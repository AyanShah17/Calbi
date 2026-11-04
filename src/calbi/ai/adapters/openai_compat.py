from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Literal

import httpx

from calbi.ai.llm import LLMError, Message
from calbi.core.schema import CalbiError


class ConfigError(CalbiError):
    """Raised when adapter or environment configuration is invalid or missing."""


class OpenAICompatLLM:
    """Universal OpenAI-compatible chat completions adapter using httpx."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
        json_mode: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.json_mode = json_mode
        self._client = client

    def __repr__(self) -> str:
        # Never leak secret API key in repr
        return (
            f"{self.__class__.__name__}(base_url={self.base_url!r}, "
            f"model={self.model!r}, json_mode={self.json_mode})"
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float = 0.0,
    ) -> str:
        """Call chat completions with retries and fallback."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        use_json = response_format == "json" and self.json_mode
        if use_json:
            payload["response_format"] = {"type": "json_object"}

        should_close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout_s)
            should_close_client = True

        try:
            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    resp = await client.post(url, headers=headers, json=payload)

                    # Check for 400 rejection of response_format json_mode
                    if resp.status_code == 400 and use_json and "response_format" in payload:
                        payload.pop("response_format", None)
                        use_json = False
                        continue

                    # Rate limited or server error -> retry
                    if resp.status_code == 429 or resp.status_code >= 500:
                        if attempt < self.max_retries:
                            retry_after_str = resp.headers.get("Retry-After")
                            if retry_after_str:
                                try:
                                    delay = float(retry_after_str)
                                except ValueError:
                                    delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                            else:
                                delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

                            await asyncio.sleep(delay)
                            continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return str(content or "")

                except httpx.HTTPStatusError as e:
                    if attempt >= self.max_retries:
                        snippet = e.response.text[:200]
                        clean_msg = (
                            f"HTTP {e.response.status_code} error from {self.base_url}: {snippet}"
                        )
                        raise LLMError(clean_msg) from None
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    if attempt >= self.max_retries:
                        raise LLMError(
                            f"Network error communicating with {self.base_url}: {e}"
                        ) from None
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

            raise LLMError(f"Exhausted retries ({self.max_retries}) calling {self.base_url}")
        finally:
            if should_close_client:
                await client.aclose()


class OpenRouterLLM(OpenAICompatLLM):
    """OpenRouter API adapter for hosted LLM models."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kw: Any,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ConfigError(
                "OPENROUTER_API_KEY environment variable or api_key parameter is required."
            )

        headers = kw.pop("extra_headers", {}) or {}
        app_url = os.getenv("CALBI_APP_URL")
        app_name = os.getenv("CALBI_APP_NAME", "Calbi")
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
            extra_headers=headers,
            **kw,
        )


class NvidiaNimLLM(OpenAICompatLLM):
    """NVIDIA NIM API adapter for NVIDIA microservice-hosted models."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kw: Any,
    ) -> None:
        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ConfigError(
                "NVIDIA_API_KEY environment variable or api_key parameter is required."
            )

        url = base_url or os.getenv("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1"

        super().__init__(
            base_url=url,
            api_key=key,
            model=model,
            **kw,
        )


def llm_from_env() -> OpenAICompatLLM:
    """Instantiate an LLM adapter from environment variables."""
    provider = os.getenv("CALBI_LLM_PROVIDER", "").strip().lower()
    if not provider:
        msg = "CALBI_LLM_PROVIDER env variable is required (e.g. 'openrouter' or 'nvidia_nim')."
        raise ConfigError(msg)

    model = os.getenv("CALBI_LLM_MODEL", "").strip()
    if not model:
        raise ConfigError("CALBI_LLM_MODEL environment variable is required.")

    if provider in ("openrouter", "open_router"):
        return OpenRouterLLM(model=model)
    elif provider in ("nvidia_nim", "nvidia", "nim"):
        return NvidiaNimLLM(model=model)
    else:
        raise ConfigError(
            f"Unsupported CALBI_LLM_PROVIDER '{provider}'. Expected 'openrouter' or 'nvidia_nim'."
        )
