import json

import httpx
import pytest

from calbi.ai.adapters.openai_compat import (
    ConfigError,
    NvidiaNimLLM,
    OpenAICompatLLM,
    OpenRouterLLM,
    llm_from_env,
)
from calbi.ai.llm import LLMError, Message


@pytest.mark.asyncio
async def test_adapter_correct_request_format() -> None:
    captured_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        body = json.loads(request.read())
        assert body["model"] == "test-model"
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"reply": "ok"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    adapter = OpenAICompatLLM(
        base_url="https://api.example.com/v1",
        api_key="secret-key-123",
        model="test-model",
        client=client,
    )

    resp = await adapter.complete(
        [Message(role="user", content="hi")],
        response_format="json",
    )
    assert resp == '{"reply": "ok"}'
    assert captured_request is not None
    assert captured_request.headers["Authorization"] == "Bearer secret-key-123"


@pytest.mark.asyncio
async def test_adapter_retries_on_429_honours_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "recovered"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    adapter = OpenAICompatLLM(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="test-model",
        client=client,
    )

    resp = await adapter.complete([Message(role="user", content="hi")])
    assert resp == "recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_adapter_falls_back_on_400_json_mode_rejection() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.read())
        if "response_format" in body:
            return httpx.Response(400, text="Unsupported parameter response_format")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback success"}}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    adapter = OpenAICompatLLM(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="test-model",
        client=client,
    )

    resp = await adapter.complete([Message(role="user", content="hi")], response_format="json")
    assert resp == "fallback success"
    assert calls == 2


@pytest.mark.asyncio
async def test_adapter_exhausted_retries_raises_llm_error_without_key_leak() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    secret = "SUPER_SECRET_TOKEN_DO_NOT_LEAK"
    adapter = OpenAICompatLLM(
        base_url="https://api.example.com/v1",
        api_key=secret,
        model="test-model",
        max_retries=2,
        client=client,
    )

    # Repr must never contain the key
    assert secret not in repr(adapter)

    with pytest.raises(LLMError) as exc_info:
        await adapter.complete([Message(role="user", content="hi")])

    assert secret not in str(exc_info.value)


def test_openrouter_and_nvidia_nim_subclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    router = OpenRouterLLM(model="meta-llama/llama-3")
    assert router.base_url == "https://openrouter.ai/api/v1"
    assert router.api_key == "router-key"

    monkeypatch.setenv("NVIDIA_API_KEY", "nim-key")
    nim = NvidiaNimLLM(model="meta/llama3-70b-instruct")
    assert nim.base_url == "https://integrate.api.nvidia.com/v1"

    # Custom NIM base URL override
    nim_custom = NvidiaNimLLM(model="model", base_url="http://local-nim:8000/v1")
    assert nim_custom.base_url == "http://local-nim:8000/v1"


def test_llm_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALBI_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("CALBI_LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key-123")

    llm = llm_from_env()
    assert isinstance(llm, OpenRouterLLM)
    assert llm.model == "openai/gpt-4o"

    # Missing model
    monkeypatch.delenv("CALBI_LLM_MODEL")
    with pytest.raises(ConfigError, match="CALBI_LLM_MODEL"):
        llm_from_env()
