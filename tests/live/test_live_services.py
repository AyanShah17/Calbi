import os

import pytest

from calbi.ai.adapters.openai_compat import NvidiaNimLLM, OpenRouterLLM
from calbi.ai.llm import Message


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_openrouter_smoke() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set.")

    model = os.getenv("CALBI_LLM_MODEL", "meta-llama/llama-3-8b-instruct:free")
    llm = OpenRouterLLM(model=model, api_key=api_key)
    res = await llm.complete([Message(role="user", content="Reply with only the word OK")])
    assert len(res) > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_nvidia_nim_smoke() -> None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY is not set.")

    model = os.getenv("CALBI_LLM_MODEL", "meta/llama3-8b-instruct")
    llm = NvidiaNimLLM(model=model, api_key=api_key)
    res = await llm.complete([Message(role="user", content="Reply with only the word OK")])
    assert len(res) > 0
