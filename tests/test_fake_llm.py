import pytest

from calbi.testing.fake_llm import FakeLLM, Message


@pytest.mark.asyncio
async def test_fake_llm_scripted_responses() -> None:
    llm = FakeLLM(responses=["hello", "world"])
    msg = [Message(role="user", content="hi")]

    resp1 = await llm.complete(msg)
    assert resp1 == "hello"

    resp2 = await llm.complete(msg, response_format="json", temperature=0.5)
    assert resp2 == "world"

    assert len(llm.calls) == 2
    assert llm.calls[0].response_format == "text"
    assert llm.calls[1].response_format == "json"
    assert llm.calls[1].temperature == 0.5

    with pytest.raises(RuntimeError, match="FakeLLM exhausted"):
        await llm.complete(msg)


@pytest.mark.asyncio
async def test_fake_llm_callable_response() -> None:
    def handler(messages: list[Message], **kwargs: object) -> str:
        return f"echo: {messages[0].content}"

    llm = FakeLLM(responses=[handler])
    resp = await llm.complete([Message(role="user", content="ping")])
    assert resp == "echo: ping"
