import pytest

from calbi.testing.fake_llm import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
