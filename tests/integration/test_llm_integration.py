"""Integration test for live LLM providers when API credentials are provided."""

import os
import pytest
from mia.config import ModelConfig
from mia.llm.client import ChatMessage, OpenAICompatibleClient


@pytest.mark.integration
@pytest.mark.llm
def test_live_openai_compatible_integration():
    """Verify live OpenAI or OpenAI-compatible generation when API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set.")

    config = ModelConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key_env_var="OPENAI_API_KEY",
        temperature=0.0,
        max_tokens=50
    )
    client = OpenAICompatibleClient(config)
    messages = [
        ChatMessage(role="user", content="Calculate 17 * 19 and output only the integer number.")
    ]
    resp = client.generate(messages)
    assert resp.content is not None
    assert "323" in resp.content
    assert resp.usage.total_tokens > 0
