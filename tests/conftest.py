import json

import pytest
from autogen_ext.models.replay import ReplayChatCompletionClient

from app.llm import CLAUDE_MODEL_INFO


def replay(*responses: dict | str) -> ReplayChatCompletionClient:
    """A fake model client that returns the given JSON objects/strings in order."""
    return ReplayChatCompletionClient(
        [r if isinstance(r, str) else json.dumps(r) for r in responses],
        model_info=CLAUDE_MODEL_INFO,
    )


@pytest.fixture
def replay_client():
    return replay
