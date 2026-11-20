from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.deps import get_llm, get_store, get_usage_log
from app.llm import LLMHTTPError, LLMResponse, Usage
from app.main import app
from app.store import InMemoryStore
from app.usage_log import UsageLog

API_KEY = "test-key"


class FakeLLM:
    """Mocked LLM injected with dependency_overrides - no network, no cost."""
    model = "gpt-4o-mini"

    def __init__(self):
        self.fail = False
        self.received = []

    async def chat(self, messages, tools=None, max_tokens=1000):
        self.received.append(messages)
        if self.fail:
            raise LLMHTTPError(500, "boom")
        return LLMResponse(content="fake reply", prompt_tokens=10, completion_tokens=5, model=self.model)

    async def stream(self, messages, max_tokens=1000):
        if self.fail:
            raise LLMHTTPError(500, "boom")
        yield "Hel"
        yield "lo"
        yield Usage(8, 2)


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=API_KEY, use_mock=True, max_input_tokens=4000,
                    usage_log_path=str(tmp_path / "usage.jsonl"))


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture(autouse=True)
def overrides(settings, fake_llm, tmp_path):
    store = InMemoryStore()
    log = UsageLog(settings.usage_log_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_usage_log] = lambda: log
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app, headers={"x-api-key": API_KEY})


@pytest.fixture
def anon_client():
    return TestClient(app)


@pytest.fixture
def with_settings(settings):
    """Helper: override settings fields for one test."""
    def apply(**changes):
        new = replace(settings, **changes)
        app.dependency_overrides[get_settings] = lambda: new
        return new
    return apply
