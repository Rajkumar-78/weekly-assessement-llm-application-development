"""Dependencies for Depends(): settings, LLM client, store, usage log."""
from functools import lru_cache

from fastapi import Depends

from .config import Settings, get_settings
from .llm import MockLLM, OpenAILLM
from .store import InMemoryStore
from .usage_log import UsageLog

_store = InMemoryStore()


@lru_cache
def _build_llm(use_mock: bool, api_key: str, model: str):
    if use_mock:
        return MockLLM(latency=0.05, ttft=0.05, token_delay=0.01)
    return OpenAILLM(api_key=api_key, model=model)       # pragma: no cover


def get_llm(settings: Settings = Depends(get_settings)):
    return _build_llm(settings.use_mock, settings.openai_api_key, settings.openai_model)


def get_store() -> InMemoryStore:
    return _store


def get_usage_log(settings: Settings = Depends(get_settings)) -> UsageLog:
    return UsageLog(settings.usage_log_path)
