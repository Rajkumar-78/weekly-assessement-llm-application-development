import asyncio
from types import SimpleNamespace

import pytest

from app.batch import run_batch, run_sequential, timed
from app.config import Settings, get_settings
from app.deps import get_llm, get_store, get_usage_log
from app.llm import LLMHTTPError, MockLLM, OpenAILLM, Usage

MSG = [{"role": "user", "content": "hi"}]


# ---- OpenAILLM with a fake SDK client (no network) ---------------------------------------
class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return self._chunks()
        call = SimpleNamespace(id="c1", function=SimpleNamespace(
            name="get_customer", arguments='{"customer_id": "C001"}'))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]),
                                     finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3), model="gpt-4o-mini")

    async def _chunks(self):
        for text in ("Hel", "lo"):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2))


def fake_openai():
    comp = FakeCompletions()
    return OpenAILLM(client=SimpleNamespace(chat=SimpleNamespace(completions=comp))), comp


def test_openai_chat_parses_tool_calls_and_usage():
    llm, comp = fake_openai()
    resp = asyncio.run(llm.chat(MSG, tools=[{"type": "function"}]))
    assert resp.tool_calls[0].name == "get_customer" and resp.tool_calls[0].id == "c1"
    assert (resp.prompt_tokens, resp.completion_tokens) == (7, 3)
    assert comp.kwargs["tools"] == [{"type": "function"}]


def test_openai_stream_yields_tokens_then_usage():
    llm, comp = fake_openai()

    async def collect():
        return [x async for x in llm.stream(MSG)]

    items = asyncio.run(collect())
    assert items[:2] == ["Hel", "lo"] and isinstance(items[2], Usage)
    assert comp.kwargs["stream_options"] == {"include_usage": True}


# ---- MockLLM stream ----------------------------------------------------------------------------
def test_mock_stream_and_failure():
    async def collect(llm):
        return [x async for x in llm.stream(MSG)]

    items = asyncio.run(collect(MockLLM()))
    assert isinstance(items[-1], Usage) and "".join(items[:-1]).startswith("(mock)")
    with pytest.raises(LLMHTTPError):
        asyncio.run(collect(MockLLM(failures=[500])))


# ---- batch: semaphore + partial failures ------------------------------------------------------
class Tracking:
    def __init__(self):
        self.running = self.peak = 0

    async def chat(self, messages, tools=None, max_tokens=1000):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.02)
        self.running -= 1
        if "bad" in messages[-1]["content"]:
            raise LLMHTTPError(400, "bad request")
        return SimpleNamespace(content="ok")


def test_semaphore_limits_concurrency_and_failure_loses_nothing():
    llm = Tracking()
    questions = ["q1", "q2", "bad q3", "q4", "q5", "q6", "q7"]
    results = asyncio.run(run_batch(llm, questions, limit=3))
    assert llm.peak == 3
    assert [r["ok"] for r in results] == [True, True, False, True, True, True, True]
    assert results[2]["error"].startswith("400")


def test_gather_is_faster_than_sequential():
    async def go():
        llm = MockLLM(latency=0.05)
        qs = ["a", "b", "c", "d"]
        _, seq = await timed(run_sequential(llm, qs))
        _, par = await timed(run_batch(llm, qs))
        return seq, par

    seq, par = asyncio.run(go())
    assert par < seq / 2


# ---- deps / config ----------------------------------------------------------------------------------
def test_default_dependencies():
    s = Settings(use_mock=True, usage_log_path="logs/x.jsonl")
    assert isinstance(get_llm(s), MockLLM)
    assert get_store() is get_store()
    assert get_usage_log(s).path == "logs/x.jsonl"


def test_more_than_two_cors_origins_is_rejected(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com,http://c.com")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError):
            get_settings()
    finally:
        monkeypatch.delenv("CORS_ORIGINS")
        get_settings.cache_clear()
