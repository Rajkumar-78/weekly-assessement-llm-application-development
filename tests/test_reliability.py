import asyncio

import pytest

from app.llm import LLMHTTPError, MockLLM
from app.reliability import acall_with_retry, call_with_retry, is_retryable


def test_retries_twice_then_succeeds():
    calls, sleeps, events = [], [], []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMHTTPError(429)
        return "ok"

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    assert asyncio.run(acall_with_retry(flaky, sleep=fake_sleep, events=events)) == "ok"
    assert len(calls) == 3
    assert sleeps == [1, 2]                                   # exponential: 1 s, 2 s
    assert events == ["Attempt 1 failed (429) -> wait 1 s",
                      "Attempt 2 failed (429) -> wait 2 s",
                      "Attempt 3 succeeded"]


@pytest.mark.parametrize("code", [401, 400, 404])
def test_client_errors_are_not_retried(code):
    calls = []

    async def bad():
        calls.append(1)
        raise LLMHTTPError(code)

    async def fake_sleep(seconds):
        raise AssertionError("must not sleep")

    with pytest.raises(LLMHTTPError):
        asyncio.run(acall_with_retry(bad, sleep=fake_sleep))
    assert len(calls) == 1


def test_gives_up_after_five_attempts_and_reraises():
    calls, sleeps = [], []

    async def down():
        calls.append(1)
        raise LLMHTTPError(503)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    with pytest.raises(LLMHTTPError) as err:
        asyncio.run(acall_with_retry(down, sleep=fake_sleep))
    assert err.value.status_code == 503
    assert len(calls) == 5 and sleeps == [1, 2, 4, 8]


def test_sync_version_retries_500():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise LLMHTTPError(500)
        return "done"

    events = []
    assert call_with_retry(flaky, sleep=lambda s: None, events=events) == "done"
    assert events[-1] == "Attempt 2 succeeded"


def test_is_retryable_rules():
    assert is_retryable(LLMHTTPError(429)) and is_retryable(LLMHTTPError(502))
    assert not is_retryable(LLMHTTPError(401)) and not is_retryable(ValueError("x"))


def test_llm_client_retries_through_mock():
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    llm = MockLLM(failures=[429, 429], sleep=fake_sleep)
    resp = asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert resp.content and llm.calls == 3 and sleeps == [1, 2]
