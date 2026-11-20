"""Retries with tenacity: only 429 and 5xx are retried; 4xx (401, 400...) fail at once."""
import asyncio
import logging

from tenacity import (AsyncRetrying, Retrying, retry_if_exception,
                      stop_after_attempt, wait_exponential)

log = logging.getLogger("llm.retry")


def is_retryable(exc: BaseException) -> bool:
    """True for HTTP 429 and any 5xx. Works with openai/anthropic errors (.status_code)."""
    code = getattr(exc, "status_code", None)
    return isinstance(code, int) and (code == 429 or 500 <= code < 600)


def _policy(sleep, events):
    def before_sleep(state):
        exc = state.outcome.exception()
        code = getattr(exc, "status_code", "?")
        wait = state.next_action.sleep
        msg = f"Attempt {state.attempt_number} failed ({code}) -> wait {wait:g} s"
        log.warning(msg)
        if events is not None:
            events.append(msg)

    return dict(
        stop=stop_after_attempt(5),
        # waits 1, 2, 4, 8 ... seconds (capped at 30)
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(is_retryable),
        before_sleep=before_sleep,
        sleep=sleep,
        reraise=True,          # after 5 failed attempts the ORIGINAL error is raised
    )


def call_with_retry(fn, *args, sleep=None, events=None, **kwargs):
    """Sync version. `events` collects one message per retry (for demos/tests)."""
    import time
    retrying = Retrying(**_policy(sleep or time.sleep, events))
    result = retrying(fn, *args, **kwargs)
    if events is not None and retrying.statistics.get("attempt_number", 1) > 1:
        events.append(f"Attempt {retrying.statistics['attempt_number']} succeeded")
    return result


async def acall_with_retry(fn, *args, sleep=None, events=None, **kwargs):
    """Async version used by every LLM client."""
    retrying = AsyncRetrying(**_policy(sleep or asyncio.sleep, events))
    result = await retrying(fn, *args, **kwargs)
    if events is not None and retrying.statistics.get("attempt_number", 1) > 1:
        events.append(f"Attempt {retrying.statistics['attempt_number']} succeeded")
    return result
