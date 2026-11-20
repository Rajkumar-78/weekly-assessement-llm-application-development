import pytest

from app.tokens import (TokenLimitExceeded, calculate_cost, check_token_limit,
                        count_text_tokens, count_tokens)
from app.usage_log import UsageLog


def test_count_tokens_grows_with_text():
    short = count_tokens([{"role": "user", "content": "hi"}])
    long = count_tokens([{"role": "user", "content": "hi " * 100}])
    assert 0 < short < long


def test_count_handles_tool_call_messages():
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]}
    assert count_tokens([msg]) > 3
    assert count_text_tokens("hello world") >= 1


def test_request_above_limit_is_refused():
    msgs = [{"role": "user", "content": "word " * 500}]
    with pytest.raises(TokenLimitExceeded) as err:
        check_token_limit(msgs, 100)
    assert err.value.limit == 100 and err.value.count > 100
    assert check_token_limit(msgs, 10_000) > 0


def test_cost_calculation():
    assert calculate_cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert calculate_cost("unknown-model", 1000, 0) == pytest.approx(0.00015)   # fallback price


def test_usage_log_records_and_totals(tmp_path):
    log = UsageLog(str(tmp_path / "logs" / "u.jsonl"))
    assert log.today_totals()["calls"] == 0
    log.record(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50, source="t", latency_s=1.234)
    log.record(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
    totals = log.today_totals()
    assert totals["calls"] == 2 and totals["total_tokens"] == 165
    assert totals["total_cost_usd"] > 0
