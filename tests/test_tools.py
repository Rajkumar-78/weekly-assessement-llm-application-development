import asyncio
import json
import time

import pytest

import app.tools as tools
from app.llm import LLMResponse, MockLLM, ToolCall
from app.tools import execute_tool_call, run_tool_loop


def run(coro):
    return asyncio.run(coro)


# ---- the 4 required error cases (+ bad JSON) ---------------------------------------
def test_error_unknown_order():
    assert run(execute_tool_call("get_order_status", '{"order_id": 9999}')) == {"error": "Order 9999 not found"}


def test_error_invalid_argument():
    out = run(execute_tool_call("calculate_discount", '{"amount": -5, "tier": "gold"}'))
    assert out["error"].startswith("Invalid argument")
    out = run(execute_tool_call("calculate_discount", '{"amount": 10, "tier": "platinum"}'))
    assert "tier must be one of" in out["error"]


def test_error_unknown_tool_name():
    assert run(execute_tool_call("refund_order", "{}")) == {"error": "Unknown tool: refund_order"}


def test_error_missing_argument():
    assert "missing a required argument" in run(execute_tool_call("get_customer", "{}"))["error"]


def test_error_bad_json_and_non_object():
    assert "not valid JSON" in run(execute_tool_call("get_customer", "{oops"))["error"]
    assert "JSON object" in run(execute_tool_call("get_customer", "[1]"))["error"]


def test_wrong_types_are_reported():
    assert "must be an integer" in run(execute_tool_call("get_order_status", '{"order_id": "abc"}'))["error"]
    assert "must be a string" in run(execute_tool_call("get_customer", '{"customer_id": 5}'))["error"]
    assert "Customer C999 not found" in run(execute_tool_call("get_customer", '{"customer_id": "C999"}'))["error"]


# ---- happy paths -------------------------------------------------------------------------
def test_tools_return_data():
    assert run(execute_tool_call("get_order_status", '{"order_id": 1001}'))["status"] == "shipped"
    assert run(execute_tool_call("get_customer", '{"customer_id": "c001"}'))["name"] == "Priya Raman"
    d = run(execute_tool_call("calculate_discount", '{"amount": 200, "tier": "gold"}'))
    assert d["final_amount"] == 180.0 and d["discount_percent"] == 10


def test_schemas_are_complete():
    names = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    assert names == set(tools.TOOL_REGISTRY)
    for t in tools.TOOL_SCHEMAS:
        fn = t["function"]
        assert fn["description"] and fn["parameters"]["required"]


# ---- the loop ----------------------------------------------------------------------------------
PROMPT = "Check order 1001, look up customer C001 and calculate the discount on 200 for a gold customer."


def test_loop_runs_three_calls_in_one_round_with_ids():
    res = run(run_tool_loop(MockLLM(), [{"role": "user", "content": PROMPT}]))
    assert res.calls_per_round == [3]
    tool_msgs = [m for m in res.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3"]
    assert "180.0" in res.content


def test_tool_calls_of_a_round_run_in_parallel(monkeypatch):
    monkeypatch.setattr(tools, "TOOL_LATENCY", 0.2)
    start = time.perf_counter()
    run(run_tool_loop(MockLLM(), [{"role": "user", "content": PROMPT}]))
    assert time.perf_counter() - start < 0.5          # sequential would take >= 0.6 s


def test_error_result_goes_back_to_the_model():
    res = run(run_tool_loop(MockLLM(), [{"role": "user", "content": "Where is order 9999?"}]))
    tool_msg = [m for m in res.messages if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"]) == {"error": "Order 9999 not found"}
    assert "Order 9999 not found" in res.content


def test_loop_stops_after_five_rounds():
    class Endless:
        async def chat(self, messages, tools=None, max_tokens=1000):
            return LLMResponse(tool_calls=[ToolCall("x", "get_customer", '{"customer_id": "C001"}')],
                               model="m")

    res = run(run_tool_loop(Endless(), [{"role": "user", "content": "go"}], max_rounds=5))
    assert res.hit_round_limit and res.rounds == 5


def test_loop_logs_each_call(tmp_path):
    from app.usage_log import UsageLog
    log = UsageLog(str(tmp_path / "u.jsonl"))
    run(run_tool_loop(MockLLM(), [{"role": "user", "content": "Where is order 1001?"}], usage_log=log))
    assert log.today_totals()["calls"] == 2           # one tool round + one final answer
