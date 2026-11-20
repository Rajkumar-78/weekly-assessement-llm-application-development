"""Task 1 demo: SDK calls, streaming vs non-streaming, token guard, cost log, /usage.
Runs with real keys if present in .env, otherwise with the offline MockLLM."""
import asyncio
import os
import time

import sdk_calls
from app.llm import MockLLM, Usage
from app.tokens import TokenLimitExceeded, check_token_limit, count_tokens
from app.usage_log import UsageLog

REAL_OPENAI = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_MOCK_LLM", "").lower() not in ("1", "true")
REAL_CLAUDE = bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("USE_MOCK_LLM", "").lower() not in ("1", "true")
PROMPT = "Explain what Server-Sent Events are in two sentences."
MSG = [{"role": "user", "content": PROMPT}]


async def mock_call(provider, model_label, system=None):
    llm = MockLLM()
    msgs = ([{"role": "system", "content": system}] if system else []) + MSG
    resp = await llm.chat(msgs)
    cost_entry = sdk_calls.LOG.record(model="mock-llm", prompt_tokens=resp.prompt_tokens,
                                      completion_tokens=resp.completion_tokens, source=f"ask_{provider.lower()}")
    sdk_calls.print_result(provider, model_label, resp.content, resp.prompt_tokens,
                           resp.completion_tokens, cost_entry["cost_usd"])


async def mock_ttft():
    llm = MockLLM(latency=4.2, ttft=0.6, token_delay=0.02)     # latencies copied from the brief
    t = time.perf_counter()
    await llm.chat(MSG)
    non_stream = time.perf_counter() - t
    t = time.perf_counter()
    first = None
    async for item in llm.stream(MSG):
        if first is None and not isinstance(item, Usage):
            first = time.perf_counter() - t
    return non_stream, first


def main():
    mode = "REAL APIs" if (REAL_OPENAI or REAL_CLAUDE) else "MOCK mode (no keys found: simulated latency, same output format)"
    print(f"=== Task 1 | {mode} ===\n")

    print("--- 1. SDK calls ---")
    if REAL_OPENAI:
        sdk_calls.ask_openai(PROMPT)
    else:
        asyncio.run(mock_call("OpenAI", "mock-gpt"))
    if REAL_CLAUDE:
        sdk_calls.ask_claude(PROMPT, system="Answer in at most two sentences.")
    else:
        asyncio.run(mock_call("Claude", "mock-claude", system="Answer in at most two sentences."))

    print("\n--- 2. Time to first token ---")
    non_stream, first = sdk_calls.measure_ttft(PROMPT) if REAL_OPENAI else asyncio.run(mock_ttft())
    print(f"Non-streaming: first output after {non_stream:.1f} s | Streaming: first token after {first:.1f} s")

    print("\n--- 3. Token guard ---")
    print(f"count_tokens(messages) = {count_tokens(MSG)}")
    huge = [{"role": "user", "content": "word " * 6000}]
    try:
        check_token_limit(huge, sdk_calls.MAX_INPUT_TOKENS)
    except TokenLimitExceeded as exc:
        print(f"Refused: {exc}")

    print("\n--- 4. Cost log (GET /usage returns this) ---")
    totals = UsageLog(sdk_calls.LOG.path).today_totals()
    print(totals)
    print("Every call is appended to", sdk_calls.LOG.path)

    print("\nNext: start the API and test streaming:")
    print("  uvicorn app.main:app --reload")
    print('  curl -N -X POST localhost:8000/chat/stream -H "x-api-key: $API_KEY" '
          '-H "Content-Type: application/json" -d \'{"prompt":"Explain SSE"}\'')


if __name__ == "__main__":
    main()
