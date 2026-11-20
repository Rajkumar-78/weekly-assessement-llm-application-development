"""Task 3 demo: tools (parallel, error recovery) + conversation memory (window, summary, budget)."""
import asyncio
import json
import os

from app.llm import MockLLM, OpenAILLM
from app.memory import Conversation, TokenBudget, naive_summarizer
from app.tools import execute_tool_call, run_tool_loop

REAL = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_MOCK_LLM", "").lower() not in ("1", "true")
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")) if REAL else MockLLM()
SYSTEM = "You are a helpful support assistant. Use the tools for order and customer questions."


def llm_summarizer(messages, previous=""):
    """Real mode: ask the model to compress old turns, keeping names/numbers/preferences."""
    import sdk_calls
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = (f"Previous summary: {previous or '(none)'}\n\nNew turns:\n{text}\n\n"
              "Merge into ONE short summary. Keep every name, number, id and user preference.")
    return sdk_calls.ask_openai(prompt, verbose=False)["answer"]


async def tools_part():
    print("--- 1. Tools: 3 calls in ONE round, executed in parallel ---")
    prompt = "Check order 1001, look up customer C001 and calculate the discount on 200 for a gold customer."
    res = await run_tool_loop(llm, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    print(f"Tool calls in one round: {res.calls_per_round[0]}")
    print("Final answer:", res.content)
    tool_ids = [m["tool_call_id"] for m in res.messages if m["role"] == "tool"]
    print("Results returned with tool_call_id:", tool_ids)

    print("\n--- 2. Tool errors are returned as results so the model can recover ---")
    res = await run_tool_loop(llm, [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": "Where is order 9999?"}])
    err = [m["content"] for m in res.messages if m["role"] == "tool"][0]
    print(f"Error result returned to model: {err}")
    print("Model recovered:", res.content)

    print("\n4 error cases tested:")
    cases = [("unknown order", "get_order_status", '{"order_id": 9999}'),
             ("invalid argument", "calculate_discount", '{"amount": -5, "tier": "gold"}'),
             ("unknown tool name", "refund_order", '{"order_id": 1001}'),
             ("missing argument", "get_customer", "{}")]
    for label, name, args in cases:
        print(f"  {label:18} -> {json.dumps(await execute_tool_call(name, args))}")


async def memory_part():
    print("\n--- 3. The LLM is stateless ---")
    r1 = await llm.chat([{"role": "user", "content": "My name is Priya and my order number is 1042."}])
    print("Call 1 sent: 'My name is Priya...'")
    r2 = await llm.chat([{"role": "user", "content": "What is my name?"}])
    print(f"Call 2 sent ONLY 'What is my name?'  -> {r2.content}")
    hist = [{"role": "user", "content": "My name is Priya and my order number is 1042."},
            {"role": "assistant", "content": r1.content},
            {"role": "user", "content": "What is my name?"}]
    r3 = await llm.chat(hist)
    print(f"Call 3 sent the full history          -> {r3.content}")

    print("\n--- 4. 15-turn conversation: sliding window + summary + token budget ---")
    conv = Conversation(SYSTEM, summarizer=llm_summarizer if REAL else naive_summarizer,
                        window_turns=4, budget=TokenBudget(system=500, history=3000, output=1000))
    topics = ["Python decorators", "async IO", "database indexes", "REST design", "caching",
              "message queues", "Docker layers", "CI pipelines", "unit testing", "logging",
              "rate limiting", "load balancing", "API versioning"]
    turns = ["My name is Priya and my order number is 1042."] + \
            [f"Tell me about {t} in detail." for t in topics] + ["What was my order number?"]
    for n, text in enumerate(turns, 1):
        conv.add_user(text)
        msgs = conv.build_messages()
        resp = await llm.chat(msgs, max_tokens=conv.budget.output)
        conv.add_assistant(resp.content)
        if n == 15:
            s = conv.stats()
            print(f"Turn 15: history {s['full_history_tokens']} tokens -> {s['sent_history_tokens']} after summarising")
            print(f"Early fact survived? Turn-15 answer: {resp.content}")
            print(f"Summary kept in memory: {conv.summary[:160]}...")


async def main():
    print(f"=== Task 3 | {'REAL APIs' if REAL else 'MOCK mode (offline model, same output format)'} ===\n")
    await tools_part()
    await memory_part()


if __name__ == "__main__":
    asyncio.run(main())
