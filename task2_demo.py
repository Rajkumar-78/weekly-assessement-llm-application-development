"""Task 2 demo: tenacity retries (429/5xx only), async speed-up, semaphore, partial failures."""
import asyncio
import logging
import os

from app.batch import run_batch, run_sequential, timed
from app.llm import LLMHTTPError, MockLLM, OpenAILLM
from app.reliability import acall_with_retry

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/retries.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")          # every retry is logged here

QUESTIONS = [f"Question {i}: give one tip about Python." for i in range(1, 11)]
MSG = [{"role": "user", "content": "hello"}]
REAL = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_MOCK_LLM", "").lower() not in ("1", "true")


class FailOnFive:
    """Wraps a client; question #5 always fails with HTTP 400 (not retryable)."""
    def __init__(self, inner):
        self.inner = inner

    async def chat(self, messages, tools=None, max_tokens=1000):
        if "Question 5:" in messages[-1]["content"]:
            raise LLMHTTPError(400, "bad request")
        return await self.inner.chat(messages, tools, max_tokens)


async def main():
    print(f"=== Task 2 | {'REAL APIs' if REAL else 'MOCK mode (simulated 2.1 s latency per call)'} ===\n")

    print("--- 1. Retry only on 429 / 5xx ---")
    events = []
    llm = MockLLM(failures=[429, 429], retry_events=events)      # fails twice, then works
    await llm.chat(MSG)
    print(" | ".join(events))

    calls = []
    async def unauthorized():
        calls.append(1)
        raise LLMHTTPError(401, "invalid api key")
    try:
        await acall_with_retry(unauthorized)
    except LLMHTTPError as exc:
        print(f"401 is NOT retried: {len(calls)} call, error raised immediately ({exc.status_code})")

    print("\n--- 2. Sequential vs asyncio.gather vs Semaphore(3) ---")
    llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")) if REAL else MockLLM(latency=2.1)
    _, seq = await timed(run_sequential(llm, QUESTIONS))
    _, par_all = await timed(run_batch(llm, QUESTIONS))
    _, par3 = await timed(run_batch(llm, QUESTIONS, limit=3))
    print(f"Sequential {seq:.0f} s | Parallel (limit 3) {par3:.0f} s")
    print(f"(extra: unlimited gather {par_all:.0f} s - fastest, but may trigger 429 on real APIs)")

    print("\n--- 3. One failing call does not lose the others ---")
    results = await run_batch(FailOnFive(MockLLM(latency=0.1)), QUESTIONS, limit=3)
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"{len(ok)} succeeded, {len(bad)} failed: " + "; ".join(f"Q{r['i']} -> {r['error']}" for r in bad))

    print("\n--- 4. Exponential backoff and jitter ---")
    print("Backoff: after each failure wait twice as long (1 s, 2 s, 4 s, 8 s) so a struggling server gets time to recover.")
    print("Without it, every client hammers the API again immediately and keeps the rate limit (429) alive.")
    print("Jitter adds a small random amount to each wait so 1000 clients do not all retry at the same instant.")
    print("We retry only 429/5xx (temporary); 4xx like 401 are our mistake, so retrying would just waste calls.")


if __name__ == "__main__":
    asyncio.run(main())
