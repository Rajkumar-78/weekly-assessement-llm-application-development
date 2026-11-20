"""Run many questions concurrently without losing results when one fails."""
import asyncio
import time


async def run_batch(llm, questions, limit=None, max_tokens=200):
    """limit=None -> plain asyncio.gather; limit=3 -> asyncio.Semaphore(3).
    Never raises: each result is {"i", "ok", "answer"|"error"} so one failure loses nothing."""
    sem = asyncio.Semaphore(limit) if limit else None

    async def one(i, question):
        try:
            if sem:
                async with sem:
                    resp = await llm.chat([{"role": "user", "content": question}], max_tokens=max_tokens)
            else:
                resp = await llm.chat([{"role": "user", "content": question}], max_tokens=max_tokens)
            return {"i": i, "ok": True, "answer": resp.content}
        except Exception as exc:                      # noqa: BLE001 - deliberate: isolate failures
            return {"i": i, "ok": False, "error": f"{getattr(exc, 'status_code', 'ERR')} {exc}"}

    return await asyncio.gather(*(one(i, q) for i, q in enumerate(questions, 1)))


async def run_sequential(llm, questions, max_tokens=200):
    out = []
    for i, question in enumerate(questions, 1):
        resp = await llm.chat([{"role": "user", "content": question}], max_tokens=max_tokens)
        out.append({"i": i, "ok": True, "answer": resp.content})
    return out


async def timed(coro):
    start = time.perf_counter()
    result = await coro
    return result, time.perf_counter() - start
