"""Task 1 - direct SDK calls (official formats) with token guard, cost and logging.

ask_openai(prompt)          -> OpenAI Chat Completions  (openai SDK)
ask_claude(prompt, system)  -> Anthropic Messages API    (anthropic SDK)
Keys come from .env (OPENAI_API_KEY / ANTHROPIC_API_KEY); nothing is hard-coded.
"""
import os
import time

from dotenv import load_dotenv

from app.tokens import calculate_cost, check_token_limit
from app.usage_log import UsageLog

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4000"))
LOG = UsageLog(os.getenv("USAGE_LOG", "logs/usage.jsonl"))


def print_result(provider, model, answer, prompt_tokens, completion_tokens, cost):
    print(f"[{provider}] model={model}")
    print(f"Answer: {answer.strip()[:300]}")
    print(f"Tokens: prompt={prompt_tokens} completion={completion_tokens} "
          f"total={prompt_tokens + completion_tokens} cost=${cost:.6f}")


def ask_openai(prompt: str, system: str = None, verbose: bool = True) -> dict:
    from openai import OpenAI                       # official SDK
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    check_token_limit(messages, MAX_INPUT_TOKENS)   # refuse BEFORE paying
    start = time.perf_counter()
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                          max_completion_tokens=500)
    latency = time.perf_counter() - start
    answer = resp.choices[0].message.content
    p, c = resp.usage.prompt_tokens, resp.usage.completion_tokens
    entry = LOG.record(model=resp.model, prompt_tokens=p, completion_tokens=c,
                       source="ask_openai", latency_s=latency)
    if verbose:
        print_result("OpenAI", resp.model, answer, p, c, entry["cost_usd"])
    return {"answer": answer, "model": resp.model, "prompt_tokens": p, "completion_tokens": c}


def ask_claude(prompt: str, system: str = None, verbose: bool = True) -> dict:
    import anthropic                                # official SDK
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    check_token_limit(([{"role": "system", "content": system}] if system else [])
                      + [{"role": "user", "content": prompt}], MAX_INPUT_TOKENS)
    kwargs = {"system": system} if system else {}   # Claude takes `system` as a separate field
    start = time.perf_counter()
    resp = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=500,
                                  messages=[{"role": "user", "content": prompt}], **kwargs)
    latency = time.perf_counter() - start
    answer = resp.content[0].text
    p, c = resp.usage.input_tokens, resp.usage.output_tokens
    LOG.record(model=resp.model, prompt_tokens=p, completion_tokens=c,
               source="ask_claude", latency_s=latency)
    if verbose:
        print_result("Claude", resp.model, answer, p, c, calculate_cost(resp.model, p, c))
    return {"answer": answer, "model": resp.model, "prompt_tokens": p, "completion_tokens": c}


def measure_ttft(prompt: str) -> tuple:
    """(seconds until a NON-streaming call returns, seconds until the FIRST streamed token)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msgs = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, max_completion_tokens=400)
    non_stream = time.perf_counter() - start

    start = time.perf_counter()
    first = None
    stream = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs,
                                            max_completion_tokens=400, stream=True)
    for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - start
    return non_stream, first
