"""LLM clients with one interface: chat() -> LLMResponse and stream() -> tokens then Usage.

* OpenAILLM  - real AsyncOpenAI client (retries via tenacity, SDK retries disabled)
* MockLLM    - offline stand-in with configurable latency / failures, used for demos & tests
"""
import asyncio
import re
from dataclasses import dataclass, field

from .reliability import acall_with_retry
from .tokens import count_text_tokens, count_tokens


class LLMHTTPError(Exception):
    """Error carrying an HTTP status code, like the SDK errors do."""

    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON string, exactly as the model produced it


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = "stop"


class BaseLLM:
    model = "base"

    def __init__(self, sleep=None, retry_events=None):
        self._sleep = sleep                 # injectable so tests never really wait
        self.retry_events = retry_events

    async def chat(self, messages, tools=None, max_tokens=1000) -> LLMResponse:
        return await acall_with_retry(self._chat_once, messages, tools, max_tokens,
                                      sleep=self._sleep, events=self.retry_events)

    async def stream(self, messages, max_tokens=1000):
        # NOTE: no retry once tokens started flowing (would duplicate output)
        async for item in self._stream(messages, max_tokens):
            yield item


# --------------------------------------------------------------------------- OpenAI
class OpenAILLM(BaseLLM):
    def __init__(self, api_key=None, model="gpt-4o-mini", client=None, **kw):
        super().__init__(**kw)
        self.model = model
        if client is None:
            from openai import AsyncOpenAI
            # max_retries=0: tenacity is the ONLY retry layer (otherwise retries multiply)
            client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self.client = client

    async def _chat_once(self, messages, tools=None, max_tokens=1000):
        kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments)
                 for tc in (choice.message.tool_calls or [])]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=calls,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model=resp.model,
            finish_reason=choice.finish_reason,
        )

    async def _stream(self, messages, max_tokens=1000):
        stream = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_completion_tokens=max_tokens,
            stream=True, stream_options={"include_usage": True})
        usage = None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):          # last chunk has empty `choices`
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        if usage:
            yield usage


# --------------------------------------------------------------------------- Mock
_FILLER = (
    "Here is a detailed explanation. First, consider the core idea and why it matters in "
    "real systems. Second, look at the trade-offs: simplicity against flexibility, latency "
    "against throughput, and cost against quality. Third, think about failure modes and how "
    "to observe them in production with logs, metrics and traces. Teams that succeed usually "
    "start with a small, well-tested version, measure it honestly, and then iterate in short "
    "cycles. They document their decisions, keep configuration outside the code, and review "
    "results together so that knowledge is shared instead of locked inside one person. "
) * 2

_ORDER_RE = re.compile(r"order\s*#?\s*(\d{3,5})", re.I)
_CUST_RE = re.compile(r"customer\s+(C\d{3})", re.I)
_DISC_RE = re.compile(r"discount[^0-9]*(\d+(?:\.\d+)?)", re.I)
_TIER_RE = re.compile(r"\b(standard|silver|gold)\b", re.I)
_FACT_ORDER_RE = re.compile(r"order (?:number )?(?:is )?#?(\d{3,5})", re.I)
_FACT_NAME_RE = re.compile(r"my name is (\w+)", re.I)


class MockLLM(BaseLLM):
    """Deterministic fake model. It only 'knows' what is inside the messages it receives,
    so it is genuinely stateless like a real LLM."""
    model = "mock-llm"

    def __init__(self, latency=0.0, ttft=0.0, token_delay=0.0, failures=None, **kw):
        super().__init__(**kw)
        self.latency, self.ttft, self.token_delay = latency, ttft, token_delay
        self.failures = list(failures or [])       # e.g. [429, 429] -> fail twice then work
        self.calls = 0

    def _maybe_fail(self):
        if self.failures:
            raise LLMHTTPError(self.failures.pop(0))

    async def _chat_once(self, messages, tools=None, max_tokens=1000):
        self.calls += 1
        await asyncio.sleep(self.latency)
        self._maybe_fail()
        return self._respond(messages, tools)

    async def _stream(self, messages, max_tokens=1000):
        self.calls += 1
        await asyncio.sleep(self.ttft)
        self._maybe_fail()
        resp = self._respond(messages, None)
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            if i:
                await asyncio.sleep(self.token_delay)
            yield word + (" " if i < len(words) - 1 else "")
        yield Usage(resp.prompt_tokens, resp.completion_tokens)

    # -- behaviour -------------------------------------------------------------
    def _respond(self, messages, tools) -> LLMResponse:
        last = messages[-1]
        prompt_tokens = count_tokens(messages)
        if last["role"] == "tool":
            text, calls = self._summarise_tools(messages), []
        else:
            calls = self._plan_tool_calls(last.get("content") or "") if tools else []
            text = "" if calls else self._answer(messages)
        return LLMResponse(
            content=text, tool_calls=calls, prompt_tokens=prompt_tokens,
            completion_tokens=count_text_tokens(text), model=self.model,
            finish_reason="tool_calls" if calls else "stop")

    def _plan_tool_calls(self, text):
        import json
        calls = []
        n = 0

        def add(name, args):
            nonlocal n
            n += 1
            calls.append(ToolCall(f"call_{n}", name, json.dumps(args)))

        for m in _ORDER_RE.finditer(text):
            add("get_order_status", {"order_id": int(m.group(1))})
        for m in _CUST_RE.finditer(text):
            add("get_customer", {"customer_id": m.group(1).upper()})
        d, t = _DISC_RE.search(text), _TIER_RE.search(text)
        if d:
            add("calculate_discount", {"amount": float(d.group(1)),
                                       "tier": (t.group(1).lower() if t else "standard")})
        return calls

    def _summarise_tools(self, messages):
        import json
        tail = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            tail.append(json.loads(m["content"]))
        parts = []
        for d in reversed(tail):
            if "error" in d:
                parts.append(f"I could not complete that request: {d['error']}.")
            elif "status" in d:
                parts.append(f"Order {d['order_id']} is {d['status']}.")
            elif "tier" in d and "name" in d:
                parts.append(f"Customer {d['name']} is on the {d['tier']} tier.")
            elif "final_amount" in d:
                parts.append(f"With a {d['discount_percent']}% discount the final amount is "
                             f"{d['final_amount']}.")
        return " ".join(parts)

    def _answer(self, messages):
        question = messages[-1].get("content") or ""
        q = question.lower()
        context = " ".join(m["content"] for m in messages[:-1] if isinstance(m.get("content"), str))
        if "order number" in q and "what" in q:
            found = _FACT_ORDER_RE.findall(context)
            return f"Your order number is {found[-1]}." if found else "I don't have that information."
        if "my name" in q and "what" in q:
            found = _FACT_NAME_RE.findall(context)
            return f"Your name is {found[-1]}." if found else "I don't have that information."
        reply = f"(mock) You asked: {question[:50]}."
        return reply + " " + _FILLER if "detail" in q else reply
