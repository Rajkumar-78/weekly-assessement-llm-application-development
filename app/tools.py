"""Three tools with precise schemas, safe execution and ONE reusable tool loop."""
import asyncio
import inspect
import json
from dataclasses import dataclass, field

TOOL_LATENCY = 0.05          # simulated I/O so parallel execution is visible

ORDERS = {
    1001: {"customer_id": "C001", "status": "shipped", "eta": "2026-11-24", "total": 200.0},
    1002: {"customer_id": "C002", "status": "processing", "eta": "2026-11-28", "total": 80.0},
    1042: {"customer_id": "C001", "status": "delivered", "eta": "2026-11-10", "total": 349.5},
}
CUSTOMERS = {
    "C001": {"name": "Priya Raman", "tier": "gold", "email": "priya@example.com"},
    "C002": {"name": "Arjun Mehta", "tier": "silver", "email": "arjun@example.com"},
    "C003": {"name": "Meena Iyer", "tier": "standard", "email": "meena@example.com"},
}
TIER_DISCOUNT = {"standard": 0, "silver": 5, "gold": 10}


class ToolError(Exception):
    """Expected, model-recoverable problem. Returned as {"error": ...}, never raised to the user."""


# --------------------------------------------------------------------------- schemas
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Look up one order by its numeric id and return its status, delivery "
                       "estimate (ETA), total and customer id. Use when the user asks where an "
                       "order is or what its status is.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "integer",
                                                   "description": "Numeric order id, e.g. 1001"}},
                       "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_customer",
        "description": "Fetch a customer profile (name, loyalty tier, email) by customer id. "
                       "Customer ids look like 'C001'. Do not use for orders.",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "string",
                                                      "description": "Customer id such as 'C001'"}},
                       "required": ["customer_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "calculate_discount",
        "description": "Calculate the loyalty discount for an amount. standard=0%, silver=5%, "
                       "gold=10%. Returns discount_percent, discount_amount and final_amount.",
        "parameters": {"type": "object",
                       "properties": {"amount": {"type": "number", "exclusiveMinimum": 0,
                                                 "description": "Pre-discount amount, > 0"},
                                      "tier": {"type": "string",
                                               "enum": ["standard", "silver", "gold"],
                                               "description": "Customer loyalty tier"}},
                       "required": ["amount", "tier"], "additionalProperties": False}}},
]


# --------------------------------------------------------------------------- implementations
async def get_order_status(order_id):
    await asyncio.sleep(TOOL_LATENCY)
    if not isinstance(order_id, int) or isinstance(order_id, bool):
        raise ToolError("Invalid argument: order_id must be an integer")
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"Order {order_id} not found")
    return {"order_id": order_id, **order}


async def get_customer(customer_id):
    await asyncio.sleep(TOOL_LATENCY)
    if not isinstance(customer_id, str):
        raise ToolError("Invalid argument: customer_id must be a string like 'C001'")
    customer = CUSTOMERS.get(customer_id.upper())
    if customer is None:
        raise ToolError(f"Customer {customer_id} not found")
    return {"customer_id": customer_id.upper(), **customer}


async def calculate_discount(amount, tier):
    await asyncio.sleep(TOOL_LATENCY)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
        raise ToolError("Invalid argument: amount must be a number greater than 0")
    if tier not in TIER_DISCOUNT:
        raise ToolError(f"Invalid argument: tier must be one of {sorted(TIER_DISCOUNT)}")
    pct = TIER_DISCOUNT[tier]
    discount = round(amount * pct / 100, 2)
    return {"amount": amount, "tier": tier, "discount_percent": pct,
            "discount_amount": discount, "final_amount": round(amount - discount, 2)}


TOOL_REGISTRY = {"get_order_status": get_order_status,
                 "get_customer": get_customer,
                 "calculate_discount": calculate_discount}


async def execute_tool_call(name: str, raw_arguments: str) -> dict:
    """Run one tool call. ALWAYS returns a dict; failures become {"error": "..."}."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "Invalid argument: arguments are not valid JSON"}
    if not isinstance(args, dict):
        return {"error": "Invalid argument: arguments must be a JSON object"}
    try:
        inspect.signature(fn).bind(**args)           # missing / unexpected parameters
    except TypeError as exc:
        return {"error": f"Invalid argument: {exc}"}
    try:
        return await fn(**args)
    except ToolError as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- the loop
@dataclass
class LoopResult:
    content: str
    messages: list
    calls_per_round: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hit_round_limit: bool = False

    @property
    def rounds(self) -> int:
        return len(self.calls_per_round)


async def run_tool_loop(llm, messages, tools=TOOL_SCHEMAS, max_rounds=5,
                        usage_log=None, max_tokens=1000) -> LoopResult:
    """Ask the model; while it requests tools run ALL calls of the round in parallel
    (asyncio.gather), append each result with its tool_call_id, and ask again.
    Stops when the model answers in text or after `max_rounds` tool rounds."""
    messages = list(messages)
    result = LoopResult(content="", messages=messages)
    for _ in range(max_rounds + 1):
        resp = await llm.chat(messages, tools=tools, max_tokens=max_tokens)
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        if usage_log is not None:
            usage_log.record(model=resp.model, prompt_tokens=resp.prompt_tokens,
                             completion_tokens=resp.completion_tokens, source="tool_loop")
        if not resp.tool_calls:
            result.content = resp.content
            messages.append({"role": "assistant", "content": resp.content})
            return result
        if result.rounds >= max_rounds:
            result.hit_round_limit = True
            result.content = f"Stopped: tool round limit ({max_rounds}) reached."
            return result
        messages.append({
            "role": "assistant", "content": resp.content or None,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments}}
                           for tc in resp.tool_calls]})
        outputs = await asyncio.gather(*(execute_tool_call(tc.name, tc.arguments)
                                         for tc in resp.tool_calls))
        for tc, out in zip(resp.tool_calls, outputs):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
        result.calls_per_round.append(len(resp.tool_calls))
    return result  # pragma: no cover
