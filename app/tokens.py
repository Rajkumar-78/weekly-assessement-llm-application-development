"""Token counting (tiktoken), request guard and cost calculation."""
import json
import math
from functools import lru_cache

DEFAULT_MODEL = "gpt-4o-mini"

# USD per 1 million tokens: (input, output). Check the providers' pricing pages
# and edit this table when prices change.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "mock-llm": (0.15, 0.60),          # so the mock demo shows realistic costs
}
FALLBACK_PRICE = PRICES["gpt-4o-mini"]


class TokenLimitExceeded(Exception):
    def __init__(self, count: int, limit: int):
        super().__init__(f"Prompt has {count} tokens, limit is {limit}")
        self.count = count
        self.limit = limit


@lru_cache(maxsize=8)
def _encoding(model: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:                       # unknown model (claude, mock...)
            return tiktoken.get_encoding("cl100k_base")
    except Exception:                          # tiktoken missing or no network for vocab
        return None


def count_text_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    enc = _encoding(model)
    if enc is None:                            # rough fallback: ~4 chars per token
        return math.ceil(len(text) / 4)
    return len(enc.encode(text))


def count_tokens(messages: list, model: str = DEFAULT_MODEL) -> int:
    """Tokens of a chat messages list (OpenAI cookbook style: +3 per message, +3 priming)."""
    total = 3
    for message in messages:
        total += 3
        for value in message.values():
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value)
            total += count_text_tokens(text, model)
    return total


def check_token_limit(messages: list, limit: int, model: str = DEFAULT_MODEL) -> int:
    """Return the token count, or raise TokenLimitExceeded when above the limit."""
    n = count_tokens(messages, model)
    if n > limit:
        raise TokenLimitExceeded(n, limit)
    return n


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, FALLBACK_PRICE)
    return round(prompt_tokens * price_in / 1e6 + completion_tokens * price_out / 1e6, 6)
