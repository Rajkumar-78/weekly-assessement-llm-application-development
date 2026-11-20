"""Conversation memory: full history + sliding window + summarisation + token budget.

The LLM is stateless, so on every call we re-send what it must know:
    [system (<=500 tok)] + [summary of old turns] + [last N turns verbatim]
and keep history <= 3000 tokens; the answer is limited to 1000 tokens.
"""
from dataclasses import dataclass
from typing import Callable

from .tokens import count_text_tokens, count_tokens


@dataclass(frozen=True)
class TokenBudget:
    system: int = 500
    history: int = 3000
    output: int = 1000


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def cap_tokens(text: str, max_tokens: int) -> str:
    """Trim text to <= max_tokens, keeping the beginning and the end."""
    if count_text_tokens(text) <= max_tokens:
        return text
    chars = max_tokens * 4
    while True:
        half = max(chars // 2 - 3, 1)
        out = text[:half] + " ... " + text[-half:]
        if count_text_tokens(out) <= max_tokens or half == 1:
            return out
        chars = int(chars * 0.8)               # tokenizer was denser than 4 chars/token


def naive_summarizer(messages: list, previous: str = "") -> str:
    """Offline extractive summary: keeps what the USER said (facts live there)."""
    parts = [previous] if previous else []
    for m in messages:
        if m["role"] == "user":
            parts.append("User said: " + _clip(m["content"], 160))
        elif m["role"] == "assistant" and m.get("content"):
            parts.append("Assistant covered: " + _clip(m["content"], 50))
    return " | ".join(parts)


class Conversation:
    def __init__(self, system: str, summarizer: Callable = naive_summarizer,
                 window_turns: int = 4, budget: TokenBudget = TokenBudget(),
                 summary_budget: int = 600):
        self.system = system
        self.summarizer = summarizer
        self.window_turns = window_turns          # turns kept word-for-word
        self.budget = budget
        self.summary_budget = summary_budget
        self.messages: list = []                  # FULL log (what GET /messages returns)
        self.summary = ""
        self._cut = 0                             # messages[:_cut] are folded into summary

    # -- recording ---------------------------------------------------------------
    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def pop_last(self):
        """Undo the last message (used when the request failed)."""
        if len(self.messages) > self._cut:
            self.messages.pop()

    # -- compaction --------------------------------------------------------------
    @property
    def active(self) -> list:
        return self.messages[self._cut:]

    def _summary_message(self) -> list:
        if not self.summary:
            return []
        return [{"role": "system", "content": "Summary of earlier conversation: " + self.summary}]

    def history_messages(self) -> list:
        return self._summary_message() + self.active

    def history_tokens(self) -> int:
        return count_tokens(self.history_messages())

    def _fold(self, n: int):
        chunk = self.messages[self._cut:self._cut + n]
        self.summary = cap_tokens(self.summarizer(chunk, self.summary), self.summary_budget)
        self._cut += n

    def _next_turn_len(self) -> int:
        """Length of the oldest active turn (up to the next user message); 0 if none can go."""
        active = self.active
        for i in range(1, len(active)):
            if active[i]["role"] == "user":
                return i
        return 0

    def compact(self):
        # 1) sliding window: fold whole turns that fall outside the last N turns
        while True:
            n = self._next_turn_len()
            users_after = sum(1 for m in self.active if m["role"] == "user")
            if n and users_after > self.window_turns:
                self._fold(n)
            else:
                break
        # 2) token budget: keep folding until history fits
        while self.history_tokens() > self.budget.history:
            n = self._next_turn_len()
            if not n:
                break
            self._fold(n)

    def build_messages(self) -> list:
        self.compact()
        system = cap_tokens(self.system, self.budget.system)
        return [{"role": "system", "content": system}] + self.history_messages()

    def stats(self) -> dict:
        """full = tokens if we had sent everything; sent = tokens actually sent as history."""
        return {"full_history_tokens": count_tokens(self.messages),
                "sent_history_tokens": self.history_tokens(),
                "folded_messages": self._cut}
