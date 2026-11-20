"""Append-only JSONL log of every LLM call + daily totals for GET /usage."""
import json
import os
from datetime import datetime
from threading import Lock

from .tokens import calculate_cost


class UsageLog:
    def __init__(self, path: str = "logs/usage.jsonl"):
        self.path = path
        self._lock = Lock()

    def record(self, *, model, prompt_tokens, completion_tokens, source="chat",
               latency_s=None, status="ok") -> dict:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": calculate_cost(model, prompt_tokens, completion_tokens),
            "latency_s": None if latency_s is None else round(latency_s, 3),
            "status": status,
        }
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def today_totals(self) -> dict:
        today = datetime.now().date().isoformat()
        calls = tokens = 0
        cost = 0.0
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row["ts"].startswith(today):
                        calls += 1
                        tokens += row["total_tokens"]
                        cost += row["cost_usd"]
        return {"date": today, "calls": calls, "total_tokens": tokens,
                "total_cost_usd": round(cost, 6)}
