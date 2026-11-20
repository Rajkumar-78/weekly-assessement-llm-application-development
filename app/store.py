"""In-memory session store (swap for a database later without touching the routes)."""
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .memory import Conversation

DEFAULT_SYSTEM = "You are a helpful, concise company support assistant."


@dataclass
class Session:
    id: str
    created_at: str
    conversation: Conversation


class InMemoryStore:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM):
        self.system_prompt = system_prompt
        self._sessions = {}

    def create(self) -> Session:
        session = Session(uuid.uuid4().hex[:12],
                          datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          Conversation(self.system_prompt))
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str):
        return self._sessions.get(session_id)
