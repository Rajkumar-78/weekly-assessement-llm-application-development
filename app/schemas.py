"""Pydantic models (request validation + response shapes)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)   # "   " counts as empty -> 422
    message: str = Field(..., min_length=1, max_length=2000)


class StreamRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    prompt: str = Field(..., min_length=1, max_length=2000)
    system: Optional[str] = Field(None, max_length=1000)


class SessionResponse(BaseModel):
    session_id: str
    created_at: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    usage: UsageInfo


class MessageOut(BaseModel):
    role: str
    content: str


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]
