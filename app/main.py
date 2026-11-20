"""Secured FastAPI backend: sessions + chat, SSE streaming, usage, auth, CORS, tests-friendly DI."""
import json
import time

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import verify_api_key
from .config import Settings, get_settings
from .deps import get_llm, get_store, get_usage_log
from .llm import Usage
from .schemas import (ChatRequest, ChatResponse, MessageOut, MessagesResponse,
                      SessionResponse, StreamRequest, UsageInfo)
from .store import InMemoryStore
from .tokens import calculate_cost, count_text_tokens, count_tokens
from .usage_log import UsageLog

app = FastAPI(title="Company Chat Service", version="1.0.0")

# ---- CORS: only the (max 2) configured browser origins ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["x-api-key", "content-type"],
)


# ---- one consistent JSON error body: {"error": {"status", "type", "message"}} ----
def _error(status: int, type_: str, message: str, headers=None, **extra) -> JSONResponse:
    return JSONResponse({"error": {"status": status, "type": type_, "message": message, **extra}},
                        status_code=status, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request, exc):
    return _error(exc.status_code, "http_error", str(exc.detail), headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    details = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return _error(422, "validation_error", "Request validation failed", details=details)


# ---- public route -----------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- protected routes (x-api-key) ----------------------------------------------------
router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(store: InMemoryStore = Depends(get_store)):
    session = store.create()
    return SessionResponse(session_id=session.id, created_at=session.created_at)


def _get_session_or_404(store: InMemoryStore, session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest,
               store: InMemoryStore = Depends(get_store),
               llm=Depends(get_llm),
               settings: Settings = Depends(get_settings),
               usage_log: UsageLog = Depends(get_usage_log)):
    conv = _get_session_or_404(store, session_id).conversation
    conv.add_user(body.message)
    messages = conv.build_messages()                      # window + summary + token budget
    tokens = count_tokens(messages)
    if tokens > settings.max_input_tokens:
        conv.pop_last()
        raise HTTPException(413, f"Prompt too long: {tokens} tokens (limit {settings.max_input_tokens})")
    start = time.perf_counter()
    try:
        resp = await llm.chat(messages, max_tokens=conv.budget.output)
    except Exception as exc:                              # provider down / retries exhausted
        conv.pop_last()
        raise HTTPException(502, "LLM provider failed, please retry later") from exc
    conv.add_assistant(resp.content)
    entry = usage_log.record(model=resp.model, prompt_tokens=resp.prompt_tokens,
                             completion_tokens=resp.completion_tokens, source="chat",
                             latency_s=time.perf_counter() - start)
    return ChatResponse(session_id=session_id, reply=resp.content,
                        usage=UsageInfo(prompt_tokens=resp.prompt_tokens,
                                        completion_tokens=resp.completion_tokens,
                                        cost_usd=entry["cost_usd"]))


@router.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_messages(session_id: str, store: InMemoryStore = Depends(get_store)):
    conv = _get_session_or_404(store, session_id).conversation
    return MessagesResponse(session_id=session_id,
                            messages=[MessageOut(**m) for m in conv.messages])


@router.post("/chat/stream")
async def chat_stream(body: StreamRequest,
                      llm=Depends(get_llm),
                      settings: Settings = Depends(get_settings),
                      usage_log: UsageLog = Depends(get_usage_log)):
    """Server-Sent Events:  data: {"token": "..."}  ... then  event: done  data: {usage}."""
    messages = ([{"role": "system", "content": body.system}] if body.system else [])
    messages.append({"role": "user", "content": body.prompt})
    prompt_tokens = count_tokens(messages)
    if prompt_tokens > settings.max_input_tokens:         # refuse BEFORE spending money
        raise HTTPException(413, f"Prompt too long: {prompt_tokens} tokens (limit {settings.max_input_tokens})")

    async def events():
        start = time.perf_counter()
        parts, usage, first, recorded = [], None, None, False

        def record():
            p = usage.prompt_tokens if usage else prompt_tokens
            c = usage.completion_tokens if usage else count_text_tokens("".join(parts))
            return usage_log.record(model=llm.model, prompt_tokens=p, completion_tokens=c,
                                    source="stream", latency_s=time.perf_counter() - start), p, c

        try:
            async for item in llm.stream(messages, max_tokens=1000):
                if isinstance(item, Usage):
                    usage = item
                    continue
                if first is None:
                    first = time.perf_counter() - start
                parts.append(item)
                yield f"data: {json.dumps({'token': item})}\n\n"
            entry, p, c = record()
            recorded = True
            done = {"prompt_tokens": p, "completion_tokens": c, "cost_usd": entry["cost_usd"],
                    "first_token_s": None if first is None else round(first, 3)}
            yield f"event: done\ndata: {json.dumps(done)}\n\n"
        except Exception as exc:       # status is already 200 once streaming began -> error event
            yield f"event: error\ndata: {json.dumps({'status': getattr(exc, 'status_code', 502), 'message': 'LLM stream failed'})}\n\n"
        finally:
            if not recorded and parts:                     # client left early: still log the call
                record()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/usage")
async def usage(usage_log: UsageLog = Depends(get_usage_log)):
    return usage_log.today_totals()


app.include_router(router)
