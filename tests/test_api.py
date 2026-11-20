from app.config import get_settings


def new_session(client):
    r = client.post("/sessions")
    assert r.status_code == 201
    return r.json()["session_id"]


# ---- auth -------------------------------------------------------------------------
def test_health_is_public(anon_client):
    assert anon_client.get("/health").json() == {"status": "ok"}


def test_missing_key_returns_401(anon_client):
    r = anon_client.post("/sessions")
    assert r.status_code == 401
    assert r.json()["error"]["status"] == 401


def test_wrong_key_returns_403(client):
    r = client.post("/sessions", headers={"x-api-key": "wrong"})
    assert r.status_code == 403
    assert r.json()["error"]["type"] == "http_error"


def test_unconfigured_server_key_returns_500(client, with_settings):
    with_settings(api_key="")
    assert client.post("/sessions").status_code == 500


# ---- sessions + chat -----------------------------------------------------------------
def test_create_session(client):
    r = client.post("/sessions")
    assert r.status_code == 201
    assert len(r.json()["session_id"]) == 12


def test_chat_returns_reply_and_stores_history(client):
    sid = new_session(client)
    r = client.post(f"/sessions/{sid}/chat", json={"message": "Hi there"})
    assert r.status_code == 200
    assert r.json()["reply"] == "fake reply"
    assert r.json()["usage"]["prompt_tokens"] == 10
    msgs = client.get(f"/sessions/{sid}/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Hi there"


def test_second_turn_sends_full_history(client, fake_llm):
    sid = new_session(client)
    client.post(f"/sessions/{sid}/chat", json={"message": "first"})
    client.post(f"/sessions/{sid}/chat", json={"message": "second"})
    sent = fake_llm.received[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


# ---- validation --------------------------------------------------------------------
def test_empty_message_returns_422(client):
    sid = new_session(client)
    r = client.post(f"/sessions/{sid}/chat", json={"message": ""})
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["type"] == "validation_error" and body["details"]


def test_whitespace_only_message_returns_422(client):
    sid = new_session(client)
    assert client.post(f"/sessions/{sid}/chat", json={"message": "    "}).status_code == 422


def test_message_length_boundaries(client):
    sid = new_session(client)
    assert client.post(f"/sessions/{sid}/chat", json={"message": "x" * 2000}).status_code == 200
    assert client.post(f"/sessions/{sid}/chat", json={"message": "x" * 2001}).status_code == 422


# ---- errors ------------------------------------------------------------------------
def test_unknown_session_returns_404(client):
    r = client.post("/sessions/nope/chat", json={"message": "hi"})
    assert r.status_code == 404
    assert "nope" in r.json()["error"]["message"]
    assert client.get("/sessions/nope/messages").status_code == 404


def test_llm_failure_returns_502_and_rolls_back(client, fake_llm):
    sid = new_session(client)
    fake_llm.fail = True
    r = client.post(f"/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert r.json()["error"]["status"] == 502
    assert client.get(f"/sessions/{sid}/messages").json()["messages"] == []


def test_prompt_over_token_limit_returns_413(client, with_settings):
    with_settings(max_input_tokens=5)
    sid = new_session(client)
    r = client.post(f"/sessions/{sid}/chat", json={"message": "this is too long for five tokens"})
    assert r.status_code == 413
    assert client.get(f"/sessions/{sid}/messages").json()["messages"] == []


# ---- streaming + usage --------------------------------------------------------------
def test_stream_sends_sse_tokens_then_done(client):
    with client.stream("POST", "/chat/stream", json={"prompt": "hi"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert 'data: {"token": "Hel"}' in body
    assert 'data: {"token": "lo"}' in body
    assert "event: done" in body


def test_stream_error_becomes_error_event(client, fake_llm):
    fake_llm.fail = True
    with client.stream("POST", "/chat/stream", json={"prompt": "hi"}) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body


def test_stream_413_when_prompt_too_long(client, with_settings):
    with_settings(max_input_tokens=3)
    assert client.post("/chat/stream", json={"prompt": "a much longer prompt"}).status_code == 413


def test_usage_totals_today(client):
    sid = new_session(client)
    client.post(f"/sessions/{sid}/chat", json={"message": "hi"})            # 10 + 5 tokens
    with client.stream("POST", "/chat/stream", json={"prompt": "hi"}) as r:  # 8 + 2 tokens
        "".join(r.iter_text())
    data = client.get("/usage").json()
    assert data["calls"] == 2
    assert data["total_tokens"] == 25
    assert data["total_cost_usd"] > 0


# ---- CORS ------------------------------------------------------------------------------
def test_cors_allows_configured_origin_only(anon_client):
    origins = get_settings().cors_origins
    assert 1 <= len(origins) <= 2
    ok = anon_client.options("/sessions", headers={
        "Origin": origins[0], "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == origins[0]
    bad = anon_client.options("/sessions", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
