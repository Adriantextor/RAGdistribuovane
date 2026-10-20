import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from services.app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    async def idle(*args):
        await asyncio.Future()
    monkeypatch.setattr(main, "worker", idle)
    with TestClient(main.app) as value:
        yield value


def test_http_idempotency_validation_and_session_isolation(client):
    first = client.post("/api/sessions").json()["session_id"]
    second = client.post("/api/sessions").json()["session_id"]
    request = str(uuid4())
    payload = {"request_id": request, "question": "What is RAG?"}
    for _ in range(2):
        assert client.post(f"/api/sessions/{first}/questions", json=payload).status_code == 202
    assert len(client.get(f"/api/sessions/{first}/messages").json()) == 1
    assert client.get(f"/api/sessions/{second}/messages").json() == []
    assert client.post(f"/api/sessions/{second}/questions", json=payload).status_code == 409
    assert client.post(f"/api/sessions/{first}/questions", json={**payload, "question": "Changed"}).status_code == 409
    assert client.post(f"/api/sessions/{first}/questions", json={**payload, "question": "  "}).status_code == 422
    assert client.get(f"/api/requests/{uuid4()}").status_code == 404


def test_retry_is_explicit_and_does_not_create_duplicate_history(client):
    session = client.post("/api/sessions").json()["session_id"]
    request = str(uuid4())
    client.post(f"/api/sessions/{session}/questions", json={"request_id": request, "question": "Question"})
    assert client.post(f"/api/requests/{request}/retry").status_code == 409
    store = main.app.state.store
    job = store.claim(60, 1)
    store.retry(request, 1, job["lease_owner"])
    response = client.post(f"/api/requests/{request}/retry")
    assert response.status_code == 202 and response.json()["status"] == "queued"
    assert len(client.get(f"/api/sessions/{session}/messages").json()) == 1


def test_sse_reconnect_snapshot_and_live_partial_frames(client):
    session = client.post("/api/sessions").json()["session_id"]
    request_id = str(uuid4())
    client.post(f"/api/sessions/{session}/questions", json={"request_id": request_id, "question": "Question"})
    store = main.app.state.store
    async def scenario():
        async def connected():
            return False
        response = await main.events(UUID(session), SimpleNamespace(is_disconnected=connected))
        iterator = response.body_iterator
        assert "retry:" in await anext(iterator)
        snapshot = await anext(iterator)
        assert "event: snapshot" in snapshot and request_id in snapshot
        job = store.claim(60, 3)
        store.generating(request_id, [], owner=job["lease_owner"])
        store.partial(request_id, job["lease_owner"], "first tokens")
        frames = [await anext(iterator) for _ in range(3)]
        assert any("first tokens" in frame and "event: job" in frame for frame in frames)
        await iterator.aclose()
    asyncio.run(scenario())
