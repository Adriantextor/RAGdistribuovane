import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from services.app.main import worker
from services.app.store import Store
from shared.config import settings


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("NODE_API_TOKEN", "test-only-token-abcdefghijklmnopqrstuvwxyz")
    settings.cache_clear()
    yield
    settings.cache_clear()


def test_offline_inference_does_not_consume_attempts(tmp_path, configured):
    store = Store(f"sqlite:///{tmp_path / 'offline.db'}")
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "What is QLoRA?")

    async def scenario():
        def respond(req):
            return httpx.Response(503 if req.url.port == 8002 else 200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            application = SimpleNamespace(state=SimpleNamespace(store=store, client=client))
            task = asyncio.create_task(worker(application))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())
    assert store.get(request)["status"] == "queued"
    assert store.get(request)["attempts"] == 0
    store.engine.dispose()


def test_worker_stores_sources_and_uses_only_its_session_history(tmp_path, configured):
    store = Store(f"sqlite:///{tmp_path / 'flow.db'}")
    first, second = [store.create_session()["session_id"] for _ in range(2)]
    earlier = str(uuid4())
    store.enqueue(earlier, second, "Secret from another session")
    store.complete(earlier, "PRIVATE ANSWER", [])
    request = str(uuid4())
    store.enqueue(request, first, "What is QLoRA?")
    source = {"id": "chunk-1", "document": "qlora.pdf", "page": 2, "text": "QLoRA uses quantization.", "score": 0.8}

    async def scenario():
        def respond(req):
            if req.url.path == "/health":
                return httpx.Response(200, json={"status": "ready"})
            if req.url.path == "/search":
                return httpx.Response(200, json={"sources": [source]})
            if req.url.path == "/generate":
                import json
                body = json.loads(req.content)
                assert body["history"] == []
                assert body["sources"] == [source]
                return httpx.Response(200, json={"answer": "QLoRA uses quantization [1]."})
            raise AssertionError(f"Unexpected endpoint: {req.url}")
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            application = SimpleNamespace(state=SimpleNamespace(store=store, client=client))
            task = asyncio.create_task(worker(application))
            try:
                for _ in range(100):
                    if store.get(request)["status"] == "completed":
                        break
                    await asyncio.sleep(0.01)
                assert store.get(request)["status"] == "completed"
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    asyncio.run(scenario())
    assert store.get(request)["sources"] == [source]
    assert store.history(second)[0]["answer"] == "PRIVATE ANSWER"
    store.engine.dispose()
