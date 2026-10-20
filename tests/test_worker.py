import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx

from services.app.store import Store
from services.app.worker import cache_key, worker


SOURCE = {"id": "chunk-1", "document_id": "qlora", "document": "qlora.pdf", "page": 2,
          "text": "QLoRA uses 4-bit quantization and LoRA adapters.", "score": .8}


def test_offline_inference_does_not_consume_attempts(tmp_path):
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
            await asyncio.sleep(.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())
    assert store.get(request)["status"] == "queued"
    assert store.get(request)["attempts"] == 0
    store.engine.dispose()


def mock_nodes(counts, stream_error=False):
    def respond(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"fingerprint": "model-v1"})
        if req.url.path == "/manifest":
            return httpx.Response(200, json={"revision": "corpus-v1"})
        if req.url.path == "/search":
            counts["search"] = counts.get("search", 0) + 1
            return httpx.Response(200, json={"sources": [SOURCE]})
        if req.url.path == "/generate/stream":
            counts["generate"] = counts.get("generate", 0) + 1
            body = json.loads(req.content)
            assert not any("PRIVATE ANSWER" in message["content"] for message in body["history"])
            frames = [{"type": "token", "text": "QLoRA "}, {"type": "token", "text": "uses quantization [1]."}]
            if stream_error:
                frames.append({"type": "error", "detail": "connection lost"})
            else:
                frames.append({"type": "done", "answer": "QLoRA uses quantization [1].", "model": "test-model", "output_tokens": 8})
            return httpx.Response(200, text="\n".join(map(json.dumps, frames)))
        raise AssertionError(f"Unexpected endpoint: {req.url}")
    return respond


async def run_until(store, client, request, expected="completed"):
    task = asyncio.create_task(worker(SimpleNamespace(state=SimpleNamespace(store=store, client=client))))
    try:
        for _ in range(400):
            if store.get(request)["status"] == expected:
                return
            await asyncio.sleep(.01)
        raise AssertionError(store.get(request))
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_worker_streams_sources_isolates_history_and_caches(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'flow.db'}")
    other = store.create_session()["session_id"]
    private = str(uuid4())
    store.enqueue(private, other, "Secret from another session")
    store.complete(private, "PRIVATE ANSWER", [])
    counts = {}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_nodes(counts))) as client:
            for index in range(2):
                session = store.create_session()["session_id"]
                request = str(uuid4())
                store.enqueue(request, session, "What is QLoRA?")
                await run_until(store, client, request)
                result = store.get(request)
                assert result["sources"] == [SOURCE]
                assert result["cache_hit"] is bool(index)
                events = store.events(session)
                if index == 0:
                    assert any(event["job"]["partial_answer"] for event in events)
                assert events[-1]["job"]["status"] == "completed"
    asyncio.run(scenario())
    assert counts == {"search": 1, "generate": 1}
    assert store.history(other)[0]["answer"] == "PRIVATE ANSWER"
    store.engine.dispose()


def test_interrupted_stream_retries_and_never_completes_partial_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_ATTEMPTS", "2")
    store = Store(f"sqlite:///{tmp_path / 'failure.db'}")
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "What is QLoRA?")
    counts = {}
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_nodes(counts, stream_error=True))) as client:
            await run_until(store, client, request, "failed")
    asyncio.run(scenario())
    result = store.get(request)
    assert result["attempts"] == 2
    assert result["answer"] is None and result["partial_answer"] == ""
    assert counts["generate"] == 2
    store.engine.dispose()


def test_cache_key_changes_with_history_corpus_model_language_and_filter():
    original = ["Question", [], [], "v1", "model1", "sk", 4]
    key = cache_key(*original)
    variants = [(1, [{"role": "user", "content": "another session context"}]),
                (2, ["doc-id"]), (3, "v2"), (4, "model2"), (5, "en"), (6, 3)]
    for index, value in variants:
        changed = list(original)
        changed[index] = value
        assert cache_key(*changed) != key
