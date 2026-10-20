import time
from uuid import uuid4

import pytest

from services.app.store import Job, Store
from services.rag.chunking import chunks


@pytest.fixture
def store(tmp_path):
    value = Store(f"sqlite:///{tmp_path / 'test.db'}")
    yield value
    value.engine.dispose()


def test_idempotent_submission_and_payload_conflict(store):
    session = store.create_session()["session_id"]
    request = str(uuid4())
    first = store.enqueue(request, session, "What is RAG?")
    assert store.enqueue(request, session, "What is RAG?")["request_id"] == first["request_id"]
    assert len(store.history(session)) == 1
    with pytest.raises(ValueError):
        store.enqueue(request, session, "What is QLoRA?")


def test_histories_are_isolated_and_persist_across_restart(store):
    first, second = [store.create_session()["session_id"] for _ in range(2)]
    request = str(uuid4())
    store.enqueue(request, first, "Question A")
    store.enqueue(str(uuid4()), second, "Question B")
    store.complete(request, "Answer A", [{"page": 2}])
    restarted = Store(str(store.engine.url))
    try:
        assert restarted.history(first)[0]["answer"] == "Answer A"
        assert restarted.history(second)[0]["question"] == "Question B"
        assert restarted.history(second)[0]["answer"] is None
    finally:
        restarted.engine.dispose()


def test_expired_lease_recovers_but_attempts_are_bounded(store):
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "Question")
    assert store.claim(600, 2)["attempts"] == 1
    assert store.claim(600, 2) is None
    with store.sessions.begin() as db:
        db.get(Job, request).lease_until = time.time() - 1
    assert store.claim(600, 2)["attempts"] == 2
    with store.sessions.begin() as db:
        db.get(Job, request).lease_until = time.time() - 1
    assert store.claim(600, 2) is None
    assert store.get(request)["status"] == "failed"


def test_chunks_cover_the_entire_text_with_bounded_size():
    text = " ".join(f"token{i}" for i in range(800))
    parts = chunks(text)
    assert len(parts) > 1
    assert all(len(part) <= 1100 for part in parts)
    assert parts[0].startswith("token0 ")
    assert parts[-1].endswith("token799")
    for word in text.split():
        assert any(word in part for part in parts)


def test_blank_pdf_text_produces_no_chunks():
    assert chunks(" \n\t") == []


def test_retry_does_not_allow_later_question_to_overtake_same_session(store):
    session = store.create_session()["session_id"]
    first, second = str(uuid4()), str(uuid4())
    store.enqueue(first, session, "First")
    store.enqueue(second, session, "Follow-up")
    assert store.claim(600, 3)["request_id"] == first
    store.retry(first, 3)
    assert store.claim(600, 3) is None
    # Another session remains eligible while the first waits for its retry.
    other = store.create_session()["session_id"]
    third = str(uuid4())
    store.enqueue(third, other, "Independent question")
    assert store.claim(600, 3)["request_id"] == third


def test_old_worker_cannot_overwrite_a_reclaimed_job(store):
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "Question")
    first = store.claim(60, 3)
    store.partial(request, first["lease_owner"], "old partial")
    with store.sessions.begin() as db:
        db.get(Job, request).lease_until = time.time() - 1
    second = store.claim(60, 3)
    assert first["lease_owner"] != second["lease_owner"]
    assert store.get(request)["partial_answer"] == ""
    with pytest.raises(LookupError):
        store.complete(request, "stale result", [], first["lease_owner"])
    store.complete(request, "current result", [], second["lease_owner"])
    assert store.get(request)["answer"] == "current result"
    assert len(store.history(session)) == 1


def test_heartbeat_extends_lease_and_manual_retry_reuses_id(store):
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "Question")
    claim = store.claim(60, 1)
    store.renew(request, claim["lease_owner"], 120)
    with store.sessions() as db:
        assert db.get(Job, request).lease_until > time.time() + 100
    store.retry(request, 1, claim["lease_owner"])
    assert store.get(request)["status"] == "failed"
    result = store.manual_retry(request)
    assert result["status"] == "queued" and result["attempts"] == 0
    assert result["request_id"] == request
    assert len(store.history(session)) == 1


def test_cache_expiration_and_metrics_are_persistent(store):
    from services.app.store import CacheEntry
    store.cache_put("key", {"answer": "answer"}, 60)
    assert store.cache_get("key")["answer"] == "answer"
    with store.sessions.begin() as db:
        db.get(CacheEntry, "key").expires_at = time.time() - 1
    assert store.cache_get("key") is None
    session = store.create_session()["session_id"]
    request = str(uuid4())
    store.enqueue(request, session, "cached")
    store.complete(request, "answer", [], cache_hit=True)
    metrics = store.metrics()
    assert metrics["completed"] == 1 and metrics["cache_hits"] == 1
    assert metrics["cache_hit_ratio"] == 1


def test_additive_migration_preserves_prototype_history(tmp_path):
    from sqlalchemy import create_engine, text
    path = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(path)
    with engine.begin() as db:
        db.execute(text("CREATE TABLE conversations (id VARCHAR(36) PRIMARY KEY, created_at FLOAT)"))
        db.execute(text("CREATE TABLE jobs (id VARCHAR(36) PRIMARY KEY, session_id VARCHAR(36), question TEXT, status VARCHAR(20), answer TEXT, sources JSON, attempts INTEGER, next_attempt FLOAT, lease_until FLOAT, created_at FLOAT, completed_at FLOAT, error TEXT)"))
        db.execute(text("INSERT INTO conversations VALUES ('session', 1)"))
        db.execute(text("INSERT INTO jobs VALUES ('request', 'session', 'old question', 'completed', 'old answer', '[]', 1, 0, 0, 1, 2, NULL)"))
    engine.dispose()
    migrated = Store(path)
    try:
        assert migrated.history("session")[0]["answer"] == "old answer"
        assert migrated.history("session")[0]["partial_answer"] == ""
        migrated.enqueue(str(uuid4()), "session", "new question")
        assert len(migrated.history("session")) == 2
    finally:
        migrated.engine.dispose()
