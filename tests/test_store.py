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
    assert store.enqueue(request, session, "What is RAG?") == first
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
