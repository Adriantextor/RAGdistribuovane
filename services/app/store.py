import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, create_engine, func, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column, sessionmaker

from shared.database import add_columns


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    title: Mapped[str] = mapped_column(String(100), default="Nová konverzácia")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt: Mapped[float] = mapped_column(Float, default=0)
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    completed_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(36), nullable=True)
    partial_answer: Mapped[str] = mapped_column(Text, default="")
    document_ids: Mapped[list] = mapped_column(JSON, default=list)
    language: Mapped[str] = mapped_column(String(8), default="auto")
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0)
    generation_ms: Mapped[float] = mapped_column(Float, default=0)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(200), default="")
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class CacheEntry(Base):
    __tablename__ = "answer_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    expires_at: Mapped[float] = mapped_column(Float, index=True)


def as_dict(job):
    return {"request_id": job.id, "session_id": job.session_id, "question": job.question,
            "status": job.status, "answer": job.answer, "partial_answer": job.partial_answer,
            "sources": job.sources, "attempts": job.attempts, "error": job.error,
            "created_at": job.created_at, "started_at": job.started_at, "completed_at": job.completed_at,
            "document_ids": job.document_ids, "language": job.language, "cache_hit": job.cache_hit,
            "retrieval_ms": job.retrieval_ms, "generation_ms": job.generation_ms,
            "ttft_ms": job.ttft_ms, "output_tokens": job.output_tokens, "model": job.model,
            "queue_ms": max(0, ((job.started_at or time.time()) - job.created_at) * 1000),
            "total_ms": max(0, ((job.completed_at or time.time()) - job.created_at) * 1000),
            "updated_at": job.updated_at}


class Store:
    def __init__(self, url):
        if url.startswith("sqlite:///./data/"):
            Path("data").mkdir(exist_ok=True)
        args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=args, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        add_columns(self.engine, "conversations", {"title": "VARCHAR(100) NOT NULL DEFAULT 'Nová konverzácia'"})
        add_columns(self.engine, "jobs", {
            "lease_owner": "VARCHAR(36)", "partial_answer": "TEXT NOT NULL DEFAULT ''",
            "document_ids": "JSON NOT NULL DEFAULT '[]'", "language": "VARCHAR(8) NOT NULL DEFAULT 'auto'",
            "cache_hit": "BOOLEAN NOT NULL DEFAULT FALSE", "started_at": "FLOAT",
            "retrieval_ms": "FLOAT NOT NULL DEFAULT 0", "generation_ms": "FLOAT NOT NULL DEFAULT 0",
            "ttft_ms": "FLOAT", "output_tokens": "INTEGER NOT NULL DEFAULT 0",
            "model": "VARCHAR(200) NOT NULL DEFAULT ''", "updated_at": "FLOAT NOT NULL DEFAULT 0",
        })

    def _emit(self, db, job):
        job.updated_at = time.time()
        db.flush()
        db.add(JobEvent(session_id=job.session_id, request_id=job.id, payload=as_dict(job)))

    @staticmethod
    def _owned(db, request_id, owner):
        query = select(Job).where(Job.id == request_id).with_for_update()
        if owner is not None:
            query = query.where(Job.lease_owner == owner, Job.status.in_(["retrieving", "generating"]))
        job = db.scalar(query)
        if job is None:
            raise LookupError("Request lease is no longer owned by this worker")
        return job

    def ping(self):
        with self.sessions() as db:
            db.execute(select(1))
        return True

    def create_session(self):
        with self.sessions.begin() as db:
            conversation = Conversation(id=str(uuid4()))
            db.add(conversation)
        return {"session_id": conversation.id}

    def conversations(self):
        with self.sessions() as db:
            rows = db.execute(select(Conversation, func.count(Job.id)).outerjoin(Job)
                              .group_by(Conversation.id).order_by(Conversation.created_at.desc()).limit(200))
            return [{"session_id": row.id, "created_at": row.created_at, "title": row.title,
                     "message_count": count} for row, count in rows]

    def enqueue(self, request_id, session_id, question, document_ids=None, language="auto"):
        document_ids = sorted(set(document_ids or []))
        with self.sessions.begin() as db:
            conversation = db.scalar(select(Conversation).where(Conversation.id == session_id).with_for_update())
            if not conversation:
                raise LookupError("Unknown conversation")
            existing = db.get(Job, request_id)
            if existing:
                if (existing.session_id, existing.question, existing.document_ids, existing.language) != (
                        session_id, question, document_ids, language):
                    raise ValueError("request_id already used for a different request")
                return as_dict(existing)
            job = Job(id=request_id, session_id=session_id, question=question,
                      document_ids=document_ids, language=language)
            db.add(job)
            if conversation.title == "Nová konverzácia":
                conversation.title = question[:90]
            self._emit(db, job)
            return as_dict(job)

    def get(self, request_id):
        with self.sessions() as db:
            job = db.get(Job, request_id)
            return as_dict(job) if job else None

    def history(self, session_id):
        with self.sessions() as db:
            if not db.get(Conversation, session_id):
                raise LookupError("Unknown conversation")
            return [as_dict(job) for job in db.scalars(
                select(Job).where(Job.session_id == session_id).order_by(Job.created_at, Job.id))]

    def events(self, session_id, after=0):
        with self.sessions() as db:
            rows = db.scalars(select(JobEvent).where(JobEvent.session_id == session_id, JobEvent.id > after)
                              .order_by(JobEvent.id).limit(100))
            return [{"id": row.id, "job": row.payload} for row in rows]

    def event_cursor(self, session_id):
        with self.sessions() as db:
            return db.scalar(select(func.max(JobEvent.id)).where(JobEvent.session_id == session_id)) or 0

    def claim(self, lease_seconds, max_attempts):
        now = time.time()
        with self.sessions.begin() as db:
            expired = db.scalars(select(Job).where(Job.status.in_(["retrieving", "generating"]),
                                                  Job.lease_until < now).with_for_update(skip_locked=True))
            for job in expired:
                job.status, job.lease_until, job.lease_owner, job.partial_answer = "queued", 0, None, ""
                job.error = "Spracovanie bolo prerušené; obnovujem požiadavku."
                self._emit(db, job)
            exhausted = db.scalars(select(Job).where(Job.status == "queued", Job.attempts >= max_attempts)
                                   .with_for_update(skip_locked=True))
            for job in exhausted:
                job.status, job.error = "failed", "Bol dosiahnutý maximálny počet pokusov."
                self._emit(db, job)
            earlier = aliased(Job)
            unfinished_predecessor = select(earlier.id).where(
                earlier.session_id == Job.session_id,
                earlier.status.in_(["queued", "retrieving", "generating"]),
                (earlier.created_at < Job.created_at) |
                ((earlier.created_at == Job.created_at) & (earlier.id < Job.id)),
            ).exists()
            job = db.scalar(select(Job).where(Job.status == "queued", Job.next_attempt <= now,
                                              ~unfinished_predecessor)
                            .order_by(Job.created_at, Job.id).with_for_update(skip_locked=True).limit(1))
            if not job:
                return None
            job.status, job.lease_owner = "retrieving", str(uuid4())
            job.attempts += 1
            job.lease_until = now + lease_seconds
            job.started_at = job.started_at or now
            job.partial_answer, job.error = "", None
            self._emit(db, job)
            result = as_dict(job)
            result["lease_owner"] = job.lease_owner
            return result

    def renew(self, request_id, owner, lease_seconds):
        with self.sessions.begin() as db:
            job = self._owned(db, request_id, owner)
            job.lease_until = time.time() + lease_seconds

    def generating(self, request_id, sources=None, retrieval_ms=0, owner=None):
        with self.sessions.begin() as db:
            job = self._owned(db, request_id, owner)
            job.status = "generating"
            if sources is not None:
                job.sources = sources
            job.retrieval_ms = retrieval_ms
            self._emit(db, job)

    def partial(self, request_id, owner, answer, ttft_ms=None):
        with self.sessions.begin() as db:
            job = self._owned(db, request_id, owner)
            job.partial_answer = answer
            if job.ttft_ms is None and ttft_ms is not None:
                job.ttft_ms = ttft_ms
            self._emit(db, job)

    def complete(self, request_id, answer, sources, owner=None, **metrics):
        with self.sessions.begin() as db:
            job = self._owned(db, request_id, owner)
            job.status, job.answer, job.sources = "completed", answer, sources
            job.completed_at, job.lease_until, job.lease_owner, job.error = time.time(), 0, None, None
            job.partial_answer = ""
            for name in ("cache_hit", "generation_ms", "output_tokens", "model", "retrieval_ms", "ttft_ms"):
                if name in metrics:
                    setattr(job, name, metrics[name])
            self._emit(db, job)

    def retry(self, request_id, max_attempts, owner=None, reason=None, base_seconds=3, consume_attempt=True):
        with self.sessions.begin() as db:
            job = self._owned(db, request_id, owner)
            if not consume_attempt:
                job.attempts = max(0, job.attempts - 1)
            job.status = "failed" if job.attempts >= max_attempts else "queued"
            job.error = reason or ("Dočasná chyba; otázku automaticky zopakujem." if job.status == "queued"
                                   else "Bol dosiahnutý maximálny počet pokusov.")
            job.next_attempt = time.time() + min(base_seconds * 2 ** max(0, job.attempts - 1), 60)
            job.lease_until, job.lease_owner, job.partial_answer = 0, None, ""
            self._emit(db, job)

    def manual_retry(self, request_id):
        with self.sessions.begin() as db:
            job = db.get(Job, request_id, with_for_update=True)
            if job is None:
                raise LookupError("Unknown request")
            if job.status != "failed":
                raise ValueError("Only a failed request can be retried")
            job.status, job.attempts, job.next_attempt, job.error = "queued", 0, 0, None
            job.partial_answer, job.answer, job.completed_at, job.started_at = "", None, None, None
            self._emit(db, job)
            return as_dict(job)

    def cache_get(self, key):
        with self.sessions() as db:
            row = db.get(CacheEntry, key)
            return row.value if row and row.expires_at > time.time() else None

    def cache_put(self, key, value, ttl):
        if ttl <= 0:
            return
        if self.engine.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        values = {"key": key, "value": value, "expires_at": time.time() + ttl}
        with self.sessions.begin() as db:
            db.execute(insert(CacheEntry).values(**values).on_conflict_do_update(
                index_elements=[CacheEntry.key], set_=values))
            db.query(CacheEntry).filter(CacheEntry.expires_at < time.time()).delete()

    def metrics(self):
        with self.sessions() as db:
            counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
            samples = list(db.execute(select(Job.created_at, Job.started_at, Job.completed_at)
                                      .where(Job.status == "completed").order_by(Job.completed_at.desc()).limit(1000)))
            durations = sorted((end - start) for start, _, end in samples if end is not None)
            queue = [(started - created) for created, started, _ in samples if started]
            hits = db.scalar(select(func.count()).select_from(Job).where(Job.cache_hit.is_(True), Job.status == "completed"))
            cache_entries = db.scalar(select(func.count()).select_from(CacheEntry).where(CacheEntry.expires_at > time.time()))
            recent = [as_dict(job) for job in db.scalars(select(Job).order_by(Job.created_at.desc()).limit(30))]
            oldest = db.scalar(select(func.min(Job.created_at)).where(Job.status == "queued"))
            return {"queue_length": counts.get("queued", 0),
                    "active": counts.get("retrieving", 0) + counts.get("generating", 0),
                    "completed": counts.get("completed", 0), "failed": counts.get("failed", 0),
                    "states": counts, "cache_hits": hits, "cache_entries": cache_entries,
                    "cache_hit_ratio": hits / counts["completed"] if counts.get("completed") else 0,
                    "mean_response_seconds": sum(durations) / len(durations) if durations else None,
                    "p50_response_seconds": durations[len(durations) // 2] if durations else None,
                    "p95_response_seconds": durations[min(len(durations) - 1, int(len(durations) * .95))] if durations else None,
                    "mean_queue_seconds": sum(queue) / len(queue) if queue else None,
                    "oldest_queued_seconds": time.time() - oldest if oldest else 0, "recent": recent}
