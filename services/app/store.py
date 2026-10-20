import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


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


def as_dict(job):
    return {"request_id": job.id, "session_id": job.session_id, "question": job.question,
            "status": job.status, "answer": job.answer, "sources": job.sources,
            "attempts": job.attempts, "error": job.error, "created_at": job.created_at,
            "completed_at": job.completed_at}


class Store:
    def __init__(self, url):
        if url.startswith("sqlite:///./data/"):
            Path("data").mkdir(exist_ok=True)
        args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=args, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def create_session(self):
        with self.sessions.begin() as db:
            conversation = Conversation(id=str(uuid4()))
            db.add(conversation)
        return {"session_id": conversation.id}

    def conversations(self):
        with self.sessions() as db:
            return [{"session_id": row.id, "created_at": row.created_at}
                    for row in db.scalars(select(Conversation).order_by(Conversation.created_at.desc()))]

    def enqueue(self, request_id, session_id, question):
        with self.sessions.begin() as db:
            # Serialize submissions per conversation, including concurrent retries.
            conversation = db.scalar(select(Conversation).where(Conversation.id == session_id).with_for_update())
            if not conversation:
                raise LookupError("Unknown conversation")
            existing = db.get(Job, request_id)
            if existing:
                if existing.session_id != session_id or existing.question != question:
                    raise ValueError("request_id already used for a different request")
                return as_dict(existing)
            job = Job(id=request_id, session_id=session_id, question=question)
            db.add(job)
            db.flush()
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

    def claim(self, lease_seconds, max_attempts):
        now = time.time()
        with self.sessions.begin() as db:
            # Expired leases recover work interrupted by a process/machine restart.
            db.execute(update(Job).where(Job.status.in_(["retrieving", "generating"]),
                                         Job.lease_until < now).values(status="queued", lease_until=0))
            db.execute(update(Job).where(Job.status == "queued", Job.attempts >= max_attempts)
                       .values(status="failed", error="Maximum processing attempts reached"))
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
            job.status = "retrieving"
            job.attempts += 1
            job.lease_until = now + lease_seconds
            return as_dict(job)

    def generating(self, request_id):
        with self.sessions.begin() as db:
            db.get(Job, request_id).status = "generating"

    def complete(self, request_id, answer, sources):
        with self.sessions.begin() as db:
            job = db.get(Job, request_id)
            job.status, job.answer, job.sources = "completed", answer, sources
            job.completed_at, job.lease_until, job.error = time.time(), 0, None

    def retry(self, request_id, max_attempts):
        with self.sessions.begin() as db:
            job = db.get(Job, request_id)
            job.status = "failed" if job.attempts >= max_attempts else "queued"
            job.error = "A processing step failed; retry scheduled" if job.status == "queued" else "Maximum processing attempts reached"
            job.next_attempt = time.time() + min(5 * 2 ** job.attempts, 60)
            job.lease_until = 0

    def metrics(self):
        with self.sessions() as db:
            jobs = list(db.scalars(select(Job)))
            completed = [job.completed_at - job.created_at for job in jobs if job.completed_at]
            return {"queue_length": sum(job.status == "queued" for job in jobs),
                    "active": sum(job.status in {"retrieving", "generating"} for job in jobs),
                    "completed": len(completed),
                    "failed": sum(job.status == "failed" for job in jobs),
                    "mean_response_seconds": sum(completed) / len(completed) if completed else None}
