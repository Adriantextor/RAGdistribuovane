import time
import threading
from functools import wraps
from pathlib import Path

from sqlalchemy import Float, Integer, String, Text, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class Corpus(Base):
    __tablename__ = "corpus"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class Catalog:
    def __init__(self, directory, model):
        self.directory, self.model = Path(directory), model
        self.lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "pdfs").mkdir(exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.directory / 'catalog.db'}",
                                    connect_args={"check_same_thread": False, "timeout": 30})
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        with self.sessions.begin() as db:
            if not db.get(Corpus, 1):
                db.add(Corpus(id=1, revision=0))
            for doc in db.scalars(select(Document).where(Document.status == "processing")):
                doc.status, doc.processed = "queued", 0
                doc.attempts = max(0, doc.attempts - 1)
                doc.error = "Import bol prerušený; automaticky sa obnoví."

    def path(self, document_id):
        return self.directory / "pdfs" / f"{document_id}.pdf"

    def data(self, doc):
        result = {name: getattr(doc, name) for name in (
            "id", "filename", "model", "status", "chunks", "processed", "pages", "attempts", "error", "created_at", "updated_at")}
        result["file_available"] = self.path(doc.id).is_file()
        if doc.model != self.model and doc.status == "completed":
            result["status"] = "needs_reindex"
        return result

    @serialized
    def adopt(self, records):
        with self.sessions.begin() as db:
            for key, record in records.items():
                if not db.get(Document, key):
                    db.add(Document(id=key, filename=record["filename"], model=self.model,
                                    status="completed", chunks=record["chunks"], processed=record["chunks"], pages=record["pages"]))
                    db.execute(update(Corpus).where(Corpus.id == 1).values(revision=Corpus.revision + 1))

    @serialized
    def enqueue(self, document_id, filename, content):
        # Persist the original first: accepted jobs can survive an API restart.
        target = self.path(document_id)
        if not target.exists():
            temp = target.with_suffix(".tmp")
            temp.write_bytes(content)
            temp.replace(target)
        with self.sessions.begin() as db:
            doc = db.get(Document, document_id)
            if doc and doc.status in {"queued", "processing"}:
                return self.data(doc)
            if doc and doc.status == "completed" and doc.model == self.model:
                return self.data(doc)
            if doc:
                doc.status, doc.processed, doc.attempts, doc.error, doc.next_attempt = "queued", 0, 0, None, 0
                doc.model = self.model
            else:
                doc = Document(id=document_id, filename=filename, model=self.model)
                db.add(doc)
            db.flush()
            return self.data(doc)

    def documents(self):
        with self.sessions() as db:
            return [self.data(doc) for doc in db.scalars(select(Document).where(Document.status != "deleted")
                                                       .order_by(Document.created_at.desc()))]

    def ready_ids(self):
        with self.sessions() as db:
            return list(db.scalars(select(Document.id).where(Document.status == "completed", Document.model == self.model)))

    def revision(self):
        with self.sessions() as db:
            return db.get(Corpus, 1).revision

    @serialized
    def claim(self):
        with self.sessions.begin() as db:
            doc = db.scalar(select(Document).where(Document.status == "queued", Document.model == self.model,
                                                   Document.next_attempt <= time.time()).order_by(Document.created_at).limit(1))
            if not doc:
                return None
            doc.status, doc.attempts, doc.processed, doc.error = "processing", doc.attempts + 1, 0, None
            doc.updated_at = time.time()
            return self.data(doc)

    @serialized
    def progress(self, key, processed, chunks, pages):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            doc.processed, doc.chunks, doc.pages, doc.updated_at = processed, chunks, pages, time.time()

    @serialized
    def complete(self, key):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            doc.status, doc.error, doc.updated_at = "completed", None, time.time()
            db.execute(update(Corpus).where(Corpus.id == 1).values(revision=Corpus.revision + 1))

    @serialized
    def fail(self, key, error, max_attempts, permanent=False):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            doc.status = "failed" if permanent or doc.attempts >= max_attempts else "queued"
            doc.error, doc.updated_at = error, time.time()
            doc.next_attempt = time.time() + min(3 * 2 ** doc.attempts, 60)

    @serialized
    def retry(self, key):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            if not doc:
                raise LookupError("Dokument neexistuje.")
            if doc.status != "failed" or not self.path(key).is_file():
                raise ValueError("Opakovať možno iba zlyhaný import s uloženým PDF.")
            doc.status, doc.attempts, doc.next_attempt, doc.error = "queued", 0, 0, None
            return self.data(doc)

    @serialized
    def remove(self, key):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            if not doc:
                raise LookupError("Dokument neexistuje.")
            if doc.status in {"queued", "processing"}:
                raise ValueError("Počkaj na dokončenie importu.")
            if doc.status != "deleted":
                doc.status, doc.updated_at = "deleted", time.time()
                db.execute(update(Corpus).where(Corpus.id == 1).values(revision=Corpus.revision + 1))
            return {"id": key, "status": "deleted"}

    @serialized
    def interrupt(self, key):
        with self.sessions.begin() as db:
            doc = db.get(Document, key)
            if doc and doc.status == "processing":
                doc.status, doc.processed, doc.next_attempt = "queued", 0, 0
                doc.attempts = max(0, doc.attempts - 1)
                doc.error = "RAG uzol sa reštartoval; import sa obnoví."
