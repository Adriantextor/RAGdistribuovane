import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from services.app.store import Store
from shared.config import settings

log = logging.getLogger(__name__)


async def node_health(client, url):
    try:
        response = await client.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def worker(app):
    config = settings()
    while True:
        try:
            ready = await asyncio.gather(node_health(app.state.client, config.rag_url),
                                         node_health(app.state.client, config.inference_url))
            if not all(ready):
                # An offline node does not exhaust the retry budget.
                await asyncio.sleep(5)
                continue
            job = await asyncio.to_thread(app.state.store.claim, config.lease_seconds, config.max_attempts)
            if not job:
                await asyncio.sleep(1)
                continue
            try:
                result = await app.state.client.post(f"{config.rag_url}/search",
                    json={"question": job["question"]}, timeout=240)
                result.raise_for_status()
                sources = result.json()["sources"]
                history = await asyncio.to_thread(app.state.store.history, job["session_id"])
                messages = []
                for item in history:
                    if item["request_id"] == job["request_id"]:
                        break
                    if item["status"] == "completed":
                        messages.extend([{"role": "user", "content": item["question"][:1000]},
                                         {"role": "assistant", "content": item["answer"][:1000]}])
                await asyncio.to_thread(app.state.store.generating, job["request_id"])
                result = await app.state.client.post(f"{config.inference_url}/generate", json={
                    "question": job["question"], "sources": sources, "history": messages[-4:],
                }, timeout=240)
                result.raise_for_status()
                await asyncio.to_thread(app.state.store.complete, job["request_id"], result.json()["answer"], sources)
            except Exception:
                log.exception("Request %s failed", job["request_id"])
                await asyncio.to_thread(app.state.store.retry, job["request_id"], config.max_attempts)
        except Exception:
            log.exception("Worker iteration failed")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app):
    app.state.store = await asyncio.to_thread(Store, settings().database_url)
    app.state.client = httpx.AsyncClient(headers=settings().headers())
    task = asyncio.create_task(worker(app))
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    await app.state.client.aclose()
    app.state.store.engine.dispose()


app = FastAPI(title="AI Research Assistant · Application node", lifespan=lifespan)


class Question(BaseModel):
    request_id: UUID
    question: str = Field(min_length=1, max_length=4000)


@app.get("/api/health")
async def health():
    rag, inference = await asyncio.gather(node_health(app.state.client, settings().rag_url),
                                           node_health(app.state.client, settings().inference_url))
    return {"application": True, "rag": rag, "inference": inference}


@app.post("/api/sessions", status_code=201)
def create_session():
    return app.state.store.create_session()


@app.get("/api/sessions")
def conversations():
    return app.state.store.conversations()


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: UUID):
    try:
        return app.state.store.history(str(session_id))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/sessions/{session_id}/questions", status_code=202)
def enqueue(session_id: UUID, body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question cannot be blank")
    try:
        return app.state.store.enqueue(str(body.request_id), str(session_id), question)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, IntegrityError) as exc:
        raise HTTPException(409, "request_id conflicts with an existing question") from exc


@app.get("/api/requests/{request_id}")
def request_status(request_id: UUID):
    result = app.state.store.get(str(request_id))
    if not result:
        raise HTTPException(404, "Unknown request")
    return result


@app.get("/api/metrics")
def metrics():
    return app.state.store.metrics()


@app.post("/api/documents")
async def upload(file: UploadFile):
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Maximum PDF size is 20 MB")
    try:
        response = await app.state.client.post(f"{settings().rag_url}/documents", files={
            "file": (file.filename or "document.pdf", content, "application/pdf")}, timeout=1200)
        if response.is_error:
            raise HTTPException(response.status_code, "Document import failed; check RAG service logs")
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(503, "RAG node unavailable or import timed out") from exc


dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
