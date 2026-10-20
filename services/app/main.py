import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal
from uuid import UUID

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from services.app.store import Store
from services.app.worker import node_health, node_probe, worker
from shared.config import settings


@asynccontextmanager
async def lifespan(app):
    app.state.store = await asyncio.to_thread(Store, settings().database_url)
    app.state.client = httpx.AsyncClient(headers=settings().headers(), limits=httpx.Limits(max_connections=30))
    tasks = [asyncio.create_task(worker(app, index)) for index in range(settings().worker_concurrency)]
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task
    await app.state.client.aclose()
    app.state.store.engine.dispose()


app = FastAPI(title="AI Research Assistant · Application node", lifespan=lifespan)


class Question(BaseModel):
    request_id: UUID
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    language: Literal["auto", "sk", "en"] = "auto"


@app.get("/api/health")
async def health():
    rag, inference = await asyncio.gather(node_probe(app.state.client, settings().rag_url),
                                         node_probe(app.state.client, settings().inference_url))
    try:
        database = await asyncio.to_thread(app.state.store.ping)
    except Exception:
        database = False
    return {"application": database, "rag": rag["available"], "inference": inference["available"],
            "details": {"rag": rag, "inference": inference}, "worker_concurrency": settings().worker_concurrency}


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
        raise HTTPException(404, "Konverzácia neexistuje.") from exc


@app.post("/api/sessions/{session_id}/questions", status_code=202)
def enqueue(session_id: UUID, body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Otázka nesmie byť prázdna.")
    try:
        return app.state.store.enqueue(str(body.request_id), str(session_id), question, body.document_ids, body.language)
    except LookupError as exc:
        raise HTTPException(404, "Konverzácia neexistuje.") from exc
    except (ValueError, IntegrityError) as exc:
        raise HTTPException(409, "request_id už patrí inej otázke alebo konfigurácii.") from exc


@app.get("/api/requests/{request_id}")
def request_status(request_id: UUID):
    result = app.state.store.get(str(request_id))
    if not result:
        raise HTTPException(404, "Požiadavka neexistuje.")
    return result


@app.post("/api/requests/{request_id}/retry", status_code=202)
def retry(request_id: UUID):
    try:
        return app.state.store.manual_retry(str(request_id))
    except LookupError as exc:
        raise HTTPException(404, "Požiadavka neexistuje.") from exc
    except ValueError as exc:
        raise HTTPException(409, "Opakovať možno iba zlyhanú požiadavku.") from exc


@app.get("/api/sessions/{session_id}/events")
async def events(session_id: UUID, request: Request, last_event_id: str = Header(default="")):
    session_id = str(session_id)
    cursor = await asyncio.to_thread(app.state.store.event_cursor, session_id)
    try:
        initial = await asyncio.to_thread(app.state.store.history, session_id)
    except LookupError as exc:
        raise HTTPException(404, "Konverzácia neexistuje.") from exc
    # A full initial snapshot makes reload and EventSource reconnection independent of old token frames.
    async def stream():
        nonlocal cursor
        yield "retry: 1500\n\n"
        yield f"event: snapshot\ndata: {json.dumps(initial, ensure_ascii=False)}\n\n"
        idle = 0
        while not await request.is_disconnected():
            rows = await asyncio.to_thread(app.state.store.events, session_id, cursor)
            for event in rows:
                cursor = event["id"]
                yield f"id: {cursor}\nevent: job\ndata: {json.dumps(event['job'], ensure_ascii=False)}\n\n"
            idle += 1
            if idle >= 40:
                yield ": heartbeat\n\n"
                idle = 0
            await asyncio.sleep(.25)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.get("/api/metrics")
def metrics():
    return app.state.store.metrics()


async def rag_proxy(method, path, **kwargs):
    try:
        response = await app.state.client.request(method, settings().rag_url + path, **kwargs)
        if response.is_error:
            detail = "RAG služba požiadavku odmietla."
            with suppress(ValueError):
                detail = response.json().get("detail", detail)
            raise HTTPException(response.status_code, detail)
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(503, "RAG uzol nie je dostupný; otázky zostávajú vo fronte.") from exc


@app.get("/api/documents")
async def documents():
    return await rag_proxy("GET", "/documents", timeout=15)


@app.post("/api/documents", status_code=202)
async def upload(file: UploadFile):
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Maximálna veľkosť PDF je 20 MB.")
    return await rag_proxy("POST", "/documents", files={
        "file": (file.filename or "document.pdf", content, "application/pdf")}, timeout=60)


@app.post("/api/documents/{document_id}/retry", status_code=202)
async def retry_document(document_id: str):
    return await rag_proxy("POST", f"/documents/{document_id}/retry", timeout=15)


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str):
    return await rag_proxy("DELETE", f"/documents/{document_id}", timeout=15)


dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
