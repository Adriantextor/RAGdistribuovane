import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager, suppress
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient, models

from services.rag.catalog import Catalog
from services.rag.chunking import chunks
from services.rag.retrieval import rank_chunks
from shared.auth import require_node
from shared.config import settings
from shared.schemas import SearchInput
from shared.text import expanded_query

log = logging.getLogger(__name__)


def collection_name():
    digest = hashlib.sha256(settings().embed_model.encode()).hexdigest()[:12]
    return f"papers_v1_{digest}"


async def all_points():
    if not await app.state.qdrant.collection_exists(collection_name()):
        return []
    records, offset = [], None
    while True:
        points, offset = await app.state.qdrant.scroll(collection_name(), limit=256, offset=offset,
                                                       with_payload=True, with_vectors=False)
        for point in points:
            payload = point.payload or {}
            if all(key in payload for key in ["document", "page", "text", "document_id"]):
                records.append({"id": str(point.id), **payload})
        if offset is None:
            return records


async def initialize():
    while True:
        try:
            records = await all_points()
            adopted = {}
            for record in records:
                item = adopted.setdefault(record["document_id"], {"filename": record["document"], "pages": 0, "chunks": 0})
                item["pages"] = max(item["pages"], record["page"])
                item["chunks"] += 1
            await asyncio.to_thread(app.state.catalog.adopt, adopted)
            app.state.ready.set()
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("Waiting for vector database before restoring document catalog")
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app):
    config = settings()
    app.state.qdrant = AsyncQdrantClient(url=config.qdrant_url, timeout=30)
    app.state.client = httpx.AsyncClient(base_url=config.inference_url,
                                        headers=config.headers(), timeout=config.request_timeout_seconds)
    app.state.catalog = Catalog(config.rag_data_dir, config.embed_model)
    app.state.ready = asyncio.Event()
    app.state.write_lock = asyncio.Lock()
    app.state.collection_lock = asyncio.Lock()
    app.state.records = None
    app.state.records_revision = None
    tasks = [asyncio.create_task(initialize())]
    tasks += [asyncio.create_task(import_worker()) for _ in range(config.import_concurrency)]
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task
    await app.state.client.aclose()
    await app.state.qdrant.close()
    app.state.catalog.engine.dispose()


app = FastAPI(title="AI Research Assistant · RAG node", lifespan=lifespan,
              dependencies=[Depends(require_node)])


async def vectors(texts, purpose):
    try:
        response = await app.state.client.post("/embed", json={"texts": texts, "purpose": purpose})
        response.raise_for_status()
        data = response.json()
        if data["model"] != settings().embed_model or len(data["vectors"]) != len(texts):
            raise ValueError("Embedding configuration mismatch")
        return data["vectors"]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(503, "Embedding uzol nie je dostupný alebo má iný model.") from exc


@app.get("/health")
async def health():
    try:
        await app.state.qdrant.get_collections()
        if not app.state.ready.is_set():
            raise HTTPException(503, "Obnovujem katalóg dokumentov.")
        return {"node": "rag", "status": "ready", "embedding_model": settings().embed_model,
                "documents": len(await asyncio.to_thread(app.state.catalog.ready_ids)),
                "import_workers": settings().import_concurrency}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Vektorová databáza nie je dostupná.") from exc


def extract(content):
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("Šifrované PDF nie je podporované.")
    if len(reader.pages) > 200:
        raise ValueError("Dokument presahuje 200 strán.")
    result = [(page_number, text) for page_number, page in enumerate(reader.pages, 1)
              for text in chunks(page.extract_text() or "")]
    if not result:
        raise ValueError("PDF neobsahuje text; sken vyžaduje OCR.")
    return result


async def process_document(doc):
    try:
        content = await asyncio.to_thread(app.state.catalog.path(doc["id"]).read_bytes)
        parts = await asyncio.to_thread(extract, content)
    except Exception:
        await asyncio.to_thread(app.state.catalog.fail, doc["id"],
            "PDF sa nepodarilo načítať. Podporovaný je textový, nešifrovaný PDF do 200 strán.",
            settings().max_attempts, True)
        return
    pages = max(page for page, _ in parts)
    await asyncio.to_thread(app.state.catalog.progress, doc["id"], 0, len(parts), pages)
    # A retried import replaces its uncommitted partial points. Search only sees completed documents.
    if await app.state.qdrant.collection_exists(collection_name()):
        await app.state.qdrant.delete(collection_name(), points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc["id"]))])), wait=True)
    size = settings().embedding_batch_size
    for start in range(0, len(parts), size):
        batch = parts[start:start + size]
        embeddings = await vectors([text for _, text in batch], "document")
        async with app.state.collection_lock:
            if not await app.state.qdrant.collection_exists(collection_name()):
                await app.state.qdrant.create_collection(collection_name(), vectors_config=models.VectorParams(
                    size=len(embeddings[0]), distance=models.Distance.COSINE))
        points = [models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{doc['id']}:{start + offset}")), vector=vector,
            payload={"document_id": doc["id"], "document": doc["filename"], "page": page,
                     "text": text, "chunk": start + offset},
        ) for offset, ((page, text), vector) in enumerate(zip(batch, embeddings))]
        await app.state.qdrant.upsert(collection_name(), points=points, wait=True)
        await asyncio.to_thread(app.state.catalog.progress, doc["id"], min(start + size, len(parts)), len(parts), pages)
    await asyncio.to_thread(app.state.catalog.complete, doc["id"])


async def import_worker():
    await app.state.ready.wait()
    while True:
        doc = None
        try:
            response = await app.state.client.get("/health", timeout=5)
            response.raise_for_status()
            await app.state.qdrant.get_collections()
            async with app.state.write_lock:
                doc = await asyncio.to_thread(app.state.catalog.claim)
            if not doc:
                await asyncio.sleep(1)
                continue
            await process_document(doc)
        except asyncio.CancelledError:
            if doc:
                await asyncio.to_thread(app.state.catalog.interrupt, doc["id"])
            raise
        except Exception as exc:
            log.warning("Document import interrupted: %s", type(exc).__name__)
            if doc:
                await asyncio.to_thread(app.state.catalog.fail, doc["id"],
                    "Import sa prerušil. Po dostupnosti uzlov sa automaticky zopakuje.", settings().max_attempts)
            await asyncio.sleep(2)


@app.get("/documents")
async def documents():
    return {"documents": await asyncio.to_thread(app.state.catalog.documents)}


@app.get("/manifest")
async def manifest():
    if not app.state.ready.is_set():
        raise HTTPException(503, "Obnovujem index.")
    revision = await asyncio.to_thread(app.state.catalog.revision)
    return {"revision": f"{collection_name()}:{revision}", "embedding_model": settings().embed_model,
            "document_ids": await asyncio.to_thread(app.state.catalog.ready_ids)}


@app.post("/documents", status_code=202)
async def import_document(file: UploadFile):
    if not app.state.ready.is_set():
        raise HTTPException(503, "Obnovujem index; skús import o chvíľu.")
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Maximálna veľkosť PDF je 20 MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(422, "Súbor nie je platný PDF.")
    document_id = hashlib.sha256(content).hexdigest()
    filename = Path(file.filename or "document.pdf").name[:200]
    async with app.state.write_lock:
        result = await asyncio.to_thread(app.state.catalog.enqueue, document_id, filename, content)
    return {**result, "document_id": document_id, "document": result["filename"]}


@app.post("/documents/{document_id}/retry", status_code=202)
async def retry_document(document_id: str):
    try:
        return await asyncio.to_thread(app.state.catalog.retry, document_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/documents/{document_id}")
async def remove_document(document_id: str):
    try:
        return await asyncio.to_thread(app.state.catalog.remove, document_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


async def ready_records():
    revision = await asyncio.to_thread(app.state.catalog.revision)
    if app.state.records_revision != revision:
        ready = set(await asyncio.to_thread(app.state.catalog.ready_ids))
        app.state.records = [record for record in await all_points() if record["document_id"] in ready]
        app.state.records_revision = revision
    return app.state.records or []


@app.post("/search")
async def search(body: SearchInput):
    records = await ready_records()
    if body.document_ids:
        selected = set(body.document_ids)
        records = [record for record in records if record["document_id"] in selected]
    if not records:
        return {"sources": []}
    embedding = (await vectors([expanded_query(body.question)], "query"))[0]
    ids = sorted({record["document_id"] for record in records})
    result = await app.state.qdrant.query_points(collection_name(), query=embedding, limit=40,
        query_filter=models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=ids))]),
        with_payload=False)
    ranked = await asyncio.to_thread(rank_chunks, body.question, records,
        [(str(point.id), point.score) for point in result.points], body.top_k, settings().retrieval_min_score)
    return {"sources": [{key: record[key] for key in ["id", "document", "document_id", "page", "text", "score"]}
                        for record in ranked]}
