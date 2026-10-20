import asyncio
import hashlib
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient, models

from services.rag.chunking import chunks
from shared.auth import require_node
from shared.config import settings
from shared.schemas import SearchInput


def collection_name():
    digest = hashlib.sha256(settings().embed_model.encode()).hexdigest()[:12]
    return f"papers_v1_{digest}"


@asynccontextmanager
async def lifespan(app):
    app.state.qdrant = AsyncQdrantClient(url=settings().qdrant_url, timeout=30)
    app.state.client = httpx.AsyncClient(base_url=settings().inference_url,
                                        headers=settings().headers(), timeout=240)
    app.state.import_lock = asyncio.Lock()
    yield
    await app.state.client.aclose()
    await app.state.qdrant.close()


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
        raise HTTPException(503, "Embedding node unavailable or misconfigured") from exc


@app.get("/health")
async def health():
    try:
        await app.state.qdrant.get_collections()
        return {"node": "rag", "status": "ready", "embedding_model": settings().embed_model}
    except Exception as exc:
        raise HTTPException(503, "Vector database unavailable") from exc


def extract(content):
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF is not supported")
    if len(reader.pages) > 200:
        raise ValueError("PDF exceeds 200 pages")
    result = [(page_number, text) for page_number, page in enumerate(reader.pages, 1)
              for text in chunks(page.extract_text() or "")]
    if not result:
        raise ValueError("PDF has no extractable text; scanned PDFs require OCR")
    return result


@app.post("/documents")
async def import_document(file: UploadFile):
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Maximum PDF size is 20 MB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(422, "Expected a PDF file")
    try:
        parts = await asyncio.to_thread(extract, content)
    except Exception as exc:
        raise HTTPException(422, "Cannot read PDF (encrypted, scanned, invalid, or over 200 pages)") from exc
    document_id = hashlib.sha256(content).hexdigest()
    filename = Path(file.filename or "document.pdf").name[:200]
    # Only one import at a time for the 8 GB development Mac. Embeddings use batches of 8.
    async with app.state.import_lock:
        for start in range(0, len(parts), 8):
            batch = parts[start:start + 8]
            embeddings = await vectors([text for _, text in batch], "document")
            if not await app.state.qdrant.collection_exists(collection_name()):
                await app.state.qdrant.create_collection(
                    collection_name(), vectors_config=models.VectorParams(
                        size=len(embeddings[0]), distance=models.Distance.COSINE))
            points = [models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{document_id}:{start + offset}")),
                vector=vector,
                payload={"document_id": document_id, "document": filename,
                         "page": page, "text": text, "chunk": start + offset},
            ) for offset, ((page, text), vector) in enumerate(zip(batch, embeddings))]
            await app.state.qdrant.upsert(collection_name(), points=points, wait=True)
    return {"document_id": document_id, "document": filename, "chunks": len(parts)}


@app.post("/search")
async def search(body: SearchInput):
    if not await app.state.qdrant.collection_exists(collection_name()):
        return {"sources": []}
    embedding = (await vectors([body.question], "query"))[0]
    result = await app.state.qdrant.query_points(
        collection_name(), query=embedding, limit=body.top_k, with_payload=True)
    return {"sources": [{"id": str(point.id), "score": point.score,
                         "document": point.payload["document"], "page": point.payload["page"],
                         "text": point.payload["text"]} for point in result.points]}
