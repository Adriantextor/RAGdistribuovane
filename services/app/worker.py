import asyncio
import hashlib
import json
import logging
import time
from contextlib import suppress

import httpx

from shared.config import settings
from shared.text import PROMPT_VERSION, RETRIEVAL_VERSION, language_of, refusal

log = logging.getLogger(__name__)


class ProcessingError(Exception):
    pass


async def node_probe(client, url):
    try:
        response = await client.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        return {**result, "available": True}
    except (httpx.HTTPError, ValueError):
        return {"available": False}


async def node_health(client, url):
    return (await node_probe(client, url))["available"]


def cache_key(question, history, documents, revision, fingerprint, language, top_k):
    payload = [" ".join(question.split()), history, sorted(documents), revision, fingerprint,
               language, top_k, PROMPT_VERSION, RETRIEVAL_VERSION]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def heartbeat(store, job):
    while True:
        await asyncio.sleep(settings().lease_seconds / 3)
        await asyncio.to_thread(store.renew, job["request_id"], job["lease_owner"], settings().lease_seconds)


async def process_job(app, job, inference):
    config, store, client = settings(), app.state.store, app.state.client
    owner, request_id = job["lease_owner"], job["request_id"]
    history = await asyncio.to_thread(store.history, job["session_id"])
    messages = []
    previous_question = ""
    for item in history:
        if item["request_id"] == request_id:
            break
        if item["status"] == "completed":
            previous_question = item["question"]
            messages.extend([{"role": "user", "content": item["question"][:700]},
                             {"role": "assistant", "content": item["answer"][:700]}])
    messages = messages[-2:]
    language = job["language"] if job["language"] != "auto" else language_of(job["question"])
    response = await client.get(f"{config.rag_url}/manifest", timeout=15)
    response.raise_for_status()
    manifest = response.json()
    key = cache_key(job["question"], messages, job["document_ids"], manifest["revision"],
                    inference.get("fingerprint", config.llm_model), language, config.retrieval_top_k)
    cached = await asyncio.to_thread(store.cache_get, key) if config.cache_ttl_seconds else None
    if cached:
        await asyncio.to_thread(store.complete, request_id, cached["answer"], cached["sources"], owner,
                                cache_hit=True, model=cached.get("model", config.llm_model))
        return
    started = time.monotonic()
    search_question = job["question"]
    if previous_question and len(search_question.split()) < 7:
        search_question = previous_question[:800] + "\n" + search_question
    response = await client.post(f"{config.rag_url}/search", json={
        "question": search_question[:4000], "document_ids": job["document_ids"], "top_k": config.retrieval_top_k,
    }, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    sources = response.json()["sources"]
    retrieval_ms = (time.monotonic() - started) * 1000
    await asyncio.to_thread(store.generating, request_id, sources, retrieval_ms, owner)
    generation_start, last_flush, answer, final = time.monotonic(), 0, "", None
    if not sources:
        final = {"answer": refusal(language), "model": config.llm_model, "output_tokens": 0}
    else:
        async with client.stream("POST", f"{config.inference_url}/generate/stream", json={
            "question": job["question"], "sources": sources, "history": messages, "language": language,
        }, timeout=config.request_timeout_seconds) as response:
            response.raise_for_status()
            first_token_ms = None
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("type") == "error":
                    raise ProcessingError("Inference prerušila generovanie.")
                if frame.get("type") == "token":
                    answer += frame.get("text", "")
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - generation_start) * 1000
                    if time.monotonic() - last_flush >= .15:
                        await asyncio.to_thread(store.partial, request_id, owner, answer, first_token_ms)
                        last_flush = time.monotonic()
                elif frame.get("type") == "done":
                    final = frame
        if not final or not final.get("answer", "").strip():
            raise ProcessingError("Stream odpovede sa skončil bez potvrdenia dokončenia.")
    generation_ms = (time.monotonic() - generation_start) * 1000
    await asyncio.to_thread(store.complete, request_id, final["answer"], sources, owner,
                            generation_ms=generation_ms, output_tokens=final.get("output_tokens", 0),
                            model=final.get("model", config.llm_model))
    await asyncio.to_thread(store.cache_put, key, {"answer": final["answer"], "sources": sources,
                                                  "model": final.get("model", config.llm_model)}, config.cache_ttl_seconds)


async def worker(app, worker_id=0):
    config = settings()
    while True:
        job = None
        try:
            rag, inference = await asyncio.gather(node_probe(app.state.client, config.rag_url),
                                                  node_probe(app.state.client, config.inference_url))
            if not rag["available"] or not inference["available"]:
                await asyncio.sleep(max(config.worker_poll_seconds, 2))
                continue
            job = await asyncio.to_thread(app.state.store.claim, config.lease_seconds, config.max_attempts)
            if not job:
                await asyncio.sleep(config.worker_poll_seconds)
                continue
            renewal = asyncio.create_task(heartbeat(app.state.store, job))
            try:
                await asyncio.wait_for(process_job(app, job, inference), timeout=config.job_timeout_seconds)
            except asyncio.CancelledError:
                # Graceful restart returns the job immediately; a hard crash uses the expiring lease.
                with suppress(LookupError):
                    await asyncio.to_thread(app.state.store.retry, job["request_id"], config.max_attempts,
                        job["lease_owner"], "Aplikácia sa reštartovala; otázka zostala vo fronte.",
                        config.retry_base_seconds, False)
                raise
            except Exception as exc:
                log.warning("Worker %s request %s: %s", worker_id, job["request_id"], type(exc).__name__)
                timeout = isinstance(exc, (TimeoutError, httpx.TimeoutException))
                reason = "Spracovanie prekročilo časový limit." if timeout else "Spracovanie sa prerušilo; pripravujem ďalší pokus."
                with suppress(LookupError):
                    await asyncio.to_thread(app.state.store.retry, job["request_id"], config.max_attempts,
                                            job["lease_owner"], reason, config.retry_base_seconds)
            finally:
                renewal.cancel()
                with suppress(asyncio.CancelledError, LookupError):
                    await renewal
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Worker loop failed")
            await asyncio.sleep(max(config.worker_poll_seconds, 2))
