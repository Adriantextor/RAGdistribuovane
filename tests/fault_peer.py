"""Loopback-only peer for deterministic deadline testing, never used by the application."""
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get('/health')
async def health():
    return {'status': 'ready', 'fingerprint': 'acceptance-timeout-peer'}


@app.post('/generate/stream')
async def generate():
    async def body():
        yield json.dumps({'type': 'token', 'text': 'incomplete test output'}) + '\n'
        await asyncio.sleep(30)
        yield json.dumps({'type': 'done', 'answer': 'late answer [1]'}) + '\n'
    return StreamingResponse(body(), media_type='application/x-ndjson')
