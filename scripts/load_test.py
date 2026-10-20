"""Parallel users, two ordered turns each, idempotent resubmission and history isolation."""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

QUESTIONS = [
    "What is double quantization in QLoRA? Answer in one sentence.",
    "Why does the Transformer need positional encoding? Answer in one sentence.",
    "What is retrieval-augmented generation? Answer in one sentence.",
    "What is the purpose of paged optimizers in QLoRA? Answer in one sentence.",
    "What is multi-head attention? Answer in one sentence.",
    "Which parameters are trained by LoRA? Answer in one sentence.",
    "Why is dot-product attention scaled by the square root of the key dimension? Answer in one sentence.",
    "What is NormalFloat4 in QLoRA? Answer in one sentence.",
    "What role does the retriever play in RAG? Answer in one sentence.",
    "Does the Transformer use recurrence? Answer in one sentence.",
]


async def run(args):
    started = time.monotonic()
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        health = (await client.get("/api/health")).json()
        if not all(health.get(key) for key in ["application", "rag", "inference"]):
            raise RuntimeError("All three nodes must be online")
        async def user(index):
            response = await client.post("/api/sessions")
            response.raise_for_status()
            session = response.json()["session_id"]
            expected, results = [], []
            for turn in range(args.turns):
                question = QUESTIONS[(index + turn) % len(QUESTIONS)] + f" (Load test user {index + 1}, turn {turn + 1}.)"
                request = str(uuid4())
                payload = {"request_id": request, "question": question, "language": "en"}
                for _ in range(2):
                    response = await client.post(f"/api/sessions/{session}/questions", json=payload)
                    response.raise_for_status()
                    assert response.json()["request_id"] == request
                expected.append((request, question))
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    response = await client.get(f"/api/requests/{request}")
                    response.raise_for_status()
                    result = response.json()
                    if result["status"] == "failed":
                        raise RuntimeError(f"User {index + 1}: request failed: {result['error']}")
                    if result["status"] == "completed":
                        assert result["answer"] and result["session_id"] == session
                        results.append({key: result[key] for key in ["request_id", "total_ms", "queue_ms", "attempts", "cache_hit"]})
                        print(f"user={index + 1} turn={turn + 1} completed {result['total_ms'] / 1000:.1f}s", flush=True)
                        break
                    await asyncio.sleep(.5)
                else:
                    raise TimeoutError(f"User {index + 1} exceeded {args.timeout}s")
            response = await client.get(f"/api/sessions/{session}/messages")
            response.raise_for_status()
            actual = [(job["request_id"], job["question"]) for job in response.json()]
            assert actual == expected, f"History isolation/order failed for {session}"
            return {"session_id": session, "requests": results}
        users = await asyncio.gather(*(user(index) for index in range(args.users)))
        all_results = [item for user_result in users for item in user_result["requests"]]
        times = sorted(item["total_ms"] / 1000 for item in all_results)
        report = {"passed": True, "users": args.users, "turns_per_user": args.turns,
                  "requests": len(times), "elapsed_seconds": time.monotonic() - started,
                  "mean_response_seconds": sum(times) / len(times),
                  "p95_response_seconds": times[min(len(times) - 1, int(len(times) * .95))],
                  "cache_hits": sum(item["cache_hit"] for item in all_results),
                  "idempotency": True, "session_isolation": True, "history_order": True, "details": users}
        directory = Path(__file__).resolve().parents[1] / "data" / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"load-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2), flush=True)
        print(f"Report: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=10, choices=range(1, 51))
    parser.add_argument("--turns", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--timeout", type=int, default=1200)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
