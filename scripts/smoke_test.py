"""Exercise the real stack. Requires running app, RAG, inference and databases."""
import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--import-papers", action="store_true")
    parser.add_argument("--question", default="What is QLoRA and how does it reduce memory usage? Give a short answer.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    with httpx.Client(base_url=args.url, timeout=1200) as client:
        response = client.get("/api/health")
        response.raise_for_status()
        health = response.json()
        print("Nodes:", health, flush=True)
        if not all(health[key] for key in ("application", "rag", "inference")):
            raise SystemExit("Not all nodes are ready")
        if args.import_papers:
            for name in ["rag-2005.11401.pdf", "qlora-2305.14314.pdf"]:
                path = root / "data" / "papers" / name
                print(f"Importing {name}…", flush=True)
                with path.open("rb") as file:
                    response = client.post("/api/documents", files={"file": (name, file, "application/pdf")})
                if response.is_error:
                    raise SystemExit(f"Import failed: {response.status_code} {response.text[:300]}")
                document_id = response.json()["document_id"]
                import_deadline = time.monotonic() + 1800
                while time.monotonic() < import_deadline:
                    listing = client.get("/api/documents").json()["documents"]
                    document = next(item for item in listing if item["id"] == document_id)
                    if document["status"] == "completed":
                        print("Imported:", document["filename"], document["chunks"], "chunks", flush=True)
                        break
                    if document["status"] == "failed":
                        raise SystemExit(document["error"])
                    time.sleep(2)
                else:
                    raise SystemExit("Timed out waiting for document import")
        response = client.post("/api/sessions")
        response.raise_for_status()
        session = response.json()["session_id"]
        request = str(uuid4())
        response = client.post(f"/api/sessions/{session}/questions", json={
            "request_id": request, "question": args.question,
        })
        response.raise_for_status()
        print("Submitted:", request, flush=True)
        deadline = time.monotonic() + 600
        previous_status = None
        while time.monotonic() < deadline:
            response = client.get(f"/api/requests/{request}", timeout=15)
            response.raise_for_status()
            result = response.json()
            if result["status"] != previous_status:
                print("Status:", result["status"], flush=True)
                previous_status = result["status"]
            if result["status"] in {"completed", "failed"}:
                break
            time.sleep(2)
        else:
            raise SystemExit("Timed out waiting for an answer")
        report = root / "data" / "last-smoke-test.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] != "completed" or not result["answer"] or not result["sources"]:
            raise SystemExit(f"Smoke test failed; see {report}")
        print("Answer:", result["answer"], flush=True)
        print("Sources:", [(source["document"], source["page"]) for source in result["sources"]], flush=True)
        print("PASS: distributed question completed with document sources", flush=True)

if __name__ == "__main__":
    main()
