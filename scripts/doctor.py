"""Readiness checks for deployment; never prints configuration secrets."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from shared.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env.mac")
    parser.add_argument("--role", choices=["app", "inference"], default="app")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    load_dotenv(root / args.env)
    config = settings()
    checks = []
    if args.role == "app":
        response = httpx.get("http://127.0.0.1:8000/api/health", timeout=20)
        response.raise_for_status()
        data = response.json()
        checks.extend((name, bool(data.get(name))) for name in ["application", "rag", "inference"])
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        bind = os.getenv("INFERENCE_BIND", os.getenv("MAC_INFERENCE_BIND", "127.0.0.1"))
        url = f"http://{bind}:8002"
        response = httpx.get(url + "/health", headers=config.headers(), timeout=20)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
        checks.append(("inference API and models", True))
        if args.probe:
            response = httpx.post(url + "/generate", headers=config.headers(), timeout=300, json={
                "question": "What is the test code? Answer in one short sentence.", "language": "en",
                "sources": [{"id": "probe", "document": "readiness-test", "page": 1,
                             "text": "The test code is READY-OK."}],
            })
            response.raise_for_status()
            checks.append(("model generates a grounded answer", "READY-OK" in response.json().get("answer", "")))
        if args.require_gpu:
            nvidia = shutil.which("nvidia-smi")
            if not nvidia:
                checks.append(("NVIDIA GPU detected", False))
            else:
                result = subprocess.run([nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=15, check=True)
                print("GPU:", result.stdout.strip())
                memory = [float(line.rsplit(",", 1)[1].strip()) for line in result.stdout.strip().splitlines()]
                checks.append(("at least 8 GiB VRAM", any(value >= 8192 for value in memory)))
            if args.probe:
                response = httpx.get(config.ollama_url + "/api/ps", timeout=10)
                response.raise_for_status()
                models = [model for model in response.json().get("models", []) if model.get("name") == config.llm_model]
                checks.append(("LLM uses GPU memory (keep MODEL_KEEP_ALIVE=5m)",
                               bool(models) and all(model.get("size_vram", 0) > 0 for model in models)))
    for name, passed in checks:
        print(("PASS" if passed else "FAIL") + ": " + name)
    raise SystemExit(0 if all(passed for _, passed in checks) else 1)


if __name__ == "__main__":
    main()
