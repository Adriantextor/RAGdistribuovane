"""One process per service. Run from the repository root."""
import argparse
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("node", choices=["app", "rag", "inference"])
parser.add_argument("--env", default=".env.mac")
args = parser.parse_args()
if not Path(args.env).is_file():
    parser.error(f"Missing {args.env}; first run python scripts/configure.py")
load_dotenv(args.env)
token = os.environ.get("NODE_API_TOKEN", "")
if len(token) < 32 or token.startswith("replace-"):
    parser.error("Configure a random NODE_API_TOKEN with at least 32 characters")
host = os.environ.get("MAC_INFERENCE_BIND", "127.0.0.1") if args.node == "inference" else "127.0.0.1"
port = {"app": 8000, "rag": 8001, "inference": 8002}[args.node]
uvicorn.run(f"services.{args.node}.main:app", host=host, port=port, workers=1)
