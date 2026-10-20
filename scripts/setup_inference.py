"""Cross-platform setup for the teammate's AI node. No secret is printed."""
import argparse
import getpass
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True, help="This notebook's Tailscale IPv4")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise SystemExit("Install Python 3.12 or 3.13 first.")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    ollama = shutil.which("ollama")
    if not ollama:
        raise SystemExit("Install and start Ollama first: https://ollama.com/download")
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements-inference.txt"], check=True)
    target = root / ".env.inference"
    if not target.exists():
        token = getpass.getpass("Shared NODE_API_TOKEN (hidden): ").strip()
        if len(token) < 32 or token.startswith("replace-"):
            raise SystemExit("Use the same random token as on the app and RAG nodes.")
        content = (root / ".env.inference.example").read_text()
        content = content.replace("replace-with-the-same-token-as-on-mac", token)
        content = content.replace("INFERENCE_BIND=127.0.0.1", f"INFERENCE_BIND={args.bind}")
        content = content.replace("LLM_MODEL=qwen3:4b", f"LLM_MODEL={args.model}")
        target.write_text(content)
        target.chmod(0o600)
    else:
        print("Keeping existing .env.inference. Check its model and bind address.")
    if not args.skip_models:
        for model in [args.model, "embeddinggemma"]:
            subprocess.run([ollama, "pull", model], check=True)
    print("AI node is prepared. Allow Python through the private-network firewall if prompted.")
    print(f'Run: "{python}" scripts/run.py inference --env .env.inference')
    print(f'From app and RAG nodes use: http://{args.bind}:8002')
    print("Verify GPU acceleration with scripts/doctor.py --env .env.inference --role inference --probe --require-gpu")


if __name__ == "__main__":
    main()
