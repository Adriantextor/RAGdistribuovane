"""Download the two starter papers from arXiv into ignored local data/."""
from pathlib import Path
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1] / "data" / "papers"
root.mkdir(parents=True, exist_ok=True)
papers = {
    "rag-2005.11401.pdf": "https://arxiv.org/pdf/2005.11401",
    "qlora-2305.14314.pdf": "https://arxiv.org/pdf/2305.14314",
}
for name, url in papers.items():
    target = root / name
    if target.exists():
        print(f"Already exists: {target.name}")
        continue
    request = Request(url, headers={"User-Agent": "AIResearchAssistant-StudentProject/0.1"})
    with urlopen(request, timeout=60) as response:
        content = response.read(20 * 1024 * 1024 + 1)
    if not content.startswith(b"%PDF-") or len(content) > 20 * 1024 * 1024:
        raise RuntimeError(f"Unexpected response downloading {url}")
    target.write_bytes(content)
    print(f"Downloaded {target.name}")
