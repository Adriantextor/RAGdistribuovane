# AI Research Assistant

Semestrálny projekt: distribuovaný RAG chatbot s lokálnym LLM.
Prvé dokumenty sú články **RAG** a **QLoRA**. Modely sa netrénujú;
články slúžia ako vyhľadávateľný kontext pri odpovedaní.

## Rozdelenie

| Uzol | Miesto | Obsah |
| --- | --- | --- |
| 1 – aplikácia | Mac | React, FastAPI, PostgreSQL, história, trvalý front |
| 2 – RAG | Azure CPU VM | FastAPI, spracovanie PDF, Qdrant |
| 3 – inference | Mac, neskôr GPU notebook | FastAPI, Ollama, Qwen3 1.7B, EmbeddingGemma |

Začni návodom **[Príprava Macu a Azure](docs/setup.md)**.
Podrobnosti a limity sú v [architektúre](docs/architecture.md).

## Dokumenty

- Patrick Lewis et al.: *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks* (2020), [arXiv](https://arxiv.org/abs/2005.11401),
  [PDF](https://arxiv.org/pdf/2005.11401).
- Tim Dettmers et al.: *QLoRA: Efficient Finetuning of Quantized LLMs* (2023),
  [arXiv](https://arxiv.org/abs/2305.14314), [PDF](https://arxiv.org/pdf/2305.14314).

`python scripts/download_papers.py` ich stiahne do ignorovaného `data/papers/`.
PDF ani modelové váhy sa neukladajú do Gitu. Autorské práva ostávajú autorom.

## Štruktúra

```text
services/app/          aplikačné API a trvalý worker
services/rag/          import, chunking, vektorové vyhľadávanie
services/inference/    autentifikované API nad lokálnou Ollamou
shared/               konfigurácia, API schémy a autentifikácia uzlov
frontend/             React + TypeScript
deploy/               Docker Compose pre Mac a Azure
scripts/              konfigurácia, spustenie a stiahnutie dokumentov
tests/                idempotencia, izolácia relácií, obnova frontu, chunking
```

## Overenie

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend run build
```

Jednotkové testy nepotrebujú model ani Azure. Overujú logiku úložiska na
dočasnej SQLite databáze; finálny integračný test musí overiť aj PostgreSQL,
Ollamu, Qdrant a sieť medzi Macom a Azure.
