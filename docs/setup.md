# Príprava Macu a Azure

Táto príručka používa **RAG + QLoRA**. Prvý nácvik môže bežať kompletne na Macu;
cieľom je presun RAG služby a jej vlastnej databázy do Azure.

## 1. Čo už bolo zistené na tomto Macu

Pri vytvorení projektu boli dostupné Git, Python 3.13 aj 3.14, Node.js 22,
npm, Docker CLI a Ollama CLI 0.18.0. Docker socket nebol dostupný.
Stav bežiacej Ollamy a stiahnutých modelov nebol overený; lokálne spojenie
v obmedzenom prostredí bolo odmietnuté. Netreba všetko inštalovať znova.

## 2. Mac: pripraviť aplikácie a modely

1. Spusti **Docker Desktop** a počkaj, kým ukáže, že engine beží.
   Ak máš len CLI, nainštaluj Docker Desktop pre Apple Silicon z
   <https://www.docker.com/products/docker-desktop/>.
2. Na tomto Macu je dostupná Ollama cez CLI; v `/Applications` nebola nájdená
   jej aplikácia. Použi `ollama serve` v samostatnom termináli a nechaj ho
   bežať. Ak už server beží, druhý nespúšťaj.
3. V termináli stiahni modely (spolu približne 2 GB na disku):

```bash
ollama pull qwen3:1.7b
ollama pull embeddinggemma
ollama list
```

Qwen3 1.7B má v knižnici Ollama približne 1,4 GB a EmbeddingGemma približne
622 MB. Pamäť pri inferencii je vyššia než veľkosť súborov. Najprv zatvor
náročné aplikácie. Používame 4096-tokenový kontext a jednu modelovú operáciu
naraz. Väčší model zatiaľ nesťahuj. Referencie:
<https://ollama.com/library/qwen3:1.7b>, <https://ollama.com/library/embeddinggemma>.

## 3. Mac: projektová konfigurácia

Príkazy spúšťaj z koreňa projektu (priečinok s README.md).
Ak `.venv` a nainštalované závislosti už existujú, prvé tri príkazy preskoč.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm --prefix frontend install
```

Konfigurácia sa vytvorí s náhodnými tajomstvami. Skript existujúci súbor
neprepíše; pri opakovanom spustení tento krok preskoč.

```bash
.venv/bin/python scripts/configure.py
.venv/bin/python scripts/download_papers.py
```

Súbor `.env.mac` patrí iba tebe. Neposielaj ho do chatu ani na GitHub.
Token budeš neskôr potrebovať skopírovať do súkromnej konfigurácie na Azure.

## 4. Prvý nácvik iba na Macu

V jednom termináli spusti PostgreSQL a lokálny Qdrant:

```bash
docker compose --env-file .env.mac -f deploy/compose.mac.yml --profile local-rag up -d
```

Potom otvor **štyri samostatné terminály** v koreňovom priečinku projektu.
Každý z týchto príkazov nechaj bežať vo vlastnom termináli:

```bash
# Terminál 1
.venv/bin/python scripts/run.py inference
```

```bash
# Terminál 2
.venv/bin/python scripts/run.py rag
```

```bash
# Terminál 3
.venv/bin/python scripts/run.py app
```

```bash
# Terminál 4
npm --prefix frontend run dev
```

Otvor <http://localhost:5173>. V bočnom paneli majú byť tri zelené body.
Nahraj PDF z `data/papers/`, počkaj na dokončenie indexácie, vytvor konverzáciu
a polož napríklad `How does QLoRA reduce memory usage?`.
Import jedného článku môže na 8 GB Macu trvať niekoľko minút. Počas prvého
importu neposielaj otázky, aby sa nezdržovali za embeddingmi.

RAG API je na porte 8001, inference na 8002 a aplikačné API na 8000.
Swagger RAG a inference vyžaduje hlavičku `Authorization: Bearer ...`.
Token nikdy nevkladaj do frontendového kódu.

## 5. GitHub

Vytvor **prázdny repozitár**, napríklad `distributed-rag-chatbot`. Nepridávaj
cez web ďalší README, .gitignore ani licenciu, pretože projekt už má obsah.
Repozitár môže byť súkromný; spolužiaka môžeš pridať ako spolupracovníka.
Pred zverejnením si prehliadni zmeny cez `git diff` / `git status`.
Skutočný remote sa nastaví až podľa URL tvojho repozitára.

Súbory `.env.mac`, `.env.azure`, stiahnuté PDF, databázy, modely a virtuálne
prostredie sú ignorované. Ukážkové `.env.*.example` sa verzujú.

## 6. Azure: vytvor iba jednu CPU VM

V <https://portal.azure.com> vyber Create a resource → Virtual machine.

| Nastavenie | Hodnota |
| --- | --- |
| Subscription | tvoje Azure for Students |
| Resource group | nová, napr. `rg-rag-project` |
| VM name | `rag-node` |
| Image | Ubuntu Server 24.04 LTS, x64 |
| Size | cieľ 2 vCPU a 4 GB RAM; napr. B2s, ak ho účet a región ponúka |
| Authentication | SSH public key |
| Username | napr. `azureuser` |
| OS disk | 30–32 GB, Standard SSD |
| Inbound ports | iba SSH 22, ideálne obmedzený na tvoju verejnú IP |

GPU, Kubernetes, Azure OpenAI ani platenú spravovanú vektorovú databázu
nepotrebujeme. Ak B2s nie je dostupná, vyber dostupnú CPU veľkosť s podobnou
RAM až po kontrole zobrazenej ceny. Kredit automaticky neznamená bezplatnú VM.
Nastav rozpočtové upozornenie a auto-shutdown na čas, keď nepracuješ.
Upozornenie na rozpočet samo VM nevypne. Po práci použi v portáli Stop
a skontroluj stav **Stopped (deallocated)**; disk môže ďalej čerpať kredit.
Stiahnutý SSH kľúč `.pem` drž mimo repozitára.

Oficiálny postup:
<https://learn.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal>.

## 7. Súkromné spojenie cez Tailscale

1. Nainštaluj Tailscale na Mac podľa <https://tailscale.com/download/mac>.
2. Nainštaluj ho na Ubuntu podľa <https://tailscale.com/download/linux>.
3. Prihlás oba stroje do rovnakej súkromnej siete (tailnet).
4. Zisti Tailscale IPv4 oboch zariadení; zvyčajne začínajú `100.`.

Na Azure potom cez SSH nainštaluj Docker a Git:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
```

Repozitár na VM naklonujeme až po nahratí kódu na tvoj GitHub. Pri súkromnom
repozitári použijeme samostatný SSH deploy key s právom čítania.

V naklonovanom repozitári na VM:

```bash
cp .env.azure.example .env.azure
chmod 600 .env.azure
nano .env.azure
```

Vyplň:

- `NODE_API_TOKEN`: ten istý tajný token ako na Macu.
- `INFERENCE_URL=http://<TAILSCALE_IP_MACU>:8002`.
- `AZURE_BIND_IP=<TAILSCALE_IP_AZURE>`.

Na Macu v `.env.mac` nastav:

- `RAG_URL=http://<TAILSCALE_IP_AZURE>:8001`.
- `MAC_INFERENCE_BIND=<TAILSCALE_IP_MACU>`.
- `INFERENCE_URL=http://<TAILSCALE_IP_MACU>:8002`.

Po zmene reštartuj aplikačnú a inference službu. Lokálnu RAG službu zastav
Ctrl+C. Lokálny Qdrant môžeš zastaviť:

```bash
docker compose --env-file .env.mac -f deploy/compose.mac.yml --profile local-rag stop qdrant
```

Na Azure spusti:

```bash
sudo docker compose --env-file .env.azure -f deploy/compose.azure.yml up -d --build
sudo docker compose --env-file .env.azure -f deploy/compose.azure.yml logs --tail 50
```

Porty 6333, 8001, 8002 ani 11434 neotváraj do verejného internetu.
RAG kontajner publikuje port iba na Tailscale IP; Qdrant nie je publikovaný.
Firewall na Macu musí dovoliť súkromné spojenie na inference API.
Spustený Tailscale je potrebný pred štartom služby bindovanej na jeho IP.

Po presune do Azure **znova nahraj oba PDF** cez web, pretože Azure má
vlastný prázdny index. Lokálny index sa automaticky neprenáša.

## 8. Overenie funkčnosti

1. Všetky tri indikátory dostupnosti svietia zeleno.
2. Import RAG a QLoRA skončí počtom chunkov.
3. Odpoveď obsahuje nájdené zdroje, PDF strany a rozkliknuteľný text chunkov.
4. Dve konverzácie si uchovajú vlastné otázky a odpovede aj po obnovení webu.
5. Zastav inference API, odošli otázku: zostane vo fronte. Znovu spusti API:
   otázka sa automaticky spracuje.
6. Reštart aplikačného API zachová históriu a čakajúce úlohy v PostgreSQL.
   Rozpracovaná úloha sa po páde obnoví po uplynutí lease (najviac 10 minút).

Streamovanie tokenov, cache a záťažový test patria do nasledujúcej etapy.
