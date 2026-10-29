# Local Rag

A local retrieval-augmented assistant over the company PDFs in `knowledge-files/`.

Questions and documents are embedded locally with `nomic-embed-text`, matched in a local Chroma index, and answered by a gpt-oss model served by Ollama through LiteLLM. FastAPI exposes that pipeline to a small web UI and to any OpenAI-compatible client. A single benchmark scores a machine so you can compare PCs.

## What you need

- Linux with an NVIDIA GPU, or another machine that can run Ollama
- [uv](https://docs.astral.sh/uv/) and Python 3.11 (this repo pins 3.11 in `.python-version`)
- [Ollama](https://ollama.com/)
- About 16 GB of memory for `gpt-oss:20b` (about 14 GB on disk). `gpt-oss:120b` is used only when a GPU has about 80 GB of memory and the 65 GB model is installed
- About 300 MB more disk for `nomic-embed-text`, which ingest downloads if it is missing

The app picks the model on startup. On a 24 GB GPU it stays on gpt-oss 20B.

## Install

From the repo root:

```bash
uv sync
```

`uv sync` creates `.venv` and installs the locked dependencies.

Install Ollama if `ollama` is not already on the machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start it if it is not already running:

```bash
ollama serve
```

## Environment file

Ingest and answers run locally. No API key is required.

These settings are optional. Defaults are shown. Put them in `.env` in the repo root if you want to change them. That file is gitignored.

```bash
RAG_MAX_INFLIGHT=2
RAG_MAX_WAITING=30
RAG_CACHE_TTL=600
```

`RAG_MAX_INFLIGHT` is how many generations run on the GPU at once. `RAG_MAX_WAITING` is how many more requests may wait before the API returns HTTP 429. `RAG_CACHE_TTL` is how long an identical answer is reused, in seconds.

## Set up a machine after clone

Dependencies are assumed installed (`uv sync` has been run). Do this once per machine:

1. Pull the model the hardware can run:

```bash
ollama pull gpt-oss:20b
```

Pull `gpt-oss:120b` only on a machine with about 80 GB of GPU memory and about 65 GB of free disk.

2. Put the company PDFs in `knowledge-files/`. That folder is gitignored because the files include employee records.

3. Build the search index. This reads each PDF, splits policies on their numbered headings, turns holiday and promotion tables into one row per chunk, and embeds the chunks locally:

```bash
uv run python -m implementation.ingest
```

The index is written to `preprocessed_db/`. Repeat this command when a PDF is added or replaced.

4. Start the API:

```bash
uv run python -m api.server --port 8000
```

5. Open http://127.0.0.1:8000/ and ask a question about the documents, for example "How many privilege leave days does an O1 employee in Nashik get?" The page shows the answer and the retrieved sources.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

A ready server looks like `{"status":"ok","model":"gpt-oss:20b",...}`.

The Gradio UI is separate and calls the pipeline directly:

```bash
uv run app.py
```

It listens on http://127.0.0.1:7860/.

## Compare machines

Run the API on that machine, then in a second terminal:

```bash
uv run python -m loadtest.benchmark
```

The benchmark asks the same fixed questions on every PC and sends `X-Bypass-Cache: 1`, so a second run cannot look faster by reusing a stored answer. It prints a score line:

```text
COMPARE latency_median_s=3.33 answers_per_min=23.5
```

| Field | Meaning | Better |
|---|---|---|
| `answers_per_min` | Questions finished per minute with 2 running at once | Higher |
| `latency_median_s` | Typical seconds for one question, alone | Lower |

Pick the machine with the highest `answers_per_min`. If two results are within about 10% of each other, pick the lower `latency_median_s`.

`average` is the mean of the five single-question times. `p95` means 95% of those single questions were faster than that many seconds. Warmup is the first request and is not part of the score.

Each run writes `benchmark-<hostname>.json`. Copy those files into one folder and sort by `answers_per_min`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Chat page |
| `GET` | `/health` | Model name, queue depth, cache size |
| `POST` | `/ask` | `{ "question": "...", "history": [] }` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat for LiteLLM |

`POST /ask` returns `answer`, `context`, `elapsed_ms`, `queue_ms`, and `cached`.

Point LiteLLM at this server with `api_base` `http://127.0.0.1:8000/v1`, model `openai/local-rag`, and any non-empty `api_key` such as `local`.
