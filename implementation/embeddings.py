"""Local embeddings through Ollama.

nomic-embed-text expects a prefix: search_document for stored chunks and
search_query for a user question. The same model has to be used for both,
so retrieval and ingestion share this module. Text is not sent to a hosted API.
"""

import json
import subprocess
import urllib.request

from implementation.llm import OLLAMA_API_BASE, ensure_ollama, ensure_server, logger


EMBED_MODEL = "nomic-embed-text"
_ready = False


def embed_texts(texts: list[str], purpose: str) -> list[list[float]]:
    if purpose not in {"document", "query"}:
        raise ValueError(f"purpose must be 'document' or 'query', got {purpose!r}")
    ensure_embed_model()

    vectors: list[list[float]] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = json.dumps(
            {
                "model": EMBED_MODEL,
                "input": [f"search_{purpose}: {text}" for text in batch],
            }
        ).encode()
        request = urllib.request.Request(
            f"{OLLAMA_API_BASE}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode())
        vectors.extend(body["embeddings"])
        logger.info("Embedded %s/%s chunks with %s.", min(start + batch_size, len(texts)), len(texts), EMBED_MODEL)
    return vectors


def ensure_embed_model() -> None:
    global _ready
    if _ready:
        return
    ensure_ollama()
    ensure_server()
    listed = subprocess.run(["ollama", "list"], check=False, capture_output=True, text=True)
    if EMBED_MODEL not in (listed.stdout or ""):
        logger.info("Downloading %s for local embeddings.", EMBED_MODEL)
        subprocess.run(["ollama", "pull", EMBED_MODEL], check=True)
    else:
        logger.info("%s is already installed.", EMBED_MODEL)
    _ready = True
