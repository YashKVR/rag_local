"""RAG API.

FastAPI accepts chat requests and runs the local LiteLLM client in a worker
thread. A small number of requests run at once, matching what one GPU can
finish. Further requests wait in a short queue, and anything beyond that is
rejected immediately instead of sitting until the client times out.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from implementation.answer import answer_question
from implementation.llm import prepare_local_model

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# Two in flight matched the best measured throughput on this GPU.
MAX_INFLIGHT = int(os.environ.get("RAG_MAX_INFLIGHT", "2"))
MAX_WAITING = int(os.environ.get("RAG_MAX_WAITING", "30"))
CACHE_TTL_SECONDS = int(os.environ.get("RAG_CACHE_TTL", "600"))
CACHE_MAX = 256

_inflight = 0
_waiting = 0
_cache_hits = 0
_inflight_lock = threading.Lock()
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, "AskResponse"]] = {}
_slots: asyncio.Semaphore | None = None
_model_name = ""


class AskRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)


class ContextItem(BaseModel):
    source: str
    text: str


class AskResponse(BaseModel):
    answer: str
    context: list[ContextItem]
    elapsed_ms: float
    queue_ms: float = 0
    cached: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "insurellm"
    messages: list[ChatMessage]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_name, _slots
    _slots = asyncio.Semaphore(MAX_INFLIGHT)
    _model_name = await asyncio.to_thread(prepare_local_model)
    yield


app = FastAPI(title="Insurellm RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _context_items(chunks) -> list[ContextItem]:
    return [
        ContextItem(source=chunk.metadata.get("source", ""), text=chunk.page_content)
        for chunk in chunks
    ]


def _cache_key(question: str, history: list[dict]) -> str:
    return json.dumps({"question": question.strip(), "history": history}, sort_keys=True, default=str)


def _cache_get(key: str) -> AskResponse | None:
    global _cache_hits
    now = time.time()
    with _cache_lock:
        found = _cache.get(key)
        if found is None:
            return None
        stored_at, response = found
        if now - stored_at > CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        _cache_hits += 1
    return response.model_copy(update={"cached": True, "elapsed_ms": 0, "queue_ms": 0})


def _cache_put(key: str, response: AskResponse) -> None:
    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            oldest = min(_cache, key=lambda item: _cache[item][0])
            _cache.pop(oldest, None)
        _cache[key] = (time.time(), response)


def _run_rag(question: str, history: list[dict]) -> AskResponse:
    global _inflight
    with _inflight_lock:
        _inflight += 1
    started = time.perf_counter()
    try:
        answer, chunks = answer_question(question, history)
    finally:
        with _inflight_lock:
            _inflight -= 1
    return AskResponse(
        answer=answer or "",
        context=_context_items(chunks),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


async def _ask(question: str, history: list[dict], bypass_cache: bool = False) -> AskResponse:
    global _waiting
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    key = _cache_key(question, history)
    if not bypass_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    if _slots is None:
        raise HTTPException(status_code=503, detail="Server is still starting.")

    with _inflight_lock:
        waiting = _waiting
        if waiting >= MAX_WAITING:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Queue is full ({MAX_INFLIGHT} running, {waiting} waiting). "
                    "Retry in a few seconds."
                ),
                headers={"Retry-After": "5"},
            )
        _waiting += 1

    queued_at = time.perf_counter()
    try:
        await _slots.acquire()
    finally:
        with _inflight_lock:
            _waiting -= 1
    queue_ms = (time.perf_counter() - queued_at) * 1000
    try:
        if not bypass_cache:
            cached = _cache_get(key)
            if cached is not None:
                cached.queue_ms = queue_ms
                return cached
        try:
            result = await asyncio.to_thread(_run_rag, question, history)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        result.queue_ms = queue_ms
        if not bypass_cache:
            _cache_put(key, result)
        return result
    finally:
        _slots.release()


@app.get("/health")
def health():
    with _inflight_lock:
        inflight = _inflight
        waiting = _waiting
    with _cache_lock:
        cache_size = len(_cache)
        cache_hits = _cache_hits
    return {
        "status": "ok",
        "model": _model_name,
        "inflight": inflight,
        "waiting": waiting,
        "max_inflight": MAX_INFLIGHT,
        "max_waiting": MAX_WAITING,
        "cache_size": cache_size,
        "cache_hits": cache_hits,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, x_bypass_cache: str | None = Header(default=None)):
    return await _ask(body.question, body.history, bypass_cache=x_bypass_cache == "1")


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    """OpenAI-compatible endpoint so LiteLLM can call this server."""
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages is empty.")
    history = [
        {"role": message.role, "content": message.content}
        for message in body.messages[:-1]
        if message.role in {"user", "assistant"}
    ]
    result = await _ask(body.messages[-1].content, history)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "elapsed_ms": result.elapsed_ms,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Insurellm RAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Uvicorn worker processes. The GPU model still runs one request at a time inside Ollama.",
    )
    args = parser.parse_args()
    uvicorn.run("api.server:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
