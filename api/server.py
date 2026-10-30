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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from implementation.answer import answer_question, stream_answer
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


class HistoryMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(examples=["How many privilege leave days does an O1 employee in Nashik get?"])
    history: list[HistoryMessage] = Field(default_factory=list)


class ContextItem(BaseModel):
    source: str
    page: int | None = None
    section: str = ""
    type: str = ""
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
    model: str = "local-rag"
    messages: list[ChatMessage]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_name, _slots
    _slots = asyncio.Semaphore(MAX_INFLIGHT)
    _model_name = await asyncio.to_thread(prepare_local_model)
    yield


app = FastAPI(title="Local Rag", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _context_items(chunks) -> list[ContextItem]:
    items = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        page = metadata.get("page")
        items.append(
            ContextItem(
                source=metadata.get("source", ""),
                page=page if isinstance(page, int) else None,
                section=metadata.get("section") or "",
                type=metadata.get("type") or "",
                text=chunk.page_content,
            )
        )
    return items


def _history(body: AskRequest) -> list[dict]:
    cleaned = []
    for item in body.history:
        if item.role not in {"user", "assistant"} or not isinstance(item.content, str):
            raise HTTPException(
                status_code=400,
                detail="Each history item needs role 'user' or 'assistant' and string content.",
            )
        cleaned.append({"role": item.role, "content": item.content})
    return cleaned


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


async def _reserve_slot() -> float:
    """Wait for a GPU slot. Returns how long the request spent waiting, in milliseconds."""
    global _waiting
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
    return (time.perf_counter() - queued_at) * 1000


def _release_slot() -> None:
    if _slots is not None:
        _slots.release()


async def _ask(question: str, history: list[dict], bypass_cache: bool = False) -> AskResponse:
    global _waiting
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    key = _cache_key(question, history)
    if not bypass_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    queue_ms = await _reserve_slot()
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
        _release_slot()


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


def _checked_ask(body: AskRequest) -> tuple[str, list[dict]]:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return body.question.strip(), _history(body)


@app.post("/ask", response_model=AskResponse)
@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest, x_bypass_cache: str | None = Header(default=None)):
    """Return the full answer and the source chunks in one JSON response."""
    question, history = _checked_ask(body)
    return await _ask(question, history, bypass_cache=x_bypass_cache == "1")


def _stream_response(events):
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _cached_events(cached: AskResponse):
    async def events():
        yield _sse("context", {"context": [item.model_dump() for item in cached.context]})
        if cached.answer:
            yield _sse("token", {"text": cached.answer})
        yield _sse("done", {"elapsed_ms": 0, "queue_ms": 0, "cached": True})

    return events()


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest, x_bypass_cache: str | None = Header(default=None)):
    """Stream the answer as server-sent events after the source chunks are known.

    Queue-full and validation errors are normal HTTP responses. A failure during
    generation is an `error` event, because the response has already started.
    """
    question, history = _checked_ask(body)
    bypass_cache = x_bypass_cache == "1"
    key = _cache_key(question, history)
    if not bypass_cache:
        cached = _cache_get(key)
        if cached is not None:
            return _stream_response(_cached_events(cached))

    queue_ms = await _reserve_slot()

    async def events():
        if not bypass_cache:
            cached = _cache_get(key)
            if cached is not None:
                cached.queue_ms = queue_ms
                yield _sse("context", {"context": [item.model_dump() for item in cached.context]})
                if cached.answer:
                    yield _sse("token", {"text": cached.answer})
                yield _sse("done", {"elapsed_ms": 0, "queue_ms": queue_ms, "cached": True})
                return

        loop = asyncio.get_running_loop()
        incoming: asyncio.Queue = asyncio.Queue()
        finished = asyncio.Event()

        def run() -> None:
            global _inflight
            answer_parts: list[str] = []
            chunks = []
            with _inflight_lock:
                _inflight += 1
            try:
                for kind, payload in stream_answer(question, history):
                    if kind == "context":
                        chunks = payload
                        items = [item.model_dump() for item in _context_items(chunks)]
                        loop.call_soon_threadsafe(incoming.put_nowait, ("context", {"context": items}))
                    else:
                        answer_parts.append(payload)
                        loop.call_soon_threadsafe(incoming.put_nowait, ("token", {"text": payload}))
                if not bypass_cache:
                    _cache_put(
                        key,
                        AskResponse(
                            answer="".join(answer_parts),
                            context=_context_items(chunks),
                            elapsed_ms=0,
                        ),
                    )
            except Exception as exc:
                loop.call_soon_threadsafe(incoming.put_nowait, ("error", {"detail": str(exc)}))
            finally:
                with _inflight_lock:
                    _inflight -= 1
                loop.call_soon_threadsafe(incoming.put_nowait, None)
                loop.call_soon_threadsafe(finished.set)

        threading.Thread(target=run, daemon=True).start()
        started = time.perf_counter()
        try:
            while True:
                item = await incoming.get()
                if item is None:
                    yield _sse(
                        "done",
                        {
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "queue_ms": queue_ms,
                            "cached": False,
                        },
                    )
                    break
                kind, payload = item
                yield _sse(kind, payload)
                if kind == "error":
                    break
        finally:
            await finished.wait()

    async def events_with_slot():
        try:
            async for piece in events():
                yield piece
        finally:
            _release_slot()

    return _stream_response(events_with_slot())


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

    parser = argparse.ArgumentParser(description="Run the Local Rag API.")
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
