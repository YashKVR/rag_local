"""Load test the RAG API with async clients, threads, and processes.

The API server is the system under test. This process only sends traffic.
LiteLLM mode calls the OpenAI-compatible /v1/chat/completions route.
HTTP mode calls POST /ask directly.
"""

import argparse
import asyncio
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool

import httpx

QUESTIONS = [
    "When was the company founded?",
    "How many employees does the company have?",
    "What is Carllm?",
    "Where is the headquarters?",
]


@dataclass
class Outcome:
    latency: float
    ok: bool
    error: str = ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(mode: str, client: str, concurrency: int, outcomes: list[Outcome], wall: float) -> dict:
    latencies = [item.latency for item in outcomes if item.ok]
    errors = [item for item in outcomes if not item.ok]
    return {
        "mode": mode,
        "client": client,
        "concurrency": concurrency,
        "requests": len(outcomes),
        "ok": len(latencies),
        "errors": len(errors),
        "rps": len(outcomes) / wall if wall else 0,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "max": max(latencies) if latencies else 0,
        "mean": statistics.fmean(latencies) if latencies else 0,
        "wall": wall,
        "sample_error": errors[0].error if errors else "",
    }


def print_row(row: dict) -> None:
    print(
        f"{row['mode']:<12} {row['client']:<8} {row['concurrency']:>4} "
        f"{row['ok']:>4}/{row['requests']:<4} {row['rps']:>6.2f} "
        f"{row['p50']:>6.1f} {row['p95']:>6.1f} {row['max']:>6.1f}  {row['sample_error'][:80]}"
    )


def http_call(base_url: str, question: str, timeout: float) -> Outcome:
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url}/ask",
            json={"question": question, "history": []},
            headers={"X-Bypass-Cache": "1"},
            timeout=timeout,
        )
        response.raise_for_status()
        if not response.json().get("answer"):
            raise RuntimeError("empty answer")
        return Outcome(time.perf_counter() - started, True)
    except Exception as exc:
        return Outcome(time.perf_counter() - started, False, f"{type(exc).__name__}: {exc}")


def litellm_call(base_url: str, question: str, timeout: float) -> Outcome:
    import litellm

    litellm.suppress_debug_info = True
    started = time.perf_counter()
    try:
        response = litellm.completion(
            model="openai/local-rag",
            api_base=f"{base_url}/v1",
            api_key="local",
            messages=[{"role": "user", "content": question}],
            timeout=timeout,
        )
        if not response.choices[0].message.content:
            raise RuntimeError("empty answer")
        return Outcome(time.perf_counter() - started, True)
    except Exception as exc:
        return Outcome(time.perf_counter() - started, False, f"{type(exc).__name__}: {exc}")


def sync_call(base_url: str, question: str, timeout: float, client: str) -> Outcome:
    if client == "litellm":
        return litellm_call(base_url, question, timeout)
    return http_call(base_url, question, timeout)


def process_job(job: tuple) -> list[Outcome]:
    base_url, questions, timeout, client, count, offset = job
    outcomes = []
    for index in range(count):
        question = questions[(offset + index) % len(questions)]
        outcomes.append(sync_call(base_url, question, timeout, client))
    return outcomes


async def async_http(base_url: str, question: str, timeout: float, client: httpx.AsyncClient) -> Outcome:
    started = time.perf_counter()
    try:
        response = await client.post(
            "/ask",
            json={"question": question, "history": []},
            headers={"X-Bypass-Cache": "1"},
            timeout=timeout,
        )
        response.raise_for_status()
        if not response.json().get("answer"):
            raise RuntimeError("empty answer")
        return Outcome(time.perf_counter() - started, True)
    except Exception as exc:
        return Outcome(time.perf_counter() - started, False, f"{type(exc).__name__}: {exc}")


async def async_litellm(base_url: str, question: str, timeout: float) -> Outcome:
    import litellm

    litellm.suppress_debug_info = True
    started = time.perf_counter()
    try:
        response = await litellm.acompletion(
            model="openai/local-rag",
            api_base=f"{base_url}/v1",
            api_key="local",
            messages=[{"role": "user", "content": question}],
            timeout=timeout,
        )
        if not response.choices[0].message.content:
            raise RuntimeError("empty answer")
        return Outcome(time.perf_counter() - started, True)
    except Exception as exc:
        return Outcome(time.perf_counter() - started, False, f"{type(exc).__name__}: {exc}")


async def run_asyncio(base_url: str, total: int, concurrency: int, timeout: float, client_name: str) -> list[Outcome]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bound(index: int, http: httpx.AsyncClient | None) -> Outcome:
        question = QUESTIONS[index % len(QUESTIONS)]
        async with semaphore:
            if client_name == "litellm":
                return await async_litellm(base_url, question, timeout)
            return await async_http(base_url, question, timeout, http)

    if client_name == "http":
        async with httpx.AsyncClient(base_url=base_url) as http:
            return list(await asyncio.gather(*(bound(i, http) for i in range(total))))
    return list(await asyncio.gather(*(bound(i, None) for i in range(total))))


def run_threads(base_url: str, total: int, concurrency: int, timeout: float, client_name: str) -> list[Outcome]:
    def job(index: int) -> Outcome:
        return sync_call(base_url, QUESTIONS[index % len(QUESTIONS)], timeout, client_name)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(job, range(total)))


def run_processes(base_url: str, total: int, concurrency: int, timeout: float, client_name: str) -> list[Outcome]:
    workers = max(1, concurrency)
    counts = [total // workers] * workers
    for index in range(total % workers):
        counts[index] += 1
    jobs = [
        (base_url, QUESTIONS, timeout, client_name, count, offset)
        for offset, count in enumerate(counts)
        if count
    ]
    with Pool(processes=len(jobs)) as pool:
        grouped = pool.map(process_job, jobs)
    return [item for group in grouped for item in group]


def execute(mode: str, base_url: str, total: int, concurrency: int, timeout: float, client_name: str) -> dict:
    started = time.perf_counter()
    if mode == "asyncio":
        outcomes = asyncio.run(run_asyncio(base_url, total, concurrency, timeout, client_name))
    elif mode == "threads":
        outcomes = run_threads(base_url, total, concurrency, timeout, client_name)
    elif mode == "processes":
        outcomes = run_processes(base_url, total, concurrency, timeout, client_name)
    else:
        raise ValueError(mode)
    return summarize(mode, client_name, concurrency, outcomes, time.perf_counter() - started)


def warmup(base_url: str, timeout: float) -> None:
    print("Warming up with one request...", flush=True)
    outcome = http_call(base_url, QUESTIONS[0], timeout)
    if not outcome.ok:
        raise SystemExit(f"Warmup failed: {outcome.error}")
    print(f"Warmup finished in {outcome.latency:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Load test the Local Rag API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["asyncio", "threads", "processes", "ramp", "all"], default="ramp")
    parser.add_argument("--client", choices=["http", "litellm"], default="http")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--slo-p95", type=float, default=20, help="Ramp stops after p95 latency exceeds this many seconds.")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    print(f"Client CPU cores: {os.cpu_count()}", flush=True)
    if not args.no_warmup:
        warmup(args.base_url, args.timeout)

    print(
        f"{'mode':<12} {'client':<8} {'n':>4} {'ok':>7} {'rps':>6} {'p50':>6} {'p95':>6} {'max':>6}  error",
        flush=True,
    )
    modes = ["asyncio", "threads", "processes"] if args.mode == "all" else [args.mode]
    if args.mode == "ramp":
        concurrency = 1
        last_good = None
        while concurrency <= args.max_concurrency:
            # Send at least one full wave so this level is actually reached.
            total = max(args.requests, concurrency)
            row = execute("asyncio", args.base_url, total, concurrency, args.timeout, args.client)
            print_row(row)
            failed = row["errors"] or row["p95"] > args.slo_p95
            if failed:
                if last_good is None:
                    print("No concurrency level stayed inside the error and latency limits.", flush=True)
                else:
                    print(
                        f"Capacity: {last_good} concurrent request(s) before errors or p95 above {args.slo_p95:.0f}s.",
                        flush=True,
                    )
                return
            last_good = concurrency
            concurrency *= 2
        print(
            f"Handled up to {last_good} concurrent request(s) within p95 {args.slo_p95:.0f}s. Raise --max-concurrency to look further.",
            flush=True,
        )
        return

    for mode in modes:
        print_row(execute(mode, args.base_url, args.requests, args.concurrency, args.timeout, args.client))


if __name__ == "__main__":
    main()
