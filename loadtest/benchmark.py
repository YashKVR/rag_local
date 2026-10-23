"""One comparable speed test for the RAG API.

Every request bypasses the answer cache and asks a different question, so the
number measures the machine rather than a cached reply. Run the same command
on each machine and compare answers_per_min. Higher is faster.
"""

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path

import httpx

from loadtest.run import percentile

# Same questions, same order, on every machine.
WARMUP = "When was Insurellm founded?"
SEQUENTIAL = [
    "How many employees does Insurellm have?",
    "Where is the Insurellm headquarters?",
    "What is Carllm?",
    "What is Homellm?",
    "What is Claimllm?",
]
CONCURRENT = [
    "What is Rellm?",
    "What is Markellm?",
    "What is Lifellm?",
    "What is Healthllm?",
    "How many products does Insurellm offer?",
    "Which cities does Insurellm have offices in?",
]
CONCURRENCY = 2


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else "unknown"


def ask(client: httpx.Client, question: str) -> float:
    started = time.perf_counter()
    response = client.post(
        "/ask",
        json={"question": question, "history": []},
        headers={"X-Bypass-Cache": "1"},
    )
    response.raise_for_status()
    if not response.json().get("answer"):
        raise RuntimeError("empty answer")
    return time.perf_counter() - started


async def ask_many(base_url: str, questions: list[str], concurrency: int, timeout: float) -> tuple[list[float], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(http: httpx.AsyncClient, question: str) -> float:
        async with semaphore:
            started = time.perf_counter()
            response = await http.post(
                "/ask",
                json={"question": question, "history": []},
                headers={"X-Bypass-Cache": "1"},
                timeout=timeout,
            )
            response.raise_for_status()
            if not response.json().get("answer"):
                raise RuntimeError("empty answer")
            return time.perf_counter() - started

    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url) as http:
        latencies = list(await asyncio.gather(*(one(http, question) for question in questions)))
    return latencies, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description="Compare RAG speed across machines.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--output",
        default="",
        help="JSON file to write. Default: benchmark-<hostname>.json",
    )
    args = parser.parse_args()

    hostname = socket.gethostname()
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            health = client.get("/health")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"Cannot reach {args.base_url}/health ({exc}). Start the API first: "
                "uv run python -m api.server --port 8000"
            ) from exc
        model = health.json().get("model", "unknown")

        print("Warming up (not scored)...", flush=True)
        warmup_s = ask(client, WARMUP)
        print(f"Warmup: {warmup_s:.1f}s", flush=True)

        print(f"Sequential: {len(SEQUENTIAL)} questions, one at a time...", flush=True)
        sequential = [ask(client, question) for question in SEQUENTIAL]

    print(f"Concurrent: {len(CONCURRENT)} questions, {CONCURRENCY} at a time...", flush=True)
    concurrent, wall = asyncio.run(ask_many(args.base_url, CONCURRENT, CONCURRENCY, args.timeout))

    answers_per_min = len(concurrent) / wall * 60 if wall else 0
    report = {
        "hostname": hostname,
        "cpu_cores": os.cpu_count(),
        "gpu": gpu_name(),
        "model": model,
        "warmup_s": round(warmup_s, 2),
        "latency_median_s": round(percentile(sequential, 50), 2),
        "latency_average_s": round(statistics.fmean(sequential), 2),
        "latency_p95_s": round(percentile(sequential, 95), 2),
        "answers_per_min": round(answers_per_min, 1),
        "concurrent_median_s": round(percentile(concurrent, 50), 2),
        "concurrent_p95_s": round(percentile(concurrent, 95), 2),
        "concurrent_wall_s": round(wall, 2),
        "sequential_samples_s": [round(value, 2) for value in sequential],
        "concurrent_samples_s": [round(value, 2) for value in concurrent],
    }

    print()
    print(f"Machine: {hostname}")
    print(f"CPU cores: {report['cpu_cores']}    GPU: {report['gpu']}    Model: {model}")
    print()
    print("One question at a time (lower is better)")
    print(f"  median   {report['latency_median_s']:.2f}s")
    print(f"  average  {report['latency_average_s']:.2f}s")
    print(f"  p95      {report['latency_p95_s']:.2f}s")
    print()
    print(f"Two questions at a time, {len(CONCURRENT)} questions finished in {wall:.1f}s")
    print(f"  answers per minute   {report['answers_per_min']:.1f}")
    print(f"  median wait          {report['concurrent_median_s']:.2f}s")
    print(f"  p95 wait             {report['concurrent_p95_s']:.2f}s")
    print()
    print(
        "COMPARE "
        f"latency_median_s={report['latency_median_s']:.2f} "
        f"answers_per_min={report['answers_per_min']:.1f}"
    )
    print()
    print("Pick the machine with the highest answers_per_min.")
    print("If two machines are close, pick the lower latency_median_s.")
    print("Warmup is model load time and is not part of the score.")

    output = Path(args.output or f"benchmark-{hostname}.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
