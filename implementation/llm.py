import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from litellm import completion

OLLAMA_API_BASE = "http://127.0.0.1:11434"
GB = 1024**3

# Download sizes are the Ollama MXFP4 builds. Memory figures are Ollama's
# published requirements: 120B fits on one 80 GB GPU, 20B needs about 16 GB.
MODELS = (
    {"tag": "gpt-oss:20b", "label": "gpt-oss 20B", "download_gb": 14, "memory_gb": 16},
    {"tag": "gpt-oss:120b", "label": "gpt-oss 120B", "download_gb": 65, "memory_gb": 80},
)

logger = logging.getLogger("rag_local.llm")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_lock = threading.Lock()
_ready = False
_selected = None
_installed: set[str] = set()
_gpu_gb = 0.0
_ram_gb = 0.0


def prepare_local_model() -> str:
    """Install Ollama if needed, pull both gpt-oss models, and pick one that fits."""
    global _ready, _selected, _installed, _gpu_gb, _ram_gb

    if _ready and _selected:
        return _selected

    with _lock:
        if _ready and _selected:
            return _selected

        logger.info("Preparing local gpt-oss models.")
        ensure_ollama()
        ensure_server()
        _installed = download_models()
        gpus, _ram_gb = read_hardware()
        _gpu_gb = sum(memory for _, memory in gpus)
        _selected = choose_model(_installed, gpus, _ram_gb)
        _ready = True
        return _selected


def local_completion(**kwargs):
    """Run a chat completion on the gpt-oss model selected for this machine."""
    global _selected

    model = prepare_local_model()
    try:
        return completion(model=f"ollama_chat/{model}", api_base=OLLAMA_API_BASE, **kwargs)
    except Exception as exc:
        alternate = fallback_model(model)
        if alternate is None:
            logger.error("%s request failed: %s", label_for(model), exc)
            raise
        logger.error(
            "%s request failed: %s. Switching to %s.",
            label_for(model),
            exc,
            label_for(alternate),
        )
        _selected = alternate
        return completion(model=f"ollama_chat/{alternate}", api_base=OLLAMA_API_BASE, **kwargs)


def ensure_ollama() -> None:
    path = shutil.which("ollama")
    if path:
        version = subprocess.run(
            ["ollama", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        detail = (version.stdout or version.stderr).strip()
        logger.info("Ollama is installed at %s (%s).", path, detail or "version unknown")
        return

    logger.info("Ollama is not installed. Installing it from https://ollama.com/install.sh.")
    command = "curl -fsSL https://ollama.com/install.sh | sh"
    if os.geteuid() != 0:
        if shutil.which("sudo") is None:
            raise RuntimeError(
                "Ollama is not installed, and sudo is unavailable. "
                "Install it with: curl -fsSL https://ollama.com/install.sh | sh"
            )
        command = "curl -fsSL https://ollama.com/install.sh | sudo sh"

    result = subprocess.run(command, shell=True)
    path = shutil.which("ollama")
    if result.returncode != 0 or path is None:
        raise RuntimeError(
            "Ollama installation failed. Install it with: curl -fsSL https://ollama.com/install.sh | sh"
        )
    logger.info("Installed Ollama at %s.", path)


def ensure_server() -> None:
    if server_ready():
        logger.info("Ollama server is running at %s.", OLLAMA_API_BASE)
        return

    logger.info("Ollama server is not running. Starting it.")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if server_ready():
            logger.info("Ollama server is running at %s.", OLLAMA_API_BASE)
            return
        time.sleep(1)
    raise RuntimeError(f"Ollama server did not start at {OLLAMA_API_BASE}.")


def server_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_API_BASE}/api/tags", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def download_models() -> set[str]:
    installed = installed_models()
    for spec in MODELS:
        tag = spec["tag"]
        label = spec["label"]
        if tag in installed:
            logger.info("%s is already installed.", label)
            continue

        need = spec["download_gb"] * GB
        free = free_bytes()
        if free < need:
            logger.error(
                "Cannot download %s. It needs about %d GB and %.0f GB is free.",
                label,
                spec["download_gb"],
                free / GB,
            )
            continue

        logger.info("Downloading %s (about %d GB).", label, spec["download_gb"])
        result = subprocess.run(["ollama", "pull", tag])
        if result.returncode != 0:
            logger.error("Download of %s failed with exit code %s.", label, result.returncode)
            continue
        logger.info("Downloaded %s.", label)
        installed.add(tag)
    return installed


def installed_models() -> set[str]:
    with urllib.request.urlopen(f"{OLLAMA_API_BASE}/api/tags", timeout=10) as response:
        payload = json.load(response)
    return {model["name"] for model in payload.get("models", [])}


def free_bytes() -> int:
    for candidate in (
        os.environ.get("OLLAMA_MODELS"),
        "/usr/share/ollama",
        os.path.expanduser("~/.ollama"),
        "/",
    ):
        if candidate and os.path.isdir(candidate):
            return shutil.disk_usage(candidate).free
    return shutil.disk_usage("/").free


def read_hardware() -> tuple[list[tuple[str, float]], float]:
    gpus = []
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                name, memory = (part.strip() for part in line.split(",", 1))
                gpus.append((name, int(memory) / 1024))

    ram_gb = 0.0
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemTotal"):
                ram_gb = int(line.split()[1]) / 1024 / 1024
                break

    if gpus:
        gpu_text = ", ".join(f"{name} with {memory:.1f} GB" for name, memory in gpus)
    else:
        gpu_text = "no NVIDIA GPU detected"
    logger.info("Hardware: %s; system RAM %.1f GB.", gpu_text, ram_gb)
    return gpus, ram_gb


def choose_model(installed: set[str], gpus: list[tuple[str, float]], ram_gb: float) -> str:
    largest_gpu = max((memory for _, memory in gpus), default=0.0)
    total_gpu = sum(memory for _, memory in gpus)
    spec_120 = MODELS[1]
    spec_20 = MODELS[0]

    fits_120 = largest_gpu >= spec_120["memory_gb"] or total_gpu >= spec_120["memory_gb"]
    fits_20 = max(ram_gb, largest_gpu, total_gpu) >= spec_20["memory_gb"]

    if fits_120 and spec_120["tag"] in installed:
        logger.info(
            "Using %s. GPU memory is %.1f GB, which meets the %d GB requirement.",
            spec_120["label"],
            max(largest_gpu, total_gpu),
            spec_120["memory_gb"],
        )
        return spec_120["tag"]

    if fits_120 and spec_120["tag"] not in installed:
        logger.error(
            "This machine can run %s, but that model is not installed.",
            spec_120["label"],
        )
    elif not fits_120:
        if gpus:
            logger.info(
                "%s needs about %d GB of GPU memory. The largest GPU has %.1f GB, so this machine cannot run it.",
                spec_120["label"],
                spec_120["memory_gb"],
                largest_gpu,
            )
        else:
            logger.info(
                "%s needs about %d GB of GPU memory. No NVIDIA GPU was detected, so this machine cannot run it.",
                spec_120["label"],
                spec_120["memory_gb"],
            )

    if fits_20 and spec_20["tag"] in installed:
        logger.info(
            "Using %s. It needs about %d GB of memory, and this machine has %.1f GB of GPU memory and %.1f GB of RAM.",
            spec_20["label"],
            spec_20["memory_gb"],
            largest_gpu,
            ram_gb,
        )
        return spec_20["tag"]

    if spec_20["tag"] not in installed:
        raise RuntimeError(f"{spec_20['label']} is not installed, and {spec_120['label']} cannot be used.")
    raise RuntimeError(
        f"{spec_20['label']} needs about {spec_20['memory_gb']} GB of memory. "
        f"This machine has {largest_gpu:.1f} GB of GPU memory and {ram_gb:.1f} GB of RAM."
    )


def fallback_model(model: str) -> str | None:
    """Return the other installed model only when this machine can run it."""
    other = "gpt-oss:20b" if model == "gpt-oss:120b" else "gpt-oss:120b"
    if other not in _installed:
        return None
    if other == "gpt-oss:120b" and _gpu_gb < 80:
        return None
    if other == "gpt-oss:20b" and max(_ram_gb, _gpu_gb) < 16:
        return None
    return other


def label_for(tag: str) -> str:
    for spec in MODELS:
        if spec["tag"] == tag:
            return spec["label"]
    return tag


if __name__ == "__main__":
    print(prepare_local_model())
