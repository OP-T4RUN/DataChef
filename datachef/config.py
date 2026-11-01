"""Application configuration loaded from environment variables."""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Auto-bootstrap .env from .env.example if missing
_root = Path(__file__).resolve().parent.parent
_env_path = _root / ".env"
_example_path = _root / ".env.example"

if not _env_path.exists() and _example_path.exists():
    with contextlib.suppress(OSError):
        shutil.copy(_example_path, _env_path)

# Load .env file if present (silently ignored when absent)
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Provider & Inference Engine (Ollama or Hugging Face)
# ---------------------------------------------------------------------------

PROVIDER: str = _env(
    "DATACHEF_PROVIDER",
    "ollama" if os.environ.get("OLLAMA_MODEL") or not os.environ.get("HF_API_KEY") else "huggingface",
).lower()

# Hugging Face Settings
HF_API_KEY: str = _env("HF_API_KEY", "")
HF_MODEL_ID: str = _env("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
HF_INFERENCE_PROVIDER: str = _env("HF_INFERENCE_PROVIDER", "auto")

# Ollama Settings (Local open-source inference)
OLLAMA_HOST: str = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "gemma4")

# ---------------------------------------------------------------------------
# Agent loop limits
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS: int = _env_int("DATACHEF_MAX_TOOL_CALLS", 10)
MAX_TURNS: int = _env_int("DATACHEF_MAX_TURNS", 8)
MAX_WALL_SECONDS: int = _env_int("DATACHEF_MAX_WALL_SECONDS", 120)
MAX_RESULT_TOKENS: int = _env_int("DATACHEF_MAX_RESULT_TOKENS", 2000)
MAX_DISCOVERY_CANDIDATES: int = _env_int("DATACHEF_MAX_DISCOVERY_CANDIDATES", 6)

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------

TEMPERATURE: float = _env_float("DATACHEF_TEMPERATURE", 0.2)
MAX_RESPONSE_TOKENS: int = _env_int("DATACHEF_MAX_RESPONSE_TOKENS", 1024)
REQUEST_TIMEOUT: int = _env_int("DATACHEF_REQUEST_TIMEOUT", 30)
MAX_RETRIES: int = _env_int("DATACHEF_MAX_RETRIES", 3)

# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

CONTEXT_HISTORY_N: int = _env_int("DATACHEF_CONTEXT_HISTORY_N", 3)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

def _resolve_storage_dir(env_key: str, subdir_name: str) -> Path:
    """Resolve storage path with graceful fallback if home directory is read-only."""
    candidates = [
        os.environ.get(env_key),
        Path.home() / ".datachef" / subdir_name,
        Path(__file__).resolve().parent.parent / ".datachef" / subdir_name,
        Path("/tmp") / "datachef" / subdir_name,
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError:
                continue
    return Path("/tmp") / "datachef" / subdir_name


INVESTIGATION_DIR: Path = _resolve_storage_dir("DATACHEF_INVESTIGATION_DIR", "investigations")
CACHE_DIR: Path = _resolve_storage_dir("DATACHEF_CACHE_DIR", "cache")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate() -> list[str]:
    """Return a list of configuration error messages (empty if config is valid)."""
    errors: list[str] = []
    positive_ints = {
        "DATACHEF_MAX_TOOL_CALLS": MAX_TOOL_CALLS,
        "DATACHEF_MAX_TURNS": MAX_TURNS,
        "DATACHEF_MAX_WALL_SECONDS": MAX_WALL_SECONDS,
        "DATACHEF_MAX_RESULT_TOKENS": MAX_RESULT_TOKENS,
        "DATACHEF_MAX_DISCOVERY_CANDIDATES": MAX_DISCOVERY_CANDIDATES,
        "DATACHEF_MAX_RESPONSE_TOKENS": MAX_RESPONSE_TOKENS,
        "DATACHEF_REQUEST_TIMEOUT": REQUEST_TIMEOUT,
        "DATACHEF_CONTEXT_HISTORY_N": CONTEXT_HISTORY_N,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            errors.append(f"{name} must be greater than zero (got {value}).")
    if MAX_RETRIES < 0:
        errors.append(f"DATACHEF_MAX_RETRIES must be zero or greater (got {MAX_RETRIES}).")
    if not 0 <= TEMPERATURE <= 2:
        errors.append(f"DATACHEF_TEMPERATURE must be between 0 and 2 (got {TEMPERATURE}).")
    if PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            errors.append("OLLAMA_MODEL is not set. Specify e.g. 'gemma4' or 'gemma2:9b'.")
        # Test connection to Ollama host (quick non-blocking check)
        try:
            import requests
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
            if resp.status_code != 200:
                errors.append(f"Ollama server returned status {resp.status_code} at {OLLAMA_HOST}.")
        except Exception:
            errors.append(
                f"Cannot connect to Ollama at {OLLAMA_HOST}. "
                "Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull "
                f"{OLLAMA_MODEL}`)."
            )
    elif PROVIDER == "huggingface":
        if not HF_API_KEY:
            errors.append(
                "HF_API_KEY is not set. "
                "Copy .env.example to .env and add your Hugging Face API key, "
                "or set DATACHEF_PROVIDER=ollama to run fully locally."
            )
        if not HF_MODEL_ID:
            errors.append("HF_MODEL_ID is not set.")
    else:
        errors.append(f"Unknown provider '{PROVIDER}'. Supported providers: 'ollama', 'huggingface'.")

    return errors
