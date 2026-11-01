"""
SessionStore — manages active session state across Streamlit reruns.

Uses Streamlit's session_state as the in-memory store.
Also provides helpers to initialize and access the shared agent components.
"""

from __future__ import annotations

import io
from typing import Any

import polars as pl
import streamlit as st

from datachef import config
from datachef.agent.investigation_agent import InvestigationAgent
from datachef.agent.prompt_builder import PromptBuilder
from datachef.analytics.profiler import DatasetProfiler
from datachef.context.dataset_store import DatasetStore
from datachef.context.models import InvestigationState
from datachef.context.profile_cache import ProfileCache
from datachef.persistence.investigation_record import InvestigationRecord
from datachef.providers.registry import get_provider
from datachef.tools.registry import ToolRegistry

_KEYS = {
    "state": "datachef_investigation_state",
    "dataset_store": "datachef_dataset_store",
    "agent": "datachef_agent",
    "profiler": "datachef_profiler",
    "record_store": "datachef_record_store",
    "tool_registry": "datachef_tool_registry",
    "config_errors": "datachef_config_errors",
}


def initialize() -> None:
    """Initialize all session state objects on first run."""
    if _KEYS["state"] not in st.session_state:
        st.session_state[_KEYS["state"]] = InvestigationState()
    if _KEYS["dataset_store"] not in st.session_state:
        st.session_state[_KEYS["dataset_store"]] = DatasetStore()
    if _KEYS["profiler"] not in st.session_state:
        st.session_state[_KEYS["profiler"]] = DatasetProfiler()
    if _KEYS["record_store"] not in st.session_state:
        st.session_state[_KEYS["record_store"]] = InvestigationRecord()
    if _KEYS["tool_registry"] not in st.session_state:
        st.session_state[_KEYS["tool_registry"]] = ToolRegistry()
    if _KEYS["config_errors"] not in st.session_state:
        st.session_state[_KEYS["config_errors"]] = config.validate()


def get_state() -> InvestigationState:
    return st.session_state[_KEYS["state"]]


def get_dataset_store() -> DatasetStore:
    return st.session_state[_KEYS["dataset_store"]]


def get_tool_registry() -> ToolRegistry:
    return st.session_state[_KEYS["tool_registry"]]


def get_record_store() -> InvestigationRecord:
    return st.session_state[_KEYS["record_store"]]


def get_config_errors() -> list[str]:
    return st.session_state.get(_KEYS["config_errors"], [])


def get_active_provider_info() -> tuple[str, str]:
    """Return (provider_name, model_name)."""
    prov = (st.session_state.get("datachef_runtime_provider") or config.PROVIDER).lower()
    if prov == "ollama":
        model = st.session_state.get("datachef_runtime_model_id") or config.OLLAMA_MODEL
    else:
        model = st.session_state.get("datachef_runtime_model_id") or config.HF_MODEL_ID
    return prov, model


def update_provider_settings(
    provider: str,
    api_key: str = "",
    model_id: str = "",
    host: str = "",
) -> list[str]:
    """Update active provider settings at runtime and re-validate."""
    provider = provider.lower()
    st.session_state["datachef_runtime_provider"] = provider
    st.session_state["datachef_runtime_api_key"] = api_key
    st.session_state["datachef_runtime_model_id"] = model_id
    st.session_state["datachef_runtime_host"] = host

    errors: list[str] = []
    if provider == "ollama":
        target_model = model_id or config.OLLAMA_MODEL
        target_host = host or config.OLLAMA_HOST
        if not target_model:
            errors.append("Ollama model name is required.")
        else:
            try:
                import requests
                resp = requests.get(f"{target_host}/api/tags", timeout=1.5)
                if resp.status_code != 200:
                    errors.append(f"Ollama server returned status {resp.status_code} at {target_host}.")
            except Exception:
                errors.append(
                    f"Cannot connect to Ollama at {target_host}. "
                    f"Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull {target_model}`)."
                )
    elif provider == "huggingface":
        target_key = api_key if api_key != "" else config.HF_API_KEY
        target_model = model_id or config.HF_MODEL_ID
        if not target_key:
            errors.append("Hugging Face API key is required. Enter your token above.")
        if not target_model:
            errors.append("Hugging Face model ID is required.")

    st.session_state[_KEYS["config_errors"]] = errors

    # Reset cached agent so the next build_agent() uses the updated provider
    if _KEYS["agent"] in st.session_state:
        del st.session_state[_KEYS["agent"]]

    return errors


def load_dataset(uploaded_file: Any) -> tuple[pl.DataFrame, bool]:
    """
    Load an uploaded Streamlit file into the DatasetStore.
    Returns (DataFrame, is_cached) tuple.
    """
    suffix = str(uploaded_file.name).lower().rsplit(".", 1)[-1]
    payload = uploaded_file.getvalue()
    if not payload:
        raise ValueError("The uploaded file is empty.")
    buffer = io.BytesIO(payload)
    if suffix == "csv":
        df = pl.read_csv(buffer, infer_schema_length=10_000, try_parse_dates=True)
    elif suffix in {"parquet", "pq"}:
        df = pl.read_parquet(buffer)
    else:
        raise ValueError("DataChef accepts CSV and Parquet files only.")

    get_dataset_store().load_dataframe(df, source_path=uploaded_file.name)
    return df, False


def profile_dataset(df: pl.DataFrame, source_path: str = "<uploaded>") -> None:
    """Run profiling and update the session state."""
    cache = ProfileCache()
    # Try cache first (won't hit for in-memory uploads, but useful for local paths)
    profile = cache.get(source_path)
    if profile is None:
        profiler: DatasetProfiler = st.session_state[_KEYS["profiler"]]
        profile = profiler.profile(df, source_path)
        cache.put(profile)

    state = get_state()
    state.dataset_profile = profile


def build_agent() -> InvestigationAgent:
    """Build and cache the InvestigationAgent for the current session."""
    if _KEYS["agent"] in st.session_state:
        return st.session_state[_KEYS["agent"]]

    state = get_state()
    store = get_dataset_store()
    registry = get_tool_registry()
    runtime_prov = st.session_state.get("datachef_runtime_provider")
    runtime_key = st.session_state.get("datachef_runtime_api_key")
    runtime_model = st.session_state.get("datachef_runtime_model_id")
    runtime_host = st.session_state.get("datachef_runtime_host")

    provider = get_provider(
        provider=runtime_prov,
        api_key=runtime_key,
        model_id=runtime_model,
        host=runtime_host,
    )
    df = store.get_frame()

    if state.dataset_profile is None:
        raise RuntimeError("No dataset profile available. Profile the dataset first.")

    prompt_builder = PromptBuilder(state.dataset_profile, registry)
    agent = InvestigationAgent(
        provider=provider,
        registry=registry,
        prompt_builder=prompt_builder,
        df=df,
        state=state,
    )
    st.session_state[_KEYS["agent"]] = agent
    return agent


def reset_session() -> None:
    """Clear all session state (e.g., when a new dataset is uploaded)."""
    for key in _KEYS.values():
        if key in st.session_state:
            del st.session_state[key]
    initialize()
