"""
Provider registry — selects and instantiates the configured provider.
"""

from __future__ import annotations

from datachef import config
from datachef.providers.base import BaseProvider
from datachef.providers.huggingface import HuggingFaceProvider
from datachef.providers.ollama import OllamaProvider


def get_provider(
    provider: str | None = None,
    api_key: str | None = None,
    model_id: str | None = None,
    host: str | None = None,
) -> BaseProvider:
    """
    Return the configured inference provider based on DATACHEF_PROVIDER or runtime overrides.
    Supports: 'ollama' (local open-source) and 'huggingface' (open-weight hosted).
    """
    active_provider = (provider or config.PROVIDER).lower()

    if active_provider == "ollama":
        return OllamaProvider(
            host=host or config.OLLAMA_HOST,
            model_id=model_id or config.OLLAMA_MODEL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
        )

    if active_provider == "huggingface":
        return HuggingFaceProvider(
            api_key=api_key if api_key is not None else config.HF_API_KEY,
            model_id=model_id or config.HF_MODEL_ID,
            inference_provider=config.HF_INFERENCE_PROVIDER,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
        )

    raise ValueError(
        f"Unknown provider '{active_provider}'. Supported providers: ollama, huggingface."
    )
