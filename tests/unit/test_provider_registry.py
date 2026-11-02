"""
Unit tests for provider registry and runtime parameter overrides.
"""

from __future__ import annotations

import pytest

from datachef import config
from datachef.providers.huggingface import HuggingFaceProvider
from datachef.providers.ollama import OllamaProvider
from datachef.providers.registry import get_provider


def test_get_provider_ollama_overrides() -> None:
    provider = get_provider(
        provider="ollama",
        model_id="llama3.2:3b",
        host="http://custom-host:11434",
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.model_id() == "llama3.2:3b"
    assert provider._host == "http://custom-host:11434"


def test_get_provider_huggingface_overrides() -> None:
    provider = get_provider(
        provider="huggingface",
        api_key="hf_test_override_token",
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    )
    assert isinstance(provider, HuggingFaceProvider)
    assert provider._api_key == "hf_test_override_token"
    assert provider.model_id() == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_get_provider_defaults_to_config() -> None:
    provider = get_provider()
    if config.PROVIDER == "ollama":
        assert isinstance(provider, OllamaProvider)
    elif config.PROVIDER == "huggingface":
        assert isinstance(provider, HuggingFaceProvider)


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'unsupported'"):
        get_provider(provider="unsupported")


def test_get_provider_blank_overrides_are_preserved() -> None:
    # Explicit blank model override must not silently fall back to config values
    ollama = get_provider(provider="ollama", model_id="")
    assert isinstance(ollama, OllamaProvider)
    assert ollama.model_id() == ""

    hf = get_provider(provider="huggingface", model_id="")
    assert isinstance(hf, HuggingFaceProvider)
    assert hf.model_id() == ""
