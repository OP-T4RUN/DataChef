"""
Unit tests for runtime provider settings, state isolation, and re-validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datachef import config
from datachef.persistence import session_store


@pytest.fixture
def mock_session_state(monkeypatch: pytest.MonkeyPatch) -> dict:
    state: dict = {}
    monkeypatch.setattr(session_store.st, "session_state", state)
    return state


def test_get_active_provider_info_isolation(mock_session_state: dict) -> None:
    mock_session_state["datachef_runtime_provider"] = "ollama"
    mock_session_state["datachef_runtime_ollama_model_id"] = "gemma2:9b"
    mock_session_state["datachef_runtime_hf_model_id"] = "Qwen/Qwen2.5-72B-Instruct"

    prov, model = session_store.get_active_provider_info()
    assert prov == "ollama"
    assert model == "gemma2:9b"

    mock_session_state["datachef_runtime_provider"] = "huggingface"
    prov, model = session_store.get_active_provider_info()
    assert prov == "huggingface"
    assert model == "Qwen/Qwen2.5-72B-Instruct"


def test_update_provider_settings_does_not_corrupt_inactive_provider(
    mock_session_state: dict,
) -> None:
    # 1. Start with Ollama configuration
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_get.return_value = mock_resp

        errors = session_store.update_provider_settings(
            provider="ollama",
            host="http://localhost:11434",
            model_id="gemma4",
        )
        assert errors == []
        assert mock_session_state["datachef_runtime_provider"] == "ollama"
        assert mock_session_state["datachef_runtime_ollama_host"] == "http://localhost:11434"
        assert mock_session_state["datachef_runtime_ollama_model_id"] == "gemma4"

    # 2. Switch to Hugging Face and apply settings
    errors = session_store.update_provider_settings(
        provider="huggingface",
        api_key="hf_test_secret_key",
        model_id="Qwen/Qwen2.5-72B-Instruct",
    )
    assert errors == []
    assert mock_session_state["datachef_runtime_provider"] == "huggingface"
    assert mock_session_state["datachef_runtime_hf_api_key"] == "hf_test_secret_key"
    assert mock_session_state["datachef_runtime_hf_model_id"] == "Qwen/Qwen2.5-72B-Instruct"

    # Crucial assertion: Ollama settings must not be cleared or overwritten
    assert mock_session_state["datachef_runtime_ollama_host"] == "http://localhost:11434"
    assert mock_session_state["datachef_runtime_ollama_model_id"] == "gemma4"

    # 3. Switch back to Ollama with an updated model
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_get.return_value = mock_resp

        session_store.update_provider_settings(
            provider="ollama",
            model_id="mistral-nemo",
        )
        assert mock_session_state["datachef_runtime_provider"] == "ollama"
        assert mock_session_state["datachef_runtime_ollama_model_id"] == "mistral-nemo"
        assert mock_session_state["datachef_runtime_ollama_host"] == "http://localhost:11434"

        # Crucial assertion: Hugging Face API key must remain intact
        assert mock_session_state["datachef_runtime_hf_api_key"] == "hf_test_secret_key"


def test_update_provider_settings_clears_cached_agent(mock_session_state: dict) -> None:
    mock_session_state["datachef_agent"] = "dummy_agent_instance"

    with patch.object(config, "validate", return_value=[]):
        session_store.update_provider_settings(provider="ollama", model_id="gemma4")

    assert "datachef_agent" not in mock_session_state


def test_reset_session_preserves_runtime_provider_validation(
    mock_session_state: dict,
) -> None:
    mock_session_state["datachef_runtime_provider"] = "huggingface"
    mock_session_state["datachef_runtime_hf_api_key"] = "hf_valid_key"
    mock_session_state["datachef_runtime_hf_model_id"] = "Qwen/Qwen2.5-72B-Instruct"

    # In .env, provider might be ollama without a running daemon
    with patch.object(config, "validate", return_value=[]) as mock_validate:
        session_store.reset_session()
        # Ensure validate was invoked for the active runtime provider (huggingface)
        mock_validate.assert_called_with(
            provider="huggingface",
            api_key="hf_valid_key",
            model_id="Qwen/Qwen2.5-72B-Instruct",
        )
        assert session_store.get_config_errors() == []


def test_update_provider_settings_unsupported_provider_leaves_session_intact(mock_session_state: dict) -> None:
    mock_session_state["datachef_runtime_provider"] = "ollama"
    errors = session_store.update_provider_settings(provider="unsupported_provider")
    assert any("Unknown provider 'unsupported_provider'" in err for err in errors)
    assert any("Unknown provider 'unsupported_provider'" in err for err in session_store.get_config_errors())
    # Session provider must not be corrupted to the invalid provider
    assert mock_session_state["datachef_runtime_provider"] == "ollama"


def test_update_provider_settings_blank_model_override(mock_session_state: dict) -> None:
    errors = session_store.update_provider_settings(provider="ollama", model_id="")
    assert any("Ollama model name is required" in err for err in errors)
    # Active provider info must reflect the blank model instead of falling back to default
    prov, model = session_store.get_active_provider_info()
    assert prov == "ollama"
    assert model == ""


def test_get_active_provider_info_unsupported(mock_session_state: dict) -> None:
    mock_session_state["datachef_runtime_provider"] = "invalid_llm"
    prov, model = session_store.get_active_provider_info()
    assert prov == "invalid_llm"
    assert model == "unknown"


def test_build_agent_unsupported_provider_raises(mock_session_state: dict) -> None:
    mock_session_state["datachef_runtime_provider"] = "invalid_llm"
    state = session_store.InvestigationState()
    state.dataset_profile = MagicMock()
    mock_session_state["datachef_investigation_state"] = state
    mock_session_state["datachef_dataset_store"] = session_store.DatasetStore()
    mock_session_state["datachef_tool_registry"] = session_store.ToolRegistry()

    with pytest.raises(ValueError, match="Unknown provider 'invalid_llm'"):
        session_store.build_agent()

