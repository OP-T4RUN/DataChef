"""
Unit tests for configuration storage resolution and dynamic validation overrides.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from datachef import config


def test_resolve_storage_dir_uses_tempdir_fallback(monkeypatch) -> None:
    # Force Home and workspace candidate paths to fail with OSError (e.g. read-only filesystem)
    orig_mkdir = Path.mkdir

    def failing_mkdir(self: Path, *args, **kwargs) -> None:
        # Allow temp dir paths to succeed, fail others
        temp_dir_str = tempfile.gettempdir()
        if temp_dir_str in str(self):
            orig_mkdir(self, *args, **kwargs)
        else:
            raise OSError("Read-only file system")

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    monkeypatch.delenv("TEST_FAILING_STORAGE_DIR", raising=False)

    resolved = config._resolve_storage_dir("TEST_FAILING_STORAGE_DIR", "test_subdir")
    assert str(resolved).startswith(tempfile.gettempdir())
    assert resolved.name == "test_subdir"


def test_config_validate_with_huggingface_overrides() -> None:
    # Valid HF credentials passed dynamically
    errors = config.validate(
        provider="huggingface",
        api_key="hf_valid_token",
        model_id="Qwen/Qwen2.5-72B-Instruct",
    )
    assert errors == []

    # Missing API key
    errors_missing_key = config.validate(
        provider="huggingface",
        api_key="",
        model_id="Qwen/Qwen2.5-72B-Instruct",
    )
    assert any("HF_API_KEY is not set" in err for err in errors_missing_key)


def test_config_validate_with_ollama_overrides() -> None:
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_get.return_value = mock_resp

        errors = config.validate(
            provider="ollama",
            host="http://custom-ollama:11434",
            model_id="llama3",
        )
        assert errors == []
        mock_get.assert_called_with("http://custom-ollama:11434/api/tags", timeout=1.5)

