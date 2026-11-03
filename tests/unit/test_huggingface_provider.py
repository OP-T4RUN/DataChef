"""
Unit tests for HuggingFaceProvider constrained decoding and degradation fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from huggingface_hub.errors import HfHubHTTPError

from datachef.providers.huggingface import HuggingFaceProvider


def test_huggingface_passes_response_format_on_final_turn():
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"conclusion": "done"}'
    mock_resp.choices[0].message.tool_calls = None
    mock_resp.choices[0].finish_reason = "stop"
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    provider = HuggingFaceProvider(api_key="mock-key", model_id="mock-model")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    provider._client = mock_client

    resp = provider.chat_completion(
        messages=[{"role": "user", "content": "Conclude"}],
        tools=[],
        tool_choice="none",
        response_format={"type": "json_object"},
    )
    assert mock_client.chat.completions.create.called
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs.get("response_format") == {"type": "json_object"}
    assert resp.content == '{"conclusion": "done"}'


def test_huggingface_degrades_gracefully_on_400_response_format():
    """If provider rejects response_format with HTTP 400, it retries without response_format."""
    mock_err_response = MagicMock(status_code=400)
    err_400 = HfHubHTTPError("Unsupported parameter: response_format", response=mock_err_response)

    mock_fallback_resp = MagicMock()
    mock_fallback_resp.choices = [MagicMock()]
    mock_fallback_resp.choices[0].message.content = "Fallback conclusion."
    mock_fallback_resp.choices[0].message.tool_calls = None
    mock_fallback_resp.choices[0].finish_reason = "stop"
    mock_fallback_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    provider = HuggingFaceProvider(api_key="mock-key", model_id="mock-model")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [err_400, mock_fallback_resp]
    provider._client = mock_client

    resp = provider.chat_completion(
        messages=[{"role": "user", "content": "Conclude"}],
        tools=[],
        tool_choice="none",
        response_format={"type": "json_object"},
    )
    assert mock_client.chat.completions.create.call_count == 2
    # First call included response_format
    assert mock_client.chat.completions.create.call_args_list[0][1].get("response_format") == {"type": "json_object"}
    # Second call omitted response_format
    assert "response_format" not in mock_client.chat.completions.create.call_args_list[1][1]
    assert resp.content == "Fallback conclusion."
