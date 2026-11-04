"""
Tests for InvestigationAgent controlled retry behavior on malformed structured output.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from datachef.agent.investigation_agent import InvestigationAgent
from datachef.agent.prompt_builder import PromptBuilder
from datachef.analytics.profiler import DatasetProfiler
from datachef.context.models import InvestigationState
from datachef.providers.base import BaseProvider, ProviderResponse
from datachef.tools.registry import ToolRegistry


class RetryTestProvider(BaseProvider):
    """Mocks a sequence of provider responses for retry testing."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.call_count = 0
        self.recorded_messages: list[list[dict[str, Any]]] = []
        self.recorded_response_formats: list[Any] = []

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.recorded_messages.append(messages)
        self.recorded_response_formats.append(response_format)
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    def model_id(self) -> str:
        return "mock-retry-model"

    def provider_name(self) -> str:
        return "mock-retry-provider"


def _build_test_agent(responses: list[ProviderResponse]) -> tuple[InvestigationAgent, RetryTestProvider]:
    df = pl.DataFrame({"x": [10, 20, 30]})
    profile = DatasetProfiler().profile(df, "test.csv")
    state = InvestigationState(dataset_profile=profile)
    registry = ToolRegistry()
    prompt_builder = PromptBuilder(profile, registry)
    provider = RetryTestProvider(responses)
    agent = InvestigationAgent(provider, registry, prompt_builder, df, state)
    return agent, provider


def test_agent_retries_once_on_malformed_json_and_succeeds():
    """Agent detects invalid JSON on final turn, requests correction, and succeeds on retry."""
    turn_1_bad_json = ProviderResponse(
        content="I think the mean of x is 20, but I forgot to output JSON.",
        finish_reason="stop",
    )
    turn_2_valid_json = ProviderResponse(
        content=json.dumps({
            "conclusion": "The mean of x is 20.0.",
            "interpretation": "Values are evenly spaced.",
            "limitations": ["Small sample."],
        }),
        finish_reason="stop",
    )

    agent, provider = _build_test_agent([turn_1_bad_json, turn_2_valid_json])
    finding = agent.investigate("What is the mean of x?")

    # Provider should have been called twice (turn 1 + 1 retry)
    assert provider.call_count == 2
    # The second call should have included the retry feedback message
    retry_prompt = provider.recorded_messages[1][-1]["content"]
    assert "could not be parsed as a valid FindingDraft JSON object" in retry_prompt
    # Response format was passed on retry turn (when forced into final conclusion mode)
    assert provider.recorded_response_formats[1] is not None

    # Finding was parsed from the structured output
    assert finding.conclusion == "The mean of x is 20.0."
    assert finding.interpretation == "Values are evenly spaced."
    assert "Small sample." in finding.limitations


def test_agent_retries_once_then_falls_back_to_regex_on_second_failure():
    """If retry also fails to produce valid JSON, agent falls back to regex parser without crashing."""
    turn_1_bad_json = ProviderResponse(
        content="Not JSON.",
        finish_reason="stop",
    )
    turn_2_still_bad_json = ProviderResponse(
        content=(
            "### CONCLUSION\n"
            "The mean of x is 20.0.\n\n"
            "### INTERPRETATION\n"
            "Interpreted via fallback.\n\n"
            "### LIMITATIONS\n"
            "- Fallback used."
        ),
        finish_reason="stop",
    )

    agent, provider = _build_test_agent([turn_1_bad_json, turn_2_still_bad_json])
    finding = agent.investigate("What is the mean of x?")

    # Exactly 2 calls: 1 initial + 1 retry (no endless loop)
    assert provider.call_count == 2
    # Fallback parsed the markdown headings
    assert "The mean of x is 20.0." in finding.conclusion
    assert "Interpreted via fallback." in finding.interpretation
    assert any("Fallback used." in lim for lim in finding.limitations)


def test_agent_does_not_retry_if_turn_limit_already_reached(monkeypatch):
    """When MAX_TURNS is reached on the first turn, agent does not exceed MAX_TURNS to retry."""
    from datachef import config
    monkeypatch.setattr(config, "MAX_TURNS", 1)

    turn_1_bad_json = ProviderResponse(
        content="Not JSON at turn limit.",
        finish_reason="stop",
    )

    agent, provider = _build_test_agent([turn_1_bad_json])
    finding = agent.investigate("What is the mean of x?")

    # Provider called only once because MAX_TURNS=1 was exhausted
    assert provider.call_count == 1
    assert finding.conclusion == "Not JSON at turn limit."
