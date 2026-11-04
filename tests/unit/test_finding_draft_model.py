"""
Unit tests for the FindingDraft Pydantic model (Objective 1).

Covers:
- Valid construction
- conclusion non-empty validator
- limitations coercion from string and list
- primary_evidence_call_ids defaults
- model_json_schema() is exportable (needed for constrained decoding)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datachef.agent.finding_composer import _parse_finding_draft
from datachef.context.models import FindingDraft

# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------


def test_finding_draft_minimal_valid():
    """Only conclusion is required; all other fields default correctly."""
    draft = FindingDraft(conclusion="Sales declined by 12% in Q3.")
    assert draft.conclusion == "Sales declined by 12% in Q3."
    assert draft.interpretation == ""
    assert draft.limitations == []
    assert draft.primary_evidence_call_ids == []


def test_finding_draft_all_fields():
    """All fields populated round-trip correctly."""
    draft = FindingDraft(
        conclusion="Revenue peaked in July.",
        interpretation="The summer promotion drove the spike.",
        limitations=["Only one year of data was available.", "No external factors modelled."],
        primary_evidence_call_ids=["call-abc", "call-def"],
    )
    assert draft.conclusion == "Revenue peaked in July."
    assert draft.interpretation == "The summer promotion drove the spike."
    assert len(draft.limitations) == 2
    assert draft.primary_evidence_call_ids == ["call-abc", "call-def"]


# ---------------------------------------------------------------------------
# conclusion validator
# ---------------------------------------------------------------------------


def test_finding_draft_empty_conclusion_raises():
    """An empty string conclusion must raise ValidationError (caught by min_length constraint)."""
    with pytest.raises(ValidationError, match="string_too_short|conclusion must not be empty"):
        FindingDraft(conclusion="")


def test_finding_draft_whitespace_only_conclusion_raises():
    """A whitespace-only conclusion must raise ValidationError."""
    with pytest.raises(ValidationError, match="conclusion must not be empty"):
        FindingDraft(conclusion="   \n\t  ")


def test_finding_draft_conclusion_stripped():
    """Leading/trailing whitespace on a non-empty conclusion is stripped."""
    draft = FindingDraft(conclusion="  Sales declined.  ")
    assert draft.conclusion == "Sales declined."


# ---------------------------------------------------------------------------
# limitations coercion
# ---------------------------------------------------------------------------


def test_finding_draft_limitations_from_string():
    """A bare multi-line string is split into a list of non-empty lines."""
    raw = "Only Q1 data.\nNo causal inference possible.\n"
    draft = FindingDraft(conclusion="OK", limitations=raw)
    assert draft.limitations == ["Only Q1 data.", "No causal inference possible."]


def test_finding_draft_limitations_from_single_line_string():
    """A single-line string becomes a one-element list."""
    draft = FindingDraft(conclusion="OK", limitations="Only one season.")
    assert draft.limitations == ["Only one season."]


def test_finding_draft_limitations_from_list_filters_blanks():
    """Empty strings and whitespace-only items are filtered out of a list."""
    draft = FindingDraft(conclusion="OK", limitations=["Valid caveat.", "", "  ", "Another."])
    assert draft.limitations == ["Valid caveat.", "Another."]


def test_finding_draft_limitations_from_list_of_non_strings():
    """Non-string items in the list are coerced to str."""
    draft = FindingDraft(conclusion="OK", limitations=[42, None, "Real caveat."])
    # None → "None" is non-empty, 42 → "42" is non-empty
    assert "Real caveat." in draft.limitations


def test_finding_draft_limitations_empty_list_default():
    """Omitting limitations produces an empty list."""
    draft = FindingDraft(conclusion="OK")
    assert draft.limitations == []


# ---------------------------------------------------------------------------
# Schema export (needed for constrained decoding)
# ---------------------------------------------------------------------------


def test_finding_draft_schema_is_exportable():
    """model_json_schema() must return a non-empty dict with required fields."""
    schema = FindingDraft.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "conclusion" in schema["properties"]
    assert "limitations" in schema["properties"]
    assert "primary_evidence_call_ids" in schema["properties"]


def test_finding_draft_schema_marks_conclusion_required():
    """conclusion must appear in the required list of the JSON schema."""
    schema = FindingDraft.model_json_schema()
    required = schema.get("required", [])
    assert "conclusion" in required



# ---------------------------------------------------------------------------
# _parse_finding_draft tests (Objective 2)
# ---------------------------------------------------------------------------


def test_parse_finding_draft_direct_json():
    raw = '{"conclusion": "Revenue rose by 10%.", "interpretation": "Strong growth.", "limitations": ["Limited data."]}'
    draft = _parse_finding_draft(raw)
    assert draft is not None
    assert draft.conclusion == "Revenue rose by 10%."
    assert draft.interpretation == "Strong growth."
    assert draft.limitations == ["Limited data."]


def test_parse_finding_draft_markdown_code_fence():
    raw = """Here is the structured conclusion:
```json
{
  "conclusion": "Churn rate decreased to 3.5%.",
  "interpretation": "Customer retention strategies were effective.",
  "limitations": ["Cohort size was small."],
  "primary_evidence_call_ids": ["call-123"]
}
```
Hope this helps!"""
    draft = _parse_finding_draft(raw)
    assert draft is not None
    assert draft.conclusion == "Churn rate decreased to 3.5%."
    assert draft.interpretation == "Customer retention strategies were effective."
    assert draft.limitations == ["Cohort size was small."]
    assert draft.primary_evidence_call_ids == ["call-123"]


def test_parse_finding_draft_markdown_fence_without_json_tag():
    raw = """```
{
  "conclusion": "No significant correlation found.",
  "interpretation": "",
  "limitations": []
}
```"""
    draft = _parse_finding_draft(raw)
    assert draft is not None
    assert draft.conclusion == "No significant correlation found."


def test_parse_finding_draft_embedded_in_prose():
    raw = """Based on the investigation results, here is the conclusion object:
{"conclusion": "Q4 performance was optimal.", "primary_evidence_call_ids": ["c1"]}
This concludes the analysis."""
    draft = _parse_finding_draft(raw)
    assert draft is not None
    assert draft.conclusion == "Q4 performance was optimal."
    assert draft.primary_evidence_call_ids == ["c1"]


def test_parse_finding_draft_missing_required_field_returns_none():
    raw = '{"interpretation": "Missing conclusion field."}'
    assert _parse_finding_draft(raw) is None


def test_parse_finding_draft_empty_conclusion_returns_none():
    raw = '{"conclusion": "   ", "interpretation": "Blank conclusion."}'
    assert _parse_finding_draft(raw) is None


def test_parse_finding_draft_truncated_json_returns_none():
    raw = '{"conclusion": "Truncated json...'
    assert _parse_finding_draft(raw) is None


def test_parse_finding_draft_empty_or_whitespace_returns_none():
    assert _parse_finding_draft("") is None
    assert _parse_finding_draft("   \n\t  ") is None


def test_parse_finding_draft_pure_prose_returns_none():
    raw = "CONCLUSION: Total revenue is $500,000.\nINTERPRETATION: Good year."
    assert _parse_finding_draft(raw) is None
