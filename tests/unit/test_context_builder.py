"""
Unit tests for ContextBuilder.
"""

from __future__ import annotations

import polars as pl

from datachef.analytics.profiler import DatasetProfiler
from datachef.context.context_builder import ContextBuilder
from datachef.context.models import InvestigationState, ToolCallRecord


def test_context_builder_schema_block():
    df = pl.DataFrame({
        "customer_id": [1, 2, 3],
        "spend": [100.5, 200.0, 300.2],
        "category": ["A", "B", "A"],
    })
    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_context.csv")

    builder = ContextBuilder(profile)
    schema_block = builder.build_schema_block()

    assert "customer_id" in schema_block
    assert "spend" in schema_block
    assert "category" in schema_block
    assert "3 rows" in schema_block


def test_context_builder_turn_context_no_raw_rows():
    df = pl.DataFrame({
        "val_1": [1, 2, 3],
        "val_2": ["x", "y", "z"],
    })
    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_no_rows.csv")

    state = InvestigationState(dataset_profile=profile)
    state.add_tool_record(
        ToolCallRecord(
            turn_index=1,
            tool_name="get_column_stats",
            arguments={"columns": ["val_1"]},
            result_summary='{"mean": 2.0, "min": 1.0, "max": 3.0}',
            validation_passed=True,
        )
    )

    builder = ContextBuilder(profile)
    summary = builder.build_profile_summary()
    assert "val_1" in summary
    assert "val_2" in summary



def test_context_builder_turn_instruction_intermediate_turn():
    df = pl.DataFrame({"x": [1, 2]})
    profile = DatasetProfiler().profile(df, source_path="test.csv")
    builder = ContextBuilder(profile)

    instruction = builder.build_turn_instruction(
        tool_calls_used=2,
        max_tool_calls=5,
        turns_used=1,
        max_turns=3,
        is_final_turn=False,
    )
    assert "2/5 tool calls" in instruction
    assert "1/3 turns" in instruction
    assert "Call a tool if more evidence is needed" in instruction


def test_context_builder_turn_instruction_final_turn_contains_json_schema():
    df = pl.DataFrame({"x": [1, 2]})
    profile = DatasetProfiler().profile(df, source_path="test.csv")
    builder = ContextBuilder(profile)

    instruction = builder.build_turn_instruction(
        tool_calls_used=5,
        max_tool_calls=5,
        turns_used=2,
        max_turns=3,
        is_final_turn=True,
    )
    assert "This is your final turn" in instruction
    assert "Do NOT call any more tools" in instruction
    assert "FindingDraft" in instruction or "conclusion" in instruction
    assert "Return ONLY valid JSON" in instruction
