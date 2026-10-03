"""Tests for M1 aggregation helpers."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("m1_r5", Path("scripts/vocab/07_m1_r5.py"))
m1_r5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_r5)

RULE = {"pass": 0.8, "conditional": 0.75}


def row(stratum, correct, sentence="s"):
    return {"stratum": stratum, "term_id": "M:x", "key": "x", "matched_tokens": "x", "sentence": sentence, "correct": correct}


def test_surface_stratum():
    assert m1_r5.surface_stratum("M", "cs") == "acronym_cs"
    assert m1_r5.surface_stratum("M", "ci") == "method_ci"
    assert m1_r5.surface_stratum("T", "ci") == "task_ci"


def test_validate_flags_bad_values_and_changed_columns():
    original = [row("method_ci", ""), row("method_ci", "")]
    judged = [row("method_ci", "2"), row("method_ci", "1", sentence="edited")]
    errors, unfilled = m1_r5.validate_attempt(judged, original)
    assert unfilled == 0
    assert any("correct='2'" in e for e in errors)
    assert any("columns changed" in e for e in errors)


def test_validate_counts_unfilled_and_accepts_spreadsheet_floats():
    original = [row("task_ci", ""), row("task_ci", "")]
    errors, unfilled = m1_r5.validate_attempt([row("task_ci", "1.0"), row("task_ci", "")], original)
    assert errors == [] and unfilled == 1


def test_stratum_stats_status():
    rows = [row("method_ci", "1")] * 90 + [row("method_ci", "0")] * 10
    stats = m1_r5.stratum_stats(rows, ["method_ci"], RULE)["method_ci"]
    assert stats["correct"] == 90 and stats["status"] == "pass"


def test_final_by_stratum_uses_latest_attempt_per_stratum():
    report = [
        {"attempt": 0, "strata": {"method_ci": {"status": "fail"}, "acronym_cs": {"status": "fail"}}},
        {"attempt": 1, "strata": {"method_ci": {"status": "fail"}, "acronym_cs": {"status": "fail"}}},
        {"attempt": 2, "strata": {"acronym_cs": {"status": "pass"}}},
    ]
    final = m1_r5.final_by_stratum(report)
    assert final["method_ci"]["attempt"] == 1
    assert final["acronym_cs"] == {"status": "pass", "attempt": 2}