"""Tests for R4 judgement validation and decision rules, using synthetic rows only."""
import importlib.util
from fractions import Fraction
from pathlib import Path

# The script name starts with a digit, so it is loaded by path
_spec = importlib.util.spec_from_file_location(
    "r4_decisions", Path(__file__).resolve().parents[1] / "scripts/vocab/03_r4_decisions.py"
)
r4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r4)

CFG = {"seed": 1, "n_per_stratum": 3, "allow_threshold": 0.6}


def row(tid, size_a, size_b, a="", b=""):
    out = {
        "term_id": tid,
        "key": tid,
        "t1_docs": str(size_a + size_b),
        "stratum_a_size": str(size_a),
        "stratum_b_size": str(size_b),
        "a_technical": a,
        "b_technical": b,
    }
    for s, size in (("a", size_a), ("b", size_b)):
        for i in range(1, 4):
            out[f"{s}{i}_sentence"] = "example" if i <= min(3, size) else ""
    return out


def errors(rows):
    return r4.validate(rows, {r["term_id"] for r in rows}, 3)[1]


def test_whole_accepts_spreadsheet_integers_only():
    assert r4.whole("2") == 2
    assert r4.whole("2.0") == 2
    assert r4.whole("2.5") is None
    assert r4.whole("1/2") is None
    assert r4.whole("-1") is None
    assert r4.whole("") is None


def test_share_at_threshold_is_allowed():
    parsed, errs = r4.validate([row("T:x", 3, 2, a="3", b="0")], {"T:x"}, 3)
    assert not errs
    share = r4.technical_share(parsed[0])
    assert share == Fraction(3, 5)
    assert share >= Fraction(str(CFG["allow_threshold"]))


def test_share_is_weighted_by_stratum_size():
    parsed, _ = r4.validate([row("T:x", 10, 90, a="3", b="1")], {"T:x"}, 3)
    assert r4.technical_share(parsed[0]) == Fraction(10 * 1 + 90 * Fraction(1, 3), 100)


def test_empty_stratum_accepts_blank_or_zero_only():
    assert not errors([row("T:x", 0, 5, a="", b="2")])
    assert not errors([row("T:x", 0, 5, a="0", b="2")])
    assert errors([row("T:x", 0, 5, a="1", b="2")])


def test_judgement_above_shown_examples_is_rejected():
    assert errors([row("T:x", 2, 5, a="3", b="2")])


def test_blank_judgement_is_unfilled_not_error():
    parsed, errs = r4.validate([row("T:x", 4, 5, a="2", b="")], {"T:x"}, 3)
    assert not errs
    assert parsed[0]["technical"]["b"] is None


def test_sheet_damage_is_detected():
    damaged = row("T:x", 4, 5, a="2", b="2")
    damaged["stratum_a_size"] = "3"
    assert errors([damaged])
    cut = row("T:x", 4, 5, a="2", b="2")
    cut["a2_sentence"] = ""
    assert errors([cut])
    short = row("T:x", 4, 5, a="2", b="2")
    short["stratum_b_size"] = None
    assert errors([short])
    assert r4.validate([row("T:x", 1, 1, a="1", b="1")], {"T:x", "T:y"}, 3)[1]


def test_source_mismatch_stops_on_vocab_or_config_change():
    meta = {"vocab_sha256": "abc", "review_config": dict(CFG)}
    assert r4.source_mismatch(meta, "abc", dict(CFG)) is None
    assert r4.source_mismatch(meta, "def", dict(CFG))
    assert r4.source_mismatch(meta, "abc", {**CFG, "allow_threshold": 0.5})
