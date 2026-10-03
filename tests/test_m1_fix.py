"""Tests for the M1 remediation filter."""
import importlib.util
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location("m1_fix", Path("scripts/vocab/06_apply_m1_fix.py"))
m1_fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_fix)

JUNK = re.compile(r"[<>\"{}\[\]|\\`]")


def term(tid, surfaces):
    return {"id": tid, "type": "M", "status": "active", "surfaces": [{"text": t, "match": m} for t, m in surfaces]}


def test_junk_surface_removes_whole_term_when_nothing_is_left():
    kept, removed, dropped = m1_fix.filter_terms([term("M:test\">", [('test">', "ci")])], JUNK, set())
    assert kept == [] and dropped == ['M:test">']
    assert removed[0]["reason"] == "junk_characters"


def test_excluded_acronym_keeps_the_expansion():
    terms = [term("M:linear discriminant analysis", [("LDA", "cs"), ("linear discriminant analysis", "ci")])]
    kept, removed, dropped = m1_fix.filter_terms(terms, JUNK, {("M:linear discriminant analysis", "LDA")})
    assert [s["text"] for s in kept[0]["surfaces"]] == ["linear discriminant analysis"]
    assert removed == [{"term_id": "M:linear discriminant analysis", "surface": "LDA", "reason": "r3_exclusion"}]
    assert dropped == []


def test_exclusion_is_specific_to_the_term():
    terms = [term("M:other", [("LDA", "cs")])]
    kept, _, _ = m1_fix.filter_terms(terms, JUNK, {("M:linear discriminant analysis", "LDA")})
    assert kept[0]["surfaces"][0]["text"] == "LDA"


def test_normal_punctuation_is_not_junk():
    for text in ["r(2+1)d", "c++", "o'neill", "u-net"]:
        assert not JUNK.search(text)
