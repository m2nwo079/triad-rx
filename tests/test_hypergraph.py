"""Tests for hypergraph helpers."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("hypergraph", Path("scripts/graph/01_build_hypergraph.py"))
hypergraph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hypergraph)


def test_preregistration_lock():
    assert not hypergraph.preregistration_frozen("고정 커밋: 미정\n")
    assert not hypergraph.preregistration_frozen("no freeze line\n")
    assert hypergraph.preregistration_frozen("intro\n고정 커밋: abc1234\n")


def test_drop_modes_removes_acronyms_and_empty_terms():
    terms = [
        {"id": "M:a", "surfaces": [{"text": "LDA", "match": "cs"}, {"text": "linear discriminant analysis", "match": "ci"}]},
        {"id": "M:b", "surfaces": [{"text": "BERT", "match": "cs"}]},
    ]
    out = hypergraph.drop_modes(terms, {"cs"})
    assert [t["id"] for t in out] == ["M:a"]
    assert [s["text"] for s in out[0]["surfaces"]] == ["linear discriminant analysis"]
