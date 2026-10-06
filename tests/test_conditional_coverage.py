"""Conditional completion coverage (scripts/engine/05_conditional_coverage.py) on a toy hypergraph."""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coverage", ROOT / "scripts/engine/05_conditional_coverage.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def test_pair_counts_ignore_duplicates_within_a_paper():
    build = pd.DataFrame({"terms": [["a", "b", "b"], ["b", "a"], ["a", "c"]]})
    counts = cc.pair_counts(build)
    assert counts[("a", "b")] == 2 and counts[("a", "c")] == 1 and counts[("b", "c")] == 0


def test_coverage_counts_qualifying_pairs_and_recall():
    counts = cc.pair_counts(pd.DataFrame({"terms": [["a", "b"], ["a", "b"], ["c", "d"]]}))
    triads = [("a", "b", "x"), ("c", "d", "y"), ("e", "f", "g")]
    out = cc.coverage(triads, lambda p: counts[p] >= 2, {("a", "b", "x")})
    assert out["by_qualifying_pairs"] == {"0": 2, "1": 1, "2": 0, "3": 0}
    assert out["coverage"] == 1 / 3
    assert out["candidate_recall"] == 1 / 3
    loose = cc.coverage(triads, lambda p: counts[p] >= 1, set())
    assert loose["coverage"] == 2 / 3


def test_coverage_of_empty_set_is_nan():
    out = cc.coverage([], lambda p: True, set())
    assert out["triads"] == 0 and out["coverage"] != out["coverage"]
