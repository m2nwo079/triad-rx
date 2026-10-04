"""Tests for triad candidate rules."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("cands", Path("scripts/graph/03_build_candidates.py"))
cands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cands)


def test_pair_threshold_uses_both_precisions():
    assert cands.pair_threshold(1.0, 1.0, 0.95) == 3
    assert cands.pair_threshold(0.75, 0.75, 0.95) == 6


def test_all_task_triads_are_flagged():
    assert cands.all_tasks(("T:a", "T:b", "T:c"))
    assert not cands.all_tasks(("M:a", "T:b", "T:c"))


def test_open_triangle_and_two_hop():
    # a-b, b-c, a-c each co-occur in separate papers (open triangle); d links only to a
    papers = [["M:a", "M:b"]] * 3 + [["M:b", "T:c"]] * 3 + [["M:a", "T:c"]] * 3 + [["M:a", "M:d"]] * 3
    edges = pd.DataFrame({"terms": papers})
    pi = {t: 1.0 for t in ["M:a", "M:b", "T:c", "M:d"]}
    rows, stats = cands.build(edges, pi, 0.95, 1.0)
    kinds = {(r["a"], r["b"], r["c"]): r["kind"] for r in rows}
    assert kinds[("M:a", "M:b", "T:c")] == "open_triangle"
    assert stats["open_triangles"] == 1
    assert stats["two_hop_selected"] == 1
    assert sum(k == "two_hop" for k in kinds.values()) == 1


def test_closed_triad_is_not_a_candidate():
    papers = [["M:a", "M:b", "T:c"]] * 3
    rows, stats = cands.build(pd.DataFrame({"terms": papers}), {"M:a": 1.0, "M:b": 1.0, "T:c": 1.0}, 0.95, 1.0)
    assert rows == [] and stats["open_triangles_before_filter"] == 0
