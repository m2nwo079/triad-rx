"""Conditional candidates (07) and the paired precision difference (08) on toy data."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cc = load("cond_candidates", "scripts/engine/07_conditional_candidates.py")
rd = load("rediagnose", "scripts/engine/08_rediagnose.py")

TERMS = ["M:a", "M:b", "M:c", "M:d"]


def toy_graph():
    n = len(TERMS)
    counts = np.array([[0, 3, 2, 0], [3, 0, 1, 0], [2, 1, 0, 0], [0, 0, 0, 0]])
    stable = counts >= 2
    m = np.arange(n * n, dtype=float).reshape(n, n) / 10
    return {"terms": TERMS, "n": n, "counts": counts, "stable": stable, "is_task": np.zeros(n, bool),
            "closed": np.array([], dtype=np.int64), "scores": {"s": (m + m.T, np.add)}}


def test_top_candidates_budget_rank_and_kind():
    out = cc.top_candidates(toy_graph(), "stable", "s", 2)
    assert len(out) == 2 and out["gen_rank"].tolist() == [1, 2]
    assert out["gen_score"].is_monotonic_decreasing
    assert (out[["a", "b", "c"]].apply(lambda r: list(r) == sorted(r), axis=1)).all()
    full = cc.top_candidates(toy_graph(), "stable", "s", 10)
    # (a, b, c) has stable pairs a-b and a-c but not b-c, so it is a completion
    abc = full[(full["a"] == "M:a") & (full["b"] == "M:b") & (full["c"] == "M:c")]
    assert abc["kind"].tolist() == ["completion"]


def test_attach_labels_fills_missing_as_false():
    cands = pd.DataFrame({"a": ["M:a", "M:a"], "b": ["M:b", "M:b"], "c": ["M:c", "M:d"]})
    labels = pd.DataFrame({"a": ["M:a"], "b": ["M:b"], "c": ["M:c"], "label_v2": [True]})
    out = cc.attach_labels(cands, labels)
    assert out["label_v2"].tolist() == [True, False] and out["realised"].tolist() == [True, False]


def test_paired_difference_is_zero_for_the_same_order_and_signed_otherwise():
    df = pd.DataFrame({"a": list("pqrstu"), "b": list("vwxyzv"), "c": list("zzzzzz")})
    y = np.array([1, 0, 0, 0, 0, 1], dtype=float)
    ref = np.arange(6)
    ev = {"bootstrap_seed": 1, "bootstrap_n": 50, "ci_level": 0.95}
    out = rd.diff_intervals(df, {"same": ref, "better": np.array([0, 5, 1, 2, 3, 4])}, ref, y, [2], ev)
    assert out["same"]["2"]["candidate"]["lower"] == 0 and out["same"]["2"]["term_block"]["upper"] == 0
    assert out["better"]["2"]["candidate"]["lower"] >= 0
