"""Tests for triad candidate features."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("feats", Path("scripts/graph/05_build_features.py"))
feats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feats)


def test_npmi_bounds():
    assert feats.npmi(0, 5, 5, 100) == -1.0
    assert abs(feats.npmi(5, 5, 5, 100) - 1.0) < 1e-12


def test_gmean_handles_zero():
    assert feats.gmean([0, 0, 0]) == 0.0
    assert abs(feats.gmean([3, 3, 3]) - 3.0) < 1e-12


def test_compute_on_toy_hypergraph():
    utc = pd.Timestamp("2018-06-01", tz="UTC")
    old = pd.Timestamp("2015-06-01", tz="UTC")
    terms = [["M:a", "M:b"]] * 3 + [["M:b", "T:c"]] * 3 + [["M:a", "T:c"]] * 3 + [["M:a", "M:b", "M:d"]]
    dates = [utc] * 5 + [old] * 5
    edges = pd.DataFrame({"terms": terms, "first_date": dates})
    cands = pd.DataFrame([{"a": "M:a", "b": "M:b", "c": "T:c", "kind": "open_triangle"}])
    pi = {t: 1.0 for t in ["M:a", "M:b", "T:c", "M:d"]}
    vectors = {t: np.eye(4)[i] for i, t in enumerate(["M:a", "M:b", "T:c", "M:d"])}
    out = feats.compute(edges, cands, pi, 0.95, pd.Timestamp("2017-01-01", tz="UTC"), vectors).iloc[0]
    assert out["H_open_triangle"] == 1
    assert out["H_pairedges_min"] == 3 and out["H_pairedges_mean"] == 10 / 3
    assert abs(out["T_volume"] - 1.0) < 1e-12 and out["T_cos_mean"] == 0.0
    # Growth per pair: a-b (3 recent of 4), b-c (2 of 3), a-c (0 of 3)
    expected = [(3 / 2 + 1) / (1 / 3 + 1), (2 / 2 + 1) / (1 / 3 + 1), (0 / 2 + 1) / (3 / 3 + 1)]
    assert math.isclose(out["R_growth_min"], min(expected))
    assert math.isclose(out["R_growth_mean"], sum(expected) / 3)
