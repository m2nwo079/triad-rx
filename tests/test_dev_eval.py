"""Development evaluation helpers (scripts/engine/01_dev_eval.py) on synthetic rankings."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev_eval", ROOT / "scripts/engine/01_dev_eval.py")
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


def toy() -> tuple[pd.DataFrame, np.ndarray]:
    rows = [("hub", "b", "c", 0.9), ("hub", "d", "e", 0.8), ("hub", "f", "g", 0.7), ("x", "y", "z", 0.1)]
    return pd.DataFrame(rows, columns=["a", "b", "c", "s"]), np.array([1.0, 1.0, 0.0, 0.0])


def order_of(df: pd.DataFrame) -> np.ndarray:
    return dev.evaluate.ranking(df["s"].to_numpy(), np.arange(len(df)))


def test_diversified_order_defers_over_cap():
    df, _ = toy()
    order = dev.diversified_order(df, order_of(df), 2)
    # The third hub triad (score 0.7) moves behind the non-hub triad
    assert order.tolist() == [0, 1, 3, 2]
    assert sorted(order.tolist()) == list(range(len(df)))


def test_concentration_counts_hub():
    df, y = toy()
    out = dev.concentration(df.iloc[:3], y[:3], 2)
    assert out["top_terms"][0] == ["hub", 3]
    assert out["max_term_triad_share"] == 1.0
    assert out["distinct_terms"] == 7
    assert out["hits"] == 2
    assert out["hit_distinct_terms"] == 5
    assert out["top_terms_slot_share"] == pytest.approx((3 + 1) / 9)


def test_ranking_metrics_precision_and_lift():
    df, y = toy()
    out = dev.ranking_metrics(df, order_of(df), y, [2, 4], 1)
    assert out["2"]["precision"] == 1.0
    assert out["2"]["lift"] == pytest.approx(1.0 / 0.5)
    assert out["4"]["lift"] == pytest.approx(1.0)


def test_only_development_timepoints_are_allowed():
    import json
    cfg = json.loads((ROOT / "config/study.json").read_text())["engine_reliability"]["dev_eval"]
    assert "T3" not in cfg["timepoints"]


def test_vectorised_precision_matches_original():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.1).astype(float)
    order = rng.permutation(n)
    for _ in range(50):
        w = rng.integers(0, 4, n).astype(float)
        for k in (1, 7, 50, 199, 1000):
            assert dev.precision_at_k(order, y, w, k) == pytest.approx(dev.evaluate.weighted_precision_at_k(order, y, w, k))


def test_power_returns_shares():
    rng = np.random.default_rng(1)
    terms = [f"t{i}" for i in range(12)]
    rows = [tuple(rng.choice(terms, 3, replace=False)) + (float(rng.random()),) for _ in range(150)]
    df = pd.DataFrame(rows, columns=["a", "b", "c", "s"]).drop_duplicates(["a", "b", "c"]).reset_index(drop=True)
    y = (df["s"] > 0.8).to_numpy().astype(float)
    out = dev.power(df, order_of(df), y, [5, 20], {"outer": 5, "inner": 20, "seed": 2}, 0.95)
    for k in ("5", "20"):
        assert set(out[k]) == {"candidate", "term_block", "both"}
        assert all(0.0 <= v <= 1.0 for v in out[k].values())
        assert out[k]["both"] <= min(out[k]["candidate"], out[k]["term_block"])
