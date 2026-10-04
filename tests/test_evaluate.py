"""Tests for validation metrics."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("ev", Path("scripts/model/02_evaluate.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_ranking_breaks_ties_by_id():
    order = ev.ranking(np.array([0.5, 0.9, 0.5]), np.array([0, 1, 2]))
    assert list(order) == [1, 0, 2]


def test_weighted_precision_counts_duplicates_and_fractions():
    order = np.array([0, 1, 2])
    y = np.array([1.0, 0.0, 1.0])
    assert ev.weighted_precision_at_k(order, y, np.array([1.0, 1.0, 1.0]), 2) == 0.5
    # Candidate 0 drawn twice fills the top 2 with positives
    assert ev.weighted_precision_at_k(order, y, np.array([2.0, 1.0, 1.0]), 2) == 1.0
    # Zero-weight candidates are skipped
    assert ev.weighted_precision_at_k(order, y, np.array([0.0, 0.0, 1.0]), 1) == 1.0


def test_lift_equals_one_for_random_like_ranking():
    y = np.array([1.0, 0.0] * 50)
    order = np.arange(100)
    m = ev.metrics(order, order, np.linspace(1, 0, 100), np.linspace(1, 0, 100), y, np.ones(100), 100)
    assert abs(m["lift"] - 1.0) < 1e-12 and abs(m["auprc_diff"]) < 1e-12


def test_term_block_weights_are_products():
    rng = np.random.default_rng(0)
    triads = np.array([[0, 1, 2]])
    w = ev.term_block_weights(rng, triads, np.array(["a", "b", "c"]))
    assert w.shape == (1,) and w[0] in {0, 1, 2, 4, 8, 3, 6, 9, 12, 18, 27}
