"""Conditional completion generator (scripts/engine/06_conditional_recall.py) on a toy hypergraph."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cond_recall", ROOT / "scripts/engine/06_conditional_recall.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
spec = importlib.util.spec_from_file_location("features", ROOT / "scripts/graph/05_build_features.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

TERMS = ["M:a", "M:b", "M:c", "T:x", "T:y", "T:z"]
INDEX = {t: i for i, t in enumerate(TERMS)}
BUILD = pd.DataFrame({"terms": [["M:a", "M:b"], ["M:a", "M:b", "T:x"], ["M:c", "T:y"], ["T:x", "T:y", "T:z"], ["M:a"]]})


def test_npmi_matrix_matches_feature_npmi():
    counts, docs = cr.cooccurrence(BUILD, INDEX)
    m = cr.npmi_matrix(counts, docs, len(BUILD))
    for i in range(len(TERMS)):
        for j in range(len(TERMS)):
            if i != j:
                assert math.isclose(m[i, j], features.npmi(int(counts[i, j]), int(docs[i]), int(docs[j]), len(BUILD)))


def test_adamic_adar_skips_degree_one():
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    aa = cr.adamic_adar_matrix(adj)
    assert math.isclose(aa[1, 2], 1 / math.log(2))
    assert aa[0, 1] == 0


def test_completions_drop_closed_task_only_and_keep_max():
    counts, docs = cr.cooccurrence(BUILD, INDEX)
    n = len(TERMS)
    is_task = np.array([t.startswith("T:") for t in TERMS])
    closed = np.sort(cr.triad_codes(np.array([0, 3]), np.array([1, 4]), np.array([3, 5]), n))
    seeds = np.array([[0, 1], [0, 3], [3, 4]])
    mats = {"min_count": (counts, np.minimum)}
    codes, scores = cr.completions(seeds, mats, is_task, closed, n)
    got = {tuple(sorted(np.unravel_index(c, (n, n, n)))) for c in codes}
    assert (0, 1, 3) not in got and (3, 4, 5) not in got
    assert (0, 1, 2) in got and (0, 3, 4) in got
    assert len(got) == len(codes)
    # Every kept triad's score is the maximum over the seeds it contains (brute force)
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    codes, scores = cr.completions(seeds, {"s": (m, np.add)}, is_task, closed, n)
    for c, got_score in zip(codes, scores["s"]):
        tri = set(np.unravel_index(c, (n, n, n)))
        expected = max(m[u, w] + m[v, w] for u, v in seeds.tolist() if {u, v} <= tri for w in tri - {u, v})
        assert math.isclose(got_score, expected)


def test_recall_at_budgets_and_cap():
    codes = np.array([10, 20, 30, 40])
    score = np.array([0.1, 0.9, 0.9, 0.5])
    out = cr.recall_at(codes, score, np.array([30, 10]), 4, [1, 2, 10])
    assert [r["hits"] for r in out] == [0, 1, 2]
    assert out[1]["recall"] == 1 / 4 and out[1]["precision"] == 1 / 2
    assert out[2]["capped"] and out[2]["precision"] == 2 / 4


def test_seed_pairs_rules():
    counts = np.array([[0, 5, 1], [5, 0, 0], [1, 0, 0]])
    stable = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=bool)
    assert cr.seed_pairs("min_5", counts, stable).tolist() == [[0, 1]]
    assert cr.seed_pairs("stable", counts, stable).tolist() == [[0, 2]]


def test_label_codes_match_completion_codes_and_skip_unknown_terms():
    rows = pd.DataFrame({"a": ["M:a", "M:a"], "b": ["M:b", "M:new"], "c": ["T:y", "T:y"]})
    codes = cr.label_codes(rows, INDEX)
    assert codes.tolist() == cr.triad_codes(np.array([0]), np.array([1]), np.array([4]), len(TERMS)).tolist()
