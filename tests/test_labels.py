"""Tests for the triadic null-model labelling."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("labels", Path("scripts/graph/04_build_labels.py"))
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def brute_upper_tail(observed, probs):
    total = 0.0
    for outcome in itertools.product([0, 1], repeat=len(probs)):
        if sum(outcome) >= observed:
            p = 1.0
            for x, q in zip(outcome, probs):
                p *= q if x else 1 - q
            total += p
    return total


def test_upper_tail_matches_brute_force():
    groups = [(3, 0.1), (2, 0.35), (1, 0.8)]
    probs = [0.1] * 3 + [0.35] * 2 + [0.8]
    for observed in range(0, 7):
        assert abs(labels.upper_tail(observed, groups) - brute_upper_tail(observed, probs)) < 1e-12


def test_upper_tail_edge_probabilities():
    assert labels.upper_tail(1, [(5, 0.0)]) == 0.0
    assert abs(labels.upper_tail(5, [(5, 1.0)]) - 1.0) < 1e-12


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    q = labels.benjamini_hochberg(p)
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_lock():
    assert not labels.preregistration_frozen("고정 커밋: 미정")
    assert labels.preregistration_frozen("고정 커밋: 1a2b3c4")
