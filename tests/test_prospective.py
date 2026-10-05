"""Prospective check (preregistration 8): windows, lock, baseline order and the verdict on synthetic data."""
import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prospective_check", ROOT / "scripts/model/07_prospective_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
register = check.register

UTC = datetime.timezone.utc
CUTOFF = datetime.datetime(2026, 9, 24, 17, 59, 54, tzinfo=UTC)
CFG = {"buffer_years": 1, "label_years": 2, "descriptive_years": [1, 2]}
EV = {"bootstrap_n": 50, "bootstrap_seed": 1, "ci_level": 0.95}


def test_verdict_and_descriptive_windows():
    start, end, verdict = check.check_window(CUTOFF, 3, CFG)
    assert (start.year, end.year, verdict) == (2027, 2029, True)
    start, end, verdict = check.check_window(CUTOFF, 1, CFG)
    assert (start, end.year, verdict) == (CUTOFF, 2027, False)
    with pytest.raises(ValueError):
        check.check_window(CUTOFF, 4, CFG)


def test_add_years_on_leap_day():
    leap = datetime.datetime(2028, 2, 29, tzinfo=UTC)
    assert check.add_years(leap, 1) == datetime.datetime(2029, 2, 28, tzinfo=UTC)


def test_lock():
    end = check.add_years(CUTOFF, 3)
    after = end + datetime.timedelta(days=1)
    assert check.lock_reason(end - datetime.timedelta(seconds=1), after, end)
    assert check.lock_reason(after, end - datetime.timedelta(days=1), end)
    assert check.lock_reason(after, end, end) is None


def toy_population() -> pd.DataFrame:
    rows = [("a", "b", "c", 1.0, 0.5), ("a", "b", "d", 2.0, 0.5), ("a", "c", "d", 3.0, 0.9),
            ("b", "c", "d", 4.0, 0.9), ("a", "b", "e", None, 9.0), ("c", "d", "e", 5.0, 0.1)]
    df = pd.DataFrame(rows, columns=["a", "b", "c", "rank_prescription", "P_aa_min"])
    return register.population(df)


def test_population_and_baseline_ties():
    pop = toy_population()
    assert len(pop) == 5
    baseline = register.triads(pop, register.baseline_order(pop, "P_aa_min"), 3)
    # Ties at 0.9 and 0.5 are broken by term ids; the redundant row (score 9.0) is excluded
    assert baseline == [["a", "c", "d"], ["b", "c", "d"], ["a", "b", "c"]]
    assert register.triads(pop, register.prescription_order(pop), 2) == [["a", "b", "c"], ["a", "b", "d"]]


def test_judge_point_values():
    pop = toy_population()
    y = np.array([float(t in {("a", "b", "c"), ("a", "b", "d")}) for t in pop[["a", "b", "c"]].itertuples(index=False, name=None)])
    out = check.judge(pop, y, 2, "P_aa_min", EV)
    assert out["point"]["precision_prescription"] == 1.0
    assert out["point"]["precision_baseline"] == 0.0
    assert out["point"]["lift"] == pytest.approx(1.0 / (2 / 5))
    assert set(out["verdict"]) == {"P1", "P2"}


def test_unseen_triad_is_not_realised():
    edges = pd.DataFrame({"terms": [["a", "b"], ["a", "c"], ["b", "c"], ["a", "d"]]})
    cands = pd.DataFrame([("a", "b", "c")], columns=["a", "b", "c"])
    assert check.null_model_labels(edges, cands, 0.05).tolist() == [0]


def test_dvf_by_score():
    cands = [{"triad": ["a", "b", "c"], "axes": {"D": {"score": 4.0}, "F": {"score": 2.0}, "V": {"score": None}}},
             {"triad": ["a", "b", "d"], "axes": {"D": {"score": 4.0}, "F": {"score": 3.0}, "V": {"score": None}}}]
    out = check.dvf_by_score(cands, {("a", "b", "c"): 1, ("a", "b", "d"): 0})
    assert out["D"]["4"] == {"n": 2, "realised": 1}
    assert out["V"]["missing"] == {"n": 2, "realised": 1}
