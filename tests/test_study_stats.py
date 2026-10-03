"""Tests for timepoint windows and statistics helpers."""
import datetime

import numpy as np
import pytest

from triadrx.stats import quality_status, r5_min_docs, wilson_interval
from triadrx.study import build_window

STUDY = {
    "timepoints": {"T1": {"build": [2014, 2018]}},
    "applied": {"build_years": 5},
}
UTC = datetime.timezone.utc


def test_t1_window_covers_whole_calendar_years():
    start, end = build_window("T1", STUDY)
    assert start == datetime.datetime(2014, 1, 1, tzinfo=UTC)
    assert end.date() == datetime.date(2018, 12, 31)


def test_applied_window_ends_at_cutoff():
    cutoff = datetime.datetime(2026, 9, 24, 17, 59, 54, tzinfo=UTC)
    start, end = build_window("applied", STUDY, cutoff)
    assert end == cutoff
    assert start == datetime.datetime(2021, 9, 24, 17, 59, 54, tzinfo=UTC)


def test_applied_window_requires_cutoff():
    with pytest.raises(ValueError):
        build_window("applied", STUDY)


def test_wilson_interval_known_value():
    low, high = wilson_interval(90, 100)
    assert round(low, 3) == 0.826
    assert round(high, 3) == 0.945


def test_quality_status_bands():
    rule = {"pass": 0.8, "conditional": 0.75}
    assert quality_status(0.8, rule) == "pass"
    assert quality_status(0.75, rule) == "conditional"
    assert quality_status(0.7499, rule) == "fail"


def test_r5_min_docs():
    assert r5_min_docs(0.95, 1.0) == 3
    assert r5_min_docs(0.95, 0.75) == 4
    with pytest.raises(ValueError):
        r5_min_docs(0.95, 0.0)


def test_r6_threshold():
    import importlib.util
    from pathlib import Path

    path = Path("scripts/vocab/04_apply_r6.py")
    spec = importlib.util.spec_from_file_location("apply_r6", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    degrees = np.array([0, 1, 2, 3, 4])
    total, w_p_star, max_w_c = module.r6_threshold(degrees, 1.0)
    assert total == 10
    assert w_p_star == 4
    assert max_w_c == 2.5
