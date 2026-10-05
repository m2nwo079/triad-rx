"""Leakage controls on the derived data (design 10.7).

The derived files are not committed, so each test is skipped when its inputs are absent.
- every paper in a build or label hypergraph is dated inside that window
- every matched term belongs to the timepoint's final vocabulary (built from the build window)
- every candidate term appears in the build window
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triadrx.study import build_window, data_cutoff, label_window, load_study

GRAPH = ROOT / "data/derived/hypergraph/main"
FEATURES = ROOT / "data/derived/features/main"
VOCAB = ROOT / "vocab/timepoints"
CASES = [("T1", "build"), ("T1", "label"), ("T2", "build"), ("T2", "label"), ("applied", "build")]


def read_or_skip(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present")
    return pd.read_parquet(path)


def window(timepoint: str, kind: str):
    study = load_study()
    if kind == "label":
        return label_window(timepoint, study)
    return build_window(timepoint, study, data_cutoff() if timepoint == "applied" else None)


def vocabulary(timepoint: str) -> set[str]:
    return {t["term_id"] for t in json.loads((VOCAB / f"{timepoint}_final.json").read_text())["kept"]}


@pytest.mark.parametrize("timepoint,kind", CASES)
def test_papers_inside_window(timepoint, kind):
    graph = read_or_skip(GRAPH / f"{timepoint}_{kind}.parquet")
    start, end = window(timepoint, kind)
    dates = pd.to_datetime(graph["first_date"], utc=True)
    assert dates.min() >= start and dates.max() <= end


@pytest.mark.parametrize("timepoint,kind", CASES)
def test_terms_from_final_vocabulary(timepoint, kind):
    graph = read_or_skip(GRAPH / f"{timepoint}_{kind}.parquet")
    terms = {t for ts in graph["terms"] for t in ts}
    assert terms <= vocabulary(timepoint)


@pytest.mark.parametrize("timepoint", ["T1", "T2", "applied"])
def test_candidate_terms_seen_in_build_window(timepoint):
    features = read_or_skip(FEATURES / f"{timepoint}.parquet")
    graph = read_or_skip(GRAPH / f"{timepoint}_build.parquet")
    seen = {t for ts in graph["terms"] for t in ts}
    assert set(features["a"]) | set(features["b"]) | set(features["c"]) <= seen
