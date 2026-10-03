"""Build-window paper loading and matching shared by vocabulary steps."""
import datetime

import pandas as pd

from triadrx.study import build_window, data_cutoff
from triadrx.text import Matcher

PAPERS_COLUMNS = ["id", "title", "abstract", "first_date"]


def window_papers(papers_path, timepoint: str, study: dict) -> tuple[pd.DataFrame, datetime.datetime, datetime.datetime]:
    """Papers first submitted inside the timepoint's build window, sorted by id."""
    cutoff = data_cutoff() if timepoint == "applied" else None
    start, end = build_window(timepoint, study, cutoff)
    papers = pd.read_parquet(papers_path, columns=PAPERS_COLUMNS)
    window = papers[(papers["first_date"] >= start) & (papers["first_date"] <= end)].sort_values("id")
    return window, start, end


def match_papers(window: pd.DataFrame, terms: list[dict]) -> pd.DataFrame:
    """One row per (paper, term, mode) found in title and abstract."""
    matcher = Matcher(terms)
    rows = []
    for pid, title, abstract in window[["id", "title", "abstract"]].itertuples(index=False):
        spans = matcher.spans(f"{title}. {abstract}")
        for tid, mode in sorted({(s["id"], s["mode"]) for s in spans}):
            rows.append((pid, tid, mode))
    return pd.DataFrame(rows, columns=["paper_id", "term_id", "mode"])
