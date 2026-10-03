"""Apply R4 decisions and rule R6 to one timepoint's build window and save its matches.

Usage: python scripts/vocab/04_apply_r6.py --timepoint T1   (T1, T2 or applied)
"""
import argparse
import collections
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import build_window, data_cutoff, load_study
from triadrx.text import Matcher

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
R4_DECISIONS = Path("vocab/r4_decisions_T1.json")
OUT_DIR = Path("vocab/timepoints")
MATCH_DIR = Path("data/derived")
RESULTS = Path("results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def r6_threshold(paper_degrees: np.ndarray, quantile: float) -> tuple[int, float, float]:
    """Return (N, w_p*, max allowed w_c) for rule R6.

    N is the total number of paper-term links. w_p* is the quantile of the number of
    matched terms per paper, taken over papers with at least one match. A term whose
    paper count exceeds N / w_p* would push P_pc = w_p * w_c / N above 1 for papers at
    or above that quantile, so it violates the null model and is removed.
    """
    linked = paper_degrees[paper_degrees > 0]
    total_links = int(linked.sum())
    w_p_star = float(np.quantile(linked, quantile, method="higher"))
    return total_links, w_p_star, total_links / w_p_star


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    args = parser.parse_args()
    tp = args.timepoint

    study = load_study()
    if not R4_DECISIONS.exists():
        sys.exit(f"{R4_DECISIONS} is missing: finish the R4 review and run 03_r4_decisions.py first")
    r4 = json.loads(R4_DECISIONS.read_text())
    if r4["meta"]["vocab_sha256"] != sha256(VOCAB):
        sys.exit("vocab/vocab_base.json changed since the R4 decisions were made")
    allowed = {d["term_id"] for d in r4["decisions"] if d["decision"] == "allow"}

    vocab = json.loads(VOCAB.read_text())
    # Step 1 of preregistration 3.4: active terms plus allowed R4 candidates
    terms = []
    for term in vocab["terms"]:
        if term["status"] == "active" or term["id"] in allowed:
            terms.append({**term, "status": "active"})
    matcher = Matcher(terms)
    term_type = {t["id"]: t["type"] for t in terms}

    cutoff = data_cutoff() if tp == "applied" else None
    start, end = build_window(tp, study, cutoff)
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"])
    # Decision-time rule: only papers first submitted inside the build window
    window = papers[(papers["first_date"] >= start) & (papers["first_date"] <= end)].sort_values("id")

    rows = []
    for pid, title, abstract in window[["id", "title", "abstract"]].itertuples(index=False):
        spans = matcher.spans(f"{title}. {abstract}")
        # Keep one entry per (term, mode) so that M1 can sample by stratum later
        seen = {(s["id"], s["mode"]) for s in spans}
        for tid, mode in sorted(seen):
            rows.append((pid, tid, mode))
    matches = pd.DataFrame(rows, columns=["paper_id", "term_id", "mode"])
    links = matches.drop_duplicates(["paper_id", "term_id"])

    degrees = links.groupby("paper_id").size().reindex(window["id"], fill_value=0).to_numpy()
    term_docs = links.groupby("term_id").size()
    quantile = study["vocab"]["r6_rule"]["paper_degree_quantile"]
    total_links, w_p_star, max_w_c = r6_threshold(degrees, quantile)
    removed = term_docs[term_docs > max_w_c]
    kept = term_docs[term_docs <= max_w_c]

    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    # Matches stay in data/ because they are derived from the paper corpus
    matches[matches["term_id"].isin(kept.index)].to_parquet(MATCH_DIR / f"matches_{tp}_after_r6.parquet")

    meta = {
        "step": "apply_r4_r6",
        "timepoint": tp,
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "window": [start.isoformat(), end.isoformat()],
        "r6_rule": study["vocab"]["r6_rule"],
        "papers_sha256": sha256(PAPERS),
        "vocab_sha256": sha256(VOCAB),
        "r4_decisions_sha256": sha256(R4_DECISIONS),
    }
    stats = {
        "window_papers": int(len(window)),
        "papers_with_matches": int((degrees > 0).sum()),
        "total_links": total_links,
        "paper_degree_quantile_value": w_p_star,
        "max_term_docs": max_w_c,
        "terms_matched": int(len(term_docs)),
        "terms_removed_r6": int(len(removed)),
        "terms_kept": int(len(kept)),
        "r4_allowed": len(allowed),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = {
        "meta": meta,
        "stats": stats,
        "removed_r6": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in removed.sort_values(ascending=False).items()],
        "kept": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in kept.sort_index().items()],
    }
    (OUT_DIR / f"{tp}_after_r6.json").write_text(json.dumps(detail, indent=1))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"vocab_r6_{tp}.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
    print(f"Saved {OUT_DIR / f'{tp}_after_r6.json'}, {RESULTS / f'vocab_r6_{tp}.json'} and matches")


if __name__ == "__main__":
    main()
