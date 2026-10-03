"""Apply the M1 remediation filter after R6 and re-match one timepoint's build window.

The filter removes surfaces with markup or broken characters and the acronym surfaces
on the R3 exclusion list (preregistration 3.3 and 3.4). The base vocabulary, the R4
decisions and the R6 result stay unchanged; terms removed by R6 are not brought back.

Usage: python scripts/vocab/06_apply_m1_fix.py --timepoint T1   (T1, T2 or applied)
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.corpus import match_papers, window_papers
from triadrx.study import load_study

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
R4_DECISIONS = Path("vocab/r4_decisions_T1.json")
TIMEPOINTS = Path("vocab/timepoints")
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


def filter_terms(terms: list[dict], junk: re.Pattern, exclusions: set[tuple[str, str]]):
    """Return (kept terms, removed surfaces, removed terms) after the M1 filter."""
    kept, removed_surfaces, removed_terms = [], [], []
    for term in terms:
        surfaces = []
        for surface in term["surfaces"]:
            if junk.search(surface["text"]):
                removed_surfaces.append({"term_id": term["id"], "surface": surface["text"], "reason": "junk_characters"})
            elif (term["id"], surface["text"]) in exclusions:
                removed_surfaces.append({"term_id": term["id"], "surface": surface["text"], "reason": "r3_exclusion"})
            else:
                surfaces.append(surface)
        if surfaces:
            kept.append({**term, "surfaces": surfaces})
        else:
            removed_terms.append(term["id"])
    return kept, removed_surfaces, removed_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    args = parser.parse_args()
    tp = args.timepoint

    study = load_study()
    cfg = study["vocab"]["m1_fix"]
    junk = re.compile(cfg["junk_pattern"])
    exclusions = {(e["term_id"], e["surface"]) for e in cfg["r3_exclude"]}

    after_r6_path = TIMEPOINTS / f"{tp}_after_r6.json"
    after_r6 = json.loads(after_r6_path.read_text())
    r4 = json.loads(R4_DECISIONS.read_text())
    if after_r6["meta"]["vocab_sha256"] != sha256(VOCAB) or after_r6["meta"]["r4_decisions_sha256"] != sha256(R4_DECISIONS):
        sys.exit("vocab_base.json or r4_decisions_T1.json changed since R6 was applied")
    r6_kept = {k["term_id"] for k in after_r6["kept"]}
    allowed = {d["term_id"] for d in r4["decisions"] if d["decision"] == "allow"}

    vocab = json.loads(VOCAB.read_text())
    # Same term set as R6 produced; only surfaces are filtered here
    terms = [
        {**t, "status": "active"}
        for t in vocab["terms"]
        if (t["status"] == "active" or t["id"] in allowed) and t["id"] in r6_kept
    ]
    kept, removed_surfaces, removed_terms = filter_terms(terms, junk, exclusions)
    term_type = {t["id"]: t["type"] for t in kept}

    window, start, end = window_papers(PAPERS, tp, study)
    matches = match_papers(window, kept)
    term_docs = matches.drop_duplicates(["paper_id", "term_id"]).groupby("term_id").size()

    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(MATCH_DIR / f"matches_{tp}_after_m1fix.parquet")

    meta = {
        "step": "apply_m1_fix",
        "timepoint": tp,
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "window": [start.isoformat(), end.isoformat()],
        "m1_fix": cfg,
        "papers_sha256": sha256(PAPERS),
        "vocab_sha256": sha256(VOCAB),
        "r4_decisions_sha256": sha256(R4_DECISIONS),
        "after_r6_sha256": sha256(after_r6_path),
    }
    stats = {
        "window_papers": int(len(window)),
        "terms_in": len(terms),
        "surfaces_removed": len(removed_surfaces),
        "terms_removed": len(removed_terms),
        "terms_matched": int(len(term_docs)),
    }
    detail = {
        "meta": meta,
        "stats": stats,
        "removed_surfaces": removed_surfaces,
        "removed_terms": removed_terms,
        "kept": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in term_docs.sort_index().items()],
    }
    (TIMEPOINTS / f"{tp}_after_m1fix.json").write_text(json.dumps(detail, indent=1))
    (RESULTS / f"vocab_m1fix_{tp}.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
    print(f"Saved {TIMEPOINTS / f'{tp}_after_m1fix.json'}, {RESULTS / f'vocab_m1fix_{tp}.json'} and matches")


if __name__ == "__main__":
    main()
