"""Draw the M1 matching-precision sample on T1 after R6 and write the review sheet.

Usage: python scripts/vocab/05_m1_sample.py --attempt 0
Attempt 0 is the first measurement; each re-measurement uses the next attempt number,
which gives a new seed (preregistration 5.1).
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study
from triadrx.text import Matcher

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
TIMEPOINTS = Path("vocab/timepoints")
MATCH_DIR = Path("data/derived")
REVIEW_DIR = Path("data/review")
RESULTS = Path("results")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


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


def stratum(term_type: str, mode: str) -> str:
    """M1 strata: case-sensitive acronyms, otherwise method or task surfaces."""
    if mode == "cs":
        return "acronym_cs"
    return "method_ci" if term_type == "M" else "task_ci"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=0)
    # Re-measurements sample from the vocabulary after the M1 remediation filter
    parser.add_argument("--stage", choices=["after_r6", "after_m1fix"], default="after_r6")
    # Stratum-level re-measurement: only strata with a remediation rule (preregistration 5.1)
    parser.add_argument("--strata", nargs="+", default=None)
    args = parser.parse_args()
    stage_file = TIMEPOINTS / f"T1_{args.stage}.json"
    matches_file = MATCH_DIR / f"matches_T1_{args.stage}.parquet"

    study = load_study()
    cfg = study["quality"]["m1"]
    if not 0 <= args.attempt <= cfg["max_remeasure"]:
        sys.exit(f"attempt must be between 0 and {cfg['max_remeasure']}")
    seed = cfg["seed"] + args.attempt
    n = cfg["n_per_stratum"]

    after = json.loads(stage_file.read_text())
    kept = {k["term_id"]: k["type"] for k in after["kept"]}
    vocab = json.loads(VOCAB.read_text())
    keys = {t["id"]: t["key"] for t in vocab["terms"]}
    matches = pd.read_parquet(matches_file)
    matches = matches[matches["term_id"].isin(kept)]
    matches["stratum"] = [stratum(kept[t], m) for t, m in zip(matches["term_id"], matches["mode"])]

    # Uniform over matches, at most one match per paper within a stratum
    rng = random.Random(seed)
    picked = []
    strata = args.strata or cfg["strata"]
    unknown = set(strata) - set(cfg["strata"])
    if unknown:
        sys.exit(f"unknown strata {sorted(unknown)}")
    for name in strata:
        pool = sorted(matches[matches["stratum"] == name][["paper_id", "term_id", "mode"]].itertuples(index=False))
        rng.shuffle(pool)
        used = set()
        for pid, tid, mode in pool:
            if pid in used:
                continue
            used.add(pid)
            picked.append((name, pid, tid, mode))
            if len(used) == n:
                break

    # Re-match only the sampled papers to find a sentence that shows the match
    terms = [{**t, "status": "active"} for t in vocab["terms"] if t["id"] in kept]
    matcher = Matcher(terms)
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"])
    papers = papers[papers["id"].isin({p for _, p, _, _ in picked})].set_index("id")
    rows = []
    for name, pid, tid, mode in picked:
        text = re.sub(r"\s+", " ", f"{papers.at[pid, 'title']}. {papers.at[pid, 'abstract']}").strip()
        sentence, surface = "", ""
        for candidate in SENTENCE.split(text):
            hits = [s for s in matcher.spans(candidate) if s["id"] == tid and s["mode"] == mode]
            if hits:
                sentence, surface = candidate, hits[0]["tokens"]
                break
        rows.append(
            {
                "stratum": name,
                "term_id": tid,
                "key": keys[tid],
                "matched_tokens": surface,
                "paper_id": pid,
                "sentence": sentence,
                "correct": "",
                "note": "",
            }
        )

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sheet = REVIEW_DIR / f"m1_review_T1_attempt{args.attempt}.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: sum(1 for r in rows if r["stratum"] == name) for name in strata}
    pool_sizes = {name: int((matches["stratum"] == name).sum()) for name in strata}
    result = {
        "meta": {
            "step": "m1_sample_T1",
            "attempt": args.attempt,
            "seed": seed,
            "created": datetime.date.today().isoformat(),
            "git_commit": git_commit(),
            "quality_m1": cfg,
            "stage": args.stage,
            "strata": strata,
            "stage_sha256": sha256(stage_file),
            "matches_sha256": sha256(matches_file),
            "sampling": "per stratum, uniform over matches of the stage vocabulary with at most one match per paper",
        },
        "sampled": counts,
        "stratum_match_counts": pool_sizes,
        "missing_sentence": sum(1 for r in rows if not r["sentence"]),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"m1_sample_T1_attempt{args.attempt}.json").write_text(json.dumps(result, indent=2))
    print("Saved", sheet, "and", RESULTS / f"m1_sample_T1_attempt{args.attempt}.json")


if __name__ == "__main__":
    main()