"""Score the Q1 review and its agreement with the automatic checks (design 9.1, 9.2).

Usage: python scripts/quality/03_q1_score.py

Checks that the reviewed file matches its read-only original except `correct`
and `note`, then reports the share of supported sentences with a Wilson 95%
interval (no pass threshold), and a 2x2 table of human judgment against
whether the automatic checks (results/briefing_checks.json) flagged the sentence.

Output: results/q1.json
"""
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx import quality as qu

REVIEW = Path("data/review/q1_review.csv")
ORIGINAL = Path("data/review/q1_review_original.csv")
CHECKS = Path("results/briefing_checks.json")
RESULTS = Path("results")
EDITABLE = {"correct", "note"}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def main() -> None:
    head_r, rows = read(REVIEW)
    head_o, orig = read(ORIGINAL)
    if head_r != head_o or len(rows) != len(orig):
        sys.exit("header or row count differs from the original")
    for i, (r, o) in enumerate(zip(rows, orig), start=2):
        changed = [k for k in head_o if k not in EDITABLE and r[k] != o[k]]
        if changed:
            sys.exit(f"line {i}: columns other than correct/note changed: {changed}")
        if r["correct"].strip() not in ("0", "1"):
            sys.exit(f"line {i}: correct must be 0 or 1, got {r['correct']!r}")

    flagged = {(f["rank"], f["rep"], f["field"], f["sentence_index"])
               for f in json.loads(CHECKS.read_text())["flags"] if f["sentence_index"] is not None}
    k = sum(int(r["correct"]) for r in rows)
    n = len(rows)
    low, high = qu.wilson(k, n)
    table = {"human_1_auto_clean": 0, "human_1_auto_flag": 0, "human_0_auto_clean": 0, "human_0_auto_flag": 0}
    for r in rows:
        key = (int(r["rank"]), int(r["rep"]), r["field"], int(r["sentence_index"]))
        auto = "flag" if key in flagged else "clean"
        table[f"human_{r['correct'].strip()}_auto_{auto}"] += 1
    unsure = sum(1 for r in rows if "애매" in r["note"])
    by_field = {}
    for r in rows:
        field = "concept_frame" if r["field"].startswith("concept_frame") else r["field"].split("[")[0].split(".")[0]
        entry = by_field.setdefault(field, {"n": 0, "supported": 0})
        entry["n"] += 1
        entry["supported"] += int(r["correct"])
    result = {
        "meta": {"step": "q1", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "review_sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
                 "original_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                 "checks_sha256": hashlib.sha256(CHECKS.read_bytes()).hexdigest()},
        "supported": k, "n": n, "rate": k / n, "wilson_95": [low, high], "unsure_notes": unsure,
        "by_field": by_field, "human_vs_auto": table,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "q1.json").write_text(json.dumps(result, indent=2))
    print(f"Q1 {k}/{n} = {k / n:.3f}, Wilson 95% [{low:.3f}, {high:.3f}]")
    print("human vs automatic checks:", table)
    print("Saved", RESULTS / "q1.json")


if __name__ == "__main__":
    main()
