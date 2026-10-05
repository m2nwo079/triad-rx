"""Score an M2 review attempt and combine attempts into the final M2 result (design 9.2).

Usage: python scripts/evidence/11_m2_score.py --attempt 1

1. Checks that the reviewed file matches its read-only original in every column
   except `correct` and `note`, and that every `correct` is 0 or 1.
2. Per category: correct / n, Wilson 95% lower bound and the verdict from
   quality.m2.pass_rule (pass, conditional, fail).
3. Final result: for each configured category, the latest scored attempt that
   measured it (preregistration 5.4.1: only failed categories are remeasured).
   Unscored attempts (for example attempt 0, voided before judging) are ignored.

Outputs
- results/m2_scores_attempt<N>.json
- results/m2_final.json
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

REVIEW_DIR = Path("data/review")
RESULTS = Path("results")
EDITABLE = {"correct", "note"}
Z = 1.959963984540054


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def wilson_lower(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - margin


def wilson_upper(k: int, n: int) -> float:
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre + margin


def verdict(lower: float, rule: dict) -> str:
    if lower >= rule["pass"]:
        return "pass"
    if lower >= rule["conditional"]:
        return "conditional"
    return "fail"


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def score_attempt(attempt: int, rule: dict) -> dict:
    review = REVIEW_DIR / f"m2_review_attempt{attempt}.csv"
    original = REVIEW_DIR / f"m2_review_attempt{attempt}_original.csv"
    head_r, rows_r = read_rows(review)
    head_o, rows_o = read_rows(original)
    if head_r != head_o:
        sys.exit(f"header differs from the original: {head_r} vs {head_o}")
    if len(rows_r) != len(rows_o):
        sys.exit(f"row count differs from the original: {len(rows_r)} vs {len(rows_o)}")
    for i, (r, o) in enumerate(zip(rows_r, rows_o), start=2):
        changed = [k for k in head_o if k not in EDITABLE and r[k] != o[k]]
        if changed:
            sys.exit(f"line {i}: columns other than correct/note changed: {changed}")
        if r["correct"].strip() not in ("0", "1"):
            sys.exit(f"line {i}: correct must be 0 or 1, got {r['correct']!r}")

    categories = {}
    for row in rows_r:
        cat = categories.setdefault(row["category"], {"n": 0, "correct": 0, "unsure_notes": 0})
        cat["n"] += 1
        cat["correct"] += int(row["correct"].strip())
        if "애매" in row["note"]:
            cat["unsure_notes"] += 1
    for cat in categories.values():
        cat["precision"] = cat["correct"] / cat["n"]
        cat["wilson_lower"] = wilson_lower(cat["correct"], cat["n"])
        cat["wilson_upper"] = wilson_upper(cat["correct"], cat["n"])
        cat["verdict"] = verdict(cat["wilson_lower"], rule)

    meta = {
        "step": "m2_scores",
        "attempt": attempt,
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "review_sha256": sha256(review),
        "original_sha256": sha256(original),
        "pass_rule": rule,
    }
    return {"meta": meta, "categories": categories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()

    m2 = load_study()["quality"]["m2"]
    scores = score_attempt(args.attempt, m2["pass_rule"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"m2_scores_attempt{args.attempt}.json").write_text(json.dumps(scores, indent=2))

    scored = sorted(
        (int(p.stem.removeprefix("m2_scores_attempt")), json.loads(p.read_text()))
        for p in RESULTS.glob("m2_scores_attempt*.json")
    )
    final, history = {}, {}
    for attempt, result in scored:
        for category, values in result["categories"].items():
            history.setdefault(category, []).append(attempt)
            final[category] = {"attempt": attempt, **values}
    over = {c: a for c, a in history.items() if len(a) > 1 + m2["max_remeasure"]}
    if over:
        sys.exit(f"more remeasurements than quality.m2.max_remeasure allows: {over}")
    missing = [c for c in m2["categories"] if c not in final]

    result = {
        "meta": {
            "step": "m2_final",
            "created": datetime.date.today().isoformat(),
            "git_commit": git_commit(),
            "scored_attempts": [a for a, _ in scored],
            "rule": "per category, the latest scored attempt that measured it",
        },
        "categories": final,
        "attempts_per_category": history,
        "not_measured": missing,
    }
    (RESULTS / "m2_final.json").write_text(json.dumps(result, indent=2))
    for category in m2["categories"]:
        if category in final:
            v = final[category]
            print(f"{category:<11} attempt {v['attempt']}: {v['correct']}/{v['n']} "
                  f"lower {v['wilson_lower']:.3f} -> {v['verdict']}")
        else:
            print(f"{category:<11} not measured")
    print("Saved", RESULTS / f"m2_scores_attempt{args.attempt}.json", RESULTS / "m2_final.json")


if __name__ == "__main__":
    main()
