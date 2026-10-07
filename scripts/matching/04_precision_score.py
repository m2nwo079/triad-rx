"""Score the matching-precision reviews (matching preregistration 3).

Usage: python scripts/matching/04_precision_score.py

Every attempt's review file is checked against its read-only _original.csv (only correct and note
may change, correct is 1 or 0). Each stratum's final result is its last measured attempt; the pass
status follows matching.precision.pass_rule (triadrx.stats.quality_status). The final lower bounds
are the precision of rule R5 for the rebuilt vocabularies.

Output: results/precision_<tp>.json
"""
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.stats import quality_status, wilson_interval
from triadrx.study import load_study, runtime_environment

REVIEW_DIR = Path("data/review")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def validate(review: pd.DataFrame, original: pd.DataFrame) -> list[str]:
    if list(review.columns) != list(original.columns):
        return ["columns differ from the original"]
    fixed = [c for c in original.columns if c not in ("correct", "note")]
    errors = []
    if not review[fixed].equals(original[fixed]):
        errors.append("columns outside correct and note were changed")
    bad = review.loc[~review["correct"].isin(["0", "1"]), "review_id"].tolist()
    if bad:
        errors.append(f"correct must be 1 or 0: {bad}")
    return errors


def stratum_results(review: pd.DataFrame, rule: dict, attempt: int) -> dict:
    out = {}
    for name, rows in review.groupby("stratum"):
        n, correct = len(rows), int((rows["correct"] == "1").sum())
        lower, upper = wilson_interval(correct, n)
        out[name] = {"attempt": attempt, "n": n, "correct": correct, "precision": correct / n,
                     "wilson_lower": lower, "wilson_upper": upper, "status": quality_status(lower, rule)}
    return out


def main() -> None:
    cfg = load_study()["matching"]["precision"]
    tp = cfg["timepoint"]
    attempts, final, hashes = [], {}, {}
    for attempt in range(cfg["max_remeasure"] + 1):
        sheet = REVIEW_DIR / f"precision_{tp}_attempt{attempt}.csv"
        if not sheet.exists():
            break
        original = sheet.with_name(sheet.stem + "_original.csv")
        review = read(sheet)
        errors = validate(review, read(original))
        if errors:
            sys.exit(f"attempt {attempt} is not ready:\n- " + "\n- ".join(errors))
        result = stratum_results(review, cfg["pass_rule"], attempt)
        attempts.append(result)
        # The last attempt that measured a stratum is its final result
        final.update(result)
        hashes[f"attempt{attempt}"] = {"review": evaluate.sha256(sheet), "original": evaluate.sha256(original)}
    if not attempts:
        sys.exit("no precision review found")
    failed = sorted(s for s, v in final.items() if v["status"] == "fail")
    out = {
        "meta": {"step": "matching_precision", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                 "precision_config": cfg, "files_sha256": hashes},
        "attempts": attempts,
        "final_by_stratum": final,
        "final_failed_strata": failed,
        "attempts_left": cfg["max_remeasure"] + 1 - len(attempts),
    }
    (RESULTS / f"precision_{tp}.json").write_text(json.dumps(out, indent=2))
    for name, v in final.items():
        print(f"{name} (attempt {v['attempt']}): {v['correct']}/{v['n']}, lower {v['wilson_lower']:.3f}, {v['status']}")


if __name__ == "__main__":
    main()
