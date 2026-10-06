"""Score the human review of the combination-evidence rule (engine preregistration 2.1.1, 2.1.2).

Usage: python scripts/engine/03_label_evidence_score.py

The review file (engine_reliability.label_v2.review_file) is checked against its read-only
`_original.csv` copy: only judgement, term_error and note may change, every judgement must be
one of the allowed values, and a term error must be judged as a mention. The rule side of each
row comes from the key file. Precision is the combination share among rule positives, the miss
is the combination share among rule negatives, both with Wilson 95% intervals. The rule is
adopted when the positive lower bound exceeds the negative upper bound.

Output: results/label_evidence_<tp>_attempt<n>_score.json (counts and hashes, no text;
attempt 0 was saved as results/label_evidence_<tp>_score.json before attempts were numbered)
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
from triadrx.stats import wilson_interval
from triadrx.study import load_study

RESULTS = Path("results")
JUDGEMENTS = {"결합": "combination", "비교": "comparison", "언급": "mention"}
EDITABLE = ["judgement", "term_error", "note"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def validate(review: pd.DataFrame, original: pd.DataFrame, key: pd.DataFrame) -> list[str]:
    """Every problem found; an empty list means the review can be scored."""
    if list(review.columns) != list(original.columns):
        return ["columns differ from the original"]
    fixed = [c for c in original.columns if c not in EDITABLE]
    errors = []
    if not review[fixed].equals(original[fixed]):
        errors.append("columns outside judgement, term_error and note were changed")
    bad = review.loc[~review["judgement"].isin(JUDGEMENTS), "review_id"].tolist()
    if bad:
        errors.append(f"missing or unknown judgement: {bad}")
    bad = review.loc[~review["term_error"].isin(["", "y"]), "review_id"].tolist()
    if bad:
        errors.append(f"term_error must be empty or y: {bad}")
    bad = review.loc[(review["term_error"] == "y") & (review["judgement"] != "언급"), "review_id"].tolist()
    if bad:
        errors.append(f"term errors must be judged as 언급: {bad}")
    if sorted(review["review_id"]) != sorted(key["review_id"]) or not key["rule"].isin(["positive", "negative"]).all():
        errors.append("key does not match the review rows")
    return errors


def side_summary(rows: pd.DataFrame) -> dict:
    n = len(rows)
    combined = int((rows["judgement"] == "결합").sum())
    lower, upper = wilson_interval(combined, n)
    return {
        "n": n,
        "judgements": {JUDGEMENTS[j]: int((rows["judgement"] == j).sum()) for j in JUDGEMENTS},
        "term_errors": int((rows["term_error"] == "y").sum()),
        "combination_share": combined / n if n else float("nan"),
        "wilson_lower": lower,
        "wilson_upper": upper,
    }


def score(review: pd.DataFrame, key: pd.DataFrame) -> dict:
    rows = review.merge(key[["review_id", "rule"]], on="review_id", validate="one_to_one")
    pos = side_summary(rows[rows["rule"] == "positive"])
    neg = side_summary(rows[rows["rule"] == "negative"])
    return {"rule_positive": pos, "rule_negative": neg,
            "adopted": bool(pos["wilson_lower"] > neg["wilson_upper"])}


def main() -> None:
    cfg = load_study()["engine_reliability"]["label_v2"]
    review_path, key_path = Path(cfg["review_file"]), Path(cfg["review_key_file"])
    original_path = review_path.with_name(review_path.stem + "_original.csv")
    review, original, key = read(review_path), read(original_path), read(key_path)
    errors = validate(review, original, key)
    if errors:
        sys.exit("review file is not ready:\n- " + "\n- ".join(errors))

    result = {
        "meta": {"step": "label_evidence_score", "timepoint": cfg["design_timepoint"],
                 "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                 "rule": "adopt when the Wilson lower bound of rule positives exceeds the upper bound of rule negatives",
                 "files_sha256": {"review": evaluate.sha256(review_path), "original": evaluate.sha256(original_path),
                                  "key": evaluate.sha256(key_path)}},
        **score(review, key),
    }
    out = RESULTS / f"label_evidence_{cfg['design_timepoint']}_attempt{cfg['attempt']}_score.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
