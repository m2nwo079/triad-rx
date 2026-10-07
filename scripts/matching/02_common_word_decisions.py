"""Validate the common-word review and save its decisions (matching preregistration 2.2).

Usage: python scripts/matching/02_common_word_decisions.py

The review file is checked against its read-only _original.csv: only decision, exclusions and note
may change, every decision must be one of matching.common_words.decisions, and exclusions are
given exactly for keep_with_exclusions. Decisions (term, surface, decision, exclusion phrases) are
saved without example sentences or notes.

Outputs: vocab/common_word_decisions.json, results/common_word_decisions.json
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
from triadrx.study import load_study, runtime_environment

OUT = Path("vocab/common_word_decisions.json")
RESULTS = Path("results")
EDITABLE = ["decision", "exclusions", "note"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def validate(review: pd.DataFrame, original: pd.DataFrame, allowed: list[str]) -> list[str]:
    if list(review.columns) != list(original.columns):
        return ["columns differ from the original"]
    fixed = [c for c in original.columns if c not in EDITABLE]
    errors = []
    if not review[fixed].equals(original[fixed]):
        errors.append("columns outside decision, exclusions and note were changed")
    bad = review.loc[~review["decision"].isin(allowed), "review_id"].tolist()
    if bad:
        errors.append(f"missing or unknown decision: {bad}")
    with_phrases = review["exclusions"].str.strip() != ""
    bad = review.loc[with_phrases != (review["decision"] == "keep_with_exclusions"), "review_id"].tolist()
    if bad:
        errors.append(f"exclusions must be given exactly for keep_with_exclusions: {bad}")
    return errors


def decisions(review: pd.DataFrame) -> list[dict]:
    return [{"term_id": r.term_id, "surface": r.surface, "decision": r.decision,
             "exclusions": [p.strip() for p in r.exclusions.split(";") if p.strip()]}
            for r in review.itertuples(index=False)]


def main() -> None:
    cfg = load_study()["matching"]["common_words"]
    review_path = Path(cfg["review_file"])
    original_path = review_path.with_name(review_path.stem + "_original.csv")
    review, original = read(review_path), read(original_path)
    errors = validate(review, original, cfg["decisions"])
    if errors:
        sys.exit("review file is not ready:\n- " + "\n- ".join(errors))
    rows = decisions(review)
    meta = {"step": "common_word_decisions", "created": datetime.date.today().isoformat(),
            "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
            "files_sha256": {"review": evaluate.sha256(review_path), "original": evaluate.sha256(original_path)}}
    OUT.write_text(json.dumps({"meta": meta, "decisions": rows}, indent=1))
    counts = review["decision"].value_counts().to_dict()
    summary = {"meta": {**meta, "output_sha256": evaluate.sha256(OUT)}, "surfaces": len(rows),
               "by_decision": {d: int(counts.get(d, 0)) for d in cfg["decisions"]},
               "exclusion_phrases": sum(len(r["exclusions"]) for r in rows)}
    (RESULTS / "common_word_decisions.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
