"""Score the label v3 review (matching preregistration 5.2).

Usage: python scripts/matching/09_label_v3_score.py [--confirm | --confirm2]

--confirm scores the relation-classifier confirmation sheet (matching.relation.confirm, attempt 1)
instead of the sentence-window sheet (attempt 0). --confirm2 scores the attempt 2 sheet
(matching.relation.confirm2, matching preregistration 5.4) with the same adoption rule.

Validation and the combination shares with Wilson intervals are those of the engine study
(scripts/engine/03_label_evidence_score.py, reused through importlib). The rule is adopted when the
rule-positive lower bound exceeds the rule-negative upper bound and reaches
matching.label_v3.target_positive_lower.

Output: results/label_v3_<tp>_score.json, results/label_v3_<tp>_attempt1_score.json or
results/label_v3_<tp>_attempt2_score.json (counts and hashes, no text)
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment

RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2score = load("label_evidence_score", ROOT / "scripts/engine/03_label_evidence_score.py")


def adopt(scored: dict, target: float) -> dict:
    separated = scored["adopted"]
    reaches = scored["rule_positive"]["wilson_lower"] >= target
    return {"separated": bool(separated), "reaches_target": bool(reaches), "adopted": bool(separated and reaches)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--confirm", action="store_true")
    group.add_argument("--confirm2", action="store_true")
    args = parser.parse_args()
    attempt = 2 if args.confirm2 else 1 if args.confirm else 0
    matching = load_study()["matching"]
    cfg = matching["label_v3"]
    files = {0: cfg, 1: matching["relation"]["confirm"], 2: matching["relation"].get("confirm2")}[attempt]
    review_path, key_path = Path(files["review_file"]), Path(files["review_key_file"])
    original_path = review_path.with_name(review_path.stem + "_original.csv")
    review, original, key = v2score.read(review_path), v2score.read(original_path), v2score.read(key_path)
    errors = v2score.validate(review, original, key)
    if errors:
        sys.exit("review file is not ready:\n- " + "\n- ".join(errors))
    scored = v2score.score(review, key)
    result = {
        "meta": {"step": "label_v3_score", "timepoint": cfg["design_timepoint"], "created": datetime.date.today().isoformat(),
                 "git_commit": v2score.evaluate.git_commit(), "environment": runtime_environment(),
                 "target_positive_lower": cfg["target_positive_lower"],
                 "files_sha256": {"review": v2score.evaluate.sha256(review_path), "original": v2score.evaluate.sha256(original_path),
                                  "key": v2score.evaluate.sha256(key_path)}},
        "rule_positive": scored["rule_positive"], "rule_negative": scored["rule_negative"],
        **adopt(scored, cfg["target_positive_lower"]),
    }
    suffix = f"_attempt{attempt}" if attempt else ""
    out = RESULTS / f"label_v3_{cfg['design_timepoint']}{suffix}_score.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
