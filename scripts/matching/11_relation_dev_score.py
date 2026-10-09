"""Agreement of the relation classifier with the human reviews (matching preregistration 5.3, development).

Usage: python scripts/matching/11_relation_dev_score.py --outputs <relation_outputs.jsonl> --meta <relation_meta.json> --tag v1

Joins the model outputs with data/derived/relation/dev_truth.parquet. Human judgements map to
combination (결합), comparison (비교) and mention (언급). Reports the confusion table, agreement, Cohen's
kappa, the precision and recall of the model's combination against human combination (Wilson 95%),
term-error agreement, invalid outputs and throughput. Development only; never a verdict.

Output: results/relation_dev_<tag>.json
"""
import argparse
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
from triadrx.study import runtime_environment

TRUTH = Path("data/derived/relation/dev_truth.parquet")
RESULTS = Path("results")
HUMAN = {"결합": "combination", "비교": "comparison", "언급": "mention"}
LABELS = ["combination", "comparison", "mention"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def kappa(a: pd.Series, b: pd.Series) -> float:
    observed = (a == b).mean()
    expected = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def score(df: pd.DataFrame) -> dict:
    """`df` has human and model columns with the three relations (model may also be invalid)."""
    model = df["model"].where(df["model"].isin(LABELS), "mention")
    table = pd.crosstab(df["human"], df["model"]).reindex(index=LABELS, fill_value=0).to_dict()
    pred = model == "combination"
    true = df["human"] == "combination"
    tp = int((pred & true).sum())
    p_lo, p_hi = wilson_interval(tp, int(pred.sum()))
    r_lo, r_hi = wilson_interval(tp, int(true.sum()))
    return {"n": int(len(df)), "confusion_human_by_model": table, "agreement": float((model == df["human"]).mean()),
            "kappa": kappa(df["human"], model),
            "combination_precision": tp / pred.sum() if pred.sum() else float("nan"), "combination_precision_ci": [p_lo, p_hi],
            "combination_recall": tp / true.sum() if true.sum() else float("nan"), "combination_recall_ci": [r_lo, r_hi],
            "invalid": int((df["model"] == "invalid").sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    outputs = pd.read_json(args.outputs, lines=True)
    truth = pd.read_parquet(TRUTH)
    df = truth.merge(outputs.rename(columns={"relation": "model", "term_error": "model_term_error"}), on="id", validate="one_to_one")
    df["human"] = df["judgement"].map(HUMAN)
    meta = json.loads(Path(args.meta).read_text())
    result = {
        "meta": {"step": "relation_dev", "tag": args.tag, "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(), "model_run": meta,
                 "outputs_sha256": evaluate.sha256(Path(args.outputs)), "truth_sha256": evaluate.sha256(TRUTH),
                 "note": "Development agreement with existing human reviews; never a verdict."},
        "all": score(df),
        "by_source": {s: score(g) for s, g in df.groupby("source")},
        "term_error_agreement": float((df["model_term_error"] == df["term_error"]).mean()),
    }
    (RESULTS / f"relation_dev_{args.tag}.json").write_text(json.dumps(result, indent=2))
    a = result["all"]
    print(f"n {a['n']} agreement {a['agreement']:.3f} kappa {a['kappa']:.3f} combination precision {a['combination_precision']:.3f} "
          f"recall {a['combination_recall']:.3f} invalid {a['invalid']} rate {meta['records_per_second']:.2f}/s")


if __name__ == "__main__":
    main()
