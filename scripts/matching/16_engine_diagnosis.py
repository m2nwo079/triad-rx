"""Development diagnosis of the third-study engine: train on T1, evaluate on T2 (matching preregistration 6.3).

Usage: python scripts/matching/16_engine_diagnosis.py

Inputs are the v3 features (scripts/matching/15_engine_features.py) of the train and evaluation timepoints of
matching.engine.dev_eval and their v3 relation labels (data/derived/labels_v3/relation/<tp>.parquet), attached
here from the label files. The run stops when a label file is missing, when the evaluation timepoint is T3, or
when the train labels end after the end of the evaluation build window (decision-time rule, design 18.4).

The models, feature rankings, k values and model settings are those of the second-study rediagnosis
(engine_reliability.conditional.rediagnosis, scripts/engine/08_rediagnose.py reused through importlib): the
engine (P, H, T, R) and the engine without H are trained on T1 with the first-study settings (trees chosen by
CV, mean over seeds) and score T2. On T2 the rankings are the generator order, both models and the feature
rankings; for each k they get the precision, lift and concentration, both bootstraps of the lift, and both
bootstraps of the precision difference against the generator order.
Generator recall on T2 is read against the first-study rule on the same v3 inputs at budgets of 1 and
`diagnosis_budget_multiple` times its candidate count, for the label v3 positives and for all realised triads
(the totals include triads no candidate can reach).

The output is descriptive and is never a verdict; it is not used to change the generator, features, model
settings or labels (preregistration 6.3).

Output: results/engine_v3_diagnosis.json
"""
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import build_window, label_window, load_study, runtime_environment

FEATURES = Path("data/derived/features_v3")
FIRST = Path("data/derived/candidates_v3/first")
LABELS = Path("data/derived/labels_v3/relation")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")
cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
prep = load("timepoint_prep", ROOT / "scripts/engine/14_timepoint_prep.py")
trainer, dev, evaluate = rd.trainer, rd.dev, rd.evaluate


def check_timepoints(train: str, eval_tp: str, study: dict) -> None:
    """Stop on T3 or on train labels that end after the evaluation build window (design 18.4)."""
    if eval_tp == "T3" or train == "T3":
        sys.exit("T3 is never used in the development diagnosis")
    if label_window(train, study)[1] > build_window(eval_tp, study)[1]:
        sys.exit(f"{train} labels end after the {eval_tp} build window")


def labelled(features: pd.DataFrame, label_path: Path) -> pd.DataFrame:
    if not label_path.exists():
        sys.exit(f"{label_path} is missing; the diagnosis needs the v3 relation labels")
    return cands_v3.attach_labels(features, pd.read_parquet(label_path))


def recall_by_budget(first: pd.DataFrame, generator: pd.DataFrame, labels: pd.DataFrame, multiples: list[int]) -> dict:
    """Recall of the first-study rule and of the generator's top budgets over all positives of the label file."""
    totals = {"label_v3": int(labels["label_v3"].sum()), "realised": int(len(labels))}
    gen = generator.sort_values("gen_rank")
    out = {"first_study_candidates": int(len(first)), "totals": totals,
           "first_study_rule": {name: {"hits": int(first[name].sum()), "recall": int(first[name].sum()) / totals[name]}
                                for name in totals},
           "generator": []}
    for m in multiples:
        top = gen.head(m * len(first))
        out["generator"].append({"multiple": m, "budget": m * len(first), "candidates": int(len(top)),
                                 **{name: {"hits": int(top[name].sum()), "recall": int(top[name].sum()) / totals[name]}
                                    for name in totals}})
    return out


def main() -> None:
    study = prep.merged_study(load_study())
    dev_cfg = study["matching"]["engine"]["dev_eval"]
    train_tp, eval_tp = dev_cfg["train"], dev_cfg["eval"]
    check_timepoints(train_tp, eval_tp, study)
    cfg = study["engine_reliability"]["conditional"]["rediagnosis"]
    multiple = study["engine_reliability"]["conditional"]["generator"]["diagnosis_budget_multiple"]
    model_cfg, feat_cfg, ev = study["model"], study["features"], study["evaluation"]
    top_terms = study["engine_reliability"]["dev_eval"]["top_terms"]

    paths = {tp: {"features": FEATURES / f"{tp}.parquet", "labels": LABELS / f"{tp}.parquet"} for tp in (train_tp, eval_tp)}
    data = {tp: labelled(pd.read_parquet(p["features"]), p["labels"]) for tp, p in paths.items()}
    train_df, eval_df = data[train_tp], data[eval_tp]
    y_train = train_df["label_v3"].to_numpy().astype(int)
    y = eval_df["label_v3"].to_numpy().astype(float)

    ids = np.arange(len(eval_df))
    orders = {"generator": evaluate.ranking(eval_df["gen_score"].to_numpy(), ids)}
    models = {}
    for name, sets in cfg["models"].items():
        cols = trainer.columns(feat_cfg, sets)
        rounds, cv_ap, cv_sd = trainer.choose_rounds(train_df[cols], y_train, model_cfg)
        boosters = [trainer.train(train_df[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
        score = np.mean([b.predict(eval_df[cols]) for b in boosters], axis=0)
        gain = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
        orders[name] = evaluate.ranking(score, ids)
        models[name] = {"sets": sets, "rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd,
                        "importance_gain_mean": dict(sorted(zip(cols, map(float, gain)), key=lambda kv: -kv[1]))}
        print(f"{name}: rounds {rounds}, CV AP {cv_ap:.4f} +/- {cv_sd:.4f}")
    for name, column in cfg["feature_rankings"].items():
        orders[name] = evaluate.ranking(eval_df[column].to_numpy(), ids)

    rankings = {name: {"by_k": dev.ranking_metrics(eval_df, order, y, cfg["k"], top_terms),
                       "lift_ci": dev.lift_intervals(eval_df, order, y, cfg["k"], ev)} for name, order in orders.items()}
    others = {name: order for name, order in orders.items() if name != "generator"}
    for name, d in rd.diff_intervals(eval_df, others, orders["generator"], y, cfg["k"], ev).items():
        rankings[name]["precision_diff_vs_generator_ci"] = d

    eval_labels = pd.read_parquet(paths[eval_tp]["labels"])
    first = cands_v3.attach_labels(pd.read_parquet(FIRST / f"{eval_tp}.parquet")[KEYS], eval_labels)
    recall = recall_by_budget(first, eval_df, eval_labels, sorted({1, multiple}))

    result = {
        "meta": {"step": "engine_v3_diagnosis", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "dev_eval": dev_cfg, "rediagnosis_config": cfg,
                 "model_config": model_cfg, "evaluation_config": ev, "environment": runtime_environment(),
                 "inputs_sha256": {tp: {k: evaluate.sha256(v) for k, v in p.items()} for tp, p in paths.items()}
                 | {"first_study_rule": evaluate.sha256(FIRST / f"{eval_tp}.parquet")},
                 "note": "Descriptive development diagnostics; never a verdict (matching preregistration 6.3)."},
        "train": {"timepoint": train_tp, "candidates": int(len(train_df)), "positives": int(y_train.sum())},
        "eval": {"timepoint": eval_tp, "candidates": int(len(eval_df)), "positives": int(y.sum()),
                 "base_rate": float(y.mean())},
        "models": models,
        "rankings": rankings,
        "generator_recall": recall,
    }
    (RESULTS / "engine_v3_diagnosis.json").write_text(json.dumps(result, indent=2))
    for name, r in rankings.items():
        for k in map(str, cfg["k"]):
            m, ci = r["by_k"][k], r["lift_ci"][k]
            d = r.get("precision_diff_vs_generator_ci", {}).get(k)
            print(f"{name:12s} k={k:>5} P {m['precision']:.3f} lift {m['lift']:6.2f} "
                  f"term [{ci['term_block']['lower']:.2f}, {ci['term_block']['upper']:.2f}]"
                  + (f" diff term [{d['term_block']['lower']:+.3f}, {d['term_block']['upper']:+.3f}]" if d else ""))
    for row in recall["generator"]:
        print(f"generator x{row['multiple']}: label_v3 recall {row['label_v3']['recall']:.4f} "
              f"(first-study rule {recall['first_study_rule']['label_v3']['recall']:.4f})")


if __name__ == "__main__":
    main()
