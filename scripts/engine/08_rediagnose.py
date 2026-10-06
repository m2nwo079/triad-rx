"""Engine and baselines re-diagnosed on the conditional completion candidates (design 18.9 step 3).

Usage: python scripts/engine/08_rediagnose.py [--candidates conditional|learned]

Candidates come from scripts/engine/07_conditional_candidates.py (conditional, the default) or from
the learned generator of scripts/engine/10_learned_generator.py (learned). Features are the first-study
features (scripts/graph/05_build_features.py, reused through importlib) computed from build-window
data only. Each model in engine_reliability.conditional.rediagnosis.models is trained on the train timepoint
with the second-study label and the first-study model settings (scripts/model/01_train.py: trees
chosen by CV, mean over seeds), then scores the evaluation timepoint. On the evaluation timepoint
the rankings are the generator order, every model and every feature ranking; for each k they get
the precision, lift and concentration of scripts/engine/01_dev_eval.py, both bootstraps of the
lift, and both bootstraps of the precision difference against the generator order (the same
weights for both orders in each draw). The output is descriptive and is never a verdict.

Outputs
- data/derived/features_v2/main/<tp>.parquet or data/derived/features_learned/main/<tp>.parquet (not committed)
- results/engine_rediagnosis.json or results/engine_rediagnosis_learned.json
"""
import argparse
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
from triadrx.study import build_window, load_study, runtime_environment

HYPERGRAPH = Path("data/derived/hypergraph/main")
# Candidate set -> (candidates, features, result file)
SETS = {
    "conditional": (Path("data/derived/candidates_v2/main"), Path("data/derived/features_v2/main"), "engine_rediagnosis.json"),
    "learned": (Path("data/derived/candidates_learned/main"), Path("data/derived/features_learned/main"),
                "engine_rediagnosis_learned.json"),
}
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = load("build_features", ROOT / "scripts/graph/05_build_features.py")
trainer = load("train", ROOT / "scripts/model/01_train.py")
dev = load("dev_eval", ROOT / "scripts/engine/01_dev_eval.py")
evaluate = dev.evaluate


def build_features(tp: str, study: dict, candidates: Path) -> pd.DataFrame:
    """First-study features of the conditional candidates, with the same recent window and leakage check."""
    cfg = study["features"]
    edges = pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet")
    edges["first_date"] = pd.to_datetime(edges["first_date"], utc=True)
    start, end = build_window(tp, study)
    if edges["first_date"].max() > end or edges["first_date"].min() < start:
        sys.exit("hypergraph contains papers outside the build window")
    recent_start = start.replace(year=end.year - cfg["recent_years"] + 1)
    cands = pd.read_parquet(candidates / f"{tp}.parquet")
    vectors, _ = features.term_vectors(edges, cfg["text"])
    feats = features.compute(edges, cands, features.cands_mod.term_pi(tp, "main"),
                             study["vocab"]["r5_rule"]["stability"], recent_start, vectors)
    return cands.merge(feats, on=KEYS, validate="one_to_one").sort_values(KEYS).reset_index(drop=True)


def diff_intervals(df: pd.DataFrame, orders: dict, ref: np.ndarray, y: np.ndarray, ks: list[int], ev: dict) -> dict:
    """Bootstrap intervals of precision(order) - precision(ref) at each k, both resampling schemes."""
    terms, triads = dev.term_index(df)
    rng = np.random.default_rng(ev["bootstrap_seed"])
    diffs = {name: {kind: {k: [] for k in ks} for kind in ("candidate", "term_block")} for name in orders}
    for _ in range(ev["bootstrap_n"]):
        for kind, w in (("candidate", evaluate.candidate_weights(rng, len(df))),
                        ("term_block", evaluate.term_block_weights(rng, triads, terms))):
            for k in ks:
                base = dev.precision_at_k(ref, y, w, k)
                for name, order in orders.items():
                    diffs[name][kind][k].append(dev.precision_at_k(order, y, w, k) - base)
    return {name: {str(k): {kind: evaluate.interval(d[kind][k], ev["ci_level"]) for kind in d} for k in ks}
            for name, d in diffs.items()}


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["conditional"]["rediagnosis"]
    model_cfg, feat_cfg, ev = study["model"], study["features"], study["evaluation"]
    top_terms = study["engine_reliability"]["dev_eval"]["top_terms"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="conditional", choices=list(SETS))
    candidates, features_dir, out_name = SETS[parser.parse_args().candidates]
    features_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for tp in (cfg["train_timepoint"], cfg["eval_timepoint"]):
        data[tp] = build_features(tp, study, candidates)
        data[tp].to_parquet(features_dir / f"{tp}.parquet", index=False)

    train_df, eval_df = data[cfg["train_timepoint"]], data[cfg["eval_timepoint"]]
    y_train = train_df["label_v2"].to_numpy().astype(int)
    y = eval_df["label_v2"].to_numpy().astype(float)
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
    for name, d in diff_intervals(eval_df, others, orders["generator"], y, cfg["k"], ev).items():
        rankings[name]["precision_diff_vs_generator_ci"] = d

    result = {
        "meta": {"step": "engine_rediagnosis", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "rediagnosis_config": cfg, "model_config": model_cfg,
                 "evaluation_config": ev,
                 "candidates": str(candidates), "environment": runtime_environment(),
                 "inputs_sha256": {tp: evaluate.sha256(candidates / f"{tp}.parquet") for tp in data},
                 "note": "Descriptive development diagnostics; never a verdict."},
        "train": {"timepoint": cfg["train_timepoint"], "candidates": int(len(train_df)), "positives": int(y_train.sum())},
        "eval": {"timepoint": cfg["eval_timepoint"], "candidates": int(len(eval_df)), "positives": int(y.sum()),
                 "base_rate": float(y.mean())},
        "models": models,
        "rankings": rankings,
    }
    (RESULTS / out_name).write_text(json.dumps(result, indent=2))
    for name, r in rankings.items():
        for k in map(str, cfg["k"]):
            m, ci = r["by_k"][k], r["lift_ci"][k]
            d = r.get("precision_diff_vs_generator_ci", {}).get(k)
            print(f"{name:12s} k={k:>5} P {m['precision']:.3f} lift {m['lift']:6.2f} "
                  f"term [{ci['term_block']['lower']:.2f}, {ci['term_block']['upper']:.2f}]"
                  + (f" diff term [{d['term_block']['lower']:+.3f}, {d['term_block']['upper']:+.3f}]" if d else ""))


if __name__ == "__main__":
    main()
