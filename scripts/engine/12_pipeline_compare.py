"""Candidate pipelines compared on the same labels and resamples (design 18.9 step 4 follow-up).

Usage: python scripts/engine/12_pipeline_compare.py

Each pipeline in engine_reliability.pipeline_compare.pipelines is a candidate set (its features file
from scripts/engine/08_rediagnose.py) and an order: "generator" ranks by the generator score, any
other name is a model of engine_reliability.conditional.rediagnosis.models trained on the train
timepoint with the second-study label and the first-study settings (scripts/model/01_train.py),
then scoring the evaluation timepoint. All pipelines are read on the evaluation timepoint over the
union of their candidates, so that one resample gives every candidate a single weight: the
candidate bootstrap draws candidates of the union, the term-block bootstrap draws terms of the
union. For each k: precision, hits, the share of all second-study positives found, concentration,
and both bootstrap intervals of the precision difference against the reference pipeline. The output
is descriptive and is never a verdict.

Output: results/pipeline_compare.json
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
from triadrx.study import load_study, runtime_environment

LABELS_V2 = Path("data/derived/labels_v2/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")
dev, evaluate, trainer = rd.dev, rd.evaluate, rd.trainer


def union_index(pools: dict) -> tuple[pd.DataFrame, dict]:
    """Union of the pools' triads and, for each pool, the union position of each of its rows."""
    union = pd.concat([p[KEYS + ["label_v2"]] for p in pools.values()]).drop_duplicates(KEYS)
    union = union.sort_values(KEYS).reset_index(drop=True)
    pos = pd.Series(np.arange(len(union)), index=pd.MultiIndex.from_frame(union[KEYS]))
    return union, {name: pos.loc[pd.MultiIndex.from_frame(p[KEYS])].to_numpy() for name, p in pools.items()}


def paired_diffs(union: pd.DataFrame, ranked: dict, ref: str, ks: list[int], ev: dict) -> dict:
    """Bootstrap intervals of precision(pipeline) - precision(reference); `ranked` holds union positions in rank order."""
    terms, triads = dev.term_index(union)
    y = union["label_v2"].to_numpy().astype(float)
    rng = np.random.default_rng(ev["bootstrap_seed"])
    others = [name for name in ranked if name != ref]
    diffs = {name: {kind: {k: [] for k in ks} for kind in ("candidate", "term_block")} for name in others}
    for _ in range(ev["bootstrap_n"]):
        for kind, w in (("candidate", evaluate.candidate_weights(rng, len(union))),
                        ("term_block", evaluate.term_block_weights(rng, triads, terms))):
            prec = {name: {k: dev.precision_at_k(np.arange(len(r)), y[r], w[r], k) for k in ks} for name, r in ranked.items()}
            for name in others:
                for k in ks:
                    diffs[name][kind][k].append(prec[name][k] - prec[ref][k])
    return {name: {str(k): {kind: evaluate.interval(d[kind][k], ev["ci_level"]) for kind in d} for k in ks}
            for name, d in diffs.items()}


def main() -> None:
    study = load_study()
    er = study["engine_reliability"]
    cfg, rcfg = er["pipeline_compare"], er["conditional"]["rediagnosis"]
    model_cfg, feat_cfg, ev = study["model"], study["features"], study["evaluation"]
    top_terms = er["dev_eval"]["top_terms"]
    train_tp, eval_tp = rcfg["train_timepoint"], rcfg["eval_timepoint"]

    pools, ranked_rows, models = {}, {}, {}
    for name, spec in cfg["pipelines"].items():
        folder = Path(spec["features"])
        train_df = pd.read_parquet(folder / f"{train_tp}.parquet")
        eval_df = pd.read_parquet(folder / f"{eval_tp}.parquet").sort_values(KEYS).reset_index(drop=True)
        ids = np.arange(len(eval_df))
        if spec["order"] == "generator":
            score = eval_df["gen_score"].to_numpy()
        else:
            cols = trainer.columns(feat_cfg, rcfg["models"][spec["order"]])
            y_train = train_df["label_v2"].to_numpy().astype(int)
            rounds, cv_ap, _ = trainer.choose_rounds(train_df[cols], y_train, model_cfg)
            boosters = [trainer.train(train_df[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
            score = np.mean([b.predict(eval_df[cols]) for b in boosters], axis=0)
            models[name] = {"rounds": rounds, "cv_average_precision_mean": cv_ap}
        pools[name] = eval_df
        ranked_rows[name] = evaluate.ranking(score, ids)
        print(f"{name}: {len(eval_df)} candidates, {int(eval_df['label_v2'].sum())} positives")

    union, positions = union_index(pools)
    ranked = {name: positions[name][order] for name, order in ranked_rows.items()}
    labels = pd.read_parquet(LABELS_V2 / f"{eval_tp}.parquet")
    total = int(labels["label_v2"].sum())
    by_pipeline = {}
    for name, order in ranked_rows.items():
        df, y = pools[name], pools[name]["label_v2"].to_numpy().astype(float)
        metrics = dev.ranking_metrics(df, order, y, cfg["k"], top_terms)
        for k in cfg["k"]:
            metrics[str(k)]["share_of_all_positives"] = metrics[str(k)]["hits"] / total
        by_pipeline[name] = {"candidates": int(len(df)), "positives_in_pool": int(y.sum()),
                             "pool_recall": float(y.sum() / total), "by_k": metrics}
    for name, d in paired_diffs(union, ranked, cfg["reference"], cfg["k"], ev).items():
        by_pipeline[name]["precision_diff_vs_reference_ci"] = d

    result = {
        "meta": {"step": "pipeline_compare", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                 "pipeline_compare_config": cfg, "rediagnosis_config": rcfg, "model_config": model_cfg, "evaluation_config": ev,
                 "inputs_sha256": {name: {tp: evaluate.sha256(Path(spec["features"]) / f"{tp}.parquet") for tp in (train_tp, eval_tp)}
                                   for name, spec in cfg["pipelines"].items()},
                 "note": "Descriptive development diagnostics; never a verdict."},
        "eval_timepoint": eval_tp, "label_v2_positives": total, "union_candidates": int(len(union)),
        "models": models, "pipelines": by_pipeline,
    }
    (RESULTS / "pipeline_compare.json").write_text(json.dumps(result, indent=2))
    for name, r in by_pipeline.items():
        line = " ".join(f"k={k}:{r['by_k'][str(k)]['precision']:.3f}" for k in cfg["k"])
        d = r.get("precision_diff_vs_reference_ci", {})
        extra = " ".join(f"[{d[str(k)]['term_block']['lower']:+.3f},{d[str(k)]['term_block']['upper']:+.3f}]" for k in cfg["k"]) if d else ""
        print(f"{name:20s} pool recall {r['pool_recall']:.3f} {line} {extra}")


if __name__ == "__main__":
    main()
