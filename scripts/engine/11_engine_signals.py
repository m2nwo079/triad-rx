"""New engine signals on the conditional candidates (design 18.9 step 4, signals b and c, hub concentration).

Usage: python scripts/engine/11_engine_signals.py

Adds two feature sets to the first-study features of the conditional candidates
(data/derived/features_v2 from scripts/engine/08_rediagnose.py), both from build-window data only:
- E (pair combination evidence, scripts/engine/09_pair_evidence.py): min and mean over the three
  pairs of n_evidence / n_cooc (zero without co-occurrence) and the number of pairs with evidence.
- S (subface context overlap): a candidate never co-occurred as a triad in the build window, so the
  papers of (u, v) and of (v, w) are disjoint. For each pivot v, the cosine between the term counts
  of the other terms in the papers of (u, v) and in the papers of (v, w), leaving out u, v and w;
  zero when a pair never co-occurred. Max and mean over the three pivots.
Each model in engine_reliability.signals.models is trained on the train timepoint with the
second-study label and the first-study settings, then scores the evaluation timepoint. Every model
order and its per-term capped versions (signals.diversify_caps, as in scripts/engine/01_dev_eval.py)
get precision, lift, concentration and both bootstraps, and the precision difference against the
plain engine order with paired bootstraps (scripts/engine/08_rediagnose.py). The output is
descriptive and is never a verdict.

Output: results/engine_signals.json
"""
import datetime
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment

HYPERGRAPH = Path("data/derived/hypergraph/main")
FEATURES = Path("data/derived/features_v2/main")
PAIR_EVIDENCE = Path("data/derived/pair_evidence/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
PAIRS = [("a", "b"), ("a", "c"), ("b", "c")]
# Pivot term and the two pairs that share it
PIVOTS = [("a", ("a", "b"), ("a", "c")), ("b", ("a", "b"), ("b", "c")), ("c", ("a", "c"), ("b", "c"))]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")
dev, evaluate, trainer = rd.dev, rd.evaluate, rd.trainer


def evidence_features(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    share = dict(zip(zip(pairs["x"], pairs["y"]), pairs["n_evidence"] / pairs["n_cooc"]))
    has = {k for k, n in zip(zip(pairs["x"], pairs["y"]), pairs["n_evidence"]) if n > 0}
    v = np.array([[share.get((r[x], r[y]), 0.0) for x, y in PAIRS] for r in df[KEYS].to_dict("records")])
    e = np.array([[(r[x], r[y]) in has for x, y in PAIRS] for r in df[KEYS].to_dict("records")])
    return pd.DataFrame({"E_share_min": v.min(axis=1), "E_share_mean": v.mean(axis=1), "E_pairs": e.sum(axis=1)}, index=df.index)


def context_features(df: pd.DataFrame, build: pd.DataFrame) -> pd.DataFrame:
    """Subface context overlap (S_ctx_max, S_ctx_mean) of each candidate."""
    terms = sorted({t for ts in build["terms"] for t in ts} | set(df["a"]) | set(df["b"]) | set(df["c"]))
    index = {t: i for i, t in enumerate(terms)}
    rows, cols = [], []
    for p, ts in enumerate(build["terms"]):
        for t in set(ts):
            rows.append(p)
            cols.append(index[t])
    incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(build), len(terms)))
    by_term = incidence.T.tocsr()
    ids = {c: df[c].map(index).to_numpy() for c in KEYS}
    pair_list = sorted({(x, y) for p, q in PAIRS for x, y in zip(ids[p], ids[q])})
    pos = {pair: n for n, pair in enumerate(pair_list)}
    px = np.array([x for x, _ in pair_list])
    py = np.array([y for _, y in pair_list])
    # Context of a pair: term counts over the papers that contain both of its terms
    ctx = (by_term[px].multiply(by_term[py]).tocsr() @ incidence).tocsr()
    sq = np.asarray(ctx.multiply(ctx).sum(axis=1)).ravel()
    pair_idx = {(p, q): np.array([pos[(x, y)] for x, y in zip(ids[p], ids[q])]) for p, q in PAIRS}
    cos = []
    for _, p1, p2 in PIVOTS:
        r1, r2 = pair_idx[p1], pair_idx[p2]
        dot = np.asarray(ctx[r1].multiply(ctx[r2]).sum(axis=1)).ravel()
        n1, n2 = sq[r1].copy(), sq[r2].copy()
        for c in KEYS:
            v1 = np.asarray(ctx[r1, ids[c]]).ravel()
            v2 = np.asarray(ctx[r2, ids[c]]).ravel()
            dot -= v1 * v2
            n1 -= v1 * v1
            n2 -= v2 * v2
        denom = np.sqrt(np.maximum(n1, 0) * np.maximum(n2, 0))
        cos.append(np.where(denom > 0, dot / np.where(denom > 0, denom, 1), 0.0))
    cos = np.column_stack(cos)
    return pd.DataFrame({"S_ctx_max": cos.max(axis=1), "S_ctx_mean": cos.mean(axis=1)}, index=df.index)


def with_signals(tp: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES / f"{tp}.parquet").sort_values(KEYS).reset_index(drop=True)
    build = pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet")
    return pd.concat([df, evidence_features(df, pd.read_parquet(PAIR_EVIDENCE / f"{tp}.parquet")),
                      context_features(df, build)], axis=1)


def main() -> None:
    study = load_study()
    er = study["engine_reliability"]
    cfg, rcfg = er["signals"], er["conditional"]["rediagnosis"]
    model_cfg, ev = study["model"], study["evaluation"]
    feat_cfg = {"sets": {**study["features"]["sets"], **cfg["sets"]}}
    top_terms = er["dev_eval"]["top_terms"]
    train_tp, eval_tp = rcfg["train_timepoint"], rcfg["eval_timepoint"]
    train_df, eval_df = with_signals(train_tp), with_signals(eval_tp)
    y_train = train_df["label_v2"].to_numpy().astype(int)
    y = eval_df["label_v2"].to_numpy().astype(float)
    ids = np.arange(len(eval_df))

    orders, models = {}, {}
    for name, sets in cfg["models"].items():
        cols = trainer.columns(feat_cfg, sets)
        rounds, cv_ap, cv_sd = trainer.choose_rounds(train_df[cols], y_train, model_cfg)
        boosters = [trainer.train(train_df[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
        score = np.mean([b.predict(eval_df[cols]) for b in boosters], axis=0)
        gain = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
        orders[name] = evaluate.ranking(score, ids)
        for cap in cfg["diversify_caps"]:
            orders[f"{name}_cap{cap}"] = dev.diversified_order(eval_df, orders[name], cap)
        models[name] = {"sets": sets, "rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd,
                        "importance_gain_mean": dict(sorted(zip(cols, map(float, gain)), key=lambda kv: -kv[1]))}
        print(f"{name}: rounds {rounds}, CV AP {cv_ap:.4f} +/- {cv_sd:.4f}")

    k = rcfg["k"]
    rankings = {name: {"by_k": dev.ranking_metrics(eval_df, order, y, k, top_terms),
                       "lift_ci": dev.lift_intervals(eval_df, order, y, k, ev)} for name, order in orders.items()}
    others = {name: order for name, order in orders.items() if name != "engine"}
    for name, d in rd.diff_intervals(eval_df, others, orders["engine"], y, k, ev).items():
        rankings[name]["precision_diff_vs_engine_ci"] = d

    signal_cols = [c for s in cfg["sets"].values() for c in s]
    result = {
        "meta": {"step": "engine_signals", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                 "signals_config": cfg, "rediagnosis_config": rcfg, "model_config": model_cfg, "evaluation_config": ev,
                 "inputs_sha256": {tp: {"features": evaluate.sha256(FEATURES / f"{tp}.parquet"),
                                        "pair_evidence": evaluate.sha256(PAIR_EVIDENCE / f"{tp}.parquet")}
                                   for tp in (train_tp, eval_tp)},
                 "note": "Descriptive development diagnostics; never a verdict."},
        "train": {"timepoint": train_tp, "candidates": int(len(train_df)), "positives": int(y_train.sum())},
        "eval": {"timepoint": eval_tp, "candidates": int(len(eval_df)), "positives": int(y.sum()), "base_rate": float(y.mean())},
        "signal_summary": {tp: d[signal_cols].describe().to_dict() for tp, d in ((train_tp, train_df), (eval_tp, eval_df))},
        "models": models,
        "rankings": rankings,
    }
    (RESULTS / "engine_signals.json").write_text(json.dumps(result, indent=2))
    for name, r in rankings.items():
        line = " ".join(f"k={kk}:{r['by_k'][str(kk)]['precision']:.3f}" for kk in k)
        print(f"{name:16s} {line}")


if __name__ == "__main__":
    main()
