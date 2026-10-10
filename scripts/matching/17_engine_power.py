"""Power of the third-study T3 verdict sequence (matching preregistration 6.4).

Usage: python scripts/matching/17_engine_power.py [--smoke]

Same method as the second study (scripts/engine/13_power.py, reused through importlib) on the v3 inputs:
the engine is trained on the train timepoint of matching.verdict.power with the v3 features and labels,
the generator's top B candidates of the evaluation timepoint are the pool, and the verdict steps run in
sequence with nested bootstraps (outer draws resample terms; inside them the candidate and term-block
bootstraps are repeated; a replication passes a step when both lower bounds clear its threshold).
Two changes, both fixed in the preregistration before this run:
- the confidence level is the verdict level (matching.t3_reuse.ci_level), not evaluation.ci_level;
- S4 (engine precision at k minus the engine without H) is added after S3, and its stand-alone power is
  reported next to the sequence powers S1, S1-S2, S1-S3 and S1-S4.
The choice rule (matching.verdict.power.rule) is the second-study rule: the budget and k with the highest
S1-and-S2 power under the conservative scenario; ties go to the smaller budget, then the smaller k.
The output is descriptive and is never a verdict.

--smoke runs 3 outer and 3 inner draws and writes data/derived/engine_v3_power_smoke.json (not committed)
to check the run and the save before the full run.

Output: results/engine_v3_power.json
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
from triadrx.study import data_cutoff, load_study, runtime_environment

FEATURES = Path("data/derived/features_v3")
LABELS = Path("data/derived/labels_v3/relation")
FIRST = Path("data/derived/candidates_v3/first")
RESULTS = Path("results")
SMOKE = Path("data/derived/engine_v3_power_smoke.json")
KEYS = ["a", "b", "c"]
THRESHOLDS = {"S1": 0.0, "S2": 1.0, "S3": 0.0, "S4": 0.0}
SEQUENCES = {"S1_S2": ("S1", "S2"), "S1_S2_S3": ("S1", "S2", "S3"), "S1_S2_S3_S4": ("S1", "S2", "S3", "S4")}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pw = load("power", ROOT / "scripts/engine/13_power.py")
cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
dev, evaluate, trainer = pw.dev, pw.evaluate, pw.trainer


def step_values(d: dict, y_pool: np.ndarray, keep_pos: np.ndarray, w_pool: np.ndarray, w_pos: np.ndarray,
                budgets: list[int], ks: list[int]) -> dict:
    """pw.step_values (S1 to S3) plus S4, the engine precision minus the engine without H."""
    out = pw.step_values(d, y_pool, keep_pos, w_pool, w_pos, budgets, ks)
    for b in budgets:
        p_eng = pw.precisions(d["orders"][b]["engine"], y_pool, w_pool, ks)
        p_noh = pw.precisions(d["orders"][b]["engine_no_h"], y_pool, w_pool, ks)
        out[b]["S4"] = [e - h for e, h in zip(p_eng, p_noh)]
    return out


def passes(draws: dict, b: int, step: str, q: float, idx: int | None = None) -> bool:
    """Both bootstrap lower bounds clear the step threshold."""
    ok = True
    for kind in draws:
        vals = np.array([v[b][step] if idx is None else v[b][step][idx] for v in draws[kind]], dtype=float)
        vals = vals[~np.isnan(vals)]
        ok = ok and bool(len(vals)) and bool(np.percentile(vals, q) > THRESHOLDS[step])
    return ok


def nested_power(d: dict, budgets: list[int], ks: list[int], cfg: dict, level: float, keep_prob: float) -> dict:
    """Share of pseudo replications passing each step alone and each sequence, per budget and k."""
    n_terms = d["n_terms"]
    rng = np.random.default_rng(cfg["seed"])
    thin = np.random.default_rng(cfg["seed"] + 1)
    q = (1 - level) / 2 * 100
    names = ["S2", "S3", "S4", *SEQUENCES]
    counts = {b: {"S1": 0, "by_k": {k: dict.fromkeys(names, 0) for k in ks}} for b in budgets}
    for _ in range(cfg["outer"]):
        c = rng.multinomial(n_terms, np.full(n_terms, 1 / n_terms)).astype(float)
        pseudo_pool = c[d["pool_terms"]].prod(axis=1)
        pseudo_pos = c[d["pos_terms"]].prod(axis=1)
        if pseudo_pool.sum() <= 0 or pseudo_pos.sum() <= 0:
            continue
        keep_pos = thin.random(len(pseudo_pos)) < keep_prob
        y_pool = d["pool_y"] * np.isin(d["pool_pos_index"], np.flatnonzero(keep_pos))
        draws = {kind: [] for kind in ("candidate", "term_block")}
        for _ in range(cfg["inner"]):
            dd = rng.multinomial(n_terms, c / c.sum()).astype(float)
            draws["term_block"].append(step_values(d, y_pool, keep_pos, dd[d["pool_terms"]].prod(axis=1),
                                                   dd[d["pos_terms"]].prod(axis=1), budgets, ks))
            w_pool = rng.multinomial(len(pseudo_pool), pseudo_pool / pseudo_pool.sum()).astype(float)
            w_pos = rng.multinomial(len(pseudo_pos), pseudo_pos / pseudo_pos.sum()).astype(float)
            draws["candidate"].append(step_values(d, y_pool, keep_pos, w_pool, w_pos, budgets, ks))
        for b in budgets:
            s1 = passes(draws, b, "S1", q)
            counts[b]["S1"] += s1
            for i, k in enumerate(ks):
                alone = {"S1": s1, **{s: passes(draws, b, s, q, i) for s in ("S2", "S3", "S4")}}
                row = counts[b]["by_k"][k]
                for s in ("S2", "S3", "S4"):
                    row[s] += alone[s]
                for name, steps in SEQUENCES.items():
                    row[name] += all(alone[s] for s in steps)
    n = cfg["outer"]
    return {str(b): {"S1": v["S1"] / n, "by_k": {str(k): {s: x / n for s, x in r.items()} for k, r in v["by_k"].items()}}
            for b, v in counts.items()}


def prepare(study: dict, cfg: dict) -> dict:
    """Pool, positives, orders and first-study membership on the v3 inputs (pw.prepare with v3 paths)."""
    rcfg = study["engine_reliability"]["conditional"]["rediagnosis"]
    model_cfg, feat_cfg = study["model"], study["features"]
    train_tp, eval_tp = cfg["train_timepoint"], cfg["eval_timepoint"]
    train_df = cands_v3.attach_labels(pd.read_parquet(FEATURES / f"{train_tp}.parquet"), pd.read_parquet(LABELS / f"{train_tp}.parquet"))
    labels = pd.read_parquet(LABELS / f"{eval_tp}.parquet")
    pool = cands_v3.attach_labels(pd.read_parquet(FEATURES / f"{eval_tp}.parquet"), labels).sort_values(KEYS).reset_index(drop=True)
    y_train = train_df["label_v3"].to_numpy().astype(int)
    scores, rounds = {}, {}
    for name in ("engine", "engine_no_h"):
        cols = trainer.columns(feat_cfg, rcfg["models"][name])
        rounds[name], _, _ = trainer.choose_rounds(train_df[cols], y_train, model_cfg)
        boosters = [trainer.train(train_df[cols], y_train, rounds[name], s, model_cfg) for s in model_cfg["seeds"]]
        scores[name] = np.mean([b.predict(pool[cols]) for b in boosters], axis=0)
    scores["baseline"] = pool[cfg["baseline_feature"]].to_numpy()

    pos = labels[labels["label_v3"]][KEYS].reset_index(drop=True)
    first = set(pd.read_parquet(FIRST / f"{eval_tp}.parquet")[KEYS].itertuples(index=False, name=None))
    first_count = json.loads((RESULTS / f"engine_v3_candidates_{eval_tp}.json").read_text())["first_study_rule"]["candidates"]
    budgets = [m * first_count for m in cfg["budget_multiples_of_first_candidates"]]
    terms = sorted(set(pool[KEYS].to_numpy().ravel()) | set(pos.to_numpy().ravel()))
    index = {t: i for i, t in enumerate(terms)}
    rank = dict(zip(pool[KEYS].itertuples(index=False, name=None), pool["gen_rank"]))
    pos_keys = list(pos.itertuples(index=False, name=None))
    pos_index = {t: i for i, t in enumerate(pos_keys)}
    ids = np.arange(len(pool))
    orders = {}
    for b in budgets:
        inside = (pool["gen_rank"] <= b).to_numpy()
        orders[b] = {name: evaluate.ranking(np.where(inside, score, -np.inf), ids)[: int(inside.sum())]
                     for name, score in scores.items()}
    return {
        "n_terms": len(terms), "budgets": budgets, "rounds": rounds, "orders": orders,
        "pool_terms": pool[KEYS].apply(lambda col: col.map(index)).to_numpy(),
        "pool_rank": pool["gen_rank"].to_numpy(), "pool_y": pool["label_v3"].to_numpy().astype(float),
        "pool_pos_index": np.array([pos_index.get(t, -1) for t in pool[KEYS].itertuples(index=False, name=None)]),
        "pos_terms": pos.apply(lambda col: col.map(index)).to_numpy(),
        "pos_in_first": np.array([t in first for t in pos_keys], dtype=float),
        "pos_rank": np.array([rank.get(t, np.inf) for t in pos_keys], dtype=float),
        "inputs": {"train_features": FEATURES / f"{train_tp}.parquet", "train_labels": LABELS / f"{train_tp}.parquet",
                   "eval_features": FEATURES / f"{eval_tp}.parquet", "eval_labels": LABELS / f"{eval_tp}.parquet",
                   "first_study_rule": FIRST / f"{eval_tp}.parquet"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    study = load_study()
    cfg = dict(study["matching"]["verdict"]["power"])
    if smoke:
        cfg.update(outer=3, inner=3)
    level = study["matching"]["t3_reuse"]["ci_level"]
    ks = cfg["k"]
    d = prepare(study, cfg)
    ratio = pw.window_ratio(study, cfg["eval_timepoint"], data_cutoff())
    ones_pool, ones_pos = np.ones(len(d["pool_y"])), np.ones(len(d["pos_terms"]))
    point = step_values(d, d["pool_y"], np.ones(len(ones_pos), bool), ones_pool, ones_pos, d["budgets"], ks)
    power = {"evaluation_like": nested_power(d, d["budgets"], ks, cfg, level, 1.0),
             "t3_conservative": nested_power(d, d["budgets"], ks, cfg, level, min(ratio, 1.0))}
    result = {
        "meta": {"step": "engine_v3_power", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                 "power_config": cfg, "ci_level": level, "engine_rounds": d["rounds"], "smoke": smoke,
                 "inputs_sha256": {k: evaluate.sha256(v) for k, v in d["inputs"].items()},
                 "note": "Descriptive power analysis on development data; never a verdict (matching preregistration 6.4)."},
        "budgets": d["budgets"], "k": ks, "t3_to_eval_label_window_ratio": ratio,
        "point_estimates": {str(b): {"S1_recall_diff": v["S1"], "S2_lift": dict(zip(map(str, ks), v["S2"])),
                                     "S3_precision_diff": dict(zip(map(str, ks), v["S3"])),
                                     "S4_precision_diff": dict(zip(map(str, ks), v["S4"]))} for b, v in point.items()},
        "power": power,
        "choice": pw.choose(power["t3_conservative"], cfg["minimum_sequence_power"]),
    }
    out = SMOKE if smoke else RESULTS / "engine_v3_power.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k in ("budgets", "t3_to_eval_label_window_ratio", "choice")}, indent=2))
    for scen, p in power.items():
        for b, v in p.items():
            for k, r in v["by_k"].items():
                print(f"{scen:16s} B={b:>7} k={k:>5} S1 {v['S1']:.2f} S1-S2 {r['S1_S2']:.2f} S1-S3 {r['S1_S2_S3']:.2f} "
                      f"S1-S4 {r['S1_S2_S3_S4']:.2f} S4 alone {r['S4']:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
