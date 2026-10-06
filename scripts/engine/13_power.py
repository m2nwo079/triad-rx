"""Power of the verdict sequence for the primary pipeline (design 18.9 step 5, 18.10).

Usage: python scripts/engine/13_power.py

Primary pipeline (engine_reliability.verdict_pipeline): the conditional nPMI generator keeps its top
B candidates and the engine (P, H, T, R; first-study settings, trained on the train timepoint with
the second-study label) ranks them. On the evaluation timepoint the verdict steps run in sequence:
- S1 recall: the share of all second-study positives inside the top B of the generator minus the
  share inside the first-study candidates (taken whole when B exceeds their number); passes when
  both bootstrap lower bounds exceed 0.
- S2 precision against random: the lift of the engine's top k inside the pool; both lower bounds > 1.
- S3 precision against the baseline: engine precision at k minus that of the baseline feature
  ranking inside the same pool; both lower bounds > 0.
Power is a nested bootstrap as in scripts/engine/01_dev_eval.py: each outer draw resamples terms to
make a pseudo replication, inside it the candidate and term-block bootstraps are repeated, and the
replication passes a step when both lower bounds clear its threshold. Recall resamples the
positives and precision resamples the pool, each with its own candidate draw; the term-block draw
is shared. The conservative scenario keeps each positive with probability equal to the T3 label
window length over the evaluation label window length (positives are fewer when the window is
shorter). The choice rule is power_v2.rule. The output is descriptive and is never a verdict.

Output: results/engine_power.json
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
from triadrx.study import data_cutoff, label_window, load_study, runtime_environment, t3_window

FEATURES = Path("data/derived/features_v2/main")
LABELS_V2 = Path("data/derived/labels_v2/main")
FIRST_CANDIDATES = Path("data/derived/candidates/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")
dev, evaluate, trainer = rd.dev, rd.evaluate, rd.trainer


def window_ratio(study: dict, eval_tp: str, cutoff: datetime.datetime) -> float:
    t3_start, t3_end = t3_window("label", study, cutoff)
    start, end = label_window(eval_tp, study)
    return (t3_end - t3_start).total_seconds() / (end - start).total_seconds()


def precisions(order: np.ndarray, y: np.ndarray, w: np.ndarray, ks: list[int]) -> list[float]:
    """dev.precision_at_k at several k from one cumulative sum (same fractional last block)."""
    w_o, y_o = w[order], y[order]
    cum = np.cumsum(w_o)
    hits = np.cumsum(w_o * y_o)
    out = []
    for k in ks:
        if cum[-1] <= 0:
            out.append(float("nan"))
            continue
        i = int(np.searchsorted(cum, k))
        if i >= len(cum):
            out.append(float(hits[-1] / cum[-1]))
            continue
        before_w = cum[i - 1] if i > 0 else 0.0
        before_h = hits[i - 1] if i > 0 else 0.0
        out.append(float((before_h + (k - before_w) * y_o[i]) / k))
    return out


def step_values(d: dict, y_pool: np.ndarray, keep_pos: np.ndarray, w_pool: np.ndarray, w_pos: np.ndarray,
                budgets: list[int], ks: list[int]) -> dict:
    """S1 recall difference per budget, S2 lift and S3 precision difference per budget and k."""
    out = {}
    wp = w_pos * keep_pos
    total = wp.sum()
    first = (wp * d["pos_in_first"]).sum()
    for b in budgets:
        cond = (wp * (d["pos_rank"] <= b)).sum()
        in_pool = d["pool_rank"] <= b
        base = (w_pool * y_pool)[in_pool].sum() / w_pool[in_pool].sum() if w_pool[in_pool].sum() > 0 else float("nan")
        p_eng = precisions(d["orders"][b]["engine"], y_pool, w_pool, ks)
        p_base = precisions(d["orders"][b]["baseline"], y_pool, w_pool, ks)
        out[b] = {"S1": (cond - first) / total if total > 0 else float("nan"),
                  "S2": [p / base if base > 0 else float("nan") for p in p_eng],
                  "S3": [e - q for e, q in zip(p_eng, p_base)]}
    return out


def nested_power(d: dict, budgets: list[int], ks: list[int], cfg: dict, level: float, keep_prob: float) -> dict:
    """Share of pseudo replications that pass S1, S1 and S2, and S1 to S3, per budget and k."""
    n_terms = d["n_terms"]
    rng = np.random.default_rng(cfg["seed"])
    thin = np.random.default_rng(cfg["seed"] + 1)
    q = (1 - level) / 2 * 100
    thresholds = {"S1": 0.0, "S2": 1.0, "S3": 0.0}
    counts = {b: {"S1": 0, "by_k": {k: {"S2": 0, "S3": 0, "S1_S2": 0, "S1_S2_S3": 0} for k in ks}} for b in budgets}
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
            def passes(step, idx=None):
                ok = True
                for kind in draws:
                    vals = np.array([v[b][step] if idx is None else v[b][step][idx] for v in draws[kind]], dtype=float)
                    vals = vals[~np.isnan(vals)]
                    ok = ok and bool(len(vals)) and bool(np.percentile(vals, q) > thresholds[step])
                return ok
            s1 = passes("S1")
            counts[b]["S1"] += s1
            for i, k in enumerate(ks):
                s2, s3 = passes("S2", i), passes("S3", i)
                row = counts[b]["by_k"][k]
                row["S2"] += s2
                row["S3"] += s3
                row["S1_S2"] += s1 and s2
                row["S1_S2_S3"] += s1 and s2 and s3
    n = cfg["outer"]
    return {str(b): {"S1": v["S1"] / n, "by_k": {str(k): {s: x / n for s, x in r.items()} for k, r in v["by_k"].items()}}
            for b, v in counts.items()}


def choose(power: dict, minimum: float) -> dict:
    """Budget and k with the highest S1-and-S2 power; ties go to the smaller budget, then the smaller k."""
    best = None
    for b, v in power.items():
        for k, r in v["by_k"].items():
            key = (r["S1_S2"], -int(b), -int(k))
            if best is None or key > best[0]:
                best = (key, b, k, r["S1_S2"])
    return {"budget": int(best[1]), "k": int(best[2]), "power_S1_S2": float(best[3]), "meets_minimum": bool(best[3] >= minimum)}


def prepare(study: dict, cfg: dict) -> dict:
    rcfg = study["engine_reliability"]["conditional"]["rediagnosis"]
    model_cfg, feat_cfg = study["model"], study["features"]
    train_df = pd.read_parquet(FEATURES / f"{cfg['train_timepoint']}.parquet")
    pool = pd.read_parquet(FEATURES / f"{cfg['eval_timepoint']}.parquet").sort_values(KEYS).reset_index(drop=True)
    cols = trainer.columns(feat_cfg, rcfg["models"]["engine"])
    y_train = train_df["label_v2"].to_numpy().astype(int)
    rounds, _, _ = trainer.choose_rounds(train_df[cols], y_train, model_cfg)
    boosters = [trainer.train(train_df[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
    engine = np.mean([b.predict(pool[cols]) for b in boosters], axis=0)

    labels = pd.read_parquet(LABELS_V2 / f"{cfg['eval_timepoint']}.parquet")
    pos = labels[labels["label_v2"]][KEYS].reset_index(drop=True)
    first = set(pd.read_parquet(FIRST_CANDIDATES / f"{cfg['eval_timepoint']}.parquet")[KEYS].itertuples(index=False, name=None))
    terms = sorted(set(pool[KEYS].to_numpy().ravel()) | set(pos.to_numpy().ravel()))
    index = {t: i for i, t in enumerate(terms)}
    rank = dict(zip(pool[KEYS].itertuples(index=False, name=None), pool["gen_rank"]))
    pos_keys = list(pos.itertuples(index=False, name=None))
    pos_index = {t: i for i, t in enumerate(pos_keys)}
    first_count = json.loads((RESULTS / f"candidates_main_{cfg['eval_timepoint']}.json").read_text())["stats"]["candidates"]
    budgets = [m * first_count for m in cfg["budget_multiples_of_first_candidates"]]
    ids = np.arange(len(pool))
    orders = {}
    for b in budgets:
        inside = (pool["gen_rank"] <= b).to_numpy()
        orders[b] = {name: evaluate.ranking(np.where(inside, score, -np.inf), ids)[: int(inside.sum())]
                     for name, score in (("engine", engine), ("baseline", pool[cfg["baseline_feature"]].to_numpy()))}
    return {
        "n_terms": len(terms), "budgets": budgets, "rounds": rounds, "orders": orders,
        "pool_terms": pool[KEYS].apply(lambda col: col.map(index)).to_numpy(),
        "pool_rank": pool["gen_rank"].to_numpy(), "pool_y": pool["label_v2"].to_numpy().astype(float),
        "pool_pos_index": np.array([pos_index.get(t, -1) for t in pool[KEYS].itertuples(index=False, name=None)]),
        "pos_terms": pos.apply(lambda col: col.map(index)).to_numpy(),
        "pos_in_first": np.array([t in first for t in pos_keys], dtype=float),
        "pos_rank": np.array([rank.get(t, np.inf) for t in pos_keys], dtype=float),
    }


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["power_v2"]
    ev = study["evaluation"]
    ks = cfg["k"]
    d = prepare(study, cfg)
    ratio = window_ratio(study, cfg["eval_timepoint"], data_cutoff())
    ones_pool, ones_pos = np.ones(len(d["pool_y"])), np.ones(len(d["pos_terms"]))
    point = step_values(d, d["pool_y"], np.ones(len(ones_pos), bool), ones_pool, ones_pos, d["budgets"], ks)
    power = {"evaluation_like": nested_power(d, d["budgets"], ks, cfg, ev["ci_level"], 1.0),
             "t3_conservative": nested_power(d, d["budgets"], ks, cfg, ev["ci_level"], min(ratio, 1.0))}
    result = {
        "meta": {"step": "engine_power", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                 "power_config": cfg, "evaluation_config": ev, "engine_rounds": d["rounds"],
                 "note": "Descriptive power analysis on development data; never a verdict."},
        "budgets": d["budgets"], "k": ks, "t3_to_eval_label_window_ratio": ratio,
        "point_estimates": {str(b): {"S1_recall_diff": v["S1"], "S2_lift": dict(zip(map(str, ks), v["S2"])),
                                     "S3_precision_diff": dict(zip(map(str, ks), v["S3"]))} for b, v in point.items()},
        "power": power,
        "choice": choose(power["t3_conservative"], cfg["minimum_sequence_power"]),
    }
    (RESULTS / "engine_power.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k in ("t3_to_eval_label_window_ratio", "point_estimates", "choice")}, indent=2))
    for scen, p in power.items():
        for b, v in p.items():
            print(scen, b, "S1", v["S1"], {k: (r["S1_S2"], r["S1_S2_S3"]) for k, r in v["by_k"].items()})


if __name__ == "__main__":
    main()
