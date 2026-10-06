"""Development evaluation of candidate rankings (design 18.4): precision, lift, hub concentration, power.

Usage: python scripts/engine/01_dev_eval.py [--timepoint T2]

Only development timepoints (engine_reliability.dev_eval.timepoints) are accepted; T3 is never
read here. Each ranking in dev_eval.rankings is a score column (engine scores or a build-window
feature), ordered descending with ties by candidate id. Diversified rankings
(dev_eval.diversify) walk a base ranking and push back a candidate when one of its terms has
already appeared `cap` times; pushed-back candidates follow in their original order. Every
ranking is a fixed order, also inside the bootstrap. For each ranking and each k:
- Precision@k and lift over the timepoint's realisation rate
- concentration of the top k: distinct terms, the most frequent terms and their share of slots,
  distinct terms among the hits
- the candidate and the term-block bootstrap interval of the lift, drawn as in the lightweight
  validation (scripts/model/02_evaluate.py, reused through importlib)
Power (rankings in dev_eval.power): a nested bootstrap. Each outer draw resamples terms to make
a pseudo replication of the timepoint; inside it both bootstraps of the verdict are repeated and
the replication counts as passing when both lower bounds exceed 1. Power is the passing share.
It assumes the next timepoint resembles this one and is an approximation.
The output is descriptive for engine development and is never a verdict.

Output: results/engine_dev_<timepoint>.json
"""
import argparse
import datetime
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

SCORES = Path("data/derived/scores/main")
FEATURES = Path("data/derived/features/main")
LABELS = Path("data/derived/labels/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def precision_at_k(order: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> float:
    """Vectorised evaluate.weighted_precision_at_k (same fractional last block)."""
    w_o, y_o = w[order], y[order]
    cum = np.cumsum(w_o)
    if cum[-1] <= 0:
        return float("nan")
    i = int(np.searchsorted(cum, k))
    if i >= len(cum):
        return float((w_o * y_o).sum() / cum[-1])
    before = cum[i - 1] if i > 0 else 0.0
    hits = (w_o[:i] * y_o[:i]).sum() + (k - before) * y_o[i]
    return float(hits / k)


def lift(order: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> float:
    total = w.sum()
    base = (w * y).sum() / total if total > 0 else float("nan")
    return precision_at_k(order, y, w, k) / base if base and base > 0 else float("nan")


def term_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    terms = np.array(sorted(set(df["a"]) | set(df["b"]) | set(df["c"])))
    index = {t: i for i, t in enumerate(terms)}
    return terms, np.array([[index[t] for t in row] for row in df[KEYS].itertuples(index=False)])


def diversified_order(df: pd.DataFrame, order: np.ndarray, cap: int) -> np.ndarray:
    """Greedy per-term cap on a base order; candidates over the cap keep their relative order after."""
    counts = Counter()
    kept, deferred = [], []
    for i in order:
        row = (df.at[i, "a"], df.at[i, "b"], df.at[i, "c"])
        if all(counts[t] < cap for t in row):
            kept.append(i)
            counts.update(row)
        else:
            deferred.append(i)
    return np.array(kept + deferred)


def concentration(top: pd.DataFrame, hits: np.ndarray, n_terms: int) -> dict:
    """How much the top k depends on a few terms (each triad has three term slots)."""
    counts = Counter(t for row in top[KEYS].itertuples(index=False) for t in row)
    leaders = counts.most_common(n_terms)
    hit_terms = {t for row, h in zip(top[KEYS].itertuples(index=False), hits) if h for t in row}
    return {
        "distinct_terms": len(counts),
        "top_terms": [[t, n] for t, n in leaders],
        "top_terms_slot_share": sum(n for _, n in leaders) / (3 * len(top)) if len(top) else float("nan"),
        "max_term_triad_share": leaders[0][1] / len(top) if leaders else float("nan"),
        "hits": int(hits.sum()),
        "hit_distinct_terms": len(hit_terms),
    }


def ranking_metrics(df: pd.DataFrame, order: np.ndarray, y: np.ndarray, ks: list[int], n_terms: int) -> dict:
    base = y.mean()
    out = {}
    for k in ks:
        top_idx = order[:k]
        p = float(y[top_idx].mean())
        out[str(k)] = {"precision": p, "lift": p / base if base > 0 else float("nan"),
                       **concentration(df.iloc[top_idx], y[top_idx], n_terms)}
    return out


def lift_intervals(df: pd.DataFrame, order: np.ndarray, y: np.ndarray, ks: list[int], ev: dict) -> dict:
    """Both bootstraps of the lift at every k, same draw order as the lightweight validation."""
    terms, triads = term_index(df)
    rng = np.random.default_rng(ev["bootstrap_seed"])
    lifts = {kind: {k: [] for k in ks} for kind in ("candidate", "term_block")}
    for _ in range(ev["bootstrap_n"]):
        for kind, w in (("candidate", evaluate.candidate_weights(rng, len(df))),
                        ("term_block", evaluate.term_block_weights(rng, triads, terms))):
            for k in ks:
                lifts[kind][k].append(lift(order, y, w, k))
    return {str(k): {kind: evaluate.interval(lifts[kind][k], ev["ci_level"]) for kind in lifts} for k in ks}


def power(df: pd.DataFrame, order: np.ndarray, y: np.ndarray, ks: list[int], cfg: dict, level: float) -> dict:
    """Share of pseudo replications whose two lower bounds both exceed 1, per k."""
    _, triads = term_index(df)
    n_terms, n_cand = triads.max() + 1, len(df)
    rng = np.random.default_rng(cfg["seed"])
    passed = {k: {"candidate": 0, "term_block": 0, "both": 0} for k in ks}
    q = (1 - level) / 2 * 100
    for _ in range(cfg["outer"]):
        c = rng.multinomial(n_terms, np.full(n_terms, 1 / n_terms)).astype(float)
        pseudo = c[triads[:, 0]] * c[triads[:, 1]] * c[triads[:, 2]]
        if pseudo.sum() <= 0:
            continue
        inner = {kind: {k: [] for k in ks} for kind in ("candidate", "term_block")}
        for _ in range(cfg["inner"]):
            d = rng.multinomial(n_terms, c / c.sum()).astype(float)
            w_term = d[triads[:, 0]] * d[triads[:, 1]] * d[triads[:, 2]]
            w_cand = rng.multinomial(n_cand, pseudo / pseudo.sum()).astype(float)
            for k in ks:
                inner["term_block"][k].append(lift(order, y, w_term, k))
                inner["candidate"][k].append(lift(order, y, w_cand, k))
        for k in ks:
            lower = {}
            for kind in inner:
                vals = np.array([v for v in inner[kind][k] if not np.isnan(v)])
                lower[kind] = np.percentile(vals, q) if len(vals) else float("nan")
                passed[k][kind] += bool(lower[kind] > 1)
            passed[k]["both"] += bool(lower["candidate"] > 1 and lower["term_block"] > 1)
    return {str(k): {kind: n / cfg["outer"] for kind, n in v.items()} for k, v in passed.items()}


def main() -> None:
    study = load_study()
    cfg, ev = study["engine_reliability"]["dev_eval"], study["evaluation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", default=cfg["timepoints"][0], choices=cfg["timepoints"])
    tp = parser.parse_args().timepoint

    paths = {"scores": SCORES / f"{tp}.parquet", "features": FEATURES / f"{tp}.parquet", "labels": LABELS / f"{tp}.parquet"}
    df = (pd.read_parquet(paths["scores"])
          .merge(pd.read_parquet(paths["features"]), on=KEYS, validate="one_to_one")
          .merge(pd.read_parquet(paths["labels"])[KEYS + ["label"]], on=KEYS, validate="one_to_one")
          .sort_values(KEYS).reset_index(drop=True))
    y = df["label"].to_numpy().astype(float)

    ids = np.arange(len(df))
    orders = {name: (f"score {column}", evaluate.ranking(df[column].to_numpy(), ids)) for name, column in cfg["rankings"].items()}
    for cap in cfg.get("diversify", {}).get("caps", []):
        base = cfg["diversify"]["base"]
        orders[f"{base}_cap{cap}"] = (f"{base} with per-term cap {cap}", diversified_order(df, orders[base][1], cap))
    rankings = {}
    for name, (source, order) in orders.items():
        rankings[name] = {"source": source, "by_k": ranking_metrics(df, order, y, cfg["k"], cfg["top_terms"]),
                          "lift_ci": lift_intervals(df, order, y, cfg["k"], ev)}
        if name in cfg["power"]["rankings"]:
            rankings[name]["power"] = power(df, order, y, cfg["k"], cfg["power"], ev["ci_level"])
    result = {
        "meta": {"step": "engine_dev_eval", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "dev_eval_config": cfg, "evaluation_config": ev,
                 "inputs_sha256": {name: evaluate.sha256(path) for name, path in paths.items()},
                 "note": "Descriptive development diagnostics; never a verdict."},
        "candidates": int(len(df)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "rankings": rankings,
    }
    out = RESULTS / f"engine_dev_{tp}.json"
    out.write_text(json.dumps(result, indent=2))
    for name, r in rankings.items():
        for k in map(str, cfg["k"]):
            m, ci = r["by_k"][k], r["lift_ci"][k]
            pw = r.get("power", {}).get(k)
            print(f"{name:11s} k={k:>3} P {m['precision']:.3f} lift {m['lift']:6.2f} "
                  f"cand [{ci['candidate']['lower']:.2f}, {ci['candidate']['upper']:.2f}] "
                  f"term [{ci['term_block']['lower']:.2f}, {ci['term_block']['upper']:.2f}]"
                  + (f" power {pw['both']:.2f}" if pw else ""))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
