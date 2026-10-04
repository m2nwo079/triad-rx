"""Lightweight validation on T2: V1 (Precision@100 lift) and V2 (AUPRC gain of H features).

Usage: python scripts/model/02_evaluate.py [--variant main]

Judgement (preregistration 4.2): V1 passes if the 95% lower bound of the lift exceeds 1, and
V2 passes if the 95% lower bound of the AUPRC difference exceeds 0, under BOTH the candidate
bootstrap and the term-block bootstrap. Scores are the mean over model seeds.
Requires T2 labels, which exist only after the preregistration freeze.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

SCORES = Path("data/derived/scores")
LABELS = Path("data/derived/labels")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def ranking(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Order by score descending, ties broken by candidate id ascending."""
    return np.lexsort((ids, -scores))


def weighted_precision_at_k(order: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> float:
    """Precision of the top k units when candidate i counts w[i] times (fractional last block)."""
    cum, hits = 0.0, 0.0
    for i in order:
        if w[i] <= 0:
            continue
        take = min(w[i], k - cum)
        hits += take * y[i]
        cum += take
        if cum >= k:
            break
    return hits / cum if cum > 0 else float("nan")


def metrics(order_e, order_b, scores_e, scores_b, y, w, k) -> dict:
    """V1 lift and V2 AUPRC difference for one (re)sample given candidate weights."""
    total = w.sum()
    base = (w * y).sum() / total if total > 0 else float("nan")
    p_at_k = weighted_precision_at_k(order_e, y, w, k)
    lift = p_at_k / base if base and base > 0 else float("nan")
    mask = w > 0
    if (w[mask] * y[mask]).sum() == 0 or (w[mask] * (1 - y[mask])).sum() == 0:
        diff = float("nan")
    else:
        ap_e = average_precision_score(y[mask], scores_e[mask], sample_weight=w[mask])
        ap_b = average_precision_score(y[mask], scores_b[mask], sample_weight=w[mask])
        diff = ap_e - ap_b
    return {"lift": lift, "auprc_diff": diff}


def candidate_weights(rng, n: int) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def term_block_weights(rng, triads: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """Resample terms with replacement; a candidate's weight is the product of its terms' counts."""
    draws = np.bincount(rng.integers(0, len(terms), len(terms)), minlength=len(terms)).astype(float)
    return draws[triads[:, 0]] * draws[triads[:, 1]] * draws[triads[:, 2]]


def interval(values: list[float], level: float) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)])
    lo, hi = np.percentile(arr, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]) if len(arr) else (float("nan"),) * 2
    return {"lower": float(lo), "upper": float(hi), "valid_resamples": int(len(arr))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    variant = args.variant

    study = load_study()
    cfg, model_cfg = study["evaluation"], study["model"]
    label_path = LABELS / variant / "T2.parquet"
    if not label_path.exists():
        sys.exit("T2 labels do not exist yet (they are created only after the preregistration freeze)")
    score_path = SCORES / variant / "T2.parquet"
    df = pd.read_parquet(score_path).merge(pd.read_parquet(label_path)[KEYS + ["label"]], on=KEYS, validate="one_to_one")
    df = df.sort_values(KEYS).reset_index(drop=True)
    ids = np.arange(len(df))
    y = df["label"].to_numpy().astype(float)
    k = cfg["top_k"]

    terms = np.array(sorted(set(df["a"]) | set(df["b"]) | set(df["c"])))
    index = {t: i for i, t in enumerate(terms)}
    triads = np.array([[index[t] for t in row] for row in df[KEYS].itertuples(index=False)])

    se, sb = df["engine_mean"].to_numpy(), df["baseline_mean"].to_numpy()
    oe, ob = ranking(se, ids), ranking(sb, ids)
    ones = np.ones(len(df))
    point = metrics(oe, ob, se, sb, y, ones, k)

    rng = np.random.default_rng(cfg["bootstrap_seed"])
    boot = {"candidate": {"lift": [], "auprc_diff": []}, "term_block": {"lift": [], "auprc_diff": []}}
    for _ in range(cfg["bootstrap_n"]):
        for kind, w in (("candidate", candidate_weights(rng, len(df))), ("term_block", term_block_weights(rng, triads, terms))):
            m = metrics(oe, ob, se, sb, y, w, k)
            boot[kind]["lift"].append(m["lift"])
            boot[kind]["auprc_diff"].append(m["auprc_diff"])
    ci = {kind: {m: interval(v, cfg["ci_level"]) for m, v in vals.items()} for kind, vals in boot.items()}

    v1 = all(ci[kind]["lift"]["lower"] > 1 for kind in ci)
    v2 = all(ci[kind]["auprc_diff"]["lower"] > 0 for kind in ci)

    # Per-seed and secondary metrics on the full T2 sample
    per_seed = {}
    for name in ["engine", "baseline"]:
        rows = []
        for seed in model_cfg["seeds"]:
            s = df[f"{name}_seed{seed}"].to_numpy()
            o = ranking(s, ids)
            rows.append({"auprc": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)),
                         **{f"precision_at_{kk}": weighted_precision_at_k(o, y, ones, kk) for kk in [k] + cfg["secondary_k"]}})
        per_seed[name] = {m: {"mean": float(np.mean([r[m] for r in rows])), "sd": float(np.std([r[m] for r in rows], ddof=1))}
                          for m in rows[0]}
    secondary = {"engine_auroc": float(roc_auc_score(y, se)),
                 **{f"engine_precision_at_{kk}": weighted_precision_at_k(oe, y, ones, kk) for kk in cfg["secondary_k"]}}

    result = {
        "meta": {"step": "validation", "variant": variant, "created": datetime.date.today().isoformat(),
                 "git_commit": git_commit(), "evaluation_config": cfg,
                 "scores_sha256": sha256(score_path), "labels_sha256": sha256(label_path)},
        "t2": {"candidates": int(len(df)), "positives": int(y.sum()), "base_rate": float(y.mean())},
        "point": {"precision_at_k": weighted_precision_at_k(oe, y, ones, k), **point},
        "ci": ci,
        "verdict": {"V1": v1, "V2": v2},
        "per_seed": per_seed,
        "secondary": secondary,
    }
    (RESULTS / f"validation_{variant}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"verdict": result["verdict"], "point": result["point"], "ci": ci}, indent=2))


if __name__ == "__main__":
    main()
