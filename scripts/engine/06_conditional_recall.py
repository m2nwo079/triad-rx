"""Recall-budget curve of conditional completion (design 18.9 step 2).

Usage: python scripts/engine/06_conditional_recall.py

For each development timepoint (engine_reliability.conditional.timepoints) and seed pair rule
(generator.seed_pairs: "stable" is the first-study pair stability rule, "min_<n>" means at least n
build-window papers), every seed pair (u, v) is completed with every other term w. Triads that
already co-occurred in a build-window paper and triads made only of tasks are dropped, as in the
first study. Each completion is scored from build-window data only:
- npmi_mean: mean nPMI of the two new pairs (u, w) and (v, w) (nPMI as in scripts/graph/05_build_features.py)
- min_count: the smaller build-window count of the two new pairs
- adamic_adar: Adamic-Adar of (u, w) plus that of (v, w) on the stable-pair graph, as in the features
A triad reached from several seed pairs keeps its highest score. Triads are ranked by score with
ties by term ids, and the recall and precision of the second-study labels (and of all realised
new triads) are read at budgets that are multiples of the first-study candidate count.
Labels are used only for this reading. The output is descriptive and is never a verdict.

Output: results/conditional_recall_budget.json
"""
import datetime
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph/main")
LABELS_V2 = Path("data/derived/labels_v2/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
CHUNK = 500


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidates = load("build_candidates", ROOT / "scripts/graph/03_build_candidates.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def cooccurrence(build: pd.DataFrame, index: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pair counts (zero diagonal) and document counts over build-window papers."""
    n = len(index)
    counts = np.zeros((n, n), dtype=np.int64)
    docs = np.zeros(n, dtype=np.int64)
    for ts in build["terms"]:
        idx = np.array(sorted({index[t] for t in ts}))
        docs[idx] += 1
        counts[np.ix_(idx, idx)] += 1
    np.fill_diagonal(counts, 0)
    return counts, docs


def npmi_matrix(counts: np.ndarray, docs: np.ndarray, n_papers: int) -> np.ndarray:
    """Vectorised features.npmi: -1 without co-occurrence, 1 when the pair is in every paper."""
    p_ab = counts / n_papers
    p = docs / n_papers
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log(p_ab / np.outer(p, p)) / -np.log(p_ab)
    value = np.where(p_ab >= 1, 1.0, value)
    return np.where(counts == 0, -1.0, value)


def adamic_adar_matrix(adj: np.ndarray) -> np.ndarray:
    """Sum over common neighbours z of 1 / log(degree z), skipping degree 1 as in the features."""
    deg = adj.sum(axis=1)
    weight = np.zeros(len(deg))
    weight[deg > 1] = 1 / np.log(deg[deg > 1])
    a = adj.astype(float)
    return (a * weight) @ a.T


def triad_codes(i: np.ndarray, j: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    s = np.sort(np.stack([i, j, k], axis=-1), axis=-1).astype(np.int64)
    return (s[..., 0] * n + s[..., 1]) * n + s[..., 2]


def completions(seeds: np.ndarray, mats: dict, is_task: np.ndarray, closed: np.ndarray, n: int):
    """Codes and per-score maxima of all completions of the seed pairs."""
    codes, scores = [], {name: [] for name in mats}
    k = np.arange(n)
    for start in range(0, len(seeds), CHUNK):
        u, v = seeds[start:start + CHUNK, 0][:, None], seeds[start:start + CHUNK, 1][:, None]
        kk = np.broadcast_to(k, (len(u), n))
        keep = (kk != u) & (kk != v) & ~(is_task[u] & is_task[v] & is_task[kk])
        code = triad_codes(np.broadcast_to(u, kk.shape), np.broadcast_to(v, kk.shape), kk, n)
        keep &= ~np.isin(code, closed)
        codes.append(code[keep])
        for name, (m, combine) in mats.items():
            scores[name].append(combine(m[u.ravel()], m[v.ravel()])[keep].astype(np.float64))
    codes = np.concatenate(codes)
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    first = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return codes[first], {name: np.maximum.reduceat(np.concatenate(s)[order], first) for name, s in scores.items()}


def recall_at(codes: np.ndarray, score: np.ndarray, positives: np.ndarray, total: int, budgets: list[int]) -> list[dict]:
    """Rank by score (ties by code) and read recall over `total` positives and precision at each budget.

    `total` also counts positives that no candidate can reach (terms outside the build window).
    """
    order = np.lexsort((codes, -score))
    hits = np.cumsum(np.isin(codes[order], positives))
    out = []
    for b in budgets:
        top = min(b, len(codes))
        h = int(hits[top - 1]) if top else 0
        out.append({"budget": b, "capped": b > len(codes), "hits": h,
                    "recall": h / total if total else float("nan"),
                    "precision": h / top if top else float("nan")})
    return out


def label_codes(rows: pd.DataFrame, index: dict) -> np.ndarray:
    """Codes of labelled triads whose three terms all occur in the build window."""
    rows = rows[rows[KEYS].isin(list(index)).all(axis=1)]
    ids = [rows[c].map(index).to_numpy(dtype=np.int64) for c in KEYS]
    return np.unique(triad_codes(*ids, len(index)))


def seed_pairs(rule: str, counts: np.ndarray, stable: np.ndarray) -> np.ndarray:
    mask = stable if rule == "stable" else counts >= int(rule.removeprefix("min_"))
    i, j = np.nonzero(np.triu(mask, 1))
    return np.stack([i, j], axis=1)


def prepare(build: pd.DataFrame, tp: str, stability: float) -> dict:
    """Term index, pair statistics, stable pairs, closed triads and score matrices of a build window."""
    terms = sorted({t for ts in build["terms"] for t in ts})
    index = {t: i for i, t in enumerate(terms)}
    n = len(terms)
    counts, docs = cooccurrence(build, index)
    pi = candidates.term_pi(tp, "main")
    pis = np.array([pi[t] for t in terms])
    stable = (counts > 0) & (counts >= np.ceil(-math.log(1 - stability) / np.outer(pis, pis)))
    closed = set()
    for ts in build["terms"]:
        idx = sorted({index[t] for t in ts})
        closed.update(itertools.combinations(idx, 3))
    closed = np.sort(np.array([(a * n + b) * n + c for a, b, c in closed], dtype=np.int64))
    return {
        "terms": terms, "index": index, "n": n, "counts": counts, "docs": docs, "stable": stable, "closed": closed,
        "is_task": np.array([t.startswith("T:") for t in terms]),
        "scores": {"npmi_mean": (npmi_matrix(counts, docs, len(build)), lambda a, b: (a + b) / 2),
                   "min_count": (counts, np.minimum),
                   "adamic_adar": (adamic_adar_matrix(stable), np.add)},
    }


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["conditional"]
    gen = cfg["generator"]
    stability = study["vocab"]["r5_rule"]["stability"]
    out, hashes = {}, {}
    for tp in cfg["timepoints"]:
        paths = {"build": HYPERGRAPH / f"{tp}_build.parquet", "labels_v2": LABELS_V2 / f"{tp}.parquet"}
        hashes[tp] = {k: evaluate.sha256(v) for k, v in paths.items()}
        g = prepare(pd.read_parquet(paths["build"]), tp, stability)
        index, n, counts, stable, is_task, closed = g["index"], g["n"], g["counts"], g["stable"], g["is_task"], g["closed"]
        mats = {name: g["scores"][name] for name in gen["scores"]}

        labels = pd.read_parquet(paths["labels_v2"])
        sets = {"label_v2": {"total": int(labels["label_v2"].sum()), "codes": label_codes(labels[labels["label_v2"]], index)},
                "realised_all": {"total": len(labels), "codes": label_codes(labels, index)}}

        first = json.loads((RESULTS / f"candidates_main_{tp}.json").read_text())["stats"]["candidates"]
        budgets = [m * first for m in gen["budget_multiples_of_first_candidates"]]
        out[tp] = {"terms": n, "first_study_candidates": first, "budgets": budgets, "by_seed_rule": {}}
        for rule in gen["seed_pairs"]:
            seeds = seed_pairs(rule, counts, stable)
            codes, scores = completions(seeds, mats, is_task, closed, n)
            entry = {"seed_pairs": int(len(seeds)), "candidates": int(len(codes)), "sets": {}}
            for name, s in sets.items():
                reached = int(np.isin(s["codes"], codes).sum())
                entry["sets"][name] = {
                    "positives": s["total"], "reached_by_all_candidates": reached,
                    "ceiling_recall": reached / s["total"] if s["total"] else float("nan"),
                    "by_score": {sc: recall_at(codes, v, s["codes"], s["total"], budgets) for sc, v in scores.items()},
                }
            out[tp]["by_seed_rule"][rule] = entry
            for sc in scores:
                line = " ".join(f"x{m}:{r['recall']:.3f}" for m, r in
                                zip(gen["budget_multiples_of_first_candidates"], entry["sets"]["label_v2"]["by_score"][sc]))
                print(f"{tp} {rule:6s} {sc:11s} seeds={len(seeds)} cands={len(codes)} label_v2 recall {line}")

    result = {
        "meta": {"step": "conditional_recall_budget", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "conditional_config": cfg, "stability": stability,
                 "inputs_sha256": hashes,
                 "note": "Descriptive development diagnostics; never a verdict. Tune on generator.tune_timepoint, "
                         "read generator.confirm_timepoint only to confirm."},
        "timepoints": out,
    }
    (RESULTS / "conditional_recall_budget.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
