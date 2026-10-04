"""Compute P, H, T, R features for one timepoint's triad candidates (design 5.3).

Usage: python scripts/graph/05_build_features.py --timepoint T1 [--variant main]

Every feature uses the build-window hypergraph and build-window abstracts only.
"""
import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import build_window, data_cutoff, load_study

HYPERGRAPH = Path("data/derived/hypergraph")
CANDIDATES = Path("data/derived/candidates")
PAPERS = Path("data/arxiv/papers.parquet")
OUT_DIR = Path("data/derived/features")
RESULTS = Path("results")

# Reuse the candidate script's precision and pair-stability helpers
_spec = importlib.util.spec_from_file_location("cands", Path(__file__).with_name("03_build_candidates.py"))
cands_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cands_mod)


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


def gmean(values) -> float:
    values = [max(v, 0.0) for v in values]
    return float(math.exp(sum(math.log1p(v) for v in values) / len(values)) - 1)


def npmi(c: int, a: int, b: int, n: int) -> float:
    if c == 0:
        return -1.0
    p_ab, p_a, p_b = c / n, a / n, b / n
    if p_ab >= 1:
        return 1.0
    return math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)


def term_vectors(edges: pd.DataFrame, cfg: dict) -> tuple[dict, list]:
    """TF-IDF on build-window abstracts, term vector = mean of its papers, reduced by SVD."""
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"]).set_index("id")
    docs = papers.loc[edges["paper_id"]]
    texts = (docs["title"] + ". " + docs["abstract"]).tolist()
    vectorizer = TfidfVectorizer(min_df=cfg["min_df"], max_df=cfg["max_df"], stop_words="english",
                                 sublinear_tf=True, max_features=cfg["max_features"])
    x = vectorizer.fit_transform(texts)
    terms = sorted({t for ts in edges["terms"] for t in ts})
    index = {t: i for i, t in enumerate(terms)}
    rows, cols = [], []
    for j, ts in enumerate(edges["terms"]):
        for t in set(ts):
            rows.append(index[t])
            cols.append(j)
    incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(terms), len(edges)))
    counts = np.asarray(incidence.sum(axis=1)).ravel()
    means = sparse.diags(1.0 / counts) @ incidence @ x
    svd = TruncatedSVD(n_components=cfg["svd_dims"], random_state=cfg["seed"])
    reduced = svd.fit_transform(means)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.where(norms == 0, 1, norms)
    return {t: reduced[i] for t, i in index.items()}, ["svd_explained_variance", float(svd.explained_variance_ratio_.sum())]


def compute(edges: pd.DataFrame, cands: pd.DataFrame, pi: dict, stability: float, recent_start, vectors: dict) -> pd.DataFrame:
    n_papers = len(edges)
    term_docs, pairs, recent_pairs = Counter(), Counter(), Counter()
    postings = defaultdict(set)
    sizes = edges["terms"].str.len().to_numpy()
    is_recent = (edges["first_date"] >= recent_start).to_numpy()
    for i, ts in enumerate(edges["terms"]):
        ts = sorted(set(ts))
        term_docs.update(ts)
        for t in ts:
            postings[t].add(i)
        for pair in itertools.combinations(ts, 2):
            pairs[pair] += 1
            if is_recent[i]:
                recent_pairs[pair] += 1

    adj = defaultdict(set)
    for (a, b), c in pairs.items():
        if c >= cands_mod.pair_threshold(pi[a], pi[b], stability):
            adj[a].add(b)
            adj[b].add(a)

    def count(a, b):
        return pairs[(a, b) if a < b else (b, a)]

    def recent(a, b):
        return recent_pairs[(a, b) if a < b else (b, a)]

    rows = []
    for a, b, c, kind in cands[["a", "b", "c", "kind"]].itertuples(index=False):
        triad = (a, b, c)
        pair_list = list(itertools.combinations(triad, 2))
        cn = [len(adj[x] & adj[y]) for x, y in pair_list]
        aa = [sum(1 / math.log(len(adj[z])) for z in adj[x] & adj[y] if len(adj[z]) > 1) for x, y in pair_list]
        npm = [npmi(count(x, y), term_docs[x], term_docs[y], n_papers) for x, y in pair_list]
        pc = [count(x, y) for x, y in pair_list]
        logf = [math.log(term_docs[x]) for x in triad]
        related = set()
        for x, y in pair_list:
            related |= postings[x] & postings[y]
        rel_sizes = sizes[list(related)] if related else np.array([])
        growth = []
        for x, y in pair_list:
            r = recent(x, y)
            growth.append((r / 2 + 1) / ((count(x, y) - r) / 3 + 1))
        v = np.stack([vectors[x] for x in triad])
        cos = [float(v[i] @ v[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        rows.append({
            "a": a, "b": b, "c": c,
            "P_aa_min": min(aa), "P_aa_mean": float(np.mean(aa)), "P_aa_gmean": gmean(aa),
            "P_cn_min": min(cn), "P_cn_mean": float(np.mean(cn)), "P_cn_gmean": gmean(cn),
            "P_npmi_min": min(npm), "P_npmi_mean": float(np.mean(npm)), "P_npmi_max": max(npm),
            "P_logfreq_min": min(logf), "P_logfreq_mean": float(np.mean(logf)),
            "H_pairedges_min": min(pc), "H_pairedges_mean": float(np.mean(pc)), "H_pairedges_gmean": gmean(pc),
            "H_triad_cn": len(adj[a] & adj[b] & adj[c]),
            "H_size_weighted_aa": float(sum(1 / math.log(s) for s in rel_sizes if s > 1)),
            "H_related_edge_size_mean": float(rel_sizes.mean()) if len(rel_sizes) else 0.0,
            "H_open_triangle": int(kind == "open_triangle"),
            "T_cos_mean": float(np.mean(cos)), "T_cos_min": min(cos),
            "T_volume": float(max(np.linalg.det(v @ v.T), 0.0)),
            "R_growth_min": min(growth), "R_growth_mean": float(np.mean(growth)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    tp, variant = args.timepoint, args.variant

    study = load_study()
    cfg = study["features"]
    stability = study["vocab"]["r5_rule"]["stability"]
    edges_path = HYPERGRAPH / variant / f"{tp}_build.parquet"
    cand_path = CANDIDATES / variant / f"{tp}.parquet"
    edges = pd.read_parquet(edges_path)
    # Hypergraph dates are stored without a timezone; they are UTC
    edges["first_date"] = pd.to_datetime(edges["first_date"], utc=True)
    cands = pd.read_parquet(cand_path)

    cutoff = data_cutoff() if tp == "applied" else None
    start, end = build_window(tp, study, cutoff)
    # Leakage check: every paper used for features lies inside the build window
    if edges["first_date"].max() > end or edges["first_date"].min() < start:
        sys.exit("hypergraph contains papers outside the build window")
    # Recent part of the build window: the last calendar years for T1/T2, the last years before the cutoff for applied
    if tp == "applied":
        recent_start = end.replace(year=end.year - cfg["recent_years"])
    else:
        recent_start = start.replace(year=end.year - cfg["recent_years"] + 1)

    vectors, svd_info = term_vectors(edges, cfg["text"])
    features = compute(edges, cands, cands_mod.term_pi(tp, variant), stability, recent_start, vectors)

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tp}.parquet"
    features.to_parquet(out)
    numeric = features.drop(columns=["a", "b", "c"])
    result = {
        "meta": {"step": "features", "timepoint": tp, "variant": variant,
                 "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "features_config": cfg, "build_window": [start.isoformat(), end.isoformat()],
                 "recent_start": recent_start.isoformat(),
                 "hypergraph_sha256": sha256(edges_path), "candidates_sha256": sha256(cand_path)},
        "stats": {"candidates": int(len(features)), "features": int(numeric.shape[1]),
                  "nan_cells": int(numeric.isna().sum().sum()), svd_info[0]: svd_info[1]},
    }
    (RESULTS / f"features_{variant}_{tp}.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {out}: {result['stats']}")


if __name__ == "__main__":
    main()
