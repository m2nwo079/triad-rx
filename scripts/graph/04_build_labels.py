"""Label triad candidates with the triadic null model on the label-window hypergraph (design 5.4).

Usage: python scripts/graph/04_build_labels.py --timepoint T1 [--variant main]

For each candidate (a, b, c) the observed count is the number of label-window papers that
contain all three terms. Under the null model a paper p contains term x with probability
P_px = min(1, w_p * w_x / N); the count is Poisson-binomial over papers. Papers with the
same degree w_p share one probability, so the exact distribution is a convolution of
binomials, one per degree. Upper-tail p-values are adjusted with Benjamini-Hochberg.
The T2 label window is locked until the preregistration freeze commit is recorded.
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph")
CANDIDATES = Path("data/derived/candidates")
PREREG = Path("preregistration.md")
OUT_DIR = Path("data/derived/labels")
RESULTS = Path("results")


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


def preregistration_frozen(text: str) -> bool:
    match = re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M)
    return bool(match) and match.group(1).strip() not in {"", "미정"}


def binomial_pmf(n: int, q: float, upto: int) -> np.ndarray:
    """P(X = k) for k = 0..upto under Binomial(n, q), computed in log space."""
    k = np.arange(min(n, upto) + 1)
    if q <= 0:
        out = np.zeros(upto + 1)
        out[0] = 1.0
        return out
    if q >= 1:
        out = np.zeros(upto + 1)
        if n <= upto:
            out[n] = 1.0
        return out
    log_coef = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    pmf = np.exp(log_coef + k * math.log(q) + (n - k) * math.log1p(-q))
    out = np.zeros(upto + 1)
    out[: len(pmf)] = pmf
    return out


def upper_tail(observed: int, groups: list[tuple[int, float]]) -> float:
    """P(X >= observed) for X = sum of independent Binomial(n_d, q_d)."""
    if observed <= 0:
        return 1.0
    upto = observed - 1
    dist = np.zeros(upto + 1)
    dist[0] = 1.0
    for n, q in groups:
        dist = np.convolve(dist, binomial_pmf(n, q, upto))[: upto + 1]
    return float(max(0.0, 1.0 - dist.sum()))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    tp, variant = args.timepoint, args.variant
    if tp == "T2" and not preregistration_frozen(PREREG.read_text()):
        sys.exit("T2 labels are locked until the preregistration freeze commit is recorded")

    study = load_study()
    alpha = study["labels"]["fdr"]
    label_path = HYPERGRAPH / variant / f"{tp}_label.parquet"
    cand_path = CANDIDATES / variant / f"{tp}.parquet"
    edges = pd.read_parquet(label_path)
    cands = pd.read_parquet(cand_path)

    # Bipartite paper-term statistics of the label window
    degree = edges["terms"].str.len().to_numpy()
    total_links = int(degree.sum())
    term_docs = Counter(t for terms in edges["terms"] for t in set(terms))
    degree_groups = sorted(Counter(degree.tolist()).items())
    postings = defaultdict(set)
    for i, terms in enumerate(edges["terms"]):
        for t in terms:
            postings[t].add(i)

    observed, expected, pvalues = [], [], []
    for a, b, c in cands[["a", "b", "c"]].itertuples(index=False):
        o = len(postings[a] & postings[b] & postings[c])
        groups = []
        e = 0.0
        for d, n_d in degree_groups:
            q = 1.0
            for x in (a, b, c):
                q *= min(1.0, d * term_docs.get(x, 0) / total_links)
            groups.append((n_d, q))
            e += n_d * q
        observed.append(o)
        expected.append(e)
        pvalues.append(upper_tail(o, groups))

    p = np.array(pvalues)
    qvals = benjamini_hochberg(p)
    labels = cands.copy()
    labels["observed"] = observed
    labels["expected"] = expected
    labels["p_value"] = p
    labels["q_value"] = qvals
    labels["label"] = (qvals <= alpha).astype(int)

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tp}.parquet"
    labels.to_parquet(out)
    positives = labels[labels["label"] == 1]
    stats = {
        "candidates": int(len(labels)),
        "observed_ge1": int((labels["observed"] >= 1).sum()),
        "positives": int(len(positives)),
        "positive_rate": float(len(positives) / len(labels)) if len(labels) else 0.0,
        "positives_by_kind": positives["kind"].value_counts().to_dict(),
        "candidates_by_kind": labels["kind"].value_counts().to_dict(),
        "label_window_papers": int(len(edges)),
        "label_window_links": total_links,
    }
    result = {
        "meta": {"step": "labels", "timepoint": tp, "variant": variant,
                 "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "labels_config": study["labels"],
                 "label_hypergraph_sha256": sha256(label_path), "candidates_sha256": sha256(cand_path)},
        "stats": stats,
    }
    (RESULTS / f"labels_{variant}_{tp}.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {out}: {stats}")


if __name__ == "__main__":
    main()
