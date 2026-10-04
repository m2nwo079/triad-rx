"""Candidate generation recall (secondary metric, design 10.4 and preregistration 4.2).

Usage: python scripts/model/03_candidate_recall.py --timepoint T2 [--variant main]

Realised new triads are all triads that co-occur in at least one label-window paper, never
co-occurred in a build-window paper, and are not made only of tasks. They are tested with the
same null model and BH correction as the labels (over this whole set). Recall is the share of
significant ones that are also candidates.
"""
import argparse
import datetime
import importlib.util
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph")
CANDIDATES = Path("data/derived/candidates")
RESULTS = Path("results")

_spec = importlib.util.spec_from_file_location("labels", Path(__file__).resolve().parents[1] / "graph" / "04_build_labels.py")
labels_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels_mod)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", default="T2", choices=["T1", "T2"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    tp, variant = args.timepoint, args.variant
    if tp == "T2" and not labels_mod.preregistration_frozen(Path("preregistration.md").read_text()):
        sys.exit("T2 recall is locked until the preregistration freeze commit is recorded")

    study = load_study()
    alpha = study["labels"]["fdr"]
    build = pd.read_parquet(HYPERGRAPH / variant / f"{tp}_build.parquet")
    label = pd.read_parquet(HYPERGRAPH / variant / f"{tp}_label.parquet")
    cands = pd.read_parquet(CANDIDATES / variant / f"{tp}.parquet")
    cand_set = set(map(tuple, cands[["a", "b", "c"]].itertuples(index=False)))

    closed = set()
    for ts in build["terms"]:
        ts = sorted(set(ts))
        if len(ts) >= 3:
            closed.update(itertools.combinations(ts, 3))

    observed = Counter()
    for ts in label["terms"]:
        ts = sorted(set(ts))
        for tri in itertools.combinations(ts, 3):
            if tri not in closed and not all(t.startswith("T:") for t in tri):
                observed[tri] += 1

    degree = label["terms"].str.len().to_numpy()
    total_links = int(degree.sum())
    term_docs = Counter(t for ts in label["terms"] for t in set(ts))
    groups = sorted(Counter(degree.tolist()).items())
    degrees = np.array([d for d, _ in groups], dtype=float)
    sizes = np.array([n for _, n in groups], dtype=float)

    triads = list(observed)
    pvalues = np.empty(len(triads))
    for i, tri in enumerate(triads):
        q = np.ones_like(degrees)
        for x in tri:
            q *= np.minimum(1.0, degrees * term_docs[x] / total_links)
        o = observed[tri]
        if o == 1:
            # P(X >= 1) = 1 - prod (1 - q_d)^n_d, computed in log space
            pvalues[i] = -np.expm1(np.sum(sizes * np.log1p(-np.minimum(q, 1 - 1e-16))))
        else:
            pvalues[i] = labels_mod.upper_tail(o, list(zip(sizes.astype(int), q)))
    qvals = labels_mod.benjamini_hochberg(pvalues) if len(triads) else np.array([])
    significant = [tri for tri, qv in zip(triads, qvals) if qv <= alpha]
    hit = sum(1 for tri in significant if tri in cand_set)

    result = {
        "meta": {"step": "candidate_recall", "timepoint": tp, "variant": variant,
                 "created": datetime.date.today().isoformat(), "git_commit": git_commit()},
        "realised_new_triads_observed": len(triads),
        "realised_new_triads_significant": len(significant),
        "significant_in_candidates": hit,
        "candidate_recall": hit / len(significant) if significant else float("nan"),
    }
    (RESULTS / f"candidate_recall_{variant}_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
