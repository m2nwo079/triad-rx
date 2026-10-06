"""Coverage of conditional completion (design 18.9 step 2): the recall ceiling of adding one term to a pair.

Usage: python scripts/engine/05_conditional_coverage.py

For each development timepoint (engine_reliability.conditional.timepoints) and each realised
triad set (second-study positives from scripts/engine/04_label_v2.py, and all realised
significant new triads as in the first-study recall), counts how many of the triad's three pairs
already co-occurred in the build window under each pair criterion: at least n papers for every n
in `pair_min_counts`, and the first-study pair stability rule (scripts/graph/03_build_candidates.py,
reused through importlib). A triad with at least one such pair can be reached by a generator that
completes realised pairs, so that share is the recall ceiling. The recall of the first-study
candidate generator on the same sets is reported next to it.
The output is descriptive for engine development and is never a verdict.

Output: results/conditional_coverage.json
"""
import datetime
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph/main")
LABELS_V2 = Path("data/derived/labels_v2/main")
CANDIDATES = Path("data/derived/candidates/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidates = load("build_candidates", ROOT / "scripts/graph/03_build_candidates.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def pair_counts(build: pd.DataFrame) -> Counter:
    counts = Counter()
    for ts in build["terms"]:
        counts.update(itertools.combinations(sorted(set(ts)), 2))
    return counts


def coverage(triads: list[tuple], qualifies, cand_set: set) -> dict:
    """Distribution of qualifying pairs per triad, the share with at least one, and candidate recall."""
    n_pairs = Counter(sum(qualifies(p) for p in itertools.combinations(t, 2)) for t in triads)
    n = len(triads)
    return {
        "triads": n,
        "by_qualifying_pairs": {str(k): n_pairs.get(k, 0) for k in range(4)},
        "coverage": (n - n_pairs.get(0, 0)) / n if n else float("nan"),
        "candidate_recall": sum(t in cand_set for t in triads) / n if n else float("nan"),
    }


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["conditional"]
    stability = study["vocab"]["r5_rule"]["stability"]
    out = {}
    hashes = {}
    for tp in cfg["timepoints"]:
        paths = {"build": HYPERGRAPH / f"{tp}_build.parquet", "labels_v2": LABELS_V2 / f"{tp}.parquet",
                 "candidates": CANDIDATES / f"{tp}.parquet"}
        hashes[tp] = {k: evaluate.sha256(v) for k, v in paths.items()}
        counts = pair_counts(pd.read_parquet(paths["build"]))
        labels = pd.read_parquet(paths["labels_v2"])
        cand_set = set(pd.read_parquet(paths["candidates"])[KEYS].itertuples(index=False, name=None))
        pi = candidates.term_pi(tp, "main")

        criteria = {f"min_{k}": (lambda p, k=k: counts[p] >= k) for k in cfg["pair_min_counts"]}
        if cfg["stable_pairs"]:
            criteria["stable"] = lambda p: counts[p] > 0 and counts[p] >= candidates.pair_threshold(pi[p[0]], pi[p[1]], stability)
        sets = {"label_v2": list(labels.loc[labels["label_v2"], KEYS].itertuples(index=False, name=None)),
                "realised_all": list(labels[KEYS].itertuples(index=False, name=None))}
        out[tp] = {name: {crit: coverage(triads, q, cand_set) for crit, q in criteria.items()}
                   for name, triads in sets.items()}

    result = {
        "meta": {"step": "conditional_coverage", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "conditional_config": cfg, "stability": stability,
                 "inputs_sha256": hashes,
                 "note": "Descriptive development diagnostics; never a verdict."},
        "timepoints": out,
    }
    (RESULTS / "conditional_coverage.json").write_text(json.dumps(result, indent=2))
    for tp, sets in out.items():
        for name, crits in sets.items():
            for crit, r in crits.items():
                print(f"{tp} {name:12s} {crit:7s} n={r['triads']:6d} coverage {r['coverage']:.3f} "
                      f"candidate recall {r['candidate_recall']:.3f}")


if __name__ == "__main__":
    main()
