"""Candidates of the third-study engine on the v3 inputs (matching preregistration 6.1-6.2).

Usage: python scripts/matching/14_engine_candidates.py --timepoint T1|T2|R2|R3|T3

Inputs are the v3 vocabulary (vocab/timepoints/<tp>_v3_final.json) and the v3 build-window hypergraph
(data/derived/hypergraph/v3/<tp>_build.parquet). Only build-window data are read; T3 labels are never read.

1. Term precision pi: the per-term values of the v3 rebuild's rule R5 (the `r5` records of the vocabulary,
   the minimum over the final stratum lower bounds of results/precision_T1.json for the term's surfaces).
   The second-study pi (first-study M1 and vocabulary) is not used.
2. First-study rule on v3 (open triangles and two-hop triads, scripts/graph/03_build_candidates.py reused
   through importlib) with this pi. Its candidate count is the base of the generator budget, as the
   first-study candidate count was in the second study.
3. Generator: engine_reliability.conditional.generator.selected (seed pair rule and third-term score), as in
   scripts/engine/06_conditional_recall.py and 07_conditional_candidates.py (reused through importlib) but
   with the pi of step 1. The top `diagnosis_budget_multiple` times the step 2 count are kept, ranked by score
   with ties by term ids.
Triads holding an excluded pair of the vocabulary (hierarchy or containment, preregistration 5.1) are dropped
from both sets before any count or budget. When the timepoint has v3 relation labels
(data/derived/labels_v3/relation/<tp>.parquet) and is not T3, `label_v3` and `realised` are attached for
evaluation only.

Outputs (not committed): data/derived/candidates_v3/{first,generator}/<tp>.parquet
Summary: results/engine_v3_candidates_<tp>.json (counts and hashes)
"""
import argparse
import datetime
import importlib.util
import json
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment

TIMEPOINTS = Path("vocab/timepoints")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
LABELS = Path("data/derived/labels_v3/relation")
OUT_DIR = Path("data/derived/candidates_v3")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
TIMEPOINT_CHOICES = ["T1", "T2", "R2", "R3", "T3"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recall = load("conditional_recall", ROOT / "scripts/engine/06_conditional_recall.py")
gen_v2 = load("conditional_candidates", ROOT / "scripts/engine/07_conditional_candidates.py")
first_rule = load("build_candidates", ROOT / "scripts/graph/03_build_candidates.py")


def term_pi_v3(vocab: dict) -> dict:
    """Per-term pi of the v3 rebuild's R5 for the final terms."""
    final = {t["id"] for t in vocab["terms"]}
    pi = {r["term_id"]: r["pi"] for r in vocab["r5"] if r["term_id"] in final}
    missing = final - set(pi)
    if missing:
        raise ValueError(f"{len(missing)} final terms have no R5 pi")
    return pi


def prepare(build: pd.DataFrame, pi: dict, stability: float) -> dict:
    """recall.prepare with the given pi instead of the second-study pi."""
    terms = sorted({t for ts in build["terms"] for t in ts})
    index = {t: i for i, t in enumerate(terms)}
    n = len(terms)
    counts, docs = recall.cooccurrence(build, index)
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
        "scores": {"npmi_mean": (recall.npmi_matrix(counts, docs, len(build)), lambda a, b: (a + b) / 2),
                   "min_count": (counts, np.minimum),
                   "adamic_adar": (recall.adamic_adar_matrix(stable), np.add)},
    }


def excluded_matrix(index: dict, excluded: set) -> np.ndarray:
    """Symmetric boolean matrix of excluded pairs among the build-window terms."""
    mat = np.zeros((len(index), len(index)), dtype=bool)
    for a, b in excluded:
        if a in index and b in index:
            mat[index[a], index[b]] = mat[index[b], index[a]] = True
    return mat


def drop_excluded_codes(codes: np.ndarray, score: np.ndarray, mat: np.ndarray, n: int):
    """Codes (and scores) of triads without an excluded pair."""
    i, j, k = codes // (n * n), codes // n % n, codes % n
    keep = ~(mat[i, j] | mat[i, k] | mat[j, k])
    return codes[keep], score[keep], int((~keep).sum())


def first_study_candidates(build: pd.DataFrame, pi: dict, stability: float, ratio: float, excluded: set):
    """First-study rule on the v3 inputs, then excluded pairs dropped."""
    rows, stats = first_rule.build(build, pi, stability, ratio)
    frame = pd.DataFrame(rows)
    held = frame[KEYS].apply(lambda r: holds_excluded(tuple(r), excluded), axis=1)
    return frame[~held].reset_index(drop=True), stats, int(held.sum())


def holds_excluded(triad: tuple, excluded: set) -> bool:
    a, b, c = triad
    return any(tuple(sorted(p)) in excluded for p in ((a, b), (a, c), (b, c)))


def generator_candidates(g: dict, rule: str, score: str, budget: int, mat: np.ndarray):
    """Top `budget` completions without excluded pairs, with kind, score and rank."""
    seeds = recall.seed_pairs(rule, g["counts"], g["stable"])
    codes, scores = recall.completions(seeds, {score: g["scores"][score]}, g["is_task"], g["closed"], g["n"])
    codes, kept_score, dropped = drop_excluded_codes(codes, scores[score], mat, g["n"])
    return gen_v2.to_frame(g, codes, kept_score, budget), int(len(seeds)), int(len(codes)), dropped


def attach_labels(cands: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    flags = labels[KEYS + ["label_v3"]].assign(realised=True)
    out = cands.merge(flags, on=KEYS, how="left", validate="one_to_one")
    # Unmatched rows are NaN; comparing with True avoids pandas' silent downcasting of fillna
    return out.assign(label_v3=out["label_v3"].eq(True), realised=out["realised"].eq(True))


def label_summary(cands: pd.DataFrame) -> dict:
    return {"label_v3_positive": int(cands["label_v3"].sum()), "realised": int(cands["realised"].sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=TIMEPOINT_CHOICES)
    tp = parser.parse_args().timepoint
    study = load_study()
    sel = study["engine_reliability"]["conditional"]["generator"]
    stability = study["vocab"]["r5_rule"]["stability"]
    ratio = study["candidates"]["two_hop"]["n_ratio_to_open_triangles"]

    vocab_path, build_path = TIMEPOINTS / f"{tp}_v3_final.json", HYPERGRAPH / f"{tp}_build.parquet"
    vocab = json.loads(vocab_path.read_text())
    build = pd.read_parquet(build_path)
    pi = term_pi_v3(vocab)
    excluded = {tuple(sorted(p)) for kind in ("hierarchy", "containment") for p in vocab["excluded_pairs"][kind]}

    first, first_stats, first_dropped = first_study_candidates(build, pi, stability, ratio, excluded)
    budget = sel["diagnosis_budget_multiple"] * len(first)
    g = prepare(build, pi, stability)
    mat = excluded_matrix(g["index"], excluded)
    cands, n_seeds, n_completions, gen_dropped = generator_candidates(
        g, sel["selected"]["seed_pairs"], sel["selected"]["score"], budget, mat)

    # Labels are attached for evaluation only and never for T3
    label_path = LABELS / f"{tp}.parquet"
    labelled = tp != "T3" and label_path.exists()
    if labelled:
        labels = pd.read_parquet(label_path)
        first, cands = attach_labels(first, labels), attach_labels(cands, labels)

    for name, frame in (("first", first), ("generator", cands)):
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(OUT_DIR / name / f"{tp}.parquet", index=False)
    inputs = {"vocab": vocab_path, "build": build_path, **({"labels": label_path} if labelled else {})}
    result = {
        "meta": {"step": "engine_v3_candidates", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": recall.evaluate.git_commit(), "environment": runtime_environment(),
                 "generator": sel["selected"], "diagnosis_budget_multiple": sel["diagnosis_budget_multiple"],
                 "stability": stability, "two_hop_ratio": ratio,
                 "inputs_sha256": {k: recall.evaluate.sha256(v) for k, v in inputs.items()},
                 "outputs_sha256": {name: recall.evaluate.sha256(OUT_DIR / name / f"{tp}.parquet") for name in ("first", "generator")}},
        "terms": g["n"],
        "pi_by_value": {str(round(v, 6)): int(c) for v, c in zip(*np.unique(list(pi.values()), return_counts=True))},
        "first_study_rule": {"rule_stats": first_stats, "candidates": int(len(first)), "dropped_excluded_pairs": first_dropped,
                             "kind": first["kind"].value_counts().to_dict(),
                             **({"labels": label_summary(first)} if labelled else {})},
        "generator": {"seed_pairs": n_seeds, "completions_without_excluded": n_completions,
                      "dropped_excluded_pairs": gen_dropped, "budget": budget, "candidates": int(len(cands)),
                      "kind": cands["kind"].value_counts().to_dict(),
                      **({"labels": label_summary(cands)} if labelled else {})},
        "labels_attached": labelled,
    }
    (RESULTS / f"engine_v3_candidates_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
