"""Candidates of the selected conditional completion generator (design 18.9 steps 2 and 3).

Usage: python scripts/engine/07_conditional_candidates.py

For each development timepoint, the generator in engine_reliability.conditional.generator.selected
(seed pair rule and third-term score) completes the seed pairs as in
scripts/engine/06_conditional_recall.py (reused through importlib) and keeps the top
`diagnosis_budget_multiple` times the first-study candidate count, ranked by score with ties by
term ids. `kind` is "open_triangle" when all three pairs are stable pairs (the first-study meaning
of the H_open_triangle feature) and "completion" otherwise. Labels are attached for evaluation
only: label_v2 (second-study label) and realised (significant realised new triad, first-study
recall definition).

Outputs
- data/derived/candidates_v2/main/<tp>.parquet (not committed): a, b, c, kind, gen_score, gen_rank, label_v2, realised
- results/candidates_v2_<tp>.json: counts and hashes
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
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph/main")
LABELS_V2 = Path("data/derived/labels_v2/main")
OUT_DIR = Path("data/derived/candidates_v2/main")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recall = load("conditional_recall", ROOT / "scripts/engine/06_conditional_recall.py")


def top_candidates(g: dict, rule: str, score: str, budget: int) -> pd.DataFrame:
    """Top `budget` completions with terms, kind, score and rank."""
    seeds = recall.seed_pairs(rule, g["counts"], g["stable"])
    codes, scores = recall.completions(seeds, {score: g["scores"][score]}, g["is_task"], g["closed"], g["n"])
    return to_frame(g, codes, scores[score], budget)


def to_frame(g: dict, codes: np.ndarray, score: np.ndarray, budget: int) -> pd.DataFrame:
    """Top `budget` triad codes by score (ties by code) as terms, kind, score and rank."""
    order = np.lexsort((codes, -score))[:budget]
    n = g["n"]
    i, j, k = codes[order] // (n * n), codes[order] // n % n, codes[order] % n
    terms = np.array(g["terms"])
    st = g["stable"]
    return pd.DataFrame({
        "a": terms[i], "b": terms[j], "c": terms[k],
        "kind": np.where(st[i, j] & st[i, k] & st[j, k], "open_triangle", "completion"),
        "gen_score": score[order], "gen_rank": np.arange(1, len(order) + 1),
    })


def attach_labels(cands: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    flags = labels[KEYS + ["label_v2"]].assign(realised=True)
    out = cands.merge(flags, on=KEYS, how="left", validate="one_to_one")
    # Unmatched rows are NaN; comparing with True avoids pandas' silent downcasting of fillna
    return out.assign(label_v2=out["label_v2"].eq(True), realised=out["realised"].eq(True))


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["conditional"]
    gen = cfg["generator"]
    sel = gen["selected"]
    stability = study["vocab"]["r5_rule"]["stability"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tp in cfg["timepoints"]:
        paths = {"build": HYPERGRAPH / f"{tp}_build.parquet", "labels_v2": LABELS_V2 / f"{tp}.parquet"}
        g = recall.prepare(pd.read_parquet(paths["build"]), tp, stability)
        first = json.loads((RESULTS / f"candidates_main_{tp}.json").read_text())["stats"]["candidates"]
        budget = gen["diagnosis_budget_multiple"] * first
        cands = attach_labels(top_candidates(g, sel["seed_pairs"], sel["score"], budget), pd.read_parquet(paths["labels_v2"]))
        out = OUT_DIR / f"{tp}.parquet"
        cands.to_parquet(out, index=False)
        result = {
            "meta": {"step": "candidates_v2", "timepoint": tp, "created": datetime.date.today().isoformat(),
                     "git_commit": recall.evaluate.git_commit(), "generator": gen, "stability": stability,
                     "inputs_sha256": {k: recall.evaluate.sha256(v) for k, v in paths.items()},
                     "output_sha256": recall.evaluate.sha256(out)},
            "budget": budget,
            "candidates": int(len(cands)),
            "kind": cands["kind"].value_counts().to_dict(),
            "label_v2_positive": int(cands["label_v2"].sum()),
            "realised": int(cands["realised"].sum()),
        }
        (RESULTS / f"candidates_v2_{tp}.json").write_text(json.dumps(result, indent=2))
        print(tp, {k: v for k, v in result.items() if k != "meta"})


if __name__ == "__main__":
    main()
