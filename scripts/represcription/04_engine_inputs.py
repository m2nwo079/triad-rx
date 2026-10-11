"""Applied-mode candidates and features for the represcription (represcription preregistration 3).

Usage: python scripts/represcription/04_engine_inputs.py

Candidates use the third-study functions (scripts/matching/14_engine_candidates.py) on the applied v3 vocabulary
and build hypergraph: the first-study rule on the same inputs gives the budget base, and the selected
conditional generator keeps its top `represcription.engine.budget_multiple` times that count; excluded pairs are
dropped from both sets before any count. No labels exist for the applied mode.
Features use the third-study feature table (scripts/matching/15_engine_features.py, first-study features)
with the applied build window (design 4.4) and the first-study applied recent window: the last
`features.recent_years` years before the cutoff (scripts/graph/05_build_features.py). The third-study rule for
calendar-year timepoints would cut the recent window of a September-to-September window to one year.

Outputs (not committed): data/derived/candidates_v3/{first,generator}/applied.parquet, data/derived/features_v3/applied.parquet
Summaries: results/engine_v3_candidates_applied.json, results/engine_v3_features_applied.json
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
from triadrx.study import build_window, data_cutoff, load_study, runtime_environment

TP = "applied"
TIMEPOINTS = Path("vocab/timepoints")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
CANDIDATES = Path("data/derived/candidates_v3")
FEATURES = Path("data/derived/features_v3")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
feats_v3 = load("engine_features", ROOT / "scripts/matching/15_engine_features.py")
sha256, git_commit = cands_v3.recall.evaluate.sha256, cands_v3.recall.evaluate.git_commit


def recent_start_applied(end: datetime.datetime, years: int) -> datetime.datetime:
    """First-study applied recent window: the last `years` years before the cutoff."""
    return end.replace(year=end.year - years)


def main() -> None:
    study = load_study()
    sel = study["engine_reliability"]["conditional"]["generator"]
    multiple = study["represcription"]["engine"]["budget_multiple"]
    stability = study["vocab"]["r5_rule"]["stability"]
    ratio = study["candidates"]["two_hop"]["n_ratio_to_open_triangles"]
    vocab_path, build_path = TIMEPOINTS / f"{TP}_v3_final.json", HYPERGRAPH / f"{TP}_build.parquet"
    vocab = json.loads(vocab_path.read_text())
    build = pd.read_parquet(build_path)
    pi = cands_v3.term_pi_v3(vocab)
    excluded = {tuple(sorted(p)) for kind in ("hierarchy", "containment") for p in vocab["excluded_pairs"][kind]}

    first, first_stats, first_dropped = cands_v3.first_study_candidates(build, pi, stability, ratio, excluded)
    budget = multiple * len(first)
    g = cands_v3.prepare(build, pi, stability)
    mat = cands_v3.excluded_matrix(g["index"], excluded)
    cands, n_seeds, n_completions, gen_dropped = cands_v3.generator_candidates(
        g, sel["selected"]["seed_pairs"], sel["selected"]["score"], budget, mat)
    for name, frame in (("first", first), ("generator", cands)):
        (CANDIDATES / name).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(CANDIDATES / name / f"{TP}.parquet", index=False)
    cand_result = {
        "meta": {"step": "engine_v3_candidates", "timepoint": TP, "created": datetime.date.today().isoformat(),
                 "git_commit": git_commit(), "environment": runtime_environment(), "generator": sel["selected"],
                 "budget_multiple": multiple, "stability": stability, "two_hop_ratio": ratio,
                 "inputs_sha256": {"vocab": sha256(vocab_path), "build": sha256(build_path)},
                 "outputs_sha256": {name: sha256(CANDIDATES / name / f"{TP}.parquet") for name in ("first", "generator")}},
        "terms": g["n"],
        "pi_by_value": {str(round(v, 6)): int(c) for v, c in zip(*np.unique(list(pi.values()), return_counts=True))},
        "first_study_rule": {"rule_stats": first_stats, "candidates": int(len(first)), "dropped_excluded_pairs": first_dropped,
                             "kind": first["kind"].value_counts().to_dict()},
        "generator": {"seed_pairs": n_seeds, "completions_without_excluded": n_completions, "dropped_excluded_pairs": gen_dropped,
                      "budget": budget, "candidates": int(len(cands)), "kind": cands["kind"].value_counts().to_dict()},
    }
    (RESULTS / f"engine_v3_candidates_{TP}.json").write_text(json.dumps(cand_result, indent=2))

    cfg = study["features"]
    edges = build.copy()
    edges["first_date"] = pd.to_datetime(edges["first_date"], utc=True)
    start, end = build_window(TP, study, data_cutoff())
    feats_v3.check_window(edges, start, end)
    recent_start = recent_start_applied(end, cfg["recent_years"])
    table, svd_info = feats_v3.feature_table(edges, cands, pi, stability, recent_start, cfg["text"])
    if len(table) != len(cands):
        sys.exit(f"{len(cands) - len(table)} candidates have no features")
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / f"{TP}.parquet"
    table.to_parquet(out, index=False)
    names = [c for c in table.columns if c not in feats_v3.CANDIDATE_COLUMNS]
    feat_result = {
        "meta": {"step": "engine_v3_features", "timepoint": TP, "created": datetime.date.today().isoformat(),
                 "git_commit": git_commit(), "environment": runtime_environment(),
                 "build_window": [start.isoformat(), end.isoformat()], "recent_start": recent_start.isoformat(),
                 "recent_rule": "first-study applied: recent_years before the cutoff", "stability": stability, "text": cfg["text"],
                 "inputs_sha256": {"vocab": sha256(vocab_path), "build": sha256(build_path),
                                   "candidates": sha256(CANDIDATES / "generator" / f"{TP}.parquet")},
                 "output_sha256": sha256(out)},
        "candidates": int(len(table)),
        "features": names,
        "matches_feature_sets": sorted(names) == sorted(f for s in cfg["sets"].values() for f in s),
        svd_info[0]: svd_info[1],
        "missing_values": int(table[names].isna().sum().sum()),
    }
    (RESULTS / f"engine_v3_features_{TP}.json").write_text(json.dumps(feat_result, indent=2))
    print(json.dumps({"first_study_rule": cand_result["first_study_rule"]["candidates"], "budget": budget,
                      "generator": cand_result["generator"]["candidates"], "features": len(names),
                      "missing_values": feat_result["missing_values"], "recent_start": feat_result["meta"]["recent_start"]}, indent=2))


if __name__ == "__main__":
    main()
