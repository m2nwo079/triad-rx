"""First-study features of the third-study candidates on the v3 inputs (matching preregistration 6.2).

Usage: python scripts/matching/15_engine_features.py --timepoint T1|T2|R2|R3|T3

The candidates are the generator candidates of scripts/matching/14_engine_candidates.py. The features are the
first-study P, H, T and R features (scripts/graph/05_build_features.py, reused through importlib, definitions
unchanged) computed from the v3 build-window hypergraph and build-window abstracts only, with the v3 term pi
(the R5 records of the v3 vocabulary, as in 14_engine_candidates.py) for the stable-pair graph. As in the second
study, the hypergraph must lie inside the build window and the recent window is the last `features.recent_years`.

Labels are not written here: the output keeps the triad, the generator columns and the features, so the labels
are attached at evaluation from the label files and never come from an earlier candidate run.

Output (not committed): data/derived/features_v3/<tp>.parquet
Summary: results/engine_v3_features_<tp>.json (counts and hashes)
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import build_window, load_study, runtime_environment

TIMEPOINTS = Path("vocab/timepoints")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
CANDIDATES = Path("data/derived/candidates_v3/generator")
OUT_DIR = Path("data/derived/features_v3")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
CANDIDATE_COLUMNS = KEYS + ["kind", "gen_score", "gen_rank"]
TIMEPOINT_CHOICES = ["T1", "T2", "R2", "R3", "T3"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = load("build_features", ROOT / "scripts/graph/05_build_features.py")
cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
prep = load("timepoint_prep", ROOT / "scripts/engine/14_timepoint_prep.py")


def check_window(edges: pd.DataFrame, start, end) -> None:
    if edges["first_date"].max() > end or edges["first_date"].min() < start:
        sys.exit("hypergraph contains papers outside the build window")


def feature_table(edges: pd.DataFrame, cands: pd.DataFrame, pi: dict, stability: float,
                  recent_start, text_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Candidates (generator columns only) joined with their first-study features."""
    vectors, svd_info = features.term_vectors(edges, text_cfg)
    feats = features.compute(edges, cands, pi, stability, recent_start, vectors)
    table = cands[CANDIDATE_COLUMNS].merge(feats, on=KEYS, validate="one_to_one")
    return table.sort_values(KEYS).reset_index(drop=True), svd_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=TIMEPOINT_CHOICES)
    tp = parser.parse_args().timepoint
    study = prep.merged_study(load_study())
    cfg = study["features"]
    stability = study["vocab"]["r5_rule"]["stability"]

    vocab_path, build_path, cand_path = TIMEPOINTS / f"{tp}_v3_final.json", HYPERGRAPH / f"{tp}_build.parquet", CANDIDATES / f"{tp}.parquet"
    edges = pd.read_parquet(build_path)
    edges["first_date"] = pd.to_datetime(edges["first_date"], utc=True)
    start, end = build_window(tp, study)
    check_window(edges, start, end)
    recent_start = start.replace(year=end.year - cfg["recent_years"] + 1)
    pi = cands_v3.term_pi_v3(json.loads(vocab_path.read_text()))
    cands = pd.read_parquet(cand_path)

    table, svd_info = feature_table(edges, cands, pi, stability, recent_start, cfg["text"])
    if len(table) != len(cands):
        sys.exit(f"{len(cands) - len(table)} candidates have no features")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    table.to_parquet(out, index=False)
    feature_names = [c for c in table.columns if c not in CANDIDATE_COLUMNS]
    result = {
        "meta": {"step": "engine_v3_features", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": features.git_commit(), "environment": runtime_environment(),
                 "build_window": [start.isoformat(), end.isoformat()], "recent_start": recent_start.isoformat(),
                 "stability": stability, "text": cfg["text"],
                 "inputs_sha256": {"vocab": features.sha256(vocab_path), "build": features.sha256(build_path),
                                   "candidates": features.sha256(cand_path)},
                 "output_sha256": features.sha256(out)},
        "candidates": int(len(table)),
        "features": feature_names,
        "matches_feature_sets": sorted(feature_names) == sorted(f for s in cfg["sets"].values() for f in s),
        svd_info[0]: svd_info[1],
        "missing_values": int(table[feature_names].isna().sum().sum()),
    }
    (RESULTS / f"engine_v3_features_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
