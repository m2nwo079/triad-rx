"""Label v3 from the relation classifier (matching preregistration 5.3).

Usage: python scripts/matching/13_label_v3_relation.py --timepoint T1|R2|R3|T3|T2 --outputs <relation_outputs.jsonl> --meta <relation_meta.json>

A realised triad (data/derived/relation/<tp>_frame.parquet) is positive when at least
engine_reliability.label_v2.min_evidence_papers of its papers are classified combination. With per-triad
early stopping, papers after the first combination may be unclassified; this does not change a positive
triad, and a negative triad has all its papers classified. Every frame record must be classified unless
its triad is already positive.

Outputs: data/derived/labels_v3/relation/<tp>.parquet (not committed), results/label_v3_relation_<tp>.json
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
from triadrx.study import load_study, runtime_environment

FRAMES = Path("data/derived/relation")
OUT_DIR = Path("data/derived/labels_v3/relation")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def label_table(frame: pd.DataFrame, outputs: pd.DataFrame, min_papers: int) -> pd.DataFrame:
    df = frame.merge(outputs[["id", "relation"]], on="id", how="left")
    df["combination"] = df["relation"] == "combination"
    table = df.groupby(KEYS).agg(n_papers=("id", "size"), n_classified=("relation", "count"),
                                 n_combination=("combination", "sum")).reset_index()
    table["label_v3"] = table["n_combination"] >= min_papers
    missing = table[(~table["label_v3"]) & (table["n_classified"] < table["n_papers"])]
    if len(missing):
        raise ValueError(f"{len(missing)} negative triads have unclassified papers")
    return table


def main() -> None:
    study = load_study()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "R2", "R3", "T3", "T2"])
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    args = parser.parse_args()
    tp = args.timepoint
    frame_path = FRAMES / f"{tp}_frame.parquet"
    table = label_table(pd.read_parquet(frame_path), pd.read_json(args.outputs, lines=True),
                        study["engine_reliability"]["label_v2"]["min_evidence_papers"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    table.to_parquet(out, index=False)
    result = {"meta": {"step": "label_v3_relation", "timepoint": tp, "created": datetime.date.today().isoformat(),
                       "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                       "model_run": json.loads(args.meta.read_text()), "frame_sha256": evaluate.sha256(frame_path),
                       "outputs_sha256": evaluate.sha256(args.outputs), "output_sha256": evaluate.sha256(out)},
              "triads": int(len(table)), "pairs": int(table["n_papers"].sum()), "pairs_classified": int(table["n_classified"].sum()),
              "label_v3_positive": int(table["label_v3"].sum()), "positive_share": float(table["label_v3"].mean())}
    (RESULTS / f"label_v3_relation_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
