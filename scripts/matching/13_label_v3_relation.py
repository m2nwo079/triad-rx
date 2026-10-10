"""Label v3 from the relation classifier (matching preregistration 5.3).

Usage: python scripts/matching/13_label_v3_relation.py --timepoint T1|R2|R3|T3|T2 --outputs <relation_outputs.jsonl> ... --meta <relation_meta.json> ...

A realised triad (data/derived/relation/<tp>_frame.parquet) is positive when at least
engine_reliability.label_v2.min_evidence_papers of its papers are classified combination. With per-triad
early stopping, papers after the first combination may be unclassified; this does not change a positive
triad, and a negative triad has all its papers classified. Every frame record must be classified unless
its triad is already positive.
A set run in parts (kaggle/relation/package.py --part) gives one outputs and one meta file per part, in the
same order; the parts must share the settings other than the part, must not repeat a record, and together
must cover the frame (a missing part leaves negative triads with unclassified papers and stops the run).

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


def merged_outputs(output_paths: list[Path], meta_paths: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """Outputs of one run or of its parts, checked for shared settings and repeated records."""
    if len(output_paths) != len(meta_paths):
        sys.exit("give one meta file per outputs file")
    metas = [json.loads(p.read_text()) for p in meta_paths]
    if len(metas) > 1:
        shared = [{k: v for k, v in m["settings"].items() if k != "part"} for m in metas]
        parts = sorted(m["settings"].get("part") for m in metas)
        if any(s != shared[0] for s in shared) or parts != list(range(len(metas))):
            sys.exit("parts differ in settings or do not cover parts 0 to n-1")
    outputs = pd.concat([pd.read_json(p, lines=True) for p in output_paths], ignore_index=True)
    if outputs["id"].duplicated().any():
        sys.exit("a record appears in more than one outputs file")
    return outputs, metas


def main() -> None:
    study = load_study()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "R2", "R3", "T3", "T2"])
    parser.add_argument("--outputs", required=True, type=Path, nargs="+")
    parser.add_argument("--meta", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    tp = args.timepoint
    frame_path = FRAMES / f"{tp}_frame.parquet"
    outputs, metas = merged_outputs(args.outputs, args.meta)
    table = label_table(pd.read_parquet(frame_path), outputs,
                        study["engine_reliability"]["label_v2"]["min_evidence_papers"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    table.to_parquet(out, index=False)
    result = {"meta": {"step": "label_v3_relation", "timepoint": tp, "created": datetime.date.today().isoformat(),
                       "git_commit": evaluate.git_commit(), "environment": runtime_environment(),
                       "model_run": metas[0] if len(metas) == 1 else metas, "frame_sha256": evaluate.sha256(frame_path),
                       "outputs_sha256": evaluate.sha256(args.outputs[0]) if len(args.outputs) == 1
                       else [evaluate.sha256(p) for p in args.outputs],
                       "output_sha256": evaluate.sha256(out)},
              "triads": int(len(table)), "pairs": int(table["n_papers"].sum()), "pairs_classified": int(table["n_classified"].sum()),
              "label_v3_positive": int(table["label_v3"].sum()), "positive_share": float(table["label_v3"].mean())}
    (RESULTS / f"label_v3_relation_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
