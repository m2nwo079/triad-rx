"""Second-study labels for realised new triads (engine preregistration 2.1, adopted at attempt 1).

Usage: python scripts/engine/04_label_v2.py --timepoint T1|T2

Only development timepoints (engine_reliability.conditional.timepoints) are accepted; T3 is never
read here. Realised new triads are the significant ones of scripts/model/03_candidate_recall.py
(BH over all realised new triads, not over the candidates as in the first-study labels). A triad
is positive when at least `min_evidence_papers` of its label-window papers carry combination
evidence under the adopted rule (scripts/engine/02_label_evidence.py, reused through importlib).

Outputs
- data/derived/labels_v2/main/<tp>.parquet (not committed): a, b, c, n_papers, n_evidence, label_v2
- results/label_v2_<tp>.json: counts and hashes
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
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph/main")
OUT_DIR = Path("data/derived/labels_v2/main")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load("label_evidence", ROOT / "scripts/engine/02_label_evidence.py")


def label_table(realised: dict, positive: dict, min_evidence: int) -> pd.DataFrame:
    """One row per realised triad with its paper and evidence-paper counts and the second-study label."""
    rows = [{"a": a, "b": b, "c": c, "n_papers": len(idxs), "n_evidence": len(positive.get((a, b, c), []))}
            for (a, b, c), idxs in sorted(realised.items())]
    df = pd.DataFrame(rows, columns=["a", "b", "c", "n_papers", "n_evidence"])
    df["label_v2"] = df["n_evidence"] >= min_evidence
    return df


def main() -> None:
    study = load_study()
    er = study["engine_reliability"]
    cfg = er["label_v2"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=er["conditional"]["timepoints"])
    tp = parser.parse_args().timepoint

    paths = {"build": HYPERGRAPH / f"{tp}_build.parquet", "label": HYPERGRAPH / f"{tp}_label.parquet"}
    build, label = pd.read_parquet(paths["build"]), pd.read_parquet(paths["label"])
    realised = evidence.significant_new_triads(build, label, study["labels"]["fdr"])
    positive, _ = evidence.split_by_evidence(tp, cfg, label, realised)

    df = label_table(realised, positive, cfg["min_evidence_papers"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    df.to_parquet(out, index=False)

    result = {
        "meta": {"step": "label_v2", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evidence.evaluate.git_commit(), "label_v2_config": cfg,
                 "realised_set": "significant new triads with BH over all realised new triads (as in candidate recall)",
                 "inputs_sha256": {k: evidence.evaluate.sha256(v) for k, v in paths.items()},
                 "output_sha256": evidence.evaluate.sha256(out)},
        "realised_significant_triads": int(len(df)),
        "triad_paper_pairs": int(df["n_papers"].sum()),
        "pairs_with_evidence": int(df["n_evidence"].sum()),
        "label_v2_positive": int(df["label_v2"].sum()),
        "label_v2_share": float(df["label_v2"].mean()) if len(df) else float("nan"),
    }
    (RESULTS / f"label_v2_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
