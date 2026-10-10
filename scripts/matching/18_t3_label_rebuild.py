"""T3 label-window hypergraph with the v3 rules (matching preregistration 6.4).

Usage: python scripts/matching/18_t3_label_rebuild.py

Locked: it stops unless the matching preregistration records its freeze commit.
The vocabulary, surface rules and exclusion phrases are read from vocab/timepoints/T3_v3_final.json,
written by scripts/matching/07_rebuild.py from the T3 build window only. Before the label window is read,
the T3 build-window hypergraph is matched again from that file and must equal the stored
data/derived/hypergraph/v3/T3_build.parquet, so the label window uses exactly the matching of the build window.
The label window runs from the T3 label start to the data cutoff (triadrx.study.t3_window).

Output (not committed): data/derived/hypergraph/v3/T3_label.parquet
Summary: results/matching_rebuild_T3_label.json
"""
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.corpus import match_papers_refined
from triadrx.study import data_cutoff, engine_prereg_frozen, load_study, runtime_environment, t3_window

TIMEPOINTS = Path("vocab/timepoints")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
PAPERS = Path("data/arxiv/papers.parquet")
RESULTS = Path("results")
TP = "T3"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebuild = load("matching_rebuild", ROOT / "scripts/matching/07_rebuild.py")
evaluate = rebuild.evaluate


def require_frozen(study: dict) -> str:
    """Freeze commit of the matching preregistration; stop while it is not recorded."""
    text = Path(study["matching"]["prereg_file"]).read_text()
    if not engine_prereg_frozen(text):
        sys.exit("T3 label window is locked until the matching preregistration records its freeze commit")
    return re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M).group(1).strip()


def matching_inputs(vocab: dict) -> tuple[list[dict], set[str], dict]:
    """Final terms, case-sensitive-only terms and exclusion phrases as 07_rebuild.py passed them."""
    terms = vocab["terms"]
    cs_only = {t["id"] for t in terms if all(s["match"] == "cs" for s in t["surfaces"])}
    phrases = {k: [tuple(p) for p in v] for k, v in vocab["exclusion_phrases"].items()}
    return terms, cs_only, phrases


def window_papers(start, end) -> pd.DataFrame:
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"],
                             filters=[("first_date", ">=", start), ("first_date", "<=", end)])
    return papers.sort_values("id").reset_index(drop=True)


def same_hypergraph(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if len(a) != len(b) or list(a["paper_id"]) != list(b["paper_id"]):
        return False
    return all(list(x) == list(y) for x, y in zip(a["terms"], b["terms"]))


def main() -> None:
    study = rebuild.prep.merged_study(load_study())
    freeze = require_frozen(study)
    rules = study["matching"]["surface_rules"]
    vocab_path, build_path = TIMEPOINTS / f"{TP}_v3_final.json", HYPERGRAPH / f"{TP}_build.parquet"
    vocab = json.loads(vocab_path.read_text())
    if vocab["meta"]["surface_rules"] != rules:
        sys.exit("surface rules differ from those of the T3 vocabulary")
    terms, cs_only, phrases = matching_inputs(vocab)

    # The build window must match exactly as in 07_rebuild.py before the label window is read
    build_window = rebuild.sample.build_window_papers(TP, study)
    again = rebuild.hyperedges(build_window, match_papers_refined(build_window, terms, cs_only, rules, phrases))
    if not same_hypergraph(again, pd.read_parquet(build_path)):
        sys.exit("re-matched T3 build window differs from the stored hypergraph; label window not read")

    start, end = t3_window("label", study, data_cutoff())
    window = window_papers(start, end)
    label = rebuild.hyperedges(window, match_papers_refined(window, terms, cs_only, rules, phrases))
    out = HYPERGRAPH / f"{TP}_label.parquet"
    label.to_parquet(out)
    result = {"meta": {"step": "matching_rebuild_T3_label", "created": datetime.date.today().isoformat(),
                       "git_commit": evaluate.git_commit(), "preregistration_freeze_commit": freeze,
                       "environment": runtime_environment(), "label_window": [start.isoformat(), end.isoformat()],
                       "inputs_sha256": {"vocab": evaluate.sha256(vocab_path), "build": evaluate.sha256(build_path)},
                       "output_sha256": evaluate.sha256(out)},
              "build_window_rematched_equal": True,
              "label_window_papers": int(len(window)),
              "label_hyperedges": int(len(label))}
    (RESULTS / "matching_rebuild_T3_label.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
