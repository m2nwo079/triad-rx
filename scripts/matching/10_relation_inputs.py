"""Inputs for the relation classifier (matching preregistration 5.3).

Usage: python scripts/matching/10_relation_inputs.py --set dev|T1|R2|R3|T3|T2

dev: every (triad, paper) of the human reviews in matching.relation.dev_reviews, with the title and
abstract shown to the reviewer and the three terms (name and type). Human judgements are written to a
separate truth file that never goes to the model.

T1, R2, R3, T3: every (triad, paper) of the realised triads of the timepoint's v3 hypergraphs with the
excluded pairs dropped (scripts/matching/08_label_v3_sample.py), each tagged with its triad as the group
for per-triad early stopping. T3 reads its label window only after preregistration_matching.md records
its freeze commit.

Outputs (not committed): data/derived/relation/<set>_inputs.jsonl, data/derived/relation/dev_truth.parquet or <set>_frame.parquet,
and results/relation_inputs_<set>.json (counts and hashes)
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
from triadrx.study import engine_prereg_frozen, load_study, runtime_environment

OUT_DIR = Path("data/derived/relation")
RESULTS = Path("results")
TYPES = {"M": "method", "T": "task"}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def term_entry(term_id: str) -> dict:
    kind, name = term_id.split(":", 1)
    return {"id": term_id, "name": name, "type": TYPES[kind]}


def record(rid: str, triad: list[str], paper_id: str, title: str, abstract: str) -> dict:
    return {"id": rid, "paper_id": paper_id, "title": " ".join(title.split()), "abstract": " ".join(abstract.split()),
            "terms": [term_entry(t) for t in triad]}


def dev_set(cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    records, truth = [], []
    for source in cfg["dev_reviews"]:
        review = pd.read_csv(source["review"], encoding="utf-8-sig", dtype=str, keep_default_na=False)
        key = pd.read_csv(source["key"], encoding="utf-8-sig", dtype=str, keep_default_na=False)
        rows = review.merge(key[["review_id", "rule"]], on="review_id", validate="one_to_one")
        tag = Path(source["review"]).stem
        for r in rows.itertuples(index=False):
            rid = f"{tag}:{r.review_id}"
            records.append(record(rid, [r.term_a, r.term_b, r.term_c], r.paper_id, r.title, r.abstract))
            truth.append({"id": rid, "source": tag, "rule": r.rule, "judgement": r.judgement, "term_error": r.term_error == "y"})
    return records, pd.DataFrame(truth)


def label_set(study: dict, tp: str) -> tuple[list[dict], pd.DataFrame]:
    """All (triad, paper) of a timepoint's kept realised triads, grouped by triad."""
    l3 = load("label_v3_sample", ROOT / "scripts/matching/08_label_v3_sample.py")
    vocab = json.loads((l3.TIMEPOINTS / f"{tp}_v3_final.json").read_text())
    build = pd.read_parquet(l3.HYPERGRAPH / f"{tp}_build.parquet")
    label = pd.read_parquet(l3.HYPERGRAPH / f"{tp}_label.parquet")
    realised = l3.evidence.significant_new_triads(build, label, study["labels"]["fdr"])
    excluded = l3.excluded_pairs(vocab)
    pairs = [(t, label.iloc[i]["paper_id"]) for t in sorted(realised) if not l3.holds_excluded_pair(t, excluded) for i in realised[t]]
    text = pd.read_parquet(l3.PAPERS, columns=["id", "title", "abstract"], filters=[("id", "in", sorted({p for _, p in pairs}))]).set_index("id")
    records, rows = [], []
    for n, (t, pid) in enumerate(pairs):
        r = record(f"{tp}-{n:06d}", list(t), pid, text.at[pid, "title"], text.at[pid, "abstract"])
        r["group"] = "|".join(t)
        records.append(r)
        rows.append({"id": r["id"], "a": t[0], "b": t[1], "c": t[2], "paper_id": pid})
    return records, pd.DataFrame(rows)


def main() -> None:
    study = load_study()
    cfg = study["matching"]["relation"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, choices=["dev", "T1", "R2", "R3", "T3", "T2"])
    name = parser.parse_args().set
    if name == "T3" and not engine_prereg_frozen(Path(study["matching"]["prereg_file"]).read_text()):
        sys.exit("T3 label window is locked until the matching preregistration records its freeze commit")
    if name == "dev":
        records, truth = dev_set(cfg)
    else:
        records, truth = label_set(study, name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = OUT_DIR / f"{name}_inputs.jsonl"
    with inputs.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    # The development set keeps human judgements (truth); label sets keep the id-to-triad frame
    truth_path = OUT_DIR / f"{name}_{'truth' if name == 'dev' else 'frame'}.parquet"
    truth.to_parquet(truth_path, index=False)
    result = {"meta": {"step": "relation_inputs", "set": name, "created": datetime.date.today().isoformat(),
                       "git_commit": evaluate.git_commit(), "environment": runtime_environment(), "relation_config": cfg,
                       "inputs_sha256": evaluate.sha256(inputs), "truth_sha256": evaluate.sha256(truth_path)},
              "records": len(records),
              "by_source": truth["source"].value_counts().to_dict() if "source" in truth else None,
              "groups": int(truth[["a", "b", "c"]].drop_duplicates().shape[0]) if "a" in truth else None}
    (RESULTS / f"relation_inputs_{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
