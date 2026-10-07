"""Validate the synonym review and turn it into merge groups and excluded pairs (matching preregistration 4.2, 4.3).

Usage: python scripts/matching/06_synonym_decisions.py

The review file is checked against its read-only _original.csv (only decision and note may change,
decisions from matching.synonyms.decisions). Pairs judged same are joined into groups (connected
pairs form one group) represented by the member whose name (vocabulary key) comes first; pairs judged hierarchy are
listed for exclusion. Containment pairs are computed later per vocabulary (scripts/matching/05_synonym_sheet.py
`containment`) because they depend on each timepoint's terms.

Outputs: vocab/synonym_decisions.json, results/synonym_decisions.json
"""
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

OUT = Path("vocab/synonym_decisions.json")
VOCAB = Path("vocab/vocab_base.json")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def validate(review: pd.DataFrame, original: pd.DataFrame, allowed: list[str]) -> list[str]:
    if list(review.columns) != list(original.columns):
        return ["columns differ from the original"]
    fixed = [c for c in original.columns if c not in ("decision", "note")]
    errors = []
    if not review[fixed].equals(original[fixed]):
        errors.append("columns outside decision and note were changed")
    bad = review.loc[~review["decision"].isin(allowed), "review_id"].tolist()
    if bad:
        errors.append(f"missing or unknown decision: {bad}")
    return errors


def merge_groups(pairs: list[tuple[str, str]], names: dict[str, str]) -> dict[str, list[str]]:
    """Connected components of the same pairs, keyed by the member whose name comes first (ties by id)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return {min(members, key=lambda m: (names[m], m)): sorted(members) for members in groups.values()}


def main() -> None:
    cfg = load_study()["matching"]["synonyms"]
    review_path = Path(cfg["review_file"])
    original_path = review_path.with_name(review_path.stem + "_original.csv")
    review = read(review_path)
    errors = validate(review, read(original_path), cfg["decisions"])
    if errors:
        sys.exit("review file is not ready:\n- " + "\n- ".join(errors))
    same = [(r.term_a, r.term_b) for r in review.itertuples(index=False) if r.decision == "same"]
    hierarchy = sorted((r.term_a, r.term_b) for r in review.itertuples(index=False) if r.decision == "hierarchy")
    names = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}
    groups = merge_groups(same, names)
    meta = {"step": "synonym_decisions", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
            "environment": runtime_environment(),
            "files_sha256": {"review": evaluate.sha256(review_path), "original": evaluate.sha256(original_path)}}
    OUT.write_text(json.dumps({"meta": meta, "merge_groups": groups, "hierarchy_pairs": [list(p) for p in hierarchy]}, indent=1))
    counts = review["decision"].value_counts().to_dict()
    summary = {"meta": {**meta, "output_sha256": evaluate.sha256(OUT)}, "pairs": len(review),
               "by_decision": {d: int(counts.get(d, 0)) for d in cfg["decisions"]},
               "merge_groups": len(groups), "terms_merged": sum(len(m) for m in groups.values())}
    (RESULTS / "synonym_decisions.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
