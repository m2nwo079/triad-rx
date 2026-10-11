"""Decisions of the applied review, combined with the third-study decisions (represcription preregistration 2).

Usage: python scripts/represcription/03_review_decisions.py

Both review files are checked against their read-only originals with the third-study rules
(scripts/matching/02_common_word_decisions.py and 06_synonym_decisions.py, reused through importlib):
only decision, note and, for common words, exclusions may change, and every row needs an allowed decision.
The new decisions are then combined with the third-study ones:
- common words: the third-study decisions followed by the new ones (no item appears in both);
- synonyms: the third-study merge groups are expanded back into pairs, the new `same` pairs are added and
  the groups are recomputed with the same function, so unchanged groups keep their names and a new pair
  that touches a group joins it; hierarchy pairs are the union.
The applied window is then rebuilt with the combined files (scripts/represcription/01_applied_rebuild.py).

Outputs: represcription.review.combined.common_words and .synonyms (committed),
results/review_decisions_applied.json
"""
import datetime
import importlib.util
import json
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment

VOCAB = Path("vocab/vocab_base.json")
COMMON = Path("vocab/common_word_decisions.json")
SYNONYMS = Path("vocab/synonym_decisions.json")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cwd = load("common_word_decisions", ROOT / "scripts/matching/02_common_word_decisions.py")
syd = load("synonym_decisions", ROOT / "scripts/matching/06_synonym_decisions.py")
evaluate = syd.evaluate


def combined_groups(groups: dict[str, list[str]], same: list[tuple[str, str]], names: dict[str, str]) -> dict[str, list[str]]:
    """Third-study groups plus new same pairs, recomputed with the third-study grouping rule."""
    pairs = [(rep, m) for rep, members in groups.items() for m in members if m != rep] + list(same)
    return syd.merge_groups(pairs, names)


def checked(path: Path, allowed: list[str], validate) -> "object":
    original = path.with_name(path.stem + "_original.csv")
    review = cwd.read(path)
    errors = validate(review, cwd.read(original), allowed)
    if errors:
        sys.exit(f"{path} is not ready:\n- " + "\n- ".join(errors))
    return review, original


def main() -> None:
    study = load_study()
    m, rcfg = study["matching"], study["represcription"]["review"]
    out_common, out_synonyms = Path(rcfg["combined"]["common_words"]), Path(rcfg["combined"]["synonyms"])
    cw_path, sy_path = Path(rcfg["common_words_file"]), Path(rcfg["synonyms_file"])
    cw_review, cw_original = checked(cw_path, m["common_words"]["decisions"], cwd.validate)
    sy_review, sy_original = checked(sy_path, m["synonyms"]["decisions"], syd.validate)

    old_common = json.loads(COMMON.read_text())
    new_common = cwd.decisions(cw_review)
    overlap = {(d["term_id"], d["surface"]) for d in old_common["decisions"]} & {(d["term_id"], d["surface"]) for d in new_common}
    if overlap:
        sys.exit(f"items already decided in the third study: {sorted(overlap)}")
    old_syn = json.loads(SYNONYMS.read_text())
    same = [(r.term_a, r.term_b) for r in sy_review.itertuples(index=False) if r.decision == "same"]
    hierarchy = sorted({tuple(p) for p in old_syn["hierarchy_pairs"]} |
                       {(r.term_a, r.term_b) for r in sy_review.itertuples(index=False) if r.decision == "hierarchy"})
    names = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}
    groups = combined_groups(old_syn["merge_groups"], same, names)

    meta = {"step": "review_decisions_applied", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
            "environment": runtime_environment(),
            "files_sha256": {"common_words_review": evaluate.sha256(cw_path), "common_words_original": evaluate.sha256(cw_original),
                             "synonyms_review": evaluate.sha256(sy_path), "synonyms_original": evaluate.sha256(sy_original),
                             "third_study_common_words": evaluate.sha256(COMMON), "third_study_synonyms": evaluate.sha256(SYNONYMS)}}
    out_common.write_text(json.dumps({"meta": meta, "decisions": old_common["decisions"] + new_common}, indent=1))
    out_synonyms.write_text(json.dumps({"meta": meta, "merge_groups": groups, "hierarchy_pairs": [list(p) for p in hierarchy]}, indent=1))

    cw_counts, sy_counts = cw_review["decision"].value_counts().to_dict(), sy_review["decision"].value_counts().to_dict()
    old_groups = old_syn["merge_groups"]
    summary = {"meta": {**meta, "outputs_sha256": {"common_words": evaluate.sha256(out_common), "synonyms": evaluate.sha256(out_synonyms)}},
               "common_words": {"surfaces": len(cw_review), "by_decision": {d: int(cw_counts.get(d, 0)) for d in m["common_words"]["decisions"]}},
               "synonyms": {"pairs": len(sy_review), "by_decision": {d: int(sy_counts.get(d, 0)) for d in m["synonyms"]["decisions"]}},
               "merge_groups": {"third_study": len(old_groups), "combined": len(groups),
                                "grown": sum(1 for rep, members in groups.items() if rep in old_groups and len(members) > len(old_groups[rep])),
                                "new": sum(1 for rep in groups if rep not in old_groups)},
               "hierarchy_pairs": {"third_study": len(old_syn["hierarchy_pairs"]), "combined": len(hierarchy)}}
    (RESULTS / "review_decisions_applied.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
