"""Applied-mode build window rebuilt with the third-study v3 rules (represcription preregistration 2).

Usage: python scripts/represcription/01_applied_rebuild.py [--with-review]

Runs the third-study rebuild (scripts/matching/07_rebuild.py, reused through importlib and left unchanged)
for the applied mode. That script reads calendar-year windows, so this script supplies only two things:
- the applied build window (design 4.4: `applied.build_years` years ending at the data cutoff), and
- an `applied` entry among the matching timepoints so the rebuild accepts it (no label window is built).
Everything else is the rebuild unchanged: refined matching with the common-word decisions, synonym merges,
R6, R5 with the third-study stratum lower bounds, and the hierarchy and containment exclusion pairs.
The final applied terms missing from every reviewed timepoint (matching.engine.reviewed_timepoints) are
counted and added to the result. With --with-review the rebuild reads the third-study decisions combined
with the applied review (represcription.review.combined, scripts/represcription/03_review_decisions.py)
instead of the third-study decision files; the files used and their hashes are recorded.

Outputs: vocab/timepoints/applied_v3_final.json, data/derived/hypergraph/v3/applied_build.parquet (not committed)
Summary: results/matching_rebuild_applied.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import build_window, data_cutoff

TP = "applied"
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebuild = load("matching_rebuild", ROOT / "scripts/matching/07_rebuild.py")


def applied_bounds(study: dict):
    """Applied build window (design 4.4). The `applied` entry added for the rebuild is ignored here,
    because build_window treats any name in `timepoints` as a calendar-year timepoint."""
    plain = {**study, "timepoints": {k: v for k, v in study["timepoints"].items() if k != TP}}
    return build_window(TP, plain, data_cutoff())


def applied_window_papers(study: dict) -> pd.DataFrame:
    """Applied build-window papers, read the same way as the third-study window loader."""
    start, end = applied_bounds(study)
    papers = pd.read_parquet(rebuild.sample.PAPERS, columns=["id", "title", "abstract", "first_date"],
                             filters=[("first_date", ">=", start), ("first_date", "<=", end)])
    return papers.sort_values("id").reset_index(drop=True)


def with_applied(study: dict) -> dict:
    """Study settings in which the rebuild accepts the applied mode as a build-only timepoint."""
    study = json.loads(json.dumps(study))
    study["matching"]["timepoints"] = [*study["matching"]["timepoints"], TP]
    study["timepoints"][TP] = {"role": "apply"}
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-review", action="store_true")
    with_review = parser.parse_args().with_review
    decisions = {"common_words": rebuild.sample.DECISIONS, "synonyms": rebuild.SYNONYMS}
    if with_review:
        combined = rebuild.load_study()["represcription"]["review"]["combined"]
        decisions = {"common_words": Path(combined["common_words"]), "synonyms": Path(combined["synonyms"])}
        for path in decisions.values():
            if not path.exists():
                sys.exit(f"{path} is missing; run scripts/represcription/03_review_decisions.py first")
    original_paths = rebuild.sample.DECISIONS, rebuild.SYNONYMS
    rebuild.sample.DECISIONS, rebuild.SYNONYMS = decisions["common_words"], decisions["synonyms"]
    original_window = rebuild.sample.build_window_papers
    original_load = rebuild.load_study
    rebuild.sample.build_window_papers = lambda tp, study: applied_window_papers(study) if tp == TP else original_window(tp, study)
    rebuild.load_study = lambda *args, **kwargs: with_applied(original_load(*args, **kwargs))
    argv = sys.argv
    sys.argv = [argv[0], "--timepoint", TP]
    try:
        rebuild.main()
    finally:
        sys.argv = argv
        rebuild.sample.build_window_papers = original_window
        rebuild.load_study = original_load
        rebuild.sample.DECISIONS, rebuild.SYNONYMS = original_paths

    study = rebuild.load_study()
    path = RESULTS / f"matching_rebuild_{TP}.json"
    result = json.loads(path.read_text())
    final_ids = {t["id"] for t in json.loads((rebuild.TIMEPOINTS / f"{TP}_v3_final.json").read_text())["terms"]}
    start, end = applied_bounds(study)
    result["meta"]["build_window"] = [start.isoformat(), end.isoformat()]
    result["meta"]["decisions"] = {name: {"file": str(path), "sha256": rebuild.evaluate.sha256(path)} for name, path in decisions.items()}
    result["terms_not_in_reviewed_timepoints"] = rebuild.unreviewed_terms(final_ids, study["matching"]["engine"]["reviewed_timepoints"])
    path.write_text(json.dumps(result, indent=2))
    print("build window:", result["meta"]["build_window"])
    print("terms not in reviewed timepoints:", result["terms_not_in_reviewed_timepoints"])


if __name__ == "__main__":
    main()
