"""Patent evidence for the represcription with v3 matching (represcription preregistration 5).

Usage (from the run folder): cd represcription && python ../scripts/represcription/07_patents.py

Runs the first-study patent step (scripts/evidence/03_patents.py, reused through importlib and left unchanged)
with the first-study principle "the same final vocabulary and matching function as the papers" (design 6.1)
applied to the represcription: the final terms are those of the applied v3 vocabulary and each title and
abstract is matched with the third-study refined matching (triadrx.text.refined_spans with the surface rules
and the exclusion phrases of that vocabulary). Selection of documents, inventions and assignees is unchanged.
The matching used is added to results/evidence_patents.json.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study
from triadrx.text import Matcher, refined_spans

APPLIED_VOCAB = Path("vocab/timepoints/applied_v3_final.json")
RESULT = Path("results/evidence_patents.json")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patents = load("patents_first", ROOT / "scripts/evidence/03_patents.py")
t3_label = load("t3_label_rebuild", ROOT / "scripts/matching/18_t3_label_rebuild.py")


class RefinedMatcher:
    """Matcher with the first-study `spans` interface that applies the refined matching rules to one text."""

    def __init__(self, terms: list[dict], cs_only: set[str], rules: dict, phrases: dict):
        self.matcher = Matcher(terms)
        self.cs_only, self.rules, self.phrases = cs_only, rules, phrases

    def spans(self, text: str) -> list[dict]:
        return refined_spans(self.matcher, [text], self.cs_only, self.rules, self.phrases)[0]

    def __getattr__(self, name: str):
        # Other attributes the first-study step reads (such as `ambiguous`) come from the underlying matcher
        return getattr(self.matcher, name)


def main() -> None:
    study = load_study()
    vocab = json.loads(APPLIED_VOCAB.read_text())
    terms, cs_only, phrases = t3_label.matching_inputs(vocab)
    rules = study["matching"]["surface_rules"]
    if vocab["meta"]["surface_rules"] != rules:
        sys.exit("surface rules differ from those of the applied v3 vocabulary")

    original_load, original_matcher = patents.load, patents.Matcher
    patents.load = lambda name, path: (SimpleNamespace(final_terms=lambda tp, base: terms) if name == "build_hypergraph"
                                       else original_load(name, path))
    patents.Matcher = lambda given: RefinedMatcher(given, cs_only, rules, phrases)
    try:
        patents.main()
    finally:
        patents.load, patents.Matcher = original_load, original_matcher

    result = json.loads(RESULT.read_text())
    result["meta"]["matching"] = {"vocabulary": str(APPLIED_VOCAB), "vocabulary_sha256": patents_sha256(APPLIED_VOCAB),
                                  "rule": "third-study refined matching (triadrx.text.refined_spans)", "terms": len(terms)}
    RESULT.write_text(json.dumps(result, indent=2))
    print("matching:", result["meta"]["matching"])


def patents_sha256(path: Path) -> str:
    return t3_label.evaluate.sha256(path)


if __name__ == "__main__":
    main()
