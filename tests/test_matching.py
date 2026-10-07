"""Third-study matching steps on toy data."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cw = load("common_word_sheet", "scripts/matching/01_common_word_sheet.py")
INFLECTIONS = [["ing", ""], ["ing", "e"], ["ed", ""], ["ed", "e"], ["s", ""], ["es", ""]]


def test_common_words_include_inflections_only_through_the_word_list():
    words = {"attention", "patch", "prune", "object"}
    assert cw.is_common("Attention", words, INFLECTIONS)
    assert cw.is_common("patching", words, INFLECTIONS)
    assert cw.is_common("pruning", words, INFLECTIONS)
    assert cw.is_common("objects", words, INFLECTIONS)
    assert not cw.is_common("sarsa", words, INFLECTIONS)


import sys

import pandas as pd

sys.path.insert(0, str(ROOT))
from triadrx.text import Matcher, ci_tokens, refine_terms, refined_spans

RULES = {"drop_nested": True, "acronym_needs_full_name": True, "skip_surface_chars": ["+"]}
TERMS = [
    {"id": "M:attention", "status": "active", "surfaces": [{"text": "attention", "match": "ci"}]},
    {"id": "M:gps", "status": "active", "surfaces": [{"text": "GPS", "match": "cs"}, {"text": "greedy policy search", "match": "ci"}]},
    {"id": "M:palm", "status": "active", "surfaces": [{"text": "palm", "match": "ci"}, {"text": "pathways language model", "match": "ci"}]},
    {"id": "M:unet++", "status": "active", "surfaces": [{"text": "unet++", "match": "ci"}]},
]


def ids(spans):
    return [{m["id"] for m in s} for s in spans]


def test_refine_terms_drops_lossy_and_excluded_surfaces_but_keeps_the_term():
    terms, cs_only = refine_terms(TERMS, ["+"], {("M:palm", "palm")})
    by_id = {t["id"]: [s["text"] for s in t["surfaces"]] for t in terms}
    assert "M:unet++" not in by_id and by_id["M:palm"] == ["pathways language model"]
    assert cs_only == set()


def test_exclusion_phrase_drops_only_the_covered_match():
    terms, cs_only = refine_terms(TERMS, ["+"])
    m = Matcher(terms)
    ex = {"M:attention": [ci_tokens("received attention")]}
    out = refined_spans(m, ["It received attention.", "Attention received praise.", "Self attention works."], cs_only, RULES, ex)
    assert ids(out) == [set(), {"M:attention"}, {"M:attention"}]


def test_acronym_needs_its_full_name_in_the_same_paper():
    terms, cs_only = refine_terms(TERMS, ["+"])
    m = Matcher(terms)
    assert ids(refined_spans(m, ["Robots use GPS."], cs_only, RULES)) == [set()]
    assert ids(refined_spans(m, ["Greedy policy search (GPS).", "GPS helps."], cs_only, RULES)) == [{"M:gps"}, {"M:gps"}]


cwd = load("common_word_decisions", "scripts/matching/02_common_word_decisions.py")
ps = load("precision_score", "scripts/matching/04_precision_score.py")


def test_common_word_review_validation():
    original = pd.DataFrame({"review_id": ["a", "b"], "surface": ["x", "y"], "term_id": ["M:x", "M:y"],
                             "decision": ["", ""], "exclusions": ["", ""], "note": ["", ""]})
    good = original.assign(decision=["keep", "keep_with_exclusions"], exclusions=["", "p; q"])
    allowed = ["keep", "exclude", "keep_with_exclusions"]
    assert cwd.validate(good, original, allowed) == []
    assert cwd.decisions(good)[1]["exclusions"] == ["p", "q"]
    assert cwd.validate(good.assign(exclusions=["p", "p"]), original, allowed)
    assert cwd.validate(good.assign(surface=["z", "y"]), original, allowed)


def test_precision_scoring_and_validation():
    original = pd.DataFrame({"review_id": [f"r{i}" for i in range(4)], "stratum": ["a", "a", "b", "b"],
                             "correct": [""] * 4, "note": [""] * 4})
    review = original.assign(correct=["1", "1", "1", "0"])
    assert ps.validate(review, original) == []
    assert ps.validate(original.assign(correct=["1", "", "1", "0"]), original)
    out = ps.stratum_results(review, {"statistic": "wilson_lower_95", "pass": 0.8, "conditional": 0.75}, 0)
    assert out["a"]["correct"] == 2 and out["b"]["precision"] == 0.5 and out["b"]["status"] == "fail"


ss = load("synonym_sheet", "scripts/matching/05_synonym_sheet.py")
sd = load("synonym_decisions", "scripts/matching/06_synonym_decisions.py")
SYN = {"npmi_min": 0.6, "min_cooccurrence": 2, "acronym_max_letters": 6, "stem_suffixes": ["ional", "ions", "ion", "al", "s"],
       "stem_min_length": 4}


def term(tid, key, surfaces):
    return {"id": tid, "key": key, "type": tid[0], "status": "active", "surfaces": [{"text": t, "match": m} for t, m in surfaces]}


def test_candidate_rules_and_containment_skip():
    terms = [term("T:lip reading", "lip reading", [("lip reading", "ci")]), term("T:lipreading", "lipreading", [("lipreading", "ci")]),
             term("T:ner", "ner", [("ner", "ci")]), term("T:named entity recognition", "named entity recognition", [("named entity recognition", "ci")]),
             term("T:hand pose estimation", "hand pose estimation", [("hand pose estimation", "ci")]),
             term("T:3d hand pose estimation", "3d hand pose estimation", [("3d hand pose estimation", "ci")])]
    build = pd.DataFrame({"terms": [["T:ner", "T:named entity recognition"]] * 3 + [["T:hand pose estimation", "T:3d hand pose estimation"]] * 3 + [["T:lip reading"]] * 4})
    found = ss.candidate_pairs(terms, build, SYN)
    assert "form" in found[("T:lip reading", "T:lipreading")]["rules"]
    assert {"initials", "cooccurrence"} <= set(found[("T:named entity recognition", "T:ner")]["rules"])
    # Containment pairs go to the rule, not to review
    assert ("T:3d hand pose estimation", "T:hand pose estimation") not in found
    assert ("T:3d hand pose estimation", "T:hand pose estimation") in ss.containment(terms, SYN)


def test_merge_groups_join_connected_pairs():
    names = {"a": "zeta", "b": "alpha", "c": "beta", "x": "same", "y": "same"}
    groups = sd.merge_groups([("b", "c"), ("a", "b"), ("x", "y")], names)
    # Represented by the first name; equal names fall back to the id
    assert groups == {"b": ["a", "b", "c"], "x": ["x", "y"]}


rb = load("rebuild", "scripts/matching/07_rebuild.py")


def test_merge_terms_unites_surfaces_under_the_representative():
    terms = [term("T:named entity recognition", "named entity recognition", [("named entity recognition", "ci")]),
             term("T:ner", "ner", [("NER", "cs")]), term("M:attention", "attention", [("attention", "ci")])]
    groups = {"T:named entity recognition": ["T:named entity recognition", "T:ner"]}
    out, cs_only, phrases = rb.merge_terms(terms, groups, {"T:ner": [("x", "y")]})
    merged = {t["id"]: t for t in out}
    assert set(merged) == {"T:named entity recognition", "M:attention"}
    assert [s["text"] for s in merged["T:named entity recognition"]["surfaces"]] == ["named entity recognition", "NER"]
    assert merged["T:named entity recognition"]["merged_from"] == ["T:named entity recognition", "T:ner"]
    assert cs_only == set() and phrases == {"T:named entity recognition": [("x", "y")]}


def test_study_timepoint_roles_keep_t3_labels_out_of_the_rebuild():
    import json
    study = rb.prep.merged_study(json.loads((ROOT / "config/study.json").read_text()))
    assert [tp for tp in study["matching"]["timepoints"] if study["timepoints"][tp].get("role") == "train"] == ["T1", "R2", "R3"]
