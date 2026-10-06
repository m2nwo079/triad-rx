"""Combination-evidence rule (scripts/engine/02_label_evidence.py) on synthetic text."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_evidence", ROOT / "scripts/engine/02_label_evidence.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)

ABBR = ["et al.", "e.g.", "vs."]
CUES = le.cue_pattern(["compared with", "outperforms", "baseline", "vs.", "compare"])


def test_split_keeps_abbreviations():
    text = "We follow Smith et al. in this work. We use BERT, e.g. for QA. It beats LSTM vs. GRU."
    assert le.split_sentences(text, ABBR) == ["We follow Smith et al. in this work.", "We use BERT, e.g. for QA.",
                                              "It beats LSTM vs. GRU."]


def test_cue_word_boundaries():
    assert CUES.search("Compared with BERT, ours is faster")
    assert CUES.search("a strong baseline.")
    assert CUES.search("LSTM vs. GRU")
    assert not CUES.search("a comparable model")
    # "baseline" must not match inside a longer word
    assert not CUES.search("baselines-free setting")


def test_evidence_needs_all_terms_in_window_without_cue():
    sents = ["We combine A with B.", "This solves C.", "Unrelated D."]
    terms = [{"A", "B"}, {"C"}, {"D"}]
    assert le.has_evidence(("A", "B", "C"), terms, sents, 2, CUES)
    assert not le.has_evidence(("A", "B", "D"), terms, sents, 2, CUES)
    assert not le.has_evidence(("A", "B", "C"), terms, sents, 1, CUES)


def test_evidence_rejects_comparison_window():
    sents = ["A outperforms B on C."]
    assert not le.has_evidence(("A", "B", "C"), [{"A", "B", "C"}], sents, 2, CUES)


def test_draw_one_paper_per_distinct_triad():
    pool = {("a", "b", "c"): ["p1", "p2"], ("a", "b", "d"): ["p3"], ("a", "c", "d"): ["p4"]}
    out = le.draw(np.random.default_rng(0), pool, 2, {("a", "c", "d")})
    assert len({t for t, _ in out}) == 2
    assert all(p in pool[t] for t, p in out)
    assert ("a", "c", "d") not in {t for t, _ in out}


def test_significant_new_triads_excludes_closed_and_task_only():
    build = pd.DataFrame({"terms": [["M:a", "M:b", "M:c"]]})
    label = pd.DataFrame({"terms": [["M:a", "M:b", "M:c"], ["T:x", "T:y", "T:z"]] + [["M:a", "M:b", "M:d"]] * 5
                          + [["M:e"], ["M:f"]] * 20})
    out = le.significant_new_triads(build, label, 0.05)
    assert ("M:a", "M:b", "M:c") not in out
    assert ("T:x", "T:y", "T:z") not in out
    assert ("M:a", "M:b", "M:d") in out and len(out[("M:a", "M:b", "M:d")]) == 5


TERMS = [
    {"id": "M:faster r cnn", "status": "active", "surfaces": [{"text": "faster r-cnn", "match": "ci"}]},
    {"id": "M:r cnn", "status": "active", "surfaces": [{"text": "R-CNN", "match": "cs"}]},
    {"id": "M:gps", "status": "active", "surfaces": [{"text": "GPS", "match": "cs"},
                                                     {"text": "greedy policy search", "match": "ci"}]},
    {"id": "M:unet++", "status": "active", "surfaces": [{"text": "unet++", "match": "ci"}]},
]
RULES = {"drop_nested": True, "acronym_needs_full_name": True, "skip_surface_chars": ["+"]}


def test_evidence_vocab_drops_lossy_surfaces_and_finds_acronym_only_terms():
    terms, cs_only = le.evidence_vocab(TERMS, ["+"])
    assert "M:unet++" not in {t["id"] for t in terms}
    assert cs_only == {"M:r cnn"}


def test_nested_match_is_dropped():
    terms, cs_only = le.evidence_vocab(TERMS, ["+"])
    m = le.Matcher(terms)
    assert le.evidence_terms(m, ["We use Faster R-CNN."], cs_only, RULES) == [{"M:faster r cnn"}]
    assert le.evidence_terms(m, ["We use R-CNN."], cs_only, RULES) == [{"M:r cnn"}]


def test_acronym_needs_full_name_in_the_paper():
    terms, cs_only = le.evidence_vocab(TERMS, ["+"])
    m = le.Matcher(terms)
    assert le.evidence_terms(m, ["Robots use GPS."], cs_only, RULES) == [set()]
    out = le.evidence_terms(m, ["Greedy policy search (GPS).", "GPS helps."], cs_only, RULES)
    assert out == [{"M:gps"}, {"M:gps"}]
    loose = {**RULES, "acronym_needs_full_name": False}
    assert le.evidence_terms(m, ["Robots use GPS."], cs_only, loose) == [{"M:gps"}]
