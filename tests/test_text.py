"""Tests for shared normalization and matching."""
from triadrx.text import Matcher, ci_tokens, cs_tokens, norm


def term(tid, surfaces, status="active"):
    return {"id": tid, "status": status, "surfaces": [{"text": t, "match": m} for t, m in surfaces]}


TERMS = [
    term("M:bert", [("BERT", "cs")]),
    term("M:r cnn", [("R-CNN", "cs")]),
    term("M:convolutional neural network", [("CNN", "cs"), ("convolutional neural network", "ci")]),
    term("T:pre training", [("pre training", "ci")]),
    term("T:image classification", [("image classification", "ci")]),
    term("T:classification", [("classification", "ci")], status="r4_candidate"),
]


def test_norm_unifies_case_and_separators():
    assert norm("Pre-Training_of  Deep/Nets.") == "pre training of deep nets"


def test_surface_and_text_tokenize_the_same_way():
    assert ci_tokens("pre-training") == ci_tokens("Pre training") == ("pre", "training")
    assert cs_tokens("R-CNN") == ("R", "CNN")


def test_acronym_inside_hyphenated_word():
    assert "M:bert" in Matcher(TERMS).terms("A BERT-based model")


def test_acronym_is_case_sensitive():
    assert "M:bert" not in Matcher(TERMS).terms("my friend bert")


def test_longest_match_wins():
    found = Matcher(TERMS).terms("We use Fast R-CNN")
    assert "M:r cnn" in found
    assert "M:convolutional neural network" not in found


def test_inactive_status_is_excluded_by_default():
    text = "image classification and classification"
    assert Matcher(TERMS).terms(text) == {"T:image classification"}
    widened = Matcher(TERMS, statuses=("active", "r4_candidate")).terms(text)
    assert widened == {"T:image classification", "T:classification"}


def test_ambiguous_sequence_is_skipped():
    terms = [term("M:a", [("foo bar", "ci")]), term("M:b", [("Foo-Bar", "ci")])]
    matcher = Matcher(terms)
    assert matcher.terms("foo bar") == set()
    assert ("foo", "bar") in matcher.ambiguous["ci"]
