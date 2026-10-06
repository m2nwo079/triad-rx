"""Review validation and adoption rule (scripts/engine/03_label_evidence_score.py) on synthetic rows."""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_score", ROOT / "scripts/engine/03_label_evidence_score.py")
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)


def sheets(pos_judgements, neg_judgements):
    judgements = list(pos_judgements) + list(neg_judgements)
    ids = [f"le-{i:03d}" for i in range(len(judgements))]
    original = pd.DataFrame({"review_id": ids, "title": "t", "abstract": "a", "judgement": "", "term_error": "", "note": ""})
    review = original.assign(judgement=judgements)
    key = pd.DataFrame({"review_id": ids, "rule": ["positive"] * len(pos_judgements) + ["negative"] * len(neg_judgements)})
    return review, original, key


def test_valid_review_passes():
    review, original, key = sheets(["결합", "비교"], ["언급"])
    assert ls.validate(review, original, key) == []


def test_validation_catches_each_problem():
    review, original, key = sheets(["결합", "비교"], ["언급"])
    edited = review.copy()
    edited.loc[0, "abstract"] = "changed"
    assert ls.validate(edited, original, key)
    blank = review.copy()
    blank.loc[1, "judgement"] = ""
    assert ls.validate(blank, original, key)
    wrong_error = review.copy()
    wrong_error.loc[0, "term_error"] = "y"
    assert ls.validate(wrong_error, original, key)
    assert ls.validate(review, original, key.iloc[:2])


def test_adopted_when_groups_separate():
    review, _, key = sheets(["결합"] * 40, ["언급"] * 40)
    out = ls.score(review, key)
    assert out["adopted"]
    assert out["rule_positive"]["combination_share"] == 1.0
    assert out["rule_negative"]["judgements"]["mention"] == 40


def test_not_adopted_when_intervals_overlap():
    review, _, key = sheets(["결합", "언급"] * 5, ["결합", "언급", "언급"] * 3)
    assert not ls.score(review, key)["adopted"]
