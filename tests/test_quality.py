import math

from triadrx import quality as q


def test_sentences_split_on_sentence_end():
    assert q.sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]
    assert q.sentences("") == []
    assert q.sentences("Uses YOLOv8. 3D detection works.") == ["Uses YOLOv8.", "3D detection works."]


def test_numbers_ignore_tokens_inside_words():
    assert q.numbers("YOLOv8 with 3D and 42 tasks, score 4.0.") == [42.0, 4.0]
    assert q.numbers("proposed threshold: 30 percent") == [30.0]
    assert q.numbers("See paper:2609.20888, repo:a/b2 and (dvf:14) for 3 tasks.") == [3.0]
    assert q.strip_ids("x (dvf:16).").strip() == "x ( )."


def test_wilson_matches_known_values():
    low, high = q.wilson(8, 8)
    assert math.isclose(low, 0.6756, abs_tol=1e-3)
    low, high = q.wilson(98, 100)
    assert math.isclose(low, 0.9300, abs_tol=1e-3)


def test_items_follow_field_classes():
    item = {"text": "x.", "evidence_ids": ["paper:1"]}
    out = {"concept_frame": {"status": "insufficient_concept", "insufficient_explanation": "y.",
                             "insufficient_evidence_ids": ["paper:1"]}, "briefing_valid": None, "briefing": None}
    assert [i["field"] for i in q.evidential_items(out)] == ["concept_frame.insufficient_explanation"]
    assert q.prescriptive_items(out) == []
    brief = {k: item for k in ("summary", "target_user", "jtbd", "value_prop", "riskiest_assumption", "mvp", "success_metric")}
    brief["kill_criteria"] = [item]
    brief["building_blocks"] = {"open_source": [{"id": "pkg:p", "use": "u.", "evidence_ids": ["pkg:p"]}], "to_build": [item]}
    out = {"concept_frame": {"status": "complete", "components": [], "target_task": item, "limitation": item,
                             "combination": item}, "briefing_valid": True, "briefing": brief}
    evid = [i["field"] for i in q.evidential_items(out)]
    pres = [i["field"] for i in q.prescriptive_items(out)]
    assert "summary" in evid and "building_blocks.open_source[pkg:p].use" in evid
    assert "mvp" in pres and "kill_criteria[0]" in pres and "summary" not in pres
