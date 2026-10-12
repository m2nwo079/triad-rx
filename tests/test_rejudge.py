"""Post-hoc blind re-judging of the faithfulness study."""
import importlib.util
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rejudge", ROOT / "scripts/faithfulness/03_rejudge.py")
rj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rj)


def rows(prefix: str) -> list[dict]:
    return [{"review_id": f"{prefix}-{i}", "rank": "1", "rep": "1", "field": f, "sentence": f"s{prefix}{i}",
             "evidence_ids": "p:1", "correct": str(i % 2), "note": ""}
            for i, f in enumerate(["concept_frame.components[M:a].role", "summary", "building_blocks.open_source[repo:x].use",
                                   "concept_frame.target_task"])]


def test_pick_keeps_only_the_chosen_fields_and_blind_hides_the_order():
    items = rj.pick(rows("a"), "represcription", ["components"]) + rj.pick(rows("b"), "faithfulness", ["components", "open_source"])
    assert [i["field"].split("[")[0] for i in items] == ["concept_frame.components", "concept_frame.components", "building_blocks.open_source"]
    out = rj.blind(items, 7)
    assert [r["review_id"] for r in out] == ["rj-000", "rj-001", "rj-002"]
    assert sorted(r["earlier_id"] for r in out) == sorted(i["earlier_id"] for i in items)
    assert rj.blind(items, 7) == out


def test_summary_counts_earlier_and_new_judgements_per_study():
    key = [{"review_id": "rj-0", "study": "a", "earlier": 1, "field": "concept_frame.components[M:x].role"},
           {"review_id": "rj-1", "study": "a", "earlier": 1, "field": "concept_frame.components[M:y].role"},
           {"review_id": "rj-2", "study": "b", "earlier": 0, "field": "concept_frame.components[M:z].role"}]
    out = rj.summary(key, {"rj-0": 1, "rj-1": 0, "rj-2": 0})
    assert out["a"]["concept_frame.components"] == {"n": 2, "earlier_supported": 2, "now_supported": 1, "agreement": 0.5}
    assert out["b"]["concept_frame.components"]["agreement"] == 1.0


def test_sheet_columns_hide_the_study_and_the_earlier_judgement():
    assert not {"study", "earlier", "earlier_id", "rank", "rep", "field"} & set(rj.COLUMNS)


def test_others_takes_every_field_outside_the_named_groups():
    picked = rj.pick(rows("a"), "x", ["others"])
    assert [i["field"] for i in picked] == ["summary", "concept_frame.target_task"]
    both = rj.pick(rows("a"), "x", ["components", "open_source"]) + picked
    assert len(both) == len(rows("a"))


def test_full_splits_business_fields_and_keeps_earlier_and_new_rates():
    items = [{"study": "a", "field": "jtbd", "earlier": 0, "now": 1}, {"study": "a", "field": "summary", "earlier": 1, "now": 1},
             {"study": "a", "field": "concept_frame.components[M:x].role", "earlier": 0, "now": 0}]
    out = rj.full(items)["a"]
    assert out["overall"]["earlier"]["supported"] == 1 and out["overall"]["now"]["supported"] == 2
    assert out["business"]["now"]["n"] == 1 and out["without_business"]["now"]["n"] == 2
