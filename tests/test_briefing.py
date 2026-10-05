import copy

import pytest

from triadrx import briefing as b


@pytest.fixture
def bundle():
    return {
        "rank": 1,
        "triad": [{"term_id": "M:a", "name": "a", "type": "M"}, {"term_id": "M:b", "name": "b", "type": "M"},
                  {"term_id": "T:c", "name": "c", "type": "T"}],
        "papers": [{"id": "paper:1"}, {"id": "paper:2"}],
        "repositories": [{"id": "repo:o/r"}],
        "packages": [{"id": "pkg:p", "version_before_cutoff": "1.2.0"}],
        "patents": [{"ids": ["patent:9", "patent:8"]}],
        "dvf": {"id": "dvf:1", "riskiest_axis": "V", "riskiest_reason": "missing", "mvp_type": "customer interviews"},
    }


def complete_frame():
    comp = lambda t: {"term_id": t, "role": "r", "evidence_ids": ["paper:1"]}
    item = {"text": "x", "evidence_ids": ["paper:2"]}
    return {"status": "complete", "insufficient_reason": "none", "components": [comp("M:a"), comp("M:b"), comp("T:c")],
            "target_task": item, "limitation": item, "combination": item}


def test_ids_exclude_dvf_in_stage_one(bundle):
    assert "dvf:1" not in b.bundle_ids(bundle, include_dvf=False)
    assert "dvf:1" in b.bundle_ids(bundle, include_dvf=True)
    assert "patent:8" in b.bundle_ids(bundle, include_dvf=False)


def test_schema_restricts_ids_and_terms(bundle):
    s = b.frame_schema(bundle)
    assert s["properties"]["components"]["items"]["properties"]["term_id"]["enum"] == ["M:a", "M:b", "T:c"]
    assert "dvf:1" not in s["properties"]["target_task"]["properties"]["evidence_ids"]["items"]["enum"]
    bs = b.briefing_schema(bundle)
    assert bs["properties"]["building_blocks"]["properties"]["open_source"]["items"]["properties"]["id"]["enum"] == ["pkg:p", "repo:o/r"]


def test_frame_validation(bundle):
    assert b.validate_frame(complete_frame(), bundle) == []
    f = complete_frame()
    f["components"] = f["components"][:2]
    assert b.validate_frame(f, bundle)
    f = complete_frame()
    f["insufficient_reason"] = "comparison_only"
    assert b.validate_frame(f, bundle)
    assert b.validate_frame({"status": "insufficient_concept", "insufficient_reason": "none"}, bundle)
    ok = {"status": "insufficient_concept", "insufficient_reason": "comparison_only",
          "insufficient_explanation": "compared", "insufficient_evidence_ids": ["paper:1"]}
    assert b.validate_frame(ok, bundle) == []
    assert b.validate_frame({**ok, "insufficient_evidence_ids": []}, bundle)
    assert b.validate_frame({**ok, "insufficient_evidence_ids": ["paper:404"]}, bundle)
    assert b.validate_frame({**complete_frame(), "insufficient_evidence_ids": ["paper:1"]}, bundle)


def test_instruction_mentions_missing(bundle):
    text = b.briefing_instruction("{dvf_id} {axis} {axis_name} {missing_note}{mvp_type}", bundle)
    assert "viability" in text and "missing evidence" in text and "customer interviews" in text


def test_attach_fixed_values(bundle):
    item = {"text": "x", "evidence_ids": ["paper:1"]}
    gen = {k: item for k in ("summary", "target_user", "jtbd", "value_prop", "riskiest_assumption", "mvp", "success_metric")}
    gen["kill_criteria"] = [item]
    gen["building_blocks"] = {"open_source": [{"id": "pkg:p", "use": "u", "evidence_ids": ["pkg:p"]},
                                              {"id": "repo:o/r", "use": "u", "evidence_ids": ["repo:o/r"]}],
                              "to_build": [item]}
    assert b.validate_briefing(gen, bundle) == []
    out = b.attach_fixed(copy.deepcopy(gen), complete_frame(), bundle)
    assert out["riskiest_assumption"]["axis"] == "V"
    assert out["riskiest_assumption"]["designated_by_missing_evidence"] is True
    assert out["mvp"]["type"] == "customer interviews"
    assert out["building_blocks"]["open_source"][0]["version"] == "1.2.0"
    assert out["building_blocks"]["open_source"][1]["version"] is None
