"""Fifth-study (faithfulness) changes: prompts v4, business basis, and unchanged handling of earlier outputs."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triadrx import briefing as br
from triadrx import quality as qu


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_evidential_items(output: dict) -> list[dict]:
    """The evidential split before the business basis existed (copied for comparison)."""
    items = []
    frame = output.get("concept_frame") or {}
    if frame.get("status") == "complete":
        for comp in frame.get("components", []):
            items.append({"field": f"concept_frame.components[{comp.get('term_id')}].role",
                          "text": comp.get("role", ""), "evidence_ids": comp.get("evidence_ids", [])})
        for name in ("target_task", "limitation", "combination"):
            part = frame.get(name) or {}
            items.append({"field": f"concept_frame.{name}", "text": part.get("text", ""), "evidence_ids": part.get("evidence_ids", [])})
    elif frame.get("status") == "insufficient_concept":
        items.append({"field": "concept_frame.insufficient_explanation", "text": frame.get("insufficient_explanation", ""),
                      "evidence_ids": frame.get("insufficient_evidence_ids", [])})
    briefing = output.get("briefing") if output.get("briefing_valid") else None
    if briefing:
        for name in ("summary", "target_user", "jtbd", "value_prop"):
            items.append({"field": name, "text": briefing[name]["text"], "evidence_ids": briefing[name]["evidence_ids"]})
        for item in briefing["building_blocks"]["open_source"]:
            items.append({"field": f"building_blocks.open_source[{item['id']}].use", "text": item["use"], "evidence_ids": item["evidence_ids"]})
    return items


def earlier_outputs() -> list[dict]:
    paths = sorted((ROOT / "briefings").glob("rank_*/rep_*.json")) + sorted((ROOT / "represcription/briefings").glob("rank_*/rep_*.json"))
    return [json.loads(p.read_text()) for p in paths]


def test_earlier_outputs_split_exactly_as_before():
    outs = earlier_outputs()
    if not outs:
        pytest.skip("no earlier briefings in this checkout")
    for out in outs:
        assert qu.evidential_items(out) == old_evidential_items(out)
        assert not any(i["field"].endswith(".hypothesis") for i in qu.prescriptive_items(out))


def test_hypothesis_business_fields_move_to_prescriptive():
    field = lambda basis: {"basis": basis, "text": "t", "evidence_ids": ["p:1"]}
    briefing = {"summary": {"text": "s", "evidence_ids": ["p:1"]}, "target_user": field("hypothesis"), "jtbd": field("evidence"),
                "value_prop": field("hypothesis"), "riskiest_assumption": {"text": "r", "evidence_ids": ["dvf"]},
                "mvp": {"text": "m", "evidence_ids": ["dvf"]}, "success_metric": {"text": "x", "evidence_ids": ["dvf"]},
                "kill_criteria": [], "building_blocks": {"open_source": [], "to_build": []}}
    out = {"concept_frame": {}, "briefing": briefing, "briefing_valid": True}
    assert [i["field"] for i in qu.evidential_items(out)] == ["summary", "jtbd"]
    assert [i["field"] for i in qu.prescriptive_items(out)][:2] == ["target_user.hypothesis", "value_prop.hypothesis"]


def test_schema_adds_the_basis_only_when_asked():
    bundle = {"papers": [{"id": "p:1"}], "repositories": [], "packages": [], "patents": [], "dvf": {"id": "dvf"}}
    try:
        plain, with_basis = br.briefing_schema(bundle), br.briefing_schema(bundle, True)
    except (KeyError, TypeError):
        pytest.skip("bundle shape needs more fields for bundle_ids")
    assert "basis" not in plain["properties"]["target_user"]["properties"]
    for f in br.BUSINESS_FIELDS:
        assert with_basis["properties"][f]["required"][0] == "basis"
        assert with_basis["properties"][f]["properties"]["basis"]["enum"] == ["evidence", "hypothesis"]
    assert plain["properties"]["summary"] == with_basis["properties"]["summary"]


def test_prompts_v4_change_only_the_planned_rules():
    f3, f4 = (ROOT / "prompts/frame_v3.txt").read_text(), (ROOT / "prompts/frame_v4.txt").read_text()
    assert f4.startswith(f3.rstrip("\n")) and f4.rstrip("\n").splitlines()[-1].startswith("6. ")
    b3 = (ROOT / "prompts/briefing_v3.txt").read_text().splitlines()
    b4 = (ROOT / "prompts/briefing_v4.txt").read_text().splitlines()
    changed = [l3[:2] for l3, l4 in zip(b3, b4) if l3 != l4]
    assert len(b3) == len(b4) and changed == ["5.", "9."]


su = load("faithfulness_setup", "scripts/faithfulness/01_setup_run.py")
cp = load("faithfulness_compare", "scripts/faithfulness/02_compare.py")


def test_run_config_swaps_only_prompt_basis_and_q1(tmp_path):
    study = json.loads((ROOT / "config/study.json").read_text())
    out = su.run_config(study, study["faithfulness"])
    gen, base = out["applied"]["briefing"]["generation"], study["applied"]["briefing"]["generation"]
    assert gen["prompt_version"] == "v4" and gen["business_basis"] is True
    keep = lambda d: {k: v for k, v in d.items() if k not in ("prompt_version", "business_basis")}
    assert keep(gen) == keep(base)
    assert out["quality"]["q1"]["seed"] == study["faithfulness"]["q1"]["seed"] != study["represcription"]["q1"]["seed"]
    prereg = tmp_path / "p.md"
    prereg.write_text("x\n고정 커밋: 미정\n")
    with pytest.raises(SystemExit):
        su.require_frozen({"faithfulness": {"prereg_file": str(prereg)}})


def test_business_basis_counts_and_field_q1():
    def out(basis):
        f = {"basis": basis, "text": "t", "evidence_ids": ["p:1"]} if basis else {"text": "t", "evidence_ids": ["p:1"]}
        return {"briefing_valid": True, "briefing": {"target_user": f, "jtbd": f, "value_prop": f}}
    counts = cp.business_basis([out("hypothesis"), out("evidence"), out(None)], ["target_user", "jtbd", "value_prop"])
    assert counts["hypothesis"] == 3 and counts["fields"] == 9
    q1 = {"by_field": {"summary": {"supported": 3, "n": 4}, "concept_frame": {"supported": 1, "n": 2}}}
    r = cp.fields_q1(q1, ["concept_frame", "summary", "building_blocks"])
    assert r["supported"] == 4 and r["n"] == 6 and cp.fields_q1(q1, ["jtbd"])["rate"] is None
