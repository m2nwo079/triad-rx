"""Response schemas and validation for the two-stage briefing generation (design 8.2, 8.3).

Schemas use the Gemini responseSchema subset (OBJECT, ARRAY, STRING with enum,
required, minItems). Evidence IDs are restricted to the IDs of the candidate's own
bundle, so an ID outside the bundle cannot be produced. DVF-derived values
(riskiest axis, MVP type, package versions) are not generated; they are attached
from the bundle after generation.
"""

AXIS_NAMES = {"D": "desirability", "F": "feasibility", "V": "viability"}

# Field classes (design 8.2, preregistration 5.5): Q1 samples sentences from evidential fields only.
# Prescriptive fields propose experiments and thresholds and are checked by rules instead.
EVIDENTIAL_FIELDS = ["concept_frame", "summary", "target_user", "jtbd", "value_prop", "building_blocks.open_source.use"]
PRESCRIPTIVE_FIELDS = ["riskiest_assumption", "mvp", "success_metric", "kill_criteria", "building_blocks.to_build"]
REASONS = ["none", "comparison_only", "term_mismatch", "no_combination_evidence", "other"]


def bundle_ids(bundle: dict, include_dvf: bool) -> list[str]:
    ids = [p["id"] for p in bundle["papers"]]
    ids += [r["id"] for r in bundle["repositories"]]
    ids += [k["id"] for k in bundle["packages"]]
    ids += [i for p in bundle["patents"] for i in p["ids"]]
    if include_dvf:
        ids.append(bundle["dvf"]["id"])
    return sorted(set(ids))


def frame_input(bundle: dict) -> dict:
    """Stage 1 sees the evidence but not the DVF table."""
    return {k: bundle[k] for k in ("rank", "triad", "papers", "repositories", "packages", "patents")}


def _ids(allowed: list[str]) -> dict:
    return {"type": "ARRAY", "items": {"type": "STRING", "enum": allowed}, "minItems": 1}


def _text(allowed: list[str]) -> dict:
    return {
        "type": "OBJECT",
        "properties": {"text": {"type": "STRING"}, "evidence_ids": _ids(allowed)},
        "required": ["text", "evidence_ids"],
        "propertyOrdering": ["text", "evidence_ids"],
    }


def frame_schema(bundle: dict) -> dict:
    allowed = bundle_ids(bundle, include_dvf=False)
    terms = [t["term_id"] for t in bundle["triad"]]
    component = {
        "type": "OBJECT",
        "properties": {
            "term_id": {"type": "STRING", "enum": terms},
            "role": {"type": "STRING"},
            "evidence_ids": _ids(allowed),
        },
        "required": ["term_id", "role", "evidence_ids"],
        "propertyOrdering": ["term_id", "role", "evidence_ids"],
    }
    properties = {
        "status": {"type": "STRING", "enum": ["complete", "insufficient_concept"]},
        "insufficient_reason": {"type": "STRING", "enum": REASONS},
        "insufficient_explanation": {"type": "STRING"},
        "insufficient_evidence_ids": {"type": "ARRAY", "items": {"type": "STRING", "enum": allowed}},
        "components": {"type": "ARRAY", "items": component},
        "target_task": _text(allowed),
        "limitation": _text(allowed),
        "combination": _text(allowed),
    }
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": ["status", "insufficient_reason"],
        "propertyOrdering": list(properties),
    }


def briefing_schema(bundle: dict) -> dict:
    allowed = bundle_ids(bundle, include_dvf=True)
    blocks = sorted([r["id"] for r in bundle["repositories"]] + [k["id"] for k in bundle["packages"]])
    open_item = {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING", "enum": blocks} if blocks else {"type": "STRING"},
            "use": {"type": "STRING"},
            "evidence_ids": _ids(allowed),
        },
        "required": ["id", "use", "evidence_ids"],
        "propertyOrdering": ["id", "use", "evidence_ids"],
    }
    properties = {
        "summary": _text(allowed),
        "target_user": _text(allowed),
        "jtbd": _text(allowed),
        "value_prop": _text(allowed),
        "riskiest_assumption": _text(allowed),
        "mvp": _text(allowed),
        "success_metric": _text(allowed),
        "kill_criteria": {"type": "ARRAY", "items": _text(allowed), "minItems": 1},
        "building_blocks": {
            "type": "OBJECT",
            "properties": {
                "open_source": {"type": "ARRAY", "items": open_item},
                "to_build": {"type": "ARRAY", "items": _text(allowed)},
            },
            "required": ["open_source", "to_build"],
            "propertyOrdering": ["open_source", "to_build"],
        },
    }
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": list(properties),
        "propertyOrdering": list(properties),
    }


def briefing_instruction(template: str, bundle: dict) -> str:
    dvf = bundle["dvf"]
    axis = dvf["riskiest_axis"]
    missing = ""
    if dvf["riskiest_reason"] == "missing":
        missing = ("It was designated because the bundle has no evidence for this axis; "
                   "say explicitly that it was designated due to missing evidence. ")
    return template.format(dvf_id=dvf["id"], axis=axis, axis_name=AXIS_NAMES[axis],
                           missing_note=missing, mvp_type=dvf["mvp_type"])


def _has_ids(obj, allowed: set) -> bool:
    return isinstance(obj, dict) and bool(obj.get("evidence_ids")) and set(obj["evidence_ids"]) <= allowed


def validate_frame(frame: dict, bundle: dict) -> list[str]:
    """Problems that make a stage-1 output unusable; empty when valid."""
    allowed = set(bundle_ids(bundle, include_dvf=False))
    problems = []
    status = frame.get("status")
    reason = frame.get("insufficient_reason")
    if status == "complete":
        if reason != "none":
            problems.append("complete frame with an insufficient_reason")
        if frame.get("insufficient_evidence_ids"):
            problems.append("complete frame with insufficient_evidence_ids")
        terms = sorted(t["term_id"] for t in bundle["triad"])
        got = sorted(c.get("term_id") for c in frame.get("components", []))
        if got != terms:
            problems.append(f"components do not cover the three terms exactly once: {got}")
        for c in frame.get("components", []):
            if not c.get("role") or not _has_ids(c, allowed):
                problems.append(f"component {c.get('term_id')} lacks role or evidence")
        for field in ("target_task", "limitation", "combination"):
            if not _has_ids(frame.get(field), allowed) or not frame[field].get("text"):
                problems.append(f"{field} lacks text or evidence")
    elif status == "insufficient_concept":
        if reason not in REASONS or reason == "none":
            problems.append("insufficient frame without a reason")
        ids = frame.get("insufficient_evidence_ids") or []
        if not ids or not set(ids) <= allowed:
            problems.append("insufficient frame without evidence ids for its reason")
        if not frame.get("insufficient_explanation"):
            problems.append("insufficient frame without an explanation")
    else:
        problems.append(f"unknown status {status!r}")
    return problems


def validate_briefing(briefing: dict, bundle: dict) -> list[str]:
    allowed = set(bundle_ids(bundle, include_dvf=True))
    problems = []
    for field in ("summary", "target_user", "jtbd", "value_prop", "riskiest_assumption", "mvp", "success_metric"):
        item = briefing.get(field)
        if not _has_ids(item, allowed) or not item.get("text"):
            problems.append(f"{field} lacks text or evidence")
    kills = briefing.get("kill_criteria") or []
    if not kills or not all(_has_ids(k, allowed) and k.get("text") for k in kills):
        problems.append("kill_criteria empty or lacking evidence")
    blocks = briefing.get("building_blocks") or {}
    for item in blocks.get("open_source", []):
        if not _has_ids(item, allowed):
            problems.append(f"open_source {item.get('id')} lacks evidence")
    for item in blocks.get("to_build", []):
        if not _has_ids(item, allowed):
            problems.append("to_build item lacks evidence")
    return problems


def attach_fixed(briefing: dict, frame: dict, bundle: dict) -> dict:
    """Add the stage-1 frame and the DVF- and bundle-derived values the model may not set."""
    dvf = bundle["dvf"]
    versions = {k["id"]: k.get("version_before_cutoff") for k in bundle["packages"]}
    out = {"concept_frame": frame, **briefing}
    out["riskiest_assumption"] = {**briefing["riskiest_assumption"], "axis": dvf["riskiest_axis"],
                                  "designated_by_missing_evidence": dvf["riskiest_reason"] == "missing"}
    out["mvp"] = {**briefing["mvp"], "type": dvf["mvp_type"]}
    blocks = briefing["building_blocks"]
    out["building_blocks"] = {
        "open_source": [{**i, "version": versions.get(i["id"], None) if i["id"].startswith("pkg:") else None}
                        for i in blocks["open_source"]],
        "to_build": blocks["to_build"],
    }
    return out
