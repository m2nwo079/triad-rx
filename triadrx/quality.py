"""Shared helpers for briefing quality checks (design 9, preregistration 5.5).

Sentence units, evidential and prescriptive items of a generated output,
numbers in text, evidence text of bundle items, and Wilson intervals.
"""
import math
import re

Z = 1.959963984540054
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
AXIS_WORDS = {"desirability": "D", "feasibility": "F", "viability": "V"}
# Evidence ids written inside a sentence (paper:2609.20888, dvf:14) are citations, not numeric claims
EVIDENCE_ID = re.compile(r"\b(?:paper|repo|pkg|patent|dvf):[^\s,;)\]]+", re.IGNORECASE)


def strip_ids(text: str) -> str:
    return EVIDENCE_ID.sub(" ", text or "")


def sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def numbers(text: str) -> list[float]:
    """Numbers in text, ignoring evidence ids and digits inside words (YOLOv8, 3D)."""
    return [float(m) for m in NUMBER.findall(strip_ids(text))]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - margin, centre + margin


def evidential_items(output: dict) -> list[dict]:
    """(field, text, evidence_ids) of the evidential fields of one generated output."""
    items = []
    frame = output.get("concept_frame") or {}
    if frame.get("status") == "complete":
        for comp in frame.get("components", []):
            items.append({"field": f"concept_frame.components[{comp.get('term_id')}].role",
                          "text": comp.get("role", ""), "evidence_ids": comp.get("evidence_ids", [])})
        for name in ("target_task", "limitation", "combination"):
            part = frame.get(name) or {}
            items.append({"field": f"concept_frame.{name}", "text": part.get("text", ""),
                          "evidence_ids": part.get("evidence_ids", [])})
    elif frame.get("status") == "insufficient_concept":
        items.append({"field": "concept_frame.insufficient_explanation", "text": frame.get("insufficient_explanation", ""),
                      "evidence_ids": frame.get("insufficient_evidence_ids", [])})
    briefing = output.get("briefing") if output.get("briefing_valid") else None
    if briefing:
        for name in ("summary", "target_user", "jtbd", "value_prop"):
            items.append({"field": name, "text": briefing[name]["text"], "evidence_ids": briefing[name]["evidence_ids"]})
        for item in briefing["building_blocks"]["open_source"]:
            items.append({"field": f"building_blocks.open_source[{item['id']}].use", "text": item["use"],
                          "evidence_ids": item["evidence_ids"]})
    return items


def prescriptive_items(output: dict) -> list[dict]:
    briefing = output.get("briefing") if output.get("briefing_valid") else None
    if not briefing:
        return []
    items = []
    for name in ("riskiest_assumption", "mvp", "success_metric"):
        items.append({"field": name, "text": briefing[name]["text"], "evidence_ids": briefing[name]["evidence_ids"]})
    for i, item in enumerate(briefing["kill_criteria"]):
        items.append({"field": f"kill_criteria[{i}]", "text": item["text"], "evidence_ids": item["evidence_ids"]})
    for i, item in enumerate(briefing["building_blocks"]["to_build"]):
        items.append({"field": f"building_blocks.to_build[{i}]", "text": item["text"], "evidence_ids": item["evidence_ids"]})
    return items


def evidence_texts(bundle: dict) -> dict[str, str]:
    """Readable text of every bundle item, keyed by evidence id."""
    texts = {}
    for p in bundle["papers"]:
        texts[p["id"]] = f"{p['title']}\n{p['abstract']}"
    for r in bundle["repositories"]:
        texts[r["id"]] = f"repository {r['id'][5:]}; stars in window {r['stars_window']:g}; description: {r.get('description') or '(none)'}"
    for k in bundle["packages"]:
        texts[k["id"]] = (f"package {k['id'][4:]}; downloads first half {k['downloads_first_half']:g}, "
                          f"second half {k['downloads_second_half']:g}; version before cutoff {k.get('version_before_cutoff') or 'unknown'}")
    for p in bundle["patents"]:
        text = f"{p['title']}\n{p['abstract']}\nassignee categories: {', '.join(p['assignee_categories']) or '(none)'}"
        for i in p["ids"]:
            texts[i] = text
    dvf = bundle["dvf"]
    axes = "; ".join(f"{a} score {v['score'] if v['score'] is not None else 'missing'}" for a, v in dvf["axes"].items())
    texts[dvf["id"]] = (f"DVF table: {axes}; riskiest axis {dvf['riskiest_axis']} ({dvf['riskiest_reason']}); "
                        f"MVP type {dvf['mvp_type']}")
    return texts


def dvf_numbers(bundle: dict) -> set[float]:
    return {float(v["score"]) for v in bundle["dvf"]["axes"].values() if v["score"] is not None}
