"""Post-hoc descriptive classification of the automatic check flags (design 9.1).

Usage: python scripts/quality/04_checks_review.py

The rules below were defined after viewing the flags of results/briefing_checks.json
(research log 2026-10-05). They explain the flags; they do not change the checks,
the flags or any pre-registered metric, and the main conclusions do not use them.

- unmarked_threshold: all numbers equal DVF axis scores -> dvf_score_only;
  else the sentence contains "propos" -> other_wording; else no_marker (violation)
- number_not_in_cited_evidence: every number outside the cited evidence and the
  DVF table is a digit of a candidate term name (for example GPT-2) -> term_name_digit;
  else violation
- dvf_number_mismatch: every available DVF axis score of the candidate appears in
  the sentence -> order_artifact; else violation
- riskiest_axis_not_named: the riskiest-assumption text names the axis by its
  letter ("axis V", "(V)") -> letter_form; else not_named (violation)
- missing_designation_not_stated: the text says "no evidence" -> other_wording;
  else not_stated (violation)

Output: results/briefing_checks_review.json
"""
import datetime
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx import quality as qu

CHECKS = Path("results/briefing_checks.json")
BUNDLE_DIR = Path("data/derived/briefing/bundles")
OUT_DIR = Path("briefings")
RESULTS = Path("results")
LETTER = re.compile(r"\baxis\s*\(?\s*[DFV]\b|\(\s*[DFV]\s*\)")
VIOLATIONS = {"no_marker", "violation", "not_named", "not_stated"}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def term_digits(bundle: dict) -> set[float]:
    digits = set()
    for term in bundle["triad"]:
        digits |= {float(d) for d in re.findall(r"\d+(?:\.\d+)?", term["name"])}
    return digits


def main() -> None:
    flags = json.loads(CHECKS.read_text())["flags"]
    bundles, outputs = {}, {}
    rows, counts = [], Counter()
    for item in flags:
        rank, rep = item["rank"], item["rep"]
        if rank not in bundles:
            bundles[rank] = json.loads((BUNDLE_DIR / f"rank_{rank:02d}.json").read_text())
        bundle = bundles[rank]
        scores = qu.dvf_numbers(bundle)
        sentence = item["sentence"] or ""
        nums = qu.numbers(sentence)
        for flag in item["flags"]:
            if flag == "unmarked_threshold":
                if nums and all(n in scores for n in nums):
                    category = "dvf_score_only"
                elif "propos" in sentence.lower():
                    category = "other_wording"
                else:
                    category = "no_marker"
            elif flag == "number_not_in_cited_evidence":
                texts = qu.evidence_texts(bundle)
                key = (rank, rep)
                if key not in outputs:
                    outputs[key] = json.loads((OUT_DIR / f"rank_{rank:02d}" / f"rep_{rep}.json").read_text())
                cited = next((it for it in qu.evidential_items(outputs[key]) if it["field"] == item["field"]), None)
                cited_text = " ".join(texts.get(i, "") for i in (cited or {}).get("evidence_ids", []))
                known = set(qu.numbers(cited_text)) | scores
                outside = {n for n in nums if n not in known}
                category = "term_name_digit" if outside and outside <= term_digits(bundle) else "violation"
            elif flag == "dvf_number_mismatch":
                category = "order_artifact" if scores and scores <= set(nums) else "violation"
            elif flag in ("riskiest_axis_not_named", "missing_designation_not_stated"):
                key = (rank, rep)
                if key not in outputs:
                    outputs[key] = json.loads((OUT_DIR / f"rank_{rank:02d}" / f"rep_{rep}.json").read_text())
                text = outputs[key]["briefing"]["riskiest_assumption"]["text"]
                if flag == "riskiest_axis_not_named":
                    category = "letter_form" if LETTER.search(text) else "not_named"
                else:
                    category = "other_wording" if "no evidence" in text.lower() else "not_stated"
            else:
                category = "unclassified"
            counts[(flag, category)] += 1
            rows.append({"rank": rank, "rep": rep, "field": item["field"], "flag": flag, "category": category})

    table = {}
    for (flag, category), n in sorted(counts.items()):
        table.setdefault(flag, {})[category] = n
    violations = sum(n for (flag, category), n in counts.items() if category in VIOLATIONS)
    result = {
        "meta": {
            "step": "briefing_checks_review",
            "created": datetime.date.today().isoformat(),
            "git_commit": git_commit(),
            "post_hoc": True,
            "note": "Descriptive classification defined after viewing the flags; not a pre-registered metric and not used for the main conclusions.",
            "checks_sha256": hashlib.sha256(CHECKS.read_bytes()).hexdigest(),
            "violation_categories": sorted(VIOLATIONS),
        },
        "flags_total": sum(counts.values()),
        "violations": violations,
        "by_flag": table,
        "items": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "briefing_checks_review.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"flags_total": result["flags_total"], "violations": violations, "by_flag": table}, indent=1))
    print("Saved", RESULTS / "briefing_checks_review.json")


if __name__ == "__main__":
    main()
