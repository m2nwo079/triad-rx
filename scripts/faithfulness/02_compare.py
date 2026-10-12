"""Decision and comparison of the faithfulness study (faithfulness preregistration 3).

Usage: python scripts/faithfulness/02_compare.py   (from the repository root)

Main decision (one): "improved" only if the Wilson 95% lower bound of this study's Q1 exceeds the Q1 point
estimate of the base run (faithfulness.decision). Everything else is descriptive and set next to the base run:
Q1 of the evidential fields without the business fields, the business fields marked as hypotheses (count and
share over all valid briefings), Q1 of the business-field sentences kept as evidence, evidential sentences by
field in the population, concept-insufficient frames with reasons, Q2 and the automatic checks.

Output: faithfulness.run_dir/results/comparison_represcription.json
"""
import datetime
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.quality import evidential_items, is_hypothesis
from triadrx.study import load_study, runtime_environment


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def fields_q1(q1: dict, fields: list[str]) -> dict:
    k = sum(q1["by_field"].get(f, {}).get("supported", 0) for f in fields)
    n = sum(q1["by_field"].get(f, {}).get("n", 0) for f in fields)
    return {"supported": k, "n": n, "rate": k / n if n else None, "wilson_95": wilson(k, n)}


def outputs(run: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run / "briefings").glob("rank_*/rep_*.json"))]


def business_basis(outs: list[dict], fields: list[str]) -> dict:
    """Basis of the business fields over all valid briefings; outputs without a basis count as evidence."""
    counts = {f: Counter() for f in fields}
    for out in outs:
        briefing = out.get("briefing") if out.get("briefing_valid") else None
        if briefing:
            for f in fields:
                counts[f]["hypothesis" if is_hypothesis(briefing[f]) else "evidence"] += 1
    total = sum(sum(c.values()) for c in counts.values())
    hyp = sum(c["hypothesis"] for c in counts.values())
    return {"by_field": {f: dict(c) for f, c in counts.items()}, "fields": total, "hypothesis": hyp,
            "hypothesis_share": hyp / total if total else None}


def evidential_counts(outs: list[dict]) -> dict:
    counts = Counter()
    for out in outs:
        for item in evidential_items(out):
            counts[item["field"].split(".")[0].split("[")[0]] += 1
    return dict(counts)


def side(run: Path, cfg: dict) -> dict:
    res = lambda name: json.loads((run / "results" / f"{name}.json").read_text())
    q1, gen, q2, checks = res("q1"), res("briefing_generation"), res("q2"), res("briefing_checks")
    outs = outputs(run)
    s = gen["summary"]
    return {"q1": {"supported": q1["supported"], "n": q1["n"], "rate": q1["rate"], "wilson_95": q1["wilson_95"]},
            "q1_without_business": fields_q1(q1, cfg["evidential_fields_without_business"]),
            "q1_business_evidence": fields_q1(q1, cfg["business_fields"]),
            "business_basis": business_basis(outs, cfg["business_fields"]),
            "evidential_items_by_field": evidential_counts(outs),
            "frames": {"runs": s["runs"], "complete": s["frame_complete"], "insufficient": s["frame_insufficient"],
                       "reasons": s["insufficient_reasons"], "briefing_valid": s["briefing_valid"]},
            "q2_frame_agreement": q2["frame_agreement"],
            "checks": {"flagged_items": checks["summary"]["flagged_items"], "flag_counts": checks["summary"]["flag_counts"],
                       "sentences": checks["summary"]["sentences"]}}


def main() -> None:
    study = load_study()
    cfg = study["faithfulness"]
    this, base = side(ROOT / cfg["run_dir"], cfg), side(ROOT / cfg["base_run"], cfg)
    lower, threshold = this["q1"]["wilson_95"][0], base["q1"]["rate"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"meta": {"step": "faithfulness_comparison", "created": datetime.date.today().isoformat(), "git_commit": commit,
                    "environment": runtime_environment(), "decision_rule": cfg["decision"]},
           "decision": {"q1_lower_bound": lower, "threshold": threshold,
                        "verdict": "improved" if lower > threshold else "not shown"},
           "faithfulness": this, "represcription": base}
    path = ROOT / cfg["run_dir"] / "results/comparison_represcription.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"decision: Q1 lower bound {lower:.3f} vs base Q1 {threshold:.3f} -> {out['decision']['verdict']}")
    for name, s in (("faithfulness", this), ("represcription", base)):
        print(name, "Q1", round(s["q1"]["rate"], 3), "without business", s["q1_without_business"]["supported"], "/",
              s["q1_without_business"]["n"], "business evidence", s["q1_business_evidence"]["supported"], "/",
              s["q1_business_evidence"]["n"], "hypothesis share", s["business_basis"]["hypothesis_share"],
              "insufficient", s["frames"]["insufficient"], "Q2", s["q2_frame_agreement"]["rate"])
    print("Saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
