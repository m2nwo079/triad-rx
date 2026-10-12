"""Descriptive comparison with the first-study prescription (represcription preregistration 7).

Usage: python scripts/represcription/08_compare.py   (from the repository root)

Nothing here is a verdict. Both prescriptions use the same data cutoff and the same downstream rules; only the
candidate engine differs. First-study term ids are mapped to the representative of their merge group in the
combined synonym decisions (represcription.review.combined.synonyms) before triads are compared, so a triad
that changed only by a synonym merge counts as shared. Items: top-k overlap, Spearman correlation of the
prescription ranks of shared triads, DVF axis scores, riskiest axis and MVP type, concept-insufficient share
with reasons, Q1 overall and by field, Q2.

Output: represcription.run_dir/results/comparison_first_study.json
"""
import datetime
import json
import math
import sys
from collections import Counter
from pathlib import Path

import subprocess

import numpy as np

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment


def canonical(groups: dict[str, list[str]]) -> dict[str, str]:
    """Each merged member id to the representative of its group."""
    return {m: rep for rep, members in groups.items() for m in members}


def triad_key(triad: list[str], rep: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted(rep.get(t, t) for t in triad))


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman correlation of two rank lists without ties; None below three pairs."""
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def overlap(first: list[dict], fourth: list[dict], rep: dict[str, str]) -> dict:
    a = {triad_key(e["triad"], rep): e["rank"] for e in first}
    b = {triad_key(e["triad"], rep): e["rank"] for e in fourth}
    shared = sorted(set(a) & set(b), key=lambda t: b[t])
    terms_a = {t for key in a for t in key}
    terms_b = {t for key in b for t in key}
    return {"top_k": len(fourth), "shared_triads": len(shared),
            "shared": [{"triad": list(t), "rank_first": a[t], "rank_fourth": b[t]} for t in shared],
            "shared_rank_spearman": spearman([a[t] for t in shared], [b[t] for t in shared]),
            "distinct_terms": {"first": len(terms_a), "fourth": len(terms_b), "shared": len(terms_a & terms_b)}}


def dvf_summary(dvf: dict) -> dict:
    cands = dvf["candidates"]
    axes = {}
    for axis in ("D", "V", "F"):
        scores = [c["axes"][axis]["score"] for c in cands if c["axes"][axis]["score"] is not None]
        axes[axis] = {"missing": len(cands) - len(scores),
                      "quartiles": [float(q) for q in np.percentile(scores, [25, 50, 75])] if scores else None,
                      "mean": float(np.mean(scores)) if scores else None}
    return {"candidates": len(cands), "axes": axes,
            "riskiest_axis": dict(Counter(c["riskiest_axis"] for c in cands)),
            "riskiest_reason": dict(Counter(c["riskiest_reason"] for c in cands)),
            "mvp_type": dict(Counter(c["mvp_type"] for c in cands))}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def quality(gen: dict, q1: dict, q2: dict) -> dict:
    s = gen["summary"]
    business = [f for f in ("target_user", "jtbd", "value_prop") if f in q1["by_field"]]
    bk = sum(q1["by_field"][f]["supported"] for f in business)
    bn = sum(q1["by_field"][f]["n"] for f in business)
    return {"concept_insufficient": {"runs": s["runs"], "insufficient": s["frame_insufficient"],
                                     "share": s["frame_insufficient"] / s["runs"], "reasons": s["insufficient_reasons"]},
            "q1": {"supported": q1["supported"], "n": q1["n"], "rate": q1["rate"], "wilson_95": q1["wilson_95"],
                   "by_field": {f: {"supported": v["supported"], "n": v["n"]} for f, v in q1["by_field"].items()},
                   "business_fields": {"supported": bk, "n": bn, "rate": bk / bn if bn else None,
                                       "wilson_95": wilson(bk, bn) if bn else None}},
            "q2_frame_agreement": q2["frame_agreement"]}


def main() -> None:
    study = load_study()
    rcfg = study["represcription"]
    run = ROOT / rcfg["run_dir"]
    run_cfg = json.loads((run / "config/study.json").read_text())
    groups = json.loads((ROOT / rcfg["review"]["combined"]["synonyms"]).read_text())["merge_groups"]
    rep = canonical(groups)
    first_reg = json.loads((ROOT / study["prospective"]["registration"]).read_text())["top_k_prescription"]
    fourth_reg = json.loads((run / run_cfg["prospective"]["registration"]).read_text())["top_k_prescription"]

    def results(base: Path, name: str) -> dict:
        return json.loads((base / "results" / f"{name}.json").read_text())

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"meta": {"step": "represcription_comparison", "created": datetime.date.today().isoformat(), "git_commit": commit,
                    "environment": runtime_environment(),
                    "note": "Descriptive only (represcription preregistration 7); same data cutoff and downstream rules, engine differs.",
                    "first_registration": study["prospective"]["registration"],
                    "fourth_registration": str(Path(rcfg["run_dir"]) / run_cfg["prospective"]["registration"])},
           "prescription": overlap(first_reg, fourth_reg, rep),
           "dvf": {"first": dvf_summary(results(ROOT, "dvf")), "fourth": dvf_summary(results(run, "dvf"))},
           "briefing_quality": {
               "first": quality(results(ROOT, "briefing_generation"), results(ROOT, "q1"), results(ROOT, "q2")),
               "fourth": quality(results(run, "briefing_generation"), results(run, "q1"), results(run, "q2"))}}
    path = run / "results/comparison_first_study.json"
    path.write_text(json.dumps(out, indent=2))
    p = out["prescription"]
    print(f"top {p['top_k']}: shared {p['shared_triads']}, spearman {p['shared_rank_spearman']}, terms {p['distinct_terms']}")
    for side in ("first", "fourth"):
        d, q = out["dvf"][side], out["briefing_quality"][side]
        print(side, "V missing", d["axes"]["V"]["missing"], "riskiest", d["riskiest_axis"],
              "insufficient", q["concept_insufficient"]["insufficient"], "Q1", round(q["q1"]["rate"], 3),
              "business", q["q1"]["business_fields"]["supported"], "/", q["q1"]["business_fields"]["n"])
    print("Saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
