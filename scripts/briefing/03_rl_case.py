"""Reinforcement learning case and comparison with tod (design 8.6; descriptive, not pre-registered).

Usage: python scripts/briefing/03_rl_case.py

RL candidates: briefing targets with a term whose id contains one of
applied.rl_case.term_patterns. Every RL candidate is reported; candidates whose
frame was complete in all repetitions are marked for the deep dive. For each RL
candidate: frame status and reason per repetition, Q2 agreement, DVF axes and the
riskiest axis with its bootstrap shares, evidence counts, and the Q1 sample
sentences of that candidate (counts only, from the local review file).

tod comparison (external/tod, checked against PROVENANCE.md hashes): the LLM-scored
DVF of tod against the indicator-based DVF here, judge agreement in both projects,
and the tod leakage result recomputed from its files (Spearman over its reliable
concepts). No metric or rule of this study is changed by this analysis.

Output: results/rl_case.json
"""
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

TOD = Path("external/tod")
RESULTS = Path("results")
OUT_DIR = Path("briefings")
REVIEW = Path("data/review/q1_review.csv")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_tod() -> dict:
    """Every tod file must match the hash recorded in PROVENANCE.md."""
    recorded = dict(re.findall(r"\| (\S+\.json) \| ([0-9a-f]{64}) \|", (TOD / "PROVENANCE.md").read_text()))
    for name, digest in recorded.items():
        if sha256(TOD / name) != digest:
            sys.exit(f"{TOD / name} differs from PROVENANCE.md")
    return recorded


def spearman(a: list[float], b: list[float]) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main() -> None:
    cfg = load_study()["applied"]["rl_case"]
    patterns = [p.lower() for p in cfg["term_patterns"]]
    recorded = check_tod()

    bundles = {b["rank"]: b for b in json.loads((RESULTS / "briefing_bundles.json").read_text())["bundles"]}
    dvf = {c["rank"]: c for c in json.loads((RESULTS / "dvf.json").read_text())["candidates"]}
    q2 = {c["rank"]: c for c in json.loads((RESULTS / "q2.json").read_text())["per_candidate"]}
    review = []
    if REVIEW.exists():
        with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
            review = list(csv.DictReader(handle))

    rl = []
    for rank in sorted(bundles):
        triad = dvf[rank]["triad"]
        matched = [t for t in triad if any(p in t.lower() for p in patterns)]
        if not matched:
            continue
        reps = [json.loads(p.read_text()) for p in sorted((OUT_DIR / f"rank_{rank:02d}").glob("rep_*.json"))]
        frames = [{"rep": o["rep"], "status": (o["concept_frame"] or {}).get("status"),
                   "reason": (o["concept_frame"] or {}).get("insufficient_reason"), "valid": o["frame_valid"]} for o in reps]
        complete = sum(1 for f in frames if f["status"] == "complete" and f["valid"])
        sample = [r for r in review if int(r["rank"]) == rank]
        d = dvf[rank]
        rl.append({
            "rank": rank, "triad": triad, "matched_terms": matched,
            "frames": frames, "complete_reps": complete, "deep_dive": complete == len(frames),
            "q2": {"mode": q2[rank]["mode"], "agreeing": q2[rank]["agreeing"], "reps": len(q2[rank]["keys"])},
            "dvf": {"axes": {a: {"score": d["axes"][a]["score"], "missing": d["axes"][a]["value"] is None} for a in "DFV"},
                    "riskiest_axis": d["riskiest_axis"], "riskiest_reason": d["riskiest_reason"],
                    "riskiest_bootstrap_share": d["riskiest_bootstrap_share"]},
            "evidence": {k: v for k, v in bundles[rank].items() if k in ("papers", "repositories", "packages", "patents")},
            "q1_sample": {"sentences": len(sample), "supported": sum(int(r["correct"]) for r in sample if r["correct"].strip() in "01" and r["correct"].strip())},
            "briefing_files": [str(OUT_DIR / f"rank_{rank:02d}" / f"rep_{o['rep']}.json") for o in reps],
        })

    # Indicator-based DVF here: how many distinct axis-score profiles among the DVF population
    profiles = Counter(tuple(c["axes"][a]["score"] for a in "DFV") for c in dvf.values())
    axis_spread = {a: dict(Counter(str(c["axes"][a]["score"]) for c in dvf.values())) for a in "DFV"}

    tod_briefs = json.loads((TOD / "briefs.json").read_text())
    tod_dvf = [{"concept_id": b["concept_id"], "rel_growth": b["rel_growth"],
                "total_median": b["dvf_stats"]["total"]["median"], "total_std": b["dvf_stats"]["total"]["std"],
                "F_median": b["dvf_stats"]["F"]["median"]} for b in tod_briefs]
    judges = json.loads((TOD / "judge_agreement.json").read_text())
    leak = json.loads((TOD / "leakage_compare.json").read_text())
    reliable = [c["concept_id"] for c in json.loads((TOD / "three_way.json").read_text()) if c["reliable"]]
    growth = [leak[c]["rel_growth"] for c in reliable]
    leakage = {f"{model}_{kind}": spearman([leak[c][f"{model}_{kind}"] for c in reliable], growth)
               for model in ("g", "q") for kind in ("orig", "blind")}
    q1 = json.loads((RESULTS / "q1.json").read_text())
    cross = json.loads((RESULTS / "q1_cross_model.json").read_text())

    result = {
        "meta": {"step": "rl_case", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "descriptive": True, "note": "Descriptive case analysis (design 8.6); not a pre-registered metric.",
                 "settings": cfg, "tod_files_sha256": recorded},
        "rl_candidates": rl,
        "comparison": {
            "dvf": {
                "tod_llm_scored": {"concepts": tod_dvf,
                                   "distinct_total_medians": len({t["total_median"] for t in tod_dvf}),
                                   "distinct_F_medians": len({t["F_median"] for t in tod_dvf})},
                "triadrx_indicator_based": {"candidates": len(dvf), "distinct_axis_profiles": len(profiles),
                                            "axis_score_counts": axis_spread},
            },
            "judges": {
                "tod_promise_scores_qwen_vs_gemini": {"spearman": judges["spearman"], "exact_match": judges["exact_match"]},
                "triadrx_sentence_faithfulness_human_vs_qwen": {"agreement": cross["agreement"], "cohen_kappa": cross["cohen_kappa"]},
            },
            "tod_leakage_spearman_reliable_concepts": {"concepts": reliable, **leakage},
            "triadrx_q1": {"rate": q1["rate"], "wilson_95": q1["wilson_95"]},
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rl_case.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    for c in rl:
        print(f"rank {c['rank']}: {c['triad']} complete {c['complete_reps']}/{len(c['frames'])} deep_dive {c['deep_dive']}")
    print("tod DVF distinct total medians:", result["comparison"]["dvf"]["tod_llm_scored"]["distinct_total_medians"],
          "| TriadRx distinct axis profiles:", len(profiles), "of", len(dvf))
    print("tod leakage (Gemini) orig -> blind:", round(leakage["g_orig"], 2), "->", round(leakage["g_blind"], 2))
    print("Saved", RESULTS / "rl_case.json")


if __name__ == "__main__":
    main()
