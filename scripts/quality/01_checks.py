"""Automatic rule checks (design 9.1) and Q2 stability (design 9.2, preregistration 5.5).

Usage: python scripts/quality/01_checks.py

Checks on every sentence of every generated output (briefings/rank_NN/rep_R.json):
- cited ids exist in the candidate's bundle
- cited papers are dated on or before data_cutoff
- evidential sentences: numbers appear in the cited evidence or the DVF table, and
  a number written next to an axis name equals that axis score
- prescriptive sentences: every number carries the threshold marker
- any sentence: superiority phrases (quality.checks) only when a cited item contains them
- riskiest_assumption names its axis and, when designated by missing evidence, says so;
  mvp mentions a keyword of its MVP type
Evidence ids written inside sentences are removed before numbers are read
(triadrx.quality.strip_ids; checker bug fixed after the first run, research log 2026-10-05).
Q2: per candidate the modal (status, insufficient_reason) over repetitions; the share
of repetitions equal to the mode, pooled over candidates, with a Wilson interval.
Invalid frames count as disagreeing. The MVP type is fixed by rule and is reported as is.

Outputs: results/briefing_checks.json, results/q2.json
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
from triadrx import briefing as br
from triadrx import quality as qu
from triadrx.study import data_cutoff, load_study

BUNDLE_DIR = Path("data/derived/briefing/bundles")
OUT_DIR = Path("briefings")
RESULTS = Path("results")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def load_outputs() -> dict:
    outputs = {}
    for path in sorted(OUT_DIR.glob("rank_*/rep_*.json")):
        out = json.loads(path.read_text())
        outputs[(out["rank"], out["rep"])] = out
    return outputs


def axis_mentions(sentence: str) -> list[tuple[str, float]]:
    """(axis, number) when a number follows an axis name within a few words."""
    found = []
    sentence = qu.strip_ids(sentence)
    for word, axis in qu.AXIS_WORDS.items():
        for m in re.finditer(word + r"\W+(?:\w+\W+){0,4}?(\d+(?:\.\d+)?)", sentence, re.IGNORECASE):
            found.append((axis, float(m.group(1))))
    return found


def main() -> None:
    study = load_study()
    checks = study["quality"]["checks"]
    cutoff = data_cutoff()
    outputs = load_outputs()
    bundles = {r: json.loads((BUNDLE_DIR / f"rank_{r:02d}.json").read_text()) for r in sorted({k[0] for k in outputs})}
    phrases = [p.lower() for p in checks["superiority_phrases"]]
    marker = checks["threshold_marker"].lower()

    flags, sentence_count = [], Counter()
    for (rank, rep), out in sorted(outputs.items()):
        bundle = bundles[rank]
        texts = qu.evidence_texts(bundle)
        allowed = set(br.bundle_ids(bundle, include_dvf=True))
        paper_dates = {p["id"]: p["first_date"] for p in bundle["papers"]}
        dvf_scores = {a: v["score"] for a, v in bundle["dvf"]["axes"].items()}
        dvf_nums = qu.dvf_numbers(bundle)
        groups = [("evidential", qu.evidential_items(out)), ("prescriptive", qu.prescriptive_items(out))]
        for kind, items in groups:
            for item in items:
                cited = [i for i in item["evidence_ids"] if i in texts]
                cited_text = " ".join(texts[i] for i in cited).lower()
                cited_nums = set(qu.numbers(cited_text)) | dvf_nums
                base = []
                if set(item["evidence_ids"]) - allowed:
                    base.append("id_not_in_bundle")
                if any(paper_dates.get(i, "") > cutoff.isoformat() for i in item["evidence_ids"] if i.startswith("paper:")):
                    base.append("evidence_after_cutoff")
                for idx, sentence in enumerate(qu.sentences(item["text"])):
                    sentence_count[kind] += 1
                    found = list(base)
                    low = sentence.lower()
                    nums = qu.numbers(sentence)
                    if kind == "evidential":
                        if [n for n in nums if n not in cited_nums]:
                            found.append("number_not_in_cited_evidence")
                        for axis, value in axis_mentions(sentence):
                            if dvf_scores.get(axis) is None or float(dvf_scores[axis]) != value:
                                found.append("dvf_number_mismatch")
                    elif nums and marker not in low:
                        found.append("unmarked_threshold")
                    for phrase in phrases:
                        if re.search(r"\b" + re.escape(phrase), low) and phrase not in cited_text:
                            found.append(f"superiority:{phrase}")
                    if found:
                        flags.append({"rank": rank, "rep": rep, "kind": kind, "field": item["field"],
                                      "sentence_index": idx, "sentence": sentence, "flags": sorted(set(found))})
        if out.get("briefing_valid"):
            b = out["briefing"]
            axis = b["riskiest_assumption"]["axis"]
            ra = b["riskiest_assumption"]["text"].lower()
            name = br.AXIS_NAMES[axis]
            problems = []
            if name not in ra:
                problems.append("riskiest_axis_not_named")
            if b["riskiest_assumption"]["designated_by_missing_evidence"] and "missing" not in ra:
                problems.append("missing_designation_not_stated")
            if not any(k in b["mvp"]["text"].lower() for k in checks["mvp_keywords"][axis]):
                problems.append("mvp_type_keyword_absent")
            if problems:
                flags.append({"rank": rank, "rep": rep, "kind": "prescriptive", "field": "riskiest_assumption/mvp",
                              "sentence_index": None, "sentence": None, "flags": problems})

    flag_counts = Counter(f for item in flags for f in item["flags"])
    meta = {"step": "briefing_checks", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "data_cutoff": cutoff.isoformat(), "checks": checks, "outputs": len(outputs)}
    summary = {"sentences": dict(sentence_count), "flagged_items": len(flags), "flag_counts": dict(flag_counts)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "briefing_checks.json").write_text(json.dumps({"meta": meta, "summary": summary, "flags": flags},
                                                             indent=1, ensure_ascii=False))

    # Q2: frame agreement with the modal (status, reason) per candidate
    per_rank, matches, total = [], 0, 0
    mvp_consistent = 0
    for rank in sorted(bundles):
        reps = [outputs[k] for k in sorted(outputs) if k[0] == rank]
        keys = [(o["concept_frame"] or {}).get("status", "invalid") + "|" + str((o["concept_frame"] or {}).get("insufficient_reason"))
                if o["frame_valid"] else "invalid" for o in reps]
        mode, count = Counter(keys).most_common(1)[0]
        if mode == "invalid":
            count = 0
        matches += count
        total += len(keys)
        types = {o["briefing"]["mvp"]["type"] for o in reps if o.get("briefing_valid")}
        mvp_consistent += int(len(types) <= 1)
        per_rank.append({"rank": rank, "keys": keys, "mode": mode, "agreeing": count, "mvp_types": sorted(types)})
    low, high = qu.wilson(matches, total)
    q2 = {
        "meta": {"step": "q2", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "definition": study["quality"]["q2"]},
        "frame_agreement": {"agreeing": matches, "total": total, "rate": matches / total, "wilson_95": [low, high]},
        "mvp_type_agreement": {"candidates_consistent": mvp_consistent, "candidates": len(per_rank),
                               "note": "fixed by the DVF rule (design 8.4); 100 percent by construction"},
        "per_candidate": per_rank,
    }
    (RESULTS / "q2.json").write_text(json.dumps(q2, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"Q2 frame agreement {matches}/{total} = {matches / total:.3f}, Wilson 95% [{low:.3f}, {high:.3f}]")
    print("Saved", RESULTS / "briefing_checks.json", RESULTS / "q2.json")


if __name__ == "__main__":
    main()
