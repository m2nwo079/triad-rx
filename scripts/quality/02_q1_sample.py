"""Q1 review sample (design 9.2, preregistration 5.5, 5.5.1).

Usage: python scripts/quality/02_q1_sample.py

Population: every sentence of the evidential fields of all generated outputs
(quality.q1.population). Draws quality.q1.n_sentences sentences without
replacement with numpy default_rng(quality.q1.seed), in a fixed population order
(rank, repetition, field order, sentence order). For each sentence the review file
carries the cited evidence ids, and a companion Markdown file shows the full text
of the cited evidence for reading while judging.

Outputs
- data/review/q1_review.csv and q1_review_original.csv (read-only copy)
- data/review/q1_review_evidence.md (reading aid)
- results/q1_sample.json
"""
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx import quality as qu
from triadrx.study import load_study

BUNDLE_DIR = Path("data/derived/briefing/bundles")
OUT_DIR = Path("briefings")
REVIEW_DIR = Path("data/review")
RESULTS = Path("results")
FIELDS = ["review_id", "rank", "rep", "field", "sentence_index", "sentence", "evidence_ids", "correct", "note"]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def main() -> None:
    q1 = load_study()["quality"]["q1"]
    if q1["population"] != "evidential_fields":
        sys.exit("only quality.q1.population == 'evidential_fields' is implemented")
    review = REVIEW_DIR / "q1_review.csv"
    original = REVIEW_DIR / "q1_review_original.csv"
    if review.exists() or original.exists():
        sys.exit(f"{review} already exists; the Q1 sample is drawn once")

    population, bundles = [], {}
    for path in sorted(OUT_DIR.glob("rank_*/rep_*.json")):
        out = json.loads(path.read_text())
        for item in qu.evidential_items(out):
            for idx, sentence in enumerate(qu.sentences(item["text"])):
                population.append({"rank": out["rank"], "rep": out["rep"], "field": item["field"],
                                   "sentence_index": idx, "sentence": sentence, "evidence_ids": item["evidence_ids"]})
    population.sort(key=lambda s: (s["rank"], s["rep"]))
    rng = np.random.default_rng(q1["seed"])
    chosen = sorted(rng.choice(len(population), size=min(q1["n_sentences"], len(population)), replace=False))
    sample = [population[i] for i in chosen]

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, s in enumerate(sample):
        rows.append({"review_id": f"q1-{n:03d}", "rank": s["rank"], "rep": s["rep"], "field": s["field"],
                     "sentence_index": s["sentence_index"], "sentence": s["sentence"],
                     "evidence_ids": " ".join(s["evidence_ids"]), "correct": "", "note": ""})
    for path in (review, original):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    original.chmod(0o444)

    lines = ["# Q1 review: cited evidence", "",
             "Judge each sentence only against the evidence cited for it (preregistration 5.5.1).", ""]
    for row, s in zip(rows, sample):
        if s["rank"] not in bundles:
            bundles[s["rank"]] = json.loads((BUNDLE_DIR / f"rank_{s['rank']:02d}.json").read_text())
        texts = qu.evidence_texts(bundles[s["rank"]])
        lines += [f"## {row['review_id']} (rank {s['rank']}, rep {s['rep']}, {s['field']})", "",
                  f"> {s['sentence']}", ""]
        for eid in s["evidence_ids"]:
            lines += [f"**{eid}**", "", texts.get(eid, "(not in bundle)"), ""]
    aid = REVIEW_DIR / "q1_review_evidence.md"
    aid.write_text("\n".join(lines))

    fields = {}
    for s in sample:
        key = s["field"].split("[")[0].split(".")[0] if not s["field"].startswith("concept_frame") else "concept_frame"
        fields[key] = fields.get(key, 0) + 1
    meta = {"step": "q1_sample", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "settings": q1, "population_order": "rank, repetition, field order, sentence order",
            "sampling": "numpy default_rng(seed).choice without replacement",
            "review_original_sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
    stats = {"population_sentences": len(population), "sample": len(sample), "sample_by_field": fields}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "q1_sample.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", review, original, aid, RESULTS / "q1_sample.json")


if __name__ == "__main__":
    main()
