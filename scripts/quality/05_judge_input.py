"""Input file for the cross-model Q1 check (design 9.1, preregistration 5.5.2).

Usage: python scripts/quality/05_judge_input.py

Reads the read-only original of the Q1 review file (no human labels or notes) and
writes, for each sampled sentence, the sentence and the text of its cited evidence.

Outputs
- data/derived/quality/q1_judge_input.jsonl (not committed; contains abstracts)
- results/q1_judge_input.json: hashes of the input file and the judge prompt
"""
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx import quality as qu
from triadrx.study import load_study

ORIGINAL = Path("data/review/q1_review_original.csv")
BUNDLE_DIR = Path("data/derived/briefing/bundles")
OUT = Path("data/derived/quality/q1_judge_input.jsonl")
RESULTS = Path("results")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def main() -> None:
    cfg = load_study()["quality"]["cross_model"]
    prompt = Path(f"prompts/q1_judge_{cfg['prompt_version']}.txt")
    with ORIGINAL.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bundles, records = {}, []
    for row in rows:
        rank = int(row["rank"])
        if rank not in bundles:
            bundles[rank] = qu.evidence_texts(json.loads((BUNDLE_DIR / f"rank_{rank:02d}.json").read_text()))
        ids = row["evidence_ids"].split()
        evidence = "\n\n".join(f"[{i}]\n{bundles[rank].get(i, '(not in bundle)')}" for i in ids)
        records.append({"review_id": row["review_id"],
                        "user": f"Sentence:\n{row['sentence']}\n\nCited evidence:\n{evidence}"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    meta = {"step": "q1_judge_input", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "settings": cfg, "records": len(records),
            "input_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
            "original_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
            "note": "built from the read-only original; human labels and notes are not included"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "q1_judge_input.json").write_text(json.dumps(meta, indent=2))
    print(f"records: {len(records)}")
    print("Saved", OUT, RESULTS / "q1_judge_input.json")


if __name__ == "__main__":
    main()
