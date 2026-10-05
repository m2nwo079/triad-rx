"""Agreement between the human Q1 judgments and the cross-model judge (design 9.1, preregistration 5.5.2).

Usage: python scripts/quality/06_cross_model_score.py

Inputs downloaded from the Colab notebook (notebooks/q1_cross_model.ipynb) into
data/derived/quality/: q1_judge_output.jsonl and q1_judge_meta.json. The judge
input and prompt hashes in the meta file must equal results/q1_judge_input.json.
Outputs without a valid label count as label 0 (preregistration 5.5.2) and are
counted separately.

Reported: raw agreement, Cohen's kappa, the 2x2 table, the judge's supported
rate with a Wilson interval next to the human Q1, and disagreements by field.

Output: results/q1_cross_model.json
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

REVIEW = Path("data/review/q1_review.csv")
INPUT_META = Path("results/q1_judge_input.json")
OUTPUT = Path("data/derived/quality/q1_judge_output.jsonl")
META = Path("data/derived/quality/q1_judge_meta.json")
RESULTS = Path("results")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def kappa(table: dict) -> float:
    n = sum(table.values())
    po = (table[(1, 1)] + table[(0, 0)]) / n
    h1 = (table[(1, 1)] + table[(1, 0)]) / n
    m1 = (table[(1, 1)] + table[(0, 1)]) / n
    pe = h1 * m1 + (1 - h1) * (1 - m1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    expected = json.loads(INPUT_META.read_text())
    meta = json.loads(META.read_text())
    if meta["input_sha256"] != expected["input_sha256"] or meta["prompt_sha256"] != expected["prompt_sha256"]:
        sys.exit("judge input or prompt differs from results/q1_judge_input.json")
    judged = {r["review_id"]: r for r in (json.loads(l) for l in OUTPUT.read_text().splitlines() if l.strip())}
    with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [r["review_id"] for r in rows if r["review_id"] not in judged]
    if missing:
        sys.exit(f"judge output lacks {len(missing)} sentences: {missing[:5]}")

    table = {(h, m): 0 for h in (0, 1) for m in (0, 1)}
    invalid, disagreements, by_field = 0, [], {}
    for r in rows:
        human = int(r["correct"])
        out = judged[r["review_id"]]
        if not out["valid"]:
            invalid += 1
        model = int(out["label"]) if out["valid"] else 0
        table[(human, model)] += 1
        field = "concept_frame" if r["field"].startswith("concept_frame") else r["field"].split("[")[0].split(".")[0]
        entry = by_field.setdefault(field, {"n": 0, "agree": 0})
        entry["n"] += 1
        entry["agree"] += int(human == model)
        if human != model:
            disagreements.append({"review_id": r["review_id"], "field": r["field"], "human": human, "model": model,
                                  "model_reason": out.get("reason")})
    n = len(rows)
    agree = table[(0, 0)] + table[(1, 1)]
    model_supported = table[(0, 1)] + table[(1, 1)]
    human_supported = table[(1, 0)] + table[(1, 1)]
    result = {
        "meta": {"step": "q1_cross_model", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "judge": meta, "review_sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
                 "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                 "invalid_rule": "outputs without a valid label count as label 0"},
        "n": n, "invalid_outputs": invalid,
        "agreement": agree / n, "agreement_wilson_95": list(qu.wilson(agree, n)),
        "cohen_kappa": kappa(table),
        "table": {f"human_{h}_model_{m}": v for (h, m), v in table.items()},
        "human_supported_rate": human_supported / n,
        "model_supported_rate": model_supported / n, "model_supported_wilson_95": list(qu.wilson(model_supported, n)),
        "by_field": by_field, "disagreements": disagreements,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "q1_cross_model.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"judge {meta['model_id']} ({meta['model_revision']}), invalid {invalid}")
    print(f"agreement {agree}/{n} = {agree / n:.3f}, kappa {result['cohen_kappa']:.3f}")
    print(f"supported rate: human {human_supported / n:.3f}, model {model_supported / n:.3f}")
    print("table:", result["table"])
    print("Saved", RESULTS / "q1_cross_model.json")


if __name__ == "__main__":
    main()
