"""Aggregate M1 judgements, decide pass status per stratum, and apply rule R5 to every timepoint.

All measurement attempts are validated and reported. Each stratum uses its latest
measurement; a stratum may be re-measured alone when only it has a remediation rule.
R5 uses those final Wilson lower bounds.

Usage: python scripts/vocab/07_m1_r5.py [--check]
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.stats import quality_status, r5_min_docs, wilson_interval
from triadrx.study import load_study

REVIEW = Path("data/review")
RESULTS = Path("results")
VOCAB = Path("vocab/vocab_base.json")
TIMEPOINTS = Path("vocab/timepoints")
# Columns the reviewer must not change; paper ids are left out because spreadsheets rewrite them
FIXED = ["stratum", "term_id", "key", "matched_tokens", "sentence"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def validate_attempt(rows: list[dict], original: list[dict]) -> tuple[list[str], int]:
    """Return (errors, unfilled count) for one judged sheet against its original."""
    errors = []
    if len(rows) != len(original):
        return [f"row count {len(rows)} differs from original {len(original)}"], 0
    unfilled = 0
    for line, (row, orig) in enumerate(zip(rows, original), start=2):
        changed = [c for c in FIXED if row.get(c, "").strip() != orig.get(c, "").strip()]
        if changed:
            errors.append(f"line {line}: columns changed {changed}")
        value = row.get("correct", "").strip()
        if value == "":
            unfilled += 1
        elif value not in {"0", "1", "0.0", "1.0"}:
            errors.append(f"line {line}: correct={value!r} must be 0 or 1")
    return errors, unfilled


def stratum_stats(rows: list[dict], strata: list[str], rule: dict) -> dict:
    out = {}
    for name in strata:
        values = [int(float(r["correct"])) for r in rows if r["stratum"] == name]
        n, k = len(values), sum(values)
        low, high = wilson_interval(k, n)
        out[name] = {
            "n": n,
            "correct": k,
            "precision": k / n,
            "wilson_lower": low,
            "wilson_upper": high,
            "status": quality_status(low, rule),
        }
    return out


def final_by_stratum(report: list[dict]) -> dict:
    """For each stratum, the measurement from the latest attempt that sampled it."""
    final = {}
    for attempt in report:
        for name, stats in attempt["strata"].items():
            final[name] = {**stats, "attempt": attempt["attempt"]}
    return final


def surface_stratum(term_type: str, mode: str) -> str:
    if mode == "cs":
        return "acronym_cs"
    return "method_ci" if term_type == "M" else "task_ci"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate sheets and report only")
    args = parser.parse_args()

    study = load_study()
    cfg = study["quality"]["m1"]
    strata, rule = cfg["strata"], cfg["pass_rule"]

    attempts = []
    for attempt in range(cfg["max_remeasure"] + 1):
        sheet = REVIEW / f"m1_review_T1_attempt{attempt}.csv"
        if not sheet.exists():
            break
        original = REVIEW / f"m1_review_T1_attempt{attempt}_original.csv"
        summary = RESULTS / f"m1_sample_T1_attempt{attempt}.json"
        if not original.exists() or not summary.exists():
            sys.exit(f"attempt {attempt}: need {original} and {summary}")
        rows, orig = read_rows(sheet), read_rows(original)
        errors, unfilled = validate_attempt(rows, orig)
        print(f"attempt {attempt}: rows {len(rows)}, unfilled {unfilled}, errors {len(errors)}")
        for error in errors:
            print("  ERROR", error)
        attempts.append({"attempt": attempt, "sheet": sheet, "rows": rows, "errors": errors,
                         "unfilled": unfilled, "summary": json.loads(summary.read_text())})
    if not attempts:
        sys.exit("no M1 review sheet found")
    if args.check:
        return
    if any(a["errors"] or a["unfilled"] for a in attempts):
        sys.exit("Not written: fix errors and fill every row first")

    report = []
    for a in attempts:
        report.append({
            "attempt": a["attempt"],
            "seed": a["summary"]["meta"]["seed"],
            "stage": a["summary"]["meta"].get("stage", "after_r6"),
            "sample_git_commit": a["summary"]["meta"]["git_commit"],
            "sheet_sha256": sha256(a["sheet"]),
            "strata": stratum_stats(a["rows"], sorted({r["stratum"] for r in a["rows"]}, key=strata.index), rule),
        })
    final = final_by_stratum(report)
    missing_strata = [s for s in strata if s not in final]
    if missing_strata:
        sys.exit(f"strata never measured: {missing_strata}")
    failed = [s for s, v in final.items() if v["status"] == "fail"]
    latest = report[-1]
    attempts_left = cfg["max_remeasure"] - latest["attempt"]
    m1_result = {
        "meta": {"step": "m1_T1", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "quality_m1": cfg},
        "attempts": report,
        "final_by_stratum": final,
        "final_failed_strata": failed,
    }
    (RESULTS / "m1_T1.json").write_text(json.dumps(m1_result, indent=2))
    print("Saved", RESULTS / "m1_T1.json")
    for name, v in final.items():
        print(f"  {name} (attempt {v['attempt']}): {v['correct']}/{v['n']}, lower {v['wilson_lower']:.3f}, {v['status']}")
    if failed and attempts_left > 0:
        sys.exit(f"Strata {failed} failed and re-measurement remains (attempts left: {attempts_left}). R5 not applied.")

    # Rule R5 with each stratum's final lower bound (preregistration 3.6, 5.1)
    stability = study["vocab"]["r5_rule"]["stability"]
    lower = {s: v["wilson_lower"] for s, v in final.items()}
    stage = latest["stage"]
    # Check every timepoint first so that a missing file never leaves partial outputs
    missing = [p for p in (TIMEPOINTS / f"{tp}_{stage}.json" for tp in ["T1", "T2", "applied"]) if not p.exists()]
    if missing:
        sys.exit(f"R5 not applied, missing stage files: {[str(p) for p in missing]}")
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    for tp in ["T1", "T2", "applied"]:
        stage_path = TIMEPOINTS / f"{tp}_{stage}.json"
        detail = json.loads(stage_path.read_text())
        dropped = {(r["term_id"], r["surface"]) for r in detail.get("removed_surfaces", [])}
        kept, removed = [], []
        for entry in detail["kept"]:
            term = vocab[entry["term_id"]]
            modes = {s["match"] for s in term["surfaces"] if (term["id"], s["text"]) not in dropped}
            term_strata = sorted({surface_stratum(term["type"], m) for m in modes})
            pi = min(lower[s] for s in term_strata)
            k = r5_min_docs(stability, pi)
            record = {**entry, "strata": term_strata, "pi": pi, "k": k}
            (kept if entry["docs"] >= k else removed).append(record)
        meta = {"step": "apply_r5", "timepoint": tp, "created": datetime.date.today().isoformat(),
                "git_commit": git_commit(), "stage": stage, "stage_sha256": sha256(stage_path),
                "m1_final_attempts": {k: v["attempt"] for k, v in final.items()}, "r5_rule": study["vocab"]["r5_rule"],
                "stratum_lower": lower}
        stats = {"terms_in": len(detail["kept"]), "terms_removed_r5": len(removed), "terms_final": len(kept)}
        (TIMEPOINTS / f"{tp}_final.json").write_text(
            json.dumps({"meta": meta, "stats": stats, "removed_r5": removed, "kept": kept}, indent=1))
        (RESULTS / f"vocab_final_{tp}.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
        print(f"Saved {TIMEPOINTS / f'{tp}_final.json'} ({stats})")


if __name__ == "__main__":
    main()