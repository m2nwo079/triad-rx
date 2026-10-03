"""Validate the reviewer's R4 judgements for T1 and decide which candidates are allowed.

Run with --check while reviewing to see progress and errors without writing anything.
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SHEET = Path("data/review/r4_review_T1.csv")
VOCAB = Path("vocab/vocab_base.json")
STUDY = Path("config/study.json")
# Summary written with the review sheet; ties the sheet to its vocabulary and sampling settings
SHEET_SUMMARY = Path("results/r4_review_T1.json")
# Only counts and decisions go here; sentences, paper ids and notes stay in data/
OUT = Path("vocab/r4_decisions_T1.json")
# Decision counts and metadata for reports; per-candidate decisions stay in OUT only
OUT_RESULT = Path("results/r4_decisions_T1.json")


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


def cell(row: dict, column: str) -> str:
    """Cell text; a short row from csv.DictReader gives None, which is treated as blank."""
    return (row.get(column) or "").strip()


def whole(value: str) -> int | None:
    """Parse a non-negative whole number; spreadsheet output such as '2.0' is accepted."""
    try:
        number = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if number.denominator != 1 or number < 0:
        return None
    return int(number)


def source_mismatch(sheet_meta: dict, vocab_sha256: str, cfg: dict) -> str | None:
    """Reason the sheet cannot be used, or None when it matches the current vocabulary and settings."""
    if sheet_meta["vocab_sha256"] != vocab_sha256:
        return f"{VOCAB} changed since the review sheet was built; rebuild the sheet"
    if sheet_meta["review_config"] != cfg:
        return f"r4_review settings in {STUDY} differ from those used to build the review sheet"
    return None


def validate(rows: list[dict], candidate_ids: set[str], n_per_stratum: int):
    """Return (parsed rows, errors). Each parsed row holds sizes, shown counts and judgements."""
    errors = []
    parsed = []
    ids = [row["term_id"] for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f"duplicate term_id rows: {duplicates}")
    missing = sorted(candidate_ids - set(ids))
    extra = sorted(set(ids) - candidate_ids)
    if missing:
        errors.append(f"R4 candidates missing from sheet: {missing}")
    if extra:
        errors.append(f"sheet rows that are not R4 candidates: {extra}")

    for line, row in enumerate(rows, start=2):
        where = f"line {line} ({row['term_id']})"
        t1_docs = whole(cell(row, "t1_docs"))
        sizes = {s: whole(cell(row, f"stratum_{s}_size")) for s in ("a", "b")}
        if t1_docs is None or None in sizes.values():
            errors.append(f"{where}: t1_docs and stratum sizes must be non-negative integers")
            continue
        # Every T1 paper falls into exactly one stratum for a term
        if sizes["a"] + sizes["b"] != t1_docs:
            errors.append(f"{where}: stratum sizes do not add up to t1_docs")
        item = {"row": row, "t1_docs": t1_docs, "size": sizes, "shown": {}, "technical": {}}
        for s in ("a", "b"):
            shown = sum(1 for i in range(1, n_per_stratum + 1) if cell(row, f"{s}{i}_sentence"))
            if shown != min(n_per_stratum, sizes[s]):
                errors.append(f"{where}: stratum {s.upper()} shows {shown} examples, expected min({n_per_stratum}, size)")
            item["shown"][s] = shown
            raw = cell(row, f"{s}_technical")
            if shown == 0:
                # Zero-weight stratum: blank or 0 is fine, anything larger is an input mistake
                if raw and whole(raw) != 0:
                    errors.append(f"{where}: {s}_technical must be blank or 0 when no example is shown")
                item["technical"][s] = 0
            elif not raw:
                item["technical"][s] = None
            else:
                value = whole(raw)
                if value is None or value > shown:
                    errors.append(f"{where}: {s}_technical={raw!r} must be an integer in 0..{shown}")
                item["technical"][s] = value
        parsed.append(item)
    return parsed, errors


def technical_share(item: dict) -> Fraction:
    """Stratum-weighted share of technical use; exact arithmetic so the threshold boundary is safe."""
    weighted = sum(
        item["size"][s] * Fraction(item["technical"][s], item["shown"][s])
        for s in ("a", "b")
        if item["shown"][s]
    )
    return weighted / item["t1_docs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report progress and errors only")
    args = parser.parse_args()

    study = json.loads(STUDY.read_text())
    cfg = study["r4_review"]
    threshold = Fraction(str(cfg["allow_threshold"]))
    vocab = json.loads(VOCAB.read_text())
    candidate_ids = {t["id"] for t in vocab["terms"] if t["status"] == "r4_candidate"}
    sheet_meta = json.loads(SHEET_SUMMARY.read_text())["meta"]
    # The sheet must come from the current vocabulary and the current sampling and threshold settings
    mismatch = source_mismatch(sheet_meta, sha256(VOCAB), cfg)
    if mismatch:
        sys.exit(mismatch)

    # Reading with csv keeps every cell as text; a .numbers or .xlsx file fails here
    with SHEET.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"term_id", "key", "t1_docs", "stratum_a_size", "stratum_b_size", "a_technical", "b_technical"}
        absent = required - set(reader.fieldnames or [])
        if absent:
            sys.exit(f"{SHEET}: missing columns {sorted(absent)}; export the sheet as CSV (UTF-8)")
        rows = list(reader)

    parsed, errors = validate(rows, candidate_ids, cfg["n_per_stratum"])
    review = [p for p in parsed if p["t1_docs"] > 0]
    unfilled = [p for p in review if None in p["technical"].values()]
    print(f"Rows to review: {len(review)}, filled: {len(review) - len(unfilled)}, errors: {len(errors)}")
    for error in errors:
        print("ERROR", error)
    if args.check:
        return
    if errors or unfilled:
        for p in unfilled:
            print("UNFILLED", p["row"]["term_id"])
        sys.exit("Not written: fix the errors and fill every row first")

    decisions = []
    for p in parsed:
        entry = {
            "term_id": p["row"]["term_id"],
            "key": p["row"]["key"],
            "t1_docs": p["t1_docs"],
            "stratum_size": p["size"],
            "shown": p["shown"],
            "technical": p["technical"],
        }
        if p["t1_docs"] == 0:
            entry.update(technical_share=None, decision="excluded_no_t1_docs")
        else:
            share = technical_share(p)
            entry.update(
                technical_share=float(share),
                technical_share_exact=str(share),
                decision="allow" if share >= threshold else "reject",
            )
        decisions.append(entry)

    meta = {
        "step": "r4_decisions_T1",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "timepoint": "T1",
        "review_config": cfg,
        "review_sheet_git_commit": sheet_meta["git_commit"],
        "sheet_sha256": sha256(SHEET),
        "vocab_sha256": sha256(VOCAB),
        "rule": "share = sum over strata of size * technical / shown, divided by t1_docs; allow when share >= allow_threshold; candidates without T1 documents are excluded",
    }
    counts = {
        d: sum(1 for e in decisions if e["decision"] == d)
        for d in ("allow", "reject", "excluded_no_t1_docs")
    }
    OUT.write_text(json.dumps({"meta": meta, "decisions": decisions}, indent=2))
    OUT_RESULT.write_text(json.dumps({"meta": meta, "counts": counts}, indent=2))
    print("Saved", OUT, "and", OUT_RESULT)


if __name__ == "__main__":
    main()
