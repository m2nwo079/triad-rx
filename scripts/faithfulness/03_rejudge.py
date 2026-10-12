"""Blind re-judging of Q1 sentences from the represcription and the faithfulness study (post hoc).

Usage (from the repository root):
  python scripts/faithfulness/03_rejudge.py sample [--fields components open_source] [--round 1]
  python scripts/faithfulness/03_rejudge.py sample --fields others --round 2
  python scripts/faithfulness/03_rejudge.py score [--round 1]
  python scripts/faithfulness/03_rejudge.py combine

Added after the faithfulness results (faithfulness preregistration 5); it does not change the main decision.
It separates two readings of the drop in component-role and open-source sentences: a change in the judge's
standard, or a change in the sentences. sample takes those sentences from both Q1 sheets, shuffles them with
faithfulness.rejudge.seed and writes a sheet without the study, rank, repetition or earlier judgement; the
cited evidence comes from the shared bundles. The key (study, earlier review id and judgement) is written to a
separate file that the judge does not open. score compares the earlier and the new judgements per study.
Round 2 takes every other field (others) with the seed plus one, so the two rounds together re-judge both
Q1 samples completely, each round mixing the two studies. combine recomputes Q1 per study from the new
judgements of both rounds (overall, without the business fields, business fields) next to the earlier Q1.

Files: faithfulness/data/review/rejudge[_2].csv (+ _original.csv, _evidence.md, _key.json; not committed)
Outputs: faithfulness/results/rejudge[_2].json, faithfulness/results/rejudge_full.json
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import math

import numpy as np

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx import quality as qu
from triadrx.study import load_study

FIELD_PREFIX = {"components": "concept_frame.components", "open_source": "building_blocks.open_source"}
BUSINESS = ("target_user", "jtbd", "value_prop")
COLUMNS = ["review_id", "sentence", "evidence_ids", "correct", "note"]


def paths(cfg: dict, round_: int = 1) -> dict:
    review = ROOT / cfg["run_dir"] / "data/review"
    tag = "" if round_ == 1 else f"_{round_}"
    return {"sheet": review / f"rejudge{tag}.csv", "original": review / f"rejudge{tag}_original.csv",
            "evidence": review / f"rejudge{tag}_evidence.md", "key": review / f"rejudge{tag}_key.json",
            "result": ROOT / cfg["run_dir"] / f"results/rejudge{tag}.json"}


def read_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def chosen(field: str, fields: list[str]) -> bool:
    """Whether a Q1 field belongs to the chosen groups; others is every field outside the named groups."""
    named = tuple(FIELD_PREFIX.values())
    return (field.startswith(tuple(FIELD_PREFIX[f] for f in fields if f != "others"))
            or ("others" in fields and not field.startswith(named)))


def pick(rows: list[dict], study: str, fields: list[str]) -> list[dict]:
    return [{"study": study, "earlier_id": r["review_id"], "earlier": int(r["correct"]), "rank": int(r["rank"]),
             "field": r["field"], "sentence": r["sentence"], "evidence_ids": r["evidence_ids"]}
            for r in rows if chosen(r["field"], fields)]


def blind(items: list[dict], seed: int) -> list[dict]:
    order = np.random.default_rng(seed).permutation(len(items))
    return [{**items[i], "review_id": f"rj-{n:03d}"} for n, i in enumerate(order)]


def summary(key: list[dict], judged: dict[str, int]) -> dict:
    out = {}
    for study in sorted({k["study"] for k in key}):
        rows = [k for k in key if k["study"] == study]
        for field in sorted({r["field"].split("[")[0] for r in rows}):
            sub = [r for r in rows if r["field"].split("[")[0] == field]
            earlier = sum(r["earlier"] for r in sub)
            now = sum(judged[r["review_id"]] for r in sub)
            same = sum(r["earlier"] == judged[r["review_id"]] for r in sub)
            out.setdefault(study, {})[field] = {"n": len(sub), "earlier_supported": earlier, "now_supported": now,
                                                "agreement": same / len(sub)}
    return out


def sample(cfg: dict, fields: list[str], round_: int) -> None:
    p = paths(cfg, round_)
    if p["sheet"].exists():
        sys.exit(f"{p['sheet']} already exists")
    items = pick(read_sheet(ROOT / cfg["base_run"] / "data/review/q1_review.csv"), cfg["base_run"], fields)
    items += pick(read_sheet(ROOT / cfg["run_dir"] / "data/review/q1_review.csv"), cfg["run_dir"], fields)
    seed = cfg["rejudge"]["seed"] + round_ - 1
    rows = blind(items, seed)
    bundle_dir = ROOT / cfg["base_run"] / "data/derived/briefing/bundles"
    p["sheet"].parent.mkdir(parents=True, exist_ok=True)
    for path in (p["sheet"], p["original"]):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows([{**r, "correct": "", "note": ""} for r in rows])
    p["original"].chmod(0o444)
    lines = ["# Re-judging: cited evidence", "", "Judge each sentence only against the evidence cited for it (Q1 rules).", ""]
    texts = {}
    for r in rows:
        if r["rank"] not in texts:
            texts[r["rank"]] = qu.evidence_texts(json.loads((bundle_dir / f"rank_{r['rank']:02d}.json").read_text()))
        lines += [f"## {r['review_id']}", "", f"> {r['sentence']}", ""]
        for eid in r["evidence_ids"].split():
            lines += [f"**{eid}**", "", texts[r["rank"]].get(eid, "(not in bundle)"), ""]
    p["evidence"].write_text("\n".join(lines))
    p["key"].write_text(json.dumps({"fields": fields, "seed": seed, "round": round_,
                                    "sheet_original_sha256": hashlib.sha256(p["original"].read_bytes()).hexdigest(),
                                    "items": [{k: r[k] for k in ("review_id", "study", "earlier_id", "earlier", "field")} for r in rows]},
                                   indent=1))
    print(f"{len(rows)} sentences ({', '.join(fields)}); sheet {p['sheet'].relative_to(ROOT)}")


def judged_sheet(p: dict) -> dict[str, int]:
    sheet, original = read_sheet(p["sheet"]), read_sheet(p["original"])
    if len(sheet) != len(original):
        sys.exit("the sheet and its original differ in length")
    judged = {}
    for row, orig in zip(sheet, original):
        if {k: row[k] for k in ("review_id", "sentence", "evidence_ids")} != {k: orig[k] for k in ("review_id", "sentence", "evidence_ids")}:
            sys.exit(f"{row['review_id']}: only correct and note may change")
        if row["correct"] not in ("0", "1"):
            sys.exit(f"{row['review_id']}: correct must be 0 or 1, got '{row['correct']}'")
        judged[row["review_id"]] = int(row["correct"])
    return judged


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def rates(items: list[dict], value: str) -> dict:
    k, n = sum(i[value] for i in items), len(items)
    return {"supported": k, "n": n, "rate": k / n if n else None, "wilson_95": wilson(k, n)}


def full(key_items: list[dict]) -> dict:
    """Earlier and new Q1 per study over the re-judged items, overall and split by business fields."""
    out = {}
    for study in sorted({i["study"] for i in key_items}):
        rows = [i for i in key_items if i["study"] == study]
        business = [i for i in rows if i["field"].startswith(BUSINESS)]
        rest = [i for i in rows if not i["field"].startswith(BUSINESS)]
        out[study] = {part: {"earlier": rates(items, "earlier"), "now": rates(items, "now")}
                      for part, items in (("overall", rows), ("without_business", rest), ("business", business))}
    return out


def combine(cfg: dict) -> None:
    items = []
    for round_ in (1, 2):
        p = paths(cfg, round_)
        judged = judged_sheet(p)
        items += [{**i, "now": judged[i["review_id"]]} for i in json.loads(p["key"].read_text())["items"]]
    counts = {s: sum(i["study"] == s for i in items) for s in {i["study"] for i in items}}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"meta": {"step": "faithfulness_rejudge_full", "created": datetime.date.today().isoformat(), "git_commit": commit,
                       "note": "Post hoc (faithfulness preregistration 5); both Q1 samples re-judged blind in two mixed rounds.",
                       "items_by_study": counts},
              "by_study": full(items)}
    out = ROOT / cfg["run_dir"] / "results/rejudge_full.json"
    out.write_text(json.dumps(result, indent=2))
    for study, parts in result["by_study"].items():
        for part, v in parts.items():
            e, n = v["earlier"], v["now"]
            print(f"{study:15s} {part:17s} earlier {e['supported']}/{e['n']}  now {n['supported']}/{n['n']} "
                  f"({n['rate']:.2f}, Wilson {n['wilson_95'][0]:.2f}-{n['wilson_95'][1]:.2f})" if n["n"] else f"{study} {part} none")
    print("Saved", out.relative_to(ROOT))


def score(cfg: dict, round_: int) -> None:
    p = paths(cfg, round_)
    key = json.loads(p["key"].read_text())
    judged = judged_sheet(p)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"meta": {"step": "faithfulness_rejudge", "created": datetime.date.today().isoformat(), "git_commit": commit,
                       "note": "Post hoc (faithfulness preregistration 5); does not change the main decision.",
                       "fields": key["fields"], "seed": key["seed"], "sheet_original_sha256": key["sheet_original_sha256"]},
              "by_study": summary(key["items"], judged)}
    p["result"].write_text(json.dumps(result, indent=2))
    for study, fields in result["by_study"].items():
        for field, v in fields.items():
            print(f"{study:15s} {field:30s} earlier {v['earlier_supported']}/{v['n']}  now {v['now_supported']}/{v['n']}  agreement {v['agreement']:.2f}")
    print("Saved", p["result"].relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["sample", "score", "combine"])
    parser.add_argument("--fields", nargs="+", choices=sorted([*FIELD_PREFIX, "others"]), default=["components", "open_source"])
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    cfg = load_study()["faithfulness"]
    if args.step == "sample":
        sample(cfg, args.fields, args.round)
    elif args.step == "score":
        score(cfg, args.round)
    else:
        combine(cfg)


if __name__ == "__main__":
    main()
