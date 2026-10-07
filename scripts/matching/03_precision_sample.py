"""Refined-matching vocabulary on the precision timepoint and its precision sample (matching preregistration 3).

Usage: python scripts/matching/03_precision_sample.py [--attempt 0] [--strata method_ci task_ci acronym_cs]

The first-study vocabulary order is repeated with the refined matching of triadrx.text:
active terms plus the T1 R4 allow decisions, the M1 remediation filter (scripts/vocab/06_apply_m1_fix.py),
the surface rules of matching.surface_rules, the common-word decisions (vocab/common_word_decisions.json:
excluded surfaces are removed, exclusion phrases drop the matches they cover), then rule R6
(scripts/vocab/04_apply_r6.py) on the build window of matching.precision.timepoint. The sample follows
the first-study M1 sampler (scripts/vocab/05_m1_sample.py): per stratum, uniform over matches with at
most one match per paper, seed matching.precision.seed plus the attempt number; the sentence shown is
the first sentence whose refined matches hold that term in that mode.

Outputs
- vocab/timepoints/<tp>_v3_after_r6.json (committed) and data/derived/matching/matches_<tp>_v3_after_r6.parquet
- data/review/precision_<tp>_attempt<n>.csv and its read-only _original.csv (not committed)
- results/precision_sample_<tp>_attempt<n>.json
"""
import argparse
import csv
import datetime
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.corpus import match_papers_refined
from triadrx.study import build_window, load_study, runtime_environment
from triadrx.text import Matcher, ci_tokens, refine_terms, refined_spans

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
R4_DECISIONS = Path("vocab/r4_decisions_T1.json")
DECISIONS = Path("vocab/common_word_decisions.json")
TIMEPOINTS = Path("vocab/timepoints")
MATCH_DIR = Path("data/derived/matching")
REVIEW_DIR = Path("data/review")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r6 = load("apply_r6", ROOT / "scripts/vocab/04_apply_r6.py")
m1fix = load("apply_m1_fix", ROOT / "scripts/vocab/06_apply_m1_fix.py")
m1 = load("m1_sample", ROOT / "scripts/vocab/05_m1_sample.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def refined_vocabulary(study: dict) -> tuple[list[dict], set[str], dict, dict]:
    """Terms, case-sensitive-only ids, exclusion phrases per term and removal counts."""
    r4 = json.loads(R4_DECISIONS.read_text())
    allowed = {d["term_id"] for d in r4["decisions"] if d["decision"] == "allow"}
    base = json.loads(VOCAB.read_text())["terms"]
    terms = [{**t, "status": "active"} for t in base if t["status"] == "active" or t["id"] in allowed]
    fix = study["vocab"]["m1_fix"]
    terms, fix_surfaces, fix_terms = m1fix.filter_terms(terms, re.compile(fix["junk_pattern"]),
                                                        {(e["term_id"], e["surface"]) for e in fix["r3_exclude"]})
    decisions = json.loads(DECISIONS.read_text())["decisions"]
    excluded = {(d["term_id"], d["surface"]) for d in decisions if d["decision"] == "exclude"}
    exclusions = {}
    for d in decisions:
        if d["decision"] == "keep_with_exclusions":
            exclusions.setdefault(d["term_id"], []).extend(ci_tokens(p) for p in d["exclusions"])
    before = len(terms)
    terms, cs_only = refine_terms(terms, study["matching"]["surface_rules"]["skip_surface_chars"], excluded)
    counts = {"m1_fix_surfaces_removed": len(fix_surfaces), "m1_fix_terms_removed": len(fix_terms),
              "terms_removed_by_surface_rules_and_decisions": before - len(terms),
              "excluded_surfaces": len(excluded), "terms_with_exclusion_phrases": len(exclusions)}
    return terms, cs_only, exclusions, counts


def build_window_papers(tp: str, study: dict) -> pd.DataFrame:
    """Same papers as triadrx.corpus.window_papers, read with a date filter to save memory."""
    start, end = build_window(tp, study)
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"],
                             filters=[("first_date", ">=", start), ("first_date", "<=", end)])
    return papers.sort_values("id").reset_index(drop=True)


def sample(matches: pd.DataFrame, kept: dict, strata: list[str], n: int, seed: int) -> list[tuple]:
    """Per stratum, uniform over matches with at most one match per paper (first-study M1 rule)."""
    matches = matches[matches["term_id"].isin(kept)].copy()
    matches["stratum"] = [m1.stratum(kept[t], mode) for t, mode in zip(matches["term_id"], matches["mode"])]
    rng = random.Random(seed)
    picked = []
    for name in strata:
        pool = sorted(matches[matches["stratum"] == name][["paper_id", "term_id", "mode"]].itertuples(index=False))
        rng.shuffle(pool)
        used = set()
        for pid, tid, mode in pool:
            if pid in used:
                continue
            used.add(pid)
            picked.append((name, pid, tid, mode))
            if len(used) == n:
                break
    return picked


def main() -> None:
    study = load_study()
    cfg = study["matching"]["precision"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--strata", nargs="+", default=cfg["strata"], choices=cfg["strata"])
    args = parser.parse_args()
    if not 0 <= args.attempt <= cfg["max_remeasure"]:
        sys.exit(f"attempt must be between 0 and {cfg['max_remeasure']}")
    tp = cfg["timepoint"]
    sheet = REVIEW_DIR / f"precision_{tp}_attempt{args.attempt}.csv"
    if sheet.exists():
        sys.exit(f"{sheet} already exists; a new attempt needs a new number and a new sample")
    rules = study["matching"]["surface_rules"]

    terms, cs_only, exclusions, removal = refined_vocabulary(study)
    window = build_window_papers(tp, study)
    matches = match_papers_refined(window, terms, cs_only, rules, exclusions)
    links = matches.drop_duplicates(["paper_id", "term_id"])
    degrees = links.groupby("paper_id").size().reindex(window["id"], fill_value=0).to_numpy()
    term_docs = links.groupby("term_id").size()
    total_links, w_p_star, max_w_c = r6.r6_threshold(degrees, study["vocab"]["r6_rule"]["paper_degree_quantile"])
    term_type = {t["id"]: t["type"] for t in terms}
    kept = {t: term_type[t] for t in term_docs[term_docs <= max_w_c].index}
    removed = term_docs[term_docs > max_w_c]

    stage = TIMEPOINTS / f"{tp}_v3_after_r6.json"
    stage.write_text(json.dumps({
        "meta": {"step": "matching_v3_after_r6", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "surface_rules": rules,
                 "decisions_sha256": evaluate.sha256(DECISIONS)},
        "stats": {"window_papers": int(len(window)), "total_links": total_links, "paper_degree_quantile_value": w_p_star,
                  "max_term_docs": max_w_c, **removal},
        "removed_r6": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in removed.sort_values(ascending=False).items()],
        "kept": [{"term_id": t, "type": term_type[t], "docs": int(term_docs[t])} for t in sorted(kept)],
    }, indent=1))
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = MATCH_DIR / f"matches_{tp}_v3_after_r6.parquet"
    matches[matches["term_id"].isin(kept)].to_parquet(matches_path, index=False)

    picked = sample(matches, kept, args.strata, cfg["n_per_stratum"], cfg["seed"] + args.attempt)
    keys = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}
    matcher = Matcher(terms)
    text = window.set_index("id")
    rows = []
    for n, (name, pid, tid, mode) in enumerate(picked):
        sentences = m1.SENTENCE.split(re.sub(r"\s+", " ", f"{text.at[pid, 'title']}. {text.at[pid, 'abstract']}").strip())
        sentence, surface = "", ""
        for s, spans in zip(sentences, refined_spans(matcher, sentences, cs_only, rules, exclusions)):
            hits = [m for m in spans if m["id"] == tid and m["mode"] == mode]
            if hits:
                sentence, surface = s, hits[0]["tokens"]
                break
        rows.append({"review_id": f"pr-{n:03d}", "stratum": name, "term_id": tid, "key": keys[tid],
                     "matched_tokens": surface, "paper_id": pid, "sentence": sentence, "correct": "", "note": ""})
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    original = sheet.with_name(sheet.stem + "_original.csv")
    original.write_bytes(sheet.read_bytes())
    original.chmod(0o444)

    result = {
        "meta": {"step": "precision_sample", "timepoint": tp, "attempt": args.attempt, "seed": cfg["seed"] + args.attempt,
                 "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                 "environment": runtime_environment(), "precision_config": cfg, "strata": args.strata,
                 "stage_sha256": evaluate.sha256(stage), "matches_sha256": evaluate.sha256(matches_path),
                 "review_sha256": evaluate.sha256(sheet)},
        "terms_after_r6": len(kept), "terms_removed_r6": int(len(removed)),
        "sampled": {name: sum(1 for r in rows if r["stratum"] == name) for name in args.strata},
        "missing_sentence": sum(1 for r in rows if not r["sentence"]),
    }
    (RESULTS / f"precision_sample_{tp}_attempt{args.attempt}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
