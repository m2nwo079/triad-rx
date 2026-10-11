"""Review sheets for the applied terms that no reviewed timepoint held (represcription preregistration 2).

Usage: python scripts/represcription/02_review_sheets.py

New terms are the final terms of the applied v3 vocabulary (vocab/timepoints/applied_v3_final.json) absent
from the v3 final vocabulary of every reviewed timepoint (matching.engine.reviewed_timepoints). Both sheets
use the third-study rules and columns, restricted to these terms:
- Common words (matching preregistration 2.2): case-insensitive single-word surfaces of new terms whose word,
  or the base of a listed inflection, is in the R4 word list, minus (term, surface) items that already have a
  decision in vocab/common_word_decisions.json. Example sentences are drawn with represcription.review.seed
  from applied build-window sentences where the shared matcher finds exactly that surface.
- Synonyms and hierarchy (matching preregistration 4.1): candidate pairs found by the third-study rules
  (scripts/matching/05_synonym_sheet.py) on the applied v3 vocabulary and applied v3 build hypergraph, keeping
  pairs with at least one new term and dropping pairs already shown in the third-study synonym sheet.
  The third study built its candidates on first-study vocabularies because v3 vocabularies did not exist yet;
  here the v3 applied vocabulary that the represcription uses is the input.

Outputs (not committed): represcription.review.common_words_file and synonyms_file with read-only _original.csv
copies. Summary: results/review_sheets_applied.json (counts, settings and hashes, no text).
"""
import csv
import datetime
import importlib.util
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment
from triadrx.text import Matcher, ci_tokens

TIMEPOINTS = Path("vocab/timepoints")
APPLIED_VOCAB = TIMEPOINTS / "applied_v3_final.json"
APPLIED_BUILD = Path("data/derived/hypergraph/v3/applied_build.parquet")
COMMON_DECISIONS = Path("vocab/common_word_decisions.json")
PAPERS = Path("data/arxiv/papers.parquet")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cw = load("common_word_sheet", ROOT / "scripts/matching/01_common_word_sheet.py")
sy = load("synonym_sheet", ROOT / "scripts/matching/05_synonym_sheet.py")
ar = load("applied_rebuild", ROOT / "scripts/represcription/01_applied_rebuild.py")


def new_term_ids(applied_ids: set[str], reviewed: list[str]) -> set[str]:
    """Applied final terms absent from every reviewed timepoint's v3 final vocabulary."""
    seen = set()
    for tp in reviewed:
        seen |= {t["id"] for t in json.loads((TIMEPOINTS / f"{tp}_v3_final.json").read_text())["terms"]}
    return set(applied_ids) - seen


def common_word_items(terms: list[dict], new: set[str], words: set[str], inflections: list, decided: set) -> list[tuple]:
    """(term id, surface) of new terms that the third-study common-word rule would list and nobody decided."""
    items = []
    for t in terms:
        if t["id"] not in new:
            continue
        for s in t["surfaces"]:
            if (s["match"] == "ci" and len(ci_tokens(s["text"])) == 1 and cw.is_common(s["text"], words, inflections)
                    and (t["id"], s["text"]) not in decided):
                items.append((t["id"], s["text"]))
    return sorted(set(items))


def synonym_items(found: dict, new: set[str], shown: set) -> list[tuple]:
    """Candidate pairs with at least one new term that the third-study sheet did not show."""
    return sorted(p for p in found if (p[0] in new or p[1] in new) and p not in shown)


def example_sentences(window: pd.DataFrame, terms: list[dict], items: list[tuple], abbreviations: list[str]) -> dict:
    """Applied build-window sentences in which the shared matcher finds each (term id, surface)."""
    matcher = Matcher(terms)
    wanted = {(tid, ci_tokens(text)[0]): (tid, text) for tid, text in items}
    words = {w for _, w in wanted}
    found = defaultdict(list)
    for pid, title, abstract in window[["id", "title", "abstract"]].itertuples(index=False):
        text = f"{title}. {abstract}"
        if not words & set(ci_tokens(text)):
            continue
        for sentence in cw.evidence.split_sentences(text, abbreviations):
            for span in matcher.spans(sentence):
                key = (span["id"], span["tokens"].lower())
                if span["mode"] == "ci" and key in wanted:
                    found[wanted[key]].append((pid, sentence))
    return found


def write_sheet(path: Path, columns: list[str], rows: list[dict], prefix: str) -> None:
    if path.exists():
        sys.exit(f"{path} already exists; a new attempt needs a new file name")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for n, row in enumerate(rows):
            writer.writerow({"review_id": f"{prefix}-{n:03d}", **row})
    original = path.with_name(path.stem + "_original.csv")
    original.write_bytes(path.read_bytes())
    original.chmod(0o444)


def main() -> None:
    study = load_study()
    m, rcfg = study["matching"], study["represcription"]["review"]
    cw_cfg, sy_cfg = m["common_words"], m["synonyms"]
    cw_path, sy_path = Path(rcfg["common_words_file"]), Path(rcfg["synonyms_file"])
    for p in (cw_path, sy_path):
        if p.exists():
            sys.exit(f"{p} already exists; a new attempt needs a new file name")
    prior_sheet = Path(sy_cfg["review_file"]).with_name(Path(sy_cfg["review_file"]).stem + "_original.csv")
    if not prior_sheet.exists():
        sys.exit(f"{prior_sheet} (third-study synonym sheet) is needed to skip pairs already shown")

    vocab = json.loads(APPLIED_VOCAB.read_text())
    terms = vocab["terms"]
    by_id = {t["id"]: t for t in terms}
    new = new_term_ids(set(by_id), m["engine"]["reviewed_timepoints"])
    window = ar.applied_window_papers(study)
    rng_cw, rng_sy = random.Random(rcfg["seed"]), random.Random(rcfg["seed"] + 1)

    wordlist = Path(study["vocab"]["r4_wordlist"])
    words = {w.strip().lower() for w in wordlist.read_text(errors="ignore").splitlines() if w.strip()}
    decided = {(d["term_id"], d["surface"]) for d in json.loads(COMMON_DECISIONS.read_text())["decisions"]}
    cw_items = common_word_items(terms, new, words, cw_cfg["inflections"], decided)
    found = example_sentences(window, terms, cw_items, study["engine_reliability"]["label_v2"]["abbreviations"])
    cw_rows = []
    for key in cw_items:
        pool = found.get(key, [])
        picked = rng_cw.sample(pool, min(cw_cfg["examples_per_surface"], len(pool)))
        cw_rows.append({"term_id": key[0], "surface": key[1], "first_timepoint": "applied", "matches": len(pool),
                        **{f"example_{i + 1}": f"[{p}] {s}" for i, (p, s) in enumerate(picked)}})
    cw_columns = ["review_id", "term_id", "surface", "first_timepoint", "matches"] + \
                 [f"example_{i + 1}" for i in range(cw_cfg["examples_per_surface"])] + ["decision", "exclusions", "note"]

    build = pd.read_parquet(APPLIED_BUILD)
    shown = {tuple(sorted((r["term_a"], r["term_b"]))) for r in pd.read_csv(prior_sheet, encoding="utf-8-sig").to_dict("records")}
    candidates = sy.candidate_pairs(terms, build, sy_cfg)
    sy_items = synonym_items(candidates, new, shown)
    holders = defaultdict(list)
    wanted = set(sy_items)
    for pid, ts in zip(build["paper_id"], build["terms"]):
        for pair in itertools.combinations(sorted(set(ts)), 2):
            if pair in wanted:
                holders[pair].append(pid)
    titles = window.set_index("id")["title"]
    sy_rows = []
    for pair in sy_items:
        info, a, b = candidates[pair], by_id[pair[0]], by_id[pair[1]]
        picked = rng_sy.sample(holders[pair], min(sy_cfg["titles_per_pair"], len(holders[pair])))
        sy_rows.append({"term_a": pair[0], "name_a": a["key"], "surfaces_a": "; ".join(s["text"] for s in a["surfaces"]),
                        "term_b": pair[1], "name_b": b["key"], "surfaces_b": "; ".join(s["text"] for s in b["surfaces"]),
                        "rules": "+".join(info["rules"]), "first_timepoint": "applied", "cooccurrence": info["cooccurrence"],
                        **{f"title_{i + 1}": f"[{p}] {' '.join(str(titles.get(p, '')).split())}" for i, p in enumerate(picked)}})
    sy_columns = ["review_id", "term_a", "name_a", "surfaces_a", "term_b", "name_b", "surfaces_b", "rules", "first_timepoint",
                  "cooccurrence"] + [f"title_{i + 1}" for i in range(sy_cfg["titles_per_pair"])] + ["decision", "note"]

    write_sheet(cw_path, cw_columns, cw_rows, "cwa")
    write_sheet(sy_path, sy_columns, sy_rows, "sya")
    evaluate = sy.evaluate
    result = {"meta": {"step": "review_sheets_applied", "created": datetime.date.today().isoformat(),
                       "git_commit": evaluate.git_commit(), "environment": runtime_environment(), "review_config": rcfg,
                       "inputs_sha256": {"applied_vocab": evaluate.sha256(APPLIED_VOCAB), "applied_build": evaluate.sha256(APPLIED_BUILD),
                                         "common_word_decisions": evaluate.sha256(COMMON_DECISIONS),
                                         "prior_synonym_sheet": evaluate.sha256(prior_sheet), "wordlist": evaluate.sha256(wordlist)},
                       "sheets_sha256": {"common_words": evaluate.sha256(cw_path), "synonyms": evaluate.sha256(sy_path)}},
              "new_terms": len(new),
              "new_terms_by_type": dict(Counter(by_id[t]["type"] for t in new)),
              "common_word_surfaces": len(cw_rows),
              "synonym_pairs": len(sy_rows),
              "synonym_pairs_by_rule": dict(Counter(r for row in sy_rows for r in row["rules"].split("+")))}
    (RESULTS / "review_sheets_applied.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
