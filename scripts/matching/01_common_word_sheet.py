"""Common-word review sheet (matching preregistration 2.2).

Usage: python scripts/matching/01_common_word_sheet.py

Candidates are case-insensitive single-word surfaces of each timepoint's final vocabulary
(matching.timepoints) whose word, or the base of a listed inflection, is in the word list of
rule R4. Each surface is shown with example sentences drawn at random (matching.common_words.seed)
from the build window of the first timepoint whose vocabulary holds it; a sentence counts when
the shared matcher finds exactly that surface in it. Windows of the engine timepoints are merged
in memory as in scripts/engine/14_timepoint_prep.py.

Outputs
- matching.common_words.review_file and its _original.csv copy (not committed, read-only copy)
- results/common_words_sheet.json: counts, settings and hashes, no text
"""
import csv
import datetime
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import build_window, load_study, runtime_environment
from triadrx.text import Matcher, ci_tokens

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load("timepoint_prep", ROOT / "scripts/engine/14_timepoint_prep.py")
evidence = prep.labels_v2.evidence
graph = prep.graph


def is_common(word: str, words: set[str], inflections: list[list[str]]) -> bool:
    """The word or the base of one of its listed inflections is in the word list."""
    word = word.lower()
    if word in words:
        return True
    return any(word.endswith(suffix) and word[: -len(suffix)] + add in words for suffix, add in inflections)


def candidates(study: dict, words: set[str], cfg: dict) -> dict[tuple[str, str], str]:
    """(term id, surface) -> first timepoint whose final vocabulary holds it."""
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    first = {}
    for tp in study["matching"]["timepoints"]:
        for term in graph.final_terms(tp, vocab):
            for s in term["surfaces"]:
                if s["match"] == "ci" and len(ci_tokens(s["text"])) == 1 and is_common(s["text"], words, cfg["inflections"]):
                    first.setdefault((term["id"], s["text"]), tp)
    return first


def example_sentences(tp: str, study: dict, surfaces: list[tuple[str, str]], abbreviations: list[str]) -> dict:
    """All build-window sentences of `tp` in which the shared matcher finds each surface."""
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    matcher = Matcher(graph.final_terms(tp, vocab))
    wanted = {(tid, ci_tokens(text)[0]): (tid, text) for tid, text in surfaces}
    start, end = build_window(tp, study)
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"],
                             filters=[("first_date", ">=", start), ("first_date", "<=", end)])
    words = {w for _, w in wanted}
    found = defaultdict(list)
    for pid, title, abstract in papers[["id", "title", "abstract"]].itertuples(index=False):
        text = f"{title}. {abstract}"
        if not words & set(ci_tokens(text)):
            continue
        for sentence in evidence.split_sentences(text, abbreviations):
            for span in matcher.spans(sentence):
                key = (span["id"], span["tokens"].lower())
                if span["mode"] == "ci" and key in wanted:
                    found[wanted[key]].append((pid, sentence))
    return found


def main() -> None:
    study = prep.merged_study(load_study())
    cfg = study["matching"]["common_words"]
    review = Path(cfg["review_file"])
    original = review.with_name(review.stem + "_original.csv")
    if review.exists():
        sys.exit(f"{review} already exists; a new attempt needs a new file name")
    wordlist = Path(study["vocab"]["r4_wordlist"])
    words = {w.strip().lower() for w in wordlist.read_text(errors="ignore").splitlines() if w.strip()}
    first = candidates(study, words, cfg)

    by_tp = defaultdict(list)
    for key, tp in first.items():
        by_tp[tp].append(key)
    abbreviations = study["engine_reliability"]["label_v2"]["abbreviations"]
    rng = random.Random(cfg["seed"])
    rows, counts = [], {}
    for tp in study["matching"]["timepoints"]:
        if not by_tp[tp]:
            continue
        found = example_sentences(tp, study, by_tp[tp], abbreviations)
        for key in sorted(by_tp[tp]):
            pool = found.get(key, [])
            counts[f"{key[0]}|{key[1]}"] = len(pool)
            picked = rng.sample(pool, min(cfg["examples_per_surface"], len(pool)))
            rows.append({"term_id": key[0], "surface": key[1], "first_timepoint": tp, "matches": len(pool),
                         **{f"example_{i + 1}": f"[{p}] {s}" for i, (p, s) in enumerate(picked)}})

    columns = ["review_id", "term_id", "surface", "first_timepoint", "matches"] + \
              [f"example_{i + 1}" for i in range(cfg["examples_per_surface"])] + ["decision", "exclusions", "note"]
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for n, row in enumerate(rows):
            writer.writerow({"review_id": f"cw-{n:03d}", **row})
    original.write_bytes(review.read_bytes())
    original.chmod(0o444)

    result = {
        "meta": {"step": "common_words_sheet", "created": datetime.date.today().isoformat(),
                 "git_commit": prep.rd.evaluate.git_commit(), "environment": runtime_environment(),
                 "common_words_config": cfg, "wordlist_sha256": prep.rd.evaluate.sha256(wordlist),
                 "review_sha256": prep.rd.evaluate.sha256(review)},
        "surfaces": len(rows),
        "by_first_timepoint": {tp: len(v) for tp, v in by_tp.items()},
        "matching_sentences": counts,
    }
    (RESULTS / "common_words_sheet.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k in ("surfaces", "by_first_timepoint")}, indent=2))


if __name__ == "__main__":
    main()
