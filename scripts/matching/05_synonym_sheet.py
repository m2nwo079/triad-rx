"""Synonym and hierarchy review sheet (matching preregistration 4.1).

Usage: python scripts/matching/05_synonym_sheet.py

For each timepoint in matching.synonyms.timepoints, the final vocabulary (vocab/timepoints/<tp>_final.json
through scripts/graph/01_build_hypergraph.py) and the build-window hypergraph
(data/derived/hypergraph/main/<tp>_build.parquet) give candidate pairs by the rules of 4.1: same surface
form without spaces, hyphens and underscores; an acronym equal to the initials of another term's
multi-word surface; the same word set; high co-occurrence (nPMI and paper count thresholds).
Containment pairs (4.3) are left out because a rule handles them. Each pair is shown with both terms'
names and surfaces and up to titles_per_pair titles of build-window papers holding both, drawn with
the configured seed, from the first timepoint where the pair appears. Candidates are only chosen
for review here; the decisions come from the review.

Outputs: matching.synonyms.review_file and its read-only _original.csv (not committed),
results/synonym_sheet.json
"""
import csv
import datetime
import importlib.util
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment
from triadrx.text import ci_tokens, norm

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
HYPERGRAPH = Path("data/derived/hypergraph/main")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load("build_hypergraph", ROOT / "scripts/graph/01_build_hypergraph.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def stem(word: str, suffixes: list[str], min_length: int) -> str:
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_length:
            return word[: -len(suffix)]
    return word


def word_set(name: str, cfg: dict) -> frozenset:
    return frozenset(stem(w, cfg["stem_suffixes"], cfg["stem_min_length"]) for w in ci_tokens(name))


def containment(terms: list[dict], cfg: dict) -> set[tuple[str, str]]:
    """Sorted term pairs where one name's word set is a proper subset of the other's (4.3)."""
    words = {t["id"]: word_set(t["key"], cfg) for t in terms}
    return {tuple(sorted((a, b))) for a in words for b in words if a != b and words[a] and words[a] < words[b]}


def candidate_pairs(terms: list[dict], build: pd.DataFrame, cfg: dict) -> dict[tuple, dict]:
    """Candidate pairs (sorted ids) with the rules that found them and their co-occurrence statistics."""
    found = defaultdict(set)
    form = defaultdict(set)
    initials = defaultdict(set)
    for t in terms:
        for s in t["surfaces"]:
            if s["match"] == "ci":
                form[re.sub(r"[\s\-_]", "", norm(s["text"]))].add(t["id"])
                tokens = ci_tokens(s["text"])
                if len(tokens) >= 2:
                    initials["".join(w[0] for w in tokens)].add(t["id"])
    for ids in form.values():
        for pair in itertools.combinations(sorted(ids), 2):
            found[pair].add("form")
    for t in terms:
        short = [s["text"].lower() for s in t["surfaces"] if s["match"] == "cs"]
        name = t["key"].replace(" ", "")
        if name.isalpha() and len(name) <= cfg["acronym_max_letters"] and " " not in t["key"]:
            short.append(name.lower())
        for acronym in short:
            for other in initials.get(acronym, ()):
                if other != t["id"]:
                    found[tuple(sorted((t["id"], other)))].add("initials")
    by_words = defaultdict(set)
    for t in terms:
        by_words[word_set(t["key"], cfg)].add(t["id"])
    for ids in by_words.values():
        for pair in itertools.combinations(sorted(ids), 2):
            found[pair].add("same_words")
    n = len(build)
    docs, pairs = Counter(), Counter()
    for ts in build["terms"]:
        ts = sorted(set(ts))
        docs.update(ts)
        pairs.update(itertools.combinations(ts, 2))
    stats = {}
    for (a, b), c in pairs.items():
        p = c / n
        npmi = 1.0 if p >= 1 else math.log(p / (docs[a] / n * docs[b] / n)) / -math.log(p)
        stats[(a, b)] = (c, npmi)
        if c >= cfg["min_cooccurrence"] and npmi >= cfg["npmi_min"]:
            found[(a, b)].add("cooccurrence")
    skip = containment(terms, cfg)
    return {pair: {"rules": sorted(rules), "cooccurrence": stats.get(pair, (0, None))[0], "npmi": stats.get(pair, (0, None))[1]}
            for pair, rules in found.items() if pair not in skip}


def main() -> None:
    study = load_study()
    cfg = study["matching"]["synonyms"]
    review = Path(cfg["review_file"])
    if review.exists():
        sys.exit(f"{review} already exists; a new attempt needs a new file name")
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    titles = pd.read_parquet(PAPERS, columns=["id", "title"]).set_index("id")["title"]
    rng = random.Random(cfg["seed"])
    rows, first_tp, per_tp = [], {}, {}
    for tp in cfg["timepoints"]:
        terms = graph.final_terms(tp, vocab)
        build = pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet")
        found = candidate_pairs(terms, build, cfg)
        per_tp[tp] = len(found)
        new = [p for p in sorted(found) if p not in first_tp]
        holders = defaultdict(list)
        wanted = set(new)
        for pid, ts in zip(build["paper_id"], build["terms"]):
            for pair in itertools.combinations(sorted(set(ts)), 2):
                if pair in wanted:
                    holders[pair].append(pid)
        for pair in new:
            first_tp[pair] = tp
            info = found[pair]
            picked = rng.sample(holders[pair], min(cfg["titles_per_pair"], len(holders[pair])))
            a, b = vocab[pair[0]], vocab[pair[1]]
            rows.append({"term_a": pair[0], "name_a": a["key"], "surfaces_a": "; ".join(s["text"] for s in a["surfaces"]),
                         "term_b": pair[1], "name_b": b["key"], "surfaces_b": "; ".join(s["text"] for s in b["surfaces"]),
                         "rules": "+".join(info["rules"]), "first_timepoint": tp, "cooccurrence": info["cooccurrence"],
                         **{f"title_{i + 1}": f"[{p}] {' '.join(str(titles.get(p, '')).split())}" for i, p in enumerate(picked)}})
    columns = ["review_id", "term_a", "name_a", "surfaces_a", "term_b", "name_b", "surfaces_b", "rules", "first_timepoint",
               "cooccurrence"] + [f"title_{i + 1}" for i in range(cfg["titles_per_pair"])] + ["decision", "note"]
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for n, row in enumerate(rows):
            writer.writerow({"review_id": f"sy-{n:03d}", **row})
    original = review.with_name(review.stem + "_original.csv")
    original.write_bytes(review.read_bytes())
    original.chmod(0o444)
    result = {"meta": {"step": "synonym_sheet", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                       "environment": runtime_environment(), "synonyms_config": cfg, "review_sha256": evaluate.sha256(review)},
              "pairs": len(rows), "candidates_by_timepoint": per_tp,
              "by_rule": dict(Counter(r for row in rows for r in row["rules"].split("+"))),
              "by_first_timepoint": dict(Counter(first_tp.values()))}
    (RESULTS / "synonym_sheet.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
