"""Combination evidence for realised triads and the human review sample (engine preregistration 2.1).

Usage: python scripts/engine/02_label_evidence.py

Design timepoint only (engine_reliability.label_v2.design_timepoint, T1); T3 is refused.
Realised new triads are defined as in the first study's candidate recall: triads that co-occur in
a label-window paper, never co-occurred in a build-window paper, are not made only of tasks, and are
significant under the same null model and BH. For each such (triad, paper) the rule checks whether
some window of `window_sentences` consecutive sentences (title first) matches all three terms and
contains no comparison cue. Matching uses the timepoint's final terms and triadrx.text, with the
evidence matching rules (label_v2.evidence_matching) applied on top. Triads in earlier attempts'
keys are excluded from the sample.

Outputs
- data/review/label_evidence_<tp>_attempt<n>.csv (not committed): blind review sheet, shuffled
- data/review/label_evidence_<tp>_attempt<n>_key.csv (not committed): rule side of each row
- results/label_evidence_<tp>_attempt<n>.json: counts, sample settings and hashes, no text
  (attempt 0 was saved as results/label_evidence_<tp>.json before attempts were numbered)
"""
import csv
import datetime
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study
from triadrx.text import Matcher

PAPERS = Path("data/arxiv/papers.parquet")
HYPERGRAPH = Path("data/derived/hypergraph/main")
VOCAB = Path("vocab/vocab_base.json")
RESULTS = Path("results")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
PLACEHOLDER = "․"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load("build_hypergraph", ROOT / "scripts/graph/01_build_hypergraph.py")
labels = load("build_labels", ROOT / "scripts/graph/04_build_labels.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")


def split_sentences(text: str, abbreviations: list[str]) -> list[str]:
    """Sentence split that does not break after the listed abbreviations."""
    protected = text or ""
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", PLACEHOLDER))
    parts = [s.strip() for s in SENTENCE_END.split(protected) if s.strip()]
    return [s.replace(PLACEHOLDER, ".") for s in parts]


def cue_pattern(cues: list[str]) -> re.Pattern:
    alternatives = sorted((re.escape(c) for c in cues), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])", re.IGNORECASE)


def evidence_terms(matcher: Matcher, sentences: list[str], cs_only: set[str], rules: dict) -> list[set]:
    """Matched terms per sentence after the evidence matching rules (engine preregistration 2.1).

    drop_nested: a match whose tokens lie inside a longer match of the other pass is dropped.
    acronym_needs_full_name: a case-sensitive match counts only when the same term also has a
    case-insensitive match in the paper, unless the term has no case-insensitive surface.
    """
    kept = []
    for s in sentences:
        spans = matcher.spans(s)
        if rules["drop_nested"]:
            spans = [m for m in spans if not any(o["start"] <= m["start"] and m["end"] <= o["end"]
                                                 and o["end"] - o["start"] > m["end"] - m["start"] for o in spans)]
        kept.append(spans)
    full = {m["id"] for spans in kept for m in spans if m["mode"] == "ci"}
    out = []
    for spans in kept:
        out.append({m["id"] for m in spans if m["mode"] == "ci" or not rules["acronym_needs_full_name"]
                    or m["id"] in cs_only or m["id"] in full})
    return out


def evidence_vocab(terms: list[dict], skip_chars: list[str]) -> tuple[list[dict], set[str]]:
    """Terms without surfaces whose characters the tokenizer drops, and the ids with only acronym surfaces."""
    out = []
    for t in terms:
        surfaces = [s for s in t["surfaces"] if not any(c in s["text"] for c in skip_chars)]
        if surfaces:
            out.append({**t, "surfaces": surfaces})
    return out, {t["id"] for t in out if all(s["match"] == "cs" for s in t["surfaces"])}


def has_evidence(triad: tuple, sentence_terms: list[set], sentences: list[str], size: int, cues: re.Pattern) -> bool:
    """Some window of `size` consecutive sentences holds all three terms and no comparison cue."""
    needed = set(triad)
    for i in range(len(sentences)):
        window = range(i, min(i + size, len(sentences)))
        if needed <= set().union(*(sentence_terms[j] for j in window)):
            if not cues.search(" ".join(sentences[j] for j in window)):
                return True
    return False


def significant_new_triads(build: pd.DataFrame, label: pd.DataFrame, fdr: float) -> dict[tuple, list[int]]:
    """Same set as scripts/model/03_candidate_recall.py, with the label-window papers of each triad."""
    closed = set()
    for ts in build["terms"]:
        ts = sorted(set(ts))
        if len(ts) >= 3:
            closed.update(itertools.combinations(ts, 3))
    papers = defaultdict(list)
    for i, ts in enumerate(label["terms"]):
        for tri in itertools.combinations(sorted(set(ts)), 3):
            if tri not in closed and not all(t.startswith("T:") for t in tri):
                papers[tri].append(i)
    degree = label["terms"].str.len().to_numpy()
    total = int(degree.sum())
    docs = Counter(t for ts in label["terms"] for t in set(ts))
    groups = sorted(Counter(degree.tolist()).items())
    deg = np.array([d for d, _ in groups], dtype=float)
    size = np.array([n for _, n in groups], dtype=float)
    triads = list(papers)
    pvalues = np.empty(len(triads))
    for i, tri in enumerate(triads):
        q = np.ones_like(deg)
        for x in tri:
            q *= np.minimum(1.0, deg * docs[x] / total)
        o = len(papers[tri])
        if o == 1:
            pvalues[i] = -np.expm1(np.sum(size * np.log1p(-np.minimum(q, 1 - 1e-16))))
        else:
            pvalues[i] = labels.upper_tail(o, list(zip(size.astype(int), q)))
    qvals = labels.benjamini_hochberg(pvalues) if triads else np.array([])
    return {tri: papers[tri] for tri, qv in zip(triads, qvals) if qv <= fdr}


def draw(rng, pool: dict[tuple, list[str]], n: int, exclude: set) -> list[tuple[tuple, str]]:
    """n distinct triads at random, then one of their papers at random."""
    keys = sorted(k for k in pool if k not in exclude)
    chosen = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    return [(keys[i], pool[keys[i]][rng.integers(len(pool[keys[i]]))]) for i in sorted(chosen)]


def split_by_evidence(tp: str, cfg: dict, label: pd.DataFrame, realised: dict[tuple, list[int]]):
    """Papers of each realised triad with and without combination evidence (label_v2 rule)."""
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    rules = cfg["evidence_matching"]
    ev_terms, cs_only = evidence_vocab(graph.final_terms(tp, vocab), rules["skip_surface_chars"])
    matcher = Matcher(ev_terms)
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"]).set_index("id")
    cues = cue_pattern(cfg["comparison_cues"])
    cache = {}

    def sentences_of(idx: int):
        if idx not in cache:
            pid = label.iloc[idx]["paper_id"]
            row = text.loc[pid]
            sents = ([row["title"]] if cfg["title_as_first_sentence"] else []) + split_sentences(row["abstract"], cfg["abbreviations"])
            cache[idx] = (pid, sents, evidence_terms(matcher, sents, cs_only, rules))
        return cache[idx]

    positive, negative = defaultdict(list), defaultdict(list)
    for tri, idxs in realised.items():
        for idx in idxs:
            pid, sents, terms = sentences_of(idx)
            if has_evidence(tri, terms, sents, cfg["window_sentences"], cues):
                positive[tri].append(pid)
            else:
                negative[tri].append(pid)
    return positive, negative


def main() -> None:
    study = load_study()
    cfg = study["engine_reliability"]["label_v2"]
    tp = cfg["design_timepoint"]
    if tp not in study["timepoints"]:
        sys.exit(f"{tp} is not a development timepoint")
    review, key = Path(cfg["review_file"]), Path(cfg["review_key_file"])
    if review.exists() or key.exists():
        sys.exit(f"{review} already exists; a new attempt needs a new file name and a new sample")

    build = pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet")
    label = pd.read_parquet(HYPERGRAPH / f"{tp}_label.parquet")
    realised = significant_new_triads(build, label, study["labels"]["fdr"])

    positive, negative = split_by_evidence(tp, cfg, label, realised)
    evidence_count = Counter({tri: len(pids) for tri, pids in positive.items()})
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"]).set_index("id")

    # Triads reviewed in earlier attempts are never drawn again
    seen = set()
    for path in cfg["previous_review_keys"]:
        prev = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        seen.update(zip(prev["term_a"], prev["term_b"], prev["term_c"]))
    rng = np.random.default_rng(cfg["sample"]["seed"])
    pos = draw(rng, positive, cfg["sample"]["positive"], seen)
    neg = draw(rng, negative, cfg["sample"]["negative"], seen | {t for t, _ in pos})
    rows = [("positive", t, p) for t, p in pos] + [("negative", t, p) for t, p in neg]
    order = rng.permutation(len(rows))
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", newline="", encoding="utf-8-sig") as fr, key.open("w", newline="", encoding="utf-8-sig") as fk:
        wr, wk = csv.writer(fr), csv.writer(fk)
        wr.writerow(["review_id", "term_a", "term_b", "term_c", "paper_id", "title", "abstract", "judgement", "term_error", "note"])
        wk.writerow(["review_id", "rule", "term_a", "term_b", "term_c", "paper_id"])
        for n, i in enumerate(order):
            side, tri, pid = rows[i]
            rid = f"le-{n:03d}"
            wr.writerow([rid, *tri, pid, text.loc[pid, "title"], text.loc[pid, "abstract"], "", "", ""])
            wk.writerow([rid, side, *tri, pid])

    n_pairs = sum(len(v) for v in realised.values())
    m = cfg["min_evidence_papers"]
    result = {
        "meta": {"step": "label_evidence", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "label_v2_config": cfg,
                 "inputs_sha256": {"build": evaluate.sha256(HYPERGRAPH / f"{tp}_build.parquet"),
                                   "label": evaluate.sha256(HYPERGRAPH / f"{tp}_label.parquet")},
                 "previous_keys_sha256": {p: evaluate.sha256(Path(p)) for p in cfg["previous_review_keys"]},
                 "review_sha256": evaluate.sha256(review), "key_sha256": evaluate.sha256(key)},
        "realised_significant_triads": len(realised),
        "triad_paper_pairs": n_pairs,
        "pairs_with_evidence": int(sum(evidence_count.values())),
        "pair_evidence_share": sum(evidence_count.values()) / n_pairs if n_pairs else float("nan"),
        "triads_meeting_min_evidence": sum(1 for t in realised if evidence_count[t] >= m),
        "sample": {"positive": len(pos), "negative": len(neg)},
    }
    (RESULTS / f"label_evidence_{tp}_attempt{cfg['attempt']}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
