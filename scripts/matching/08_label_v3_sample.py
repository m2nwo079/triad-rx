"""Label v3 on the design timepoint and its blind review sample (matching preregistration 5).

Usage: python scripts/matching/08_label_v3_sample.py

Realised triads come from the v3 hypergraphs (data/derived/hypergraph/v3) with the first-study null
model and BH (scripts/engine/02_label_evidence.py `significant_new_triads`, reused through importlib).
Triads holding an excluded pair of the timepoint's v3 vocabulary (hierarchy or containment) are
dropped. Combination evidence uses the adopted engine rule (engine_reliability.label_v2) with the
v3 terms, the refined matching (triadrx.text.refined_spans, surface rules and exclusion phrases)
and the comparison cues. The sample is drawn as in the engine study (`draw`): distinct triads first,
then one paper each, rule positives and rule negatives, shuffled into one blind sheet.

Outputs
- data/derived/labels_v3/<tp>.parquet (not committed): a, b, c, n_papers, n_evidence, label_v3
- matching.label_v3.review_file and review_key_file, plus the read-only _original.csv (not committed)
- results/label_v3_<tp>.json: counts, settings and hashes, no text
"""
import csv
import datetime
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment
from triadrx.text import Matcher, refined_spans

PAPERS = Path("data/arxiv/papers.parquet")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
TIMEPOINTS = Path("vocab/timepoints")
OUT_DIR = Path("data/derived/labels_v3")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load("label_evidence", ROOT / "scripts/engine/02_label_evidence.py")
labels_v2 = load("label_v2", ROOT / "scripts/engine/04_label_v2.py")
evaluate = evidence.evaluate


def excluded_pairs(vocab: dict) -> set[tuple[str, str]]:
    return {tuple(sorted(p)) for kind in ("hierarchy", "containment") for p in vocab["excluded_pairs"][kind]}


def holds_excluded_pair(triad: tuple, excluded: set) -> bool:
    a, b, c = triad
    return any(tuple(sorted(p)) in excluded for p in ((a, b), (a, c), (b, c)))


def split_by_evidence(vocab: dict, ev_cfg: dict, rules: dict, label: pd.DataFrame, realised: dict) -> tuple[dict, dict]:
    """Papers of each realised triad with and without combination evidence under the v3 matching."""
    terms = vocab["terms"]
    matcher = Matcher(terms)
    cs_only = {t["id"] for t in terms if all(s["match"] == "cs" for s in t["surfaces"])}
    phrases = {k: [tuple(p) for p in v] for k, v in vocab["exclusion_phrases"].items()}
    cues = evidence.cue_pattern(ev_cfg["comparison_cues"])
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"],
                           filters=[("id", "in", list(label["paper_id"]))]).set_index("id")
    cache = {}
    positive, negative = defaultdict(list), defaultdict(list)
    for tri, idxs in realised.items():
        for idx in idxs:
            if idx not in cache:
                pid = label.iloc[idx]["paper_id"]
                row = text.loc[pid]
                sents = ([row["title"]] if ev_cfg["title_as_first_sentence"] else []) + evidence.split_sentences(row["abstract"], ev_cfg["abbreviations"])
                terms_by_sentence = [{m["id"] for m in spans} for spans in refined_spans(matcher, sents, cs_only, rules, phrases)]
                cache[idx] = (pid, sents, terms_by_sentence)
            pid, sents, terms_by_sentence = cache[idx]
            (positive if evidence.has_evidence(tri, terms_by_sentence, sents, ev_cfg["window_sentences"], cues) else negative)[tri].append(pid)
    return positive, negative


def main() -> None:
    study = load_study()
    cfg = study["matching"]["label_v3"]
    ev_cfg = study["engine_reliability"]["label_v2"]
    rules = study["matching"]["surface_rules"]
    tp = cfg["design_timepoint"]
    review, key = Path(cfg["review_file"]), Path(cfg["review_key_file"])
    if review.exists() or key.exists():
        sys.exit(f"{review} already exists; a new attempt needs a new file name and a new sample")
    vocab = json.loads((TIMEPOINTS / f"{tp}_v3_final.json").read_text())
    paths = {"build": HYPERGRAPH / f"{tp}_build.parquet", "label": HYPERGRAPH / f"{tp}_label.parquet"}
    build, label = pd.read_parquet(paths["build"]), pd.read_parquet(paths["label"])
    realised_all = evidence.significant_new_triads(build, label, study["labels"]["fdr"])
    excluded = excluded_pairs(vocab)
    realised = {t: v for t, v in realised_all.items() if not holds_excluded_pair(t, excluded)}
    positive, negative = split_by_evidence(vocab, ev_cfg, rules, label, realised)
    table = labels_v2.label_table(realised, positive, ev_cfg["min_evidence_papers"]).rename(columns={"label_v2": "label_v3"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    table.to_parquet(out, index=False)

    rng = np.random.default_rng(cfg["sample"]["seed"])
    pos = evidence.draw(rng, positive, cfg["sample"]["positive"], set())
    neg = evidence.draw(rng, negative, cfg["sample"]["negative"], {t for t, _ in pos})
    rows = [("positive", t, p) for t, p in pos] + [("negative", t, p) for t, p in neg]
    order = rng.permutation(len(rows))
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"],
                           filters=[("id", "in", [p for _, _, p in rows])]).set_index("id")
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", newline="", encoding="utf-8-sig") as fr, key.open("w", newline="", encoding="utf-8-sig") as fk:
        wr, wk = csv.writer(fr), csv.writer(fk)
        wr.writerow(["review_id", "term_a", "term_b", "term_c", "paper_id", "title", "abstract", "judgement", "term_error", "note"])
        wk.writerow(["review_id", "rule", "term_a", "term_b", "term_c", "paper_id"])
        for n, i in enumerate(order):
            side, tri, pid = rows[i]
            rid = f"l3-{n:03d}"
            wr.writerow([rid, *tri, pid, text.loc[pid, "title"], text.loc[pid, "abstract"], "", "", ""])
            wk.writerow([rid, side, *tri, pid])
    original = review.with_name(review.stem + "_original.csv")
    original.write_bytes(review.read_bytes())
    for path in (original, key):
        path.chmod(0o444)

    n_pairs = int(table["n_papers"].sum())
    result = {
        "meta": {"step": "label_v3", "timepoint": tp, "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                 "environment": runtime_environment(), "label_v3_config": cfg, "evidence_config": ev_cfg, "surface_rules": rules,
                 "inputs_sha256": {k: evaluate.sha256(v) for k, v in paths.items()},
                 "vocab_sha256": evaluate.sha256(TIMEPOINTS / f"{tp}_v3_final.json"), "output_sha256": evaluate.sha256(out),
                 "review_sha256": evaluate.sha256(review), "key_sha256": evaluate.sha256(key)},
        "realised_significant_triads": len(realised_all),
        "dropped_for_excluded_pairs": len(realised_all) - len(realised),
        "realised_kept": len(realised),
        "triad_paper_pairs": n_pairs,
        "pairs_with_evidence": int(table["n_evidence"].sum()),
        "label_v3_positive": int(table["label_v3"].sum()),
        "sample": {"positive": len(pos), "negative": len(neg)},
    }
    (RESULTS / f"label_v3_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
