"""Prepare a second-study timepoint with the first-study rules (design 18.4, 18.9 step 6).

Usage: python scripts/engine/14_timepoint_prep.py --timepoint R2|R3|T3
       python scripts/engine/14_timepoint_prep.py --check T1

The timepoint windows come from engine_reliability.timepoints; config timepoints stay unchanged and
the windows are only merged in memory, so the first-study helpers (triadrx.corpus, triadrx.study)
read them. The first-study scripts are not modified; their functions are reused through importlib
in the first-study order:
1. vocabulary: active terms plus the T1 R4 allow decisions, rule R6 on the build window
   (scripts/vocab/04_apply_r6.py), the M1 remediation filter and re-matching
   (scripts/vocab/06_apply_m1_fix.py), rule R5 with the final M1 lower bounds of results/m1_T1.json
   (scripts/vocab/07_m1_r5.py); written as vocab/timepoints/<tp>_{after_r6,after_m1fix,final}.json
2. build-window hypergraph, and for training timepoints the label-window hypergraph and the
   second-study labels (scripts/engine/04_label_v2.py). T3 never reads its label window here
3. first-study candidates (scripts/graph/03_build_candidates.py, their count sets the budget) and
   the selected conditional generator's top budget (scripts/engine/07_conditional_candidates.py)
4. first-study features of those candidates (scripts/engine/08_rediagnose.py)
--check T1 runs step 1 for T1 and compares it with the committed T1 vocabulary files; it writes
nothing and exits with an error on any difference.

Outputs: vocab/timepoints/<tp>_*.json, results/engine_prep_<tp>.json, and under data/derived
(not committed) hypergraph/main/<tp>_{build,label}.parquet, labels_v2/main/<tp>.parquet,
candidates/main/<tp>.parquet, candidates_v2/main/<tp>.parquet, features_v2/main/<tp>.parquet
"""
import argparse
import copy
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.corpus import label_papers, match_papers, window_papers
from triadrx.stats import r5_min_docs
from triadrx.study import load_study, runtime_environment

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
R4_DECISIONS = Path("vocab/r4_decisions_T1.json")
M1 = Path("results/m1_T1.json")
TIMEPOINTS = Path("vocab/timepoints")
DERIVED = Path("data/derived")
RESULTS = Path("results")
STAGE = "after_m1fix"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r6 = load("apply_r6", ROOT / "scripts/vocab/04_apply_r6.py")
m1fix = load("apply_m1_fix", ROOT / "scripts/vocab/06_apply_m1_fix.py")
graph = load("build_hypergraph", ROOT / "scripts/graph/01_build_hypergraph.py")
first_cands = load("build_candidates", ROOT / "scripts/graph/03_build_candidates.py")
labels_v2 = load("label_v2", ROOT / "scripts/engine/04_label_v2.py")
recall = load("conditional_recall", ROOT / "scripts/engine/06_conditional_recall.py")
cond = load("conditional_candidates", ROOT / "scripts/engine/07_conditional_candidates.py")
rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")


def merged_study(study: dict) -> dict:
    """Study with the engine timepoints added to `timepoints` in memory only."""
    out = copy.deepcopy(study)
    out["timepoints"].update(study["engine_reliability"]["timepoints"])
    return out


def vocabulary(tp: str, study: dict) -> dict:
    """The three first-study vocabulary stages of one timepoint, in the first-study file formats."""
    r4 = json.loads(R4_DECISIONS.read_text())
    allowed = {d["term_id"] for d in r4["decisions"] if d["decision"] == "allow"}
    base = json.loads(VOCAB.read_text())["terms"]
    terms = [{**t, "status": "active"} for t in base if t["status"] == "active" or t["id"] in allowed]
    term_type = {t["id"]: t["type"] for t in terms}
    window, start, end = window_papers(PAPERS, tp, study)

    # R6 on all matched terms (scripts/vocab/04_apply_r6.py)
    links = match_papers(window, terms).drop_duplicates(["paper_id", "term_id"])
    degrees = links.groupby("paper_id").size().reindex(window["id"], fill_value=0).to_numpy()
    term_docs = links.groupby("term_id").size()
    total_links, w_p_star, max_w_c = r6.r6_threshold(degrees, study["vocab"]["r6_rule"]["paper_degree_quantile"])
    removed, kept = term_docs[term_docs > max_w_c], term_docs[term_docs <= max_w_c]
    after_r6 = {"removed_r6": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in removed.sort_values(ascending=False).items()],
                "kept": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in kept.sort_index().items()],
                "stats": {"window_papers": int(len(window)), "total_links": total_links,
                          "paper_degree_quantile_value": w_p_star, "max_term_docs": max_w_c}}

    # M1 remediation filter and re-matching (scripts/vocab/06_apply_m1_fix.py)
    cfg = study["vocab"]["m1_fix"]
    r6_kept = set(kept.index)
    fix_terms, removed_surfaces, removed_terms = m1fix.filter_terms(
        [t for t in terms if t["id"] in r6_kept], re.compile(cfg["junk_pattern"]),
        {(e["term_id"], e["surface"]) for e in cfg["r3_exclude"]})
    fix_docs = match_papers(window, fix_terms).drop_duplicates(["paper_id", "term_id"]).groupby("term_id").size()
    after_fix = {"removed_surfaces": removed_surfaces, "removed_terms": removed_terms,
                 "kept": [{"term_id": t, "type": term_type[t], "docs": int(n)} for t, n in fix_docs.sort_index().items()]}

    # R5 with the final M1 lower bounds (scripts/vocab/07_m1_r5.py)
    lower = {s: v["wilson_lower"] for s, v in json.loads(M1.read_text())["final_by_stratum"].items()}
    stability = study["vocab"]["r5_rule"]["stability"]
    vocab = {t["id"]: t for t in base}
    dropped = {(r["term_id"], r["surface"]) for r in removed_surfaces}
    final_kept, final_removed = [], []
    for entry in after_fix["kept"]:
        term = vocab[entry["term_id"]]
        modes = {s["match"] for s in term["surfaces"] if (term["id"], s["text"]) not in dropped}
        strata = sorted({graph.surface_stratum(term["type"], m) for m in modes})
        pi = min(lower[s] for s in strata)
        k = r5_min_docs(stability, pi)
        (final_kept if entry["docs"] >= k else final_removed).append({**entry, "strata": strata, "pi": pi, "k": k})
    final = {"removed_r5": final_removed, "kept": final_kept}
    return {"window": [start.isoformat(), end.isoformat()], "after_r6": after_r6, "after_m1fix": after_fix, "final": final}


def compare_with_committed(tp: str, v: dict) -> list[str]:
    """Differences between a rebuilt vocabulary and the committed first-study files."""
    errors = []
    checks = {"after_r6": ["kept", "removed_r6"], "after_m1fix": ["kept", "removed_surfaces", "removed_terms"],
              "final": ["kept", "removed_r5"]}
    for stage, keys in checks.items():
        committed = json.loads((TIMEPOINTS / f"{tp}_{stage}.json").read_text())
        for key in keys:
            if committed[key] != v[stage][key]:
                errors.append(f"{stage}.{key} differs")
    return errors


def write_vocabulary(tp: str, v: dict, meta: dict) -> None:
    TIMEPOINTS.mkdir(parents=True, exist_ok=True)
    for stage in ("after_r6", "after_m1fix", "final"):
        body = {k: x for k, x in v[stage].items() if k != "stats"}
        stats = v[stage].get("stats", {})
        out = {"meta": {**meta, "step": f"engine_prep_{stage}", "stage": STAGE if stage == "final" else stage},
               "stats": stats, **body}
        (TIMEPOINTS / f"{tp}_{stage}.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--timepoint", choices=["R2", "R3", "T3"])
    group.add_argument("--check", choices=["T1"])
    args = parser.parse_args()
    study = merged_study(load_study())

    if args.check:
        errors = compare_with_committed(args.check, vocabulary(args.check, study))
        if errors:
            sys.exit(f"{args.check} vocabulary differs from the committed files: {errors}")
        print(f"{args.check} vocabulary matches the committed files")
        return

    tp = args.timepoint
    er = study["engine_reliability"]
    commit = rd.evaluate.git_commit()
    meta = {"timepoint": tp, "created": datetime.date.today().isoformat(), "git_commit": commit,
            "papers_sha256": rd.evaluate.sha256(PAPERS), "vocab_sha256": rd.evaluate.sha256(VOCAB),
            "r4_decisions_sha256": rd.evaluate.sha256(R4_DECISIONS), "m1_sha256": rd.evaluate.sha256(M1)}
    v = vocabulary(tp, study)
    write_vocabulary(tp, v, {**meta, "window": v["window"]})
    summary = {"vocabulary": {stage: len(v[stage]["kept"]) for stage in ("after_r6", "after_m1fix", "final")}}

    # Hypergraphs: the node set is the build-time final vocabulary, also for the label window
    base = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    terms = graph.final_terms(tp, base)
    hyper = DERIVED / "hypergraph/main"
    hyper.mkdir(parents=True, exist_ok=True)
    window, _, _ = window_papers(PAPERS, tp, study)
    build = graph.hyperedges(window, terms)
    build.to_parquet(hyper / f"{tp}_build.parquet")
    summary["build_hyperedges"] = int(len(build))
    is_training = er["timepoints"][tp]["role"] == "train"
    if is_training:
        lwin, _, _ = label_papers(PAPERS, tp, study)
        label = graph.hyperedges(lwin, terms)
        label.to_parquet(hyper / f"{tp}_label.parquet")
        realised = labels_v2.evidence.significant_new_triads(build, label, study["labels"]["fdr"])
        positive, _ = labels_v2.evidence.split_by_evidence(tp, er["label_v2"], label, realised)
        table = labels_v2.label_table(realised, positive, er["label_v2"]["min_evidence_papers"])
        (DERIVED / "labels_v2/main").mkdir(parents=True, exist_ok=True)
        table.to_parquet(DERIVED / f"labels_v2/main/{tp}.parquet", index=False)
        summary.update(label_hyperedges=int(len(label)), realised_significant_triads=int(len(table)),
                       label_v2_positive=int(table["label_v2"].sum()))

    # First-study candidates set the budget; the conditional generator fills it
    stability = study["vocab"]["r5_rule"]["stability"]
    rows, stats = first_cands.build(build, first_cands.term_pi(tp, "main"), stability,
                                    study["candidates"]["two_hop"]["n_ratio_to_open_triangles"])
    (DERIVED / "candidates/main").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(DERIVED / f"candidates/main/{tp}.parquet")
    budget = er["verdict"]["budget_multiple_of_first_candidates"] * stats["candidates"]
    sel = er["conditional"]["generator"]["selected"]
    g = recall.prepare(build, tp, stability)
    cands = cond.top_candidates(g, sel["seed_pairs"], sel["score"], budget)
    if is_training:
        cands = cond.attach_labels(cands, pd.read_parquet(DERIVED / f"labels_v2/main/{tp}.parquet"))
    (DERIVED / "candidates_v2/main").mkdir(parents=True, exist_ok=True)
    cands.to_parquet(DERIVED / f"candidates_v2/main/{tp}.parquet", index=False)
    summary.update(first_study_candidates=int(stats["candidates"]), budget=int(budget), conditional_candidates=int(len(cands)))
    if is_training:
        summary["conditional_label_v2_positive"] = int(cands["label_v2"].sum())

    feats = rd.build_features(tp, study, DERIVED / "candidates_v2/main")
    (DERIVED / "features_v2/main").mkdir(parents=True, exist_ok=True)
    feats.to_parquet(DERIVED / f"features_v2/main/{tp}.parquet", index=False)

    result = {"meta": {**meta, "step": "engine_prep", "environment": runtime_environment(), "window": v["window"],
                       "timepoint_config": er["timepoints"][tp]},
              **summary}
    (RESULTS / f"engine_prep_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: x for k, x in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
