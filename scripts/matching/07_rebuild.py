"""Rebuild a timepoint's vocabulary and hypergraphs with refined matching and merged synonyms (matching preregistration 2-4).

Usage: python scripts/matching/07_rebuild.py --timepoint T1|R2|R3|T3

1. Vocabulary: the refined vocabulary of scripts/matching/03_precision_sample.py (R4 allow decisions,
   M1 remediation filter, surface rules, common-word decisions), then the same-concept groups of
   vocab/synonym_decisions.json are merged into one term each (all surfaces, the representative's id,
   name and type; exclusion phrases follow their terms). Rule R6 on the build window, then rule R5
   with the final stratum lower bounds of results/precision_T1.json.
2. Excluded pairs (4.3): hierarchy pairs of the review whose two terms are in the final vocabulary
   (mapped to representatives) and containment pairs of the final vocabulary
   (scripts/matching/05_synonym_sheet.py `containment`).
3. Hypergraphs: the build window, and the label window for training timepoints, re-matched with the
   final vocabulary and the refined matching. T3 never reads its label window here.
Windows of the engine timepoints are merged in memory as in scripts/engine/14_timepoint_prep.py.

Outputs: vocab/timepoints/<tp>_v3_final.json (terms, R5 details, excluded pairs), results/matching_rebuild_<tp>.json,
data/derived/hypergraph/v3/<tp>_{build,label}.parquet (not committed)
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.corpus import match_papers_refined
from triadrx.stats import r5_min_docs
from triadrx.study import label_window, load_study, runtime_environment

PAPERS = Path("data/arxiv/papers.parquet")
SYNONYMS = Path("vocab/synonym_decisions.json")
PRECISION = Path("results/precision_T1.json")
TIMEPOINTS = Path("vocab/timepoints")
OUT_DIR = Path("data/derived/hypergraph/v3")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sample = load("precision_sample", ROOT / "scripts/matching/03_precision_sample.py")
synonym = load("synonym_sheet", ROOT / "scripts/matching/05_synonym_sheet.py")
prep = load("timepoint_prep", ROOT / "scripts/engine/14_timepoint_prep.py")
r6, graph, evaluate = sample.r6, prep.graph, sample.evaluate


def merge_terms(terms: list[dict], groups: dict[str, list[str]], exclusions: dict) -> tuple[list[dict], set[str], dict]:
    """One term per same-concept group (representative id, name and type; union of surfaces)."""
    by_id = {t["id"]: t for t in terms}
    rep_of = {m: rep for rep, members in groups.items() for m in members}
    merged = {}
    for t in terms:
        rep = rep_of.get(t["id"], t["id"])
        if rep not in merged:
            base = by_id.get(rep, t)
            merged[rep] = {**base, "id": rep, "surfaces": [], "merged_from": []}
        target = merged[rep]
        target["surfaces"] += [s for s in t["surfaces"] if s not in target["surfaces"]]
        target["merged_from"].append(t["id"])
    out = list(merged.values())
    cs_only = {t["id"] for t in out if all(s["match"] == "cs" for s in t["surfaces"])}
    phrases = {}
    for tid, items in exclusions.items():
        phrases.setdefault(rep_of.get(tid, tid), []).extend(items)
    return out, cs_only, phrases


def label_window_papers(tp: str, study: dict) -> pd.DataFrame:
    start, end = label_window(tp, study)
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"],
                             filters=[("first_date", ">=", start), ("first_date", "<=", end)])
    return papers.sort_values("id").reset_index(drop=True)


def hyperedges(window: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    grouped = matches.groupby("paper_id")["term_id"].apply(lambda s: sorted(set(s)))
    dates = window.set_index("id")["first_date"]
    return pd.DataFrame({"paper_id": grouped.index, "first_date": dates.loc[grouped.index].values, "terms": grouped.values})


def main() -> None:
    study = prep.merged_study(load_study())
    cfg = study["matching"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=cfg["timepoints"])
    tp = parser.parse_args().timepoint
    rules = cfg["surface_rules"]
    decisions = json.loads(SYNONYMS.read_text())

    terms, _, exclusions, removal = sample.refined_vocabulary(study)
    terms, cs_only, exclusions = merge_terms(terms, decisions["merge_groups"], exclusions)
    window = sample.build_window_papers(tp, study)
    matches = match_papers_refined(window, terms, cs_only, rules, exclusions)
    links = matches.drop_duplicates(["paper_id", "term_id"])
    degrees = links.groupby("paper_id").size().reindex(window["id"], fill_value=0).to_numpy()
    term_docs = links.groupby("term_id").size()
    total_links, w_p_star, max_w_c = r6.r6_threshold(degrees, study["vocab"]["r6_rule"]["paper_degree_quantile"])
    after_r6 = {t["id"]: t for t in terms if t["id"] in term_docs.index and term_docs[t["id"]] <= max_w_c}

    lower = {s: v["wilson_lower"] for s, v in json.loads(PRECISION.read_text())["final_by_stratum"].items()}
    stability = study["vocab"]["r5_rule"]["stability"]
    final, removed_r5 = [], []
    for tid, t in sorted(after_r6.items()):
        strata = sorted({graph.surface_stratum(t["type"], s["match"]) for s in t["surfaces"]})
        pi = min(lower[s] for s in strata)
        k = r5_min_docs(stability, pi)
        record = {"term_id": tid, "type": t["type"], "docs": int(term_docs[tid]), "strata": strata, "pi": pi, "k": k}
        (final if term_docs[tid] >= k else removed_r5).append(record)
    final_terms = [{**after_r6[r["term_id"]], "status": "active"} for r in final]
    final_ids = {t["id"] for t in final_terms}
    final_cs = {t["id"] for t in final_terms if all(s["match"] == "cs" for s in t["surfaces"])}
    rep_of = {m: rep for rep, members in decisions["merge_groups"].items() for m in members}
    hierarchy = {tuple(sorted((rep_of.get(a, a), rep_of.get(b, b)))) for a, b in decisions["hierarchy_pairs"]}
    hierarchy = {p for p in hierarchy if p[0] != p[1] and set(p) <= final_ids}
    contained = synonym.containment(final_terms, cfg["synonyms"])

    commit = evaluate.git_commit()
    vocab_path = TIMEPOINTS / f"{tp}_v3_final.json"
    vocab_path.write_text(json.dumps({
        "meta": {"step": "matching_v3_final", "timepoint": tp, "created": datetime.date.today().isoformat(), "git_commit": commit,
                 "surface_rules": rules, "synonyms_sha256": evaluate.sha256(SYNONYMS), "precision_sha256": evaluate.sha256(PRECISION),
                 "stratum_lower": lower},
        "stats": {"window_papers": int(len(window)), "total_links": total_links, "paper_degree_quantile_value": w_p_star,
                  "max_term_docs": max_w_c, "terms_matched": int(len(term_docs)), "terms_after_r6": len(after_r6),
                  "terms_final": len(final), **removal},
        "terms": final_terms, "r5": final, "removed_r5": removed_r5,
        "excluded_pairs": {"hierarchy": sorted(map(list, hierarchy)), "containment": sorted(map(list, contained))},
        "exclusion_phrases": {k: [list(p) for p in v] for k, v in exclusions.items() if k in final_ids},
    }, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phrases = {k: v for k, v in exclusions.items() if k in final_ids}
    build = hyperedges(window, match_papers_refined(window, final_terms, final_cs, rules, phrases))
    build.to_parquet(OUT_DIR / f"{tp}_build.parquet")
    summary = {"build_hyperedges": int(len(build))}
    if study["timepoints"][tp].get("role") == "train":
        lwin = label_window_papers(tp, study)
        label = hyperedges(lwin, match_papers_refined(lwin, final_terms, final_cs, rules, phrases))
        label.to_parquet(OUT_DIR / f"{tp}_label.parquet")
        summary["label_hyperedges"] = int(len(label))

    result = {"meta": {"step": "matching_rebuild", "timepoint": tp, "created": datetime.date.today().isoformat(), "git_commit": commit,
                       "environment": runtime_environment(), "vocab_sha256": evaluate.sha256(vocab_path)},
              "terms_after_r6": len(after_r6), "terms_final": len(final), "terms_removed_r5": len(removed_r5),
              "merged_groups_in_final": sum(1 for t in final_terms if len(t.get("merged_from", [])) > 1),
              "excluded_pairs": {"hierarchy": len(hierarchy), "containment": len(contained)}, **summary}
    (RESULTS / f"matching_rebuild_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
