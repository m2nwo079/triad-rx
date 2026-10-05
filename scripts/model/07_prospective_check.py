"""Prospective check of the registered prescription against the proximity baseline (preregistration 8).

Usage: python scripts/model/07_prospective_check.py --horizon 3 --papers <filtered papers parquet> \
           --filter-summary <its results/arxiv_filter_*.json>

Written and fixed before any outcome exists. Locked until the label window has ended and a
new arXiv snapshot (filtered with scripts/collect/01_filter_arxiv.py) covers it.
- horizon = buffer_years + label_years: verdict window (cutoff + buffer, cutoff + buffer + label]
- horizon in descriptive_years: descriptive window (cutoff, cutoff + horizon], no verdict
Matching, null-model labels and the bootstrap reuse the frozen scripts through importlib.
P1: lift of the prescription top k over the population realisation rate, lower bound > 1.
P2: prescription minus baseline Precision@k, lower bound > 0. Both bootstraps must pass.
DVF: realisation by axis score within the DVF population, descriptive only.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("results")
VOCAB_BASE = Path("vocab/vocab_base.json")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load("build_hypergraph", ROOT / "scripts/graph/01_build_hypergraph.py")
labels = load("build_labels", ROOT / "scripts/graph/04_build_labels.py")
evaluate = load("evaluate", ROOT / "scripts/model/02_evaluate.py")
register = load("prospective_register", ROOT / "scripts/model/06_prospective_register.py")


def add_years(moment: datetime.datetime, years: int) -> datetime.datetime:
    try:
        return moment.replace(year=moment.year + years)
    except ValueError:
        # 29 February in a non-leap target year
        return moment.replace(year=moment.year + years, day=28)


def check_window(cutoff: datetime.datetime, horizon: int, cfg: dict) -> tuple[datetime.datetime, datetime.datetime, bool]:
    """(start, end] of the label window and whether it is the verdict window."""
    primary = cfg["buffer_years"] + cfg["label_years"]
    if horizon == primary:
        return add_years(cutoff, cfg["buffer_years"]), add_years(cutoff, primary), True
    if horizon in cfg["descriptive_years"]:
        return cutoff, add_years(cutoff, horizon), False
    raise ValueError(f"horizon {horizon} is neither the verdict horizon {primary} nor in descriptive_years")


def lock_reason(today: datetime.datetime, snapshot_cutoff: datetime.datetime, end: datetime.datetime) -> str | None:
    if today < end:
        return f"locked until {end.isoformat()} (label window has not ended)"
    if snapshot_cutoff < end:
        return f"the snapshot ends at {snapshot_cutoff.isoformat()}, before the window end {end.isoformat()}"
    return None


def null_model_labels(edges: pd.DataFrame, cands: pd.DataFrame, fdr: float) -> np.ndarray:
    """Same triadic null model and BH as scripts/graph/04_build_labels.py (design 5.4)."""
    degree = edges["terms"].str.len().to_numpy()
    total_links = int(degree.sum())
    term_docs = Counter(t for terms in edges["terms"] for t in set(terms))
    degree_groups = sorted(Counter(degree.tolist()).items())
    postings = defaultdict(set)
    for i, terms in enumerate(edges["terms"]):
        for t in terms:
            postings[t].add(i)
    pvalues = []
    for a, b, c in cands[KEYS].itertuples(index=False):
        observed = len(postings[a] & postings[b] & postings[c])
        groups = []
        for d, n_d in degree_groups:
            q = 1.0
            for x in (a, b, c):
                q *= min(1.0, d * term_docs.get(x, 0) / total_links) if total_links else 0.0
            groups.append((n_d, q))
        pvalues.append(labels.upper_tail(observed, groups))
    return (labels.benjamini_hochberg(np.array(pvalues)) <= fdr).astype(int)


def scores(order_p: np.ndarray, order_b: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> dict:
    total = w.sum()
    base = (w * y).sum() / total if total > 0 else float("nan")
    p_presc = evaluate.weighted_precision_at_k(order_p, y, w, k)
    p_base = evaluate.weighted_precision_at_k(order_b, y, w, k)
    lift = p_presc / base if base and base > 0 else float("nan")
    return {"precision_prescription": p_presc, "precision_baseline": p_base, "base_rate": base,
            "lift": lift, "precision_diff": p_presc - p_base}


def judge(pop: pd.DataFrame, y: np.ndarray, k: int, feature: str, ev: dict) -> dict:
    order_p = np.array(register.prescription_order(pop))
    order_b = np.array(register.baseline_order(pop, feature))
    terms = np.array(sorted(set(pop["a"]) | set(pop["b"]) | set(pop["c"])))
    index = {t: i for i, t in enumerate(terms)}
    triads = np.array([[index[t] for t in row] for row in pop[KEYS].itertuples(index=False)])
    point = scores(order_p, order_b, y, np.ones(len(pop)), k)
    rng = np.random.default_rng(ev["bootstrap_seed"])
    boot = {"candidate": {"lift": [], "precision_diff": []}, "term_block": {"lift": [], "precision_diff": []}}
    for _ in range(ev["bootstrap_n"]):
        for kind, w in (("candidate", evaluate.candidate_weights(rng, len(pop))),
                        ("term_block", evaluate.term_block_weights(rng, triads, terms))):
            s = scores(order_p, order_b, y, w, k)
            boot[kind]["lift"].append(s["lift"])
            boot[kind]["precision_diff"].append(s["precision_diff"])
    ci = {kind: {m: evaluate.interval(v, ev["ci_level"]) for m, v in vals.items()} for kind, vals in boot.items()}
    return {"point": point, "ci": ci,
            "verdict": {"P1": all(ci[kind]["lift"]["lower"] > 1 for kind in ci),
                        "P2": all(ci[kind]["precision_diff"]["lower"] > 0 for kind in ci)}}


def dvf_by_score(dvf_candidates: list[dict], realised: dict) -> dict:
    """Descriptive only: realisation by axis score within the DVF population (missing axes apart)."""
    out = {}
    for axis in ("D", "F", "V"):
        cells = defaultdict(lambda: {"n": 0, "realised": 0})
        for c in dvf_candidates:
            score = c["axes"][axis]["score"]
            key = "missing" if score is None else str(int(score))
            cells[key]["n"] += 1
            cells[key]["realised"] += realised[tuple(c["triad"])]
        out[axis] = dict(sorted(cells.items()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--filter-summary", type=Path, required=True)
    args = parser.parse_args()

    study = load_study()
    cfg, ev = study["prospective"], study["evaluation"]
    cutoff = data_cutoff()
    start, end, is_verdict = check_window(cutoff, args.horizon, cfg)
    snapshot_cutoff = datetime.datetime.fromisoformat(json.loads(args.filter_summary.read_text())["meta"]["data_cutoff"])
    reason = lock_reason(datetime.datetime.now(datetime.timezone.utc), snapshot_cutoff, end)
    if reason:
        sys.exit(reason)

    registered = json.loads((RESULTS / "prospective_register.json").read_text())["sha256"]
    paths = {"ranking": Path(cfg["ranking"]), "vocab": Path(cfg["vocab"]),
             "registration": Path(cfg["registration"]), "baseline": Path(cfg["baseline"]["file"])}
    for name, path in paths.items():
        if register.sha256(path) != registered[name]:
            sys.exit(f"{path} differs from the registered file")

    papers = pd.read_parquet(args.papers, columns=["id", "title", "abstract", "first_date"])
    window = papers[(papers["first_date"] > start) & (papers["first_date"] <= end)].sort_values("id")
    vocab = {t["id"]: t for t in json.loads(VOCAB_BASE.read_text())["terms"]}
    edges = graph.hyperedges(window, graph.final_terms("applied", vocab))

    ranking = pd.read_parquet(paths["ranking"])
    ranking["label"] = null_model_labels(edges, ranking, study["labels"]["fdr"])
    pop = register.population(ranking)
    y = pop["label"].to_numpy().astype(float)
    result = {
        "meta": {"step": "prospective_check", "created": datetime.date.today().isoformat(),
                 "git_commit": register.git_commit(), "horizon_years": args.horizon, "verdict_window": is_verdict,
                 "window": [start.isoformat(), end.isoformat()], "data_cutoff": cutoff.isoformat(),
                 "snapshot_cutoff": snapshot_cutoff.isoformat(), "papers_sha256": register.sha256(args.papers),
                 "prospective_config": cfg, "evaluation_config": ev},
        "label_window_papers": int(len(edges)),
        "population": int(len(pop)),
        "positives": int(y.sum()),
        **judge(pop, y, cfg["top_k"], cfg["baseline"]["feature"], ev),
    }
    realised = {tuple(t): int(v) for t, v in zip(pop[KEYS].itertuples(index=False, name=None), y)}
    result["dvf_by_score"] = dvf_by_score(json.loads((RESULTS / "dvf.json").read_text())["candidates"], realised)
    if not is_verdict:
        result["verdict"] = None
    out = RESULTS / f"prospective_check_{args.horizon}y.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"verdict": result["verdict"], "point": result["point"]}, indent=2))


if __name__ == "__main__":
    main()
