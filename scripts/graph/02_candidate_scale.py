"""Measure how many triad candidates each build-window hypergraph would produce.

Exploratory sizing for the candidate rule (design section 5.2). Uses build windows only,
so no label information is involved.

Usage: python scripts/graph/02_candidate_scale.py
"""
import datetime
import itertools
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HYPERGRAPH = Path("data/derived/hypergraph/main")
RESULTS = Path("results/candidate_scale_main.json")
MIN_SUPPORTS = [1, 2, 3, 5]
# Listing triangles in pure Python is slow; above this many we only count them
LIST_LIMIT = 20_000_000
# Same stability level as rule R5 (config vocab.r5_rule.stability)
STABILITY = json.loads(Path("config/study.json").read_text())["vocab"]["r5_rule"]["stability"]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def triads_and_pairs(edges: pd.DataFrame) -> tuple[Counter, set]:
    """Pair co-occurrence counts and the set of triads that already co-occur in one paper."""
    pairs, closed = Counter(), set()
    for terms in edges["terms"]:
        terms = sorted(set(terms))
        pairs.update(itertools.combinations(terms, 2))
        if len(terms) >= 3:
            closed.update(itertools.combinations(terms, 3))
    return pairs, closed


def scale(pairs: Counter, closed: set, min_support: int) -> dict:
    adj = defaultdict(set)
    for (a, b), c in pairs.items():
        if c >= min_support:
            adj[a].add(b)
            adj[b].add(a)
    order = {n: i for i, n in enumerate(sorted(adj, key=lambda n: (len(adj[n]), n)))}
    # Wedges (paths of length 2) bound the 2-hop candidate space
    wedges = sum(len(v) * (len(v) - 1) // 2 for v in adj.values())
    triangles = open_tri = open_with_task = 0
    listed = True
    started = time.time()
    for a in adj:
        higher = {b for b in adj[a] if order[b] > order[a]}
        for b in higher:
            for c in higher & adj[b]:
                if order[c] <= order[b]:
                    continue
                triangles += 1
                if triangles > LIST_LIMIT:
                    listed = False
                    break
                tri = tuple(sorted((a, b, c)))
                if tri not in closed:
                    open_tri += 1
                    if any(t.startswith("T:") for t in tri):
                        open_with_task += 1
            if not listed:
                break
        if not listed:
            break
    return {
        "min_support": min_support,
        "nodes": len(adj),
        "pair_edges": sum(len(v) for v in adj.values()) // 2,
        "wedges": wedges,
        "triangles": triangles if listed else None,
        "open_triangles": open_tri if listed else None,
        "open_triangles_with_task": open_with_task if listed else None,
        "listed": listed,
        "seconds": round(time.time() - started, 1),
    }


def term_pi(tp: str) -> dict:
    """Per-term precision lower bound used by R5 (stored in the final vocabulary)."""
    final = json.loads((Path("vocab/timepoints") / f"{tp}_final.json").read_text())
    return {k["term_id"]: k["pi"] for k in final["kept"]}


def stability_scale(pairs: Counter, closed: set, pi: dict, stability: float) -> dict:
    """Candidate sizes when a pair counts as linked only if it passes the pair stability rule.

    Rule: k_pair = ceil(-ln(1 - stability) / (pi_a * pi_b)), the R5 idea applied to pairs.
    """
    adj = defaultdict(set)
    for (a, b), c in pairs.items():
        if c >= math.ceil(-math.log(1 - stability) / (pi[a] * pi[b])):
            adj[a].add(b)
            adj[b].add(a)
    order = {n: i for i, n in enumerate(sorted(adj, key=lambda n: (len(adj[n]), n)))}
    wedges_open = 0
    for b in adj:
        nbrs = sorted(adj[b])
        for i, a in enumerate(nbrs):
            for c in nbrs[i + 1:]:
                if c not in adj[a]:
                    wedges_open += 1
    open_tri = open_mixed = 0
    for a in adj:
        higher = {x for x in adj[a] if order[x] > order[a]}
        for b in higher:
            for c in higher & adj[b]:
                if order[c] <= order[b]:
                    continue
                tri = tuple(sorted((a, b, c)))
                if tri in closed:
                    continue
                open_tri += 1
                kinds = {t.split(":", 1)[0] for t in tri}
                if {"M", "T"} <= kinds:
                    open_mixed += 1
    return {
        "rule": "pair_stability",
        "stability": stability,
        "nodes": len(adj),
        "pair_edges": sum(len(v) for v in adj.values()) // 2,
        "open_triangles": open_tri,
        "open_triangles_mixed_method_task": open_mixed,
        "two_hop_wedges_with_missing_pair": wedges_open,
    }


def main() -> None:
    out = {"meta": {"step": "candidate_scale", "created": datetime.date.today().isoformat(),
                    "git_commit": git_commit(), "list_limit": LIST_LIMIT}, "timepoints": {}}
    for tp in ["T1", "T2", "applied"]:
        edges = pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet")
        pairs, closed = triads_and_pairs(edges)
        rows = []
        for s in MIN_SUPPORTS:
            row = scale(pairs, closed, s)
            rows.append(row)
            print(tp, row, flush=True)
        stab = stability_scale(pairs, closed, term_pi(tp), STABILITY)
        print(tp, stab, flush=True)
        out["timepoints"][tp] = {"closed_triads": len(closed), "by_min_support": rows, "pair_stability": stab}
    RESULTS.write_text(json.dumps(out, indent=2))
    print("Saved", RESULTS)


if __name__ == "__main__":
    main()