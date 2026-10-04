"""Build triad candidates for one timepoint from its build-window hypergraph (design 5.2).

Usage: python scripts/graph/03_build_candidates.py --timepoint T1 [--variant main]

Candidates are open triangles on pair-stability edges plus the top-N two-hop triads by
the Adamic-Adar index of the missing pair. Triads made only of tasks are excluded.
Only build-window data is used, so no label information enters.
"""
import argparse
import datetime
import hashlib
import heapq
import itertools
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

HYPERGRAPH = Path("data/derived/hypergraph")
TIMEPOINTS = Path("vocab/timepoints")
VOCAB = Path("vocab/vocab_base.json")
M1 = Path("results/m1_T1.json")
OUT_DIR = Path("data/derived/candidates")
RESULTS = Path("results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def surface_stratum(term_type: str, mode: str) -> str:
    if mode == "cs":
        return "acronym_cs"
    return "method_ci" if term_type == "M" else "task_ci"


def term_pi(tp: str, variant: str) -> dict:
    """Per-term precision lower bound pi (minimum over the strata of its surfaces)."""
    lower = {s: v["wilson_lower"] for s, v in json.loads(M1.read_text())["final_by_stratum"].items()}
    if variant == "main":
        final = json.loads((TIMEPOINTS / f"{tp}_final.json").read_text())
        return {k["term_id"]: k["pi"] for k in final["kept"]}
    terms = json.loads((TIMEPOINTS / f"{tp}_final_{variant}.json").read_text())["terms"]
    return {t["id"]: min(lower[surface_stratum(t["type"], s["match"])] for s in t["surfaces"]) for t in terms}


def pair_threshold(pi_a: float, pi_b: float, stability: float) -> int:
    """Pair stability rule: k_pair = ceil(-ln(1 - s) / (pi_a * pi_b))."""
    return math.ceil(-math.log(1 - stability) / (pi_a * pi_b))


def all_tasks(triad: tuple[str, str, str]) -> bool:
    return all(t.startswith("T:") for t in triad)


def build(edges: pd.DataFrame, pi: dict, stability: float, n_ratio: float):
    """Return (candidate rows, stats)."""
    pairs, closed = Counter(), set()
    for terms in edges["terms"]:
        terms = sorted(set(terms))
        pairs.update(itertools.combinations(terms, 2))
        if len(terms) >= 3:
            closed.update(itertools.combinations(terms, 3))

    adj = defaultdict(set)
    for (a, b), c in pairs.items():
        if c >= pair_threshold(pi[a], pi[b], stability):
            adj[a].add(b)
            adj[b].add(a)

    def weight(a: str, b: str) -> int:
        return pairs[(a, b) if a < b else (b, a)]

    # Open triangles
    rows, open_raw = [], 0
    order = {n: i for i, n in enumerate(sorted(adj, key=lambda n: (len(adj[n]), n)))}
    for a in adj:
        higher = {x for x in adj[a] if order[x] > order[a]}
        for b in higher:
            for c in higher & adj[b]:
                if order[c] <= order[b]:
                    continue
                tri = tuple(sorted((a, b, c)))
                if tri in closed:
                    continue
                open_raw += 1
                if all_tasks(tri):
                    continue
                rows.append({"a": tri[0], "b": tri[1], "c": tri[2], "kind": "open_triangle",
                             "missing_pair_aa": None, "weakest_pair_count": min(weight(*p) for p in itertools.combinations(tri, 2))})
    n_open = len(rows)
    n_two_hop = int(round(n_open * n_ratio))

    # Two-hop triads: centre b linked to a and c, a-c not linked; keep the top N
    aa_cache = {}

    def adamic_adar(a: str, c: str) -> float:
        key = (a, c) if a < c else (c, a)
        if key not in aa_cache:
            aa_cache[key] = sum(1 / math.log(len(adj[z])) for z in adj[a] & adj[c])
        return aa_cache[key]

    heap, space = [], 0
    for centre in sorted(adj):
        nbrs = sorted(adj[centre])
        for i, a in enumerate(nbrs):
            for c in nbrs[i + 1:]:
                if c in adj[a]:
                    continue
                tri = tuple(sorted((a, centre, c)))
                if tri in closed or all_tasks(tri):
                    continue
                space += 1
                aa = adamic_adar(a, c)
                weakest = min(weight(a, centre), weight(centre, c))
                # Min-heap on (score, tie-break); triad ids make ordering deterministic
                key = (aa, weakest, tuple(-ord(ch) for ch in "|".join(tri)))
                item = (key, tri, centre, aa, weakest)
                if len(heap) < n_two_hop:
                    heapq.heappush(heap, item)
                elif n_two_hop and key > heap[0][0]:
                    heapq.heapreplace(heap, item)
    for _, tri, centre, aa, weakest in sorted(heap, reverse=True):
        rows.append({"a": tri[0], "b": tri[1], "c": tri[2], "kind": "two_hop",
                     "missing_pair_aa": aa, "weakest_pair_count": weakest})

    kinds = Counter()
    for r in rows:
        kinds["".join(sorted(t[0] for t in (r["a"], r["b"], r["c"])))] += 1
    stats = {
        "nodes_linked": len(adj),
        "pair_edges": sum(len(v) for v in adj.values()) // 2,
        "open_triangles_before_filter": open_raw,
        "open_triangles": n_open,
        "two_hop_space_after_filter": space,
        "two_hop_selected": len(rows) - n_open,
        "candidates": len(rows),
        "type_mix": dict(sorted(kinds.items())),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    tp, variant = args.timepoint, args.variant

    study = load_study()
    cfg = study["candidates"]
    stability = study["vocab"]["r5_rule"]["stability"]
    edges_path = HYPERGRAPH / variant / f"{tp}_build.parquet"
    edges = pd.read_parquet(edges_path)
    rows, stats = build(edges, term_pi(tp, variant), stability, cfg["two_hop"]["n_ratio_to_open_triangles"])

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tp}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    result = {
        "meta": {"step": "candidates", "timepoint": tp, "variant": variant,
                 "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
                 "candidates_config": cfg, "stability": stability,
                 "hypergraph_sha256": sha256(edges_path), "m1_sha256": sha256(M1)},
        "stats": stats,
    }
    (RESULTS / f"candidates_{variant}_{tp}.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {out}: {stats}")


if __name__ == "__main__":
    main()
