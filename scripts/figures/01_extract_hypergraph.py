"""Hypergraph neighbourhood of the top represcription for the paper figure.

Usage: python scripts/figures/01_extract_hypergraph.py [--rank 4]   (from the repository root; needs local derived data)

Reads one triad of the represcription registration (--rank, default 4: the top triad whose terms are widely
known; rank 1 holds the ambiguous acronym "pc", which a reader would need explained) and the applied v3 build hypergraph, in which each
paper is a hyperedge over the final terms it mentions. Writes, for the three terms of that triad: the number
of papers holding each pair, the number holding all three (zero for a candidate: an open triangle), and for
each pair up to `examples` papers with their other terms, so the figure can draw real hyperedges. Papers with
at least one other term come first, then ascending paper id. The output is small and committed, so the figure can be redrawn without the derived data.

Output: figures/data/hypergraph_example.json
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

BUILD = Path("data/derived/hypergraph/v3/applied_build.parquet")
OUT = Path("figures/data/hypergraph_example.json")


def neighbourhood(edges: pd.DataFrame, triad: list[str], examples: int) -> dict:
    """Pair and triple counts of a triad and example hyperedges for each pair."""
    sets = [(pid, set(ts)) for pid, ts in zip(edges["paper_id"], edges["terms"])]
    triad_set = set(triad)
    pairs = {}
    for a, b in itertools.combinations(sorted(triad), 2):
        terms_of = {pid: ts for pid, ts in sets}
        holders = sorted((pid for pid, ts in sets if a in ts and b in ts), key=lambda pid: (not (terms_of[pid] - triad_set), pid))
        pairs[f"{a}|{b}"] = {"papers": len(holders),
                             "examples": [{"paper_id": pid, "other_terms": sorted(terms_of[pid] - triad_set)}
                                          for pid in holders[:examples]]}
    return {"triad": sorted(triad), "pairs": pairs,
            "papers_with_all_three": sum(1 for _, ts in sets if triad_set <= ts),
            "papers_in_window": len(sets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=4)
    rank = parser.parse_args().rank
    study = load_study()
    run = Path(study["represcription"]["run_dir"])
    reg = json.loads((run / json.loads((run / "config/study.json").read_text())["prospective"]["registration"]).read_text())
    top = next(e for e in reg["top_k_prescription"] if e["rank"] == rank)
    edges = pd.read_parquet(BUILD, columns=["paper_id", "terms"])
    out = {"meta": {"step": "figure_hypergraph_example", "rank": top["rank"], "score": top["score"],
                    "build_sha256": hashlib.sha256(BUILD.read_bytes()).hexdigest()},
           **neighbourhood(edges, top["triad"], examples=3)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}), {k: v["papers"] for k, v in out["pairs"].items()})
    print("Saved", OUT)


if __name__ == "__main__":
    main()
