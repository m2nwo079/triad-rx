"""Exploratory plot of the projected co-occurrence network of one hypergraph.

Usage: python scripts/viz/01_plot_network.py --timepoint T1 --window build [--top 60]

Each paper (hyperedge) is expanded into term pairs. Edge weight is the number of papers
in which two terms co-occur. Figures are for inspection only and do not feed any rule.
"""
import argparse
import itertools
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

# Render to files without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

HYPERGRAPH = Path("data/derived/hypergraph")
OUT_DIR = Path("data/derived/figures")
TYPE_COLOR = {"M": "#2b6cb0", "T": "#e05252"}


def projected_counts(edges: pd.DataFrame) -> tuple[Counter, Counter, int]:
    """Return (term paper counts, pair co-occurrence counts, number of papers)."""
    term_counts, pair_counts = Counter(), Counter()
    for terms in edges["terms"]:
        terms = sorted(set(terms))
        term_counts.update(terms)
        pair_counts.update(itertools.combinations(terms, 2))
    return term_counts, pair_counts, len(edges)


def npmi(pair: int, a: int, b: int, n: int) -> float:
    """Normalized pointwise mutual information of two terms over papers."""
    p_ab, p_a, p_b = pair / n, a / n, b / n
    if p_ab >= 1:
        return 1.0
    return math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)


def build_graph(term_counts, pair_counts, n, min_count: int, min_npmi: float | None) -> nx.Graph:
    graph = nx.Graph()
    for (a, b), c in pair_counts.items():
        if c < min_count:
            continue
        score = npmi(c, term_counts[a], term_counts[b], n)
        if min_npmi is not None and score < min_npmi:
            continue
        graph.add_edge(a, b, weight=c, npmi=score)
    for node in graph.nodes:
        graph.nodes[node]["docs"] = term_counts[node]
    return graph


def component_layout(graph: nx.Graph, seed: int = 7) -> dict:
    """Lay out each connected component separately and place them on a grid, largest first."""
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    cols = max(1, math.ceil(math.sqrt(len(components))))
    pos = {}
    for i, nodes in enumerate(components):
        sub = graph.subgraph(nodes)
        if len(sub) == 1:
            local = {next(iter(nodes)): (0.0, 0.0)}
        else:
            local = nx.spring_layout(sub, seed=seed, k=1.5 / math.sqrt(len(sub)), weight="weight")
        # Larger components get more room
        scale = 0.5 + 0.5 * math.sqrt(len(sub) / len(components[0]))
        row, col = divmod(i, cols)
        for node, (x, y) in local.items():
            pos[node] = (col * 2.6 + x * scale, -row * 2.6 + y * scale)
    return pos


def draw(graph: nx.Graph, top: int, title: str, out: Path) -> None:
    strength = dict(graph.degree(weight="weight"))
    keep = sorted(strength, key=strength.get, reverse=True)[:top]
    sub = graph.subgraph(keep).copy()
    # Fixed seed so the layout is reproducible
    pos = component_layout(sub)
    max_docs = max((sub.nodes[n]["docs"] for n in sub), default=1)
    sizes = [200 + 2500 * sub.nodes[n]["docs"] / max_docs for n in sub]
    colors = [TYPE_COLOR.get(n.split(":", 1)[0], "#888888") for n in sub]
    max_w = max((d["weight"] for _, _, d in sub.edges(data=True)), default=1)
    widths = [0.3 + 3 * d["weight"] / max_w for _, _, d in sub.edges(data=True)]

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw_networkx_edges(sub, pos, width=widths, alpha=0.25, edge_color="#666666", ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(sub, pos, labels={n: n.split(":", 1)[1] for n in sub}, font_size=9, ax=ax)
    for kind, color in TYPE_COLOR.items():
        ax.scatter([], [], c=color, s=120, label={"M": "method", "T": "task"}[kind])
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    parser.add_argument("--window", default="build", choices=["build", "label"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-npmi", type=float, default=0.2)
    args = parser.parse_args()

    # Exploration must not touch the T2 label window before the preregistration freeze
    if args.timepoint == "T2" and args.window == "label":
        sys.exit("T2 label window is not plotted")

    path = HYPERGRAPH / args.variant / f"{args.timepoint}_{args.window}.parquet"
    edges = pd.read_parquet(path)
    term_counts, pair_counts, n = projected_counts(edges)
    tag = f"{args.variant} {args.timepoint} {args.window}"

    raw = build_graph(term_counts, pair_counts, n, args.min_count, None)
    draw(raw, args.top, f"{tag}: top {args.top} terms by co-occurrence strength (pairs in >= {args.min_count} papers)",
         OUT_DIR / f"network_{args.variant}_{args.timepoint}_{args.window}_count.png")

    assoc = build_graph(term_counts, pair_counts, n, args.min_count, args.min_npmi)
    draw(assoc, args.top, f"{tag}: top {args.top} terms, edges with nPMI >= {args.min_npmi}",
         OUT_DIR / f"network_{args.variant}_{args.timepoint}_{args.window}_npmi.png")
    print(f"papers {n}, terms {len(term_counts)}, pairs {len(pair_counts)}, "
          f"edges kept: count {raw.number_of_edges()}, nPMI {assoc.number_of_edges()}")


if __name__ == "__main__":
    main()
