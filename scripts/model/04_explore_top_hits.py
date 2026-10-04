"""Post-hoc exploration of the T2 top-k hits (NOT used for any verdict).

Usage: python scripts/model/04_explore_top_hits.py [--variant main]

Explains the V1 term-block result: which terms the hits in the engine's top k depend on,
how few terms cover all hits, and how often a term-block resample leaves no hit at all.
"""
import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

SCORES = Path("data/derived/scores")
LABELS = Path("data/derived/labels")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("02_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def greedy_cover(hits: list[tuple[str, str, str]]) -> list[str]:
    """Small set of terms such that every hit contains at least one of them (greedy)."""
    remaining, chosen = list(hits), []
    while remaining:
        term, _ = Counter(t for h in remaining for t in h).most_common(1)[0]
        chosen.append(term)
        remaining = [h for h in remaining if term not in h]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    variant = args.variant

    study = load_study()
    cfg = study["evaluation"]
    k = cfg["top_k"]
    df = pd.read_parquet(SCORES / variant / "T2.parquet").merge(
        pd.read_parquet(LABELS / variant / "T2.parquet")[KEYS + ["kind", "label", "observed", "expected", "q_value"]],
        on=KEYS, validate="one_to_one")
    df = df.sort_values(KEYS).reset_index(drop=True)
    ids = np.arange(len(df))
    order = ev.ranking(df["engine_mean"].to_numpy(), ids)
    top = df.iloc[order[:k]].copy()
    top["rank"] = np.arange(1, k + 1)
    hits = top[top["label"] == 1]
    hit_triads = [tuple(r) for r in hits[KEYS].itertuples(index=False)]

    term_in_top = Counter(t for r in top[KEYS].itertuples(index=False) for t in r)
    term_in_hits = Counter(t for h in hit_triads for t in h)
    cover = greedy_cover(hit_triads)

    # Share of term-block resamples whose weighted top k contains no hit
    terms = np.array(sorted(set(df["a"]) | set(df["b"]) | set(df["c"])))
    index = {t: i for i, t in enumerate(terms)}
    triads = np.array([[index[t] for t in r] for r in df[KEYS].itertuples(index=False)])
    y = df["label"].to_numpy().astype(float)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    zero = 0
    for _ in range(cfg["bootstrap_n"]):
        w = ev.term_block_weights(rng, triads, terms)
        if ev.weighted_precision_at_k(order, y, w, k) == 0:
            zero += 1

    all_pos = df[df["label"] == 1]
    pos_terms = Counter(t for r in all_pos[KEYS].itertuples(index=False) for t in r)
    result = {
        "meta": {"step": "explore_top_hits", "variant": variant, "post_hoc": True,
                 "note": "Exploratory, after the verdict; not used for any judgement.",
                 "created": datetime.date.today().isoformat(), "git_commit": git_commit()},
        "top_k": k,
        "hits_in_top_k": [
            {"rank": int(r["rank"]), "triad": [r["a"], r["b"], r["c"]], "kind": r["kind"],
             "observed": int(r["observed"]), "expected": float(r["expected"]), "q_value": float(r["q_value"])}
            for _, r in hits.iterrows()
        ],
        "terms_most_frequent_in_top_k": term_in_top.most_common(15),
        "terms_in_hits": term_in_hits.most_common(),
        "greedy_cover_of_hits": cover,
        "term_block_resamples_with_no_hit_in_top_k": {"count": zero, "of": cfg["bootstrap_n"]},
        "top_k_kind_mix": top["kind"].value_counts().to_dict(),
        "all_t2_positives": {"count": int(len(all_pos)), "distinct_terms": len(pos_terms),
                             "terms_most_frequent": pos_terms.most_common(15)},
    }
    out = RESULTS / f"explore_top_hits_{variant}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k2: result[k2] for k2 in ["hits_in_top_k", "terms_in_hits", "greedy_cover_of_hits",
                                               "term_block_resamples_with_no_hit_in_top_k", "terms_most_frequent_in_top_k",
                                               "top_k_kind_mix"]}, indent=1))
    print("Saved", out)


if __name__ == "__main__":
    main()
