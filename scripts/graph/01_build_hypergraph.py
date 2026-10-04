"""Build one hypergraph: each paper is a hyperedge holding its matched final-vocabulary terms.

Usage: python scripts/graph/01_build_hypergraph.py --timepoint T1 --window build [--variant main]

The node set is always the timepoint's build-time vocabulary, also for the label window.
The T2 label window can only be built after the preregistration is frozen.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.corpus import label_papers, match_papers, window_papers
from triadrx.stats import r5_min_docs
from triadrx.study import load_study

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
TIMEPOINTS = Path("vocab/timepoints")
M1 = Path("results/m1_T1.json")
PREREG = Path("preregistration.md")
OUT_DIR = Path("data/derived/hypergraph")
RESULTS = Path("results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def preregistration_frozen(text: str) -> bool:
    """True once the preregistration names its freeze commit (section 7)."""
    match = re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M)
    return bool(match) and match.group(1).strip() not in {"", "미정"}


def final_terms(tp: str, vocab: dict) -> list[dict]:
    """Main-analysis node set: final terms with the surfaces left after the M1 filter."""
    final = json.loads((TIMEPOINTS / f"{tp}_final.json").read_text())
    fix = json.loads((TIMEPOINTS / f"{tp}_after_m1fix.json").read_text())
    dropped = {(r["term_id"], r["surface"]) for r in fix["removed_surfaces"]}
    terms = []
    for entry in final["kept"]:
        term = vocab[entry["term_id"]]
        surfaces = [s for s in term["surfaces"] if (term["id"], s["text"]) not in dropped]
        terms.append({**term, "status": "active", "surfaces": surfaces})
    return terms


def drop_modes(terms: list[dict], modes: set[str]) -> list[dict]:
    """Remove surfaces with the given match modes; terms left without surfaces are removed."""
    out = []
    for term in terms:
        surfaces = [s for s in term["surfaces"] if s["match"] not in modes]
        if surfaces:
            out.append({**term, "surfaces": surfaces})
    return out


def surface_stratum(term_type: str, mode: str) -> str:
    if mode == "cs":
        return "acronym_cs"
    return "method_ci" if term_type == "M" else "task_ci"


def hyperedges(window: pd.DataFrame, terms: list[dict]) -> pd.DataFrame:
    """One row per paper with at least one matched term."""
    matches = match_papers(window, terms)
    grouped = matches.groupby("paper_id")["term_id"].apply(lambda s: sorted(set(s)))
    dates = window.set_index("id")["first_date"]
    return pd.DataFrame({"paper_id": grouped.index, "first_date": dates.loc[grouped.index].values, "terms": grouped.values})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=["T1", "T2", "applied"])
    parser.add_argument("--window", required=True, choices=["build", "label"])
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    args = parser.parse_args()
    tp, win, variant = args.timepoint, args.window, args.variant

    if tp == "applied" and win == "label":
        sys.exit("the applied mode has no label window")
    if tp == "T2" and win == "label" and not preregistration_frozen(PREREG.read_text()):
        sys.exit("T2 label window is locked until the preregistration freeze commit is recorded")

    study = load_study()
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    terms = final_terms(tp, vocab)
    variant_vocab = TIMEPOINTS / f"{tp}_final_{variant}.json"
    extra_meta = {}

    if variant == "S1":
        # Secondary analysis S1 (preregistration 4.1)
        modes = set(study["secondary_analyses"]["S1"]["drop_match_modes"])
        if win == "build":
            terms = drop_modes(terms, modes)
            m1 = json.loads(M1.read_text())
            lower = {s: v["wilson_lower"] for s, v in m1["final_by_stratum"].items()}
            stability = study["vocab"]["r5_rule"]["stability"]
            window, start, end = window_papers(PAPERS, tp, study)
            edges = hyperedges(window, terms)
            docs = edges.explode("terms").groupby("terms").size()
            kept = []
            for term in terms:
                pi = min(lower[surface_stratum(term["type"], s["match"])] for s in term["surfaces"])
                if docs.get(term["id"], 0) >= r5_min_docs(stability, pi):
                    kept.append(term)
            variant_vocab.write_text(json.dumps({"timepoint": tp, "variant": variant, "terms": kept}, indent=1))
            extra_meta = {"s1_terms_before_r5": len(terms), "s1_terms_after_r5": len(kept)}
            terms = kept
        else:
            if not variant_vocab.exists():
                sys.exit(f"build the S1 build window first ({variant_vocab})")
            terms = json.loads(variant_vocab.read_text())["terms"]

    if win == "build":
        window, start, end = window_papers(PAPERS, tp, study)
    else:
        window, start, end = label_papers(PAPERS, tp, study)
    edges = hyperedges(window, terms)

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tp}_{win}.parquet"
    edges.to_parquet(out)

    sizes = edges["terms"].str.len()
    summary = {
        "window": [start.isoformat(), end.isoformat()],
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "papers_in_window": int(len(window)),
        "papers_with_terms": int(len(edges)),
        "hyperedges_size_ge2": int((sizes >= 2).sum()),
        "hyperedges_size_ge3": int((sizes >= 3).sum()),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
        "total_links": int(sizes.sum()),
        "nodes_in_vocabulary": len(terms),
        "nodes_matched": int(edges.explode("terms")["terms"].nunique()),
        **extra_meta,
    }
    result_path = RESULTS / f"hypergraph_{variant}_{tp}.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"meta": {}, "windows": {}}
    result["meta"] = {
        "step": "hypergraph",
        "timepoint": tp,
        "variant": variant,
        "papers_sha256": sha256(PAPERS),
        "final_vocab_sha256": sha256(TIMEPOINTS / f"{tp}_final.json"),
    }
    result["windows"][win] = summary
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out} and {result_path}")


if __name__ == "__main__":
    main()
