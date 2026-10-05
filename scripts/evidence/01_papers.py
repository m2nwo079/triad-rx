"""Paper evidence for the DVF population (design 6.1, 7.2 F axis, 7.3).

Usage: python scripts/evidence/01_papers.py

The DVF population is the candidate list in the prospective registration file
(config key applied.dvf). Paper evidence is taken directly from the applied-mode
build hypergraph, so no re-matching is done: a paper is evidence for a pair when
both terms are in its matched term set.

Outputs
- data/derived/evidence/papers.parquet (not committed, contains abstracts):
  one row per (pair, paper) with evidence id, date, title and abstract.
- results/evidence_papers.json: per-candidate counts (pair papers, triad papers,
  latest date) used as F-axis inputs. No titles, abstracts or paper ids.
"""
import datetime
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.corpus import PAPERS_COLUMNS
from triadrx.study import data_cutoff, load_study

PAPERS = Path("data/arxiv/papers.parquet")
HYPERGRAPH = Path("data/derived/hypergraph/main/applied_build.parquet")
RANKING = Path("data/derived/applied/ranking.parquet")
OUT_DIR = Path("data/derived/evidence")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
EVIDENCE_ID_FORMAT = {
    "paper": "paper:<arXiv id>",
    "patent": "patent:<grant or publication number>",
    "repo": "repo:<owner/name>",
    "package": "pkg:<PyPI name>",
}


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


def as_utc(series: pd.Series) -> pd.Series:
    """Timezone-aware UTC timestamps; naive values are taken as UTC (arXiv v1 times are UTC)."""
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC")
    return series.dt.tz_convert("UTC")


def pair_key(x: str, y: str) -> tuple[str, str]:
    return (x, y) if x <= y else (y, x)


def main() -> None:
    study = load_study()
    cutoff = data_cutoff()
    prospective_path = Path(study["applied"]["dvf"]["prospective_file"])
    prospective = json.loads(prospective_path.read_text())
    candidates = prospective["top_k_prescription"]

    # The registration must come from the same paper snapshot
    if datetime.datetime.fromisoformat(prospective["meta"]["data_cutoff"]) != cutoff:
        sys.exit("data_cutoff of the prospective file differs from the current arXiv filter result")

    terms_needed = sorted({t for c in candidates for t in c["triad"]})
    edges = pd.read_parquet(HYPERGRAPH)
    edges["first_date"] = as_utc(edges["first_date"])

    # Papers per needed term, from the build hypergraph
    exploded = edges[["paper_id", "terms"]].explode("terms")
    exploded = exploded[exploded["terms"].isin(terms_needed)]
    papers_of = exploded.groupby("terms")["paper_id"].agg(set).to_dict()
    missing = [t for t in terms_needed if t not in papers_of]
    if missing:
        sys.exit(f"candidate terms absent from the build hypergraph: {missing}")

    # Open-triangle flag from the ranking file, for the consistency check
    ranking = pd.read_parquet(RANKING, columns=KEYS + ["H_open_triangle"])
    open_flag = {tuple(r[:3]): int(r[3]) for r in ranking.itertuples(index=False)}

    pairs = sorted({pair_key(x, y) for c in candidates for x, y in itertools.combinations(c["triad"], 2)})
    pair_papers = {p: papers_of[p[0]] & papers_of[p[1]] for p in pairs}
    dates = edges.set_index("paper_id")["first_date"]

    summary = []
    for cand in candidates:
        a, b, c = cand["triad"]
        triad = tuple(cand["triad"])
        if triad not in open_flag:
            sys.exit(f"candidate {triad} not found in {RANKING}")
        triad_papers = papers_of[a] & papers_of[b] & papers_of[c]
        if open_flag[triad] == 1 and triad_papers:
            sys.exit(f"open triangle {triad} has papers containing all three terms")
        pair_rows = []
        for x, y in itertools.combinations(cand["triad"], 2):
            found = pair_papers[pair_key(x, y)]
            latest = dates.loc[sorted(found)].max().isoformat() if found else None
            pair_rows.append({"pair": [x, y], "papers": len(found), "latest": latest})
        union = set().union(*(pair_papers[pair_key(x, y)] for x, y in itertools.combinations(cand["triad"], 2)))
        summary.append({
            "rank": cand["rank"],
            "triad": cand["triad"],
            "open_triangle": open_flag[triad],
            "pairs": pair_rows,
            "triad_papers": len(triad_papers),
            "distinct_papers": len(union),
        })

    # Evidence rows with titles and abstracts (local only)
    rows = [(x, y, pid) for (x, y), found in pair_papers.items() for pid in sorted(found)]
    evidence = pd.DataFrame(rows, columns=["term_x", "term_y", "paper_id"])
    papers = pd.read_parquet(PAPERS, columns=PAPERS_COLUMNS).rename(columns={"id": "paper_id"})
    papers = papers[papers["paper_id"].isin(set(evidence["paper_id"]))]
    papers["first_date"] = as_utc(papers["first_date"])
    evidence = evidence.merge(papers, on="paper_id", how="left", validate="many_to_one")
    if evidence["title"].isna().any():
        sys.exit("evidence papers missing from papers.parquet")
    if (evidence["first_date"] > cutoff).any():
        sys.exit("evidence papers dated after data_cutoff")
    evidence.insert(0, "evidence_id", "paper:" + evidence["paper_id"])
    evidence = evidence.sort_values(["term_x", "term_y", "first_date", "paper_id"], ascending=[True, True, False, True])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(OUT_DIR / "papers.parquet", index=False)

    meta = {
        "step": "evidence_papers",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "reliability_flag": prospective["meta"].get("reliability_flag"),
        "population": study["applied"]["dvf"],
        "evidence_id_format": EVIDENCE_ID_FORMAT,
        "prospective_sha256": sha256(prospective_path),
        "hypergraph_sha256": sha256(HYPERGRAPH),
        "ranking_sha256": sha256(RANKING),
        "papers_sha256": sha256(PAPERS),
        "evidence_parquet_sha256": sha256(OUT_DIR / "papers.parquet"),
    }
    stats = {
        "candidates": len(candidates),
        "terms": len(terms_needed),
        "pairs": len(pairs),
        "pairs_without_papers": sum(1 for p in pairs if not pair_papers[p]),
        "evidence_rows": int(len(evidence)),
        "distinct_papers": int(evidence["paper_id"].nunique()),
    }
    result = {"meta": meta, "stats": stats, "candidates": summary}
    (RESULTS / "evidence_papers.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", OUT_DIR / "papers.parquet", RESULTS / "evidence_papers.json")


if __name__ == "__main__":
    main()
