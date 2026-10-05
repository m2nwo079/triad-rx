"""Papers with Code paper-to-repository links for the DVF population (design 4.1, 6.1).

Usage: python scripts/evidence/04_pwc_links.py

Scope: applied-mode build-window papers whose matched terms include at least
one DVF-population term (term level). Papers that are pair evidence from
01_papers.py are flagged, so later steps can aggregate at pair or term level.
Only GitHub repositories are kept, since repository metadata, README search and
star history (later steps) use the GitHub API; other hosts are counted.

Repository creation dates (the date basis for paper-code links, design 4.4) are
not in the snapshot; they are fetched in the next step, and links to repositories
created after data_cutoff are dropped there.

Outputs
- data/derived/evidence/pwc_links.parquet (not committed): one row per (paper, repository)
- results/evidence_pwc_links.json: counts only
"""
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
from triadrx.study import data_cutoff, load_study

LINKS = Path("data/scope/links.parquet")
HYPERGRAPH = Path("data/derived/hypergraph/main/applied_build.parquet")
PAIR_EVIDENCE = Path("data/derived/evidence/papers.parquet")
OUT_DIR = Path("data/derived/evidence")
RESULTS = Path("results")
LINK_COLUMNS = ["paper_arxiv_id", "repo_url", "is_official", "mentioned_in_paper", "mentioned_in_github", "framework"]
GITHUB = re.compile(r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)
VERSION = re.compile(r"v\d+$")


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


def github_repo(url: str) -> str | None:
    """owner/name in lower case (GitHub names are case-insensitive), or None for other hosts."""
    match = GITHUB.match(url.strip()) if isinstance(url, str) else None
    if not match:
        return None
    owner, name = match.group(1), match.group(2)
    if name.endswith(".git"):
        name = name[:-4]
    return f"{owner}/{name}".lower()


def main() -> None:
    study = load_study()
    cutoff = data_cutoff()
    prospective_path = Path(study["applied"]["dvf"]["prospective_file"])
    candidates = json.loads(prospective_path.read_text())["top_k_prescription"]
    dvf_terms = {t for c in candidates for t in c["triad"]}

    # Build-window papers that mention at least one DVF-population term
    edges = pd.read_parquet(HYPERGRAPH, columns=["paper_id", "first_date", "terms"])
    mentions = edges["terms"].map(lambda ts: bool(dvf_terms.intersection(ts)))
    papers = edges.loc[mentions, ["paper_id", "first_date"]]
    pair_papers = set(pd.read_parquet(PAIR_EVIDENCE, columns=["paper_id"])["paper_id"])

    links = pd.read_parquet(LINKS, columns=LINK_COLUMNS)
    total_links = len(links)
    links = links[links["paper_arxiv_id"].notna() & (links["paper_arxiv_id"] != "")].copy()
    links["paper_id"] = links["paper_arxiv_id"].str.strip().str.replace(VERSION, "", regex=True)
    links = links[links["paper_id"].isin(set(papers["paper_id"]))].copy()
    links_in_scope = len(links)
    links["repo"] = links["repo_url"].map(github_repo)
    non_github = int(links["repo"].isna().sum())
    links = links[links["repo"].notna()].drop_duplicates(["paper_id", "repo"])

    links = links.merge(papers, on="paper_id", how="left", validate="many_to_one")
    links["pair_evidence"] = links["paper_id"].isin(pair_papers)
    links.insert(0, "evidence_id", "repo:" + links["repo"])
    links = links[
        ["evidence_id", "repo", "paper_id", "first_date", "pair_evidence", "repo_url",
         "is_official", "mentioned_in_paper", "mentioned_in_github", "framework"]
    ].sort_values(["paper_id", "repo"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    links.to_parquet(OUT_DIR / "pwc_links.parquet", index=False)

    linked_papers = set(links["paper_id"])
    latest_linked = pd.to_datetime(links["first_date"]).max() if len(links) else None
    meta = {
        "step": "evidence_pwc_links",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "scope": "build-window papers matching at least one DVF-population term",
        "links_sha256": sha256(LINKS),
        "hypergraph_sha256": sha256(HYPERGRAPH),
        "pair_evidence_sha256": sha256(PAIR_EVIDENCE),
        "pwc_links_parquet_sha256": sha256(OUT_DIR / "pwc_links.parquet"),
    }
    stats = {
        "dvf_terms": len(dvf_terms),
        "scope_papers": int(len(papers)),
        "pair_evidence_papers": len(pair_papers),
        "snapshot_links": int(total_links),
        "links_to_scope_papers": int(links_in_scope),
        "non_github_links_dropped": non_github,
        "github_links": int(len(links)),
        "distinct_repos": int(links["repo"].nunique()),
        "linked_scope_papers": len(linked_papers),
        "linked_pair_evidence_papers": len(linked_papers & pair_papers),
        "latest_linked_paper_date": latest_linked.isoformat() if latest_linked is not None else None,
    }
    result = {"meta": meta, "stats": stats}
    (RESULTS / "evidence_pwc_links.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", OUT_DIR / "pwc_links.parquet", RESULTS / "evidence_pwc_links.json")


if __name__ == "__main__":
    main()
