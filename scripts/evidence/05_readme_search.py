"""GitHub README search for arXiv ids of pair-evidence papers after the PWC snapshot (design 6.2).

Usage: python scripts/evidence/05_readme_search.py

Scope: papers from 01_papers.py (pair evidence of the DVF population) first
submitted after the latest linked paper of the PWC snapshot (read from
results/evidence_pwc_links.json).

Phase 1, search: one repository search per paper,
  "<arXiv id>" in:readme created:<=<data_cutoff date>
first page only (up to 100 repositories, best match); total_count is kept so
truncation is visible. Repositories created after data_cutoff are excluded by the
query (design 4.4).
Phase 2, verify: each hit's README is fetched and checked for the exact id with a
regex (search tokenizes on punctuation, so "2509.12345" can match other numbers).
README text is not stored; only whether the id is present and how many distinct
arXiv ids the README contains (to separate paper lists from implementations later).

Both phases append to JSON-lines caches in data/derived/evidence/ and skip work
already done, so the script can be stopped and rerun. Star counts returned by the
search are values after data_cutoff and are not stored.

Needs GITHUB_TOKEN in .env (or in the environment).

Outputs
- data/derived/evidence/readme_search.jsonl, readme_verify.jsonl (caches, not committed)
- data/derived/evidence/readme_links.parquet (not committed): one row per (paper, repository)
- results/evidence_log.json: key "github_readme_search", one entry per query (design 6.2)
- results/evidence_readme_links.json: counts only
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff

PAIR_EVIDENCE = Path("data/derived/evidence/papers.parquet")
PWC_SUMMARY = Path("results/evidence_pwc_links.json")
OUT_DIR = Path("data/derived/evidence")
SEARCH_CACHE = OUT_DIR / "readme_search.jsonl"
VERIFY_CACHE = OUT_DIR / "readme_verify.jsonl"
RESULTS = Path("results")
EVIDENCE_LOG = RESULTS / "evidence_log.json"
LOG_KEY = "github_readme_search"
API = "https://api.github.com"
PER_PAGE = 100
ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")


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


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    env = Path(".env")
    if not token and env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "GITHUB_TOKEN":
                token = value.strip().strip('"').strip("'")
    if not token:
        sys.exit("GITHUB_TOKEN not found in the environment or .env")
    return token


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def request(url: str, token: str, accept: str) -> tuple[int, bytes, object]:
    """GET with rate-limit handling; returns (status, body, headers)."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "triad-rx-evidence",
    })
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Header lookups must be case-insensitive (HTTP/2 sends lower-case names)
                headers = resp.headers
                body = resp.read()
            remaining = int(headers.get("X-RateLimit-Remaining", "1"))
            if remaining == 0:
                wait = max(0, int(headers.get("X-RateLimit-Reset", "0")) - int(time.time())) + 2
                print(f"rate limit reached, sleeping {wait}s")
                time.sleep(wait)
            return 200, body, headers
        except urllib.error.HTTPError as err:
            headers = err.headers
            if err.code in (403, 429) and ("Retry-After" in headers or headers.get("X-RateLimit-Remaining") == "0"):
                if "Retry-After" in headers:
                    wait = int(headers["Retry-After"]) + 1
                else:
                    wait = max(0, int(headers.get("X-RateLimit-Reset", "0")) - int(time.time())) + 2
                print(f"HTTP {err.code}, sleeping {wait}s")
                time.sleep(wait)
                continue
            if err.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return err.code, b"", headers
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    return -1, b"", None


def read_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def search_phase(paper_ids: list[str], cutoff_date: str, token: str) -> list[dict]:
    done = {r["paper_id"] for r in read_cache(SEARCH_CACHE) if r["status"] == 200}
    todo = [p for p in paper_ids if p not in done]
    print(f"search: {len(done)} cached, {len(todo)} to run")
    for i, pid in enumerate(todo, 1):
        query = f'"{pid}" in:readme created:<={cutoff_date}'
        url = f"{API}/search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": PER_PAGE})
        status, body, _ = request(url, token, "application/vnd.github+json")
        record = {"paper_id": pid, "query": query, "queried_at": now_utc(), "status": status}
        if status == 200:
            data = json.loads(body)
            record["total_count"] = data["total_count"]
            record["incomplete_results"] = data["incomplete_results"]
            record["items"] = [
                {"repo": item["full_name"].lower(), "created_at": item["created_at"]}
                for item in data["items"]
            ]
        append(SEARCH_CACHE, record)
        # Search API allows 30 requests per minute with a token
        time.sleep(2.1)
        if i % 100 == 0:
            print(f"search {i}/{len(todo)}")
    return read_cache(SEARCH_CACHE)


def verify_phase(hits: pd.DataFrame, token: str) -> list[dict]:
    done = {(r["repo"], r["paper_id"]) for r in read_cache(VERIFY_CACHE) if r["status"] in (200, 404)}
    readme_cache = {}
    todo = [(r, p) for r, p in hits[["repo", "paper_id"]].itertuples(index=False) if (r, p) not in done]
    print(f"verify: {len(done)} cached, {len(todo)} to run")
    for i, (repo, pid) in enumerate(todo, 1):
        if repo not in readme_cache:
            status, body, _ = request(f"{API}/repos/{repo}/readme", token, "application/vnd.github.raw")
            text = body.decode("utf-8", errors="replace") if status == 200 else ""
            readme_cache[repo] = (status, text)
        status, text = readme_cache[repo]
        ids = set(ARXIV_NEW.findall(text))
        append(VERIFY_CACHE, {
            "repo": repo,
            "paper_id": pid,
            "checked_at": now_utc(),
            "status": status,
            "id_found": pid in ids,
            "readme_arxiv_ids": len(ids),
        })
        if i % 200 == 0:
            print(f"verify {i}/{len(todo)}")
    return read_cache(VERIFY_CACHE)


def main() -> None:
    cutoff = data_cutoff()
    token = github_token()
    snapshot = pd.Timestamp(json.loads(PWC_SUMMARY.read_text())["stats"]["latest_linked_paper_date"])
    if snapshot.tzinfo is None:
        snapshot = snapshot.tz_localize("UTC")

    papers = pd.read_parquet(PAIR_EVIDENCE, columns=["paper_id", "first_date"]).drop_duplicates("paper_id")
    papers = papers[papers["first_date"] > snapshot].sort_values("paper_id")
    paper_ids = papers["paper_id"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    searches = [r for r in search_phase(paper_ids, cutoff.date().isoformat(), token) if r["paper_id"] in set(paper_ids)]
    latest = {}
    for r in searches:
        if r["status"] == 200:
            latest[r["paper_id"]] = r
    failed = sorted(set(paper_ids) - set(latest))
    if failed:
        sys.exit(f"{len(failed)} searches did not succeed; rerun to retry them")

    hits = pd.DataFrame(
        [(pid, item["repo"], item["created_at"]) for pid, r in latest.items() for item in r["items"]],
        columns=["paper_id", "repo", "created_at"],
    ).drop_duplicates(["paper_id", "repo"])
    print(f"hits: {len(hits)} (paper, repo) pairs, {hits['repo'].nunique()} repositories")

    check_columns = ["repo", "paper_id", "checked_at", "status", "id_found", "readme_arxiv_ids"]
    checks = pd.DataFrame(verify_phase(hits, token), columns=check_columns).drop_duplicates(["repo", "paper_id"], keep="last")
    links = hits.merge(checks[["repo", "paper_id", "status", "id_found", "readme_arxiv_ids"]], on=["repo", "paper_id"], how="left")
    links["created_at"] = pd.to_datetime(links["created_at"], utc=True)
    if (links["created_at"] > cutoff).any():
        sys.exit("search returned repositories created after data_cutoff")
    links = links.merge(papers, on="paper_id", how="left")
    links.insert(0, "evidence_id", "repo:" + links["repo"])
    links = links.sort_values(["paper_id", "repo"]).reset_index(drop=True)
    links.to_parquet(OUT_DIR / "readme_links.parquet", index=False)

    log = json.loads(EVIDENCE_LOG.read_text()) if EVIDENCE_LOG.exists() else {}
    log[LOG_KEY] = [
        {
            "query": latest[p]["query"],
            "queried_at": latest[p]["queried_at"],
            "total_count": latest[p]["total_count"],
            "returned": len(latest[p]["items"]),
            "incomplete_results": latest[p]["incomplete_results"],
        }
        for p in paper_ids
    ]
    EVIDENCE_LOG.write_text(json.dumps(log, indent=1))

    verified = links[links["id_found"] == True]
    totals = [latest[p]["total_count"] for p in paper_ids]
    meta = {
        "step": "evidence_readme_search",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "pwc_snapshot_latest_paper": snapshot.isoformat(),
        "query_template": '"<arXiv id>" in:readme created:<=<data_cutoff date>',
        "first_page_only": PER_PAGE,
        "queried_from": min(latest[p]["queried_at"] for p in paper_ids) if paper_ids else None,
        "queried_to": max(latest[p]["queried_at"] for p in paper_ids) if paper_ids else None,
        "pair_evidence_sha256": sha256(PAIR_EVIDENCE),
        "readme_links_parquet_sha256": sha256(OUT_DIR / "readme_links.parquet"),
    }
    stats = {
        "papers_searched": len(paper_ids),
        "papers_with_hits": int(sum(1 for t in totals if t > 0)),
        "papers_truncated": int(sum(1 for t in totals if t > PER_PAGE)),
        "hit_pairs": int(len(links)),
        "hit_repos": int(links["repo"].nunique()),
        "readme_unavailable": int((links["status"] != 200).sum()),
        "verified_pairs": int(len(verified)),
        "verified_repos": int(verified["repo"].nunique()),
        "verified_papers": int(verified["paper_id"].nunique()),
    }
    result = {"meta": meta, "stats": stats}
    (RESULTS / "evidence_readme_links.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", OUT_DIR / "readme_links.parquet", EVIDENCE_LOG, RESULTS / "evidence_readme_links.json")


if __name__ == "__main__":
    main()
