"""Star counts in the window before data_cutoff for pair-evidence repositories (design 7.2 D).

Usage: python scripts/evidence/06_star_history.py

Repositories (config applied.evidence, research log 2026-10-05):
- PWC links of pair-evidence papers (04_pwc_links.py, pair_evidence == True)
- README-search links whose README contains the exact arXiv id and at most
  readme_max_arxiv_ids distinct arXiv ids (05_readme_search.py)

For each repository, GET /repos/{owner}/{repo}/stargazers/history returns weeks
(newest first) with per-day counts starting on Sunday. Pages are read until the
oldest week reaches the window start. The value is the sum of daily counts for
days d with cutoff - star_window_days < d <= cutoff (UTC dates). Individual
stargazer timestamps are no longer available to non-collaborators (GitHub, July
2026), so this aggregate is used.

Only per-repository values are produced; candidate-level aggregation is done in
the DVF step. Requests are cached in a JSON-lines file and skipped on rerun.

Needs GITHUB_TOKEN in .env (or in the environment).

Outputs
- data/derived/evidence/star_history.jsonl (cache, not committed)
- data/derived/evidence/repo_stars.parquet (not committed)
- results/evidence_log.json: key "github_star_history"
- results/evidence_repo_stars.json: counts only
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study

PWC_LINKS = Path("data/derived/evidence/pwc_links.parquet")
README_LINKS = Path("data/derived/evidence/readme_links.parquet")
OUT_DIR = Path("data/derived/evidence")
CACHE = OUT_DIR / "star_history.jsonl"
RESULTS = Path("results")
EVIDENCE_LOG = RESULTS / "evidence_log.json"
LOG_KEY = "github_star_history"
API = "https://api.github.com"
# The star history endpoint is documented under this REST API version
API_VERSION = "2026-03-10"
PER_PAGE = 30
MAX_PAGES = 100
# Stay under 5,000 core requests per hour
PAUSE = 0.75


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


def request(url: str, token: str) -> tuple[int, bytes]:
    """GET with rate-limit handling; header lookups are case-insensitive."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "triad-rx-evidence",
    })
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                headers = resp.headers
                body = resp.read()
            if headers.get("X-RateLimit-Remaining") == "0":
                wait = max(0, int(headers.get("X-RateLimit-Reset", "0")) - int(time.time())) + 2
                print(f"rate limit reached, sleeping {wait}s")
                time.sleep(wait)
            return 200, body
        except urllib.error.HTTPError as err:
            headers = err.headers
            limited = "Retry-After" in headers or headers.get("X-RateLimit-Remaining") == "0"
            if err.code in (403, 429) and limited:
                if "Retry-After" in headers:
                    wait = int(headers["Retry-After"]) + 1
                else:
                    wait = max(0, int(headers.get("X-RateLimit-Reset", "0")) - int(time.time())) + 2
                print(f"HTTP {err.code}, sleeping {wait}s")
                time.sleep(wait)
                continue
            if err.code >= 500 or err.code == 422:
                time.sleep(10 * (attempt + 1))
                continue
            return err.code, b""
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    return -1, b""


def read_cache() -> dict:
    if not CACHE.exists():
        return {}
    records = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["repo"]] = record
    return records


def week_start(week: int) -> datetime.date:
    return datetime.datetime.fromtimestamp(week, datetime.timezone.utc).date()


def fetch(repo: str, token: str, start: datetime.date, cutoff: datetime.date) -> dict:
    """Weeks overlapping (start, cutoff] and the star sum over those days."""
    kept, pages, failed = [], 0, None
    for page in range(1, MAX_PAGES + 1):
        url = f"{API}/repos/{repo}/stargazers/history?per_page={PER_PAGE}&page={page}"
        status, body = request(url, token)
        time.sleep(PAUSE)
        if status != 200:
            failed = status
            break
        pages += 1
        weeks = json.loads(body)
        for w in weeks:
            first = week_start(w["week"])
            if first + datetime.timedelta(days=6) > start and first <= cutoff:
                kept.append({"week": w["week"], "days": w["days"]})
        if not weeks or week_start(weeks[-1]["week"]) <= start or len(weeks) < PER_PAGE:
            break
    stars = 0
    for w in kept:
        first = week_start(w["week"])
        # days[0] is the Sunday that starts the week
        for i, n in enumerate(w["days"]):
            day = first + datetime.timedelta(days=i)
            if start < day <= cutoff:
                stars += n
    return {
        "repo": repo,
        "fetched_at": now_utc(),
        "status": failed if failed is not None else 200,
        "pages": pages,
        "stars_window": None if failed is not None else stars,
        "weeks": kept,
    }


def main() -> None:
    study = load_study()
    cfg = study["applied"]["evidence"]
    if cfg["repo_unit"] != "pair_evidence_papers" or cfg["star_aggregate"] != "sum":
        sys.exit("only repo_unit 'pair_evidence_papers' with star_aggregate 'sum' is implemented")
    cutoff_dt = data_cutoff()
    cutoff = cutoff_dt.astimezone(datetime.timezone.utc).date()
    start = cutoff - datetime.timedelta(days=cfg["star_window_days"])
    token = github_token()

    pwc = pd.read_parquet(PWC_LINKS, columns=["repo", "pair_evidence"])
    pwc_repos = set(pwc.loc[pwc["pair_evidence"], "repo"])
    readme = pd.read_parquet(README_LINKS, columns=["repo", "id_found", "readme_arxiv_ids"])
    readme_ok = readme[(readme["id_found"] == True) & (readme["readme_arxiv_ids"] <= cfg["readme_max_arxiv_ids"])]
    readme_repos = set(readme_ok["repo"])
    readme_list_repos = set(readme.loc[(readme["id_found"] == True) & (readme["readme_arxiv_ids"] > cfg["readme_max_arxiv_ids"]), "repo"])
    repos = sorted(pwc_repos | readme_repos)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = read_cache()
    # 404 and 451 mean the repository is gone or blocked; retrying will not help
    final = {200, 404, 451}
    todo = [r for r in repos if r not in cache or cache[r]["status"] not in final]
    print(f"repositories: {len(repos)}, cached: {len(repos) - len(todo)}, to fetch: {len(todo)}")
    for i, repo in enumerate(todo, 1):
        record = fetch(repo, token, start, cutoff)
        with CACHE.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if i % 100 == 0:
            print(f"fetched {i}/{len(todo)}")

    cache = read_cache()
    rows = [
        {
            "evidence_id": "repo:" + r,
            "repo": r,
            "source_pwc": r in pwc_repos,
            "source_readme": r in readme_repos,
            "status": cache[r]["status"],
            "stars_window": cache[r]["stars_window"],
            "fetched_at": cache[r]["fetched_at"],
        }
        for r in repos
    ]
    stars = pd.DataFrame(rows)
    stars.to_parquet(OUT_DIR / "repo_stars.parquet", index=False)

    log = json.loads(EVIDENCE_LOG.read_text()) if EVIDENCE_LOG.exists() else {}
    log[LOG_KEY] = [
        {
            "request": f"GET /repos/{r}/stargazers/history",
            "fetched_at": cache[r]["fetched_at"],
            "status": cache[r]["status"],
            "pages": cache[r]["pages"],
        }
        for r in repos
    ]
    EVIDENCE_LOG.write_text(json.dumps(log, indent=1))

    ok = stars[stars["status"] == 200]
    meta = {
        "step": "evidence_repo_stars",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff_dt.isoformat(),
        "window_days": [start.isoformat(), cutoff.isoformat()],
        "window_rule": "start < day <= cutoff, UTC dates; GitHub day boundaries may not align with UTC",
        "api_version": API_VERSION,
        "settings": cfg,
        "fetched_from": stars["fetched_at"].min() if len(stars) else None,
        "fetched_to": stars["fetched_at"].max() if len(stars) else None,
        "pwc_links_sha256": sha256(PWC_LINKS),
        "readme_links_sha256": sha256(README_LINKS),
        "repo_stars_parquet_sha256": sha256(OUT_DIR / "repo_stars.parquet"),
    }
    result_stats = {
        "repositories": len(repos),
        "from_pwc": len(pwc_repos),
        "from_readme": len(readme_repos),
        "in_both": len(pwc_repos & readme_repos),
        "readme_list_repos_excluded": len(readme_list_repos - pwc_repos),
        "fetched_ok": int(len(ok)),
        "unavailable": int(len(stars) - len(ok)),
        "unavailable_status": {str(k): int(v) for k, v in stars.loc[stars["status"] != 200, "status"].value_counts().items()},
        "zero_stars_in_window": int((ok["stars_window"] == 0).sum()),
    }
    result = {"meta": meta, "stats": result_stats}
    (RESULTS / "evidence_repo_stars.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result_stats, indent=1))
    print("Saved", OUT_DIR / "repo_stars.parquet", EVIDENCE_LOG, RESULTS / "evidence_repo_stars.json")


if __name__ == "__main__":
    main()
