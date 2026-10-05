"""PyPI packages linked to pair-evidence repositories (design 7.2 D, package indicator).

Usage:
  python scripts/evidence/07_pypi_links.py          # dry run only: prints the estimated bytes
  python scripts/evidence/07_pypi_links.py --run    # runs the query (after checking the estimate)

A package is linked to a repository when any of its distributions uploaded on or
before data_cutoff has home_page, download_url or a project_urls entry pointing
to github.com/<owner>/<name> of a repository in data/derived/evidence/repo_stars.parquet
(the same repository set as the star indicator). Package names are normalized as
in PEP 503 (lower case, runs of - _ . replaced by -), which is the form used by
the download table.

Only the four needed columns are read; the table is partitioned by upload month,
so the filter on upload_time also limits the scan.

Needs GCP_PROJECT in .env and application-default credentials (gcloud).

Outputs
- data/derived/evidence/pypi_links.parquet (not committed)
- results/evidence_log.json: key "bigquery_pypi_metadata"
- results/evidence_pypi_links.json: estimate, processed bytes and counts
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study

REPO_STARS = Path("data/derived/evidence/repo_stars.parquet")
OUT_DIR = Path("data/derived/evidence")
RESULTS = Path("results")
EVIDENCE_LOG = RESULTS / "evidence_log.json"
LOG_KEY = "bigquery_pypi_metadata"
TABLE = "bigquery-public-data.pypi.distribution_metadata"
# Safety cap for this query; the sandbox allows 1 TB of queries per month
MAX_BYTES = 50 * 10**9

QUERY = f"""
WITH urls AS (
  SELECT
    REGEXP_REPLACE(LOWER(name), r'[-_.]+', '-') AS package,
    upload_time,
    LOWER(url) AS url
  FROM `{TABLE}`,
  UNNEST(ARRAY_CONCAT(project_urls, [IFNULL(home_page, ''), IFNULL(download_url, '')])) AS url
  WHERE upload_time <= @cutoff
),
repos AS (
  SELECT
    package,
    upload_time,
    REGEXP_REPLACE(REGEXP_EXTRACT(url, r'github\\.com/([a-z0-9_.-]+/[a-z0-9_.-]+)'), r'\\.git$', '') AS repo
  FROM urls
)
SELECT
  package,
  repo,
  MIN(upload_time) AS first_upload,
  MAX(upload_time) AS last_upload_before_cutoff
FROM repos
WHERE repo IN UNNEST(@repos)
GROUP BY package, repo
ORDER BY package, repo
"""


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


def gcp_project() -> str:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "GCP_PROJECT":
                return value.strip().strip('"').strip("'")
    sys.exit("GCP_PROJECT not found in .env")


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the query after the dry run")
    args = parser.parse_args()

    study = load_study()
    if study["applied"]["evidence"]["package_link"] != "pypi_metadata_urls":
        sys.exit("only applied.evidence.package_link == 'pypi_metadata_urls' is implemented")
    cutoff = data_cutoff()

    stars = pd.read_parquet(REPO_STARS, columns=["repo", "status"])
    repos = sorted(stars.loc[stars["status"] == 200, "repo"])
    params = [
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
        bigquery.ArrayQueryParameter("repos", "STRING", repos),
    ]
    client = bigquery.Client(project=gcp_project())

    dry = client.query(QUERY, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False,
    ))
    estimate = dry.total_bytes_processed
    print(f"repositories: {len(repos)}")
    print(f"dry run estimate: {estimate / 1e9:.2f} GB (cap {MAX_BYTES / 1e9:.0f} GB)")
    if not args.run:
        print("dry run only; rerun with --run to execute")
        return
    if estimate > MAX_BYTES:
        sys.exit("estimate exceeds the cap; not running")

    started = now_utc()
    job = client.query(QUERY, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=int(estimate * 1.1) + 10**9,
    ))
    # Build the frame from rows; to_dataframe() would need the extra db-dtypes package
    columns = ["package", "repo", "first_upload", "last_upload_before_cutoff"]
    links = pd.DataFrame([dict(row) for row in job.result()], columns=columns)
    finished = now_utc()

    links.insert(0, "evidence_id", "pkg:" + links["package"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    links.to_parquet(OUT_DIR / "pypi_links.parquet", index=False)

    log = json.loads(EVIDENCE_LOG.read_text()) if EVIDENCE_LOG.exists() else {}
    log[LOG_KEY] = [{
        "query": QUERY.strip(),
        "parameters": {"cutoff": cutoff.isoformat(), "repos": f"{len(repos)} repositories from {REPO_STARS}"},
        "job_id": job.job_id,
        "started_at": started,
        "finished_at": finished,
        "dry_run_bytes": estimate,
        "bytes_processed": job.total_bytes_processed,
        "bytes_billed": job.total_bytes_billed,
        "cache_hit": job.cache_hit,
        "result_rows": int(len(links)),
    }]
    EVIDENCE_LOG.write_text(json.dumps(log, indent=1))

    meta = {
        "step": "evidence_pypi_links",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "table": TABLE,
        "rule": "home_page, download_url or project_urls points to a repository; distributions uploaded on or before data_cutoff; PEP 503 names",
        "dry_run_bytes": estimate,
        "bytes_processed": job.total_bytes_processed,
        "cache_hit": job.cache_hit,
        "repo_stars_sha256": sha256(REPO_STARS),
        "pypi_links_parquet_sha256": sha256(OUT_DIR / "pypi_links.parquet"),
    }
    stats = {
        "repositories_searched": len(repos),
        "links": int(len(links)),
        "packages": int(links["package"].nunique()),
        "repositories_with_package": int(links["repo"].nunique()),
        "packages_with_several_repositories": int((links.groupby("package")["repo"].nunique() > 1).sum()),
    }
    (RESULTS / "evidence_pypi_links.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", OUT_DIR / "pypi_links.parquet", EVIDENCE_LOG, RESULTS / "evidence_pypi_links.json")


if __name__ == "__main__":
    main()
