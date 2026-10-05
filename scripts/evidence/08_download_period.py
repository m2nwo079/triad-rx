"""D-axis download period judgment by dry run (design 7.4, preregistration 6).

Usage: python scripts/evidence/08_download_period.py

Estimates, without running it, the bytes of the 12-month download query for all
packages linked in 07_pypi_links.py, and applies the preregistered rule:
- estimate <= half of the BigQuery sandbox monthly free query volume -> 12 months (BigQuery)
- otherwise -> 6 months (pypistats)

The free volume is 1 TiB per month (BigQuery free tier). The query has the same
shape as the one that would be used for collection: daily-partitioned timestamp
filter over the window before data_cutoff and the package list. A dry-run
estimate does not reflect clustering, so it is an upper bound; the rule is
applied to the estimate as written.

Nothing is billed; dry runs are free.

Output: results/download_period.json (verdict, estimate, threshold, query)
"""
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

PYPI_LINKS = Path("data/derived/evidence/pypi_links.parquet")
RESULTS = Path("results")
TABLE = "bigquery-public-data.pypi.file_downloads"
FREE_BYTES_PER_MONTH = 2**40
THRESHOLD = FREE_BYTES_PER_MONTH // 2
WINDOW_DAYS = 365

QUERY = f"""
SELECT
  project,
  COUNTIF(timestamp <= @mid) AS downloads_first_half,
  COUNTIF(timestamp > @mid) AS downloads_second_half
FROM `{TABLE}`
WHERE timestamp > @start
  AND timestamp <= @cutoff
  AND project IN UNNEST(@packages)
GROUP BY project
ORDER BY project
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


def main() -> None:
    study = load_study()
    if study["applied"]["evidence"]["download_growth"] != "half_over_half":
        sys.exit("only applied.evidence.download_growth == 'half_over_half' is implemented")
    cutoff = data_cutoff()
    start = cutoff - datetime.timedelta(days=WINDOW_DAYS)
    mid = cutoff - datetime.timedelta(days=WINDOW_DAYS / 2)

    packages = sorted(pd.read_parquet(PYPI_LINKS, columns=["package"])["package"].unique())
    params = [
        bigquery.ScalarQueryParameter("start", "TIMESTAMP", start),
        bigquery.ScalarQueryParameter("mid", "TIMESTAMP", mid),
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
        bigquery.ArrayQueryParameter("packages", "STRING", packages),
    ]
    client = bigquery.Client(project=gcp_project())
    checked_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    dry = client.query(QUERY, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False,
    ))
    estimate = dry.total_bytes_processed
    verdict = "12_months_bigquery" if estimate <= THRESHOLD else "6_months_pypistats"

    meta = {
        "step": "download_period",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "rule": "preregistration 6: 12 months if the dry-run estimate is at most half of the sandbox monthly free query volume, else 6 months (pypistats)",
        "free_bytes_per_month": FREE_BYTES_PER_MONTH,
        "threshold_bytes": THRESHOLD,
        "checked_at": checked_at,
        "query": QUERY.strip(),
        "parameters": {
            "start": start.isoformat(),
            "mid": mid.isoformat(),
            "cutoff": cutoff.isoformat(),
            "packages": f"{len(packages)} packages from {PYPI_LINKS}",
        },
        "pypi_links_sha256": sha256(PYPI_LINKS),
    }
    result = {
        "meta": meta,
        "estimate_bytes": estimate,
        "estimate_tib": round(estimate / 2**40, 3),
        "threshold_tib": THRESHOLD / 2**40,
        "verdict": verdict,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "download_period.json").write_text(json.dumps(result, indent=2))
    print(f"packages: {len(packages)}")
    print(f"dry run estimate: {estimate / 2**40:.3f} TiB, threshold {THRESHOLD / 2**40:.3f} TiB")
    print(f"verdict: {verdict}")
    print("Saved", RESULTS / "download_period.json")


if __name__ == "__main__":
    main()
