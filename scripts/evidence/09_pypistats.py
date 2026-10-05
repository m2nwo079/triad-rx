"""Package downloads from pypistats for the D-axis growth indicator (design 7.2, 7.4).

Usage: python scripts/evidence/09_pypistats.py

Runs only when results/download_period.json says 6_months_pypistats. For each
package linked in 07_pypi_links.py, GET /api/packages/<name>/overall?mirrors=false
and keep the without_mirrors daily series.

Window (research log 2026-10-05): two halves of K weeks each ending on the UTC
date of data_cutoff (inclusive), K = applied.evidence.download_half_weeks. pypistats
keeps about 180 days, so if the earliest date served is later than the window
start, K is reduced to the largest number of whole weeks that fits both halves.
Days missing from a package's series count as zero downloads.

Raw responses are saved because the data ages out of pypistats; only
per-package half sums are produced here (candidate aggregation is in the DVF step).

Outputs
- data/derived/evidence/pypistats_raw/<package>.json (raw responses, not committed; back them up)
- data/derived/evidence/pypi_downloads.parquet (not committed)
- results/evidence_log.json: key "pypistats_overall"
- results/evidence_pypi_downloads.json: window, counts and per-package half sums
"""
import datetime
import hashlib
import json
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
from triadrx.study import data_cutoff, load_study

PYPI_LINKS = Path("data/derived/evidence/pypi_links.parquet")
PERIOD = Path("results/download_period.json")
OUT_DIR = Path("data/derived/evidence")
RAW_DIR = OUT_DIR / "pypistats_raw"
RESULTS = Path("results")
EVIDENCE_LOG = RESULTS / "evidence_log.json"
LOG_KEY = "pypistats_overall"
API = "https://pypistats.org/api/packages"
# About 30 requests per minute are allowed per IP
PAUSE = 2.1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def request(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "triad-rx-evidence", "Accept": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return 200, resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 429:
                wait = int(err.headers.get("Retry-After", "60")) + 1
                print(f"HTTP 429, sleeping {wait}s")
                time.sleep(wait)
                continue
            if err.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            return err.code, b""
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    return -1, b""


def fetch_all(packages: list[str]) -> dict:
    """Raw response per package, reusing files already saved."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    todo = []
    for pkg in packages:
        path = RAW_DIR / f"{pkg}.json"
        if path.exists():
            records[pkg] = json.loads(path.read_text())
        else:
            todo.append(pkg)
    print(f"packages: {len(packages)}, saved: {len(records)}, to fetch: {len(todo)}")
    for i, pkg in enumerate(todo, 1):
        url = f"{API}/{urllib.parse.quote(pkg)}/overall?mirrors=false"
        fetched_at = now_utc()
        status, body = request(url)
        record = {"package": pkg, "url": url, "fetched_at": fetched_at, "status": status}
        if status == 200:
            record["body_sha256"] = sha256_bytes(body)
            record["response"] = json.loads(body)
        # 404 is final (unknown package); other failures are not saved so a rerun retries them
        if status in (200, 404):
            (RAW_DIR / f"{pkg}.json").write_text(json.dumps(record))
        records[pkg] = record
        time.sleep(PAUSE)
        if i % 50 == 0:
            print(f"fetched {i}/{len(todo)}")
    return records


def daily_series(record: dict) -> dict[datetime.date, int]:
    rows = record.get("response", {}).get("data", [])
    series = {}
    for row in rows:
        if row.get("category") != "without_mirrors":
            continue
        day = datetime.date.fromisoformat(row["date"])
        series[day] = series.get(day, 0) + int(row["downloads"])
    return series


def main() -> None:
    study = load_study()
    cfg = study["applied"]["evidence"]
    verdict = json.loads(PERIOD.read_text())["verdict"]
    if verdict != "6_months_pypistats":
        sys.exit(f"download period verdict is {verdict}; this script is for 6_months_pypistats")
    cutoff_dt = data_cutoff()
    cutoff = cutoff_dt.astimezone(datetime.timezone.utc).date()

    packages = sorted(pd.read_parquet(PYPI_LINKS, columns=["package"])["package"].unique())
    records = fetch_all(packages)
    failed = sorted(p for p, r in records.items() if r["status"] not in (200, 404))
    if failed:
        sys.exit(f"{len(failed)} packages failed; rerun to retry them: {failed[:10]}")

    series = {p: daily_series(r) for p, r in records.items() if r["status"] == 200}
    served = [d for s in series.values() for d in s]
    earliest = min(served)
    latest = max(served)
    if latest < cutoff:
        sys.exit(f"pypistats data ends on {latest}, before the cutoff date {cutoff}")

    weeks = cfg["download_half_weeks"]
    while weeks > 0 and cutoff - datetime.timedelta(days=14 * weeks - 1) < earliest:
        weeks -= 1
    if weeks == 0:
        sys.exit("pypistats does not cover a single pair of weeks before the cutoff date")
    second_start = cutoff - datetime.timedelta(days=7 * weeks - 1)
    first_start = cutoff - datetime.timedelta(days=14 * weeks - 1)
    first_end = second_start - datetime.timedelta(days=1)

    rows = []
    for pkg in packages:
        record = records[pkg]
        if record["status"] != 200:
            rows.append({"evidence_id": "pkg:" + pkg, "package": pkg, "status": record["status"],
                         "downloads_first_half": None, "downloads_second_half": None,
                         "days_with_data": 0, "fetched_at": record["fetched_at"]})
            continue
        s = series[pkg]
        first = sum(n for d, n in s.items() if first_start <= d <= first_end)
        second = sum(n for d, n in s.items() if second_start <= d <= cutoff)
        days = sum(1 for d in s if first_start <= d <= cutoff)
        rows.append({"evidence_id": "pkg:" + pkg, "package": pkg, "status": 200,
                     "downloads_first_half": first, "downloads_second_half": second,
                     "days_with_data": days, "fetched_at": record["fetched_at"]})
    downloads = pd.DataFrame(rows)
    downloads.to_parquet(OUT_DIR / "pypi_downloads.parquet", index=False)

    log = json.loads(EVIDENCE_LOG.read_text()) if EVIDENCE_LOG.exists() else {}
    log[LOG_KEY] = [
        {"request": records[p]["url"], "fetched_at": records[p]["fetched_at"], "status": records[p]["status"],
         "body_sha256": records[p].get("body_sha256")}
        for p in packages
    ]
    EVIDENCE_LOG.write_text(json.dumps(log, indent=1))

    ok = downloads[downloads["status"] == 200]
    meta = {
        "step": "evidence_pypi_downloads",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff_dt.isoformat(),
        "source": "pypistats.org /api/packages/<name>/overall?mirrors=false, category without_mirrors",
        "verdict": verdict,
        "half_weeks_configured": cfg["download_half_weeks"],
        "half_weeks_used": weeks,
        "first_half": [first_start.isoformat(), first_end.isoformat()],
        "second_half": [second_start.isoformat(), cutoff.isoformat()],
        "earliest_date_served": earliest.isoformat(),
        "latest_date_served": latest.isoformat(),
        "fetched_from": min(r["fetched_at"] for r in records.values()),
        "fetched_to": max(r["fetched_at"] for r in records.values()),
        "pypi_links_sha256": sha256(PYPI_LINKS),
        "pypi_downloads_parquet_sha256": sha256(OUT_DIR / "pypi_downloads.parquet"),
    }
    stats = {
        "packages": len(packages),
        "fetched_ok": int(len(ok)),
        "not_found": int((downloads["status"] == 404).sum()),
        "zero_in_both_halves": int(((ok["downloads_first_half"] == 0) & (ok["downloads_second_half"] == 0)).sum()),
    }
    per_package = {
        r.package: [int(r.downloads_first_half), int(r.downloads_second_half)]
        for r in ok.itertuples(index=False)
    }
    result = {"meta": meta, "stats": stats, "half_sums": per_package}
    (RESULTS / "evidence_pypi_downloads.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps({"half_weeks_used": weeks, **stats}, indent=1))
    print("Saved", OUT_DIR / "pypi_downloads.parquet", EVIDENCE_LOG, RESULTS / "evidence_pypi_downloads.json")


if __name__ == "__main__":
    main()
