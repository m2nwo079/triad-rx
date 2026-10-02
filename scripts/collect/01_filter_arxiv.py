"""Filter the Kaggle arXiv metadata snapshot to the target categories."""
import collections
import datetime
import email.utils
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW_ZIP = Path("data/arxiv/raw/arxiv.zip")
OUT_PARQUET = Path("data/arxiv/papers.parquet")
STUDY = Path("config/study.json")
SCOPE_RESULTS = Path("results/scope_survey_2026-10-01.json")
RESULTS = Path("results")
# Rows buffered before each parquet write, to keep memory flat
CHUNK = 50_000
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("categories", pa.string()),
        ("first_date", pa.timestamp("s", tz="UTC")),
    ]
)


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


def first_version_date(record: dict) -> datetime.datetime | None:
    # Submission time of v1; public announcement is usually the next business day
    for version in record.get("versions") or []:
        if version.get("version") == "v1" and version.get("created"):
            try:
                return email.utils.parsedate_to_datetime(version["created"]).astimezone(
                    datetime.timezone.utc
                )
            except (TypeError, ValueError):
                return None
    return None


def main() -> None:
    study = json.loads(STUDY.read_text())
    targets = set(study["arxiv_categories"])

    with zipfile.ZipFile(RAW_ZIP) as archive:
        members = [m for m in archive.infolist() if m.filename.endswith(".json")]
        if len(members) != 1:
            raise SystemExit(f"Expected one JSON file in the archive, found {len(members)}")
        member = members[0]
        snapshot_date = datetime.date(*member.date_time[:3]).isoformat()

        counts = collections.Counter()
        skipped = collections.Counter()
        by_year = collections.Counter()
        # Latest v1 submission time kept; defines the data cutoff (design 4.4)
        data_cutoff = None
        buffer = []
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(OUT_PARQUET, SCHEMA)
        with archive.open(member) as handle:
            for line_no, raw in enumerate(handle, start=1):
                counts["lines"] += 1
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    skipped["invalid_json"] += 1
                    continue
                cats = (record.get("categories") or "").split()
                if not targets.intersection(cats):
                    continue
                counts["matched"] += 1
                date = first_version_date(record)
                if date is None:
                    skipped["no_v1_date"] += 1
                    continue
                if not record.get("title") or not record.get("abstract"):
                    skipped["missing_text"] += 1
                    continue
                buffer.append(
                    {
                        "id": record["id"],
                        "title": record["title"],
                        "abstract": record["abstract"],
                        "categories": record["categories"],
                        "first_date": date,
                    }
                )
                by_year[date.year] += 1
                if data_cutoff is None or date > data_cutoff:
                    data_cutoff = date
                if len(buffer) >= CHUNK:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                    buffer.clear()
                if line_no % 500_000 == 0:
                    print(f"{line_no:,} lines read, {counts['matched']:,} matched", flush=True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        writer.close()

    counts["kept"] = sum(by_year.values())
    by_year_sorted = {str(y): by_year[y] for y in sorted(by_year)}
    comparison = {}
    if SCOPE_RESULTS.exists():
        api = json.loads(SCOPE_RESULTS.read_text())["arxiv_counts_by_year"]
        for year, api_count in api.items():
            snap = by_year_sorted.get(year)
            if snap is not None and api_count:
                comparison[year] = {
                    "snapshot": snap,
                    "api": api_count,
                    "ratio": round(snap / api_count, 3),
                }

    result = {
        "meta": {
            "step": "collect_arxiv_filter",
            "snapshot_date": snapshot_date,
            "data_cutoff": data_cutoff.isoformat() if data_cutoff else None,
            "git_commit": git_commit(),
            "source_zip": RAW_ZIP.as_posix(),
            "source_zip_sha256": sha256(RAW_ZIP),
            "source_member": member.filename,
            "source_member_bytes": member.file_size,
            "categories": sorted(targets),
            "date_definition": "v1 submission time (UTC) from versions; announcement is usually the next business day",
        },
        "counts": dict(counts),
        "skipped": dict(skipped),
        "kept_by_first_year": by_year_sorted,
        "comparison_with_api_counts": comparison,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"arxiv_filter_{snapshot_date}.json"
    out.write_text(json.dumps(result, indent=2))
    print("Saved", out, "and", OUT_PARQUET)


if __name__ == "__main__":
    main()