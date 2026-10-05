"""Inventory of the manually downloaded PatentsView tables (design 4.5).

Usage: python scripts/evidence/02_patent_files.py --release-date YYYY-MM-DD

The nine tables are downloaded once by hand from the USPTO Open Data Portal
(datasets PVGPATDIS and PVPGPUBDIS) into data/patents/raw/ and kept compressed.
This script checks that each table is present exactly once and records file
name, size and SHA-256 together with the release date shown on the portal, so
later steps can verify they read the same files.

Output: results/patent_files.json
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAW = Path("data/patents/raw")
RESULTS = Path("results")
TABLES = {
    "PVGPATDIS": ["g_patent", "g_patent_abstract", "g_cpc_at_issue", "g_assignee_disambiguated"],
    "PVPGPUBDIS": [
        "pg_published_application",
        "pg_published_application_abstract",
        "pg_cpc_at_issue",
        "pg_assignee_disambiguated",
        "pg_granted_pgpubs_crosswalk",
    ],
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


def table_name(path: Path) -> str:
    """Table name is the file name before the first dot (g_patent.tsv.zip -> g_patent)."""
    return path.name.split(".", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-date", required=True, help="release date shown on the portal, YYYY-MM-DD")
    args = parser.parse_args()
    release = datetime.date.fromisoformat(args.release_date)

    if not RAW.is_dir():
        sys.exit(f"{RAW} does not exist")
    files = [p for p in sorted(RAW.iterdir()) if p.is_file() and not p.name.startswith(".")]
    by_table = {}
    for path in files:
        by_table.setdefault(table_name(path), []).append(path)

    expected = [t for tables in TABLES.values() for t in tables]
    missing = [t for t in expected if t not in by_table]
    duplicated = {t: [p.name for p in by_table[t]] for t in expected if len(by_table.get(t, [])) > 1}
    unexpected = sorted(p.name for t, paths in by_table.items() if t not in expected for p in paths)
    if missing or duplicated:
        sys.exit(f"missing tables: {missing}; tables with more than one file: {duplicated}")

    inventory = []
    for product, tables in TABLES.items():
        for table in tables:
            path = by_table[table][0]
            inventory.append({
                "product": product,
                "table": table,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    meta = {
        "step": "patent_files",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "source": "USPTO Open Data Portal bulk datasets, downloaded manually",
        "release_date": release.isoformat(),
    }
    result = {"meta": meta, "files": inventory, "unexpected_files": unexpected}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "patent_files.json").write_text(json.dumps(result, indent=2))
    for item in inventory:
        print(f"{item['table']:<36} {item['bytes']:>14,d}  {item['file']}")
    if unexpected:
        print("Not part of the nine tables (ignored):", unexpected)
    print("Saved", RESULTS / "patent_files.json")


if __name__ == "__main__":
    main()
