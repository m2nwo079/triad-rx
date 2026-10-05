"""Patent evidence for the DVF population (design 4.5, 6.1, 7.5).

Usage: python scripts/evidence/03_patents.py

Reads the nine PatentsView tables recorded by 02_patent_files.py and keeps a
document when all of the following hold (research log, 2026-10-04):
- utility document; granted patents with withdrawn == 1 are dropped
- its own at-issue CPC contains subclass G06N
- its date (grant date or publication date, design 4.4) is inside the applied
  build window; a date is taken as 00:00 UTC of that day
- at least one final-vocabulary term matches its title or abstract, using the
  same final terms and matcher as the applied hypergraph

Documents of the same invention (pre-grant publication and granted patent) are
grouped by application_id; granted patents without a crosswalk entry use their
own id. Which documents count as "related" to a candidate (pair or term level)
is decided in the DVF step, so this step stores matched terms per document and
writes no candidate-level counts.

Outputs
- data/derived/evidence/patents.parquet (not committed): one row per document
- data/derived/evidence/patent_assignees.parquet (not committed): assignees per document
- results/evidence_patents.json: coverage dates and document counts per step
"""
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import build_window, data_cutoff, load_study
from triadrx.text import Matcher

RAW = Path("data/patents/raw")
INVENTORY = Path("results/patent_files.json")
VOCAB = Path("vocab/vocab_base.json")
HYPERGRAPH_SCRIPT = Path("scripts/graph/01_build_hypergraph.py")
OUT_DIR = Path("data/derived/evidence")
RESULTS = Path("results")
CHUNK = 200_000
CPC_SUBCLASS = "G06N"

# Per document kind: id column, tables and the columns read from each
KINDS = {
    "grant": {
        "id": "patent_id",
        "doc": ("g_patent", ["patent_id", "patent_type", "patent_date", "patent_title", "withdrawn"]),
        "date": "patent_date",
        "title": "patent_title",
        "abstract": ("g_patent_abstract", "patent_abstract"),
        "cpc": "g_cpc_at_issue",
        "assignee": "g_assignee_disambiguated",
    },
    "pgpub": {
        "id": "pgpub_id",
        "doc": ("pg_published_application", ["pgpub_id", "application_id", "patent_type", "published_date", "application_title"]),
        "date": "published_date",
        "title": "application_title",
        "abstract": ("pg_published_application_abstract", "application_abstract"),
        "cpc": "pg_cpc_at_issue",
        "assignee": "pg_assignee_disambiguated",
    },
}
ASSIGNEE_COLUMNS = [
    "assignee_sequence",
    "assignee_id",
    "disambig_assignee_individual_name_first",
    "disambig_assignee_individual_name_last",
    "disambig_assignee_organization",
    "assignee_type",
]


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


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table_path(inventory: dict, table: str) -> Path:
    return RAW / inventory[table]["file"]


def read_chunks(path: Path, usecols: list[str]):
    """All values as strings; empty fields stay empty strings."""
    return pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK
    )


def as_day_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce").dt.tz_localize("UTC")


def check_inventory() -> dict:
    """Every table must be the file recorded by 02_patent_files.py."""
    recorded = json.loads(INVENTORY.read_text())
    inventory = {item["table"]: item for item in recorded["files"]}
    for table, item in inventory.items():
        path = RAW / item["file"]
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            sys.exit(f"{path} differs from {INVENTORY}")
    return inventory


def g06n_ids(path: Path, id_col: str) -> set[str]:
    ids = set()
    for chunk in read_chunks(path, [id_col, "cpc_subclass"]):
        ids.update(chunk.loc[chunk["cpc_subclass"] == CPC_SUBCLASS, id_col])
    return ids


def select_documents(kind: str, spec: dict, inventory: dict, cpc_ids: set[str], start, end, steps: dict) -> pd.DataFrame:
    """Utility, not withdrawn, G06N and in-window documents with title."""
    table, cols = spec["doc"]
    id_col, date_col = spec["id"], spec["date"]
    counts = {"rows": 0, "utility": 0, "not_withdrawn": 0, "g06n": 0, "in_window": 0}
    max_date = None
    kept = []
    for chunk in read_chunks(table_path(inventory, table), cols):
        counts["rows"] += len(chunk)
        chunk = chunk[chunk["patent_type"] == "utility"]
        counts["utility"] += len(chunk)
        if "withdrawn" in chunk:
            chunk = chunk[chunk["withdrawn"] != "1"]
        counts["not_withdrawn"] += len(chunk)
        dates = as_day_utc(chunk[date_col])
        chunk_max = dates.max()
        if pd.notna(chunk_max) and (max_date is None or chunk_max > max_date):
            max_date = chunk_max
        chunk = chunk.assign(date=dates)[chunk[id_col].isin(cpc_ids)]
        counts["g06n"] += len(chunk)
        chunk = chunk[(chunk["date"] >= start) & (chunk["date"] <= end)]
        counts["in_window"] += len(chunk)
        kept.append(chunk)
    docs = pd.concat(kept, ignore_index=True)
    if docs[id_col].duplicated().any():
        sys.exit(f"duplicate {id_col} in {table}")
    out = pd.DataFrame({
        "doc_kind": kind,
        "doc_id": docs[id_col],
        "application_id": docs["application_id"] if "application_id" in docs else "",
        "date": docs["date"],
        "title": docs[spec["title"]],
    })
    steps[kind] = counts
    steps[kind]["max_date_all_utility"] = max_date.date().isoformat() if max_date is not None else None
    return out


def attach_abstracts(docs: pd.DataFrame, spec: dict, inventory: dict) -> pd.DataFrame:
    table, col = spec["abstract"]
    wanted = set(docs["doc_id"])
    parts = []
    for chunk in read_chunks(table_path(inventory, table), [spec["id"], col]):
        parts.append(chunk[chunk[spec["id"]].isin(wanted)])
    abstracts = pd.concat(parts, ignore_index=True).drop_duplicates(spec["id"])
    abstracts = abstracts.rename(columns={spec["id"]: "doc_id", col: "abstract"})
    return docs.merge(abstracts, on="doc_id", how="left", validate="one_to_one").fillna({"abstract": ""})


def main() -> None:
    study = load_study()
    if study["applied"]["evidence"]["patent_window"] != "build_window":
        sys.exit("only applied.evidence.patent_window == 'build_window' is implemented")
    cutoff = data_cutoff()
    start, end = build_window("applied", study, cutoff)
    inventory = check_inventory()

    # Same final terms as the applied hypergraph (no change to that frozen script)
    graph = load("build_hypergraph", HYPERGRAPH_SCRIPT)
    vocab = {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]}
    terms = graph.final_terms("applied", vocab)
    matcher = Matcher(terms)
    term_type = {t["id"]: t["type"] for t in terms}

    steps = {}
    frames = []
    for kind, spec in KINDS.items():
        cpc_ids = g06n_ids(table_path(inventory, spec["cpc"]), spec["id"])
        docs = select_documents(kind, spec, inventory, cpc_ids, start, end, steps)
        docs = attach_abstracts(docs, spec, inventory)
        steps[kind]["with_abstract"] = int((docs["abstract"] != "").sum())
        matched = [
            sorted({s["id"] for s in matcher.spans(f"{title}. {abstract}")})
            for title, abstract in docs[["title", "abstract"]].itertuples(index=False)
        ]
        docs["terms"] = matched
        docs = docs[docs["terms"].map(len) > 0]
        steps[kind]["matched"] = int(len(docs))
        frames.append(docs)
    docs = pd.concat(frames, ignore_index=True)
    docs.insert(0, "evidence_id", "patent:" + docs["doc_id"])

    # Invention key: application_id; granted patents get it from the crosswalk
    grant_ids = set(docs.loc[docs["doc_kind"] == "grant", "doc_id"])
    pgpub_apps = set(docs.loc[docs["doc_kind"] == "pgpub", "application_id"]) - {""}
    parts = []
    for chunk in read_chunks(table_path(inventory, "pg_granted_pgpubs_crosswalk"), ["application_id", "patent_id"]):
        keep = chunk["patent_id"].isin(grant_ids) | chunk["application_id"].isin(pgpub_apps)
        parts.append(chunk[keep])
    cross = pd.concat(parts, ignore_index=True)
    # Share of kept publications whose application_id appears in the crosswalk (format check)
    pgpub_apps_in_crosswalk = len(pgpub_apps & set(cross["application_id"]))
    cross = cross[(cross["patent_id"] != "") & (cross["application_id"] != "")].drop_duplicates()
    cross = cross[cross["patent_id"].isin(grant_ids)]
    grant_apps = cross.groupby("patent_id")["application_id"].agg(lambda s: sorted(set(s)))
    multi = {pid for pid, apps in grant_apps.items() if len(apps) > 1}
    grant_app = grant_apps.map(lambda apps: apps[0])
    is_grant = docs["doc_kind"] == "grant"
    docs.loc[is_grant, "application_id"] = docs.loc[is_grant, "doc_id"].map(grant_app).fillna("")
    has_app = docs["application_id"] != ""
    docs["invention_id"] = ("app:" + docs["application_id"]).where(has_app, "grant:" + docs["doc_id"])
    linked = docs.groupby("invention_id")["doc_kind"].nunique()

    # Assignees of the kept documents
    assignee_parts = []
    for kind, spec in KINDS.items():
        wanted = set(docs.loc[docs["doc_kind"] == kind, "doc_id"])
        for chunk in read_chunks(table_path(inventory, spec["assignee"]), [spec["id"]] + ASSIGNEE_COLUMNS):
            chunk = chunk[chunk[spec["id"]].isin(wanted)].rename(columns={spec["id"]: "doc_id"})
            assignee_parts.append(chunk.assign(doc_kind=kind))
    assignees = pd.concat(assignee_parts, ignore_index=True)
    assignees = assignees.merge(
        docs[["doc_kind", "doc_id", "evidence_id", "invention_id"]], on=["doc_kind", "doc_id"], how="left", validate="many_to_one"
    )

    if (docs["date"] > end).any() or (docs["date"] < start).any():
        sys.exit("patent document outside the build window")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = docs.sort_values(["date", "evidence_id"]).reset_index(drop=True)
    docs.to_parquet(OUT_DIR / "patents.parquet", index=False)
    assignees.to_parquet(OUT_DIR / "patent_assignees.parquet", index=False)

    matched_terms = {t for row in docs["terms"] for t in row}
    meta = {
        "step": "evidence_patents",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "window_rule": study["applied"]["evidence"]["patent_window"],
        "cpc_subclass": CPC_SUBCLASS,
        "release_date": json.loads(INVENTORY.read_text())["meta"]["release_date"],
        "inventory_sha256": sha256(INVENTORY),
        "vocab_sha256": sha256(VOCAB),
        "final_terms": len(terms),
        "ambiguous_surfaces": {mode: len(table) for mode, table in matcher.ambiguous.items()},
        "patents_parquet_sha256": sha256(OUT_DIR / "patents.parquet"),
        "assignees_parquet_sha256": sha256(OUT_DIR / "patent_assignees.parquet"),
    }
    stats = {
        "steps": steps,
        "documents": int(len(docs)),
        "inventions": int(docs["invention_id"].nunique()),
        "inventions_with_grant_and_pgpub": int((linked > 1).sum()),
        "grants_without_crosswalk": int((is_grant & ~has_app).sum()),
        "pgpub_applications": len(pgpub_apps),
        "pgpub_applications_in_crosswalk": pgpub_apps_in_crosswalk,
        "grants_with_several_applications": len(multi),
        "documents_without_assignee": int(len(set(docs["evidence_id"]) - set(assignees["evidence_id"]))),
        "matched_terms": len(matched_terms),
        "matched_terms_by_type": {k: sum(1 for t in matched_terms if term_type[t] == k) for k in ("M", "T")},
    }
    result = {"meta": meta, "stats": stats}
    (RESULTS / "evidence_patents.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", OUT_DIR / "patents.parquet", OUT_DIR / "patent_assignees.parquet", RESULTS / "evidence_patents.json")


if __name__ == "__main__":
    main()
