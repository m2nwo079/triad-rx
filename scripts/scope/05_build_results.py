"""Assemble the scope survey outputs in data/scope into one results JSON."""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

DATA = Path("data/scope")
RESULTS = Path("results")
SOURCE_FILES = [
    "methods.parquet",
    "links.parquet",
    "pwa_0.parquet",
    "pwa_1.parquet",
    "pwa_2.parquet",
    "pwa_3.parquet",
]
# Same spam pattern as the matching script, applied to method names
SPAM = re.compile(
    r"[☎→【】]|\+\d|\d{3}\D{0,3}\d{3}\D{0,3}\d{4}|customer|airlines|phone|call ",
    re.I,
)
# Facts read from external documentation, not computed by this project
EXTERNAL_FACTS = {
    "checked_on": "2026-10-01",
    "patents": {
        "status": "PatentsView migrated to USPTO Open Data Portal on 2026-03-20",
        "access": "ODP requires registration and sign-in since 2026-06-18",
        "bulk_same_file_downloads_per_year_per_key": 20,
        "source": "https://data.uspto.gov/apis/api-rate-limits",
        # Zipped file sizes in bytes, read from the ODP dataset pages
        "tables": {
            "release_date": "2026-04-10",
            "source": [
                "https://data.uspto.gov/bulkdata/datasets/pvgpatdis",
                "https://data.uspto.gov/bulkdata/datasets/pvpgpubdis",
            ],
            "zip_bytes": {
                "g_patent": 232076712,
                "g_patent_abstract": 1708330746,
                "g_cpc_at_issue": 338958464,
                "g_assignee_disambiguated": 362741666,
                "pg_published_application": 301006689,
                "pg_published_application_abstract": 1597376724,
                "pg_cpc_at_issue": 202652269,
                "pg_assignee_disambiguated": 122366357,
                "pg_granted_pgpubs_crosswalk": 112671324,
            },
        },
    },
    "pypi": {
        "pypistats_overall_history_days": 186,
        "source": "https://pypistats.org/api/",
    },
    "github": {
        "search_requests_per_min_authenticated": 30,
        "search_requests_per_min_unauthenticated": 10,
        "code_search_requests_per_min": 10,
        "source": "https://docs.github.com/rest/search/search",
    },
    "gemini": {
        "public_free_tier_table": False,
        "limits_location": "Google AI Studio, per project",
        "source": "https://ai.google.dev/gemini-api/docs/rate-limits",
        # Free tier limits read from https://aistudio.google.com/rate-limit
        "free_tier_limits": {
            "project": "Default Gemini Project",
            "text_models": {
                "gemini-3.5-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500},
                "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500},
                "gemini-2.5-flash-lite": {"rpm": 10, "tpm": 250000, "rpd": 20},
                "gemini-3.8-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemini-3.7-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemini-3.6-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemini-3.5-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemini-3-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemini-2.5-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
                "gemma-4-26b": {"rpm": 30, "tpm": 16000, "rpd": 14400},
                "gemma-4-31b": {"rpm": 30, "tpm": 16000, "rpd": 14400},
            },
            "pro_models_available": False,
            "embedding_models": {"rpm": 100, "rpd": 1000},
        },
    },
}


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


def vocab_summary() -> dict:
    methods = pd.read_parquet(DATA / "methods.parquet")
    counts = json.loads((DATA / "pwc_term_counts.json").read_text())
    tasks, methods_used = counts["tasks"], counts["methods"]
    spam_mask = methods["name"].str.contains(SPAM) | methods["full_name"].fillna("").str.contains(SPAM)
    return {
        "methods_rows": int(len(methods)),
        "methods_num_papers_ge10": int((methods["num_papers"] >= 10).sum()),
        "methods_spam_like_rows": int(spam_mask.sum()),
        "methods_introduced_year_share_2000": round(float((methods["introduced_year"] == 2000).mean()), 3),
        "tasks_unique_in_annotations": len(tasks),
        "tasks_ge10_papers": sum(v >= 10 for v in tasks.values()),
        "tasks_ge50_papers": sum(v >= 50 for v in tasks.values()),
        "methods_unique_in_annotations": len(methods_used),
        "methods_ge10_papers_in_annotations": sum(v >= 10 for v in methods_used.values()),
    }


def annotation_summary() -> tuple[dict, dict, dict]:
    profile = pd.read_parquet(DATA / "pwc_profile.parquet")
    links = pd.read_parquet(DATA / "links.parquet", columns=["paper_arxiv_id"])
    profile["nterm"] = profile["ntask"] + profile["nmeth"]
    profile["has_code"] = profile["arxiv_id"].isin(set(links["paper_arxiv_id"].dropna()))
    per_year = {}
    for year in [2016, 2018, 2020, 2022, 2024, 2025]:
        sub = profile[profile["year"] == year]
        per_year[year] = {
            "n": int(len(sub)),
            "mean_terms": round(float(sub["nterm"].mean()), 3),
            "share_ge3": round(float((sub["nterm"] >= 3).mean()), 3),
            "share_zero": round(float((sub["nterm"] == 0).mean()), 3),
        }
    window = profile[profile["year"].between(2014, 2025)]
    arxiv_papers = window[window["arxiv_id"].notna()].groupby("year").size()
    code_share = window[window["year"] >= 2016].groupby("year")["has_code"].mean()
    return (
        per_year,
        {int(k): int(v) for k, v in arxiv_papers.items()},
        {int(k): round(float(v), 3) for k, v in code_share.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default=datetime.date.today().isoformat())
    args = parser.parse_args()

    arxiv = json.loads((DATA / "arxiv_counts.json").read_text())
    missing = [y for y, n in arxiv.items() if n is None]
    if missing:
        raise SystemExit(f"arxiv_counts.json has missing years: {missing}")

    annotations, pwc_arxiv, code_share = annotation_summary()
    result = {
        "meta": {
            "survey": "stage1_scope",
            "snapshot_date": args.snapshot_date,
            "git_commit": git_commit(),
            "source_files_sha256": {name: sha256(DATA / name) for name in SOURCE_FILES},
            "scripts": sorted(p.name for p in Path(__file__).parent.glob("0*")),
        },
        "arxiv_counts_by_year": arxiv,
        "pwc_vocab": vocab_summary(),
        "pwc_annotations_by_year": annotations,
        "pwc_arxiv_papers_by_year": pwc_arxiv,
        "pwc_code_link_share_by_year": code_share,
        "dictionary_matching_by_year": json.loads((DATA / "match_by_year.json").read_text()),
        "external_facts": EXTERNAL_FACTS,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scope_survey_{args.snapshot_date}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved", out)


if __name__ == "__main__":
    main()
