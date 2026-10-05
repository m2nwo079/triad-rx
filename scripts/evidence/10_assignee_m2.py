"""Assignee classification (design 7.5) and the M2 review sample (design 9.2, preregistration 5.4.1).

Usage: python scripts/evidence/10_assignee_m2.py --attempt 1
       python scripts/evidence/10_assignee_m2.py --attempt 2 --categories government

1. Classifies every assignee of the patent evidence with triadrx.assignees.
2. Builds the M2 population (config quality.m2.population):
   - all_patent_evidence_assignees: every assignee_id of the patent evidence
   - dvf_related_assignees: assignee_id of the related inventions of the DVF
     population, where an invention is related to a candidate when both terms of one
     of its three pairs are among the invention's matched terms (union over the
     invention's documents). Attempt 0 used this; it was voided before judging.
   related_inventions (count of related inventions per assignee) is kept in both cases.
3. Draws up to quality.m2.n_per_category assignees per category (all when fewer),
   with the configured seed, and writes the review file plus a read-only copy.
   --categories limits the sample to the categories being remeasured (preregistration
   5.4.1: only failed categories are remeasured).

The reviewer fills `correct` (1 or 0) and `note` following preregistration 5.4.1.
Only population and sample counts are written to results; no candidate-level values.

Outputs
- data/derived/evidence/assignee_classes.parquet (not committed)
- data/review/m2_review_attempt<N>.csv and m2_review_attempt<N>_original.csv
- results/m2_sample_attempt<N>.json
"""
import argparse
import datetime
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.assignees import AssigneeClassifier
from triadrx.study import load_study

PATENTS = Path("data/derived/evidence/patents.parquet")
ASSIGNEES = Path("data/derived/evidence/patent_assignees.parquet")
OUT_DIR = Path("data/derived/evidence")
REVIEW_DIR = Path("data/review")
RESULTS = Path("results")
CATEGORY_ORDER = ["company", "academic", "government", "individual", "ambiguous", "unknown"]


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


def display_name(row) -> str:
    org = row.disambig_assignee_organization or ""
    if org:
        return org
    first = row.disambig_assignee_individual_name_first or ""
    last = row.disambig_assignee_individual_name_last or ""
    return f"{first} {last}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--categories", nargs="+", help="remeasured categories (default: all configured)")
    args = parser.parse_args()

    study = load_study()
    m2 = study["quality"]["m2"]
    populations = {"dvf_related_assignees", "all_patent_evidence_assignees"}
    if m2["population"] not in populations or study["applied"]["dvf"]["patent_unit"] != "pair":
        sys.exit(f"population must be one of {sorted(populations)} with patent_unit 'pair'")
    review = REVIEW_DIR / f"m2_review_attempt{args.attempt}.csv"
    original = REVIEW_DIR / f"m2_review_attempt{args.attempt}_original.csv"
    if review.exists() or original.exists():
        sys.exit(f"{review} already exists; use a new attempt number")

    clf = AssigneeClassifier(study["applied"]["assignee_classes"])
    assignees = pd.read_parquet(ASSIGNEES)
    assignees = assignees[assignees["assignee_id"] != ""].copy()
    assignees["name"] = [display_name(r) for r in assignees.itertuples(index=False)]

    # One record per assignee_id: the most frequent (name, type) across its rows
    variants = assignees.groupby(["assignee_id", "name", "assignee_type"]).size().reset_index(name="rows")
    variants = variants.sort_values(["assignee_id", "rows", "name"], ascending=[True, False, True])
    per_id = variants.drop_duplicates("assignee_id").copy()
    type_conflicts = int((variants.groupby("assignee_id")["assignee_type"].nunique() > 1).sum())
    per_id["category"] = [clf.classify(t, n) for n, t in zip(per_id["name"], per_id["assignee_type"])]
    per_id = per_id[["assignee_id", "name", "assignee_type", "category"]].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_id.to_parquet(OUT_DIR / "assignee_classes.parquet", index=False)

    # Related inventions of the DVF population (pair unit)
    candidates = json.loads(Path(study["applied"]["dvf"]["prospective_file"]).read_text())["top_k_prescription"]
    pairs = {tuple(sorted(p)) for c in candidates for p in itertools.combinations(c["triad"], 2)}
    docs = pd.read_parquet(PATENTS, columns=["invention_id", "terms"])
    inv_terms = docs.explode("terms").dropna().groupby("invention_id")["terms"].agg(set)
    related = {inv for inv, ts in inv_terms.items() if any(a in ts and b in ts for a, b in pairs)}
    pool_ids = set(assignees.loc[assignees["invention_id"].isin(related), "assignee_id"])
    if m2["population"] == "all_patent_evidence_assignees":
        pool = per_id.copy()
    else:
        pool = per_id[per_id["assignee_id"].isin(pool_ids)].copy()
    inv_count = assignees[assignees["invention_id"].isin(related)].groupby("assignee_id")["invention_id"].nunique()
    pool["related_inventions"] = pool["assignee_id"].map(inv_count).fillna(0).astype(int)

    # A different seed per attempt keeps a remeasurement from redrawing the same sample
    seed = m2["seed"] + args.attempt
    rng = np.random.default_rng(seed)
    samples = []
    categories = args.categories or m2["categories"]
    unknown = sorted(set(categories) - set(m2["categories"]))
    if unknown:
        sys.exit(f"unknown categories: {unknown}")
    for category in categories:
        members = pool[pool["category"] == category].sort_values("assignee_id").reset_index(drop=True)
        n = min(m2["n_per_category"], len(members))
        chosen = members.iloc[sorted(rng.choice(len(members), size=n, replace=False))] if n else members
        samples.append(chosen.assign(census=len(members) <= m2["n_per_category"]))
    sample = pd.concat(samples, ignore_index=True)
    sample.insert(0, "review_id", [f"m2-{args.attempt}-{i:03d}" for i in range(len(sample))])
    sample["correct"] = ""
    sample["note"] = ""
    sample = sample[["review_id", "category", "assignee_id", "name", "assignee_type",
                     "related_inventions", "census", "correct", "note"]]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(review, index=False, encoding="utf-8-sig")
    sample.to_csv(original, index=False, encoding="utf-8-sig")
    original.chmod(0o444)

    def counts(frame: pd.DataFrame) -> dict:
        vc = frame["category"].value_counts()
        return {c: int(vc.get(c, 0)) for c in CATEGORY_ORDER}

    meta = {
        "step": "m2_sample",
        "attempt": args.attempt,
        "categories": categories,
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "population": m2["population"],
        "seed": seed,
        "n_per_category": m2["n_per_category"],
        "sampling": "numpy default_rng(quality.m2.seed + attempt).choice without replacement per category in config order, members sorted by assignee_id",
        "patents_sha256": sha256(PATENTS),
        "assignees_sha256": sha256(ASSIGNEES),
        "review_file": str(review),
        "review_original_sha256": sha256(original),
    }
    stats = {
        "assignee_ids_all": int(len(per_id)),
        "categories_all": counts(per_id),
        "assignee_ids_with_type_conflict": type_conflicts,
        "related_inventions": len(related),
        "dvf_related_assignees": len(pool_ids),
        "population": int(len(pool)),
        "categories_population": counts(pool),
        "sample": {c: int((sample["category"] == c).sum()) for c in categories},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"m2_sample_attempt{args.attempt}.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2))
    print(json.dumps(stats, indent=1))
    print("Saved", review, original, RESULTS / f"m2_sample_attempt{args.attempt}.json")


if __name__ == "__main__":
    main()
