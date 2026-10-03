"""Build the R4 review sheet from T1 build-window sentences, using stratified random examples."""
import csv
import datetime
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.text import Matcher

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
STUDY = Path("config/study.json")
# The sheet quotes abstract sentences, so it stays in data/ and is not committed;
# only the reviewer's counts are committed later
OUT_SHEET = Path("data/review/r4_review_T1.csv")
OUT_RESULT = Path("results/r4_review_T1.json")
# Sentence boundary: end punctuation followed by whitespace and an upper-case letter or digit
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


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


def sentences(title: str, abstract: str) -> list[str]:
    text = re.sub(r"\s+", " ", f"{title}. {abstract}").strip()
    return [s for s in SENTENCE.split(text) if s]


def main() -> None:
    study = json.loads(STUDY.read_text())
    cfg = study["r4_review"]
    build_start, build_end = study["timepoints"]["T1"]["build"]
    vocab = json.loads(VOCAB.read_text())
    terms = vocab["terms"]
    candidates = {t["id"]: t for t in terms if t["status"] == "r4_candidate"}

    # Candidates are matched together with active terms so that longest-match rules
    # behave as they will in the final pipeline
    matcher = Matcher(terms, statuses=("active", "r4_candidate"))

    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"])
    years = papers["first_date"].dt.year
    # Decision-time rule: only papers first submitted inside the T1 build window
    t1 = papers[(years >= build_start) & (years <= build_end)].sort_values("id")

    docs = defaultdict(set)
    # (term id, stratum) -> list of (paper id, sentence), at most one sentence per paper
    pool = defaultdict(list)
    seen = set()
    for pid, title, abstract in t1[["id", "title", "abstract"]].itertuples(index=False):
        for sentence in sentences(title, abstract):
            found = matcher.terms(sentence)
            hits = found & candidates.keys()
            if not hits:
                continue
            for tid in sorted(hits):
                docs[tid].add(pid)
                if (tid, pid) in seen:
                    continue
                seen.add((tid, pid))
                # Stratum A: the sentence also contains another active term (technical context)
                others = {o for o in found if o != tid and o not in candidates}
                stratum = "A" if others else "B"
                pool[(tid, stratum)].append((pid, sentence))

    rng = random.Random(cfg["seed"])
    n = cfg["n_per_stratum"]
    rows = []
    for tid in sorted(candidates, key=lambda t: (-len(docs[t]), t)):
        term = candidates[tid]
        row = {
            "term_id": tid,
            "key": term["key"],
            "type": term["type"],
            "t1_docs": len(docs[tid]),
            "stratum_a_size": len(pool[(tid, "A")]),
            "stratum_b_size": len(pool[(tid, "B")]),
        }
        for stratum in ("A", "B"):
            items = pool[(tid, stratum)]
            picked = rng.sample(items, min(n, len(items)))
            for i in range(n):
                pid, text = picked[i] if i < len(picked) else ("", "")
                row[f"{stratum.lower()}{i + 1}_paper"] = pid
                row[f"{stratum.lower()}{i + 1}_sentence"] = text
        # Columns for the reviewer: how many shown examples use the technical sense
        row["a_technical"] = ""
        row["b_technical"] = ""
        row["note"] = ""
        rows.append(row)

    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SHEET.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "meta": {
            "step": "r4_review_sheet_T1",
            "created": datetime.date.today().isoformat(),
            "git_commit": git_commit(),
            "timepoint": "T1",
            "build_years": [build_start, build_end],
            "review_config": cfg,
            "papers_sha256": sha256(PAPERS),
            "vocab_sha256": sha256(VOCAB),
            "sampling": "per term, stratum A = sentence also contains another active term, stratum B = otherwise; one sentence per paper; uniform random within stratum with the configured seed",
        },
        "t1_papers": int(len(t1)),
        "candidates": len(candidates),
        "candidates_with_t1_docs": sum(1 for t in candidates if docs[t]),
        "ambiguous_sequences": {k: len(v) for k, v in matcher.ambiguous.items()},
    }
    OUT_RESULT.parent.mkdir(exist_ok=True)
    OUT_RESULT.write_text(json.dumps(result, indent=2))
    print("Saved", OUT_SHEET, "and", OUT_RESULT)


if __name__ == "__main__":
    main()
