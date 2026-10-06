"""Build the private Kaggle dataset for engine step 4 outside the repository.

Usage: python kaggle/step4/package.py <output dir> <kaggle user>

Writes repo.bundle (the current branch as committed), data.tar.gz (the derived inputs and the papers
first submitted in the years of the development build and label windows and the T3 build window,
plus the T3 label window only once the engine preregistration is frozen) and dataset-metadata.json. Nothing is written inside
the repository.
"""
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import engine_prereg_frozen

INPUTS = [
    "data/derived/hypergraph/main/T1_build.parquet", "data/derived/hypergraph/main/T2_build.parquet",
    "data/derived/labels_v2/main/T1.parquet", "data/derived/labels_v2/main/T2.parquet",
    "data/derived/features_v2/main/T1.parquet", "data/derived/features_v2/main/T2.parquet",
    "data/derived/pair_evidence/main/T1.parquet", "data/derived/pair_evidence/main/T2.parquet",
    "data/derived/features_learned/main/T1.parquet", "data/derived/features_learned/main/T2.parquet",
    "data/derived/candidates/main/T2.parquet",
    "data/derived/hypergraph/main/T3_build.parquet", "data/derived/candidates/main/T3.parquet",
    "data/derived/features_v2/main/R2.parquet", "data/derived/features_v2/main/R3.parquet",
    "data/derived/features_v2/main/T3.parquet",
]
DATASET = "triad-rx-engine-step4-data"


def main() -> None:
    out, user = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "bundle", "create", str(out / "repo.bundle"), branch], check=True)
    study = json.loads(Path("config/study.json").read_text())
    years = [y for tp in study["timepoints"].values() if "label" in tp for y in (*tp["build"], *tp["label"])]
    er = study["engine_reliability"]["timepoints"]
    years += [y for tp in er.values() for key in ("build", "label") if key in tp for y in tp[key]]
    first, last = min(years), max(years)
    # The T3 label window leaves this machine only after the engine preregistration is frozen
    assert last < er["T3"]["label_start"], "development years reach the T3 label window"
    papers = pd.read_parquet("data/arxiv/papers.parquet", columns=["id", "title", "abstract", "first_date"])
    year = pd.to_datetime(papers["first_date"], utc=True).dt.year
    frozen = engine_prereg_frozen(Path(study["engine_reliability"]["prereg_file"]).read_text())
    keep = (year >= first) & ((year <= last) | (frozen & (year >= er["T3"]["label_start"])))
    subset = papers[keep].reset_index(drop=True)
    buf = io.BytesIO()
    subset.to_parquet(buf, index=False)
    with tarfile.open(out / "data.tar.gz", "w:gz") as tar:
        for rel in INPUTS:
            tar.add(rel, arcname=rel)
        info = tarfile.TarInfo("data/arxiv/papers.parquet")
        info.size = buf.getbuffer().nbytes
        buf.seek(0)
        tar.addfile(info, buf)
    meta = {"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{branch}: {len(subset)} papers from {first} to {last}, files in {out}")


if __name__ == "__main__":
    main()
