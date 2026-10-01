"""Profile PWC vocabulary and paper annotations (C2, C3 part 1)."""
import collections
import json
from pathlib import Path

import pandas as pd

DATA = Path("data/scope")
# title and abstract are not used in this script
COLS = ["arxiv_id", "tasks", "date", "methods"]


def load_papers(cols=COLS) -> pd.DataFrame:
    frames = [pd.read_parquet(DATA / f"pwa_{i}.parquet", columns=cols) for i in range(4)]
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df


def main() -> None:
    df = load_papers()
    df["ntask"] = df["tasks"].apply(lambda x: len(x) if x is not None else 0)
    df["meth"] = df["methods"].apply(lambda x: [m["name"] for m in x] if x is not None else [])
    df["nmeth"] = df["meth"].apply(len)
    task_counts = collections.Counter(t for x in df["tasks"] if x is not None for t in x)
    method_counts = collections.Counter(m for x in df["meth"] for m in x)
    (DATA / "pwc_term_counts.json").write_text(
        json.dumps({"tasks": task_counts, "methods": method_counts})
    )
    df[["arxiv_id", "year", "ntask", "nmeth"]].to_parquet(DATA / "pwc_profile.parquet")
    print("papers", len(df))


if __name__ == "__main__":
    main()
