"""Register the proximity baseline and input hashes for the prospective check (preregistration 8).

Usage: python scripts/model/06_prospective_register.py

Run once, before any prospective outcome exists. The population is the applied-mode
candidates that pass the redundancy filter (applied.prescription). The baseline ranks that
population by prospective.baseline.feature (descending, ties by term ids) and keeps the top
prospective.top_k. The prescription top k must equal the existing registration file.

Outputs
- prospective/<registration date>_baseline.json: baseline top k (the registration file is not changed)
- results/prospective_register.json: SHA-256 of the ranking, vocabulary and registration files,
  overlap between the prescription and the baseline, data_cutoff
The ranking file is not committed; copy it to the backup folder and keep its hash.
"""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study

RESULTS = Path("results")
KEYS = ["a", "b", "c"]


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


def population(ranking: pd.DataFrame) -> pd.DataFrame:
    """Candidates passing the redundancy filter, in a fixed order (term ids)."""
    kept = ranking[ranking["rank_prescription"].notna()]
    return kept.sort_values(KEYS).reset_index(drop=True)


def prescription_order(pop: pd.DataFrame) -> list[int]:
    return pop.sort_values("rank_prescription", kind="mergesort").index.tolist()


def baseline_order(pop: pd.DataFrame, feature: str) -> list[int]:
    """Feature descending; ties by term ids (pop is already sorted by term ids)."""
    return pop.sort_values(feature, ascending=False, kind="mergesort").index.tolist()


def triads(pop: pd.DataFrame, order: list[int], k: int) -> list[list[str]]:
    return [list(pop.loc[i, KEYS]) for i in order[:k]]


def main() -> None:
    study = load_study()
    cfg = study["prospective"]
    k = cfg["top_k"]
    ranking_path, vocab_path = Path(cfg["ranking"]), Path(cfg["vocab"])
    registration_path, baseline_path = Path(cfg["registration"]), Path(cfg["baseline"]["file"])
    if baseline_path.exists():
        sys.exit(f"{baseline_path} already exists; the baseline is registered once")

    pop = population(pd.read_parquet(ranking_path))
    registration = json.loads(registration_path.read_text())
    registered = [e["triad"] for e in sorted(registration["top_k_prescription"], key=lambda e: e["rank"])]
    prescribed = triads(pop, prescription_order(pop), len(registered))
    if prescribed != registered:
        sys.exit("the ranking file does not reproduce the registered prescription top k")

    baseline = triads(pop, baseline_order(pop, cfg["baseline"]["feature"]), k)
    overlap = len({tuple(t) for t in baseline} & {tuple(t) for t in prescribed[:k]})
    cutoff = data_cutoff().isoformat()
    meta = {"step": "prospective_register", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "data_cutoff": cutoff, "prospective_config": cfg}
    baseline_path.write_text(json.dumps({"meta": meta, "baseline_top_k": [
        {"rank": i + 1, "triad": t, cfg["baseline"]["feature"]: float(pop.loc[j, cfg["baseline"]["feature"]])}
        for i, (t, j) in enumerate(zip(baseline, baseline_order(pop, cfg["baseline"]["feature"])))]}, indent=2))
    result = {
        "meta": meta,
        "population": int(len(pop)),
        "top_k": k,
        "overlap_prescription_baseline": overlap,
        "sha256": {"ranking": sha256(ranking_path), "vocab": sha256(vocab_path),
                   "registration": sha256(registration_path), "baseline": sha256(baseline_path)},
    }
    (RESULTS / "prospective_register.json").write_text(json.dumps(result, indent=2))
    print(f"population {len(pop)}, top {k}, overlap {overlap}; saved {baseline_path}")


if __name__ == "__main__":
    main()
