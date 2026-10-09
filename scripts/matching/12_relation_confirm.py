"""Confirmation sample of the relation classifier (matching preregistration 5.3).

Usage:
  python scripts/matching/12_relation_confirm.py frame    # choose triads, write classifier inputs
  python scripts/matching/12_relation_confirm.py sample --outputs <relation_outputs.jsonl>
  python scripts/matching/12_relation_confirm.py sample2 --outputs <T1 relation_outputs.jsonl>

frame: realised T1 triads as in scripts/matching/08_label_v3_sample.py (v3 hypergraphs, excluded pairs
dropped), minus the triads of the development reviews, then matching.relation.confirm.frame_triads of
them at random (seed confirm.seed). Every (triad, paper) of those triads becomes a classifier input.
sample: a triad is model-positive when one of its papers is classified combination. As in the engine
study's draw, distinct model-positive triads are drawn with one of their combination papers, and
distinct model-negative triads with one of their papers; rows are shuffled into one blind sheet with
the columns of the label v3 sheet, and the key records the model side.
sample2 (attempt 2, matching preregistration 5.4): the frame is the full T1 classification
(data/derived/relation/T1_frame.parquet and its early-stopped outputs) minus the development triads and
the attempt 1 frame triads. Inputs are checked against the hashes in matching.relation.confirm2, and the
model-positive sample size is recomputed from the power rule before drawing.

Outputs (not committed): data/derived/relation/confirm_inputs.jsonl, data/derived/relation/confirm_frame.parquet,
confirm.review_file, confirm.review_key_file and the read-only _original.csv; summaries
results/relation_confirm_frame.json and results/relation_confirm_sample.json; for sample2,
confirm2.review_file, confirm2.review_key_file, the read-only _original.csv and results/relation_confirm2_sample.json
"""
import argparse
import csv
import datetime
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.stats import wilson_interval
from triadrx.study import load_study, runtime_environment

PAPERS = Path("data/arxiv/papers.parquet")
HYPERGRAPH = Path("data/derived/hypergraph/v3")
TIMEPOINTS = Path("vocab/timepoints")
OUT_DIR = Path("data/derived/relation")
RESULTS = Path("results")
TP = "T1"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


l3 = load("label_v3_sample", ROOT / "scripts/matching/08_label_v3_sample.py")
rel_in = load("relation_inputs", ROOT / "scripts/matching/10_relation_inputs.py")
evidence, evaluate = l3.evidence, l3.evaluate


def dev_triads(cfg: dict) -> set[tuple]:
    out = set()
    for source in cfg["dev_reviews"]:
        key = pd.read_csv(source["key"], encoding="utf-8-sig", dtype=str, keep_default_na=False)
        out |= {tuple(sorted(t)) for t in zip(key["term_a"], key["term_b"], key["term_c"])}
    return out


def frame(study: dict) -> None:
    cfg = study["matching"]["relation"]
    vocab = json.loads((TIMEPOINTS / f"{TP}_v3_final.json").read_text())
    build = pd.read_parquet(HYPERGRAPH / f"{TP}_build.parquet")
    label = pd.read_parquet(HYPERGRAPH / f"{TP}_label.parquet")
    realised = evidence.significant_new_triads(build, label, study["labels"]["fdr"])
    excluded = l3.excluded_pairs(vocab)
    seen = dev_triads(cfg)
    pool = sorted(t for t in realised if not l3.holds_excluded_pair(t, excluded) and t not in seen)
    rng = np.random.default_rng(cfg["confirm"]["seed"])
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(cfg["confirm"]["frame_triads"], len(pool)), replace=False))]
    pairs = [(t, label.iloc[i]["paper_id"]) for t in chosen for i in realised[t]]
    ids = sorted({p for _, p in pairs})
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"], filters=[("id", "in", ids)]).set_index("id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = OUT_DIR / "confirm_inputs.jsonl"
    rows = []
    with inputs.open("w", encoding="utf-8") as handle:
        for n, (t, pid) in enumerate(pairs):
            rid = f"cf-{n:05d}"
            handle.write(json.dumps(rel_in.record(rid, list(t), pid, text.at[pid, "title"], text.at[pid, "abstract"]), ensure_ascii=False) + "\n")
            rows.append({"id": rid, "a": t[0], "b": t[1], "c": t[2], "paper_id": pid})
    frame_path = OUT_DIR / "confirm_frame.parquet"
    pd.DataFrame(rows).to_parquet(frame_path, index=False)
    result = {"meta": {"step": "relation_confirm_frame", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                       "environment": runtime_environment(), "confirm_config": cfg["confirm"],
                       "inputs_sha256": evaluate.sha256(inputs), "frame_sha256": evaluate.sha256(frame_path)},
              "eligible_triads": len(pool),
              "frame_triads": len(chosen), "frame_pairs": len(pairs)}
    (RESULTS / "relation_confirm_frame.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


def split(frame_rows: pd.DataFrame, outputs: pd.DataFrame) -> tuple[dict, dict]:
    """Model-positive triads with their combination papers, model-negative triads with all their papers."""
    df = frame_rows.merge(outputs[["id", "relation"]], on="id", validate="one_to_one")
    positive, negative = defaultdict(list), defaultdict(list)
    for t, g in df.groupby(["a", "b", "c"]):
        combos = g.loc[g["relation"] == "combination", "paper_id"].tolist()
        if combos:
            positive[t] = combos
        else:
            negative[t] = g["paper_id"].tolist()
    return dict(positive), dict(negative)


def write_sheet(rng, pos: list, neg: list, review: Path, key: Path, prefix: str) -> None:
    """Shuffle the drawn rows into one blind sheet and its key; the original and key become read-only."""
    rows = [("positive", t, p) for t, p in pos] + [("negative", t, p) for t, p in neg]
    order = rng.permutation(len(rows))
    text = pd.read_parquet(PAPERS, columns=["id", "title", "abstract"],
                           filters=[("id", "in", sorted({p for _, _, p in rows}))]).set_index("id")
    with review.open("w", newline="", encoding="utf-8-sig") as fr, key.open("w", newline="", encoding="utf-8-sig") as fk:
        wr, wk = csv.writer(fr), csv.writer(fk)
        wr.writerow(["review_id", "term_a", "term_b", "term_c", "paper_id", "title", "abstract", "judgement", "term_error", "note"])
        wk.writerow(["review_id", "rule", "term_a", "term_b", "term_c", "paper_id"])
        width = max(3, len(str(len(rows) - 1)))
        for n, i in enumerate(order):
            side, t, pid = rows[i]
            rid = f"{prefix}-{n:0{width}d}"
            wr.writerow([rid, *t, pid, text.at[pid, "title"], text.at[pid, "abstract"], "", "", ""])
            wk.writerow([rid, side, *t, pid])
    original = review.with_name(review.stem + "_original.csv")
    original.write_bytes(review.read_bytes())
    for path in (original, key):
        path.chmod(0o444)


def sample(study: dict, outputs_path: Path) -> None:
    cfg = study["matching"]["relation"]
    conf = cfg["confirm"]
    review, key = Path(conf["review_file"]), Path(conf["review_key_file"])
    if review.exists() or key.exists():
        sys.exit(f"{review} already exists")
    frame_rows = pd.read_parquet(OUT_DIR / "confirm_frame.parquet")
    outputs = pd.read_json(outputs_path, lines=True)
    positive, negative = split(frame_rows, outputs)
    rng = np.random.default_rng(conf["seed"] + 1)
    pos = evidence.draw(rng, positive, conf["positive"], set())
    neg = evidence.draw(rng, negative, conf["negative"], set())
    write_sheet(rng, pos, neg, review, key, "rc")
    meta = json.loads((outputs_path.parent / "confirm_meta.json").read_text()) if (outputs_path.parent / "confirm_meta.json").exists() else None
    result = {"meta": {"step": "relation_confirm_sample", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                       "environment": runtime_environment(), "confirm_config": conf, "model_run": meta,
                       "outputs_sha256": evaluate.sha256(outputs_path), "review_sha256": evaluate.sha256(review), "key_sha256": evaluate.sha256(key)},
              "frame_pairs": int(len(frame_rows)), "invalid": int((outputs["relation"] == "invalid").sum()),
              "model_positive_triads": len(positive), "model_negative_triads": len(negative),
              "sample": {"positive": len(pos), "negative": len(neg)}}
    (RESULTS / "relation_confirm_sample.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


def power(n: int, precision: float, target: float, z: float) -> float:
    """Probability that the Wilson lower bound of n draws reaches the target at the given true precision."""
    passing = [k for k in range(n + 1) if wilson_interval(k, n, z)[0] >= target]
    return float(binom.sf(passing[0] - 1, n, precision)) if passing else 0.0


def sample_size(conf: dict, target: float) -> int:
    """Smallest model-positive sample whose power reaches confirm2.power (matching preregistration 5.4)."""
    z = norm.ppf(1 - (1 - conf["confidence"]) / 2)
    n = conf["size_search"]["start"]
    while power(n, conf["assumed_precision"], target, z) < conf["power"]:
        n += conf["size_search"]["step"]
    return n


def drop_triads(frame_rows: pd.DataFrame, used: set[tuple]) -> pd.DataFrame:
    """Rows of triads not used before (development reviews or the attempt 1 frame)."""
    keys = [tuple(sorted(t)) for t in zip(frame_rows["a"], frame_rows["b"], frame_rows["c"])]
    return frame_rows[[k not in used for k in keys]]


def check_hash(path: Path, expected: str) -> None:
    actual = evaluate.sha256(path)
    if actual != expected:
        sys.exit(f"{path} does not match the preregistered hash ({actual})")


def seen_papers(cfg: dict) -> set[str]:
    """Papers of every earlier review sheet (development reviews and attempt 1)."""
    keys = [Path(source["key"]) for source in cfg["dev_reviews"]] + [Path(cfg["confirm"]["review_key_file"])]
    out = set()
    for path in keys:
        if path.exists():
            out |= set(pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)["paper_id"])
    return out


def sample_attempt2(study: dict, outputs_path: Path) -> None:
    cfg = study["matching"]["relation"]
    conf = cfg["confirm2"]
    target = study["matching"]["label_v3"]["target_positive_lower"]
    review, key = Path(conf["review_file"]), Path(conf["review_key_file"])
    if review.exists() or key.exists():
        sys.exit(f"{review} already exists")
    size = sample_size(conf, target)
    if size != conf["positive"]:
        sys.exit(f"power rule gives {size} model-positive draws, config has {conf['positive']}")
    frame_path, attempt1_path = OUT_DIR / f"{TP}_frame.parquet", OUT_DIR / "confirm_frame.parquet"
    check_hash(outputs_path, conf["outputs_sha256"])
    check_hash(frame_path, conf["frame_sha256"])
    check_hash(attempt1_path, conf["attempt1_frame_sha256"])
    attempt1 = pd.read_parquet(attempt1_path)
    attempt1_triads = {tuple(sorted(t)) for t in zip(attempt1["a"], attempt1["b"], attempt1["c"])}
    dev = dev_triads(cfg)
    full = pd.read_parquet(frame_path)
    frame_rows = drop_triads(full, dev | attempt1_triads)
    outputs = pd.read_json(outputs_path, lines=True)
    positive, negative = split(frame_rows, outputs)
    rng = np.random.default_rng(conf["seed"])
    pos = evidence.draw(rng, positive, conf["positive"], set())
    neg = evidence.draw(rng, negative, conf["negative"], set())
    write_sheet(rng, pos, neg, review, key, "rd")
    seen = seen_papers(cfg)
    drawn = [p for _, p in pos + neg]
    z = norm.ppf(1 - (1 - conf["confidence"]) / 2)
    all_triads = {tuple(sorted(t)) for t in zip(full["a"], full["b"], full["c"])}
    result = {"meta": {"step": "relation_confirm2_sample", "created": datetime.date.today().isoformat(), "git_commit": evaluate.git_commit(),
                       "environment": runtime_environment(), "confirm2_config": conf,
                       "outputs_sha256": conf["outputs_sha256"], "frame_sha256": conf["frame_sha256"],
                       "attempt1_frame_sha256": conf["attempt1_frame_sha256"],
                       "review_sha256": evaluate.sha256(review), "key_sha256": evaluate.sha256(key)},
              "frame_triads": len(all_triads),
              "excluded_triads": {"development": len(all_triads & dev), "attempt1_frame": len(all_triads & attempt1_triads)},
              "model_positive_triads": len(positive), "model_negative_triads": len(negative),
              "positive_triads_with_several_combination_papers": sum(len(v) > 1 for v in positive.values()),
              "power_at_sample_size": power(conf["positive"], conf["assumed_precision"], target, z),
              "sample": {"positive": len(pos), "negative": len(neg)},
              "papers_seen_in_earlier_sheets": sum(p in seen for p in drawn)}
    (RESULTS / "relation_confirm2_sample.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["frame", "sample", "sample2"])
    parser.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    study = load_study()
    if args.step == "frame":
        frame(study)
    elif args.step == "sample":
        sample(study, args.outputs)
    else:
        sample_attempt2(study, args.outputs)


if __name__ == "__main__":
    main()
