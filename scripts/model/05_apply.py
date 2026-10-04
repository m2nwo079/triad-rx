"""Applied mode (design 5.5, 5.6, 10.6): retrain on T1+T2 and rank the current candidates.

Usage: python scripts/model/05_apply.py

- Feature sets follow the V2 verdict: the engine without H features (P+T+R) when V2 failed.
- The model is retrained on T1 and T2 labels with the fixed settings; the number of trees is
  chosen by the same stratified CV on the combined training set.
- The prescription ranking drops redundant triads (rules D1-D3, applied mode only).
- When V1 failed, every output carries the flag "engine reliability not established".
"""
import datetime
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study
from triadrx.text import norm

FEATURES = Path("data/derived/features/main")
LABELS = Path("data/derived/labels/main")
SCORES = Path("data/derived/scores/main")
HYPERGRAPH = Path("data/derived/hypergraph/main")
VOCAB = Path("vocab/vocab_base.json")
OUT_DIR = Path("data/derived/applied")
RESULTS = Path("results")
PROSPECTIVE = Path("prospective")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_mod = load("train", Path(__file__).with_name("01_train.py"))
eval_mod = load("ev", Path(__file__).with_name("02_evaluate.py"))
feat_mod = load("feats", Path(__file__).resolve().parents[1] / "graph" / "05_build_features.py")


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


def name_tokens(term_id: str, keys: dict) -> list[str]:
    return norm(keys[term_id]).split()


def redundant_pair(a: str, b: str, keys: dict, vectors: dict, threshold: float) -> str | None:
    """Return the rule (D1, D2, D3) under which two terms name the same concept, else None."""
    ta, tb = name_tokens(a, keys), name_tokens(b, keys)
    if "".join(ta) == "".join(tb):
        return "D1"
    kinds = (a.split(":", 1)[0], b.split(":", 1)[0])
    if kinds == ("M", "T") and set(ta) <= set(tb):
        return "D2"
    if kinds == ("T", "M") and set(tb) <= set(ta):
        return "D2"
    if kinds[0] == kinds[1] and float(vectors[a] @ vectors[b]) >= threshold:
        return "D3"
    return None


def main() -> None:
    study = load_study()
    model_cfg, feat_cfg, cfg = study["model"], study["features"], study["applied"]["prescription"]
    validation = json.loads((RESULTS / "validation_main.json").read_text())
    sets = model_cfg["engine_sets"] if validation["verdict"]["V2"] else model_cfg["v2_baseline_sets"]
    reliability_flag = None if validation["verdict"]["V1"] else "engine reliability not established (V1 failed)"
    cols = train_mod.columns(feat_cfg, sets)

    # Retrain on T1 + T2 labels
    frames = []
    for tp in ["T1", "T2"]:
        f = pd.read_parquet(FEATURES / f"{tp}.parquet").merge(
            pd.read_parquet(LABELS / f"{tp}.parquet")[KEYS + ["label"]], on=KEYS, validate="one_to_one")
        frames.append(f)
    train_df = pd.concat(frames, ignore_index=True)
    y = train_df["label"].to_numpy()
    rounds, cv_ap, cv_sd = train_mod.choose_rounds(train_df[cols], y, model_cfg)
    boosters = [train_mod.train(train_df[cols], y, rounds, s, model_cfg) for s in model_cfg["seeds"]]

    applied = pd.read_parquet(FEATURES / "applied.parquet")
    applied["score"] = np.column_stack([b.predict(applied[cols]) for b in boosters]).mean(axis=1)
    applied = applied.sort_values(KEYS).reset_index(drop=True)
    order = eval_mod.ranking(applied["score"].to_numpy(), np.arange(len(applied)))
    applied["rank_engine"] = 0
    applied.loc[order, "rank_engine"] = np.arange(1, len(applied) + 1)

    # Redundancy filter (D1-D3)
    keys = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}
    edges = pd.read_parquet(HYPERGRAPH / "applied_build.parquet")
    vectors, _ = feat_mod.term_vectors(edges, feat_cfg["text"])
    same_type = set()
    for a, b, c in applied[KEYS].itertuples(index=False):
        for x, z in itertools.combinations((a, b, c), 2):
            if x.split(":", 1)[0] == z.split(":", 1)[0]:
                same_type.add((x, z))
    cos = np.array([float(vectors[x] @ vectors[z]) for x, z in same_type])
    threshold = float(np.quantile(cos, cfg["d3_same_type_cosine_quantile"]))
    reasons = []
    for a, b, c in applied[KEYS].itertuples(index=False):
        found = [r for x, z in itertools.combinations((a, b, c), 2) if (r := redundant_pair(x, z, keys, vectors, threshold))]
        reasons.append(",".join(sorted(set(found))))
    applied["redundant"] = reasons
    kept = applied[applied["redundant"] == ""].copy()
    kept = kept.loc[kept["rank_engine"].sort_values().index]
    kept["rank_prescription"] = np.arange(1, len(kept) + 1)
    applied = applied.merge(kept[KEYS + ["rank_prescription"]], on=KEYS, how="left")

    # Expected hit rate (design 10.5): T2 Precision@k of the engine that is used
    t2 = pd.read_parquet(SCORES / "T2.parquet").merge(pd.read_parquet(LABELS / "T2.parquet")[KEYS + ["label"]], on=KEYS)
    t2 = t2.sort_values(KEYS).reset_index(drop=True)
    name = "engine" if sets == model_cfg["engine_sets"] else "baseline"
    k = study["evaluation"]["top_k"]
    t2_order = eval_mod.ranking(t2[f"{name}_mean"].to_numpy(), np.arange(len(t2)))
    expected = eval_mod.weighted_precision_at_k(t2_order, t2["label"].to_numpy().astype(float), np.ones(len(t2)), k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    applied.to_parquet(OUT_DIR / "ranking.parquet")
    top = applied.dropna(subset=["rank_prescription"]).sort_values("rank_prescription").head(k)
    top_list = [{"rank": int(r.rank_prescription), "rank_engine": int(r.rank_engine), "triad": [r.a, r.b, r.c],
                 "score": float(r.score)} for r in top.itertuples()]
    counts = pd.Series([x for r in reasons for x in r.split(",") if x]).value_counts().to_dict()
    meta = {"step": "applied", "created": datetime.date.today().isoformat(), "git_commit": git_commit(),
            "data_cutoff": data_cutoff().isoformat(), "feature_sets": sets, "reliability_flag": reliability_flag,
            "applied_config": study["applied"], "model_params": model_cfg["params"], "seeds": model_cfg["seeds"],
            "applied_features_sha256": sha256(FEATURES / "applied.parquet")}
    result = {
        "meta": meta,
        "training": {"candidates": int(len(y)), "positives": int(y.sum()), "rounds": rounds,
                     "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd},
        "redundancy": {"d3_threshold": threshold, "triads_flagged": int((applied["redundant"] != "").sum()),
                       "triads_flagged_by_rule": counts, "candidates": int(len(applied)), "kept": int(len(kept))},
        "expected_hit_rate_at_k": {"k": k, "value": expected, "source": f"T2 Precision@{k} of {name} (reference only)"},
        "top_k_prescription": top_list,
    }
    (RESULTS / "applied_main.json").write_text(json.dumps(result, indent=2))
    PROSPECTIVE.mkdir(exist_ok=True)
    prospective = PROSPECTIVE / f"{datetime.date.today().isoformat()}.json"
    prospective.write_text(json.dumps({"meta": meta, "expected_hit_rate_at_k": result["expected_hit_rate_at_k"],
                                       "top_k_prescription": top_list}, indent=2))
    print(json.dumps({"training": result["training"], "redundancy": result["redundancy"],
                      "expected_hit_rate_at_k": result["expected_hit_rate_at_k"], "reliability_flag": reliability_flag,
                      "top_10": top_list[:10]}, indent=1))
    print("Saved", OUT_DIR / "ranking.parquet", RESULTS / "applied_main.json", prospective)


if __name__ == "__main__":
    main()
