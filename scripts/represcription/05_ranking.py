"""Prescription ranking with the third-study engine and its prospective registration (represcription preregistration 3-4).

Usage: python scripts/represcription/05_ranking.py

Locked: it stops unless the represcription preregistration records its freeze commit.
1. Training: the engine model (engine_reliability.conditional.rediagnosis.models.engine, P+H+T+R) on the v3
   features and relation labels of represcription.engine.training pooled; first-study model settings, trees by
   stratified cross-validation on the pooled set, mean over seeds (as the third-study verdict).
2. Ranking: the applied generator candidates (scripts/represcription/04_engine_inputs.py) by engine score,
   ties by term ids; the first-study redundancy filter D1-D3 (scripts/model/05_apply.py, reused through importlib)
   with the D3 quantile recomputed on these candidates and term vectors of the applied v3 build window.
3. Registration: the prescription top k and the baseline (same filtered population, prospective baseline feature
   descending, ties by term ids; scripts/model/06_prospective_register.py) in the run folder, in the first-study
   file formats so the first-study downstream scripts read them unchanged.

Outputs in represcription.run_dir: data/derived/applied/ranking.parquet (not committed), results/applied_main.json,
prospective/<date>.json, prospective/<date>_baseline.json, results/prospective_register.json
"""
import datetime
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import data_cutoff, engine_prereg_frozen, load_study, runtime_environment

TP = "applied"
FEATURES = Path("data/derived/features_v3")
LABELS = Path("data/derived/labels_v3/relation")
VOCAB = Path("vocab/vocab_base.json")
APPLIED_VOCAB = Path("vocab/timepoints/applied_v3_final.json")
APPLIED_BUILD = Path("data/derived/hypergraph/v3/applied_build.parquet")
VERDICT = Path("results/engine_v3_verdict_T3.json")
KEYS = ["a", "b", "c"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


apply_mod = load("apply_first", ROOT / "scripts/model/05_apply.py")
register_mod = load("register_first", ROOT / "scripts/model/06_prospective_register.py")
cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
verdict_v3 = load("t3_verdict_v3", ROOT / "scripts/matching/19_t3_verdict.py")
trainer, evaluate = verdict_v3.trainer, verdict_v3.evaluate


def require_frozen(study: dict) -> str:
    text = Path(study["represcription"]["prereg_file"]).read_text()
    if not engine_prereg_frozen(text):
        sys.exit("the prescription ranking is locked until the represcription preregistration records its freeze commit")
    return re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M).group(1).strip()


def redundancy(applied: pd.DataFrame, keys: dict, vectors: dict, quantile: float) -> tuple[list[str], float]:
    """First-study D1-D3 reasons per candidate, with the D3 threshold from these candidates' same-type pairs."""
    same_type = {(x, z) for a, b, c in applied[KEYS].itertuples(index=False)
                 for x, z in itertools.combinations((a, b, c), 2) if x.split(":", 1)[0] == z.split(":", 1)[0]}
    threshold = float(np.quantile([float(vectors[x] @ vectors[z]) for x, z in sorted(same_type)], quantile))
    reasons = []
    for a, b, c in applied[KEYS].itertuples(index=False):
        found = [r for x, z in itertools.combinations((a, b, c), 2) if (r := apply_mod.redundant_pair(x, z, keys, vectors, threshold))]
        reasons.append(",".join(sorted(set(found))))
    return reasons, threshold


def prescription_ranks(applied: pd.DataFrame) -> pd.DataFrame:
    """Engine rank over all candidates and prescription rank over the candidates that pass the filter."""
    applied = applied.sort_values(KEYS).reset_index(drop=True)
    order = evaluate.ranking(applied["score"].to_numpy(), np.arange(len(applied)))
    applied["rank_engine"] = 0
    applied.loc[order, "rank_engine"] = np.arange(1, len(applied) + 1)
    kept = applied[applied["redundant"] == ""].sort_values("rank_engine")
    applied["rank_prescription"] = np.nan
    applied.loc[kept.index, "rank_prescription"] = np.arange(1, len(kept) + 1)
    return applied


def main() -> None:
    study = load_study()
    freeze = require_frozen(study)
    rcfg = study["represcription"]
    run = Path(rcfg["run_dir"])
    model_cfg, feat_cfg = study["model"], study["features"]
    cols = trainer.columns(feat_cfg, study["engine_reliability"]["conditional"]["rediagnosis"]["models"]["engine"])

    frames = [cands_v3.attach_labels(pd.read_parquet(FEATURES / f"{tp}.parquet"), pd.read_parquet(LABELS / f"{tp}.parquet"))
              for tp in rcfg["engine"]["training"]]
    train = pd.concat(frames, ignore_index=True)
    y = train["label_v3"].to_numpy().astype(int)
    rounds, cv_ap, cv_sd = trainer.choose_rounds(train[cols], y, model_cfg)
    boosters = [trainer.train(train[cols], y, rounds, s, model_cfg) for s in model_cfg["seeds"]]

    applied = pd.read_parquet(FEATURES / f"{TP}.parquet")
    applied["score"] = np.mean([b.predict(applied[cols]) for b in boosters], axis=0)
    keys = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}
    vectors, _ = apply_mod.feat_mod.term_vectors(pd.read_parquet(APPLIED_BUILD), feat_cfg["text"])
    applied = applied.sort_values(KEYS).reset_index(drop=True)
    applied["redundant"], threshold = redundancy(applied, keys, vectors, study["applied"]["prescription"]["d3_same_type_cosine_quantile"])
    applied = prescription_ranks(applied)

    k = rcfg["prescription"]["top_k"]
    (run / "data/derived/applied").mkdir(parents=True, exist_ok=True)
    ranking_path = run / "data/derived/applied/ranking.parquet"
    applied.to_parquet(ranking_path)
    top = applied.dropna(subset=["rank_prescription"]).sort_values("rank_prescription").head(k)
    top_list = [{"rank": int(r.rank_prescription), "rank_engine": int(r.rank_engine), "triad": [r.a, r.b, r.c],
                 "score": float(r.score)} for r in top.itertuples()]
    flagged = applied["redundant"] != ""
    counts = pd.Series([x for r in applied["redundant"] for x in r.split(",") if x]).value_counts().to_dict()
    verdict = json.loads(VERDICT.read_text())
    expected = {"k": k, "value": verdict["by_k"][str(k)]["engine"]["precision"] if str(k) in verdict["by_k"] else None,
                "source": f"T3 precision@{k} of the third-study verdict (T3 reused; reference only)"}
    today = datetime.date.today().isoformat()
    meta = {"step": "represcription_ranking", "created": today, "git_commit": evaluate.git_commit(),
            "preregistration_freeze_commit": freeze, "environment": runtime_environment(), "data_cutoff": data_cutoff().isoformat(),
            "feature_sets": study["engine_reliability"]["conditional"]["rediagnosis"]["models"]["engine"],
            "reliability_flag": None, "engine_label": rcfg["label"], "applied_config": study["applied"],
            "represcription_config": rcfg, "model_params": model_cfg["params"], "seeds": model_cfg["seeds"],
            "inputs_sha256": {"applied_features": evaluate.sha256(FEATURES / f"{TP}.parquet"), "applied_vocab": evaluate.sha256(APPLIED_VOCAB),
                              "applied_build": evaluate.sha256(APPLIED_BUILD),
                              **{f"features_{tp}": evaluate.sha256(FEATURES / f"{tp}.parquet") for tp in rcfg["engine"]["training"]},
                              **{f"labels_{tp}": evaluate.sha256(LABELS / f"{tp}.parquet") for tp in rcfg["engine"]["training"]}}}
    result = {"meta": meta,
              "training": {"timepoints": rcfg["engine"]["training"], "candidates": int(len(y)), "positives": int(y.sum()),
                           "rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd},
              "redundancy": {"d3_threshold": threshold, "triads_flagged": int(flagged.sum()), "triads_flagged_by_rule": counts,
                             "candidates": int(len(applied)), "kept": int((~flagged).sum())},
              "expected_hit_rate_at_k": expected,
              "top_k_prescription": top_list}
    (run / "results").mkdir(parents=True, exist_ok=True)
    (run / "results/applied_main.json").write_text(json.dumps(result, indent=2))

    (run / "prospective").mkdir(parents=True, exist_ok=True)
    registration = run / "prospective" / f"{today}.json"
    baseline_path = run / "prospective" / f"{today}_baseline.json"
    if registration.exists() or baseline_path.exists():
        sys.exit(f"{registration} or its baseline already exists; the prescription is registered once")
    registration.write_text(json.dumps({"meta": meta, "expected_hit_rate_at_k": expected, "top_k_prescription": top_list}, indent=2))
    pop = register_mod.population(applied)
    feature = rcfg["prospective"]["baseline_feature"]
    border = register_mod.baseline_order(pop, feature)
    baseline = register_mod.triads(pop, border, k)
    prescribed = register_mod.triads(pop, register_mod.prescription_order(pop), k)
    if prescribed != [e["triad"] for e in top_list]:
        sys.exit("the ranking does not reproduce the registered prescription top k")
    baseline_path.write_text(json.dumps({"meta": meta, "baseline_top_k": [
        {"rank": i + 1, "triad": t, feature: float(pop.loc[j, feature])} for i, (t, j) in enumerate(zip(baseline, border))]}, indent=2))
    register = {"meta": meta, "population": int(len(pop)), "top_k": k,
                "overlap_prescription_baseline": len({tuple(t) for t in baseline} & {tuple(t) for t in prescribed}),
                "sha256": {"ranking": evaluate.sha256(ranking_path), "vocab": evaluate.sha256(APPLIED_VOCAB),
                           "registration": evaluate.sha256(registration), "baseline": evaluate.sha256(baseline_path)}}
    (run / "results/prospective_register.json").write_text(json.dumps(register, indent=2))
    print(json.dumps({"training": result["training"], "redundancy": result["redundancy"],
                      "overlap_prescription_baseline": register["overlap_prescription_baseline"], "top_10": top_list[:10]}, indent=1))
    print("Saved", ranking_path, registration, baseline_path)


if __name__ == "__main__":
    main()
