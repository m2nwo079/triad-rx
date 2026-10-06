"""T3 verdict of the primary pipeline (engine preregistration 3, design 18.10).

Usage: python scripts/engine/15_t3_verdict.py

Locked: it stops unless preregistration_engine.md records its freeze commit (require_t3_label_unlocked).
1. T3 label window: papers first submitted from T3 label_start to the data cutoff, matched with the T3
   build-time final vocabulary; realised significant new triads and second-study labels as in
   scripts/engine/04_label_v2.py.
2. Training: the engine and the engine without H (conditional.rediagnosis.models) on the training
   timepoints of engine_reliability.training.T3 pooled, first-study model settings, mean over seeds.
   Out-of-fold engine scores on the pooled training data fit the Platt calibration.
3. Verdict (verdict.sequence at verdict.k, in order, a step only if all earlier ones passed; both
   bootstrap lower bounds must exceed the threshold):
   S1 recall of the conditional candidates (the generator's top B) minus that of the first-study
   candidates over all T3 second-study positives; S2 lift of the engine's top k; S3 engine minus
   P_npmi_mean precision at k; S4 engine minus engine-without-H precision at k. The same statistics
   at verdict.descriptive_k are reported and never judged.
4. Calibration (secondary): Brier score and an equal-count reliability curve of the calibrated
   engine probabilities on the T3 pool.

Outputs: data/derived/hypergraph/main/T3_label.parquet and data/derived/labels_v2/main/T3.parquet
(not committed), results/engine_verdict_T3.json
"""
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import data_cutoff, load_study, require_t3_label_unlocked, runtime_environment, t3_window

PAPERS = Path("data/arxiv/papers.parquet")
VOCAB = Path("vocab/vocab_base.json")
DERIVED = Path("data/derived")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
TP = "T3"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


labels_v2 = load("label_v2", ROOT / "scripts/engine/04_label_v2.py")
cond = load("conditional_candidates", ROOT / "scripts/engine/07_conditional_candidates.py")
rd = load("rediagnose", ROOT / "scripts/engine/08_rediagnose.py")
graph = labels_v2.evidence.graph
dev, evaluate, trainer = rd.dev, rd.evaluate, rd.trainer


def recall_diff_intervals(pos: pd.DataFrame, in_pool: np.ndarray, in_first: np.ndarray, ev: dict) -> dict:
    """Point and both bootstrap intervals of recall(pool) - recall(first) over the positives."""
    terms, triads = dev.term_index(pos)
    rng = np.random.default_rng(ev["bootstrap_seed"])
    vals = {"candidate": [], "term_block": []}
    for _ in range(ev["bootstrap_n"]):
        for kind, w in (("candidate", evaluate.candidate_weights(rng, len(pos))),
                        ("term_block", evaluate.term_block_weights(rng, triads, terms))):
            total = w.sum()
            vals[kind].append((w @ in_pool - w @ in_first) / total if total > 0 else float("nan"))
    point = (in_pool.sum() - in_first.sum()) / len(pos) if len(pos) else float("nan")
    return {"point": float(point), "recall_pool": float(in_pool.mean()), "recall_first": float(in_first.mean()),
            **{kind: evaluate.interval(v, ev["ci_level"]) for kind, v in vals.items()}}


def passes(ci: dict, threshold: float) -> bool:
    return bool(ci["candidate"]["lower"] > threshold and ci["term_block"]["lower"] > threshold)


def reliability(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-count bins of calibrated probability with mean probability and observed rate."""
    order = np.argsort(prob, kind="stable")
    return [{"n": int(len(part)), "mean_probability": float(prob[part].mean()), "observed_rate": float(y[part].mean())}
            for part in np.array_split(order, bins) if len(part)]


def t3_labels(study: dict, build: pd.DataFrame) -> pd.DataFrame:
    start, end = t3_window("label", study, data_cutoff())
    papers = pd.read_parquet(PAPERS, columns=["id", "title", "abstract", "first_date"])
    window = papers[(papers["first_date"] >= start) & (papers["first_date"] <= end)].sort_values("id")
    terms = graph.final_terms(TP, {t["id"]: t for t in json.loads(VOCAB.read_text())["terms"]})
    label = graph.hyperedges(window, terms)
    label.to_parquet(DERIVED / f"hypergraph/main/{TP}_label.parquet")
    cfg = study["engine_reliability"]["label_v2"]
    realised = labels_v2.evidence.significant_new_triads(build, label, study["labels"]["fdr"])
    positive, _ = labels_v2.evidence.split_by_evidence(TP, cfg, label, realised)
    table = labels_v2.label_table(realised, positive, cfg["min_evidence_papers"])
    table.to_parquet(DERIVED / f"labels_v2/main/{TP}.parquet", index=False)
    return table


def main() -> None:
    study = load_study()
    require_t3_label_unlocked(study)
    er = study["engine_reliability"]
    verdict, rcfg = er["verdict"], er["conditional"]["rediagnosis"]
    model_cfg, feat_cfg, ev = study["model"], study["features"], study["evaluation"]
    freeze = re.search(r"^고정 커밋:\s*(.+)$", Path(er["prereg_file"]).read_text(), flags=re.M).group(1).strip()

    build = pd.read_parquet(DERIVED / f"hypergraph/main/{TP}_build.parquet")
    table = t3_labels(study, build)
    pool = cond.attach_labels(pd.read_parquet(DERIVED / f"features_v2/main/{TP}.parquet"), table)
    pool = pool.sort_values(KEYS).reset_index(drop=True)
    y = pool["label_v2"].to_numpy().astype(float)

    train = pd.concat([pd.read_parquet(DERIVED / f"features_v2/main/{tp}.parquet") for tp in er["training"][TP]], ignore_index=True)
    y_train = train["label_v2"].to_numpy().astype(int)
    ids = np.arange(len(pool))
    scores, models = {}, {}
    for name in ("engine", "engine_no_h"):
        cols = trainer.columns(feat_cfg, rcfg["models"][name])
        rounds, cv_ap, cv_sd = trainer.choose_rounds(train[cols], y_train, model_cfg)
        boosters = [trainer.train(train[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
        scores[name] = np.mean([b.predict(pool[cols]) for b in boosters], axis=0)
        models[name] = {"rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd}
        if name == "engine":
            oof = np.zeros(len(train))
            folds = StratifiedKFold(n_splits=model_cfg["cv_folds"], shuffle=True, random_state=model_cfg["seeds"][0])
            for fit, held in folds.split(train[cols], y_train):
                oof[held] = trainer.train(train[cols].iloc[fit], y_train[fit], rounds, model_cfg["seeds"][0], model_cfg).predict(train[cols].iloc[held])
            platt = LogisticRegression().fit(oof.reshape(-1, 1), y_train)
    orders = {"engine": evaluate.ranking(scores["engine"], ids), "engine_no_h": evaluate.ranking(scores["engine_no_h"], ids),
              "baseline": evaluate.ranking(pool["P_npmi_mean"].to_numpy(), ids)}

    pos = table[table["label_v2"]][KEYS].reset_index(drop=True)
    keys = list(pos.itertuples(index=False, name=None))
    pool_set = set(pool[KEYS].itertuples(index=False, name=None))
    first_set = set(pd.read_parquet(DERIVED / f"candidates/main/{TP}.parquet")[KEYS].itertuples(index=False, name=None))
    s1 = recall_diff_intervals(pos, np.array([t in pool_set for t in keys], float), np.array([t in first_set for t in keys], float), ev)

    ks = [verdict["k"]] + verdict["descriptive_k"]
    s2 = dev.lift_intervals(pool, orders["engine"], y, ks, ev)
    s3 = rd.diff_intervals(pool, {"engine": orders["engine"]}, orders["baseline"], y, ks, ev)["engine"]
    s4 = rd.diff_intervals(pool, {"engine": orders["engine"]}, orders["engine_no_h"], y, ks, ev)["engine"]
    point = dev.ranking_metrics(pool, orders["engine"], y, ks, er["dev_eval"]["top_terms"])
    by_k = {str(k): {"engine": point[str(k)], "S2_lift_ci": s2[str(k)], "S3_diff_ci": s3[str(k)], "S4_diff_ci": s4[str(k)],
                     "baseline_precision": float(y[orders["baseline"][:k]].mean()),
                     "engine_no_h_precision": float(y[orders["engine_no_h"][:k]].mean())} for k in ks}

    k = str(verdict["k"])
    thresholds = {s["step"]: s["threshold"] for s in verdict["sequence"]}
    raw = {"S1_recall": passes(s1, thresholds["S1_recall"]), "S2_random": passes(s2[k], thresholds["S2_random"]),
           "S3_npmi": passes(s3[k], thresholds["S3_npmi"]), "S4_h_contribution": passes(s4[k], thresholds["S4_h_contribution"])}
    sequence, open_ = {}, True
    for s in verdict["sequence"]:
        sequence[s["step"]] = ("pass" if raw[s["step"]] else "fail") if open_ else "not tested"
        open_ = open_ and raw[s["step"]]

    prob = platt.predict_proba(scores["engine"].reshape(-1, 1))[:, 1]
    result = {
        "meta": {"step": "engine_verdict_T3", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "preregistration_freeze_commit": freeze,
                 "environment": runtime_environment(), "verdict_config": verdict, "model_config": model_cfg,
                 "evaluation_config": ev, "training_timepoints": er["training"][TP],
                 "inputs_sha256": {"build": evaluate.sha256(DERIVED / f"hypergraph/main/{TP}_build.parquet"),
                                   "label": evaluate.sha256(DERIVED / f"hypergraph/main/{TP}_label.parquet"),
                                   "features": evaluate.sha256(DERIVED / f"features_v2/main/{TP}.parquet")}},
        "labels": {"realised_significant_triads": int(len(table)), "label_v2_positive": int(table["label_v2"].sum()),
                   "pool_candidates": int(len(pool)), "pool_positive": int(y.sum())},
        "train": {"candidates": int(len(train)), "positives": int(y_train.sum())},
        "models": models,
        "S1_recall": s1,
        "by_k": by_k,
        "sequence": sequence,
        "calibration": {"brier": float(np.mean((prob - y) ** 2)), "reliability": reliability(prob, y),
                        "platt": {"coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0])}},
    }
    (RESULTS / "engine_verdict_T3.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"labels": result["labels"], "sequence": sequence, "S1_recall": s1["point"]}, indent=2))


if __name__ == "__main__":
    main()
