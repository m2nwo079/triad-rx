"""T3 verdict of the third study (matching preregistration 6.4).

Usage: python scripts/matching/19_t3_verdict.py

Locked: it stops unless the matching preregistration records its freeze commit, and unless the power
result file matches the hash recorded in matching.verdict.power_result.
Inputs are the v3 features of the training timepoints (matching.engine.t3_training) and of T3, the v3
relation labels (data/derived/labels_v3/relation/<tp>.parquet, T3 made after the freeze), and the
first-study rule candidates of T3 on the v3 inputs.

1. Training: the engine and the engine without H (engine_reliability.conditional.rediagnosis.models) on the
   training timepoints pooled, first-study model settings, mean over seeds, as in the second-study verdict.
2. Pool: the generator's top B (matching.verdict.budget_multiple times the first-study rule count of T3).
3. Verdict (engine_reliability.verdict.sequence at matching.verdict.k, in order, a step only if all earlier
   ones passed; both bootstrap lower bounds must exceed the threshold) with the second-study statistics
   (scripts/engine/15_t3_verdict.py, reused through importlib) at the confidence level
   matching.t3_reuse.ci_level instead of evaluation.ci_level. The same statistics at
   matching.verdict.descriptive_k are reported and never judged.
4. Reading: a failed step whose sequence power in the power result is below minimum_sequence_power is
   reported as "insufficient power"; a failed S4 in that case reads as "undetermined" (preregistration 6.4).
No calibration and no secondary analysis (preregistration 6.4).

Output: results/engine_v3_verdict_T3.json
"""
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import engine_prereg_frozen, load_study, runtime_environment

FEATURES = Path("data/derived/features_v3")
LABELS = Path("data/derived/labels_v3/relation")
FIRST = Path("data/derived/candidates_v3/first")
RESULTS = Path("results")
KEYS = ["a", "b", "c"]
TP = "T3"
SEQUENCE_POWER = {"S1_recall": None, "S2_random": "S1_S2", "S3_npmi": "S1_S2_S3", "S4_h_contribution": "S1_S2_S3_S4"}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load("t3_verdict_v2", ROOT / "scripts/engine/15_t3_verdict.py")
cands_v3 = load("engine_candidates", ROOT / "scripts/matching/14_engine_candidates.py")
rd, dev, evaluate, trainer = v2.rd, v2.dev, v2.evaluate, v2.trainer


def require_frozen(study: dict) -> str:
    text = Path(study["matching"]["prereg_file"]).read_text()
    if not engine_prereg_frozen(text):
        sys.exit("T3 verdict is locked until the matching preregistration records its freeze commit")
    return re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M).group(1).strip()


def power_at_choice(verdict: dict) -> dict:
    """Sequence powers at the chosen budget and k from the recorded power result (hash checked)."""
    rec = verdict["power_result"]
    path = Path(rec["file"])
    if evaluate.sha256(path) != rec["sha256"]:
        sys.exit(f"{path} does not match the recorded hash")
    power = json.loads(path.read_text())
    budget = str(power["budgets"][power["budgets"].index(power["choice"]["budget"])])
    cell = power["power"]["t3_conservative"][budget]
    row = cell["by_k"][str(verdict["k"])]
    return {"S1": cell["S1"], **{name: row[name] for name in ("S1_S2", "S1_S2_S3", "S1_S2_S3_S4", "S4")},
            "minimum": power["meta"]["power_config"]["minimum_sequence_power"]}


def run_sequence(raw: dict, sequence: list[dict]) -> dict:
    """Pass, fail or not tested, in order; a step is tested only if every earlier step passed."""
    out, open_ = {}, True
    for s in sequence:
        out[s["step"]] = ("pass" if raw[s["step"]] else "fail") if open_ else "not tested"
        open_ = open_ and raw[s["step"]]
    return out


def reading(sequence: dict, power: dict) -> dict:
    """Preregistered reading of each step (6.4): failures below the minimum power read as insufficient power."""
    out = {}
    for step, status in sequence.items():
        key = SEQUENCE_POWER[step]
        p = power["S1"] if key is None else power[key]
        if status != "fail":
            out[step] = status
        elif p < power["minimum"]:
            out[step] = "undetermined (insufficient power)" if step == "S4_h_contribution" else "fail (insufficient power)"
        else:
            out[step] = "fail"
    return out


def main() -> None:
    study = load_study()
    freeze = require_frozen(study)
    m = study["matching"]
    verdict, er = m["verdict"], study["engine_reliability"]
    rcfg, model_cfg, feat_cfg = er["conditional"]["rediagnosis"], study["model"], study["features"]
    ev = {**study["evaluation"], "ci_level": m["t3_reuse"]["ci_level"]}
    power = power_at_choice(verdict)

    labels = {tp: pd.read_parquet(LABELS / f"{tp}.parquet") for tp in [*m["engine"]["t3_training"], TP]}
    train = pd.concat([cands_v3.attach_labels(pd.read_parquet(FEATURES / f"{tp}.parquet"), labels[tp])
                       for tp in m["engine"]["t3_training"]], ignore_index=True)
    first_count = json.loads((RESULTS / f"engine_v3_candidates_{TP}.json").read_text())["first_study_rule"]["candidates"]
    budget = verdict["budget_multiple"] * first_count
    pool = cands_v3.attach_labels(pd.read_parquet(FEATURES / f"{TP}.parquet"), labels[TP])
    pool = pool[pool["gen_rank"] <= budget].sort_values(KEYS).reset_index(drop=True)
    if len(pool) != budget:
        sys.exit(f"pool has {len(pool)} candidates, budget is {budget}")
    y = pool["label_v3"].to_numpy().astype(float)
    y_train = train["label_v3"].to_numpy().astype(int)

    ids = np.arange(len(pool))
    scores, models = {}, {}
    for name in ("engine", "engine_no_h"):
        cols = trainer.columns(feat_cfg, rcfg["models"][name])
        rounds, cv_ap, cv_sd = trainer.choose_rounds(train[cols], y_train, model_cfg)
        boosters = [trainer.train(train[cols], y_train, rounds, s, model_cfg) for s in model_cfg["seeds"]]
        scores[name] = np.mean([b.predict(pool[cols]) for b in boosters], axis=0)
        models[name] = {"rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd}
    orders = {"engine": evaluate.ranking(scores["engine"], ids), "engine_no_h": evaluate.ranking(scores["engine_no_h"], ids),
              "baseline": evaluate.ranking(pool["P_npmi_mean"].to_numpy(), ids)}

    table = labels[TP]
    pos = table[table["label_v3"]][KEYS].reset_index(drop=True)
    keys = list(pos.itertuples(index=False, name=None))
    pool_set = set(pool[KEYS].itertuples(index=False, name=None))
    first_set = set(pd.read_parquet(FIRST / f"{TP}.parquet")[KEYS].itertuples(index=False, name=None))
    s1 = v2.recall_diff_intervals(pos, np.array([t in pool_set for t in keys], float),
                                  np.array([t in first_set for t in keys], float), ev)

    ks = [verdict["k"]] + verdict["descriptive_k"]
    s2 = dev.lift_intervals(pool, orders["engine"], y, ks, ev)
    s3 = rd.diff_intervals(pool, {"engine": orders["engine"]}, orders["baseline"], y, ks, ev)["engine"]
    s4 = rd.diff_intervals(pool, {"engine": orders["engine"]}, orders["engine_no_h"], y, ks, ev)["engine"]
    point = dev.ranking_metrics(pool, orders["engine"], y, ks, er["dev_eval"]["top_terms"])
    by_k = {str(k): {"engine": point[str(k)], "S2_lift_ci": s2[str(k)], "S3_diff_ci": s3[str(k)], "S4_diff_ci": s4[str(k)],
                     "baseline_precision": float(y[orders["baseline"][:k]].mean()),
                     "engine_no_h_precision": float(y[orders["engine_no_h"][:k]].mean())} for k in ks}

    k = str(verdict["k"])
    thresholds = {s["step"]: s["threshold"] for s in er["verdict"]["sequence"]}
    raw = {"S1_recall": v2.passes(s1, thresholds["S1_recall"]), "S2_random": v2.passes(s2[k], thresholds["S2_random"]),
           "S3_npmi": v2.passes(s3[k], thresholds["S3_npmi"]), "S4_h_contribution": v2.passes(s4[k], thresholds["S4_h_contribution"])}
    sequence = run_sequence(raw, er["verdict"]["sequence"])
    inputs = {f"features_{tp}": FEATURES / f"{tp}.parquet" for tp in [*m["engine"]["t3_training"], TP]}
    inputs |= {f"labels_{tp}": LABELS / f"{tp}.parquet" for tp in [*m["engine"]["t3_training"], TP]}
    inputs["first_study_rule_T3"] = FIRST / f"{TP}.parquet"
    result = {
        "meta": {"step": "engine_v3_verdict_T3", "created": datetime.date.today().isoformat(),
                 "git_commit": evaluate.git_commit(), "preregistration_freeze_commit": freeze,
                 "environment": runtime_environment(), "verdict_config": verdict, "evaluation_config": ev,
                 "model_config": model_cfg, "training_timepoints": m["engine"]["t3_training"],
                 "inputs_sha256": {name: evaluate.sha256(path) for name, path in inputs.items()},
                 "note": "T3 was judged once in the second study; not an independent confirmation (preregistration 6.4)."},
        "labels": {"realised_triads": int(len(table)), "label_v3_positive": int(table["label_v3"].sum()),
                   "pool_candidates": int(len(pool)), "pool_positive": int(y.sum())},
        "train": {"candidates": int(len(train)), "positives": int(y_train.sum())},
        "models": models,
        "S1_recall": s1,
        "by_k": by_k,
        "sequence": sequence,
        "power_at_choice": power,
        "reading": reading(sequence, power),
    }
    (RESULTS / "engine_v3_verdict_T3.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"labels": result["labels"], "sequence": sequence, "reading": result["reading"],
                      "S1_recall": s1["point"]}, indent=2))


if __name__ == "__main__":
    main()
