"""Learned third-term score for conditional completion (design 18.9 step 4, signal a).

Usage: python scripts/engine/10_learned_generator.py

All completions of the selected seed pairs (scripts/engine/06_conditional_recall.py, reused through
importlib) get symmetric triad features from build-window pair matrices: log pair count, nPMI,
Adamic-Adar on the stable-pair graph and the pair combination-evidence share
(scripts/engine/09_pair_evidence.py), each as min, mean and max over the three pairs, the log
document count of the three terms (min, mean, max) and the generator score. A LightGBM model with
the first-study settings (scripts/model/01_train.py) is trained on the train timepoint with the
second-study label. Recall and precision at the generator budgets are compared with the nPMI
generator order: out-of-fold scores on the train timepoint (for choosing) and the mean over seeds
on the evaluation timepoint (for confirming). The output is descriptive and is never a verdict.
The top `diagnosis_budget_multiple` times the first-study candidate count by learned score are saved
as candidates in the format of scripts/engine/07_conditional_candidates.py (reused through
importlib): out-of-fold scores on the train timepoint, so that no candidate is chosen by a model
that saw its own label, and the mean over seeds on the evaluation timepoint.

Outputs
- results/learned_generator.json
- data/derived/candidates_learned/main/<tp>.parquet (not committed), summary results/candidates_learned_<tp>.json
"""
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment

HYPERGRAPH = Path("data/derived/hypergraph/main")
LABELS_V2 = Path("data/derived/labels_v2/main")
PAIR_EVIDENCE = Path("data/derived/pair_evidence/main")
CANDIDATES = Path("data/derived/candidates_learned/main")
RESULTS = Path("results")
PAIR_MATRICES = ["log_count", "npmi", "adamic_adar", "evidence_share"]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recall = load("conditional_recall", ROOT / "scripts/engine/06_conditional_recall.py")
trainer = load("train", ROOT / "scripts/model/01_train.py")
cands = load("conditional_candidates", ROOT / "scripts/engine/07_conditional_candidates.py")


def evidence_share(pairs: pd.DataFrame, index: dict) -> np.ndarray:
    """Symmetric matrix of n_evidence / n_cooc; zero for pairs that never co-occurred."""
    n = len(index)
    share = np.zeros((n, n), dtype=np.float32)
    i, j = pairs["x"].map(index).to_numpy(), pairs["y"].map(index).to_numpy()
    v = (pairs["n_evidence"] / pairs["n_cooc"]).to_numpy(dtype=np.float32)
    share[i, j] = v
    share[j, i] = v
    return share


def feature_names() -> list[str]:
    return [f"{m}_{s}" for m in PAIR_MATRICES for s in ("min", "mean", "max")] + \
           ["log_docs_min", "log_docs_mean", "log_docs_max", "gen_score"]


def triad_features(codes: np.ndarray, gen_score: np.ndarray, mats: dict, log_docs: np.ndarray, n: int, chunk: int) -> np.ndarray:
    """Symmetric features of each triad code; the term order inside a triad does not matter."""
    out = np.empty((len(codes), len(feature_names())), dtype=np.float32)
    for start in range(0, len(codes), chunk):
        c = codes[start:start + chunk]
        i, j, k = c // (n * n), c // n % n, c % n
        cols = []
        for name in PAIR_MATRICES:
            m = mats[name]
            v = np.stack([m[i, j], m[i, k], m[j, k]], axis=1)
            cols += [v.min(axis=1), v.mean(axis=1), v.max(axis=1)]
        d = np.stack([log_docs[i], log_docs[j], log_docs[k]], axis=1)
        cols += [d.min(axis=1), d.mean(axis=1), d.max(axis=1), gen_score[start:start + chunk]]
        out[start:start + chunk] = np.column_stack(cols)
    return out


def completion_table(tp: str, study: dict, first: int | None = None) -> dict:
    """All completions with features and labels; `first` overrides the first-study candidate count."""
    gen = study["engine_reliability"]["conditional"]["generator"]
    sel = gen["selected"]
    g = recall.prepare(pd.read_parquet(HYPERGRAPH / f"{tp}_build.parquet"), tp, study["vocab"]["r5_rule"]["stability"])
    seeds = recall.seed_pairs(sel["seed_pairs"], g["counts"], g["stable"])
    codes, scores = recall.completions(seeds, {sel["score"]: g["scores"][sel["score"]]}, g["is_task"], g["closed"], g["n"])
    mats = {"log_count": np.log1p(g["counts"]).astype(np.float32), "npmi": g["scores"]["npmi_mean"][0].astype(np.float32),
            "adamic_adar": recall.adamic_adar_matrix(g["stable"]).astype(np.float32),
            "evidence_share": evidence_share(pd.read_parquet(PAIR_EVIDENCE / f"{tp}.parquet"), g["index"])}
    log_docs = np.log(np.maximum(g["docs"], 1)).astype(np.float32)
    chunk = study["engine_reliability"]["signals"]["learned_generator"]["chunk_rows"]
    x = triad_features(codes, scores[sel["score"]].astype(np.float32), mats, log_docs, g["n"], chunk)
    labels = pd.read_parquet(LABELS_V2 / f"{tp}.parquet")
    positives = recall.label_codes(labels[labels["label_v2"]], g["index"])
    if first is None:
        first = json.loads((RESULTS / f"candidates_main_{tp}.json").read_text())["stats"]["candidates"]
    return {"codes": codes, "gen_score": scores[sel["score"]], "x": x, "y": np.isin(codes, positives).astype(int),
            "positives": positives, "total": int(labels["label_v2"].sum()), "g": g, "labels": labels,
            "budgets": [m * first for m in gen["budget_multiples_of_first_candidates"]],
            "diagnosis_budget": gen["diagnosis_budget_multiple"] * first}


def save_candidates(tp: str, t: dict, learned: np.ndarray, commit: str | None) -> None:
    out = cands.attach_labels(cands.to_frame(t["g"], t["codes"], learned, t["diagnosis_budget"]), t["labels"])
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES / f"{tp}.parquet"
    out.to_parquet(path, index=False)
    summary = {"meta": {"step": "candidates_learned", "timepoint": tp, "created": datetime.date.today().isoformat(),
                        "git_commit": commit, "environment": runtime_environment(),
                        "output_sha256": recall.evaluate.sha256(path)},
               "budget": t["diagnosis_budget"], "candidates": int(len(out)), "kind": out["kind"].value_counts().to_dict(),
               "label_v2_positive": int(out["label_v2"].sum()), "realised": int(out["realised"].sum())}
    (RESULTS / f"candidates_learned_{tp}.json").write_text(json.dumps(summary, indent=2))


def compare(t: dict, learned: np.ndarray) -> dict:
    return {"generator": recall.recall_at(t["codes"], t["gen_score"], t["positives"], t["total"], t["budgets"]),
            "learned": recall.recall_at(t["codes"], learned, t["positives"], t["total"], t["budgets"])}


def main() -> None:
    study = load_study()
    er = study["engine_reliability"]
    cfg, model_cfg = er["signals"]["learned_generator"], study["model"]
    train_tp, eval_tp = er["conditional"]["generator"]["tune_timepoint"], er["conditional"]["generator"]["confirm_timepoint"]
    names = feature_names()
    tr = completion_table(train_tp, study)
    x_train = pd.DataFrame(tr["x"], columns=names)
    rounds, cv_ap, cv_sd = trainer.choose_rounds(x_train, tr["y"], model_cfg)
    print(f"rounds {rounds}, CV AP {cv_ap:.4f} +/- {cv_sd:.4f}")

    oof = np.zeros(len(tr["y"]))
    folds = StratifiedKFold(n_splits=cfg["oof_folds"], shuffle=True, random_state=model_cfg["seeds"][0])
    for fit, held in folds.split(x_train, tr["y"]):
        booster = trainer.train(x_train.iloc[fit], tr["y"][fit], rounds, model_cfg["seeds"][0], model_cfg)
        oof[held] = booster.predict(x_train.iloc[held])
    boosters = [trainer.train(x_train, tr["y"], rounds, s, model_cfg) for s in model_cfg["seeds"]]
    gain = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
    train_cmp = compare(tr, oof)
    commit = recall.evaluate.git_commit()
    save_candidates(train_tp, tr, oof, commit)
    del x_train, tr

    ev = completion_table(eval_tp, study)
    x_eval = pd.DataFrame(ev["x"], columns=names)
    learned = np.mean([b.predict(x_eval) for b in boosters], axis=0)
    eval_cmp = compare(ev, learned)
    save_candidates(eval_tp, ev, learned, commit)

    result = {
        "meta": {"step": "learned_generator", "created": datetime.date.today().isoformat(),
                 "git_commit": commit, "environment": runtime_environment(),
                 "signals_config": er["signals"], "generator": er["conditional"]["generator"], "model_config": model_cfg,
                 "features": names,
                 "note": "Descriptive development diagnostics; never a verdict. Choose on the train timepoint (out-of-fold), "
                         "confirm on the evaluation timepoint."},
        "model": {"rounds": rounds, "cv_average_precision_mean": cv_ap, "cv_average_precision_sd": cv_sd,
                  "importance_gain_mean": dict(sorted(zip(names, map(float, gain)), key=lambda kv: -kv[1]))},
        "train": {"timepoint": train_tp, "completions": int(len(oof)), "by_order": train_cmp},
        "eval": {"timepoint": eval_tp, "completions": int(len(ev["codes"])), "by_order": eval_cmp},
    }
    (RESULTS / "learned_generator.json").write_text(json.dumps(result, indent=2))
    for part in ("train", "eval"):
        for order, rows in result[part]["by_order"].items():
            print(part, f"{order:9s}", " ".join(f"{r['budget']}:{r['recall']:.3f}" for r in rows))


if __name__ == "__main__":
    main()
