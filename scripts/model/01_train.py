"""Train the engine (P+H+T+R) and the V2 baseline (P+T+R) on T1 and score other timepoints.

Usage: python scripts/model/01_train.py [--variant main] [--score T2 applied]

The number of trees is chosen once per feature set by stratified cross-validation on T1
(average precision, early stopping); every other LightGBM setting is fixed in the config.
Scoring uses build-window features only, so no label of the scored timepoint is needed.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

FEATURES = Path("data/derived/features")
LABELS = Path("data/derived/labels")
OUT_DIR = Path("data/derived/scores")
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


def columns(cfg_features: dict, sets: list[str]) -> list[str]:
    return [c for s in sets for c in cfg_features["sets"][s]]


def choose_rounds(x: pd.DataFrame, y: np.ndarray, model_cfg: dict) -> tuple[int, float, float]:
    """Number of trees by stratified CV with early stopping on average precision."""
    params = {**model_cfg["params"], "objective": "binary", "metric": "average_precision",
              "verbosity": -1, "seed": model_cfg["seeds"][0], "deterministic": True}
    folds = StratifiedKFold(n_splits=model_cfg["cv_folds"], shuffle=True, random_state=model_cfg["seeds"][0])
    result = lgb.cv(params, lgb.Dataset(x, label=y), num_boost_round=model_cfg["n_estimators_max"],
                    folds=list(folds.split(x, y)),
                    callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"], verbose=False)])
    key = next(k for k in result if k.endswith("average_precision-mean"))
    best = len(result[key])
    return best, float(result[key][-1]), float(result[key.replace("-mean", "-stdv")][-1])


def train(x: pd.DataFrame, y: np.ndarray, rounds: int, seed: int, model_cfg: dict) -> lgb.Booster:
    params = {**model_cfg["params"], "objective": "binary", "verbosity": -1, "seed": seed, "deterministic": True}
    return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="main", choices=["main", "S1"])
    parser.add_argument("--score", nargs="+", default=["T2", "applied"])
    args = parser.parse_args()
    variant = args.variant

    study = load_study()
    model_cfg, feat_cfg = study["model"], study["features"]
    feat_path = FEATURES / variant / "T1.parquet"
    label_path = LABELS / variant / "T1.parquet"
    train_df = pd.read_parquet(feat_path).merge(pd.read_parquet(label_path)[KEYS + ["label"]], on=KEYS, validate="one_to_one")
    y = train_df["label"].to_numpy()

    models = {"engine": model_cfg["engine_sets"], "baseline": model_cfg["v2_baseline_sets"]}
    summary, boosters = {}, {}
    for name, sets in models.items():
        cols = columns(feat_cfg, sets)
        rounds, cv_ap, cv_sd = choose_rounds(train_df[cols], y, model_cfg)
        boosters[name] = [train(train_df[cols], y, rounds, s, model_cfg) for s in model_cfg["seeds"]]
        gain = np.mean([b.feature_importance(importance_type="gain") for b in boosters[name]], axis=0)
        summary[name] = {"sets": sets, "features": cols, "rounds": rounds, "cv_average_precision_mean": cv_ap,
                         "cv_average_precision_sd": cv_sd,
                         "importance_gain_mean": dict(sorted(zip(cols, map(float, gain)), key=lambda kv: -kv[1]))}
        print(f"{name}: rounds {rounds}, CV AP {cv_ap:.4f} +/- {cv_sd:.4f}")

    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = {}
    for tp in args.score:
        path = FEATURES / variant / f"{tp}.parquet"
        feats = pd.read_parquet(path)
        out = feats[KEYS].copy()
        for name, sets in models.items():
            cols = columns(feat_cfg, sets)
            preds = np.column_stack([b.predict(feats[cols]) for b in boosters[name]])
            for i, seed in enumerate(model_cfg["seeds"]):
                out[f"{name}_seed{seed}"] = preds[:, i]
            # Judgement uses the mean over seeds (preregistration 4.2)
            out[f"{name}_mean"] = preds.mean(axis=1)
        out.to_parquet(out_dir / f"{tp}.parquet")
        scored[tp] = {"candidates": int(len(out)), "features_sha256": sha256(path)}
        print(f"Saved scores for {tp}: {len(out)} candidates")

    result = {
        "meta": {"step": "train", "variant": variant, "created": datetime.date.today().isoformat(),
                 "git_commit": git_commit(), "model_config": model_cfg, "lightgbm": lgb.__version__,
                 "train_features_sha256": sha256(feat_path), "train_labels_sha256": sha256(label_path)},
        "train": {"candidates": int(len(y)), "positives": int(y.sum())},
        "models": summary,
        "scored": scored,
    }
    (RESULTS / f"model_{variant}.json").write_text(json.dumps(result, indent=2))
    print("Saved", RESULTS / f"model_{variant}.json")


if __name__ == "__main__":
    main()
