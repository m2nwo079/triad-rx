"""T3 secondary analysis: recall of the learned third-term score at the verdict budget (engine preregistration 3.5).

Usage: python scripts/engine/16_t3_secondary.py   (after scripts/engine/15_t3_verdict.py)

Locked like the verdict. The learned score is trained as in development (scripts/engine/10_learned_generator.py,
reused through importlib): all completions of T1 with the second-study label and the first-study model
settings, mean over seeds. T3 pair combination evidence is computed from the T3 build window
(scripts/engine/09_pair_evidence.py). On T3, recall and precision of the learned order and of the nPMI
generator order are read at the verdict budget (verdict.budget_multiple_of_first_candidates times the T3
first-study candidate count) and at the development budget multiples. Reported only; never part of the verdict.

Outputs: data/derived/pair_evidence/main/T3.parquet (not committed), results/engine_secondary_T3.json
"""
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, require_t3_label_unlocked, runtime_environment

RESULTS = Path("results")
TRAIN_TP, TP = "T1", "T3"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pe = load("pair_evidence", ROOT / "scripts/engine/09_pair_evidence.py")
lg = load("learned_generator", ROOT / "scripts/engine/10_learned_generator.py")


def main() -> None:
    study = load_study()
    require_t3_label_unlocked(study)
    er = study["engine_reliability"]
    model_cfg = study["model"]
    if not (lg.LABELS_V2 / f"{TP}.parquet").exists():
        sys.exit("run scripts/engine/15_t3_verdict.py first: it builds the T3 labels")
    build = pd.read_parquet(Path("data/derived/hypergraph/main") / f"{TP}_build.parquet")
    lg.PAIR_EVIDENCE.mkdir(parents=True, exist_ok=True)
    pe.pair_table(TP, er["label_v2"], build).to_parquet(lg.PAIR_EVIDENCE / f"{TP}.parquet", index=False)

    names = lg.feature_names()
    tr = lg.completion_table(TRAIN_TP, study)
    x_train = pd.DataFrame(tr["x"], columns=names)
    rounds, _, _ = lg.trainer.choose_rounds(x_train, tr["y"], model_cfg)
    boosters = [lg.trainer.train(x_train, tr["y"], rounds, s, model_cfg) for s in model_cfg["seeds"]]
    del x_train, tr

    first = json.loads((RESULTS / f"engine_prep_{TP}.json").read_text())["first_study_candidates"]
    t3 = lg.completion_table(TP, study, first)
    verdict_budget = er["verdict"]["budget_multiple_of_first_candidates"] * first
    t3["budgets"] = sorted(set(t3["budgets"]) | {verdict_budget})
    learned = np.mean([b.predict(pd.DataFrame(t3["x"], columns=names)) for b in boosters], axis=0)
    result = {
        "meta": {"step": "engine_secondary_T3", "created": datetime.date.today().isoformat(),
                 "git_commit": lg.recall.evaluate.git_commit(), "environment": runtime_environment(),
                 "train_timepoint": TRAIN_TP, "rounds": rounds, "secondary": er["verdict_pipeline"]["secondary"],
                 "note": "Secondary analysis; reported, never part of the verdict."},
        "verdict_budget": verdict_budget, "completions": int(len(t3["codes"])), "label_v2_positive": t3["total"],
        "by_order": lg.compare(t3, learned),
    }
    (RESULTS / "engine_secondary_T3.json").write_text(json.dumps(result, indent=2))
    for order, rows in result["by_order"].items():
        print(order, " ".join(f"{r['budget']}:{r['recall']:.3f}" for r in rows))


if __name__ == "__main__":
    main()
