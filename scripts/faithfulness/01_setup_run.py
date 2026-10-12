"""Run folder of the faithfulness study (faithfulness preregistration 1, 4).

Usage: python scripts/faithfulness/01_setup_run.py   (from the repository root)

Locked: it stops unless the faithfulness preregistration records its freeze commit.
The first-study briefing, check and Q1 scripts run unchanged with faithfulness.run_dir as working directory:
- config/study.json: the config of the base run (faithfulness.base_run) with the prompt version, the business
  basis and the Q1 sampling of faithfulness swapped in; everything else, including model, temperature,
  repetitions, seed base and pause, stays as in the base run;
- links to the base run's bundles and bundle record (results/briefing_bundles.json), so generation reads the
  very same bundles and checks their hashes, and to shared scripts, prompts, templates and .env.
Outputs (briefings, raw generations, checks, Q1) stay in the run folder.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import engine_prereg_frozen, load_study

LINKS = ["scripts", "prompts", "templates", "vocab", ".env"]
BASE_LINKS = ["data/derived/briefing/bundles", "results/briefing_bundles.json"]
# The data cutoff is read from the arXiv filter summary in the working directory's results
RESULT_INPUTS = ["results/arxiv_filter_*.json"]


def require_frozen(study: dict) -> str:
    text = Path(study["faithfulness"]["prereg_file"]).read_text()
    if not engine_prereg_frozen(text):
        sys.exit("the faithfulness run is locked until its preregistration records its freeze commit")
    return re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M).group(1).strip()


def run_config(base: dict, cfg: dict) -> dict:
    """Base-run config with only the prompt version, the business basis and the Q1 sampling replaced."""
    out = json.loads(json.dumps(base))
    gen = out["applied"]["briefing"]["generation"]
    gen["prompt_version"] = cfg["generation"]["prompt_version"]
    gen["business_basis"] = cfg["generation"]["business_basis"]
    out["quality"]["q1"] = dict(cfg["q1"])
    out["run"] = {"study": "faithfulness", "base_config": f"{cfg['base_run']}/config/study.json"}
    return out


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and Path(os.readlink(link_path)) == target:
            return
        sys.exit(f"{link_path} exists and is not the expected link to {target}")
    link_path.symlink_to(target)


def main() -> None:
    study = load_study()
    freeze = require_frozen(study)
    cfg = study["faithfulness"]
    run, base = Path(cfg["run_dir"]), ROOT / cfg["base_run"]
    for name in ["prompts/frame_" + cfg["generation"]["prompt_version"] + ".txt",
                 "prompts/briefing_" + cfg["generation"]["prompt_version"] + ".txt"]:
        if not (ROOT / name).exists():
            sys.exit(f"{ROOT / name} is missing")
    for name in LINKS:
        link(ROOT / name, run / name)
    for name in BASE_LINKS:
        if not (base / name).exists():
            sys.exit(f"{base / name} is missing")
        link(base / name, run / name)
    for pattern in RESULT_INPUTS:
        files = sorted(glob.glob(str(ROOT / pattern)))
        if len(files) != 1:
            sys.exit(f"expected one file for {pattern}, found {len(files)}")
        link(Path(files[0]), run / "results" / Path(files[0]).name)
    (run / "config").mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads((base / "config/study.json").read_text())
    (run / "config/study.json").write_text(json.dumps(run_config(base_cfg, cfg), indent=2, ensure_ascii=False))
    print(f"run folder {run} (preregistration freeze {freeze}), bundles from {cfg['base_run']}")


if __name__ == "__main__":
    main()
