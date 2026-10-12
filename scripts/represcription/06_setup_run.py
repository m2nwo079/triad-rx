"""Run folder for the first-study downstream steps of the represcription (represcription preregistration 1, 5).

Usage: python scripts/represcription/06_setup_run.py   (from the repository root)

The first-study evidence, DVF, briefing and quality scripts read and write paths relative to the working
directory, including config/study.json. They run unchanged with represcription.run_dir as working directory:
- config/study.json: the repository config with only the fourth-study values replaced (prospective registration,
  baseline, ranking and vocabulary files; Q1 sampling from represcription.q1);
- links to shared inputs (scripts, prompts, templates, vocabularies, arXiv, patent and Papers with Code data, .env)
  and to result files that are inputs (arXiv filter summary for the data cutoff, patent file inventory);
- data/derived/hypergraph/main/applied_build.parquet linked to the applied v3 build hypergraph, where the
  first-study steps read the applied hypergraph.
Outputs of the downstream steps stay in the run folder. Running this again rewrites the config and keeps links.
"""
import glob
import json
import os
import sys
from pathlib import Path

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

LINKS = ["scripts", "prompts", "templates", "vocab", "data/arxiv", "data/patents", "data/scope", ".env"]
RESULT_INPUTS = ["results/arxiv_filter_*.json", "results/patent_files.json"]
HYPERGRAPH_LINK = ("data/derived/hypergraph/main/applied_build.parquet", "data/derived/hypergraph/v3/applied_build.parquet")


def run_config(study: dict, registration: str, baseline: str) -> dict:
    """Repository config with the fourth-study files and Q1 sampling swapped in."""
    cfg = json.loads(json.dumps(study))
    cfg["applied"]["dvf"]["prospective_file"] = registration
    cfg["prospective"]["registration"] = registration
    cfg["prospective"]["baseline"]["file"] = baseline
    cfg["prospective"]["ranking"] = "data/derived/applied/ranking.parquet"
    cfg["prospective"]["vocab"] = "vocab/timepoints/applied_v3_final.json"
    cfg["quality"]["q1"] = dict(study["represcription"]["q1"])
    cfg["run"] = {"study": "represcription", "base_config": "config/study.json of the repository root"}
    return cfg


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and Path(os.readlink(link_path)) == target:
            return
        sys.exit(f"{link_path} exists and is not the expected link to {target}")
    link_path.symlink_to(target)


def main() -> None:
    study = load_study()
    run = Path(study["represcription"]["run_dir"])
    registrations = sorted(p for p in (run / "prospective").glob("*.json") if not p.stem.endswith("_baseline"))
    if len(registrations) != 1:
        sys.exit(f"expected one registration file in {run / 'prospective'}, found {len(registrations)}")
    registration = registrations[0].relative_to(run)
    baseline = registration.with_name(registration.stem + "_baseline.json")
    if not (run / baseline).exists():
        sys.exit(f"{run / baseline} is missing")

    for name in LINKS:
        if not (ROOT / name).exists():
            sys.exit(f"{ROOT / name} is missing")
        link(ROOT / name, run / name)
    for pattern in RESULT_INPUTS:
        files = sorted(glob.glob(str(ROOT / pattern)))
        if len(files) != 1:
            sys.exit(f"expected one file for {pattern}, found {len(files)}")
        link(Path(files[0]), run / "results" / Path(files[0]).name)
    link(ROOT / HYPERGRAPH_LINK[1], run / HYPERGRAPH_LINK[0])

    (run / "config").mkdir(parents=True, exist_ok=True)
    (run / "config/study.json").write_text(json.dumps(run_config(study, str(registration), str(baseline)), indent=2, ensure_ascii=False))
    print(f"run folder {run}: registration {registration}, baseline {baseline}")


if __name__ == "__main__":
    main()
