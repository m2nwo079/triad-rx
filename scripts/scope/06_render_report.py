"""Render the stage 1 scope report from the results JSON and the study config."""
import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path("templates")
REPORTS = Path("reports")
STUDY = Path("config/study.json")


def run_checks(result: dict, study: dict) -> dict:
    arxiv = result["arxiv_counts_by_year"]
    years_needed = set()
    for t in study["timepoints"].values():
        years_needed.update(range(t["build"][0], t["label"][1] + 1))
    return {
        "arXiv: no missing years": all(n is not None for n in arxiv.values()),
        "arXiv: all timepoint years present": all(str(y) in arxiv for y in years_needed),
        "Result records a source commit": bool(result["meta"]["git_commit"]),
        "Timepoints: buffer between build and label": all(
            t["build"][1] < t["buffer"][0] <= t["buffer"][1] < t["label"][0]
            for t in study["timepoints"].values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    result = json.loads(args.results.read_text())
    study = json.loads(STUDY.read_text())
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    text = env.get_template("scope_report.md.j2").render(
        r=result,
        study=study,
        results_path=args.results.as_posix(),
        checks=run_checks(result, study),
    )
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{args.results.stem}.md"
    out.write_text(text)
    print("Saved", out)


if __name__ == "__main__":
    main()
