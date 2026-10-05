"""Build the Korean and English reports from the result files (design 12).

Usage: python scripts/report/build_report.py

Renders templates/report.<lang>.md.j2 into reports/report.<lang>.md. Numbers enter
the text only through Values.v (triadrx.report), which records every value used;
the records are written to reports/report_values.json for the consistency tests.
V1/V2 verdicts are recomputed from the bootstrap lower bounds and must equal the
stored verdicts.
"""
import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.report import AREA_WORDS, VERDICT_WORDS, Values, verdicts

TEMPLATES = Path("templates")
OUT = Path("reports")
LANGS = ("ko", "en")
# Result files listed in the provenance appendix (each has meta.git_commit and meta.created)
REPORT_FILES = [
    "vocab_final_applied", "m1_T1", "validation_main", "validation_S1", "candidate_recall_main_T2",
    "applied_main", "candidates_main_applied", "evidence_papers", "evidence_patents", "evidence_repo_stars",
    "evidence_pypi_links", "evidence_pypi_downloads", "download_period", "dvf", "m2_final",
    "briefing_bundles", "briefing_generation", "briefing_checks", "briefing_checks_review",
    "q1", "q2", "q1_cross_model", "rl_case", "prospective_register",
]


def render(lang: str, root: Path = Path(".")) -> tuple[str, list]:
    values = Values(root)
    env = Environment(loader=FileSystemLoader(str(root / TEMPLATES)), undefined=StrictUndefined,
                      keep_trailing_newline=True, trim_blocks=False)
    template = env.get_template(f"report.{lang}.md.j2")
    text = template.render(V=values, verdicts=verdicts(values), report_files=REPORT_FILES, word=VERDICT_WORDS[lang],
                           area=AREA_WORDS[lang])
    return text, values.records


def main() -> None:
    OUT.mkdir(exist_ok=True)
    used = {}
    for lang in LANGS:
        text, records = render(lang)
        (OUT / f"report.{lang}.md").write_text(text)
        used[lang] = [{"file": f, "path": p, "text": t} for f, p, _, t in records]
        print(f"{lang}: {len(records)} values")
    (OUT / "report_values.json").write_text(json.dumps(used, indent=1, ensure_ascii=False))
    print("Saved", ", ".join(str(OUT / f"report.{lang}.md") for lang in LANGS), OUT / "report_values.json")


if __name__ == "__main__":
    main()
