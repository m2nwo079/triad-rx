"""Report consistency tests (design 12): numbers only from result files, same values in both languages."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "templates").glob("report.*.md.j2"))
REQUIRED = ["validation_main", "applied_main", "dvf", "q1", "q2", "q1_cross_model", "rl_case"]

# Digits allowed outside template expressions: headings, list numbers, identifiers (Q1, V2, D3, R5),
# design section references (7.1절, 16절 11·14·15번, §16 items 11, 14 and 15, section 7)
ALLOWED = [
    r"^#.*$",
    r"^\s*\d+\.\s",
    r"(?<![A-Za-z0-9])[A-Za-z]+\d+[A-Za-z0-9]*(?![A-Za-z0-9])",
    r"\d+(?:\.\d+)*절(?:\s*\d+(?:·\d+)*번)?",
    r"\d+(?:·\d+)*번",
    r"§\d+(?:\.\d+)*(?:\s+items?\s+\d+(?:(?:,\s*|\s+and\s+)\d+)*)?",
    r"section \d+",
]


def strip_allowed(text: str) -> str:
    for pattern in ALLOWED:
        text = re.sub(pattern, " ", text, flags=re.MULTILINE)
    return text


def load_builder():
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts/report/build_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def results_present() -> bool:
    return all((ROOT / "results" / f"{name}.json").exists() for name in REQUIRED)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_have_no_literal_numbers(path):
    source = path.read_text()
    outside = re.sub(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", " ", source, flags=re.DOTALL)
    leftover = [line for line in strip_allowed(outside).splitlines() if re.search(r"\d", line)]
    assert not leftover, f"literal numbers in {path.name}: {leftover[:5]}"


@pytest.fixture(scope="module")
def rendered():
    if not results_present():
        pytest.skip("result files not present")
    builder = load_builder()
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return {lang: builder.render(lang) for lang in builder.LANGS}
    finally:
        os.chdir(cwd)


def test_every_number_comes_from_a_result_file(rendered):
    for lang, (text, records) in rendered.items():
        remaining = text
        for value_text in sorted({t for _, _, _, t in records}, key=len, reverse=True):
            remaining = remaining.replace(value_text, " ")
        leftover = [line for line in strip_allowed(remaining).splitlines() if re.search(r"\d", line)]
        assert not leftover, f"{lang}: numbers not read from result files: {leftover[:5]}"


def test_languages_use_the_same_values(rendered):
    sets = {lang: {(f, p) for f, p, _, _ in records} for lang, (_, records) in rendered.items()}
    ko, en = sets["ko"], sets["en"]
    assert ko == en, f"only in ko: {sorted(ko - en)[:5]}, only in en: {sorted(en - ko)[:5]}"


def test_recorded_values_match_result_files(rendered):
    from triadrx.report import Values
    fresh = Values(ROOT)
    for _, (_, records) in rendered.items():
        for name, path, value, _ in records:
            assert fresh.raw(name, path) == value


def test_prereg_changes_counts_by_area():
    import sys
    sys.path.insert(0, str(ROOT))
    from triadrx.report import prereg_changes
    head = "## 7. log\n| 날짜 | 영역 | 항목 | 결과 확인 후 변경 여부 | 커밋 |\n|---|---|---|---|---|\n"
    text = head + "| 2026-10-04 | 엔진 | a | 예 (after) | |\n| 2026-10-05 | 엔진 | b | 아니오 | |\n"
    out = prereg_changes(text)
    assert (out["rows"], out["after_seeing_results"]) == (2, 1)
    assert out["by_area"]["엔진"] == {"rows": 2, "after_seeing_results": 1}
    assert sum(a["rows"] for a in out["by_area"].values()) == out["rows"]
    with pytest.raises(ValueError):
        prereg_changes(head + "| 2026-10-04 | 기타 | a | 예 | |\n")
