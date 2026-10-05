"""Report generation helpers (design 12).

Every number in a report comes from a result file through `Values.v(file, path)`.
Each call is recorded (file, path, raw value, rendered text) so tests can check
that the report contains no number that was not read from a result file, and that
the Korean and English reports use the same values.

Sources: results/<name>.json, config/study.json ("config"), the prospective
registration file ("prospective"), and "derived" values computed here from those
files by named functions (for example counts from the preregistration change log).
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

RESULTS = Path("results")
CONFIG = Path("config/study.json")
PREREG = Path("preregistration.md")
# Descriptive DVF stability thresholds (design 7.3, 16 item 23): a riskiest-axis designation is
# called stable when the bootstrap picks the same axis in at least this share of replicates, and an
# axis-score interval is called wide when it spans at least this many score levels
STABLE_SHARE = 0.95
WIDE_LEVELS = 2


def _resolve(obj, path: str):
    for part in path.split(".") if path else []:
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def fmt(value, style: str) -> str:
    if value is None:
        return "—"
    if style == "int":
        return f"{int(value):,}"
    if style == "d1":
        return f"{float(value):.1f}"
    if style == "d2":
        return f"{float(value):.2f}"
    if style == "d3":
        return f"{float(value):.3f}"
    if style == "pct1":
        return f"{100 * float(value):.1f}%"
    if style == "short":
        return str(value)[:7]
    if style == "date":
        return str(value)[:10]
    if style == "triad":
        return " · ".join(t.split(":", 1)[1] for t in value)
    if style == "join":
        return "+".join(str(x) for x in value)
    if style == "str":
        return str(value)
    raise ValueError(f"unknown format {style!r}")


class Values:
    def __init__(self, root: Path = Path(".")):
        self.root = root
        self.cache: dict[str, object] = {}
        self.records: list[tuple[str, str, object, str]] = []

    def load(self, name: str):
        if name not in self.cache:
            if name == "config":
                self.cache[name] = json.loads((self.root / CONFIG).read_text())
            elif name == "prospective":
                rel = json.loads((self.root / CONFIG).read_text())["applied"]["dvf"]["prospective_file"]
                self.cache[name] = json.loads((self.root / rel).read_text())
            elif name == "derived":
                self.cache[name] = derived(self)
            else:
                self.cache[name] = json.loads((self.root / RESULTS / f"{name}.json").read_text())
        return self.cache[name]

    def raw(self, name: str, path: str):
        """Value without rendering (for conditions and loops); not printed."""
        return _resolve(self.load(name), path)

    def v(self, name: str, path: str, style: str = "str") -> str:
        value = self.raw(name, path)
        text = fmt(value, style)
        self.records.append((name, path, value, text))
        return text

    def ci(self, name: str, lower: str, upper: str, style: str = "d3") -> str:
        return f"[{self.v(name, lower, style)}, {self.v(name, upper, style)}]"


def change_rows(text: str) -> list[list[str]]:
    """Cells of the dated rows in the change log (preregistration section 7), in log order."""
    section = text.split("## 7.", 1)[1]
    lines = [line for line in section.splitlines() if re.match(r"^\| \d{4}-\d{2}-\d{2} \|", line)]
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]


def prereg_changes(text: str) -> dict:
    """Rows of the change log and how many were made after seeing results (last-but-one
    column starting with 예), in total and per area (second column, one of AREA_WORDS)."""
    rows = change_rows(text)
    seen = 0
    by_area = {area: {"rows": 0, "after_seeing_results": 0} for area in AREA_WORDS["ko"]}
    for cells in rows:
        if cells[1] not in by_area:
            raise ValueError(f"unknown change-log area {cells[1]!r}")
        after = cells[-2].startswith("예")
        by_area[cells[1]]["rows"] += 1
        by_area[cells[1]]["after_seeing_results"] += after
        seen += after
    return {"rows": len(rows), "after_seeing_results": seen, "by_area": by_area}


def rows_at_freeze(root: Path, text: str) -> int:
    """Change-log rows already present in preregistration.md at the freeze commit (from git)."""
    freeze = re.search(r"^고정 커밋:\s*(\S+)", text, flags=re.M).group(1)
    frozen = subprocess.run(["git", "show", f"{freeze}:{PREREG}"], cwd=root, capture_output=True, text=True, check=True).stdout
    return len(change_rows(frozen))


def verdicts(values: Values) -> dict:
    """V1/V2 recomputed from the bootstrap lower bounds (design 10.3); both methods must pass."""
    out = {}
    for variant in ("validation_main", "validation_S1"):
        ci = values.raw(variant, "ci")
        v1_by = {m: ci[m]["lift"]["lower"] > 1 for m in ("candidate", "term_block")}
        v2_by = {m: ci[m]["auprc_diff"]["lower"] > 0 for m in ("candidate", "term_block")}
        v1, v2 = all(v1_by.values()), all(v2_by.values())
        stored = values.raw(variant, "verdict")
        if (v1, v2) != (stored["V1"], stored["V2"]):
            raise ValueError(f"{variant}: recomputed verdicts {v1, v2} differ from the stored {stored}")
        out[variant] = {"V1": v1, "V2": v2, "V1_by": v1_by, "V2_by": v2_by}
    return out


def derived(values: Values) -> dict:
    prereg_text = (values.root / PREREG).read_text()
    changes = prereg_changes(prereg_text)
    # Rows are appended in order, so the rows after the first n_frozen were added after the freeze
    n_frozen = rows_at_freeze(values.root, prereg_text)
    post_after = sum(1 for cells in change_rows(prereg_text)[n_frozen:] if cells[-2].startswith("예"))
    study = json.loads((values.root / CONFIG).read_text())
    prospective = study["prospective"]
    horizon = prospective["buffer_years"] + prospective["label_years"]
    cutoff = datetime.datetime.fromisoformat(values.raw("applied_main", "meta.data_cutoff"))
    try:
        verdict_date = cutoff.replace(year=cutoff.year + horizon)
    except ValueError:
        verdict_date = cutoff.replace(year=cutoff.year + horizon, day=28)
    dvf = values.raw("dvf", "candidates")
    rl = values.raw("rl_case", "rl_candidates")
    q1 = values.raw("q1", "by_field")
    q1_rate = values.raw("q1", "rate")

    def field_rate(name: str) -> float | None:
        entry = q1.get(name)
        return entry["supported"] / entry["n"] if entry and entry["n"] else None

    summary_rates = [r for r in (field_rate("concept_frame"),) if r is not None]
    synthesis_rates = [r for r in (field_rate("target_user"), field_rate("value_prop")) if r is not None]
    deep = [c for c in rl if c["deep_dive"] and c["q1_sample"]["sentences"]]
    full = [c for c in dvf if all(a["score"] is not None for a in c["axes"].values())]
    unstable = [c for c in full if c["riskiest_bootstrap_share"][c["riskiest_axis"]] < STABLE_SHARE]
    intervals = [a["bootstrap_score_95"] for c in dvf for a in c["axes"].values() if a["bootstrap_score_95"]]
    briefing_top = study["applied"]["briefing"]["top_n"]
    return {
        # Conditions for interpretive sentences; they are not printed as numbers
        "synthesis_fields_lower": bool(summary_rates and synthesis_rates and max(synthesis_rates) < min(summary_rates)),
        "deep_dive_less_faithful": bool(deep) and all(c["q1_sample"]["supported"] / c["q1_sample"]["sentences"] < q1_rate for c in deep),
        "prereg_changes": changes["rows"],
        "prereg_rows_at_freeze": n_frozen,
        "prereg_rows_after_freeze": changes["rows"] - n_frozen,
        "prereg_after_freeze_after_results": post_after,
        "prospective_horizon": horizon,
        "prospective_verdict_date": verdict_date.date().isoformat(),
        "prereg_changes_after_results": changes["after_seeing_results"],
        "prereg_changes_by_area": changes["by_area"],
        "rl_candidates": len(rl),
        "rl_deep_dive": sum(1 for c in rl if c["deep_dive"]),
        "dvf_candidates": len(dvf),
        "dvf_stable_share": STABLE_SHARE,
        "dvf_full_axes": len(full),
        "dvf_unstable_riskiest": len(unstable),
        "dvf_unstable_briefing": sum(1 for c in unstable if c["rank"] <= briefing_top),
        "dvf_min_riskiest_share": min((c["riskiest_bootstrap_share"][c["riskiest_axis"]] for c in full), default=None),
        "dvf_wide_levels": WIDE_LEVELS,
        "dvf_axis_scores": len(intervals),
        "dvf_axis_wide": sum(1 for lo, hi in intervals if hi - lo >= WIDE_LEVELS),
    }


# Display words for verdict values stored in result files (same value, language-specific word)
VERDICT_WORDS = {
    "ko": {"pass": "합격", "conditional": "조건부 합격", "fail": "불합격"},
    "en": {"pass": "pass", "conditional": "conditional pass", "fail": "fail"},
}

# Areas of the preregistration change log, in report order (Korean key as written in the log)
AREA_WORDS = {
    "ko": {"어휘·매칭": "어휘·매칭", "엔진": "엔진", "적용 모드·근거": "적용 모드·근거",
           "DVF·출원인": "DVF·출원인", "브리핑·품질": "브리핑·품질"},
    "en": {"어휘·매칭": "Vocabulary and matching", "엔진": "Engine", "적용 모드·근거": "Applied mode and evidence",
           "DVF·출원인": "DVF and assignees", "브리핑·품질": "Briefings and quality"},
}
