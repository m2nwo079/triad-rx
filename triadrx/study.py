"""Study settings and timepoint windows read from config/study.json."""
import datetime
import json
import re
from pathlib import Path

STUDY = Path("config/study.json")
FILTER_RESULTS = Path("results")


def load_study(path: Path = STUDY) -> dict:
    return json.loads(path.read_text())


def data_cutoff(results_dir: Path = FILTER_RESULTS) -> datetime.datetime:
    """Collection cutoff (latest v1 submission) recorded by the arXiv filter step."""
    files = sorted(results_dir.glob("arxiv_filter_*.json"))
    if len(files) != 1:
        raise SystemExit(f"Expected one arxiv_filter_*.json in {results_dir}, found {len(files)}")
    meta = json.loads(files[0].read_text())["meta"]
    return datetime.datetime.fromisoformat(meta["data_cutoff"])


def build_window(timepoint: str, study: dict, cutoff: datetime.datetime | None = None):
    """Inclusive build window as timezone-aware datetimes (start, end).

    T1 and T2 use calendar years from the config. The applied mode ends at the data
    cutoff and starts the configured number of years earlier (design section 4.4).
    """
    utc = datetime.timezone.utc
    if timepoint in study["timepoints"]:
        first, last = study["timepoints"][timepoint]["build"]
        start = datetime.datetime(first, 1, 1, tzinfo=utc)
        end = datetime.datetime(last, 12, 31, 23, 59, 59, tzinfo=utc)
        return start, end
    if timepoint == "applied":
        if cutoff is None:
            raise ValueError("applied mode needs the data cutoff")
        years = study["applied"]["build_years"]
        start = cutoff.replace(year=cutoff.year - years)
        return start, cutoff
    raise ValueError(f"unknown timepoint {timepoint!r}")


def label_window(timepoint: str, study: dict):
    """Inclusive label window for T1 or T2 as timezone-aware datetimes (start, end)."""
    if timepoint not in study["timepoints"]:
        raise ValueError(f"{timepoint!r} has no label window")
    utc = datetime.timezone.utc
    first, last = study["timepoints"][timepoint]["label"]
    return datetime.datetime(first, 1, 1, tzinfo=utc), datetime.datetime(last, 12, 31, 23, 59, 59, tzinfo=utc)


def t3_window(kind: str, study: dict, cutoff: datetime.datetime | None = None):
    """Inclusive T3 window of the engine reliability study (design 18.2) as (start, end).

    The build window covers whole calendar years; the label window runs from the start of
    label_start to the data cutoff. T3 lives in engine_reliability, not in timepoints, so the
    first-study scripts cannot reach it.
    """
    utc = datetime.timezone.utc
    t3 = study["engine_reliability"]["timepoints"]["T3"]
    if kind == "build":
        first, last = t3["build"]
        return datetime.datetime(first, 1, 1, tzinfo=utc), datetime.datetime(last, 12, 31, 23, 59, 59, tzinfo=utc)
    if kind == "label":
        if t3["label_end"] != "data_cutoff":
            raise ValueError(f"unsupported T3 label_end {t3['label_end']!r}")
        end = cutoff if cutoff is not None else data_cutoff()
        return datetime.datetime(t3["label_start"], 1, 1, tzinfo=utc), end
    raise ValueError(f"unknown T3 window {kind!r}")


def engine_prereg_frozen(text: str) -> bool:
    """True once the engine preregistration records its freeze commit (design 18.3)."""
    match = re.search(r"^고정 커밋:\s*(.+)$", text, flags=re.M)
    return bool(match) and match.group(1).strip() not in {"", "미정"}


def require_t3_label_unlocked(study: dict, root: Path = Path(".")) -> None:
    """Stop unless the engine preregistration is frozen; every T3 label step calls this first."""
    prereg = root / study["engine_reliability"]["prereg_file"]
    if not engine_prereg_frozen(prereg.read_text()):
        raise SystemExit(f"T3 label window is locked until {prereg} records its freeze commit")


def runtime_environment() -> dict:
    """Python, platform and library versions, recorded because results may come from a remote run."""
    import platform
    from importlib import metadata

    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
