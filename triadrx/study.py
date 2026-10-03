"""Study settings and timepoint windows read from config/study.json."""
import datetime
import json
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
