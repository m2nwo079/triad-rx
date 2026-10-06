"""T3 windows and the engine preregistration lock (design 18.2, 18.3)."""
import datetime
from pathlib import Path

import pytest

from triadrx.study import engine_prereg_frozen, label_window, load_study, require_t3_label_unlocked, t3_window

ROOT = Path(__file__).resolve().parents[1]
UTC = datetime.timezone.utc
CUTOFF = datetime.datetime(2026, 9, 24, 17, 59, 54, tzinfo=UTC)
STUDY = {
    "timepoints": {"T1": {"label": [2020, 2021]}},
    "engine_reliability": {"timepoints": {"T3": {"build": [2019, 2023], "buffer": [2024, 2024],
                                                  "label_start": 2025, "label_end": "data_cutoff"}}},
}


def test_t3_build_window_covers_whole_years():
    start, end = t3_window("build", STUDY)
    assert start == datetime.datetime(2019, 1, 1, tzinfo=UTC)
    assert end == datetime.datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)


def test_t3_label_window_ends_at_cutoff():
    start, end = t3_window("label", STUDY, CUTOFF)
    assert start == datetime.datetime(2025, 1, 1, tzinfo=UTC)
    assert end == CUTOFF


def test_t3_unknown_window():
    with pytest.raises(ValueError):
        t3_window("buffer", STUDY)


def test_first_study_label_window_cannot_reach_t3():
    with pytest.raises(ValueError):
        label_window("T3", STUDY)


@pytest.mark.parametrize("text,frozen", [
    ("고정 커밋: 미정\n", False),
    ("고정 커밋:\n", False),
    ("no freeze line\n", False),
    ("intro\n고정 커밋: abc1234\n", True),
])
def test_engine_prereg_frozen(text, frozen):
    assert engine_prereg_frozen(text) is frozen


def test_t3_label_guard(tmp_path):
    study = {"engine_reliability": {"prereg_file": "prereg.md"}}
    (tmp_path / "prereg.md").write_text("고정 커밋: 미정\n")
    with pytest.raises(SystemExit):
        require_t3_label_unlocked(study, tmp_path)
    (tmp_path / "prereg.md").write_text("고정 커밋: abc1234\n")
    require_t3_label_unlocked(study, tmp_path)


def test_t3_label_guard_opens_after_the_freeze():
    require_t3_label_unlocked(load_study(ROOT / "config/study.json"), ROOT)


def test_engine_preregistration_is_frozen_and_config_unchanged_since():
    """The freeze line names a real commit, and the engine settings still equal those at that commit."""
    import json
    import re
    import subprocess
    study = load_study(ROOT / "config/study.json")
    text = (ROOT / study["engine_reliability"]["prereg_file"]).read_text()
    assert engine_prereg_frozen(text)
    commit = re.search(r"^고정 커밋:\s*(\S+)", text, flags=re.M).group(1)
    frozen = subprocess.run(["git", "show", f"{commit}:config/study.json"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert json.loads(frozen)["engine_reliability"] == study["engine_reliability"]


def test_config_t3_label_does_not_overlap_dev_labels():
    """Development label windows (T1, T2) end before T3's label window starts."""
    study = load_study(ROOT / "config/study.json")
    t3_start = study["engine_reliability"]["timepoints"]["T3"]["label_start"]
    assert all(tp["label"][1] < t3_start for tp in study["timepoints"].values())


def test_t3_training_labels_end_by_decision_time():
    """Training label windows must end by the end of T3's build window (design 18.4)."""
    study = load_study(ROOT / "config/study.json")
    er = study["engine_reliability"]
    decision_year = er["timepoints"]["T3"]["build"][1]
    for tp in er["training"]["T3"]:
        spec = er["timepoints"].get(tp) or study["timepoints"][tp]
        assert spec["label"][1] <= decision_year, tp
    assert "T2" not in er["training"]["T3"]
