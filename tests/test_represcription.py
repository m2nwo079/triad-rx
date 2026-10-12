"""Fourth-study (represcription) steps on toy data."""
import datetime
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ar = load("applied_rebuild", "scripts/represcription/01_applied_rebuild.py")


def test_applied_mode_is_added_as_a_build_only_timepoint_without_changing_the_input():
    study = json.loads((ROOT / "config/study.json").read_text())
    out = ar.with_applied(study)
    assert out["matching"]["timepoints"][-1] == "applied" and out["timepoints"]["applied"] == {"role": "apply"}
    assert "applied" not in study["matching"]["timepoints"] and "applied" not in study["timepoints"]


def test_applied_window_ends_at_the_data_cutoff(tmp_path, monkeypatch):
    cutoff = datetime.datetime(2026, 9, 24, 17, 0, tzinfo=datetime.timezone.utc)
    dates = pd.to_datetime(["2021-09-24T16:00Z", "2021-09-24T18:00Z", "2026-09-24T16:00Z", "2026-09-25T00:00Z"])
    papers = pd.DataFrame({"id": ["a", "b", "c", "d"], "title": "t", "abstract": "x", "first_date": dates})
    path = tmp_path / "papers.parquet"
    papers.to_parquet(path)
    monkeypatch.setattr(ar.rebuild.sample, "PAPERS", path)
    monkeypatch.setattr(ar, "data_cutoff", lambda: cutoff)
    study = {"applied": {"build_years": 5}, "timepoints": {}, "matching": {"timepoints": []}}
    assert ar.applied_window_papers(study)["id"].tolist() == ["b", "c"]
    # The rebuild sees the study with the added `applied` entry; the window must stay the same
    assert ar.applied_window_papers(ar.with_applied(study))["id"].tolist() == ["b", "c"]


def test_represcription_config_points_to_existing_settings():
    study = json.loads((ROOT / "config/study.json").read_text())
    cfg = study["represcription"]
    assert cfg["engine"]["training"] == ["T1", "R2", "R3", "T2", "T3"]
    assert cfg["engine"]["budget_multiple"] == study["matching"]["verdict"]["budget_multiple"]
    assert cfg["q1"]["seed"] != study["quality"]["q1"]["seed"]
    assert (ROOT / cfg["prereg_file"]).exists()


rs = load("review_sheets", "scripts/represcription/02_review_sheets.py")


def test_review_items_are_limited_to_new_terms_and_skip_decided_ones():
    terms = [{"id": "M:old", "surfaces": [{"text": "attention", "match": "ci"}]},
             {"id": "M:new", "surfaces": [{"text": "mixture", "match": "ci"}, {"text": "MoX", "match": "cs"},
                                          {"text": "gated mixture", "match": "ci"}]},
             {"id": "T:new2", "surfaces": [{"text": "routing", "match": "ci"}]}]
    words = {"attention", "mixture", "route"}
    items = rs.common_word_items(terms, {"M:new", "T:new2"}, words, [["ing", ""], ["ing", "e"]], {("T:new2", "routing")})
    assert items == [("M:new", "mixture")]
    found = {("M:a", "M:new"): {}, ("M:a", "M:b"): {}, ("M:new", "T:new2"): {}, ("M:c", "T:new2"): {}}
    assert rs.synonym_items(found, {"M:new", "T:new2"}, {("M:c", "T:new2")}) == [("M:a", "M:new"), ("M:new", "T:new2")]


def test_review_config_keeps_the_third_study_rules_and_new_files():
    study = json.loads((ROOT / "config/study.json").read_text())
    review = study["represcription"]["review"]
    assert review["common_words_file"] != study["matching"]["common_words"]["review_file"]
    assert review["synonyms_file"] != study["matching"]["synonyms"]["review_file"]


rd = load("review_decisions", "scripts/represcription/03_review_decisions.py")


def test_new_same_pairs_join_existing_groups_and_keep_their_names():
    names = {"M:a": "alpha", "M:b": "beta", "M:c": "gamma", "M:d": "delta", "M:e": "epsilon"}
    groups = {"M:a": ["M:a", "M:b"]}
    out = rd.combined_groups(groups, [("M:b", "M:c"), ("M:e", "M:d")], names)
    assert out == {"M:a": ["M:a", "M:b", "M:c"], "M:d": ["M:d", "M:e"]}
    assert rd.combined_groups(groups, [], names) == groups


def test_review_files_must_keep_fixed_columns(tmp_path):
    original = tmp_path / "s_original.csv"
    review = tmp_path / "s.csv"
    original.write_text("review_id,term_a,term_b,decision,note\nsya-000,M:a,M:b,,\n", encoding="utf-8-sig")
    review.write_text("review_id,term_a,term_b,decision,note\nsya-000,M:a,M:b,same,\n", encoding="utf-8-sig")
    out, _ = rd.checked(review, ["same", "hierarchy", "different"], rd.syd.validate)
    assert out["decision"].tolist() == ["same"]
    review.write_text("review_id,term_a,term_b,decision,note\nsya-000,M:a,M:c,same,\n", encoding="utf-8-sig")
    import pytest
    with pytest.raises(SystemExit):
        rd.checked(review, ["same", "hierarchy", "different"], rd.syd.validate)


ei = load("engine_inputs", "scripts/represcription/04_engine_inputs.py")
rk = load("ranking", "scripts/represcription/05_ranking.py")


def test_applied_recent_window_is_the_last_years_before_the_cutoff():
    end = datetime.datetime(2026, 9, 24, 17, 59, 54, tzinfo=datetime.timezone.utc)
    assert ei.recent_start_applied(end, 2) == datetime.datetime(2024, 9, 24, 17, 59, 54, tzinfo=datetime.timezone.utc)


def test_ranking_is_locked_until_the_freeze_commit(tmp_path):
    import pytest
    prereg = tmp_path / "p.md"
    prereg.write_text("x\n고정 커밋: 미정\n")
    with pytest.raises(SystemExit):
        rk.require_frozen({"represcription": {"prereg_file": str(prereg)}})
    prereg.write_text("x\n고정 커밋: abc1234\n")
    assert rk.require_frozen({"represcription": {"prereg_file": str(prereg)}}) == "abc1234"


def test_prescription_ranks_break_ties_by_term_ids_and_skip_redundant():
    applied = pd.DataFrame({"a": ["M:b", "M:a", "M:c", "M:d"], "b": ["T:x"] * 4, "c": ["T:y"] * 4,
                            "score": [0.5, 0.5, 0.9, 0.1], "redundant": ["", "", "D1", ""]})
    out = rk.prescription_ranks(applied).set_index("a")
    assert out["rank_engine"].to_dict() == {"M:a": 2, "M:b": 3, "M:c": 1, "M:d": 4}
    assert out["rank_prescription"].to_dict()["M:a"] == 1 and pd.isna(out["rank_prescription"]["M:c"])
    assert out["rank_prescription"]["M:d"] == 3


def test_redundancy_flags_same_names_and_uses_candidate_quantile():
    import numpy as np
    keys = {"M:a": "batch norm", "M:b": "batchnorm", "T:x": "image classification", "M:c": "dropout"}
    vectors = {"M:a": np.array([1.0, 0.0]), "M:b": np.array([0.0, 1.0]), "T:x": np.array([1.0, 0.0]), "M:c": np.array([0.0, 1.0])}
    applied = pd.DataFrame({"a": ["M:a", "M:a"], "b": ["M:b", "M:c"], "c": ["T:x", "T:x"]})
    reasons, threshold = rk.redundancy(applied, keys, vectors, 0.985)
    assert reasons[0].startswith("D1") and threshold <= 1.0


su = load("setup_run", "scripts/represcription/06_setup_run.py")
pt = load("patents_v3", "scripts/represcription/07_patents.py")


def test_run_config_swaps_only_the_fourth_study_files_and_q1():
    study = json.loads((ROOT / "config/study.json").read_text())
    out = su.run_config(study, "prospective/2026-10-11.json", "prospective/2026-10-11_baseline.json")
    assert out["applied"]["dvf"]["prospective_file"] == "prospective/2026-10-11.json" == out["prospective"]["registration"]
    assert out["prospective"]["vocab"] == "vocab/timepoints/applied_v3_final.json"
    assert out["quality"]["q1"]["seed"] == study["represcription"]["q1"]["seed"]
    changed = {k for k in study if study[k] != out[k]}
    assert changed == {"applied", "prospective", "quality"} and "run" in out
    assert out["applied"]["briefing"] == study["applied"]["briefing"] and out["evaluation"] == study["evaluation"]


def test_links_are_created_once_and_wrong_files_stop_the_setup(tmp_path):
    import pytest
    target = tmp_path / "target"
    target.mkdir()
    su.link(target, tmp_path / "run" / "x")
    su.link(target, tmp_path / "run" / "x")
    (tmp_path / "run" / "y").write_text("not a link")
    with pytest.raises(SystemExit):
        su.link(target, tmp_path / "run" / "y")


def test_refined_matcher_gives_the_same_terms_as_refined_paper_matching():
    from triadrx.corpus import match_papers_refined
    study = json.loads((ROOT / "config/study.json").read_text())
    rules = study["matching"]["surface_rules"]
    terms = [{"id": "M:bert", "type": "M", "key": "bert", "status": "active",
              "surfaces": [{"text": "BERT", "match": "cs"}, {"text": "bidirectional encoder representations", "match": "ci"}]},
             {"id": "T:text classification", "type": "T", "key": "text classification", "status": "active",
              "surfaces": [{"text": "text classification", "match": "ci"}]}]
    text = "BERT for text classification. Bidirectional encoder representations help text classification."
    spans = pt.RefinedMatcher(terms, {"M:bert"}, rules, {}).spans(text)
    papers = pd.DataFrame({"id": ["p"], "title": ["BERT for text classification"],
                           "abstract": ["Bidirectional encoder representations help text classification."]})
    expected = set(match_papers_refined(papers, terms, {"M:bert"}, rules, {})["term_id"])
    assert {s["id"] for s in spans} == expected
    refined = pt.RefinedMatcher(terms, {"M:bert"}, rules, {})
    assert refined.ambiguous == refined.matcher.ambiguous


cp = load("compare_first", "scripts/represcription/08_compare.py")


def test_comparison_maps_merged_terms_before_counting_shared_triads():
    rep = cp.canonical({"M:a": ["M:a", "M:a2"]})
    first = [{"rank": 1, "triad": ["M:a2", "T:x", "T:y"]}, {"rank": 2, "triad": ["M:b", "T:x", "T:y"]},
             {"rank": 3, "triad": ["M:c", "T:x", "T:y"]}, {"rank": 4, "triad": ["M:d", "T:x", "T:y"]}]
    fourth = [{"rank": 1, "triad": ["M:d", "T:x", "T:y"]}, {"rank": 2, "triad": ["T:y", "M:a", "T:x"]},
              {"rank": 3, "triad": ["M:c", "T:x", "T:y"]}, {"rank": 4, "triad": ["M:e", "T:x", "T:y"]}]
    out = cp.overlap(first, fourth, rep)
    assert out["shared_triads"] == 3 and [s["rank_fourth"] for s in out["shared"]] == [1, 2, 3]
    assert out["shared_rank_spearman"] is not None and out["distinct_terms"]["shared"] == 5


def test_comparison_runs_on_first_study_files(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    run = root / "represcription"
    for d in ("results", "prospective", "config"):
        (run / d).mkdir(parents=True)
    study = json.loads((ROOT / "config/study.json").read_text())
    for name in ("dvf", "briefing_generation", "q1", "q2"):
        shutil.copy(ROOT / "results" / f"{name}.json", root / "results")
        shutil.copy(ROOT / "results" / f"{name}.json", run / "results")
    reg = study["prospective"]["registration"]
    (root / reg).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / reg, root / reg)
    shutil.copy(ROOT / reg, run / "prospective" / "x.json")
    (run / "config/study.json").write_text(json.dumps({"prospective": {"registration": "prospective/x.json"}}))
    syn = root / study["represcription"]["review"]["combined"]["synonyms"]
    syn.parent.mkdir(parents=True, exist_ok=True)
    syn.write_text(json.dumps({"merge_groups": {}}))
    monkeypatch.setattr(cp, "ROOT", root)
    monkeypatch.setattr(cp, "load_study", lambda: study)
    cp.main()
    out = json.loads((run / "results/comparison_first_study.json").read_text())
    assert out["prescription"]["shared_triads"] == 100 and abs(out["prescription"]["shared_rank_spearman"] - 1.0) < 1e-9
    assert out["dvf"]["first"] == out["dvf"]["fourth"] and out["briefing_quality"]["first"]["q1"]["rate"] == 0.61
