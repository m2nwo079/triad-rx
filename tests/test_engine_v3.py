"""Third-study engine steps on toy data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec = load("engine_candidates", "scripts/matching/14_engine_candidates.py")


def toy_build() -> pd.DataFrame:
    papers = [["M:a", "M:b"], ["M:a", "M:b"], ["M:b", "T:c"], ["M:b", "T:c"], ["M:a", "T:c"], ["M:a", "T:c"],
              ["M:d", "M:a"], ["M:d", "M:a"], ["M:d", "T:c"], ["M:a", "M:b", "T:e"], ["M:d", "T:e"], ["M:d", "T:e"]]
    return pd.DataFrame({"paper_id": [f"p{i}" for i in range(len(papers))], "terms": papers})


def test_term_pi_reads_the_r5_records_of_final_terms():
    vocab = {"terms": [{"id": "M:a"}, {"id": "T:c"}],
             "r5": [{"term_id": "M:a", "pi": 0.7}, {"term_id": "T:c", "pi": 0.9}, {"term_id": "M:gone", "pi": 0.1}]}
    assert ec.term_pi_v3(vocab) == {"M:a": 0.7, "T:c": 0.9}


def test_prepare_matches_the_second_study_prepare_for_the_same_pi(monkeypatch):
    build = toy_build()
    pi = {t: 0.9 for ts in build["terms"] for t in ts}
    monkeypatch.setattr(ec.recall.candidates, "term_pi", lambda tp, variant: pi)
    ours, theirs = ec.prepare(build, pi, 0.5), ec.recall.prepare(build, "T1", 0.5)
    for key in ("terms", "n"):
        assert ours[key] == theirs[key]
    for key in ("counts", "docs", "stable", "closed", "is_task"):
        assert np.array_equal(ours[key], theirs[key])
    for name in ours["scores"]:
        assert np.allclose(ours["scores"][name][0], theirs["scores"][name][0], equal_nan=True)


def test_excluded_pairs_are_dropped_from_generator_codes():
    index = {"M:a": 0, "M:b": 1, "T:c": 2, "M:d": 3}
    mat = ec.excluded_matrix(index, {("M:a", "M:b")})
    n = 4
    codes = np.array([(0 * n + 1) * n + 2, (0 * n + 2) * n + 3, (1 * n + 2) * n + 3], dtype=np.int64)
    kept, score, dropped = ec.drop_excluded_codes(codes, np.array([3.0, 2.0, 1.0]), mat, n)
    assert kept.tolist() == codes[1:].tolist() and score.tolist() == [2.0, 1.0] and dropped == 1


def test_generator_budget_ranks_by_score_without_excluded_pairs():
    build = toy_build()
    pi = {t: 0.99 for ts in build["terms"] for t in ts}
    g = ec.prepare(build, pi, 0.1)
    mat = ec.excluded_matrix(g["index"], set())
    cands, seeds, total, dropped = ec.generator_candidates(g, "stable", "npmi_mean", 2, mat)
    assert len(cands) == min(2, total) and dropped == 0
    assert cands["gen_rank"].tolist() == list(range(1, len(cands) + 1))
    assert (cands["gen_score"].diff().dropna() <= 0).all()


def test_labels_are_attached_by_sorted_triad():
    cands = pd.DataFrame({"a": ["M:a", "M:b"], "b": ["M:b", "M:d"], "c": ["T:c", "T:e"]})
    labels = pd.DataFrame({"a": ["M:a"], "b": ["M:b"], "c": ["T:c"], "label_v3": [True]})
    out = ec.attach_labels(cands, labels)
    assert out["label_v3"].tolist() == [True, False] and out["realised"].tolist() == [True, False]


ef = load("engine_features", "scripts/matching/15_engine_features.py")


def toy_papers(build: pd.DataFrame, path: Path) -> None:
    words = ["graph", "neural", "attention", "policy", "image", "speech", "transformer", "reward"]
    rows = [{"id": pid, "title": f"paper {i}", "abstract": " ".join(words[(i + j) % len(words)] for j in range(12))}
            for i, pid in enumerate(build["paper_id"])]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_feature_table_keeps_generator_columns_and_all_feature_sets(tmp_path, monkeypatch):
    build = toy_build().assign(first_date=pd.Timestamp("2017-06-01", tz="UTC"))
    toy_papers(build, tmp_path / "papers.parquet")
    monkeypatch.setattr(ef.features, "PAPERS", tmp_path / "papers.parquet")
    pi = {t: 0.99 for ts in build["terms"] for t in ts}
    g = ec.prepare(build, pi, 0.1)
    cands, _, _, _ = ec.generator_candidates(g, "stable", "npmi_mean", 3, ec.excluded_matrix(g["index"], set()))
    cands = cands.assign(label_v3=True, realised=True)
    text_cfg = {"min_df": 1, "max_df": 1.0, "max_features": 100, "svd_dims": 2, "seed": 0}
    table, _ = ef.feature_table(build, cands, pi, 0.1, pd.Timestamp("2017-01-01", tz="UTC"), text_cfg)
    study = json.loads(Path("config/study.json").read_text())
    expected = sorted(f for s in study["features"]["sets"].values() for f in s)
    assert sorted(c for c in table.columns if c not in ef.CANDIDATE_COLUMNS) == expected
    assert "label_v3" not in table.columns and len(table) == len(cands)


def test_window_check_rejects_papers_outside_the_build_window():
    edges = pd.DataFrame({"first_date": pd.to_datetime(["2016-01-01", "2020-01-01"], utc=True)})
    with pytest.raises(SystemExit):
        ef.check_window(edges, pd.Timestamp("2014-01-01", tz="UTC"), pd.Timestamp("2018-12-31", tz="UTC"))


ed = load("engine_diagnosis", "scripts/matching/16_engine_diagnosis.py")


def test_diagnosis_refuses_t3_and_train_labels_after_the_evaluation_build_window():
    study = ed.prep.merged_study(json.loads(Path("config/study.json").read_text()))
    ed.check_timepoints("T1", "T2", study)
    with pytest.raises(SystemExit):
        ed.check_timepoints("T1", "T3", study)
    with pytest.raises(SystemExit):
        ed.check_timepoints("R2", "T2", study)


def test_diagnosis_stops_without_label_file(tmp_path):
    with pytest.raises(SystemExit):
        ed.labelled(pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["z"]}), tmp_path / "missing.parquet")


def test_recall_by_budget_counts_unreachable_positives_in_the_totals():
    labels = pd.DataFrame({"a": ["a1", "a2", "a3", "a4"], "b": ["b"] * 4, "c": ["c"] * 4, "label_v3": [True, True, False, True]})
    first = ec.attach_labels(pd.DataFrame({"a": ["a1", "z"], "b": ["b", "b"], "c": ["c", "c"]}), labels)
    gen = ec.attach_labels(pd.DataFrame({"a": ["a2", "a1", "a3", "q"], "b": ["b"] * 4, "c": ["c"] * 4}), labels)
    gen = gen.assign(gen_rank=[1, 2, 3, 4])
    out = ed.recall_by_budget(first, gen, labels, [1, 2])
    assert out["totals"] == {"label_v3": 3, "realised": 4}
    assert out["first_study_rule"]["label_v3"]["hits"] == 1
    assert [r["label_v3"]["hits"] for r in out["generator"]] == [2, 2]
    assert [r["realised"]["hits"] for r in out["generator"]] == [2, 3]


ep = load("engine_power", "scripts/matching/17_engine_power.py")


def toy_power_input(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    n_terms, n_pool, n_pos = 12, 300, 40
    pool_terms = np.array([np.sort(rng.choice(n_terms, 3, replace=False)) for _ in range(n_pool)])
    pos_terms = np.array([np.sort(rng.choice(n_terms, 3, replace=False)) for _ in range(n_pos)])
    pool_pos_index = np.full(n_pool, -1)
    hit = rng.choice(n_pool, 25, replace=False)
    pool_pos_index[hit] = np.arange(25)
    pool_y = (pool_pos_index >= 0).astype(float)
    gen_rank = np.arange(1, n_pool + 1)
    scores = {"engine": pool_y + rng.normal(0, 0.6, n_pool), "engine_no_h": pool_y + rng.normal(0, 0.7, n_pool),
              "baseline": rng.normal(size=n_pool)}
    budgets = [100, 300]
    orders = {b: {name: np.argsort(-np.where(gen_rank <= b, s, -np.inf), kind="stable")[:b] for name, s in scores.items()}
              for b in budgets}
    pos_rank = np.full(n_pos, np.inf)
    pos_rank[:25] = gen_rank[hit]
    return {"n_terms": n_terms, "budgets": budgets, "orders": orders, "pool_terms": pool_terms, "pool_rank": gen_rank,
            "pool_y": pool_y, "pool_pos_index": pool_pos_index, "pos_terms": pos_terms,
            "pos_in_first": (rng.random(n_pos) < 0.2).astype(float), "pos_rank": pos_rank}


def test_power_steps_s1_to_s3_match_the_second_study_and_add_s4():
    d = toy_power_input()
    ones_pool, ones_pos = np.ones(len(d["pool_y"])), np.ones(len(d["pos_terms"]))
    keep = np.ones(len(ones_pos), bool)
    ours = ep.step_values(d, d["pool_y"], keep, ones_pool, ones_pos, d["budgets"], [10, 50])
    theirs = ep.pw.step_values(d, d["pool_y"], keep, ones_pool, ones_pos, d["budgets"], [10, 50])
    for b in d["budgets"]:
        for s in ("S1", "S2", "S3"):
            assert np.allclose(ours[b][s], theirs[b][s], equal_nan=True)
        assert len(ours[b]["S4"]) == 2


def test_power_sequences_match_the_second_study_for_the_same_seed():
    d = toy_power_input()
    cfg = {"outer": 4, "inner": 15, "seed": 7}
    ours = ep.nested_power(d, d["budgets"], [10, 50], cfg, 0.975, 0.9)
    theirs = ep.pw.nested_power(d, d["budgets"], [10, 50], cfg, 0.975, 0.9)
    for b in map(str, d["budgets"]):
        assert ours[b]["S1"] == theirs[b]["S1"]
        for k in ("10", "50"):
            for name in ("S2", "S3", "S1_S2", "S1_S2_S3"):
                assert ours[b]["by_k"][k][name] == theirs[b]["by_k"][k][name]
            assert ours[b]["by_k"][k]["S1_S2_S3_S4"] <= ours[b]["by_k"][k]["S1_S2_S3"]


def test_verdict_config_is_unset_before_the_power_run():
    verdict = json.loads(Path("config/study.json").read_text())["matching"]["verdict"]
    assert verdict["eval"] == "T3" and verdict["power"]["eval_timepoint"] == "T2"
    assert verdict["power"]["budget_multiples_of_first_candidates"] == [1, 5]


tl = load("t3_label_rebuild", "scripts/matching/18_t3_label_rebuild.py")
tv = load("t3_verdict_v3", "scripts/matching/19_t3_verdict.py")
pk = load("relation_package", "kaggle/relation/package.py")
lr = load("label_v3_relation", "scripts/matching/13_label_v3_relation.py")


def test_t3_steps_stay_locked_until_the_freeze_commit_is_recorded(tmp_path, monkeypatch):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("# x\n\n고정 커밋: 미정\n")
    study = {"matching": {"prereg_file": str(prereg)}}
    for module in (tl, tv):
        with pytest.raises(SystemExit):
            module.require_frozen(study)
    prereg.write_text("# x\n\n고정 커밋: abc1234\n")
    assert tl.require_frozen(study) == "abc1234" and tv.require_frozen(study) == "abc1234"


def test_t3_matching_inputs_restore_phrase_tuples_and_case_sensitive_terms():
    vocab = {"terms": [{"id": "M:a", "surfaces": [{"match": "cs"}]}, {"id": "M:b", "surfaces": [{"match": "cs"}, {"match": "ci"}]}],
             "exclusion_phrases": {"M:b": [["left", "right"]]}}
    terms, cs_only, phrases = tl.matching_inputs(vocab)
    assert cs_only == {"M:a"} and phrases == {"M:b": [("left", "right")]} and len(terms) == 2


def test_parts_keep_groups_whole_and_cover_every_record():
    records = [{"id": f"r{i}", "group": f"g{i // 3}"} for i in range(300)]
    parts = [pk.part_of(r, 3) for r in records]
    by_group = {}
    for r, p in zip(records, parts):
        by_group.setdefault(r["group"], set()).add(p)
    assert all(len(v) == 1 for v in by_group.values()) and set(parts) == {0, 1, 2}


def test_part_outputs_merge_only_when_settings_match_and_parts_are_complete(tmp_path):
    def write(i, ids, part, extra=None):
        out, meta = tmp_path / f"o{i}.jsonl", tmp_path / f"m{i}.json"
        pd.DataFrame({"id": ids, "relation": "mention"}).to_json(out, orient="records", lines=True)
        meta.write_text(json.dumps({"settings": {"set": "T3", "gpus": 2, "part": part, "parts": 2, **(extra or {})}}))
        return out, meta
    o0, m0 = write(0, ["a", "b"], 0)
    o1, m1 = write(1, ["c"], 1)
    outputs, metas = lr.merged_outputs([o0, o1], [m0, m1])
    assert sorted(outputs["id"]) == ["a", "b", "c"] and len(metas) == 2
    o2, m2 = write(2, ["c"], 0)
    with pytest.raises(SystemExit):
        lr.merged_outputs([o0, o2], [m0, m2])
    o3, m3 = write(3, ["d"], 1, {"gpus": 1})
    with pytest.raises(SystemExit):
        lr.merged_outputs([o0, o3], [m0, m3])


def test_verdict_reading_follows_the_preregistered_power_rule():
    seq = [{"step": s} for s in ("S1_recall", "S2_random", "S3_npmi", "S4_h_contribution")]
    power = {"S1": 1.0, "S1_S2": 1.0, "S1_S2_S3": 0.33, "S1_S2_S3_S4": 0.0, "S4": 0.0, "minimum": 0.8}
    s = tv.run_sequence({"S1_recall": True, "S2_random": True, "S3_npmi": False, "S4_h_contribution": False}, seq)
    assert s == {"S1_recall": "pass", "S2_random": "pass", "S3_npmi": "fail", "S4_h_contribution": "not tested"}
    assert tv.reading(s, power)["S3_npmi"] == "fail (insufficient power)"
    s = tv.run_sequence({"S1_recall": True, "S2_random": True, "S3_npmi": True, "S4_h_contribution": False}, seq)
    r = tv.reading(s, power)
    assert r["S3_npmi"] == "pass" and r["S4_h_contribution"] == "undetermined (insufficient power)"
    s = tv.run_sequence({"S1_recall": True, "S2_random": False, "S3_npmi": True, "S4_h_contribution": True}, seq)
    assert tv.reading(s, power)["S2_random"] == "fail" and s["S3_npmi"] == "not tested"
