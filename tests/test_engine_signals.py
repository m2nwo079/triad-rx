"""Step 4 signals (scripts/engine/09, 10, 11) on toy data."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pe = load("pair_evidence", "scripts/engine/09_pair_evidence.py")
lg = load("learned_generator", "scripts/engine/10_learned_generator.py")
es = load("engine_signals", "scripts/engine/11_engine_signals.py")
CUES = pe.evidence.cue_pattern(["compared with", "baseline"])


def test_evidence_pairs_respect_window_cue_and_allowed_terms():
    sents = ["We combine A with B.", "C is compared with D.", "E helps."]
    terms = [{"A", "B"}, {"C", "D"}, {"E"}]
    allowed = {"A", "B", "C", "D"}
    # One-sentence windows: C-D sits in a sentence with a cue; E is not one of the paper's terms
    assert pe.evidence_pairs(terms, sents, 1, CUES, allowed) == {("A", "B")}
    # Two-sentence windows: every window that holds A and B also holds the cue sentence
    assert pe.evidence_pairs(terms, sents, 2, CUES, allowed) == set()


def test_evidence_share_matrix_is_symmetric():
    pairs = pd.DataFrame({"x": ["a"], "y": ["b"], "n_cooc": [4], "n_evidence": [1]})
    m = lg.evidence_share(pairs, {"a": 0, "b": 1, "c": 2})
    assert m[0, 1] == m[1, 0] == np.float32(0.25) and m[0, 2] == 0


def test_triad_features_ignore_term_order_and_chunking():
    n = 4
    rng = np.random.default_rng(0)
    mats = {}
    for name in lg.PAIR_MATRICES:
        m = rng.random((n, n)).astype(np.float32)
        mats[name] = m + m.T
    log_docs = rng.random(n).astype(np.float32)
    codes = np.array([(0 * n + 1) * n + 2, (1 * n + 2) * n + 3])
    gen = np.array([0.5, 0.7], dtype=np.float32)
    whole = lg.triad_features(codes, gen, mats, log_docs, n, 10)
    chunked = lg.triad_features(codes, gen, mats, log_docs, n, 1)
    assert np.array_equal(whole, chunked)
    v = [mats["npmi"][0, 1], mats["npmi"][0, 2], mats["npmi"][1, 2]]
    col = lg.feature_names().index("npmi_mean")
    assert math.isclose(whole[0, col], float(np.mean(v)), rel_tol=1e-6)


def test_context_overlap_excludes_triad_terms():
    # Papers of (u, v) and (v, w) share the context term x; u, v, w themselves are left out
    build = pd.DataFrame({"terms": [["u", "v", "x"], ["v", "w", "x"], ["u", "w", "y"]]})
    df = pd.DataFrame({"a": ["u"], "b": ["v"], "c": ["w"]})
    out = es.context_features(df, build)
    # Pivot v: contexts {x} and {x} give 1; pivot u: (u,v) {x} vs (u,w) {y} give 0; pivot w: (u,w) {y} vs (v,w) {x} give 0
    assert math.isclose(out["S_ctx_max"].iloc[0], 1.0)
    assert math.isclose(out["S_ctx_mean"].iloc[0], 1 / 3)


def test_context_overlap_is_zero_for_a_missing_pair():
    build = pd.DataFrame({"terms": [["u", "v", "x"]]})
    out = es.context_features(pd.DataFrame({"a": ["u"], "b": ["v"], "c": ["w"]}), build)
    assert out["S_ctx_max"].iloc[0] == 0


def test_evidence_features_count_pairs_with_evidence():
    df = pd.DataFrame({"a": ["u"], "b": ["v"], "c": ["w"]})
    pairs = pd.DataFrame({"x": ["u", "v"], "y": ["v", "w"], "n_cooc": [2, 4], "n_evidence": [1, 0]})
    out = es.evidence_features(df, pairs)
    assert out["E_pairs"].iloc[0] == 1 and math.isclose(out["E_share_mean"].iloc[0], 0.5 / 3)


def test_config_has_the_keys_the_engine_scripts_read():
    import json
    er = json.loads((ROOT / "config/study.json").read_text())["engine_reliability"]
    assert {"timepoints", "sets", "models", "diversify_caps", "learned_generator"} <= set(er["signals"])
    assert {"selected", "diagnosis_budget_multiple", "tune_timepoint", "confirm_timepoint"} <= set(er["conditional"]["generator"])
    assert {"train_timepoint", "eval_timepoint", "models", "k"} <= set(er["conditional"]["rediagnosis"])
    assert {"reference", "pipelines", "k"} <= set(er["pipeline_compare"])
    assert er["verdict_pipeline"]["primary"] in er["pipeline_compare"]["pipelines"]
    assert {"budget_multiples_of_first_candidates", "k", "outer", "inner", "seed", "minimum_sequence_power"} <= set(er["power_v2"])


def test_pipeline_compare_union_and_paired_difference():
    pc = load("pipeline_compare", "scripts/engine/12_pipeline_compare.py")
    p1 = pd.DataFrame({"a": ["a", "a", "b"], "b": ["b", "c", "c"], "c": ["x", "x", "x"], "label_v2": [True, False, False]})
    p2 = pd.DataFrame({"a": ["a", "d"], "b": ["c", "e"], "c": ["x", "x"], "label_v2": [False, True]})
    union, pos = pc.union_index({"p1": p1, "p2": p2})
    assert len(union) == 4
    # The shared triad (a, c, x) maps to the same union row in both pools
    assert pos["p1"][1] == pos["p2"][0]
    ev = {"bootstrap_seed": 0, "bootstrap_n": 30, "ci_level": 0.95}
    ranked = {"p1": pos["p1"], "same": pos["p1"].copy(), "p2": pos["p2"]}
    out = pc.paired_diffs(union, ranked, "p1", [1], ev)
    assert out["same"]["1"]["candidate"]["lower"] == 0 == out["same"]["1"]["term_block"]["upper"]
    assert set(out) == {"same", "p2"}


def test_power_precisions_match_dev_precision_and_choice_rule():
    pw = load("engine_power", "scripts/engine/13_power.py")
    rng = np.random.default_rng(3)
    y = (rng.random(50) < 0.3).astype(float)
    w = rng.integers(0, 3, 50).astype(float)
    order = rng.permutation(50)
    for k, p in zip([1, 5, 20, 200], pw.precisions(order, y, w, [1, 5, 20, 200])):
        expected = pw.dev.precision_at_k(order, y, w, k)
        assert (np.isnan(p) and np.isnan(expected)) or math.isclose(p, expected)
    power = {"10": {"by_k": {"1": {"S1_S2": 0.7}, "2": {"S1_S2": 0.9}}}, "20": {"by_k": {"1": {"S1_S2": 0.9}}}}
    assert pw.choose(power, 0.8) == {"budget": 10, "k": 2, "power_S1_S2": 0.9, "meets_minimum": True}
    # Power shares come out as numpy values; the choice must still be JSON serialisable
    numpy_power = {"10": {"by_k": {"1": {"S1_S2": np.float64(0.5)}}}}
    import json
    json.dumps(pw.choose(numpy_power, np.float64(0.8)))


def test_power_step_values_on_a_toy_pool():
    pw = load("engine_power", "scripts/engine/13_power.py")
    d = {"pos_in_first": np.array([1.0, 0.0]), "pos_rank": np.array([1.0, np.inf]), "pool_rank": np.array([1, 2]),
         "orders": {2: {"engine": np.array([0, 1]), "baseline": np.array([1, 0])}}}
    out = pw.step_values(d, np.array([1.0, 0.0]), np.array([True, True]), np.ones(2), np.ones(2), [2], [1])
    # Recall: generator holds 1 of 2 positives, first-study candidates also 1 of 2
    assert out[2]["S1"] == 0 and out[2]["S2"] == [2.0] and out[2]["S3"] == [1.0]


def test_verdict_values_follow_the_power_choice():
    import hashlib
    import json
    study = json.loads((ROOT / "config/study.json").read_text())
    er = study["engine_reliability"]
    verdict, power_cfg = er["verdict"], er["power_v2"]
    power_file = ROOT / verdict["power_result"]["file"]
    assert hashlib.sha256(power_file.read_bytes()).hexdigest() == verdict["power_result"]["sha256"]
    choice = json.loads(power_file.read_text())["choice"]
    first = json.loads((ROOT / f"results/candidates_main_{power_cfg['eval_timepoint']}.json").read_text())["stats"]["candidates"]
    assert choice["budget"] == verdict["budget_multiple_of_first_candidates"] * first
    assert choice["k"] == verdict["k"] and verdict["k"] not in verdict["descriptive_k"]
    assert [s["step"] for s in verdict["sequence"]] == ["S1_recall", "S2_random", "S3_npmi", "S4_h_contribution"]


def test_timepoint_prep_merges_windows_in_memory_and_keeps_t3_labels_closed():
    import json
    from triadrx.study import build_window, label_window
    prep = load("timepoint_prep", "scripts/engine/14_timepoint_prep.py")
    study = json.loads((ROOT / "config/study.json").read_text())
    merged = prep.merged_study(study)
    assert "R2" not in study["timepoints"] and {"R2", "R3", "T3"} <= set(merged["timepoints"])
    assert build_window("R2", merged)[0].year == study["engine_reliability"]["timepoints"]["R2"]["build"][0]
    assert merged["engine_reliability"]["timepoints"]["T3"]["role"] != "train"
    with pytest.raises(Exception):
        label_window("T3", merged)


def test_t3_verdict_helpers():
    tv = load("t3_verdict", "scripts/engine/15_t3_verdict.py")
    pos = pd.DataFrame({"a": list("pqrs"), "b": list("tuvw"), "c": list("xxyy")})
    ev = {"bootstrap_seed": 0, "bootstrap_n": 50, "ci_level": 0.95}
    out = tv.recall_diff_intervals(pos, np.array([1.0, 1, 1, 0]), np.array([1.0, 0, 0, 0]), ev)
    assert out["point"] == 0.5 and out["recall_pool"] == 0.75 and out["recall_first"] == 0.25
    assert out["candidate"]["lower"] >= 0
    ci = {"candidate": {"lower": 0.2}, "term_block": {"lower": -0.1}}
    assert not tv.passes(ci, 0.0) and tv.passes({"candidate": {"lower": 0.2}, "term_block": {"lower": 0.1}}, 0.0)
    bins = tv.reliability(np.linspace(0, 1, 20), np.r_[np.zeros(10), np.ones(10)], bins=4)
    assert [b["n"] for b in bins] == [5, 5, 5, 5] and bins[0]["observed_rate"] == 0 and bins[-1]["observed_rate"] == 1
