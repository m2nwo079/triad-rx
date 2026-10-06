"""Pair combination-evidence counts in the build window (design 18.9 step 4, signal b).

Usage: python scripts/engine/09_pair_evidence.py --timepoint T1|T2

For every build-window paper, the adopted combination-evidence rule of engine preregistration 2.1
(scripts/engine/02_label_evidence.py, reused through importlib: title as the first sentence,
windows of `window_sentences` sentences, no comparison cue, evidence matching rules) is applied to
pairs: a pair of the paper's hypergraph terms has evidence in the paper when some window without
a cue matches both. Only build-window papers are read, so no label information enters.

Outputs
- data/derived/pair_evidence/main/<tp>.parquet (not committed): x, y, n_cooc, n_evidence
- results/pair_evidence_<tp>.json: counts, hashes and the runtime environment
"""
import argparse
import datetime
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study, runtime_environment
from triadrx.text import Matcher

HYPERGRAPH = Path("data/derived/hypergraph/main")
OUT_DIR = Path("data/derived/pair_evidence/main")
RESULTS = Path("results")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load("label_evidence", ROOT / "scripts/engine/02_label_evidence.py")


def evidence_pairs(sentence_terms: list[set], sentences: list[str], size: int, cues, allowed: set) -> set[tuple]:
    """Pairs of `allowed` terms that share some window of `size` sentences without a comparison cue."""
    out = set()
    for i in range(len(sentences)):
        window = range(i, min(i + size, len(sentences)))
        terms = set().union(*(sentence_terms[j] for j in window)) & allowed
        if len(terms) >= 2 and not cues.search(" ".join(sentences[j] for j in window)):
            out.update(itertools.combinations(sorted(terms), 2))
    return out


def pair_table(tp: str, cfg: dict, build: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence and combination-evidence paper counts of every build-window pair."""
    vocab = {t["id"]: t for t in json.loads(evidence.VOCAB.read_text())["terms"]}
    rules = cfg["evidence_matching"]
    ev_terms, cs_only = evidence.evidence_vocab(evidence.graph.final_terms(tp, vocab), rules["skip_surface_chars"])
    matcher = Matcher(ev_terms)
    cues = evidence.cue_pattern(cfg["comparison_cues"])
    text = pd.read_parquet(evidence.PAPERS, columns=["id", "title", "abstract"]).set_index("id")
    cooc, ev = Counter(), Counter()
    for pid, ts in zip(build["paper_id"], build["terms"]):
        allowed = set(ts)
        if len(allowed) < 2:
            continue
        cooc.update(itertools.combinations(sorted(allowed), 2))
        row = text.loc[pid]
        sents = ([row["title"]] if cfg["title_as_first_sentence"] else []) + evidence.split_sentences(row["abstract"], cfg["abbreviations"])
        ev.update(evidence_pairs(evidence.evidence_terms(matcher, sents, cs_only, rules), sents,
                                 cfg["window_sentences"], cues, allowed))
    return pd.DataFrame([(x, y, n, ev[(x, y)]) for (x, y), n in sorted(cooc.items())],
                        columns=["x", "y", "n_cooc", "n_evidence"])


def main() -> None:
    study = load_study()
    er = study["engine_reliability"]
    cfg = er["label_v2"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timepoint", required=True, choices=er["signals"]["timepoints"])
    tp = parser.parse_args().timepoint

    path = HYPERGRAPH / f"{tp}_build.parquet"
    df = pair_table(tp, cfg, pd.read_parquet(path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tp}.parquet"
    df.to_parquet(out, index=False)
    result = {
        "meta": {"step": "pair_evidence", "timepoint": tp, "created": datetime.date.today().isoformat(),
                 "git_commit": evidence.evaluate.git_commit(), "label_v2_config": cfg,
                 "environment": runtime_environment(),
                 "inputs_sha256": {"build": evidence.evaluate.sha256(path)},
                 "output_sha256": evidence.evaluate.sha256(out)},
        "pairs": int(len(df)),
        "pairs_with_evidence": int((df["n_evidence"] > 0).sum()),
        "pair_paper_links": int(df["n_cooc"].sum()),
        "pair_paper_links_with_evidence": int(df["n_evidence"].sum()),
    }
    (RESULTS / f"pair_evidence_{tp}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
