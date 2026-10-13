"""Paper figures: hypergraph extraction and a full build from the committed result files."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = load("figure_extract", "scripts/figures/01_extract_hypergraph.py")
fg = load("figure_build", "scripts/figures/02_build_figures.py")


def test_neighbourhood_counts_pairs_and_the_open_triangle():
    edges = pd.DataFrame({"paper_id": ["p3", "p1", "p2", "p4"],
                          "terms": [["M:a", "M:b", "T:x"], ["M:a", "M:b"], ["M:b", "T:c"], ["M:a", "T:c", "M:b", "T:y"]]})
    out = ex.neighbourhood(edges, ["T:c", "M:a", "M:b"], examples=2)
    assert out["triad"] == ["M:a", "M:b", "T:c"] and out["papers_with_all_three"] == 1
    assert out["pairs"]["M:a|M:b"]["papers"] == 3
    # papers with another term come first, then ascending id
    assert [e["paper_id"] for e in out["pairs"]["M:a|M:b"]["examples"]] == ["p3", "p4"]
    assert out["pairs"]["M:a|M:b"]["examples"][0]["other_terms"] == ["T:x"]


def test_all_figures_build_from_result_files(tmp_path, monkeypatch):
    if not (ROOT / "figures/data/hypergraph_example.json").exists():
        pytest.skip("figures/data/hypergraph_example.json is made by scripts/figures/01_extract_hypergraph.py")
    monkeypatch.setattr("sys.argv", ["02_build_figures.py", "--out", str(tmp_path)])
    try:
        fg.main()
    except FileNotFoundError as error:
        pytest.skip(f"result file missing in this checkout: {error.filename}")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["figures"]) == 16
    assert all((tmp_path / name).stat().st_size > 0 for name in manifest["figures"])
    assert "results/engine_v3_verdict_T3.json" in manifest["inputs_sha256"]
