"""DVF indicators, axis scores, riskiest axis and Poisson bootstrap (design 7.2-7.5, 8.4).

Usage: python scripts/dvf/01_dvf.py

Population: the prospective registration file (applied.dvf). Indicators (all
"higher is better"):
- D1 stars: sum of stars in the window over distinct GitHub repositories linked to
  the pair-evidence papers of the candidate's three pairs (PWC links, and README
  links whose README has the exact id and at most readme_max_arxiv_ids ids).
- D2 download growth: (second-half sum + s) / (first-half sum + s) over distinct
  PyPI packages linked to those repositories, s = growth_smoothing.
- F1 pair realization: minimum over the three pairs of the pair-evidence paper count.
- F2 open triangle: 1 or 0 (structural, fixed in the bootstrap).
- F3 implementations: number of the three terms with at least one build-window
  paper that has a GitHub repository (PWC, or README verified and not a list).
- V1 company assignees: distinct company assignee_id over related inventions
  (an invention is related when both terms of one of the pairs are among its
  matched terms); inventions without an assignee are excluded and counted.
- V2 company share: company / (company + academic + government) over distinct
  assignees of the related inventions; ambiguous and unknown are excluded and
  counted, with the ambiguous-as-company and ambiguous-as-academic sensitivities.

Percentiles: average rank among candidates with a value. Axis value: mean of the
available indicator percentiles; axis score 1-5 from five equal-width bins of the
axis value. Riskiest axis (8.4): a missing axis first; otherwise the lowest score,
ties by the axis value, then F, V, D order.

Bootstrap (7.3 step 4): each evidence unit (paper, repository, package,
invention) gets a Poisson(1) weight per replicate, shared across candidates; F2 is
fixed. Reported: 95% intervals of axis values and scores, and how often each axis
is the riskiest.

Output: results/dvf.json
"""
import datetime
import hashlib
import itertools
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import load_study

EV = Path("data/derived/evidence")
INPUTS = {
    "papers": EV / "papers.parquet",
    "hypergraph": Path("data/derived/hypergraph/main/applied_build.parquet"),
    "pwc_links": EV / "pwc_links.parquet",
    "readme_links": EV / "readme_links.parquet",
    "repo_stars": EV / "repo_stars.parquet",
    "pypi_links": EV / "pypi_links.parquet",
    "pypi_downloads": EV / "pypi_downloads.parquet",
    "patents": EV / "patents.parquet",
    "patent_assignees": EV / "patent_assignees.parquet",
    "assignee_classes": EV / "assignee_classes.parquet",
    "evidence_papers": Path("results/evidence_papers.json"),
}
RESULTS = Path("results")
AXES = {"D": ["D1", "D2"], "F": ["F1", "F2", "F3"], "V": ["V1", "V2"]}
MVP = {
    "F": "technical proof of concept from existing packages",
    "V": "customer interviews and price acceptance test",
    "D": "demand landing page and problem interviews",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def percentiles(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile among non-missing values (NaN stays NaN)."""
    s = pd.Series(values)
    return s.rank(method="average", pct=True).to_numpy()


def axis_score(value: float) -> float:
    if np.isnan(value):
        return np.nan
    return float(min(5, max(1, math.ceil(5 * value))))


def riskiest(values: dict, scores: dict, order: list[str]) -> tuple[str, str]:
    missing = [a for a in order if np.isnan(values[a])]
    if missing:
        return missing[0], "missing"
    best = min(order, key=lambda a: (scores[a], values[a], order.index(a)))
    return best, "lowest"


class Evidence:
    """Index structures linking candidates to weighted evidence units."""

    def __init__(self, study: dict):
        ev_cfg = study["applied"]["evidence"]
        self.candidates = json.loads(Path(study["applied"]["dvf"]["prospective_file"]).read_text())["top_k_prescription"]
        self.n = len(self.candidates)
        self.pairs = [[pair_key(a, b) for a, b in itertools.combinations(c["triad"], 2)] for c in self.candidates]
        open_flag = {tuple(c["triad"]): c["open_triangle"] for c in json.loads(INPUTS["evidence_papers"].read_text())["candidates"]}
        self.F2 = np.array([float(open_flag[tuple(c["triad"])]) for c in self.candidates])

        # Papers: unit index over pair-evidence papers and implementation papers
        papers = pd.read_parquet(INPUTS["papers"], columns=["term_x", "term_y", "paper_id"])
        pwc = pd.read_parquet(INPUTS["pwc_links"], columns=["paper_id", "repo", "pair_evidence"])
        readme = pd.read_parquet(INPUTS["readme_links"], columns=["paper_id", "repo", "id_found", "readme_arxiv_ids"])
        readme = readme[(readme["id_found"] == True) & (readme["readme_arxiv_ids"] <= ev_cfg["readme_max_arxiv_ids"])]
        impl_papers = set(pwc["paper_id"]) | set(readme["paper_id"])
        self.paper_ids = sorted(set(papers["paper_id"]) | impl_papers)
        pidx = {p: i for i, p in enumerate(self.paper_ids)}
        pair_papers = papers.groupby([papers["term_x"], papers["term_y"]])["paper_id"].agg(list).to_dict()
        self.pair_paper_idx = [[np.array([pidx[p] for p in pair_papers.get(k, [])], dtype=int) for k in ps] for ps in self.pairs]

        terms = sorted({t for c in self.candidates for t in c["triad"]})
        edges = pd.read_parquet(INPUTS["hypergraph"], columns=["paper_id", "terms"])
        edges = edges[edges["paper_id"].isin(impl_papers)].explode("terms")
        edges = edges[edges["terms"].isin(terms)]
        term_impl = edges.groupby("terms")["paper_id"].agg(list).to_dict()
        self.term_impl_idx = {t: np.array([pidx[p] for p in term_impl.get(t, [])], dtype=int) for t in terms}

        # Repositories linked to pair-evidence papers, with stars
        stars = pd.read_parquet(INPUTS["repo_stars"], columns=["repo", "status", "stars_window"])
        stars = stars[stars["status"] == 200].set_index("repo")["stars_window"].astype(float)
        links = pd.concat([pwc.loc[pwc["pair_evidence"], ["paper_id", "repo"]], readme[["paper_id", "repo"]]])
        links = links[links["repo"].isin(stars.index)].drop_duplicates()
        paper_repos = links.groupby("paper_id")["repo"].agg(set).to_dict()
        self.repo_ids = sorted(stars.index)
        ridx = {r: i for i, r in enumerate(self.repo_ids)}
        self.repo_stars = stars.reindex(self.repo_ids).to_numpy()
        cand_repos = []
        for ps in self.pairs:
            repos = set()
            for k in ps:
                for p in pair_papers.get(k, []):
                    repos |= paper_repos.get(p, set())
            cand_repos.append(sorted(repos))
        self.cand_repo_idx = [np.array([ridx[r] for r in rs], dtype=int) for rs in cand_repos]

        # Packages linked to those repositories, with half sums
        plinks = pd.read_parquet(INPUTS["pypi_links"], columns=["package", "repo"])
        downloads = pd.read_parquet(INPUTS["pypi_downloads"])
        downloads = downloads[downloads["status"] == 200].set_index("package")
        self.package_ids = sorted(downloads.index)
        kidx = {k: i for i, k in enumerate(self.package_ids)}
        self.pkg_first = downloads.reindex(self.package_ids)["downloads_first_half"].astype(float).to_numpy()
        self.pkg_second = downloads.reindex(self.package_ids)["downloads_second_half"].astype(float).to_numpy()
        repo_pkgs = plinks[plinks["package"].isin(kidx)].groupby("repo")["package"].agg(set).to_dict()
        self.cand_pkg_idx = [
            np.array(sorted({kidx[k] for r in rs for k in repo_pkgs.get(r, set())}), dtype=int) for rs in cand_repos
        ]
        self.smoothing = float(ev_cfg["growth_smoothing"])

        # Related inventions and their assignees
        docs = pd.read_parquet(INPUTS["patents"], columns=["invention_id", "terms"])
        inv_terms = docs.explode("terms").dropna().groupby("invention_id")["terms"].agg(set).to_dict()
        assignees = pd.read_parquet(INPUTS["patent_assignees"], columns=["invention_id", "assignee_id"])
        assignees = assignees[assignees["assignee_id"] != ""].drop_duplicates()
        inv_assignees = assignees.groupby("invention_id")["assignee_id"].agg(set).to_dict()
        classes = pd.read_parquet(INPUTS["assignee_classes"], columns=["assignee_id", "category"]).set_index("assignee_id")["category"].to_dict()
        self.invention_ids = sorted(inv_terms)
        iidx = {v: i for i, v in enumerate(self.invention_ids)}
        self.cand_inv = []
        self.no_assignee = []
        for ps in self.pairs:
            related = [v for v, ts in inv_terms.items() if any(a in ts and b in ts for a, b in ps)]
            with_a = [v for v in related if v in inv_assignees]
            self.no_assignee.append(len(related) - len(with_a))
            # (invention index, assignee category) for every assignee of every related invention
            rows = [(iidx[v], a, classes.get(a, "unknown")) for v in with_a for a in sorted(inv_assignees[v])]
            self.cand_inv.append(rows)

    def indicators(self, wp: np.ndarray, wr: np.ndarray, wk: np.ndarray, wi: np.ndarray) -> dict:
        n = self.n
        out = {k: np.full(n, np.nan) for k in ["D1", "D2", "F1", "F3", "V1", "V2"]}
        extra = {"V_ambiguous": np.zeros(n), "V_unknown": np.zeros(n),
                 "V2_ambiguous_as_company": np.full(n, np.nan), "V2_ambiguous_as_academic": np.full(n, np.nan)}
        # Per-term and per-pair values are computed once and shared by candidates
        has_impl = {t: bool((wp[idx] > 0).any()) for t, idx in self.term_impl_idx.items()}
        pair_count = {}
        for c in range(n):
            counts = []
            for key, idx in zip(self.pairs[c], self.pair_paper_idx[c]):
                if key not in pair_count:
                    pair_count[key] = float(wp[idx].sum())
                counts.append(pair_count[key])
            out["F1"][c] = min(counts)
            triad = self.candidates[c]["triad"]
            out["F3"][c] = float(sum(has_impl[t] for t in triad))
            ridx = self.cand_repo_idx[c]
            active = ridx[wr[ridx] > 0]
            if len(active):
                out["D1"][c] = float((wr[active] * self.repo_stars[active]).sum())
            kidx = self.cand_pkg_idx[c]
            kact = kidx[wk[kidx] > 0]
            if len(kact):
                first = (wk[kact] * self.pkg_first[kact]).sum()
                second = (wk[kact] * self.pkg_second[kact]).sum()
                out["D2"][c] = (second + self.smoothing) / (first + self.smoothing)
            present = {}
            for inv, assignee, category in self.cand_inv[c]:
                if wi[inv] > 0:
                    present[assignee] = category
            if present:
                cats = list(present.values())
                company = cats.count("company")
                academic = cats.count("academic")
                government = cats.count("government")
                ambiguous = cats.count("ambiguous")
                out["V1"][c] = float(company)
                extra["V_ambiguous"][c] = ambiguous
                extra["V_unknown"][c] = cats.count("unknown")
                base = company + academic + government
                if base:
                    out["V2"][c] = company / base
                if base + ambiguous:
                    extra["V2_ambiguous_as_company"][c] = (company + ambiguous) / (base + ambiguous)
                    extra["V2_ambiguous_as_academic"][c] = company / (base + ambiguous)
        out["F2"] = self.F2.copy()
        return out, extra


def score(ind: dict, order: list[str]) -> dict:
    pct = {k: percentiles(v) for k, v in ind.items()}
    n = len(ind["F2"])
    axes = {}
    for axis, keys in AXES.items():
        stack = np.vstack([pct[k] for k in keys])
        # nanmean warns on all-missing columns; those become NaN on purpose
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            value = np.where(np.isnan(stack).all(axis=0), np.nan, np.nanmean(stack, axis=0))
        axes[axis] = {
            "value": value,
            "score": np.array([axis_score(v) for v in value]),
            "missing_indicators": np.isnan(stack).sum(axis=0),
        }
    risk = [
        riskiest({a: axes[a]["value"][c] for a in order}, {a: axes[a]["score"][c] for a in order}, order)
        for c in range(n)
    ]
    return {"pct": pct, "axes": axes, "riskiest": risk}


def clean(x):
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main() -> None:
    study = load_study()
    dvf = study["applied"]["dvf"]
    expected = {"patent_unit": "pair", "missing_assignee": "exclude_and_report", "f_pair_aggregate": "min",
                "f_implementation": "terms_with_repo", "percentile_ties": "average"}
    for key, value in expected.items():
        if dvf[key] != value:
            sys.exit(f"applied.dvf.{key} = {dvf[key]!r} is not implemented")
    if dvf["riskiest"]["missing_axis"] != "first" or dvf["riskiest"]["tie_break"] != "axis_mean_percentile":
        sys.exit("only riskiest = missing first, ties by axis mean percentile is implemented")
    if dvf["bootstrap"]["method"] != "poisson":
        sys.exit("only the Poisson bootstrap is implemented")
    order = dvf["riskiest"]["axis_order"]

    ev = Evidence(study)
    ones = [np.ones(len(ev.paper_ids)), np.ones(len(ev.repo_ids)), np.ones(len(ev.package_ids)), np.ones(len(ev.invention_ids))]
    point_ind, point_extra = ev.indicators(*ones)
    point = score(point_ind, order)

    reps = dvf["bootstrap"]["replicates"]
    rng = np.random.default_rng(dvf["bootstrap"]["seed"])
    boot_values = {a: np.full((reps, ev.n), np.nan) for a in AXES}
    boot_scores = {a: np.full((reps, ev.n), np.nan) for a in AXES}
    boot_risk = np.empty((reps, ev.n), dtype=object)
    for b in range(reps):
        weights = [rng.poisson(1.0, size=len(w)).astype(float) for w in ones]
        ind, _ = ev.indicators(*weights)
        res = score(ind, order)
        for a in AXES:
            boot_values[a][b] = res["axes"][a]["value"]
            boot_scores[a][b] = res["axes"][a]["score"]
        boot_risk[b] = [r[0] for r in res["riskiest"]]
        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{reps}")

    candidates = []
    for c, cand in enumerate(ev.candidates):
        axis_out = {}
        for a in AXES:
            vals = boot_values[a][:, c]
            scs = boot_scores[a][:, c]
            ok = ~np.isnan(vals)
            axis_out[a] = {
                "value": clean(point["axes"][a]["value"][c]),
                "score": clean(point["axes"][a]["score"][c]),
                "missing_indicators": int(point["axes"][a]["missing_indicators"][c]),
                "bootstrap_value_95": [clean(np.quantile(vals[ok], 0.025)), clean(np.quantile(vals[ok], 0.975))] if ok.any() else None,
                "bootstrap_score_95": [clean(np.quantile(scs[ok], 0.025)), clean(np.quantile(scs[ok], 0.975))] if ok.any() else None,
                "bootstrap_missing_share": float(1 - ok.mean()),
            }
        risk_axis, risk_reason = point["riskiest"][c]
        counts = pd.Series(boot_risk[:, c]).value_counts()
        candidates.append({
            "rank": cand["rank"],
            "triad": cand["triad"],
            "indicators": {k: clean(point_ind[k][c]) for k in ["D1", "D2", "F1", "F2", "F3", "V1", "V2"]},
            "percentiles": {k: clean(point["pct"][k][c]) for k in ["D1", "D2", "F1", "F2", "F3", "V1", "V2"]},
            "axes": axis_out,
            "riskiest_axis": risk_axis,
            "riskiest_reason": risk_reason,
            "mvp_type": MVP[risk_axis],
            "riskiest_bootstrap_share": {a: float(counts.get(a, 0) / reps) for a in order},
            "evidence_counts": {
                "related_repositories": int(len(ev.cand_repo_idx[c])),
                "related_packages": int(len(ev.cand_pkg_idx[c])),
                "related_inventions_without_assignee": int(ev.no_assignee[c]),
                "V_ambiguous_assignees": int(point_extra["V_ambiguous"][c]),
                "V_unknown_assignees": int(point_extra["V_unknown"][c]),
            },
            "V2_sensitivity": {
                "ambiguous_as_company": clean(point_extra["V2_ambiguous_as_company"][c]),
                "ambiguous_as_academic": clean(point_extra["V2_ambiguous_as_academic"][c]),
            },
        })

    prospective = json.loads(Path(dvf["prospective_file"]).read_text())
    meta = {
        "step": "dvf",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "reliability_flag": prospective["meta"].get("reliability_flag"),
        "note": "DVF is a transparent indicator-based priority; its predictive validity is not tested (design 7.1)",
        "settings": {"dvf": dvf, "evidence": study["applied"]["evidence"]},
        "axis_bins": "score = ceil(5 * axis value), clipped to 1-5 (five equal-width bins of the percentile scale)",
        "inputs_sha256": {k: sha256(p) for k, p in INPUTS.items()},
        "units": {"papers": len(ev.paper_ids), "repositories": len(ev.repo_ids),
                  "packages": len(ev.package_ids), "inventions": len(ev.invention_ids)},
    }
    summary = {
        "candidates": ev.n,
        "axis_missing": {a: int(sum(1 for c in candidates if c["axes"][a]["value"] is None)) for a in AXES},
        "riskiest_axis": {a: int(sum(1 for c in candidates if c["riskiest_axis"] == a)) for a in order},
        "riskiest_by_missing": int(sum(1 for c in candidates if c["riskiest_reason"] == "missing")),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dvf.json").write_text(json.dumps({"meta": meta, "summary": summary, "candidates": candidates}, indent=1))
    print(json.dumps(summary, indent=1))
    print("Saved", RESULTS / "dvf.json")


if __name__ == "__main__":
    main()
