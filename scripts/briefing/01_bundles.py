"""Evidence bundles for the briefing targets (design 8.1, 8.1.1).

Usage: python scripts/briefing/01_bundles.py

Targets: prescription ranks 1..applied.briefing.top_n from the prospective file.
Candidate-to-evidence links are taken from the DVF script's Evidence class
(scripts/dvf/01_dvf.py, loaded with importlib), so bundles use exactly the links
the DVF table used. Selection rules are in applied.briefing.bundle:
- papers: per pair, the most recent papers_per_pair pair-evidence papers, full abstract
- repositories: top repos_max by stars in the window, with description
- packages: top packages_max by second-half downloads, with the last version
  uploaded on or before data_cutoff (PyPI JSON API)
- patents: top patents_max related inventions by latest document date
- DVF table: axis scores, riskiest axis and reason, MVP type
Ties are broken by evidence id. Repository descriptions are current values
(design 16.7). Supplement requests are cached and logged.

Needs GITHUB_TOKEN in .env (or in the environment).

Outputs
- data/derived/briefing/bundles/rank_<NN>.json (not committed; contain abstracts)
- data/derived/briefing/repo_meta.jsonl, pypi_meta.jsonl (caches, not committed)
- results/evidence_log.json: keys "github_repo_meta", "pypi_json"
- results/briefing_bundles.json: settings, counts and bundle hashes
"""
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx.study import data_cutoff, load_study

DVF_SCRIPT = Path("scripts/dvf/01_dvf.py")
DVF_RESULT = Path("results/dvf.json")
VOCAB = Path("vocab/vocab_base.json")
EV = Path("data/derived/evidence")
OUT_DIR = Path("data/derived/briefing")
BUNDLE_DIR = OUT_DIR / "bundles"
REPO_CACHE = OUT_DIR / "repo_meta.jsonl"
PYPI_CACHE = OUT_DIR / "pypi_meta.jsonl"
RESULTS = Path("results")
EVIDENCE_LOG = RESULTS / "evidence_log.json"
PAUSE = 0.75


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


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    env = Path(".env")
    if not token and env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "GITHUB_TOKEN":
                token = value.strip().strip('"').strip("'")
    if not token:
        sys.exit("GITHUB_TOKEN not found in the environment or .env")
    return token


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def get_json(url: str, headers: dict) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "triad-rx-evidence", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return 200, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code in (403, 429) and ("Retry-After" in err.headers or err.headers.get("X-RateLimit-Remaining") == "0"):
                wait = int(err.headers.get("Retry-After", "0")) or max(0, int(err.headers.get("X-RateLimit-Reset", "0")) - int(time.time())) + 2
                print(f"HTTP {err.code}, sleeping {wait}s")
                time.sleep(wait)
                continue
            if err.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return err.code, None
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    return -1, None


def read_cache(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[record[key]] = record
    return out


def append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def fetch_repos(repos: list[str], token: str) -> dict:
    cache = read_cache(REPO_CACHE, "repo")
    for repo in [r for r in repos if r not in cache or cache[r]["status"] not in (200, 404)]:
        status, data = get_json(f"https://api.github.com/repos/{repo}", {
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"})
        record = {"repo": repo, "url": f"GET /repos/{repo}", "fetched_at": now_utc(), "status": status,
                  "description": (data or {}).get("description"), "archived": (data or {}).get("archived")}
        append(REPO_CACHE, record)
        cache[repo] = record
        time.sleep(PAUSE)
    return cache


def last_version_before(releases: dict, cutoff: datetime.datetime) -> tuple[str | None, str | None]:
    """Latest non-yanked version whose first file was uploaded on or before cutoff."""
    best = None
    for version, files in releases.items():
        uploads = [f for f in files if not f.get("yanked")]
        if not uploads:
            continue
        first = min(datetime.datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00")) for f in uploads)
        if first <= cutoff and (best is None or first > best[1]):
            best = (version, first)
    return (best[0], best[1].isoformat()) if best else (None, None)


def fetch_packages(packages: list[str], cutoff: datetime.datetime) -> dict:
    cache = read_cache(PYPI_CACHE, "package")
    for pkg in [p for p in packages if p not in cache or cache[p]["status"] not in (200, 404)]:
        url = f"https://pypi.org/pypi/{urllib.parse.quote(pkg)}/json"
        status, data = get_json(url, {"Accept": "application/json"})
        version, uploaded = last_version_before(data.get("releases", {}), cutoff) if data else (None, None)
        record = {"package": pkg, "url": url, "fetched_at": now_utc(), "status": status,
                  "version_before_cutoff": version, "version_uploaded_at": uploaded}
        append(PYPI_CACHE, record)
        cache[pkg] = record
        time.sleep(PAUSE)
    return cache


def by_value_then_id(items: list[tuple[float, str]], k: int) -> list[str]:
    """Highest value first, ties by evidence id ascending."""
    return [i for _, i in sorted(items, key=lambda x: (-x[0], x[1]))[:k]]


def main() -> None:
    study = load_study()
    briefing = study["applied"]["briefing"]
    rules = briefing["bundle"]
    expected = {"paper_order": "first_date_desc", "abstract": "full", "repo_order": "stars_window_desc",
                "package_order": "second_half_downloads_desc", "patent_order": "latest_document_date_desc",
                "tie_break": "evidence_id_asc"}
    for key, value in expected.items():
        if rules[key] != value:
            sys.exit(f"applied.briefing.bundle.{key} = {rules[key]!r} is not implemented")
    if briefing["order"] != "rank_prescription":
        sys.exit("only applied.briefing.order == 'rank_prescription' is implemented")
    cutoff = data_cutoff()
    token = github_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dvf_mod = load("dvf", DVF_SCRIPT)
    ev = dvf_mod.Evidence(study)
    dvf = {c["rank"]: c for c in json.loads(DVF_RESULT.read_text())["candidates"]}
    keys = {t["id"]: t["key"] for t in json.loads(VOCAB.read_text())["terms"]}

    order = sorted(range(ev.n), key=lambda c: ev.candidates[c]["rank"])[: briefing["top_n"]]

    papers = pd.read_parquet(EV / "papers.parquet", columns=["evidence_id", "term_x", "term_y", "paper_id", "first_date", "title", "abstract"])
    by_pair = {k: g.sort_values(["first_date", "evidence_id"], ascending=[False, True])
               for k, g in papers.groupby(["term_x", "term_y"])}
    stars = pd.read_parquet(EV / "repo_stars.parquet", columns=["repo", "stars_window"]).set_index("repo")["stars_window"]
    docs = pd.read_parquet(EV / "patents.parquet", columns=["evidence_id", "invention_id", "date", "title", "abstract"])
    latest_doc = docs.sort_values(["date", "evidence_id"], ascending=[False, True]).drop_duplicates("invention_id").set_index("invention_id")
    inv_docs = docs.groupby("invention_id")["evidence_id"].agg(sorted).to_dict()

    selections = {}
    for c in order:
        repos = by_value_then_id([(float(ev.repo_stars[i]), ev.repo_ids[i]) for i in ev.cand_repo_idx[c]], rules["repos_max"])
        pkgs = by_value_then_id([(float(ev.pkg_second[i]), ev.package_ids[i]) for i in ev.cand_pkg_idx[c]], rules["packages_max"])
        invs = sorted({ev.invention_ids[i] for i, _, _ in ev.cand_inv[c]},
                      key=lambda v: (-latest_doc.loc[v, "date"].timestamp(), v))[: rules["patents_max"]]
        selections[c] = (repos, pkgs, invs)
    repo_meta = fetch_repos(sorted({r for s in selections.values() for r in s[0]}), token)
    pypi_meta = fetch_packages(sorted({p for s in selections.values() for p in s[1]}), cutoff)

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for c in order:
        cand = ev.candidates[c]
        repos, pkgs, invs = selections[c]
        paper_items = {}
        for key in ev.pairs[c]:
            group = by_pair.get(key)
            if group is None:
                continue
            for row in group.head(rules["papers_per_pair"]).itertuples(index=False):
                item = paper_items.setdefault(row.evidence_id, {
                    "id": row.evidence_id, "first_date": row.first_date.isoformat(), "title": row.title,
                    "abstract": row.abstract, "pairs": []})
                item["pairs"].append([keys.get(key[0], key[0]), keys.get(key[1], key[1])])
        pidx = {k: i for i, k in enumerate(ev.package_ids)}
        cats = {}
        for i, assignee, category in ev.cand_inv[c]:
            cats.setdefault(ev.invention_ids[i], set()).add(category)
        d = dvf[cand["rank"]]
        bundle = {
            "rank": cand["rank"],
            "triad": [{"term_id": t, "name": keys.get(t, t), "type": t.split(":", 1)[0]} for t in cand["triad"]],
            "reliability_flag": json.loads(Path(study["applied"]["dvf"]["prospective_file"]).read_text())["meta"].get("reliability_flag"),
            "papers": sorted(paper_items.values(), key=lambda x: (x["first_date"], x["id"]), reverse=True),
            "repositories": [{"id": "repo:" + r, "stars_window": float(stars[r]),
                              "description": repo_meta[r].get("description"),
                              "description_fetched_at": repo_meta[r]["fetched_at"]} for r in repos],
            "packages": [{"id": "pkg:" + p,
                          "downloads_first_half": float(ev.pkg_first[pidx[p]]),
                          "downloads_second_half": float(ev.pkg_second[pidx[p]]),
                          "version_before_cutoff": pypi_meta[p].get("version_before_cutoff")} for p in pkgs],
            "patents": [{"ids": inv_docs[v], "title": latest_doc.loc[v, "title"], "abstract": latest_doc.loc[v, "abstract"],
                         "assignee_categories": sorted(cats.get(v, []))} for v in invs],
            "dvf": {"id": f"dvf:{cand['rank']}",
                    "axes": {a: {"score": d["axes"][a]["score"], "missing": d["axes"][a]["value"] is None} for a in "DFV"},
                    "riskiest_axis": d["riskiest_axis"], "riskiest_reason": d["riskiest_reason"], "mvp_type": d["mvp_type"]},
        }
        path = BUNDLE_DIR / f"rank_{cand['rank']:02d}.json"
        path.write_text(json.dumps(bundle, indent=1, ensure_ascii=False))
        summary.append({"rank": cand["rank"], "papers": len(bundle["papers"]), "repositories": len(repos),
                        "packages": len(pkgs), "patents": len(invs),
                        "repos_without_description": sum(1 for r in repos if not repo_meta[r].get("description")),
                        "packages_without_version": sum(1 for p in pkgs if not pypi_meta[p].get("version_before_cutoff")),
                        "bundle_sha256": sha256(path)})

    used_repos = sorted({r for s in selections.values() for r in s[0]})
    used_pkgs = sorted({p for s in selections.values() for p in s[1]})
    log = json.loads(EVIDENCE_LOG.read_text()) if EVIDENCE_LOG.exists() else {}
    log["github_repo_meta"] = [{k: repo_meta[r][k] for k in ("url", "fetched_at", "status")} for r in used_repos]
    log["pypi_json"] = [{k: pypi_meta[p][k] for k in ("url", "fetched_at", "status")} for p in used_pkgs]
    EVIDENCE_LOG.write_text(json.dumps(log, indent=1))

    meta = {
        "step": "briefing_bundles",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "data_cutoff": cutoff.isoformat(),
        "briefing": briefing,
        "dvf_sha256": sha256(DVF_RESULT),
        "note": "repository descriptions are values at fetch time (design 16.7)",
    }
    (RESULTS / "briefing_bundles.json").write_text(json.dumps({"meta": meta, "bundles": summary}, indent=1))
    print(pd.DataFrame(summary).drop(columns="bundle_sha256").to_string(index=False))
    print("Saved", BUNDLE_DIR, RESULTS / "briefing_bundles.json")


if __name__ == "__main__":
    main()
