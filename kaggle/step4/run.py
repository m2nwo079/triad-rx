"""Kaggle runner for the engine study (design 18.9): T3 preparation (step 6).

Earlier runs: commit eb28253 ran pair evidence (T1, T2), the learned generator and the engine signals;
commit a91868c ran the learned generator candidates and the engine rediagnosed on them; commit 2b54e12
compared the candidate pipelines; commit 1725368 ran the power analysis (its output failed to save;
the analysis was rerun locally).

The private dataset holds repo.bundle (the committed branch) and data.tar.gz (derived inputs and the
build-window abstracts). The runner installs the pinned versions of requirements.lock when possible,
clones the bundle, unpacks the data, runs each step, and copies the results, the pair-evidence files
and a run log (time and peak child memory per step, installed versions) to /kaggle/working.
"""
import glob
import json
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/triad-rx")
PINNED = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm")
STEPS = [
    ["scripts/engine/14_timepoint_prep.py", "--check", "T1"],
    ["scripts/engine/14_timepoint_prep.py", "--timepoint", "R2"],
    ["scripts/engine/14_timepoint_prep.py", "--timepoint", "R3"],
    ["scripts/engine/14_timepoint_prep.py", "--timepoint", "T3"],
]
# Glob patterns relative to the repository; matches are copied to /kaggle/working with their paths
OUTPUTS = ["results/engine_prep_*.json", "vocab/timepoints/R2_*.json", "vocab/timepoints/R3_*.json",
           "vocab/timepoints/T3_*.json", "data/derived/*/main/R2*.parquet", "data/derived/*/main/R3*.parquet",
           "data/derived/*/main/T3*.parquet"]


def run(cmd: list[str], cwd: Path | None = None) -> dict:
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "seconds": round(time.time() - start, 1),
            "max_child_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-20000:]}


def main() -> None:
    source = Path(glob.glob("/kaggle/input/**/repo.bundle", recursive=True)[0]).parent
    log = {"python": sys.version, "steps": []}
    branch = subprocess.run(["git", "bundle", "list-heads", str(source / "repo.bundle")],
                            capture_output=True, text=True).stdout.split()[1].removeprefix("refs/heads/")
    log["clone"] = run(["git", "clone", "-b", branch, str(source / "repo.bundle"), str(REPO)])
    lock = (REPO / "requirements.lock").read_text().splitlines()
    pins = [line for line in lock if line.split("==")[0].lower() in PINNED]
    log["install"] = run([sys.executable, "-m", "pip", "install", "-q", "--retries", "1", "--timeout", "10", *pins])
    # Kaggle unpacks uploaded archives into a folder named after the archive
    if (source / "data.tar.gz").exists():
        with tarfile.open(source / "data.tar.gz") as tar:
            tar.extractall(REPO)
    else:
        shutil.copytree(source / "data" / "data", REPO / "data", dirs_exist_ok=True)
    log["head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    for step in STEPS:
        entry = run([sys.executable, *step], cwd=REPO)
        log["steps"].append(entry)
        print(step, entry["returncode"], entry["seconds"], entry["stdout"][-2000:], entry["stderr"][-2000:], flush=True)
        if entry["returncode"] != 0:
            break
    for pattern in OUTPUTS:
        for path in REPO.glob(pattern):
            target = WORK / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    (WORK / "run_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
