"""Two-stage briefing generation with Gemini (design 8.2, 8.3, 8.5).

Usage: python scripts/briefing/02_generate.py [--ranks 1 2] [--reps 1]

--ranks/--reps limit a run (for a pilot). A pilot call is a real call with the
final settings; its saved response is reused by the full run.

For each briefing target and repetition 1..R (applied.briefing.generation):
1. Stage 1 (combination frame): system instruction prompts/frame_<version>.txt,
   input = the bundle without the DVF table, output forced by triadrx.briefing.frame_schema.
2. Stage 2 (briefing), only when the frame is complete and valid: system instruction
   prompts/briefing_<version>.txt filled with the bundle's DVF values, input = bundle
   and frame, output forced by briefing_schema. Riskiest axis, MVP type and package
   versions are attached from the bundle, not generated.
Seed = seed_base + repetition (same for both stages). Malformed or invalid outputs
are recorded as such and not retried; only HTTP failures are retried. Every request
and response is saved; reruns reuse saved responses. The run stops if the model
version reported by the API differs from the configured model.

Needs GEMINI_API_KEY in .env (or in the environment).

Outputs
- data/derived/briefing/generations/rank_NN_rep_R_{frame,briefing}_<version>.json (raw, not committed)
- briefings/rank_NN/rep_R.json (frame and briefing, committed)
- results/briefing_generation.json: settings, prompt and bundle hashes, statuses, tokens
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Shared code lives in the triadrx package at the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triadrx import briefing as br
from triadrx.study import load_study

BUNDLES_RESULT = Path("results/briefing_bundles.json")
BUNDLE_DIR = Path("data/derived/briefing/bundles")
RAW_DIR = Path("data/derived/briefing/generations")
OUT_DIR = Path("briefings")
PROMPTS = Path("prompts")
RESULTS = Path("results")
API = "https://generativelanguage.googleapis.com/v1beta/models"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "GEMINI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        sys.exit("GEMINI_API_KEY not found in the environment or .env")
    return key


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def call(model: str, body: dict, key: str) -> tuple[int, dict]:
    """POST generateContent; retries rate limits and server errors only."""
    url = f"{API}/{model}:generateContent"
    data = json.dumps(body).encode("utf-8")
    for attempt in range(8):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return 200, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            text = err.read().decode("utf-8", errors="replace")
            if err.code == 429:
                print(f"HTTP 429, sleeping 60s: {text[:200]}")
                time.sleep(60)
                continue
            if err.code >= 500:
                time.sleep(15 * (attempt + 1))
                continue
            return err.code, {"error_body": text}
        except urllib.error.URLError:
            time.sleep(15 * (attempt + 1))
    return -1, {"error_body": "retries exhausted"}


def parse(response: dict) -> tuple[dict | None, str | None]:
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text), None
    except Exception as exc:
        return None, f"unparseable response: {type(exc).__name__}"


def run_stage(path: Path, model: str, body: dict, key: str, pause: float) -> dict:
    if path.exists():
        saved = json.loads(path.read_text())
        # A saved response is reused only for an identical request
        if saved["request"] != body:
            sys.exit(f"{path} was made with a different request; remove it or use a new prompt version")
        if saved["http_status"] == 200:
            return saved
    started = now_utc()
    status, response = call(model, body, key)
    record = {"requested_at": started, "http_status": status, "request": body, "response": response,
              "model_version": response.get("modelVersion"), "usage": response.get("usageMetadata")}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    time.sleep(pause)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", help="limit to these prescription ranks")
    parser.add_argument("--reps", type=int, nargs="+", help="limit to these repetitions")
    args = parser.parse_args()
    study = load_study()
    gen = study["applied"]["briefing"]["generation"]
    model = gen["model"]
    key = api_key()
    frame_prompt = (PROMPTS / f"frame_{gen['prompt_version']}.txt").read_text()
    briefing_template = (PROMPTS / f"briefing_{gen['prompt_version']}.txt").read_text()

    recorded = json.loads(BUNDLES_RESULT.read_text())["bundles"]
    bundles = {}
    for item in recorded:
        path = BUNDLE_DIR / f"rank_{item['rank']:02d}.json"
        if sha256(path) != item["bundle_sha256"]:
            sys.exit(f"{path} differs from the bundle recorded in {BUNDLES_RESULT}")
        bundles[item["rank"]] = json.loads(path.read_text())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    runs, versions = [], set()
    for rank in sorted(bundles):
        if args.ranks and rank not in args.ranks:
            continue
        bundle = bundles[rank]
        for rep in range(1, gen["repetitions"] + 1):
            if args.reps and rep not in args.reps:
                continue
            seed = gen["seed_base"] + rep
            config = {"temperature": gen["temperature"], "seed": seed, "responseMimeType": "application/json"}
            body1 = {
                "systemInstruction": {"parts": [{"text": frame_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": "Evidence bundle:\n" + json.dumps(br.frame_input(bundle), ensure_ascii=False)}]}],
                "generationConfig": {**config, "responseSchema": br.frame_schema(bundle)},
            }
            rec1 = run_stage(RAW_DIR / f"rank_{rank:02d}_rep_{rep}_frame_{gen['prompt_version']}.json", model, body1, key, gen["pause_seconds"])
            run = {"rank": rank, "rep": rep, "seed": seed, "frame_http": rec1["http_status"],
                   "model_version": rec1["model_version"], "tokens": [rec1.get("usage")]}
            if rec1["model_version"]:
                versions.add(rec1["model_version"])
            frame, error = parse(rec1["response"]) if rec1["http_status"] == 200 else (None, "http error")
            problems = [error] if error else br.validate_frame(frame, bundle)
            run["frame_status"] = frame.get("status") if frame else None
            run["insufficient_reason"] = frame.get("insufficient_reason") if frame else None
            run["frame_problems"] = problems
            output = {"rank": rank, "rep": rep, "seed": seed, "model": model, "prompt_version": gen["prompt_version"],
                      "reliability_flag": bundle.get("reliability_flag"), "concept_frame": frame,
                      "frame_valid": not problems, "briefing": None, "briefing_valid": None}

            if frame and not problems and frame["status"] == "complete":
                instruction = br.briefing_instruction(briefing_template, bundle)
                user = ("Evidence bundle:\n" + json.dumps(bundle, ensure_ascii=False)
                        + "\n\nCombination frame:\n" + json.dumps(frame, ensure_ascii=False))
                body2 = {
                    "systemInstruction": {"parts": [{"text": instruction}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {**config, "responseSchema": br.briefing_schema(bundle)},
                }
                rec2 = run_stage(RAW_DIR / f"rank_{rank:02d}_rep_{rep}_briefing_{gen['prompt_version']}.json", model, body2, key, gen["pause_seconds"])
                run["briefing_http"] = rec2["http_status"]
                run["tokens"].append(rec2.get("usage"))
                if rec2["model_version"]:
                    versions.add(rec2["model_version"])
                draft, error = parse(rec2["response"]) if rec2["http_status"] == 200 else (None, "http error")
                bproblems = [error] if error else br.validate_briefing(draft, bundle)
                run["briefing_problems"] = bproblems
                output["briefing_valid"] = not bproblems
                if draft and not bproblems:
                    output["briefing"] = br.attach_fixed(draft, frame, bundle)
                else:
                    output["briefing"] = draft
            if versions - {model}:
                sys.exit(f"API reported model versions {sorted(versions)} other than {model}; stopping")

            target = OUT_DIR / f"rank_{rank:02d}"
            target.mkdir(parents=True, exist_ok=True)
            (target / f"rep_{rep}.json").write_text(json.dumps(output, ensure_ascii=False, indent=1))
            runs.append(run)
            print(f"rank {rank:2d} rep {rep}: frame {run['frame_status']} {run['insufficient_reason']}"
                  f"{' problems' if problems else ''}"
                  f"{'' if 'briefing_problems' not in run else (' briefing ok' if not run['briefing_problems'] else ' briefing problems')}")

    def tokens(field: str) -> int:
        return sum((u or {}).get(field, 0) for r in runs for u in r["tokens"])

    if args.ranks or args.reps:
        print("Limited run (pilot); results/briefing_generation.json not written")
        return
    meta = {
        "step": "briefing_generation",
        "created": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "generation": gen,
        "model_versions": sorted(versions),
        "frame_prompt_sha256": sha256_text(frame_prompt),
        "briefing_template_sha256": sha256_text(briefing_template),
        "bundles_result_sha256": sha256(BUNDLES_RESULT),
    }
    summary = {
        "runs": len(runs),
        "frame_complete": sum(1 for r in runs if r["frame_status"] == "complete" and not r["frame_problems"]),
        "frame_insufficient": sum(1 for r in runs if r["frame_status"] == "insufficient_concept" and not r["frame_problems"]),
        "frame_invalid": sum(1 for r in runs if r["frame_problems"]),
        "briefing_valid": sum(1 for r in runs if r.get("briefing_problems") == []),
        "briefing_invalid": sum(1 for r in runs if r.get("briefing_problems")),
        "insufficient_reasons": {k: sum(1 for r in runs if r["insufficient_reason"] == k and r["frame_status"] == "insufficient_concept")
                                 for k in br.REASONS if k != "none"},
        "prompt_tokens": tokens("promptTokenCount"),
        "output_tokens": tokens("candidatesTokenCount"),
    }
    (RESULTS / "briefing_generation.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": runs}, indent=1))
    print(json.dumps(summary, indent=1))
    print("Saved", OUT_DIR, RESULTS / "briefing_generation.json")


if __name__ == "__main__":
    main()
