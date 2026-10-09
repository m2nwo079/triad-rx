"""Build the private Kaggle dataset and kernel metadata for the relation classifier.

Usage: python kaggle/relation/package.py <set> <output dir> <kaggle user> [--prompt path] [--max-new-tokens n] [--limit n] [--gpus n] [--early-stop]

Copies data/derived/relation/<set>_inputs.jsonl and the prompt (matching.relation.prompt unless given) into the output
directory with relation_settings.json and dataset-metadata.json, and writes the untracked
kaggle/relation/kernel-metadata.json for the same account (GPU and internet on).
"""
import argparse
import json
import shutil
from pathlib import Path

DATASET = "triad-rx-relation-data"
KERNEL = "triad-rx-relation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set")
    parser.add_argument("out", type=Path)
    parser.add_argument("user")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--early-stop", action="store_true")
    args = parser.parse_args()
    name, out, user = args.set, args.out, args.user
    cfg = json.loads(Path("config/study.json").read_text())["matching"]["relation"]
    prompt = Path(args.prompt or cfg["prompt"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("relation_v*.txt"):
        old.unlink()
    shutil.copy(Path(f"data/derived/relation/{name}_inputs.jsonl"), out / "relation_inputs.jsonl")
    shutil.copy(prompt, out / prompt.name)
    settings = {k: cfg[k] for k in ("model_id", "quantization", "decoding", "max_new_tokens", "batch_size")}
    settings.update(prompt=str(prompt), set=name, limit=args.limit, gpus=args.gpus, early_stop=args.early_stop)
    if args.max_new_tokens:
        settings["max_new_tokens"] = args.max_new_tokens
    (out / "relation_settings.json").write_text(json.dumps(settings, indent=2))
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]}, indent=2))
    kernel = {"id": f"{user}/{KERNEL}", "title": KERNEL, "code_file": "run.py", "language": "python", "kernel_type": "script",
              "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{user}/{DATASET}"],
              "competition_sources": [], "kernel_sources": []}
    (Path(__file__).parent / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    print(f"{name}: files in {out}")


if __name__ == "__main__":
    main()
