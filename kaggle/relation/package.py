"""Build the private Kaggle dataset and kernel metadata for the relation classifier.

Usage: python kaggle/relation/package.py <set> <output dir> <kaggle user> [--prompt path] [--max-new-tokens n] [--limit n] [--gpus n] [--early-stop] [--part i --parts n]

Copies data/derived/relation/<set>_inputs.jsonl and the prompt (matching.relation.prompt unless given) into the output
directory with relation_settings.json and dataset-metadata.json, and writes the untracked
kaggle/relation/kernel-metadata.json for the same account (GPU and internet on).
With --part i --parts n only the groups (triads) of part i are copied, so a set that would exceed one
Kaggle session runs as n kernel runs; a group never spans two parts, and the part hash differs from the
runner's GPU shard hash so each part still splits evenly across GPUs. The label step merges the parts.
"""
import argparse
import json
import shutil
import zlib
from pathlib import Path

DATASET = "triad-rx-relation-data"
KERNEL = "triad-rx-relation"


def part_of(record: dict, parts: int) -> int:
    """Part of a record's group; salted so it is independent of the runner's GPU shard (crc32 % shards)."""
    return zlib.crc32(("part|" + record.get("group", record["id"])).encode("utf-8")) % parts


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
    parser.add_argument("--part", type=int)
    parser.add_argument("--parts", type=int)
    args = parser.parse_args()
    if (args.part is None) != (args.parts is None) or (args.parts is not None and not 0 <= args.part < args.parts):
        parser.error("--part and --parts go together, with 0 <= part < parts")
    name, out, user = args.set, args.out, args.user
    cfg = json.loads(Path("config/study.json").read_text())["matching"]["relation"]
    prompt = Path(args.prompt or cfg["prompt"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("relation_v*.txt"):
        old.unlink()
    source = Path(f"data/derived/relation/{name}_inputs.jsonl")
    if args.parts is None:
        shutil.copy(source, out / "relation_inputs.jsonl")
    else:
        lines = [line for line in source.read_text().splitlines() if line.strip()]
        kept = [line for line in lines if part_of(json.loads(line), args.parts) == args.part]
        (out / "relation_inputs.jsonl").write_text("\n".join(kept) + "\n")
        print(f"part {args.part} of {args.parts}: {len(kept)} of {len(lines)} records")
    shutil.copy(prompt, out / prompt.name)
    settings = {k: cfg[k] for k in ("model_id", "quantization", "decoding", "max_new_tokens", "batch_size")}
    settings.update(prompt=str(prompt), set=name, limit=args.limit, gpus=args.gpus, early_stop=args.early_stop,
                    part=args.part, parts=args.parts)
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
