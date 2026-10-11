"""Kaggle GPU runner for the relation classifier (matching preregistration 5.3).

The private dataset holds relation_inputs.jsonl, the prompt file and relation_settings.json (model id,
quantization, decoding settings). The model is downloaded from Hugging Face, quantized to 4-bit nf4
and run greedily with thinking disabled, in batches. Execution settings (matching preregistration 5.3,
never the prompt text): with `gpus` > 1 the records are split by group (triad) into one worker process
per GPU; with `early_stop` the records of a group are classified in rounds and a group stops once one
of its papers is classified combination. Outputs go to /kaggle/working: relation_outputs.jsonl (raw
text and parsed relation per classified record) and relation_meta.json (model revision, library
versions, GPUs, timing and throughput).
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

WORK = Path("/kaggle/working")
RELATIONS = {"combination", "comparison", "mention"}


ROLES = {"core", "background", "alternative", "wrong_sense"}


def relation_from_roles(roles: list[str]) -> tuple[str, bool]:
    """Prompt v2 rule (matching preregistration 5.3): wrong sense first, then all core, then any alternative."""
    if "wrong_sense" in roles:
        return "mention", True
    if all(r == "core" for r in roles):
        return "combination", False
    if "alternative" in roles:
        return "comparison", False
    return "mention", False


def parse(text: str) -> tuple[str, bool]:
    """Relation and term-error flag from the first JSON object (v1: relation, v2: three roles); anything else is invalid."""
    match = re.search(r"\{.*?\}", text, flags=re.S)
    try:
        obj = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        obj = {}
    roles = obj.get("roles")
    if roles is not None:
        if isinstance(roles, list) and len(roles) == 3 and all(r in ROLES for r in roles):
            return relation_from_roles(roles)
        return "invalid", False
    relation = obj.get("relation")
    return (relation if relation in RELATIONS else "invalid"), bool(obj.get("term_error", False))


def user_message(r: dict) -> str:
    terms = "\n".join(f"{i + 1}. {t['name']} ({t['type']})" for i, t in enumerate(r["terms"]))
    return f"Terms:\n{terms}\n\nTitle: {r['title']}\n\nAbstract: {r['abstract']}"


def groups_of(records: list[dict]) -> dict[str, list[dict]]:
    """Records by group in input order; a record without a group is its own group."""
    groups = {}
    for r in records:
        groups.setdefault(r.get("group", r["id"]), []).append(r)
    return groups


def shard_of(key: str, shards: int) -> int:
    return zlib.crc32(key.encode("utf-8")) % shards


def classify(settings: dict, prompt: str, records: list[dict], step) -> list[dict]:
    """Classify in rounds over groups; `step` maps a list of records to their output rows."""
    groups = groups_of(records)
    done, out, k = set(), [], 0
    while True:
        todo = [g[k] for key, g in groups.items() if len(g) > k and not (settings.get("early_stop") and key in done)]
        if not todo:
            return out
        for start in range(0, len(todo), settings["batch_size"]):
            chunk = todo[start:start + settings["batch_size"]]
            for r, row in zip(chunk, step(chunk)):
                out.append(row)
                if row["relation"] == "combination":
                    done.add(r.get("group", r["id"]))
            print(f"round {k} {len(out)} classified", flush=True)
        k += 1


def worker(settings: dict, prompt: str, records: list[dict], tag: str) -> None:
    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"])
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(settings["model_id"], quantization_config=quant, device_map={"": 0},
                                                 torch_dtype=torch.float16)
    model.eval()

    def step(chunk):
        texts = [tokenizer.apply_chat_template([{"role": "system", "content": prompt}, {"role": "user", "content": user_message(r)}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=settings["max_new_tokens"], do_sample=False)
        rows = []
        for r, ids in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            raw = tokenizer.decode(ids, skip_special_tokens=True)
            relation, term_error = parse(raw)
            rows.append({"id": r["id"], "raw": raw, "relation": relation, "term_error": term_error})
        return rows

    t0 = time.time()
    out = classify(settings, prompt, records, step)
    with (WORK / f"{tag}_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"records_given": len(records), "records_classified": len(out), "seconds": round(time.time() - t0, 1),
            "model_revision": model_info(settings["model_id"]).sha, "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__, "torch": torch.__version__}
    (WORK / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    args = parser.parse_args()
    source = Path(glob.glob("/kaggle/input/**/relation_settings.json", recursive=True)[0]).parent
    settings = json.loads((source / "relation_settings.json").read_text())
    prompt = (source / Path(settings["prompt"]).name).read_text()
    records = [json.loads(line) for line in (source / "relation_inputs.jsonl").read_text().splitlines() if line.strip()]
    # Format check runs only the first records (matching preregistration 5.3)
    if settings.get("limit"):
        records = records[: settings["limit"]]
    if args.shard is not None:
        mine = [r for r in records if shard_of(r.get("group", r["id"]), args.shards) == args.shard]
        worker(settings, prompt, mine, f"shard{args.shard}")
        return

    install = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"],
                             capture_output=True, text=True)
    import torch

    started, t0 = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), time.time()
    shards = min(settings.get("gpus", 1), torch.cuda.device_count())
    if shards > 1:
        # Expandable segments avoid out-of-memory errors from fragmentation; memory placement only
        procs = [subprocess.Popen([sys.executable, __file__, "--shard", str(i), "--shards", str(shards)],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i),
                                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}) for i in range(shards)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"worker exit codes {codes}")
    else:
        worker(settings, prompt, records, "shard0")
    out, shard_meta = [], []
    for i in range(max(shards, 1)):
        out += [json.loads(line) for line in (WORK / f"shard{i}_outputs.jsonl").read_text().splitlines() if line.strip()]
        shard_meta.append(json.loads((WORK / f"shard{i}_meta.json").read_text()))
    seconds = time.time() - t0
    with (WORK / "relation_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"model_id": settings["model_id"], "model_revision": shard_meta[0]["model_revision"], "settings": settings,
            "started": started, "seconds": round(seconds, 1), "records_given": len(records), "records": len(out),
            "records_per_second": len(out) / seconds, "invalid": sum(r["relation"] == "invalid" for r in out),
            "gpu": shard_meta[0]["gpu"], "gpus": shards, "shards": shard_meta, "transformers": shard_meta[0]["transformers"],
            "torch": shard_meta[0]["torch"], "python": platform.python_version(), "install_returncode": install.returncode}
    (WORK / "relation_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
