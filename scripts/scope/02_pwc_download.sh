#!/usr/bin/env bash
# Download the frozen Papers with Code archive (CC-BY-SA-4.0) into data/scope
set -euo pipefail
mkdir -p data/scope
BASE="https://huggingface.co/datasets/pwc-archive"
curl -sL -o data/scope/methods.parquet "$BASE/methods/resolve/main/data/train-00000-of-00001.parquet"
curl -sL -o data/scope/links.parquet "$BASE/links-between-paper-and-code/resolve/main/data/train-00000-of-00001.parquet"
for i in 0 1 2 3; do
  curl -sL -o "data/scope/pwa_$i.parquet" "$BASE/papers-with-abstracts/resolve/main/data/train-0000$i-of-00004.parquet"
done
