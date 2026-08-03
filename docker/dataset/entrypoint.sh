#!/usr/bin/env bash
# docker/dataset/entrypoint.sh — umi-dataset stage CLI.
#
#   entrypoint.sh <episode_dir> <out_dir>
#     <episode_dir>  raw episode,  e.g. /data/raw_episodes/<id>
#     <out_dir>      stage output, e.g. /data/derived/<id>/dataset_v1
#
# Thin wrapper: the real work lives in robot_umi_server.dataset_stage (module
# in development — this wrapper fixes the container contract now). Touches
# <out_dir>/COMPLETE only if the module exits 0; harmless if the module also
# writes its own marker.

set -euo pipefail

usage() { echo "usage: $0 <episode_dir> <out_dir>" >&2; exit 64; }
[[ $# -eq 2 ]] || usage
EPISODE_DIR=$1
OUT_DIR=$2

[[ -d "$EPISODE_DIR" ]] || { echo "error: episode dir not found: $EPISODE_DIR" >&2; exit 66; }

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE"

python -m robot_umi_server.dataset_stage "$EPISODE_DIR" "$OUT_DIR"

touch "$OUT_DIR/COMPLETE"
echo "dataset: done -> $OUT_DIR"
