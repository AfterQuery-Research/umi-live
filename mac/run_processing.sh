#!/usr/bin/env bash
# mac/run_processing.sh — the committed Mac processing pipeline: ingest + the runner with
# the arm64-GLIM engine, turning uploaded episodes into a HuggingFace dataset WITHOUT a
# GPU box. Run on the Mac. Reuses the server ingest + gripper + dataset stages verbatim;
# only SLAM is Mac-local (mac/Dockerfile -> umi-glim-live-arm64, pcap replay).
#
#   cp .env.example .env && $EDITOR .env
#     # ROBOT_UMI_REPO_ID, HF_TOKEN, ROBOT_UMI_PUSH_TO_HUB=1, ROBOT_UMI_UPLOAD_TOKEN
#     # (UPLOAD_TOKEN must match the recorder Pi's — the same secrets value)
#   mac/run_processing.sh
#
# The Pi (RIG_TARGET=mac) tunnels its uploads to this Mac's localhost:8091 (the
# umi-upload-tunnel unit bin/provision-pi installs). Ctrl-C stops both ingest + runner.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

IMG=umi-glim-live-arm64
DATA_DIR="${DATA_DIR:-$REPO/data}"
INGEST_HOST="${INGEST_HOST:-127.0.0.1}"
INGEST_PORT="${INGEST_PORT:-8091}"
PYBIN="$REPO/.venv/bin/python"

[ -x "$PYBIN" ] || { echo "no venv at $PYBIN — run 'uv sync --package robot-umi-server' first." >&2; exit 1; }
[ -f "$REPO/.env" ] || { echo "no .env — cp .env.example .env and set ROBOT_UMI_REPO_ID / HF_TOKEN / ROBOT_UMI_UPLOAD_TOKEN." >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is required (the arm64 GLIM stage runs in a container)." >&2; exit 1; }

# arm64 GLIM image — also carries the replay entrypoint + export_tum (build once, ~1h).
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "=== building $IMG from mac/Dockerfile (one-time, ~1h) ==="
  docker build -f "$HERE/Dockerfile" -t "$IMG" "$HERE"
fi

mkdir -p "$DATA_DIR/raw_episodes" "$DATA_DIR/derived"
set -a; . "$REPO/.env"; set +a

echo "=== ingest on http://$INGEST_HOST:$INGEST_PORT  (data: $DATA_DIR) ==="
"$PYBIN" server/scripts/server.py --host "$INGEST_HOST" --port "$INGEST_PORT" --data-dir "$DATA_DIR" &
INGEST=$!
trap 'kill $INGEST 2>/dev/null || true' EXIT INT TERM

echo "=== runner --engine mac${PAIR:+ --pair}  (replay -> GLIM -> gripper -> dataset -> HF) ==="
"$PYBIN" server/scripts/runner.py --engine mac ${PAIR:+--pair} --data-dir "$DATA_DIR"
