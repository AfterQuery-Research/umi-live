# robot-umi-server

The GPU processing box: a durable ingest inbox plus a filesystem-state-machine
pipeline runner that turns uploaded episodes into LeRobot datasets. Part of the
[robot-umi](../README.md) uv workspace.

## Setup

Don't set this up by hand — [`install_server.sh`](scripts/install_server.sh) does the
whole bring-up (see the [top-level README](../README.md#setup)): Docker (+
nvidia-container-toolkit on GPU hosts), the stage images, the host Python env the
runner needs, `DATA_DIR`, the always-up ingest container, and the `umi-runner`
systemd service. It detects GPU vs CPU and works the same on a Lambda instance or an
on-prem NVIDIA machine.

## The two processes

- **`umi-ingest`** (container, always up on `:8080`) — extract + validate + publish,
  nothing else. It streams an uploaded `.tar` to `data/uploads/`, then a worker
  extracts it into a quarantine dir, **verifies checksums there**, and only then
  atomic-renames it into `data/raw_episodes/<id>/`. No ROS/CUDA/lerobot here. Refuses
  to bind `0.0.0.0` without `ROBOT_UMI_UPLOAD_TOKEN` (shared with the Pi).

- **`umi-runner`** (`server/scripts/runner.py`, systemd service) — the filesystem
  state machine. It rescans `raw_episodes/`, and for each episode runs the stages
  whose inputs are `COMPLETE` and outputs aren't: **decode** and **glim** as
  `docker compose run --rm` stage containers, **gripper** and **dataset** in-process.
  Recovery is the rescan; stages meet only at directories (`.staging` → rename →
  `COMPLETE`). The dataset stage appends to a LeRobot dataset (idempotent via an
  append-ledger) and, when `ROBOT_UMI_PUSH_TO_HUB=1`, pushes to a private HF repo.

Dispatch is by stream **presence** in the manifest, never the mode label.

## Configuration

All from the repo-root `.env` (see `.env.example`): `ROBOT_UMI_UPLOAD_TOKEN`,
`DATA_DIR`, `ROBOT_UMI_REPO_ID`, `HF_TOKEN`, `ROBOT_UMI_PUSH_TO_HUB`,
`ROBOT_UMI_PUBLIC`, `ROBOT_UMI_DATASET_ROOT`, `ROBOT_UMI_ENGINE`. Deployment knobs
live here, never in the rig config.

## Manual operation

```sh
docker compose up -d umi-ingest                 # ingest only
sudo systemctl status umi-runner                # runner service
journalctl -u umi-runner -f                     # runner logs
```

## Test

```sh
uv run --package robot-umi-server pytest server/tests -q   # incl. device→dataset e2e
```

## Data layout

```
DATA_DIR/
  uploads/                     upload tars (transient)
  raw_episodes/<id>/           hermetic episodes (manifest, config_snapshot/, umiN_*.pcap/avi/jsonl)
  derived/<id>/<umi>/decode/   rslidar pcap replay -> rosbag2
  derived/<id>/<umi>/glim/     trajectory.tum
  derived/<id>/gripper/        umiN_gripper.jsonl
  derived/<id>/dataset/        receipt.json (+ LeRobot append under ROBOT_UMI_DATASET_ROOT)
```
