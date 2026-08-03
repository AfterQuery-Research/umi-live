# umi-flow

UMI-style handheld data collection with **offline GPU SLAM**: Raspberry Pi rigs
(camera + RoboSense lidar, one arm each; a coordinated leader/follower pair for
bimanual) record raw, PTP-timestamped episodes to an SSD and upload them; a
processing host (GPU box, or a Mac in CPU mode) decodes, runs **GLIM**
(lidar-inertial SLAM with loop closure) offline, registers the two arms into one
frame, extracts ArUco gripper widths, and builds
[LeRobot](https://github.com/huggingface/lerobot) datasets pushed to the
HuggingFace Hub.

> **Setting this up? Start at [SETUP.md](SETUP.md)** — the step-by-step guide
> (which value goes in which file, bimanual pairing, target switching,
> troubleshooting). The sections below are the architecture reference.

| Package | Runs on | What it does |
| --- | --- | --- |
| [`common/robot-umi-core`](common/robot-umi-core) | both | shared truths: episode layout, manifest v2, rig config + schema derivation, pose geometry (numpy+pyyaml only) |
| [`device/`](device/README.md) | Raspberry Pi (ROS-free) | pcap lidar capture, Rust V4L2 camera, Vive receiver, buttons + LCD (temp/mode), episode spooling, resilient upload |
| [`server/`](server/README.md) | GPU box / cloud | ingest inbox, pipeline runner (filesystem state machine), stage containers (decode → GLIM/LVI-SAM → gripper → dataset → compare) |

## Architecture principles

1. **Hermetic episodes** — each episode dir carries its raw streams + a config/calib
   snapshot + a versioned manifest, all checksummed; any archived episode reprocesses
   identically forever.
2. **Record raw, process later** — the Pi writes verbatim wire formats (pcap, MJPEG,
   raw Vive samples) on one PTP time axis; all decode/geometry/SLAM/dataset math is
   server-side.
3. **Standard formats** — pcap, rosbag2, TUM, LeRobot v3, evo.
4. **One home for shared truths** — if device and server both need it, it lives in
   `robot-umi-core` exactly once.
5. **The filesystem is the queue** — stages communicate through directories
   (`.staging` → atomic rename → `COMPLETE`); recovery is a rescan.
6. **Presence over mode** — the server dispatches on which streams exist in the
   episode, never on the device's mode label.
7. **One sensor seam** — every sensor implements the same `SensorRecorder` Protocol;
   1-vs-2 UMIs is config data (`state_dim = 6·N` is derived, never hardcoded).

## Setup

Follow **[SETUP.md](SETUP.md)**. In brief: the processing host gets `.env` +
`server/scripts/install_server.sh` (or `mac/run_processing.sh`); each Pi is
provisioned **from your workstation** with `./provision_pi.sh <user@pi>`
(`./provision_pair.sh` for a bimanual pair), driven by `provisioning/secrets` +
`provisioning/rigs/<pi>.env`; already-provisioned Pis flip between processing
hosts with `./switch_target.sh --target mac|gpu <user@pi>...`.

Do **not** hand-configure a Pi (clone + `install_pi.sh` + editing files on it) —
that path is the provisioner's internal machinery and skips the SSD mount, upload
tunnel, chrony sync, and required reboot.

## Configure the rig

`config/umi.yaml` (seeded by the installer from `umi.single.example.yaml`, or
`umi.dual.example.yaml` for two UMIs) declares the rig. `N = len(umis)` drives
everything (state dim, feature names, capture plan). Camera fisheye calibs live in
`config/calib/`; per-UMI Vive EE calibs in `config/ee_calib.umiN.json`. Unknown keys
are rejected (the file is snapshotted into every episode — no stray secrets).

## Record

Buttons: record tap = start/stop, record hold = cancel, quit hold = end session,
**mode tap = cycle lidar → vive → lidar+vive** (idle only). The LCD shows per-lidar
presence/type/**temperature**, the mode, REC/IDLE, pending episodes, and upload state.
Episodes spool to the SSD and upload to the server; the runner processes each one as
it lands.

## Pipeline (server)

```
raw_episodes/<id>/                      derived/<id>/
  umiN_lidar.pcap        ── decode ──►    umiN/decode/bag        (rslidar pcap replay, per-point t)
  (bag)                  ── glim ────►    umiN/glim/trajectory.tum
  umiN_camera.mjpeg.avi  ── gripper ─►    gripper/umiN_gripper.jsonl
  everything             ── dataset ─►    dataset/receipt.json + LeRobot append (+ HF push)
  trajectories           ── compare ─►    compare/metrics.json   (evo ATE/RPE, LI vs LIV)
```

`server/scripts/runner.py` (the `umi-runner` service) drives this: **decode** and
**glim** run as `docker compose run --rm` stage containers (see `docker/`),
**gripper** and **dataset** run in-process. GLIM's known RoboSense-Airy timestamp
pitfall (upstream issue [#257](https://github.com/koide3/glim/issues/257)) and its
per-unit IMU extrinsic are handled in decode + the GLIM config. The CPU fallback for
the shipped pipeline is **GLIM-CPU** (`install_server.sh` writes the compose override
automatically on a GPU-less box).

**Scaffolded, not yet wired into the runner:** **Livox decode** (only RoboSense pcap
replay is implemented — the decode stage refuses Livox episodes, though capture to
pcap works), the **LVI-SAM** lidar-inertial-visual engine, the evo **compare** stage
(`docker/compare/`), and **FAST-LIO2** (`docker/slam-fastlio/`, proven on hardware but
not containerized). These are future LI-vs-LIV / multi-lidar work, not shipped features.

## Test

```sh
uv run --package robot-umi-core pytest common/robot-umi-core/tests -q
uv run --package robot-umi-device pytest device/tests -q
cd server && uv run --package robot-umi-server pytest -q   # incl. device→dataset e2e
```

The CI mock lidar emits synthetic RoboSense-shaped **pcaps**, so mock and hardware
exercise the same pipeline bytes.

## Acknowledgments

The device/server skeleton (episode spooling, resilient upload, secure ingest,
Rust V4L2 recorder) is based on the architecture of `branyang02/robot-umi`.
Sensor/UX/SLAM logic (buttons, ST7789 LCD, Vive receiver, fisheye/ArUco gripper
math, RoboSense bring-up) is carried over from the AfterQuery robot-umi rigs.
