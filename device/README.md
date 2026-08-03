# robot-umi-device

The Raspberry Pi recorder: ROS-free capture of 1–2 UMIs to hermetic, PTP-timestamped
episodes, spooled to the SSD and uploaded to the [server](../server/README.md). Part
of the [robot-umi](../README.md) uv workspace.

## Setup

Don't set this up by hand — [`install_pi.sh`](scripts/install_pi.sh) does the whole
fresh-Pi bring-up (see the [top-level README](../README.md#setup)). It handles the
pieces that used to be tribal knowledge: `setcap` on tcpdump, the SPI overlay + GPIO
groups, the lidar network + PTP units, the Rust recorder build, and a `umi-recorder`
systemd unit driven by the repo-root `.env`.

## What runs here

One `record.py` session wires the whole stack from one rig config (`config/umi.yaml`):

- **camera** — `v4l2_recorder/` (Rust): MJPEG capture with monotonic start-of-exposure
  timestamps, `--exposure` fixed-shutter control, per-frame stats + drop detection.
- **lidar** — `PcapCapture`: raw wire packets to a standard `.pcap` via `tcpdump`
  (needs the capture capability `install_pi.sh` grants). Passive: unplug = packet gap.
- **vive** (optional) — `NetworkPoseSensor`: kernel-stamped UDP pose datagrams, one
  per-tracker queue per UMI, raw samples to `<key>.jsonl` (EE calibration is server-side).
- **buttons + LCD** — GPIO buttons (gpiozero/lgpio) and the ST7789 panel: REC/IDLE,
  per-lidar type/**temperature**, mode, pending-episode + upload status.
- **spool + upload** — resilient episode spool (`in_progress/` → `complete/` → `sent/`,
  with `staging/` for reused upload tars) and a background uploader with retry. A
  camera/lidar failure mid-episode never discards data — the episode is kept and
  marked `dataset_eligible: false`.

Every sensor sits behind one `SensorRecorder` Protocol; N UMIs × M sensors is config
data, not code branches. `--mock-lidar` swaps in a synthetic-pcap recorder for
dev/CI (real RoboSense-shaped packets, no lidar hardware needed).

## Run

The `umi-recorder` service is the normal path (`sudo systemctl enable --now umi-recorder`),
and it loads `.env` via the unit's `EnvironmentFile`. For a manual dev session with
keyboard controls instead of GPIO, `record.py` reads only the process environment —
export `.env` yourself first, from the repo root:

```sh
set -a; . .env; set +a
.venv/bin/python device/scripts/record.py --buttons keyboard
```

Keys / buttons: `s`/tap = start/stop, `c`/hold = cancel, `q`/quit = end session,
`m`/mode-tap = cycle **lidar → vive → lidar+vive** (idle only). Server URL, rig config,
and spool dir come from `ROBOT_UMI_SERVER_URL` / `ROBOT_UMI_CONFIG` /
`ROBOT_UMI_SPOOL_DIR`; any can be overridden with the matching CLI flag.

## Test

```sh
uv run --package robot-umi-device pytest device/tests -q
```
