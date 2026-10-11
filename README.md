# UMI-Live

UMI-Live records synchronized bimanual demonstrations from two handheld UMI clients and produces a local [LeRobot](https://github.com/huggingface/lerobot) dataset. Each arm contributes a 6-D pose plus a calibrated 0–1 gripper value to the 14-D `observation.state`; pose-delta `action` remains 12-D.

## Requirements

| Device | Hardware | Operating system |
| --- | --- | --- |
| Leader client | Raspberry Pi 5; 64 GB or larger microSD card | Ubuntu Server 24.04 arm64 |
| Follower client | Raspberry Pi 5; 64 GB or larger microSD card | Ubuntu Server 24.04 arm64 |
| Processing server | x86_64 machine; NVIDIA GPU optional | Ubuntu 24.04 |

Each client requires:

- [RoboSense Airy](https://store.robosense.ai/products/airy) lidar
- [Arducam B0495 AR0234](https://www.arducam.com/downloads/datasheet/B0495_AR0234_Global_Shutter_USB3.0_Camera_Module_Datasheet.pdf) USB global-shutter camera
- Two printed [ArUco gripper tags](tools/make_gripper_tags.py), using the per-arm pairs in [config/gripper_tags.yaml](config/gripper_tags.yaml)

| Role | Arm | Dataset name | Controls |
| --- | --- | --- | --- |
| Leader | Left | `umi1` | Record button and LCD |
| Follower | Right | `umi2` | Controlled by the leader |

The leader refuses to record unless the follower acknowledges the take. Both clients spool to their microSD cards and upload through SSH tunnels. The server pairs the two uploads, runs GLIM for both arms, extracts gripper state, and appends one local dataset episode.

Before installation:

- The machine running these commands can reach the server and both clients over SSH.
- The leader can reach the follower; both clients can reach the processing server over SSH.
- All three devices have internet access during installation.
- The processing-server user has passwordless `sudo`.

Network setup and SSH key management are outside this guide.

## Install

Clone only on the machine running setup. The provisioning commands copy the repository to all three devices.

```sh
git clone https://github.com/AfterQuery-Research/umi-live.git
cd umi-live
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --package umi-live-core
```

The physical gripper-label assignments are defined once in `config/gripper_tags.yaml`.
Its checked-in values match this rig. If the labels are replaced, update that file before
generating a new tag sheet or provisioning either client.

```sh
uv run --group tools python tools/make_gripper_tags.py
```

Create the pair configuration:

```sh
cp provisioning/pair.env.example provisioning/pair.env
```

Set its deployment values:

```sh
SERVER="<server-user>@<server-address>"
LEADER="<leader-user>@<leader-address>"
FOLLOWER="<follower-user>@<follower-address>"
TASK="<task name>"
ALIGNMENT_ENABLED="true"
LEADER_HOTSPOT_ENABLED="true"
```

Set `ALIGNMENT_ENABLED=true` to register both LiDAR trajectories into umi1's metric
frame during processing. Set it to `false` to retain independent episode-relative frames.

Set `LEADER_HOTSPOT_ENABLED=false` to use the follower address in `FOLLOWER` instead. When
enabled, the leader hosts the private pair network at `10.42.0.1`; the follower joins at
`10.42.0.2`. Recording coordination and follower clock sync therefore work without
infrastructure Wi-Fi. Provisioning automatically uses the leader's sole USB Wi-Fi adapter,
keeps its built-in `wlan0` as the uplink, and generates a private credential that is reused
from the gitignored `provisioning/.pair-wifi-psk`. The follower dedicates `wlan0` to the
private link and uploads through the leader.

Install the processing server, then both clients:

On freshly flashed Pis, wait for Ubuntu's first-boot upgrades to finish before running `provision-pair`, or `apt` may fail because `unattended-upgr` holds the package-manager lock.

```sh
./bin/provision-server
./bin/provision-pair
```

The server installer selects GPU or CPU GLIM automatically. The client installer detects each camera, assigns roles and UMI names, reads tag IDs from the hardware config, sets ports and calibration paths, configures clock synchronization, and reboots both Pis.

The repository includes default camera intrinsics and provisioning deploys them to both
clients. To replace them with intrinsics calibrated for your cameras, run:

```sh
./bin/calibrate-pair
```

This opens the [ChArUco board](tools/make_charuco.py) and a live coverage viewer for each
idle camera. Sweep the board until every calibration passes; the command then atomically updates
`config/calib/umi1_calib.npz`, `config/calib/umi2_calib.npz`, and their corresponding clients.

Before collecting a new dataset, calibrate each physical gripper's fully closed/open endpoints:

```sh
uv run --package umi-live-server python \
  dataset-filtering/scripts/calibrate_gripper_range.py --gripper umi1
uv run --package umi-live-server python \
  dataset-filtering/scripts/calibrate_gripper_range.py --gripper umi2
./bin/provision-pair
```

Each calibration asks you to close the selected gripper and press Enter, then open it and press
Enter. Provisioning embeds those endpoints in every recording snapshot. The server clips the
physical width to `[0, 1]` and stores it after that arm's pose in `observation.state`.

## Verify

```sh
ssh <server-user>@<server-address> 'systemctl is-active umi-runner && curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/upload'
ssh <leader-user>@<leader-address> 'systemctl is-active umi-recorder && systemctl is-active umi-upload-tunnel'
ssh <follower-user>@<follower-address> 'systemctl is-active umi-recorder && systemctl is-active umi-upload-tunnel'
```

Expected output: every service is `active`; the empty upload returns HTTP `400`.

## Record

Tap the leader record button to start both clients. Tap it again to stop. Repeat for additional episodes.

Before either client records, each camera checks five frames for its configured gripper-tag
pair. Every tag must appear at least once in that window. If a tag is missing, the take does
not start and the leader LCD identifies the UMI and missing tag IDs.

Follow processing on the server:

```sh
ssh <server-user>@<server-address> 'journalctl -u umi-runner -f'
```

Each take runs through `pair -> decode -> GLIM -> gripper -> dataset`.

## Results

| Data | Location |
| --- | --- |
| Client recordings | `~/umi-recordings/sent/` on each client |
| Paired recordings | `~/umi-live/data/raw_episodes/` on the server |
| Processing outputs | `~/umi-live/data/derived/` on the server |
| LeRobot dataset | `~/umi-live/data/lerobot/local/umi-live/` on the server |

A take is complete when `data/derived/<paired-session>/dataset/COMPLETE` exists.
One hour later, the server automatically removes its duplicate paired sources and
decoded bags; canonical merged raw data, SLAM outputs, and the published dataset remain.

## Publishing to HuggingFace (optional)

Set `HF_REPO_ID` (e.g. `youruser/umi-bimanual`) and `HF_TOKEN` (a write-scope token) in
`provisioning/pair.env` and re-run `bin/provision-server`. Every take then mirrors the
LeRobot dataset to that Hub repo (created private on first push) and archives the raw
episode under `raw_episodes/` — the offsite copy of everything. Blank fields (the
default) keep the dataset local-only. A failed push never loses the local append; the
next take's push re-syncs.

For a new physical pair or collection campaign, set `DATA_NAMESPACE` as well. This gives
the server an isolated raw/derived/LeRobot data tree, so publishing to a new Hub repo cannot
mirror episodes from an earlier collection. Existing data is left untouched.
