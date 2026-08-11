# UMI Flow

UMI Flow records synchronized bimanual demonstrations from two handheld UMI clients and produces a local 12-D [LeRobot](https://github.com/huggingface/lerobot) dataset.

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
git clone https://github.com/AfterQuery-Research/umi-flow.git
cd umi-flow
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --package umi-flow-core
```

Replace the two camera calibrations at these fixed paths:

```text
config/calib/umi1_calib.npz
config/calib/umi2_calib.npz
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

Set its four values:

```sh
SERVER="<server-user>@<server-address>"
LEADER="<leader-user>@<leader-address>"
FOLLOWER="<follower-user>@<follower-address>"
TASK="<task name>"
```

Install the processing server, then both clients:

```sh
./bin/provision-server
./bin/provision-pair
```

The server installer selects GPU or CPU GLIM automatically. The client installer detects each camera, assigns roles and UMI names, reads tag IDs from the hardware config, sets ports and calibration paths, configures clock synchronization, and reboots both Pis.

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
| Paired recordings | `~/umi-flow/data/raw_episodes/` on the server |
| Processing outputs | `~/umi-flow/data/derived/` on the server |
| LeRobot dataset | `~/umi-flow/data/lerobot/local/umi-flow/` on the server |

A take is complete when `data/derived/<paired-session>/dataset/COMPLETE` exists.

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
