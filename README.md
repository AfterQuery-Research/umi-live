# UMI Flow

UMI Flow records synchronized bimanual demonstrations from two handheld UMI clients and processes them into a 12-D [LeRobot](https://github.com/huggingface/lerobot) dataset.

## Requirements

| Device | Requirements |
| --- | --- |
| Leader client | Raspberry Pi 5, Ubuntu Server 24.04 arm64, 64 GB or larger microSD card |
| Follower client | Raspberry Pi 5, Ubuntu Server 24.04 arm64, 64 GB or larger microSD card |
| Processing server | x86_64 Ubuntu 24.04; NVIDIA GPU optional |

Each client uses:

- [RoboSense Airy](https://store.robosense.ai/products/airy) lidar
- [Arducam B0495 AR0234](https://www.arducam.com/downloads/datasheet/B0495_AR0234_Global_Shutter_USB3.0_Camera_Module_Datasheet.pdf) USB global-shutter camera
- Two printed [ArUco tags](tools/make_gripper_tags.py) per client: IDs `0` and `1` for the leader, and `2` and `3` for the follower
- Network connection to the other client and server

Run the setup commands from any macOS or Linux machine with Git, SSH, rsync, and network access to all three devices. The clients and server must have internet access during installation.

## Roles

| Role | Arm | Dataset name | Controls |
| --- | --- | --- | --- |
| Leader | Left | `umi1` | Record buttons and LCD |
| Follower | Right | `umi2` | Headless; controlled by the leader |

The leader sends start and stop commands to the follower. Each client uploads one recording. The server pairs them by session ID, runs GLIM for both arms, extracts gripper state, and creates one dataset episode. The current dataset timeline and origin use `umi1`.

Required connections:

| From | To | Port |
| --- | --- | --- |
| Setup machine | Clients and server | TCP 22 |
| Leader | Follower | TCP 9140 |
| Clients | Server | TCP 8080, or TCP 22 when using an SSH tunnel |
| Clients and server | Internet | TCP 443 |

## Install

Clone the repository once on the setup machine. The install commands copy it to the server and both clients.

```sh
git clone https://github.com/AfterQuery-Research/umi-flow.git
cd umi-flow
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --package robot-umi-core
```

### 1. Configure the server

```sh
cp provisioning/secrets.example provisioning/secrets
cp provisioning/server.env.example provisioning/server.env
```

Set these values in `provisioning/secrets`:

```sh
PI_PASSWORD="<client sudo password>"
UPLOAD_TOKEN="<output of: openssl rand -hex 32>"
HF_TOKEN=""
```

`UPLOAD_TOKEN` is shared automatically with the server and both clients. It is required by the current ingest service.

Set these values in `provisioning/server.env`:

```sh
REPO_ID="local/umi-flow"
PUSH_TO_HUB="0"
PUBLIC="0"
RUNNER_ARGS="--pair"
```

Install the server:

```sh
./bin/provision-server <server-user>@<server-ip>
```

The installer uses the NVIDIA GPU when available and otherwise builds GLIM for CPU processing.

### 2. Configure the clients

Create one inventory file per client. Each filename must match the host used in the provision command.

```sh
cp provisioning/rigs/example.env provisioning/rigs/<leader-host>.env
cp provisioning/rigs/example.env provisioning/rigs/<follower-host>.env
```

Set the following fields. Leave `SERVER_URL` empty when using `SERVER_SSH`; leave `SERVER_SSH` empty when port 8080 is directly reachable.

| Field | Leader | Follower |
| --- | --- | --- |
| `RIG_TARGET` | `gpu` | `gpu` |
| `SERVER_SSH` | `<server-user>@<server-ip>` | `<server-user>@<server-ip>` |
| `SERVER_URL` | `http://<server-ip>:8080/upload` | `http://<server-ip>:8080/upload` |
| `ALLOW_SD_SPOOL` | `1` | `1` |
| `SPOOL_DIR` | `/home/<leader-user>/umi-recordings` | `/home/<follower-user>/umi-recordings` |
| `UMI_NAME` | `umi1` | `umi2` |
| `CAMERA_DEVICE` | blank for auto-detection | blank for auto-detection |
| `CAMERA_CALIB` | `config/calib/umi1_calib.npz` | `config/calib/umi2_calib.npz` |
| `LIDAR_TYPE` | `robosense` | `robosense` |
| `GRIPPER_TAGS` | IDs mounted on the left UMI | IDs mounted on the right UMI |
| `LCD_KIND` | `lcd` | `none` |
| `ROLE` | `leader` | `follower` |
| `PEER_HOST` | `<follower-ip>` | empty |
| `TIME_REF` | same reachable NTP server | same reachable NTP server |

Install the follower first, then the leader:

```sh
./bin/provision-pair <leader-user>@<leader-host> <follower-user>@<follower-host>
```

Both clients reboot at the end of installation.

## Verify

```sh
ssh <server-user>@<server-ip> 'systemctl is-active umi-runner && curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8080/upload'
ssh <leader-user>@<leader-host> 'systemctl is-active umi-recorder'
ssh <follower-user>@<follower-host> 'systemctl is-active umi-recorder'
```

Expected output: active services and HTTP `401` from the unauthenticated server probe.

## Record

Tap the leader record button to start. Move both UMIs through the shared workspace for at least 30 seconds, keeping both ArUco pairs visible. Tap the button again to stop.

Follow processing on the server:

```sh
ssh <server-user>@<server-ip> 'journalctl -u umi-runner -f'
```

Expected stages:

```text
pair -> decode -> glim -> gripper -> dataset
```

## Results

| Data | Location |
| --- | --- |
| Client recordings | `<SPOOL_DIR>/sent/` |
| Uploaded episodes | `~/umi-flow/data/raw_episodes/` on the server |
| Processing outputs | `~/umi-flow/data/derived/` on the server |
| LeRobot dataset | `~/umi-flow/data/lerobot/local/umi-flow/` on the server |

The dataset episode is complete when its derived directory contains `dataset/COMPLETE`.
