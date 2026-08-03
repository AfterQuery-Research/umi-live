# UMI Flow — Setup

End-to-end setup for the UMI data-collection stack: handheld gripper rigs (Raspberry Pi +
lidar + camera) that record demonstration episodes and stream them to a processing host,
which runs SLAM and publishes LeRobot datasets to the HuggingFace Hub.

```
 Pi rig(s)                    processing host (pick ONE)              output
┌──────────────┐  upload     ┌─────────────────────────────┐
│ leader  Pi   │ ──────────► │ GPU box  (CUDA GLIM)   or   │   12-D (bimanual) or 6-D
│ follower Pi  │  http/ssh   │ Mac      (CPU GLIM)         │ ► LeRobot dataset
└──────────────┘  tunnel     │ ingest → pair → decode →    │   → HuggingFace Hub
  one lidar+camera each      │ GLIM → gripper → align →    │
  buttons + LCD on leader    │ dataset → HF push           │
                             └─────────────────────────────┘
```

Everything is driven from **your workstation** (laptop/Mac). You never hand-edit files on a
Pi: `bin/provision-pi` renders and pushes them. (Ignore any older instructions telling you to
clone + configure on the Pi itself — that path skips the SSD mount, upload tunnel, chrony
sync, and the required reboot.)

---

## 0. What goes where (read this first)

**All configuration lives in one inventory on your workstation** — three small files under
`provisioning/`. The provisioning scripts render and push each machine's config; you never
hand-edit a file on a Pi or the box.

| Inventory file (workstation)       | What goes in it | Fanned out to |
|------------------------------------|-----------------|---------------|
| `provisioning/secrets`             | ALL secrets: `PI_PASSWORD` (Pi sudo), `UPLOAD_TOKEN` (upload auth), `HF_TOKEN` (Hub push) | token → box **and** every Pi; HF token → box only, never a Pi |
| `provisioning/server.env`          | box settings: `REPO_ID` (HF dataset), `PUSH_TO_HUB`, `PUBLIC`, `RUNNER_ARGS` (`--pair` for bimanual) | the processing box |
| `provisioning/rigs/<pi-host>.env`  | one file per Pi: rig hardware + **where it uploads** | that Pi |

Concretely:

- **The processing box's IP** appears in exactly two commands/fields:
  `./bin/provision-server ubuntu@<box-ip>` (sets the box up), and the Pi rig file's
  `SERVER_SSH="ubuntu@<box-ip>"` (cloud box where only port 22 is open — installs a
  boot-persistent SSH tunnel) **or** `SERVER_URL="http://<box-ip>:8080/upload"` (LAN box
  with 8080 reachable). Never both fields.
- **The Mac's `user@host`** goes in the rig file as `MAC_HOST` (only for `RIG_TARGET=mac`).
- **The follower Pi's IP** goes in the *leader's* rig file as `PEER_HOST`.
- **The upload token is written once**, in `provisioning/secrets`; `bin/provision-server`
  and `bin/provision-pi` deliver it to both sides, so they cannot mismatch. (If you ever
  bypass the provisioners and hand-edit a `.env`, a token mismatch fails *silently* —
  recording works, uploads 401-loop forever.)
- **The HuggingFace token** also lives in `provisioning/secrets`; by design it is fanned
  out ONLY to the processing host, never to a Pi.

---

## 1. Prerequisites

**Hardware (per arm):** Raspberry Pi 5 + SD card, RoboSense Airy lidar (on the Pi's `eth0`),
UVC USB camera (MJPEG-capable; width:height must be 4:3 when gripper tags are used), USB SSD
for the episode spool, gripper with two ArUco tags (`tools/make_gripper_tags.py` prints them;
tag ids must be **disjoint across arms** — e.g. umi1 `[0,1]`, umi2 `[2,3]`). The leader rig
also has 3 push buttons + an ST7789 LCD.

**Processing host:** an x86_64 Linux box with an NVIDIA GPU (CUDA GLIM), or a Mac
(CPU GLIM, slower). The GLIM images are **amd64-only** — an ARM cloud box will not work as
the `gpu` target.

**Accounts:** a HuggingFace account and a **write**-scope token (a read token fails only at
push time, after a full episode of processing).

**Network reachability** (check with `nc -vz <host> <port>` before starting):

| From → To | Port | Why |
|---|---|---|
| workstation → each Pi | 22 | provisioning |
| workstation → box | 22 | box install + key setup |
| Pi → box | 22 (cloud) or 8080 (LAN) | episode upload |
| leader Pi → follower Pi | TCP 9140 | record coordination |
| Pi → internet | 443 | apt, uv, docker install |
| box → internet | 443 | HF push, docker pulls |

**Per-camera calibration:** each camera needs its own fisheye calib `.npz`, referenced by
`CAMERA_CALIB` in the rig file (the repo ships calibs for the two original cameras only).
For a new camera, wave a checkerboard in front of it — covering the whole frame, especially
edges/corners — and run:

```sh
uv run tools/calibrate_camera.py --video <recording.avi> --board 9x6 \
    --out config/calib/<umiN>_calib.npz
```

The natural capture flow: record a throwaway episode while waving the board and point the
tool at its `umiN_camera.mjpeg.avi` (`--images 'shots/*.jpg'` also works). Aim for 20+
accepted views and RMS < 1 px; the tool warns otherwise. Do **not** reuse another camera's
file; it silently corrupts gripper geometry.

---

## 2. Processing host — GPU box

Fill the inventory on your workstation, then one command does the whole box:

```sh
cp provisioning/secrets.example    provisioning/secrets      # UPLOAD_TOKEN (openssl rand -hex 32),
                                                             # HF_TOKEN, PI_PASSWORD
cp provisioning/server.env.example provisioning/server.env   # REPO_ID=<user>/<dataset>,
                                                             # RUNNER_ARGS="--pair" for bimanual
./bin/provision-server ubuntu@<box-ip>
```

`bin/provision-server` needs only SSH key access + passwordless sudo on the box (standard on
cloud images). It rsyncs the repo (never the inventory — secrets stay on your workstation
except the two values the box needs), renders the box's `.env` from the inventory, and runs
`server/scripts/install_server.sh` remotely. The installer is idempotent and handles: Docker
(+ the compose v2 plugin — pre-installed docker on cloud images usually lacks it), the
NVIDIA container toolkit, the stage images, the host venv **including the align extra**
(open3d + gtsam, required for bimanual), the always-up ingest on `:8080`, and the
`umi-runner` systemd service. It ends with a self-test.

Changing a setting later (repo id, pair mode, rotating the token): edit the inventory and
re-run `./bin/provision-server` — don't edit files on the box. A token rotation also needs
`./bin/provision-pi` re-run per Pi (same single source).

Sanity check:

```sh
ssh ubuntu@<box-ip> "curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8080/upload"  # 401 = up
ssh ubuntu@<box-ip> "journalctl -u umi-runner -n 5"   # "... watching .../raw_episodes (pairing=True)"
```

### Alternative: Mac as the processing host

The Mac host is usually this same workstation, so `--local` just renders `./.env` from the
inventory (same single-source token), then you run the pipeline in the foreground:

```sh
./bin/provision-server --local
uv sync --package robot-umi-server
uv pip install --prerelease=allow "open3d>=0.18" "gtsam>=4.3a0"   # bimanual align
PAIR=1 INGEST_HOST=0.0.0.0 mac/run_processing.sh
```

Two Mac-only caveats: the first run builds the arm64 CPU GLIM image (~1 h), and
`run_processing.sh` is a **foreground** process — nothing restarts it for you. Keep it
running (`caffeinate -i mac/run_processing.sh`) or episodes queue on the Pis until it's back.
The Mac also needs Remote Login (sshd) ON for the Pi tunnel, and note that a DHCP Mac's IP
changes — prefer the tunnel (`RIG_TARGET=mac`) over a direct URL.

---

## 3. Recorder Pis

Flash Ubuntu Server 24.04 (arm64) with your SSH key, boot, and confirm `ssh <user>@<pi>`
works. The secrets file already exists from step 2 (the Pis reuse the same
`UPLOAD_TOKEN` — the provisioner delivers it, no re-entry). Then, on your workstation:

```sh
cp provisioning/rigs/example.env provisioning/rigs/<pi-host>.env
# minimum edits: RIG_TARGET (+ SERVER_SSH / SERVER_URL / MAC_HOST), UMI_NAME, CAMERA_CALIB,
# GRIPPER_TAGS. For a pair, also ROLE / PEER_HOST / TIME_REF (see below).

./bin/provision-pi --target gpu <user>@<pi-host>
```

`bin/provision-pi` does everything: rsyncs the repo, renders + validates `config/umi.yaml`
and the Pi `.env`, mounts the SSD (fstab, `nofail`; exfat gets uid/gid mount options and
`passno=0` so a dirty volume can't stall boot), runs the on-Pi installer (~20-40 min:
apt/rust/venv/lidar network/PTP/systemd), installs the upload tunnel, then **reboots** (the
reboot is required — GPIO/SPI group membership and the LCD overlay only apply at boot) and
verifies. Plug the camera in **before** provisioning; a missing camera is only a soft
warning ("CHECK camera") and you'd otherwise discover it at record time.

### Bimanual pair

Give each Pi its own rig file, using the pair example configs:

```sh
# leader (has buttons + LCD, drives the take):
#   ROLE="leader"  PEER_HOST="<follower-ip>"  TIME_REF="<any common NTP host, e.g. the box>"
#   RIG_CONFIG_FILE="config/umi.pair-leader.example.yaml"   # umi1, tags [0,1]
# follower:
#   ROLE="follower"  TIME_REF="<same>"
#   RIG_CONFIG_FILE="config/umi.pair-follower.example.yaml" # umi2, tags [2,3]
./bin/provision-pair <user>@<leader> <user>@<follower>
```

The pair contract, spelled out once: the leader dials the follower on TCP `9140`
(`CONTROL_PORT`, must match on both and not be firewalled); both Pis chrony-sync to
`TIME_REF` so their clocks are comparable; the follower records headless (its LCD, if any,
shows only lidar health); and the **server** merges the two uploads into one 12-D episode —
which only happens if the runner has `--pair` (see step 2). Miss any one of these and you
get lone 6-D orphans, not bimanual data.

Second lidar note: every RoboSense Airy ships at `192.168.1.200`. Two lidars on two separate
Pis can both keep the factory IP (separate networks). Only a dual-lidar-on-one-Pi rig needs
the second unit re-IP'd via its web UI first.

---

## 4. Record and verify (do this before real data collection)

Buttons (leader only, in a pair): **record tap** = start/stop episode · **record hold 2 s** =
cancel take · **mode tap** = cycle mode · **quit hold** = end session + batch-upload. The LCD
shows per-lidar type/temp, REC/IDLE, pending-episode count, and an upload banner
(processing / uploading / ok / fail — a red fail retries automatically; nothing is lost).

Smoke test: record a ~30 s throwaway take, then watch it traverse the pipeline on the box:

```sh
journalctl -u umi-runner -f
# expect, in order:  pair: merged session ...   (pair rigs)
#                    decode(umi1) done ... decode(umi2) done
#                    glim(umi1) done   ... glim(umi2) done   (T_lidar_imu source=difop)
#                    gripper() done → align() done → dataset: appended N frames pushed=True
```

Then confirm the episode on the Hub (`https://huggingface.co/datasets/<ROBOT_UMI_REPO_ID>`):
`meta/info.json` should show your episode count and `observation.state` shape `[12]`
(bimanual) or `[6]` (solo). A useful visual check: `uv run mac/replay_grippers.py --repo
<repo> --episode 0` replays the recorded claws in 3D (Rerun).

Timing reference (H100-class GPU, ~30 s episode): decode ~50 s/arm, GLIM ~10 s/arm,
align ~1 min, dataset+push ~30 s — about 3-4 min end-to-end.

---

## 5. Switching processing hosts (Mac ↔ box)

Pis and the processing host are decoupled: each Pi just uploads wherever its `.env` points.
To move already-provisioned Pis without a full re-provision:

```sh
./bin/switch-target --target gpu <user>@<pi1> <user>@<pi2>   # → the box
./bin/switch-target --target mac <user>@<pi1> <user>@<pi2>   # → the Mac
```

It rewrites the Pi's upload URL, flips the tunnel unit, restarts the recorder, and verifies
the target answers through the tunnel. Episodes already spooled upload to the *new* target
automatically. Both targets can coexist; just don't point two processing hosts at the **same**
HF repo — the push mirrors the local dataset and would clobber whichever host pushed first
(one repo per processing host, or seed the new host's local dataset from the old one first).

---

## 6. Monitoring

| Where | Command |
|---|---|
| box: pipeline | `journalctl -u umi-runner -f` |
| box: uploads arriving | `docker compose logs -f umi-ingest` |
| box: stage state | `ls data/derived/<episode>/*/COMPLETE` |
| Pi: recorder | `journalctl -u umi-recorder -f` |
| Pi: tunnel | `systemctl status umi-upload-tunnel` |
| Pi: pending uploads | `ls <spool>/pending/` (uploaded episodes move to `sent/`) |
| Pi: cross-Pi clock | `chronyc tracking` (pair rigs; want <5 ms) |

Storage: episodes are ~0.5 GB each and accumulate in the Pi spool's `sent/` (safe to prune
once confirmed on the Hub) and in the box's `data/raw_episodes` + `data/derived` (prune old
`derived/` freely — it's all recomputable; `derived/` is root-owned, use sudo).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No lidar packets | NetworkManager wiped eth0's static IP (`nmcli con show`; installer pins a `lidar` connection) or `rp_filter` drops unicast (`sysctl net.ipv4.conf.all.rp_filter` must be 0). Quick check: `sudo tcpdump -ni eth0 udp port 6699 -c 3` |
| Recorder fails right after manual install (status=216/GROUP or GPIO EPERM) | You skipped the reboot. GPIO/SPI/video groups + the LCD overlay only apply at boot |
| Recording works, nothing reaches the box, LCD shows pending | Token mismatch (silent 401 loop) — only possible if a `.env` was hand-edited out of sync with the inventory; re-run `./bin/provision-server` + `./bin/provision-pi` to re-fan the single `UPLOAD_TOKEN`. Verify from the Pi: `curl -m5 -s -o /dev/null -w '%{http_code}' -H "X-Upload-Token: $tok" -X POST http://localhost:8080/upload` → 400 good, 401 mismatch, 000 tunnel down |
| Tunnel `activating`/restart-looping | Box host key changed (rebuilt box): clear `~/.ssh/known_hosts` on the Pi; or the Pi's key isn't in the box's `authorized_keys` (re-run `bin/provision-pi` or `bin/switch-target`) |
| Pair never merges; episodes land as lone arms | Runner missing `--pair` (`ROBOT_UMI_RUNNER_ARGS=--pair` + `systemctl restart umi-runner`); or role/UMI-name misprovision → check `data/failed_pairs/`; or the mate never uploaded → `data/orphans/` after the timeout |
| GLIM trajectory teleports / diverges | Lidar UDP capture gap mid-take (flaky cable/NIC): check the pcap for >200 ms holes. Also expected for episodes recorded <60 s after Pi boot (PTP not locked — manifest says `locked: false`) |
| Bimanual arms are misaligned by ~a meter in the dataset | Cross-lidar map registration locked onto the wrong alias of a repetitive room. Set `alignment.workspace_bbox` in the pair config to crop registration to the shared workspace; verify with `mac/replay_compare.py` |
| Every point timestamp off by exactly 37 s | Someone removed `utc_offset 0` from `config/ptp/umi-gptp.cfg` (TAI-UTC offset). Put it back |
| `align() FAILED: No module named 'open3d'` | The align extra isn't installed on the host venv (installer does it now): `uv pip install --prerelease=allow "open3d>=0.18" "gtsam>=4.3a0"` |
| `docker compose` = "unknown command" | Pre-installed docker without the compose v2 plugin — re-run `install_server.sh` (it now installs the plugin binary) |
| Runner: `decode(...) FAILED to clear stale output ... Permission denied` | Only happens when REPROCESSING an episode id: decode/GLIM containers run as root, so stale `derived/<id>` needs `sudo rm -rf` first |
| Dataset stage: `state_dim mismatch` | The HF repo's arm count is frozen at creation (6-D vs 12-D). Use a fresh `ROBOT_UMI_REPO_ID` |
| Hub dataset lost episodes after switching processing hosts | The push mirrors the host's local dataset (deletes remote files not present locally). Never point a fresh host at an existing repo without seeding its `data/lerobot/<repo>` from the old host first |
| SSD not found during provisioning | Set `SSD_DEVICE=/dev/sdX1` in the rig file; or `ALLOW_SD_SPOOL=1` (dev only — spools to the SD card) |

**Reprocessing an episode:** `sudo rm -rf data/derived/<id>` on the box, then restart
`umi-runner` — the episode in `raw_episodes/` reruns through every stage. The dataset stage
keeps an append ledger, so a reprocessed episode is *not* double-appended; to force a fresh
append, remove its line from `data/lerobot/<repo>/.umi_appended_episodes.txt` (and expect a
duplicate unless you wiped the dataset).

---

## 8. Out of scope

The HTC Vive tracking path (`motion_source: vive`) requires an external Windows/SteamVR box
running a pose sender that is not part of this repository. This repo's supported motion
source is **lidar SLAM**.
