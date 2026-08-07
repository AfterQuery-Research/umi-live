"""Dataset stage: paired trajectories, frames, and gripper widths -> local LeRobot v3.

Consumes ONLY per-episode artifacts (hermetic): the manifest + config snapshot, each
UMI's camera video/frames, GLIM ``trajectory.tum`` per lidar, and the gripper stage's
width jsonls. The LeRobot feature schema comes from
``robot_umi_core.config.derive()`` on the SNAPSHOT config — the schema exists in
exactly one place.

Timeline: the first UMI's camera frames (unix ns, PTP axis). Every pose track is
resampled onto it (interp1d + Slerp); a second UMI's video uses its nearest-past
frame. States are episode-relative, zeroed to umi1's first pose.

Cross-arm frame: each lidar is SLAM'd independently, so two lidars land in two
unrelated GLIM frames. Zeroing both to umi1's origin only yields metric cross-arm
geometry when both tracks already share a world frame. That holds when the optional
``align`` stage ran (its ``align/<umi>/trajectory.tum`` are pre-registered into one
common frame — read in preference here); without it (alignment disabled, the default)
the per-lidar GLIM trajectories are used directly and cross-arm geometry is only
correct up to each lidar's arbitrary start frame.

The dataset is always written under ``$DATA_DIR/lerobot/local/umi-flow``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from robot_umi_core import config as core_config
from robot_umi_core import geom, layout, lidar_clock, manifest
from scipy.spatial.transform import Rotation

from robot_umi_server.trajectory import (
    PoseTrack,
    actions_from_states,
    read_tum,
    resample,
    states_from_track,
)

DEFAULT_ENGINE = "glim"
LOCAL_REPO_ID = "local/umi-flow"
DEFAULT_RGB_ENCODER_VCODEC = "h264"


@dataclass(frozen=True)
class FrameStream:
    """One UMI's decoded video + per-frame unix timestamps (seconds)."""

    images: list[np.ndarray]
    t: np.ndarray


def decode_video(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        frames: list[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def load_frame_stream(episode_dir: Path, umi: str) -> FrameStream:
    files = layout.sensor_files(layout.sensor_key(umi, layout.KIND_CAMERA))
    images = decode_video(episode_dir / files["video"])
    ts = []
    for line in (episode_dir / files["frames"]).read_text().splitlines():
        if not line.strip():
            continue
        ts.append(int(json.loads(line)["time_unix_ns"]) / 1e9)
    if not images:
        raise ValueError(f"{umi}: video has no frames")
    if len(images) != len(ts):
        raise ValueError(
            f"{umi}: frames jsonl count {len(ts)} != decoded video frames {len(images)}"
        )
    return FrameStream(images=images, t=np.asarray(ts, dtype=np.float64))


def pose_track_for_umi(
    episode_dir: Path,
    derived_episode_dir: Path,
    m: dict[str, Any],
    umi: core_config.UmiConfig,
) -> tuple[PoseTrack, np.ndarray]:
    """This UMI's world poses + the body-convention basis to apply to its states.

    Lidar tracks carry the lidar-body -> UMI-convention basis."""
    lidar_key = layout.sensor_key(umi.name, layout.KIND_LIDAR)
    if lidar_key not in manifest.present_streams(m, layout.KIND_LIDAR):
        raise ValueError(f"{umi.name}: required lidar stream is missing")
    # Prefer the optional common-frame trajectory, otherwise use this arm's GLIM output.
    aligned_tum = derived_episode_dir / "align" / umi.name / "trajectory.tum"
    traj_tum = (
        aligned_tum
        if aligned_tum.is_file()
        else derived_episode_dir / umi.name / DEFAULT_ENGINE / "trajectory.tum"
    )
    track = read_tum(traj_tum)
    # GLIM stamps poses in the lidar clock; realign onto the host/camera clock.
    pcap = episode_dir / layout.sensor_files(lidar_key)["pcap"]
    skew = lidar_clock.pcap_host_skew(pcap) if pcap.is_file() else None
    if skew is not None:
        print(f"  {umi.name}: lidar->host clock skew {skew:+.3f}s (realigned)", flush=True)
        track = PoseTrack(t=track.t + skew, pos=track.pos, quat=track.quat)
    # Shift the tracked lidar body to the gripper end-effector (TCP).
    lever_body = geom.UMI_FROM_LIDAR_BODY.T @ geom.LIDAR_EE_LEVER_M
    ee_pos = track.pos + Rotation.from_quat(track.quat).apply(lever_body)
    track = PoseTrack(t=track.t, pos=ee_pos, quat=track.quat)
    return track, geom.UMI_FROM_LIDAR_BODY


def load_gripper_widths(
    derived_episode_dir: Path, umi: str, timeline_t: np.ndarray
) -> np.ndarray | None:
    path = derived_episode_dir / "gripper" / f"{umi}_gripper.jsonl"
    if not path.exists():
        return None
    ts, ws = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ts.append(int(d["time_unix_ns"]) / 1e9)
        ws.append(float(d["width_mm"]))
    if not ws:
        return None
    return np.interp(timeline_t, np.asarray(ts), np.asarray(ws)).astype(np.float32)


# The LeRobot append is a global side effect that lives OUTSIDE the
# runner's staging dir, so the runner's "tear down a stage dir without COMPLETE and
# re-run it" crash recovery would otherwise re-append the same episode's frames.
# This append-only ledger (one episode_id per line, next to the dataset) records
# what has already landed so a re-run skips cleanly. It travels with the dataset.
APPENDED_LEDGER = ".umi_appended_episodes.txt"


def already_appended(dataset_dir: Path, episode_id: str) -> bool:
    ledger = dataset_dir / APPENDED_LEDGER
    if not ledger.exists():
        return False
    return episode_id in ledger.read_text(encoding="utf-8").split()


def mark_appended(dataset_dir: Path, episode_id: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / APPENDED_LEDGER).open("a", encoding="utf-8") as f:
        f.write(f"{episode_id}\n")
        f.flush()


def _assert_local_dataset_consistent(dataset_dir: Path) -> None:
    """Guard against leaving a half-written LeRobot v3.0 dataset.

    info.json claims `total_episodes`, and every episode 0..N-1 must have a row in the
    meta/episodes/*.parquet footer (a v3.0 file may batch several episodes, so we compare
    the UNION of episode_index across all footer files, not file counts). A mismatch means
    the writer wasn't flushed (missing finalize()).
    """
    import pandas as pd

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    total = int(info.get("total_episodes", 0))
    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    have: set[int] = set()
    for p in ep_files:
        col = pd.read_parquet(p, columns=["episode_index"])["episode_index"]
        have.update(int(x) for x in col)
    missing = set(range(total)) - have
    if missing:
        raise RuntimeError(
            f"local dataset inconsistent after finalize: info.total_episodes={total} but "
            f"meta/episodes is missing episode_index {sorted(missing)} "
            f"(footer files: {[p.name for p in ep_files]})."
        )


def open_dataset(
    repo_id: str,
    root: Path,
    derived: core_config.DerivedSchema,
    rgb_encoder_vcodec: str = DEFAULT_RGB_ENCODER_VCODEC,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import RGBEncoderConfig

    dataset_root = root / repo_id
    rgb_encoder = RGBEncoderConfig(vcodec=rgb_encoder_vcodec)
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if int(info["fps"]) != derived.fps:
            raise ValueError(
                f"episode fps {derived.fps} != dataset fps {info['fps']}; "
                "the dataset timebase cannot change"
            )
        existing_state = tuple(info["features"]["observation.state"]["shape"])
        if existing_state != (derived.state_dim,):
            raise ValueError(
                f"episode state_dim {derived.state_dim} != dataset {existing_state}; "
                "the local dataset schema is fixed at 12-D"
            )
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
            rgb_encoder=rgb_encoder,
            streaming_encoding=True,
            encoder_queue_maxsize=480,
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=derived.fps,
        features=derived.features,
        root=dataset_root,
        use_videos=True,
        rgb_encoder=rgb_encoder,
        streaming_encoding=True,
        encoder_queue_maxsize=480,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def run(episode_dir: Path, out_dir: Path, scope: str = "") -> None:
    """StageFn entrypoint. ``out_dir`` is the runner's staging dir (receipt lands
    there); the derived episode dir (glim/gripper outputs) is its parent."""
    del scope
    derived_episode_dir = out_dir.resolve().parent

    m = manifest.read_manifest(episode_dir / layout.MANIFEST_FILE)
    rig = core_config.load_rig_config(
        episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    )
    derived = core_config.require_bimanual(rig)

    receipt: dict[str, Any] = {"episode_id": m["episode_id"], "engine": DEFAULT_ENGINE}
    if not m["dataset_eligible"]:
        receipt["skipped"] = "episode not dataset_eligible"
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(f"dataset: skipping ineligible episode {m['episode_id']}", flush=True)
        return
    if not manifest.lidar_ptp_acceptable(m):
        receipt["skipped"] = "PTP not locked / residual too large for lidar SLAM"
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(f"dataset: skipping PTP-unacceptable episode {m['episode_id']}", flush=True)
        return
    # The follower arm is resampled onto the leader's unix-ns timeline.
    if not manifest.clock_sync_acceptable(m):
        receipt["skipped"] = "cross-Pi clock offset too large / unsynced (bimanual merge)"
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(f"dataset: skipping clock-desynced episode {m['episode_id']}", flush=True)
        return

    repo_id = LOCAL_REPO_ID
    dataset_root = Path(os.environ.get("DATA_DIR", "data")) / "lerobot"
    dataset_dir = dataset_root / repo_id

    # Idempotence: if this episode already appended (e.g. the runner re-runs the
    # stage after a crash between save_episode and the COMPLETE marker), skip the
    # whole append rather than duplicate its frames into the dataset.
    if already_appended(dataset_dir, m["episode_id"]):
        receipt["skipped"] = "already appended to dataset (idempotent re-run)"
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(f"dataset: {m['episode_id']} already in dataset; skipping", flush=True)
        return

    # --- assemble per-UMI streams on the umi1 timeline --------------------------------
    streams = {u.name: load_frame_stream(episode_dir, u.name) for u in rig.umis}
    timeline = streams[rig.umis[0].name].t
    n_frames = len(timeline)

    ref_pos: np.ndarray | None = None
    ref_quat: np.ndarray | None = None
    state_blocks: list[np.ndarray] = []
    for u in rig.umis:
        track, basis = pose_track_for_umi(episode_dir, derived_episode_dir, m, u)
        aligned = resample(track, timeline)
        if ref_pos is None:
            ref_pos, ref_quat = aligned.pos[0], aligned.quat[0]
        state_blocks.append(states_from_track(aligned, ref_pos, ref_quat, basis=basis))
    states = np.concatenate(state_blocks, axis=1)
    if states.shape != (n_frames, 12):
        raise RuntimeError(f"expected bimanual state shape {(n_frames, 12)}, got {states.shape}")
    actions = actions_from_states(states)

    grippers = {
        u.name: load_gripper_widths(derived_episode_dir, u.name, timeline)
        for u in rig.umis
        if u.name in derived.gripper_keys
    }

    # --- write the LeRobot episode -----------------------------------------------------
    dataset = open_dataset(repo_id, dataset_root, derived)

    task = m.get("task", rig.task)
    for i in range(n_frames):
        frame: dict[str, Any] = {
            "observation.state": states[i],
            "action": actions[i],
            "task": task,
        }
        for u in rig.umis:
            stream = streams[u.name]
            # umi1 defines the timeline; other UMIs use their nearest-past frame.
            j = i if u.name == rig.umis[0].name else _nearest_past(stream.t, timeline[i])
            frame[derived.video_keys[u.name]] = stream.images[j]
            width = grippers.get(u.name)
            if width is not None:
                frame[derived.gripper_keys[u.name]] = np.asarray([width[i]], dtype=np.float32)
            elif u.name in derived.gripper_keys:
                frame[derived.gripper_keys[u.name]] = np.zeros(1, dtype=np.float32)
        dataset.add_frame(frame)
    dataset.save_episode()
    # LeRobot v3.0 buffers each episode's footer metadata in the DatasetWriter: save_episode()
    # writes data/*.parquet + info.json but buffers the episode footer until finalize().
    dataset.finalize()
    _assert_local_dataset_consistent(dataset_dir)
    mark_appended(dataset_dir, m["episode_id"])
    hf_repo = os.environ.get("HF_REPO_ID", "").strip()
    pushed = _push_to_hub(dataset_dir, episode_dir, hf_repo) if hf_repo else False
    receipt.update(frames=n_frames, state_dim=int(states.shape[1]), pushed=pushed)
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        f"dataset: appended {n_frames} frames ({m['episode_id']}) pushed={pushed}",
        flush=True,
    )


def _push_to_hub(dataset_dir: Path, episode_dir: Path, repo_id: str) -> bool:
    """Mirror the local dataset to the HF Hub + archive the raw episode (the offsite copy).

    Best-effort: a failed push never loses the local append — the next episode's push
    re-syncs the whole dataset. Auth comes from the HF_TOKEN env var (rendered into .env
    by bin/provision-server; huggingface_hub reads it natively). Blank HF_REPO_ID in
    pair.env = never called = the local-only default."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        # NEVER lerobot's own push_to_hub: it skips meta/episodes/ and can mangle data/,
        # leaving the Hub inconsistent ("episode N not found in metadata"). The LOCAL
        # dataset is the single source of truth and is self-consistent after finalize();
        # sync the Hub to EXACTLY match it — delete_patterns drops stale remote files,
        # and LFS dedups unchanged videos so the per-episode re-upload is cheap.
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(dataset_dir),
            ignore_patterns=[APPENDED_LEDGER],
            delete_patterns=["meta/**", "data/**", "videos/**"],
        )
        # Raw episode archive: its own accumulating namespace, untouched by the deletes
        # above — the only off-box copy of the raw pcaps/video once Pi spools are pruned.
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(episode_dir),
            path_in_repo=f"raw_episodes/{episode_dir.name}",
        )
        # LeRobot dataset card: the tags/configs front-matter is what lights up the Hub
        # page (robotics + LeRobot chips, dataset viewer, Visualize badge). Same card
        # lerobot's own push_to_hub would write — only its data upload is unsafe, not this.
        from lerobot.datasets.utils import DatasetInfo, create_lerobot_dataset_card

        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        fields = {k: info[k] for k in DatasetInfo.__dataclass_fields__ if k in info}
        card = create_lerobot_dataset_card(
            dataset_info=DatasetInfo(**fields), license="apache-2.0", repo_id=repo_id
        )
        card.push_to_hub(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception as exc:  # push failure must not take down the pipeline
        print(f"dataset: hub push failed ({exc}); local dataset is intact", flush=True)
        return False


def _nearest_past(stream_t: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(stream_t, t, side="right")) - 1
    return max(0, idx)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m robot_umi_server.dataset_stage <episode_dir> <out_dir>")
        raise SystemExit(2)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
