"""Dataset stage: trajectories + frames + gripper widths -> LeRobot v3 (+ HF push).

Consumes ONLY per-episode artifacts (hermetic): the manifest + config snapshot, each
UMI's camera video/frames, the chosen engine's ``trajectory.tum`` per lidar UMI (or
the raw Vive jsonl + episode-shipped EE calib for vive-mode episodes), and the
gripper stage's width jsonls. The LeRobot feature schema comes from
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

Deployment knobs come from env (not the rig config): ROBOT_UMI_REPO_ID,
ROBOT_UMI_DATASET_ROOT, ROBOT_UMI_PUSH_TO_HUB, ROBOT_UMI_PUBLIC.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_umi_core import config as core_config
from robot_umi_core import geom, layout, lidar_clock, manifest

from robot_umi_server.trajectory import (
    PoseTrack,
    actions_from_states,
    read_tum,
    read_vive_track,
    resample,
    states_from_track,
)

DEFAULT_ENGINE = "glim"
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
    engine: str,
) -> tuple[PoseTrack, np.ndarray | None] | None:
    """This UMI's world poses + the body-convention basis to apply to its states.

    Presence-based, never mode-based. Lidar tracks carry the lidar-body ->
    UMI-convention basis (forward=+Z, right=+Y, up=+X); vive tracks are already
    in that convention via the EE calibration."""
    lidar_key = layout.sensor_key(umi.name, layout.KIND_LIDAR)
    vive_key = layout.sensor_key(umi.name, layout.KIND_VIVE)
    if lidar_key in manifest.present_streams(m, layout.KIND_LIDAR):
        # Prefer the align stage's common-frame trajectory when it ran (multi-lidar rigs
        # with alignment enabled); otherwise read this lidar's own GLIM trajectory. The
        # fallback IS the non-joint path — with no align output the behavior is identical
        # to before (incl. solo / single-lidar episodes, which never run align).
        aligned_tum = derived_episode_dir / "align" / umi.name / "trajectory.tum"
        traj_tum = (
            aligned_tum
            if aligned_tum.is_file()
            else derived_episode_dir / umi.name / engine / "trajectory.tum"
        )
        track = read_tum(traj_tum)
        # GLIM stamps poses in the LIDAR clock; realign onto the host/camera clock via the
        # pcap arrival-vs-payload skew, so a lagging lidar PTP clock can't push the whole
        # trajectory outside the camera window (which would clamp every pose to one value).
        pcap = episode_dir / layout.sensor_files(lidar_key)["pcap"]
        skew = lidar_clock.pcap_host_skew(pcap) if pcap.is_file() else None
        if skew is not None:
            print(f"  {umi.name}: lidar->host clock skew {skew:+.3f}s (realigned)", flush=True)
            track = PoseTrack(t=track.t + skew, pos=track.pos, quat=track.quat)
        # GLIM tracks the LIDAR body; shift to the gripper end-effector (TCP) so the state is
        # the true EE pose. The EE is a fixed lever in the lidar body frame, so
        # ee_pos = lidar_pos + R_lidar @ lever_body (orientation unchanged — UMI_FROM_LIDAR_BODY
        # re-expresses axes downstream). LIDAR_EE_LEVER_M is in UMI axes, so rotate it back into
        # the lidar body frame. Matches the mac live/offline EE viewers. (The vive branch below
        # applies its own tracker->EE calibration, so this is lidar-only.)
        lever_body = geom.UMI_FROM_LIDAR_BODY.T @ geom.LIDAR_EE_LEVER_M
        ee_pos = track.pos + Rotation.from_quat(track.quat).apply(lever_body)
        track = PoseTrack(t=track.t, pos=ee_pos, quat=track.quat)
        return track, geom.UMI_FROM_LIDAR_BODY
    if vive_key in manifest.present_streams(m, layout.KIND_VIVE):
        if umi.vive is None:
            raise ValueError(f"{umi.name}: vive stream present but not configured")
        ee_calib = episode_dir / layout.CONFIG_SNAPSHOT_DIR / umi.vive.ee_calib
        poses_jsonl = episode_dir / layout.sensor_files(vive_key)["poses"]
        return read_vive_track(poses_jsonl, ee_calib), None
    return None  # camera-only: state stays zero


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


# The LeRobot append + HF push are global side effects that live OUTSIDE the
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
    """Guard against pushing a half-written LeRobot v3.0 dataset to the Hub.

    info.json claims `total_episodes`, and every episode 0..N-1 must have a row in the
    meta/episodes/*.parquet footer (a v3.0 file may batch several episodes, so we compare
    the UNION of episode_index across all footer files, not file counts). A mismatch means
    the writer wasn't flushed (missing finalize()) and the Hub would load with
    "episode N not found in metadata" — raise so the push is skipped and the failure is loud.
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
            f"(footer files: {[p.name for p in ep_files]}). Refusing to push."
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
                "N UMIs is fixed per repo_id"
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


def run(episode_dir: Path, out_dir: Path, scope: str = "", engine: str | None = None) -> None:
    """StageFn entrypoint. ``out_dir`` is the runner's staging dir (receipt lands
    there); the derived episode dir (glim/gripper outputs) is its parent."""
    del scope
    engine = engine or os.environ.get("ROBOT_UMI_ENGINE", DEFAULT_ENGINE)
    derived_episode_dir = out_dir.resolve().parent

    m = manifest.read_manifest(episode_dir / layout.MANIFEST_FILE)
    rig = core_config.load_rig_config(
        episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    )
    derived = core_config.derive(rig)

    receipt: dict[str, Any] = {"episode_id": m["episode_id"], "engine": engine}
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
    # Merged bimanual episodes: the follower arm is resampled onto the leader's unix-ns
    # timeline, so a large cross-Pi clock skew would silently misalign the two arms. A
    # no-op for solo episodes (no clock_sync block).
    if not manifest.clock_sync_acceptable(m):
        receipt["skipped"] = "cross-Pi clock offset too large / unsynced (bimanual merge)"
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(f"dataset: skipping clock-desynced episode {m['episode_id']}", flush=True)
        return

    repo_id = os.environ.get("ROBOT_UMI_REPO_ID")
    if not repo_id:
        raise RuntimeError("ROBOT_UMI_REPO_ID is required for the dataset stage")
    dataset_root = Path(os.environ.get("ROBOT_UMI_DATASET_ROOT", "data/lerobot"))
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
        source = pose_track_for_umi(episode_dir, derived_episode_dir, m, u, engine)
        if source is None:
            state_blocks.append(np.zeros((n_frames, 6), dtype=np.float32))
            continue
        track, basis = source
        aligned = resample(track, timeline)
        if ref_pos is None:
            ref_pos, ref_quat = aligned.pos[0], aligned.quat[0]
        state_blocks.append(states_from_track(aligned, ref_pos, ref_quat, basis=basis))
    states = np.concatenate(state_blocks, axis=1)
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
    # writes data/*.parquet + info.json but NOT meta/episodes/file-NNN.parquet until the writer
    # is flushed (otherwise only DatasetWriter.__del__ flushes it, at GC/process-exit). The hub
    # push below runs BEFORE that, so without an explicit flush it mirrors a half-written dataset
    # (new episode's data present, its meta/episodes row absent) -> the Hub loads with
    # "episode N not found in metadata". finalize() flushes it; it is idempotent and safe to call
    # per-append (the next run re-opens the writer via open_dataset).
    dataset.finalize()
    _assert_local_dataset_consistent(dataset_dir)  # fail loud rather than push an invalid dataset
    # Record the append BEFORE the (slow, fallible) hub push: if the push crashes,
    # the local frames are already committed and must not be re-appended on re-run.
    mark_appended(dataset_dir, m["episode_id"])

    pushed = False
    if os.environ.get("ROBOT_UMI_PUSH_TO_HUB", "0") == "1":
        try:
            private = os.environ.get("ROBOT_UMI_PUBLIC", "0") != "1"
            from huggingface_hub import HfApi

            api = HfApi()
            # Create the repo with the intended visibility, then ENFORCE it (create_repo is
            # a no-op on an existing repo, so it won't flip a pre-existing repo's visibility).
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
            api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)

            # NEVER use lerobot's push_to_hub: it skips meta/episodes/ and can mangle data/,
            # leaving the Hub inconsistent ("episode N not found in metadata"). The LOCAL
            # dataset is the single source of truth and is always self-consistent (info.json +
            # meta/episodes/ + data/ + videos/ are all written together by save_episode). Sync
            # the Hub to EXACTLY match it: delete_patterns drops any stale/corrupt remote file
            # the local no longer has, and upload_folder overwrites the rest (LFS dedups
            # unchanged videos by hash, so re-uploading the full dataset each episode is cheap).
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(dataset_dir),
                ignore_patterns=[".umi_appended_episodes.txt"],
                delete_patterns=["meta/**", "data/**", "videos/**"],
            )
            # Archive the raw episode alongside (best-effort; separate raw_episodes/ namespace,
            # untouched by the delete_patterns above so prior episodes accumulate).
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(episode_dir),
                path_in_repo=f"raw_episodes/{episode_dir.name}",
            )
            pushed = True
        except Exception as exc:  # push failure must not lose the local append
            print(f"dataset: hub push failed ({exc}); local dataset is intact", flush=True)

    receipt.update(frames=n_frames, state_dim=int(states.shape[1]), pushed=pushed)
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        f"dataset: appended {n_frames} frames ({m['episode_id']}) pushed={pushed}",
        flush=True,
    )


def _nearest_past(stream_t: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(stream_t, t, side="right")) - 1
    return max(0, idx)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m robot_umi_server.dataset_stage <episode_dir> <out_dir>")
        raise SystemExit(2)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
