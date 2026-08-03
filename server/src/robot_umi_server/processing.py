"""Episode processing into LeRobot datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import HfApi
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERA_FILE = "camera_front.mjpeg.avi"
CAMERA_FRAMES_FILE = "camera_frames.jsonl"
LIDAR_FILE = "mock_lidar.jsonl"
LIDAR_POINTS_FILE = "mock_lidar_points.bin"
LIDAR_POINT_BYTES = 16  # packed float32 x, y, z, intensity
DEFAULT_RGB_ENCODER_VCODEC = "h264"
PUSH_PENDING_SUFFIX = ".push_pending"
# Maximum tolerated gap between a camera frame and its aligned lidar sample.
# Nominal spacing is ~100 ms at 10 Hz; this bounds both a frame arriving before
# the first sample (warm-up clamp) and a stalled/dead lidar stream (staleness).
DEFAULT_MAX_ALIGNMENT_GAP_NS = 1_000_000_000


@dataclass(frozen=True)
class LidarSample:
    """One lidar frame: legacy 2D scans carry ``ranges``; 3D point-cloud
    episodes carry an offset/length window into the packed points file."""

    time_monotonic_ns: int
    ranges: np.ndarray | None = None
    points_offset: int = 0
    points_length: int = 0
    point_count: int = 0


@dataclass(frozen=True)
class CameraFrame:
    frame_index: int
    v4l2_sequence: int
    time_monotonic_ns: int
    timestamp_clock: str
    timestamp_source: str


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    timestamp_monotonic_ns: int
    state: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class ProcessResult:
    episode_index: int
    frame_count: int
    pushed: bool = True


def process_episode(
    episode_dir: Path,
    repo_id: str,
    task: str,
    dataset_root: Path,
    push_to_hub: bool,
    private: bool,
    rgb_encoder_vcodec: str = DEFAULT_RGB_ENCODER_VCODEC,
) -> ProcessResult:
    # The local append is the irreversible commit. Do it first, then attempt
    # the Hub push. A transient push failure must NOT fail the job (the frames
    # are already committed exactly once): instead we mark the episode
    # push-pending and let startup recovery retry the push, so re-uploading the
    # same episode is never required and can never duplicate frames.
    result = append_lerobot_episode_onepass(
        episode_dir=episode_dir,
        repo_id=repo_id,
        task=task,
        root=dataset_root,
        rgb_encoder_vcodec=rgb_encoder_vcodec,
    )
    pushed = True
    if push_to_hub:
        try:
            push_episode(result.dataset, repo_id, episode_dir, private)
        except Exception as exc:
            pushed = False
            mark_push_pending(episode_dir)
            print(
                f"hub push failed for {episode_dir.name}; kept for retry: {exc}",
                flush=True,
            )
    return ProcessResult(
        episode_index=result.episode_index,
        frame_count=result.frame_count,
        pushed=pushed,
    )


def push_episode(
    dataset: LeRobotDataset, repo_id: str, episode_dir: Path, private: bool
) -> None:
    dataset.push_to_hub(private=private)
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=episode_dir,
        path_in_repo=f"raw_episodes/{episode_dir.name}",
    )


def push_pending_marker(episode_dir: Path) -> Path:
    # Sibling of the episode dir, so it is never swept into the pushed folder.
    return episode_dir.parent / f"{episode_dir.name}{PUSH_PENDING_SUFFIX}"


def mark_push_pending(episode_dir: Path) -> None:
    push_pending_marker(episode_dir).write_text("push pending\n", encoding="utf-8")


def retry_pending_pushes(raw_root: Path, push_fn: Callable[[Path], None]) -> list[str]:
    """Retry every episode whose Hub push previously failed. Returns the ids
    that were pushed successfully this pass."""
    pushed: list[str] = []
    if not raw_root.exists():
        return pushed
    for marker in sorted(raw_root.glob(f"*{PUSH_PENDING_SUFFIX}")):
        name = marker.name[: -len(PUSH_PENDING_SUFFIX)]
        episode_dir = raw_root / name
        if not episode_dir.exists():
            marker.unlink(missing_ok=True)
            continue
        try:
            push_fn(episode_dir)
            marker.unlink(missing_ok=True)
            pushed.append(name)
        except Exception as exc:
            print(f"pending hub push retry failed for {name}: {exc}", flush=True)
    return pushed


@dataclass(frozen=True)
class AppendResult:
    dataset: LeRobotDataset
    episode_index: int
    frame_count: int


def append_lerobot_episode_onepass(
    episode_dir: Path,
    repo_id: str,
    task: str,
    root: Path,
    rgb_encoder_vcodec: str = DEFAULT_RGB_ENCODER_VCODEC,
) -> AppendResult:
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest_camera_eligible(manifest)
    camera_frames = load_camera_frames(episode_dir / CAMERA_FRAMES_FILE)
    lidar = load_lidar(episode_dir / LIDAR_FILE)
    if not lidar:
        raise ValueError("episode lidar stream has no samples")

    validate_lidar_points_file(episode_dir, lidar)
    cap = cv2.VideoCapture(str(episode_dir / CAMERA_FILE))
    try:
        ok, frame_bgr = cap.read()
        if not ok:
            raise ValueError("episode video has no frames")
        first_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dataset = open_dataset(
            first_rgb,
            repo_id,
            root,
            fps=int(manifest["camera_fps"]),
            rgb_encoder_vcodec=rgb_encoder_vcodec,
            streaming_encoding=True,
        )
        episode_index = dataset.meta.total_episodes

        lidar_idx = 0
        previous_image: np.ndarray | None = None
        previous_state: np.ndarray | None = None
        frame_count = 0
        while ok:
            if frame_count >= len(camera_frames):
                raise ValueError("video has more frames than camera_frames.jsonl")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_time = camera_frames[frame_count].time_monotonic_ns
            lidar_idx = latest_past_lidar_index(lidar, lidar_idx, frame_time)
            state = lidar_slam_state(lidar, lidar_idx)
            if previous_image is not None and previous_state is not None:
                add_lerobot_frame(
                    dataset,
                    previous_image,
                    previous_state,
                    (state - previous_state).astype(np.float32),
                    task,
                )
            previous_image = frame_rgb
            previous_state = state
            frame_count += 1
            ok, frame_bgr = cap.read()
    finally:
        cap.release()

    if frame_count != len(camera_frames):
        raise ValueError(
            "camera_frames.jsonl count does not match decoded video frames"
        )
    if previous_image is not None and previous_state is not None:
        add_lerobot_frame(
            dataset,
            previous_image,
            previous_state,
            np.zeros_like(previous_state, dtype=np.float32),
            task,
        )
    dataset.save_episode()
    dataset.finalize()
    return AppendResult(
        dataset=dataset,
        episode_index=episode_index,
        frame_count=frame_count,
    )


def add_lerobot_frame(
    dataset: LeRobotDataset,
    image: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
) -> None:
    dataset.add_frame(
        {
            "observation.images.front": image,
            "observation.state": state,
            "action": action,
            "task": task,
        }
    )


def episode_to_records(episode_dir: Path) -> list[FrameRecord]:
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest_camera_eligible(manifest)
    frames = decode_video(episode_dir / CAMERA_FILE)
    camera_frames = load_camera_frames(episode_dir / CAMERA_FRAMES_FILE)
    lidar = load_lidar(episode_dir / LIDAR_FILE)
    if not frames:
        raise ValueError("episode video has no frames")
    if not lidar:
        raise ValueError("episode lidar stream has no samples")
    if len(frames) != len(camera_frames):
        raise ValueError(
            "camera_frames.jsonl count does not match decoded video frames"
        )

    validate_lidar_points_file(episode_dir, lidar)
    states = []
    timestamps = []
    lidar_idx = 0
    for camera_frame in camera_frames:
        frame_time = camera_frame.time_monotonic_ns
        lidar_idx = latest_past_lidar_index(lidar, lidar_idx, frame_time)
        timestamps.append(frame_time)
        states.append(lidar_slam_state(lidar, lidar_idx))

    state_array = np.stack(states).astype(np.float32)
    actions = np.diff(state_array, axis=0, append=state_array[-1:]).astype(np.float32)
    return [
        FrameRecord(
            image=frame,
            timestamp_monotonic_ns=timestamps[i],
            state=state_array[i],
            action=actions[i],
        )
        for i, frame in enumerate(frames)
    ]


def decode_video(path: Path) -> list[np.ndarray]:
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


def load_lidar(path: Path) -> list[LidarSample]:
    """Load the lidar frame index. Supports both formats:

    - legacy 2D scans: one JSON line per scan with a ``ranges`` array;
    - 3D point clouds: one JSON line per frame with ``point_count``/``offset``/
      ``length`` into the packed points file (``mock_lidar_points.bin``).

    Frames whose packets were all lost (``point_count`` 0) carry no state and
    are skipped; the resulting timestamp gap is policed by the alignment
    staleness bound like any other gap.
    """
    samples = []
    previous_timestamp: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        time_monotonic_ns = int(item["time_monotonic_ns"])
        if previous_timestamp is not None and time_monotonic_ns <= previous_timestamp:
            raise ValueError("lidar timestamps are not monotonic")
        previous_timestamp = time_monotonic_ns
        if "ranges" in item:
            samples.append(
                LidarSample(
                    time_monotonic_ns=time_monotonic_ns,
                    ranges=np.array(item["ranges"], dtype=np.float32),
                )
            )
            continue
        point_count = int(item["point_count"])
        length = int(item["length"])
        if length != point_count * LIDAR_POINT_BYTES:
            raise ValueError("lidar index length does not match point count")
        if point_count == 0:
            continue
        samples.append(
            LidarSample(
                time_monotonic_ns=time_monotonic_ns,
                points_offset=int(item["offset"]),
                points_length=length,
                point_count=point_count,
            )
        )
    return samples


def validate_lidar_points_file(episode_dir: Path, samples: list[LidarSample]) -> None:
    """Integrity check for the point-cloud file that SLAM will consume."""
    if not any(sample.ranges is None for sample in samples):
        return  # legacy 2D episode: no points file
    points_path = episode_dir / LIDAR_POINTS_FILE
    if not points_path.exists():
        raise ValueError("episode is missing mock_lidar_points.bin")
    size = points_path.stat().st_size
    for sample in samples:
        if sample.ranges is None and (
            sample.points_offset + sample.points_length > size
        ):
            raise ValueError("lidar index references beyond the points file")


def lidar_slam_state(lidar: list[LidarSample], index: int) -> np.ndarray:
    """Device state for one aligned lidar frame - SLAM placeholder.

    TODO(SLAM): implement lidar(-inertial) odometry over the episode's point
    clouds (FAST-LIO2 / KISS-ICP class) and return the device pose
    [x, y, z, roll, pitch, yaw] in the episode's world frame; ``action`` then
    becomes the relative pose delta between consecutive camera frames. The
    full cloud for every aligned frame is stored in ``mock_lidar_points.bin``
    at ``lidar[index].points_offset``/``points_length`` (packed float32
    x, y, z, intensity) and is archived raw with each episode, so
    already-recorded data can be re-processed once SLAM lands. Until then
    this returns a zero pose so the pipeline stays runnable end to end.
    """
    del lidar, index  # inputs reserved for the SLAM implementation
    return np.zeros(6, dtype=np.float32)


def latest_past_lidar_index(
    lidar: list[LidarSample],
    lidar_idx: int,
    frame_time: int,
    max_gap_ns: int = DEFAULT_MAX_ALIGNMENT_GAP_NS,
) -> int:
    """Index of the newest lidar sample at or before ``frame_time``.

    A frame slightly before the first sample (sensor warm-up) clamps to sample
    0 as long as it is within ``max_gap_ns``; earlier than that is a real
    desync and raises. A frame whose newest past sample is more than
    ``max_gap_ns`` old (the lidar stalled or died mid-episode) also raises,
    instead of silently freezing the state at a stale scan.
    """
    if lidar[lidar_idx].time_monotonic_ns > frame_time:
        gap = lidar[lidar_idx].time_monotonic_ns - frame_time
        if gap > max_gap_ns:
            raise ValueError(
                "camera frame precedes the lidar stream by more than the "
                "allowed alignment gap"
            )
        return lidar_idx
    while (
        lidar_idx + 1 < len(lidar)
        and lidar[lidar_idx + 1].time_monotonic_ns <= frame_time
    ):
        lidar_idx += 1
    gap = frame_time - lidar[lidar_idx].time_monotonic_ns
    if gap > max_gap_ns:
        raise ValueError(
            "lidar sample is stale for camera frame; lidar stream gap exceeds "
            "the allowed alignment gap"
        )
    return lidar_idx


def validate_manifest_camera_eligible(manifest: dict[str, Any]) -> None:
    if manifest.get("camera_sync_mode") != "v4l2_monotonic_start_of_exposure_v1":
        raise ValueError("episode is missing V4L2 camera sync metadata")
    if manifest.get("camera_dataset_eligible") is not True:
        raise ValueError("episode camera data is not dataset eligible")


def load_camera_frames(path: Path) -> list[CameraFrame]:
    if not path.exists():
        raise ValueError("episode is missing camera_frames.jsonl")
    frames = []
    previous_timestamp: int | None = None
    # Count only non-blank records: a stray blank line must not desynchronize
    # the contiguity check against frame_index.
    expected_index = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        frame = CameraFrame(
            frame_index=int(item["frame_index"]),
            v4l2_sequence=int(item["v4l2_sequence"]),
            time_monotonic_ns=int(item["time_monotonic_ns"]),
            timestamp_clock=str(item["timestamp_clock"]),
            timestamp_source=str(item["timestamp_source"]),
        )
        if frame.frame_index != expected_index:
            raise ValueError("camera_frames.jsonl frame_index is not contiguous")
        expected_index += 1
        if (
            previous_timestamp is not None
            and frame.time_monotonic_ns <= previous_timestamp
        ):
            raise ValueError("camera_frames.jsonl timestamps are not monotonic")
        if frame.timestamp_clock != "monotonic":
            raise ValueError("camera_frames.jsonl timestamp clock is not monotonic")
        if frame.timestamp_source != "start_of_exposure":
            raise ValueError(
                "camera_frames.jsonl timestamp source is not start_of_exposure"
            )
        previous_timestamp = frame.time_monotonic_ns
        frames.append(frame)
    if not frames:
        raise ValueError("camera_frames.jsonl has no frames")
    return frames


def append_lerobot_episode(
    records: list[FrameRecord],
    repo_id: str,
    task: str,
    root: Path,
    fps: int = 30,
    rgb_encoder_vcodec: str = DEFAULT_RGB_ENCODER_VCODEC,
) -> LeRobotDataset:
    if not records:
        raise ValueError("cannot append an empty episode")
    dataset = open_dataset(
        records[0].image,
        repo_id,
        root,
        fps,
        rgb_encoder_vcodec=rgb_encoder_vcodec,
        streaming_encoding=True,
    )
    for record in records:
        add_lerobot_frame(dataset, record.image, record.state, record.action, task)
    dataset.save_episode()
    dataset.finalize()
    return dataset


def open_dataset(
    first_image: np.ndarray,
    repo_id: str,
    root: Path,
    fps: int,
    rgb_encoder_vcodec: str = DEFAULT_RGB_ENCODER_VCODEC,
    streaming_encoding: bool = False,
) -> LeRobotDataset:
    dataset_root = root / repo_id
    rgb_encoder = RGBEncoderConfig(vcodec=rgb_encoder_vcodec)
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        # Resume ignores the fps argument, so appending an episode recorded at a
        # different rate would silently write frames under the original
        # timebase (wrong timestamps / playback speed). Reject the mismatch.
        existing_fps = int(json.loads(info_path.read_text(encoding="utf-8"))["fps"])
        if existing_fps != fps:
            raise ValueError(
                f"episode fps {fps} does not match existing dataset fps "
                f"{existing_fps}; the dataset timebase cannot change"
            )
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
            rgb_encoder=rgb_encoder,
            streaming_encoding=streaming_encoding,
            encoder_queue_maxsize=480,
        )
    height, width, channels = first_image.shape
    features: dict[str, dict[str, Any]] = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
        # TODO(SLAM): pose state/action, filled by lidar odometry; zeros
        # until SLAM lands (see lidar_slam_state).
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
        use_videos=True,
        rgb_encoder=rgb_encoder,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=480,
        image_writer_processes=0,
        image_writer_threads=0 if streaming_encoding else 2,
    )
