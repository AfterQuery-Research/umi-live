"""Gripper-width stage: ArUco tag detection over an episode's camera streams.

Runs after ingest, entirely from the hermetic episode directory: the manifest says
which camera streams exist, the config SNAPSHOT (never a live file) says which UMIs
have a gripper and which calib blob to use, and the per-frame widths land in
``<out_dir>/<umiN>_gripper.jsonl`` — one line per camera frame::

    {"frame_index": 0, "time_unix_ns": 1234567890, "width_mm": 71.3}

UMIs with no gripper config, or whose camera stream is absent from the episode,
are skipped with a note. ``stride`` (from the config) detects only every Nth frame;
``gripper_widths`` interpolates the rest exactly like frames that missed a tag.

Usage::

    python -m robot_umi_server.gripper_stage <episode_dir> <out_dir>
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from robot_umi_core import layout
from robot_umi_core.config import UmiConfig, load_rig_config
from robot_umi_core.manifest import present_streams, read_manifest

from robot_umi_server import gripper


class _StridedWidthSource:
    """Detect on every Nth frame only; skipped frames read as misses (None).

    ``gripper_widths`` already interpolates over frames that missed a tag, so the
    stride's skipped frames are filled by the same gap interpolation. Relies on
    ``gripper_widths`` calling ``width_mm`` exactly once per frame, in order.
    """

    def __init__(self, detector: gripper.GripperDetector, stride: int):
        self._detector = detector
        self._stride = max(1, int(stride))
        self._count = 0

    def width_mm(self, rgb: np.ndarray) -> float | None:
        index = self._count
        self._count += 1
        if index % self._stride:
            return None
        return self._detector.width_mm(rgb)


def _load_frame_times(path: Path) -> list[int]:
    """Per-frame ``time_unix_ns`` from a camera ``*_frames.jsonl`` sidecar."""
    times = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        times.append(int(json.loads(line)["time_unix_ns"]))
    return times


def _sequential_decoder(
    cap: cv2.VideoCapture, video_path: Path
) -> Callable[[int], np.ndarray]:
    """Decode frames one at a time, in order, without holding the video in memory."""

    def decode(index: int) -> np.ndarray:
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(
                f"{video_path}: video ended at frame {index} (frames.jsonl has more)"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return decode


def _process_umi(umi: UmiConfig, episode_dir: Path, out_dir: Path) -> None:
    assert umi.gripper is not None
    key = layout.sensor_key(umi.name, layout.KIND_CAMERA)
    files = layout.sensor_files(key)
    video_path = episode_dir / files["video"]
    times = _load_frame_times(episode_dir / files["frames"])
    calib_path = episode_dir / layout.CONFIG_SNAPSHOT_DIR / umi.camera.calib

    cap = cv2.VideoCapture(str(video_path))
    try:
        # The detector's K must match the decoded frame size, whatever the camera
        # actually published (detector_from_config rescales the snapshot calib).
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or umi.camera.width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or umi.camera.height
        detector = gripper.detector_from_config(
            gripper.GripperConfig(
                geom=gripper.GripperGeometry(),
                roi_top_frac=umi.gripper.roi_top_frac,
                calib_path=calib_path,
            ),
            width,
            height,
            target_ids=umi.gripper.tag_ids,
        )
        widths = gripper.gripper_widths(
            list(range(len(times))),
            _StridedWidthSource(detector, umi.gripper.stride),
            _sequential_decoder(cap, video_path),
        )
    finally:
        cap.release()

    if widths is None:
        print(
            f"{umi.name}: no frame saw both tags {umi.gripper.tag_ids}; no output written"
        )
        return
    out_path = out_dir / f"{umi.name}_gripper.jsonl"
    lines = [
        json.dumps(
            {"frame_index": i, "time_unix_ns": ts, "width_mm": float(widths[i, 0])}
        )
        for i, ts in enumerate(times)
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{umi.name}: wrote {out_path} ({len(lines)} frames)")


def run(episode_dir: Path, out_dir: Path) -> None:
    manifest = read_manifest(episode_dir / layout.MANIFEST_FILE)
    cfg = load_rig_config(
        episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    )
    cameras = set(present_streams(manifest, layout.KIND_CAMERA))
    out_dir.mkdir(parents=True, exist_ok=True)
    for umi in cfg.umis:
        if umi.gripper is None:
            print(f"{umi.name}: no gripper config; skipping")
            continue
        if layout.sensor_key(umi.name, layout.KIND_CAMERA) not in cameras:
            print(f"{umi.name}: no camera stream in this episode; skipping")
            continue
        _process_umi(umi, episode_dir, out_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "usage: python -m robot_umi_server.gripper_stage <episode_dir> <out_dir>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
