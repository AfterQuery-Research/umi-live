#!/usr/bin/env python3
"""Render one synchronized, four-panel UMI Flow conference demo.

The paired episode's leader-camera timeline is the master clock. Both onboard cameras
and both processed SLAM trajectories are sampled onto that clock; the external operator
video uses one explicit offset because it has no shared hardware timestamp.

    uv run --package umi-flow-server python tools/render_slam_demo.py \
      RAW_EPISODE DERIVED_EPISODE final.mp4 \
      --operator-video operator.mov --operator-offset-s 2.4

``--operator-offset-s`` is the time in the operator video corresponding to the first
UMI frame. A negative value pads the operator panel until its video begins.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from umi_flow_core import config as core_config
from umi_flow_core import layout
from umi_flow_server.dataset_stage import load_frame_stream, pose_track_for_umi
from umi_flow_server.trajectory import resample, states_from_track

LEFT_COLOR = (255, 178, 50)  # BGR cyan-blue
RIGHT_COLOR = (105, 92, 255)  # BGR coral
BG = (0, 0, 0)
PANEL_BG = BG
GRID = (35, 35, 35)
AXIS_COLORS = (
    (65, 65, 245),  # x: red (BGR)
    (80, 210, 110),  # y: green
    (245, 140, 55),  # z: blue
)


def nearest_past_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Index the most recent source frame at every target timestamp."""
    if len(source_t) == 0:
        raise ValueError("source timeline is empty")
    return np.clip(np.searchsorted(source_t, target_t, side="right") - 1, 0, len(source_t) - 1)


def operator_frame_index(elapsed_s: float, offset_s: float, fps: float) -> int | None:
    """External frame for an elapsed UMI time; None means operator video has not begun."""
    operator_t = elapsed_s + offset_s
    if operator_t < 0:
        return None
    return int(math.floor(operator_t * fps + 1e-9))


def shared_camera_duration(episode_dir: Path, master: str = "umi1") -> float:
    """Seconds from the master start until both onboard cameras still have frames."""
    streams = {name: load_frame_stream(episode_dir, name) for name in ("umi1", "umi2")}
    start = float(streams[master].t[0])
    end = min(float(stream.t[-1]) for stream in streams.values())
    return end - start


class IndexedVideo:
    """Decode monotonically requested video frames without retaining the full video."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = -1
        self.image: np.ndarray | None = None

    def frame(self, index: int | None) -> np.ndarray | None:
        if index is None or index < 0:
            return None
        if self.frame_count > 0 and index >= self.frame_count:
            return None
        if index < self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.index = index - 1
            self.image = None
        while self.index < index:
            ok, image = self.cap.read()
            if not ok:
                return None
            self.image = image
            self.index += 1
        return self.image

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> IndexedVideo:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize into an exact BGR canvas."""
    if width <= 0 or height <= 0:
        raise ValueError("letterbox dimensions must be positive")
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    x, y = (width - rw) // 2, (height - rh) // 2
    canvas[y : y + rh, x : x + rw] = resized
    return canvas


def video_panel(
    image: np.ndarray | None,
    panel_size: tuple[int, int],
) -> np.ndarray:
    width, height = panel_size
    if image is None:
        return np.full((height, width, 3), BG, dtype=np.uint8)
    return letterbox(image, width, height)


@dataclass(frozen=True)
class IsoProjector:
    """Fixed orthographic isometric projection fitted to a complete trajectory."""

    right: np.ndarray
    up: np.ndarray
    center: np.ndarray
    scale: float
    screen_center: np.ndarray

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        rect: tuple[int, int, int, int],
        *,
        azimuth_deg: float = 180.0,
        elevation_deg: float = 90.0,
    ) -> IsoProjector:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            finite = np.zeros((1, 3), dtype=np.float64)
        lower, upper = finite.min(axis=0), finite.max(axis=0)
        span = np.maximum(upper - lower, 0.25)
        center = (lower + upper) / 2.0
        lower, upper = center - span * 0.62, center + span * 0.62

        az, el = np.radians([azimuth_deg, elevation_deg])
        view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        # The analytic horizontal basis remains well-defined for a true top view,
        # where cross(view, world-up) would collapse to zero.
        right = np.array([math.sin(az), -math.cos(az), 0.0])
        up = np.cross(right, view)
        up /= np.linalg.norm(up)
        corners = np.asarray(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        projected = np.column_stack(((corners - center) @ right, (corners - center) @ up))
        extent = np.maximum(np.ptp(projected, axis=0), 1e-6)
        x, y, width, height = rect
        scale = min(width / extent[0], height / extent[1])
        return cls(
            right=right,
            up=up,
            center=center,
            scale=float(scale),
            screen_center=np.array([x + width / 2.0, y + height / 2.0]),
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        relative = points - self.center
        projected = np.column_stack((relative @ self.right, relative @ self.up))
        projected[:, 0] = self.screen_center[0] + projected[:, 0] * self.scale
        projected[:, 1] = self.screen_center[1] - projected[:, 1] * self.scale
        return np.rint(projected).astype(np.int32)


def _umi_to_display(points: np.ndarray) -> np.ndarray:
    """UMI (up, right, forward) -> display world (forward, right, up)."""
    points = np.asarray(points)
    return points[..., [2, 1, 0]]


def _draw_alpha_line(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    if len(points) < 2:
        return
    overlay = image.copy()
    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _draw_grid(
    panel: np.ndarray,
    projector: IsoProjector,
    positions: np.ndarray,
) -> None:
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    span = np.maximum(upper - lower, 0.25)
    lower -= span * 0.18
    upper += span * 0.18
    floor_z = min(0.0, float(lower[2]))
    for x in np.linspace(lower[0], upper[0], 7):
        pts = projector.project(np.array([[x, lower[1], floor_z], [x, upper[1], floor_z]]))
        cv2.line(panel, tuple(pts[0]), tuple(pts[1]), GRID, 1, cv2.LINE_AA)
    for y in np.linspace(lower[1], upper[1], 7):
        pts = projector.project(np.array([[lower[0], y, floor_z], [upper[0], y, floor_z]]))
        cv2.line(panel, tuple(pts[0]), tuple(pts[1]), GRID, 1, cv2.LINE_AA)


def _orientation_endpoints(state: np.ndarray, length: float) -> np.ndarray:
    """Display-space endpoints of the pose's local x/y/z frame."""
    rotation = Rotation.from_rotvec(state[3:6])
    axes_umi = rotation.apply(np.eye(3) * length)
    return _umi_to_display(state[:3] + axes_umi)


def slam_panel(
    states: dict[str, np.ndarray],
    index: int,
    projector: IsoProjector,
    panel_size: tuple[int, int],
) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)

    all_positions = np.concatenate([_umi_to_display(value[:, :3]) for value in states.values()])
    _draw_grid(panel, projector, all_positions)
    overall_span = max(float(np.ptp(all_positions, axis=0).max()), 0.25)

    colors = {"umi1": LEFT_COLOR, "umi2": RIGHT_COLOR}
    for name, state in states.items():
        color = colors.get(name, (255, 255, 255))
        path = projector.project(_umi_to_display(state[:, :3]))
        _draw_alpha_line(panel, path, color, 2, 0.18)
        _draw_alpha_line(panel, path[: index + 1], color, 4, 0.72)
        recent = path[max(0, index - 60) : index + 1]
        _draw_alpha_line(panel, recent, color, 6, 0.98)
        current = path[index]
        glow = panel.copy()
        cv2.circle(glow, tuple(current), 16, color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.22, panel, 0.78, 0.0, dst=panel)
        cv2.circle(panel, tuple(current), 7, color, -1, cv2.LINE_AA)
        endpoints = projector.project(
            _orientation_endpoints(state[index], max(0.09, overall_span * 0.1))
        )
        for endpoint, axis_color in zip(endpoints, AXIS_COLORS, strict=True):
            if np.linalg.norm(endpoint - current) < 7:
                cv2.circle(panel, tuple(current), 11, axis_color, 3, cv2.LINE_AA)
                cv2.circle(panel, tuple(current), 3, axis_color, -1, cv2.LINE_AA)
            else:
                cv2.arrowedLine(
                    panel,
                    tuple(current),
                    tuple(endpoint),
                    axis_color,
                    4,
                    cv2.LINE_AA,
                    tipLength=0.22,
                )
    return panel


@dataclass(frozen=True)
class DemoData:
    rig: core_config.RigConfig
    manifest: dict[str, Any]
    timeline: np.ndarray
    camera_paths: dict[str, Path]
    camera_indices: dict[str, np.ndarray]
    states: dict[str, np.ndarray]


def load_demo_data(
    episode_dir: Path,
    derived_episode_dir: Path,
    output_t: np.ndarray,
) -> DemoData:
    manifest_path = episode_dir / layout.MANIFEST_FILE
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    rig = core_config.load_rig_config(config_path)
    if tuple(u.name for u in rig.umis) != ("umi1", "umi2"):
        raise ValueError("demo renderer requires one paired umi1/umi2 episode")
    streams = {u.name: load_frame_stream(episode_dir, u.name) for u in rig.umis}

    sampled_tracks: dict[str, tuple[Any, np.ndarray]] = {}
    for umi in rig.umis:
        track, basis = pose_track_for_umi(episode_dir, derived_episode_dir, manifest_data, umi)
        sampled_tracks[umi.name] = (resample(track, output_t), basis)

    states: dict[str, np.ndarray] = {}
    shared_pos: np.ndarray | None = None
    shared_quat: np.ndarray | None = None
    for umi in rig.umis:
        track, basis = sampled_tracks[umi.name]
        if rig.alignment.enabled:
            if shared_pos is None:
                shared_pos, shared_quat = track.pos[0], track.quat[0]
            states[umi.name] = states_from_track(track, shared_pos, shared_quat, basis=basis)
        else:
            states[umi.name] = states_from_track(track, basis=basis)

    return DemoData(
        rig=rig,
        manifest=manifest_data,
        timeline=output_t,
        camera_paths={name: stream.video_path for name, stream in streams.items()},
        camera_indices={
            name: nearest_past_indices(stream.t, output_t) for name, stream in streams.items()
        },
        states=states,
    )


def _ffmpeg_process(
    output: Path,
    size: tuple[int, int],
    fps: float,
    preset: str,
) -> subprocess.Popen[bytes]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    width, height = size
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=UMI Flow synchronized dual-SLAM demo",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render(args: argparse.Namespace) -> None:
    episode_dir = args.episode_dir.resolve()
    derived_dir = args.derived_episode_dir.resolve()
    leader_stream = load_frame_stream(episode_dir, "umi1")
    duration = shared_camera_duration(episode_dir)
    if args.preview_seconds is not None:
        duration = min(duration, args.preview_seconds)
    if duration <= 0:
        raise ValueError("episode camera timeline has no positive duration")
    frame_count = int(math.floor(duration * args.fps)) + 1
    elapsed = np.arange(frame_count, dtype=np.float64) / args.fps
    output_t = leader_stream.t[0] + elapsed
    data = load_demo_data(episode_dir, derived_dir, output_t)

    if args.width % 4:
        raise ValueError("output width must be divisible by four")
    panel_size = (args.width // 4, args.height)
    positions = np.concatenate([_umi_to_display(state[:, :3]) for state in data.states.values()])
    projector = IsoProjector.fit(
        positions,
        (
            round(panel_size[0] * 0.05),
            round(panel_size[1] * 0.05),
            round(panel_size[0] * 0.86),
            round(panel_size[1] * 0.90),
        ),
    )

    operator_path = args.operator_video.resolve() if args.operator_video else None
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(IndexedVideo(path))
            for name, path in data.camera_paths.items()
        }
        operator = stack.enter_context(IndexedVideo(operator_path)) if operator_path else None
        output = args.output.resolve()
        partial = output.with_name(f".{output.stem}.partial{output.suffix}")
        partial.unlink(missing_ok=True)
        encoder = _ffmpeg_process(partial, (args.width, args.height), args.fps, args.preset)
        assert encoder.stdin is not None
        try:
            for i, time_s in enumerate(elapsed):
                operator_image = None
                if operator is not None:
                    operator_image = operator.frame(
                        operator_frame_index(float(time_s), args.operator_offset_s, operator.fps)
                    )
                left = readers["umi1"].frame(int(data.camera_indices["umi1"][i]))
                right = readers["umi2"].frame(int(data.camera_indices["umi2"][i]))
                panels = [
                    video_panel(operator_image, panel_size),
                    slam_panel(
                        data.states,
                        i,
                        projector,
                        panel_size,
                    ),
                    video_panel(left, panel_size),
                    video_panel(right, panel_size),
                ]
                encoder.stdin.write(np.hstack(panels).tobytes())
                if i == 0 or (i + 1) % max(1, round(args.fps * 2)) == 0:
                    print(
                        f"render {i + 1}/{frame_count} ({100 * (i + 1) / frame_count:.1f}%)",
                        flush=True,
                    )
        finally:
            encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        partial.replace(output)
    print(f"wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path, help="processed paired raw episode directory")
    parser.add_argument("derived_episode_dir", type=Path, help="matching derived episode directory")
    parser.add_argument("output", type=Path, help="single synchronized MP4 output")
    parser.add_argument("--operator-video", type=Path, help="external operator video")
    parser.add_argument(
        "--operator-offset-s",
        type=float,
        default=0.0,
        help="time in operator video corresponding to the first UMI frame",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-seconds", type=float)
    parser.add_argument("--preset", choices=("veryfast", "fast", "medium", "slow"), default="fast")
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps, width, and height must be positive")
    if args.preview_seconds is not None and args.preview_seconds <= 0:
        parser.error("--preview-seconds must be positive")
    return args


if __name__ == "__main__":
    try:
        render(parse_args())
    except (BrokenPipeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"render_slam_demo: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
