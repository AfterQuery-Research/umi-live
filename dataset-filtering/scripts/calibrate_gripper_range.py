#!/usr/bin/env python3
"""Interactively calibrate one UMI gripper's closed/open width.

Run from the umi-flow repository root while the selected UMI camera is idle:

    uv run --package umi-flow-server python \
        dataset-filtering/scripts/calibrate_gripper_range.py --gripper umi1

Close the selected gripper and press Enter, then open it and press Enter. Each
press records a short burst using the same ArUco/fisheye width detector as the
dataset pipeline. The median endpoints are written to one JSON calibration file.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
from umi_flow_core.config import load_gripper_tag_pairs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server" / "src"))
sys.path.insert(0, str(ROOT / "dataset-filtering" / "src"))

from umi_flow_server import gripper as gripper_measurement  # noqa: E402

from dataset_filtering.gripper_calibration import (  # noqa: E402
    build_two_point_calibration_report,
    write_gripper_calibration,
)

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_SAMPLES = 30
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_STD_MM = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one UMI gripper's 0/1 width range")
    parser.add_argument("--gripper", choices=("umi1", "umi2"), required=True)
    parser.add_argument(
        "--host", help="SSH target; defaults to LEADER/FOLLOWER in provisioning/pair.env"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-std-mm", type=float, default=DEFAULT_MAX_STD_MM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--plane-mm", type=float, default=61.9)
    parser.add_argument("--front-mm", type=float, default=30.1)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if args.timeout_seconds <= 0 or args.max_std_mm <= 0:
        parser.error("timeout and stability threshold must be positive")

    name = args.gripper
    host = args.host or _configured_host(name)
    output = args.output or ROOT / "config" / "calib" / f"{name}_gripper_range.json"
    calibration_path = ROOT / "config" / "calib" / f"{name}_calib.npz"
    tag_pairs = load_gripper_tag_pairs(ROOT / "config" / "gripper_tags.yaml")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"camera calibration not found: {calibration_path}")

    camera_calibration = gripper_measurement.scale_calib(
        gripper_measurement.load_calib(calibration_path), args.width, args.height
    )
    geometry = gripper_measurement.GripperGeometry(
        plane_mm=args.plane_mm, front_mm=args.front_mm
    )
    detector = gripper_measurement.GripperDetector(
        camera_calibration, geometry, tag_pairs[name], roi_top_frac=0.0
    )

    print(f"Calibrating {name} using {host} and tags {list(tag_pairs[name])}.")
    print("The camera must be idle and both gripper tags must be visible.")
    closed = _prompt_and_capture(
        "Fully CLOSE the gripper, hold it still, then press Enter... ",
        host=host,
        detector=detector,
        width=args.width,
        height=args.height,
        samples=args.samples,
        timeout_seconds=args.timeout_seconds,
        max_std_mm=args.max_std_mm,
    )
    opened = _prompt_and_capture(
        "Fully OPEN the gripper, hold it still, then press Enter... ",
        host=host,
        detector=detector,
        width=args.width,
        height=args.height,
        samples=args.samples,
        timeout_seconds=args.timeout_seconds,
        max_std_mm=args.max_std_mm,
    )
    report = build_two_point_calibration_report(
        gripper=name,
        closed_samples_mm=closed,
        open_samples_mm=opened,
        measurement={
            "host": host,
            "camera_calibration": str(calibration_path.relative_to(ROOT)),
            "tag_ids": list(tag_pairs[name]),
            "frame_size": [args.width, args.height],
            "geometry": {"plane_mm": args.plane_mm, "front_mm": args.front_mm},
            "requested_samples_per_endpoint": args.samples,
            "maximum_endpoint_std_mm": args.max_std_mm,
        },
    )
    path = write_gripper_calibration(report, output)
    print(
        json.dumps(
            {
                "gripper": name,
                "closed_width_mm": report["closed_width_mm"],
                "open_width_mm": report["open_width_mm"],
                "span_mm": report["span_mm"],
                "output": str(path),
            },
            indent=2,
        )
    )


def _configured_host(gripper: str) -> str:
    key = "LEADER" if gripper == "umi1" else "FOLLOWER"
    inventory = ROOT / "provisioning" / "pair.env"
    if inventory.is_file():
        for raw_line in inventory.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key:
                continue
            parsed = shlex.split(value, comments=True)
            if len(parsed) == 1 and "@" in parsed[0]:
                return parsed[0]
    raise RuntimeError(
        f"could not resolve {key} from {inventory}; pass --host user@hostname explicitly"
    )


def _prompt_and_capture(prompt: str, **capture_options: Any) -> np.ndarray:
    while True:
        input(prompt)
        values = _capture_width_burst(**capture_options)
        median = float(np.median(values))
        std = float(np.std(values))
        print(
            f"  captured {len(values)} samples: median={median:.3f} mm, "
            f"std={std:.3f} mm, range={np.min(values):.3f}–{np.max(values):.3f} mm"
        )
        if std <= float(capture_options["max_std_mm"]):
            return values
        print(
            f"  Endpoint was not stable (std > {capture_options['max_std_mm']:.3f} mm). "
            "Hold it still and retry."
        )


def _capture_width_burst(
    *,
    host: str,
    detector: Any,
    width: int,
    height: int,
    samples: int,
    timeout_seconds: float,
    max_std_mm: float,
) -> np.ndarray:
    del max_std_mm  # checked by the caller after a complete burst
    stream_command = (
        "dev=$(ls /dev/v4l/by-path/*usb-*-video-index0 | head -1); "
        f'v4l2-ctl -d "$dev" --set-fmt-video=width={width},height={height},pixelformat=MJPG '
        "--stream-mmap --stream-to=- 2>/dev/null"
    )
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, stream_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("could not open SSH camera stream")
    buffer = b""
    values: list[float] = []
    deadline = monotonic() + timeout_seconds
    try:
        while len(values) < samples and monotonic() < deadline:
            if process.poll() is not None:
                detail = process.stderr.read().decode(errors="replace") if process.stderr else ""
                raise RuntimeError(f"camera stream ended before calibration: {detail.strip()}")
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                continue
            buffer += chunk
            while len(values) < samples:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buffer = buffer[start:]
                    break
                encoded = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                value = detector.width_mm(rgb)
                if value is not None and np.isfinite(value):
                    values.append(float(value))
    finally:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
    if len(values) < samples:
        raise RuntimeError(
            f"only detected both tags in {len(values)}/{samples} frames within "
            f"{timeout_seconds:g}s; improve tag visibility and retry"
        )
    return np.asarray(values, dtype=np.float64)


if __name__ == "__main__":
    main()
