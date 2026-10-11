#!/usr/bin/env python3
"""Calibrate one UMI fisheye camera from a live stream, video, or image set.

Live mode reads MJPEG directly from an idle Pi over SSH and opens a local browser
viewer. Sweep the matching ChArUco board across the whole image, especially its edges
and bottom. Red cells still need coverage; green cells are covered; yellow cells contain
the board now. Capture stops automatically when coverage is sufficient.

    uv run --group tools python tools/calibrate_camera.py \
        --host umi1@umi1.local --out config/calib/umi1_calib.npz

Offline inputs use the same board and solver:

    uv run --group tools python tools/calibrate_camera.py \
        --video umi1_camera.mjpeg.avi --out config/calib/umi1_calib.npz

Generate the matching board with tools/make_charuco.py. The output contract is the one
server/src/umi_live_server/gripper.py reads: K, dist, image_size, model="fisheye".
"""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import tempfile
import threading
import time
import webbrowser
from collections.abc import Generator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 8, 5
SQUARE_LEN, MARKER_LEN = 0.030, 0.022
BOARD_DICT = "DICT_5X5_100"
MIN_CORNERS = 8
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 800, 600, 30
MAX_GOOD_RMS_PX = 1.5
MIN_INVERTIBLE_DEG = 73.0
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "StrictHostKeyChecking=accept-new",
)


def ssh_command(host: str, remote: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, remote]


def preflight_live_host(host: str) -> None:
    """Read-only proof that provisioning completed and the camera is idle."""
    remote = (
        "set -eu; "
        "command -v v4l2-ctl >/dev/null || "
        "{ echo 'v4l2-ctl missing; run ./bin/provision-pair first' >&2; exit 20; }; "
        "systemctl is-active --quiet umi-recorder || "
        "{ echo 'umi-recorder is not active; finish provisioning first' >&2; exit 21; }; "
        "if pgrep -f '[u]mi-flow-v4l2-recorder' >/dev/null; then "
        "echo 'camera is recording; stop the take before calibration' >&2; exit 22; fi; "
        "dev=$(ls /dev/v4l/by-path/*usb-*-video-index0 2>/dev/null | head -1 || true); "
        "test -n \"$dev\" || { echo 'no UMI camera found' >&2; exit 23; }"
    )
    result = subprocess.run(
        ssh_command(host, remote),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise SystemExit(f"{host} is not ready for calibration: {detail}")


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)


def coverage_cell(point, width: int, height: int, cols: int, rows: int) -> tuple[int, int]:
    return (
        min(max(int(point[0] / width * cols), 0), cols - 1),
        min(max(int(point[1] / height * rows), 0), rows - 1),
    )


def turning_point_deg(dist: np.ndarray) -> float:
    """Angle at which the Kannala-Brandt radius stops being invertible."""
    theta = np.linspace(0, np.radians(110), 20_000)
    d = np.asarray(dist).reshape(4)
    radius = theta * (1 + d[0] * theta**2 + d[1] * theta**4 + d[2] * theta**6 + d[3] * theta**8)
    return float(np.degrees(theta[np.argmax(radius)]))


def draw_overlay(
    frame: np.ndarray,
    covered: np.ndarray,
    current: set[tuple[int, int]],
    view_count: int,
    board_found: bool,
) -> np.ndarray:
    rows, cols = covered.shape
    height, width = frame.shape[:2]
    cell_width, cell_height = width / cols, height / rows
    fill = frame.copy()
    for row in range(rows):
        for col in range(cols):
            p0 = (int(col * cell_width), int(row * cell_height))
            p1 = (int((col + 1) * cell_width), int((row + 1) * cell_height))
            if covered[row, col]:
                cv2.rectangle(fill, p0, p1, (0, 150, 0), -1)
            else:
                cv2.rectangle(frame, p0, p1, (0, 0, 255), 1)
            if (col, row) in current:
                cv2.rectangle(frame, p0, p1, (0, 255, 255), 2)
    result = cv2.addWeighted(fill, 0.25, frame, 0.75, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        result,
        f"views {view_count}   coverage {covered.mean() * 100:.0f}%",
        (12, 38),
        font,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        result,
        "board detected" if board_found else "no board - aim the camera at it",
        (12, 72),
        font,
        0.7,
        (0, 220, 0) if board_found else (0, 0, 255),
        2,
    )
    cv2.putText(
        result,
        "fill the RED cells, especially the edges and bottom",
        (12, height - 16),
        font,
        0.55,
        (0, 255, 255),
        2,
    )
    return result


def split_jpegs(data: bytes) -> tuple[list[bytes], bytes]:
    """Extract complete JPEGs and retain a possibly incomplete final JPEG."""
    frames: list[bytes] = []
    while True:
        start = data.find(b"\xff\xd8")
        if start < 0:
            return frames, data[-1:] if data.endswith(b"\xff") else b""
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            return frames, data[start:]
        frames.append(data[start : end + 2])
        data = data[end + 2 :]


def frames_from_ssh(
    host: str,
    device: str,
    width: int,
    height: int,
    fps: int,
) -> Generator[tuple[str, np.ndarray], None, None]:
    if device == "auto":
        select_device = "dev=$(ls /dev/v4l/by-path/*usb-*-video-index0 2>/dev/null | head -1)"
    else:
        select_device = f"dev={shlex.quote(device)}"
    remote = (
        f'{select_device}; test -n "$dev" || {{ echo "no camera found" >&2; exit 1; }}; '
        f'exec v4l2-ctl -d "$dev" --set-fmt-video=width={width},height={height},pixelformat=MJPG '
        f"--set-parm={fps} --stream-mmap --stream-to=-"
    )
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ssh_command(host, remote),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        assert process.stdout is not None
        buffer = b""
        frame_index = 0
        try:
            while chunk := process.stdout.read(65_536):
                buffer += chunk
                encoded_frames, buffer = split_jpegs(buffer)
                # Staying live is more useful than processing a backlog.
                for encoded in encoded_frames[-1:]:
                    frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield f"{host}#{frame_index}", frame
                        frame_index += 1
                if len(buffer) > 8_000_000:
                    buffer = buffer[-2_000_000:]
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if frame_index == 0:
            errors.seek(0)
            detail = errors.read().decode(errors="replace").strip()
            raise SystemExit(
                f"camera stream from {host} produced no frames" + (f": {detail}" if detail else "")
            )


def frames_from_video(path: str, stride: int) -> Generator[tuple[str, np.ndarray], None, None]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield f"{path}#{index}", frame
            index += 1
    finally:
        capture.release()


def frames_from_images(pattern: str) -> Generator[tuple[str, np.ndarray], None, None]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no images match {pattern!r}")
    for path in paths:
        frame = cv2.imread(path)
        if frame is not None:
            yield path, frame


class BrowserViewer:
    def __init__(self, port: int):
        self.port = port
        self.latest: bytes | None = None
        self.status = "waiting for camera"
        self.done = False
        self.condition = threading.Condition()
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self):
                if self.path == "/":
                    body = (
                        b"<!doctype html><title>UMI camera calibration</title>"
                        b'<body style="margin:0;background:#111;color:#eee;text-align:center;'
                        b'font-family:sans-serif"><h2>UMI camera calibration</h2>'
                        b'<img src="/stream.mjpg" style="max-width:96vw;max-height:82vh">'
                        b'<p id="status">waiting for camera</p><script>setInterval(async()=>{'
                        b"const r=await fetch('/status');const j=await r.json();"
                        b"document.getElementById('status').textContent=j.status},500)</script></body>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/status":
                    body = json.dumps({"status": viewer.status}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        with viewer.condition:
                            viewer.condition.wait_for(
                                lambda last_frame=last: (
                                    viewer.latest is not last_frame or viewer.done
                                ),
                                timeout=1,
                            )
                            frame = viewer.latest
                            done = viewer.done
                        if frame is not None and frame is not last:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                            )
                            last = frame
                        if done:
                            return
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self, open_browser: bool) -> None:
        self.thread.start()
        print(f"live coverage viewer: {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def update(self, frame: np.ndarray, status: str) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.condition:
            if ok:
                self.latest = encoded.tobytes()
            self.status = status
            self.condition.notify_all()

    def close(self, status: str) -> None:
        with self.condition:
            self.status = status
            self.done = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def min_spread(corners: np.ndarray) -> float:
    points = corners.reshape(-1, 2)
    eigenvalues = np.linalg.eigvalsh(np.cov((points - points.mean(0)).T))
    return float(np.sqrt(max(float(eigenvalues.min()), 0.0)))


def usable_view(corners: np.ndarray, min_std_px: float = 4.0) -> bool:
    return len(corners.reshape(-1, 2)) >= MIN_CORNERS and min_spread(corners) > min_std_px


def save_raw_views(
    path: Path, views: list[tuple[np.ndarray, np.ndarray]], image_size: tuple[int, int]
) -> None:
    corners = np.empty(len(views), dtype=object)
    ids = np.empty(len(views), dtype=object)
    corners[:] = [view[0] for view in views]
    ids[:] = [view[1] for view in views]
    np.savez(path, corners=corners, ids=ids, imsize=np.array(image_size))


def load_raw_views(path: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    with np.load(path, allow_pickle=True) as data:
        views = [
            (np.asarray(corners), np.asarray(ids))
            for corners, ids in zip(data["corners"], data["ids"], strict=True)
        ]
        size_values = [int(value) for value in data["imsize"]]
    if len(size_values) != 2:
        raise SystemExit(f"{path}: imsize must contain width and height")
    image_size = (size_values[0], size_values[1])
    return views, image_size


def solve_and_save(
    captured: list[tuple[np.ndarray, np.ndarray]], image_size: tuple[int, int], output: Path
) -> tuple[float, np.ndarray, np.ndarray]:
    """Robust Kannala-Brandt solve for partial ChArUco views."""
    board_points = make_board().getChessboardCorners()
    views = [(corners, ids) for corners, ids in captured if usable_view(corners)]
    sparse = len(captured) - len(views)
    if len(views) < 8:
        raise SystemExit(f"only {len(views)} usable views; sweep more of the frame")

    width, height = image_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )
    focal_seeds = [width * scale for scale in (0.30, 0.40, 0.22, 0.50, 0.18)]
    outliers_dropped = 0
    rms = None
    matrix = distortion = rotation_vectors = translation_vectors = None

    for _attempt in range(6):
        object_points = [
            board_points[ids.flatten()].reshape(-1, 1, 3).astype(np.float64) for _, ids in views
        ]
        image_points = [corners.reshape(-1, 1, 2).astype(np.float64) for corners, _ in views]
        for focal in focal_seeds:
            initial = np.array(
                [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]],
                dtype=np.float64,
            )
            try:
                rms, matrix, distortion, rotation_vectors, translation_vectors = (
                    cv2.fisheye.calibrate(
                        object_points,
                        image_points,
                        image_size,
                        initial,
                        np.zeros((4, 1)),
                        flags=flags,
                        criteria=criteria,
                    )
                )
                break
            except cv2.error:
                rms = None
        if rms is None:
            if len(views) <= 10:
                raise SystemExit("fisheye solve failed; capture more varied board views") from None
            views.pop(min(range(len(views)), key=lambda index: min_spread(views[index][0])))
            outliers_dropped += 1
            continue

        assert matrix is not None and distortion is not None
        assert rotation_vectors is not None and translation_vectors is not None
        errors = np.array(
            [
                np.linalg.norm(
                    cv2.fisheye.projectPoints(
                        object_points[index],
                        rotation_vectors[index],
                        translation_vectors[index],
                        matrix,
                        distortion,
                    )[0].reshape(-1, 2)
                    - image_points[index].reshape(-1, 2),
                    axis=1,
                ).mean()
                for index in range(len(views))
            ]
        )
        threshold = max(2.0, float(np.median(errors)) * 1.8)
        outliers = {index for index, error in enumerate(errors) if error > threshold}
        if not outliers or len(views) - len(outliers) < 12:
            break
        views = [view for index, view in enumerate(views) if index not in outliers]
        outliers_dropped += len(outliers)

    if rms is None or matrix is None or distortion is None:
        raise SystemExit("fisheye solve failed; capture more varied board views")
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise SystemExit("fisheye solve produced non-finite parameters; recapture")

    invertible_to = turning_point_deg(distortion)
    rms_good = rms < MAX_GOOD_RMS_PX
    invertible = invertible_to > MIN_INVERTIBLE_DEG
    quality = "GOOD" if rms_good else ("marginal" if rms < 4 else "BAD - recapture")
    print(
        f"RMS {rms:.2f} px at {width}x{height} ({len(views)} views; "
        f"dropped {sparse} sparse + {outliers_dropped} outliers): {quality}"
    )
    print(
        f"K: fx={matrix[0, 0]:.1f} fy={matrix[1, 1]:.1f} "
        f"cx={matrix[0, 2]:.1f} cy={matrix[1, 2]:.1f}"
    )
    print(f"dist: {distortion.ravel().round(6).tolist()}")
    print(
        f"invertible to {invertible_to:.1f} deg: "
        + ("GOOD" if invertible else "LOW - capture more edge and bottom coverage")
    )
    if not rms_good or not invertible:
        raise SystemExit(
            "calibration rejected; existing output was not changed. "
            "Capture more varied views with strong edge and bottom coverage."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez(
            file,
            K=matrix,
            dist=distortion,
            image_size=np.array(image_size, dtype=np.int64),
            model="fisheye",
        )
    temporary.replace(output)
    print(f"wrote {output}")
    return float(rms), matrix, distortion


def collect_offline(
    frames: Iterator[tuple[str, np.ndarray]], detector, limit: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    views: list[tuple[np.ndarray, np.ndarray]] = []
    image_size = None
    seen = 0
    for name, frame in frames:
        seen += 1
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise SystemExit(f"{name}: frame size {size} differs from {image_size}")
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if corners is not None and ids is not None and usable_view(corners):
            views.append((corners, ids))
            if len(views) >= limit:
                break
    if image_size is None:
        raise SystemExit("input contained no readable frames")
    print(f"board found in {len(views)}/{seen} sampled frames at {image_size[0]}x{image_size[1]}")
    return views, image_size


def collect_live(
    frames: Iterator[tuple[str, np.ndarray]],
    detector,
    viewer: BrowserViewer,
    grid: tuple[int, int],
    target_coverage: float,
    limit: int,
    seconds: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    cols, rows = grid
    covered = np.zeros((rows, cols), dtype=bool)
    views: list[tuple[np.ndarray, np.ndarray]] = []
    image_size = None
    last_centroid = None
    last_capture = 0.0
    started = last_progress = time.monotonic()

    for _name, frame in frames:
        now = time.monotonic()
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise SystemExit(f"camera resolution changed from {image_size} to {size}")

        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        board_found = corners is not None and ids is not None and usable_view(corners)
        current: set[tuple[int, int]] = set()
        if board_found:
            current = {
                coverage_cell(point, size[0], size[1], cols, rows)
                for point in corners.reshape(-1, 2)
            }
            new_cells = [cell for cell in current if not covered[cell[1], cell[0]]]
            centroid = corners.reshape(-1, 2).mean(0)
            moved = last_centroid is None or np.linalg.norm(centroid - last_centroid) > 100
            should_capture = (new_cells and now - last_capture > 0.3) or (
                moved and now - last_capture > 1.0
            )
            if should_capture and len(views) < limit:
                views.append((corners, ids))
                last_centroid = centroid
                last_capture = now
                if new_cells:
                    last_progress = now
                for col, row in current:
                    covered[row, col] = True
                print(
                    f"view {len(views)}/{limit}: coverage {covered.mean() * 100:.0f}% "
                    f"(+{len(new_cells)} cells)",
                    flush=True,
                )

        status = f"{len(views)} views; {covered.mean() * 100:.0f}% coverage"
        viewer.update(draw_overlay(frame, covered, current, len(views), board_found), status)
        coverage = float(covered.mean())
        if len(views) >= limit:
            break
        if len(views) >= 20 and (coverage >= target_coverage or now - last_progress > 25):
            break
        if now - started >= seconds:
            break

    if image_size is None:
        raise SystemExit("camera stream ended before producing a frame")
    return views, image_size


def parse_grid(value: str) -> tuple[int, int]:
    try:
        cols, rows = (int(part) for part in value.lower().split("x"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("grid must be COLSxROWS, for example 8x6") from None
    if cols < 1 or rows < 1:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return cols, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="idle Pi camera over SSH, e.g. umi1@umi1.local")
    source.add_argument("--video", help="recorded camera video")
    source.add_argument("--images", help="quoted glob of still frames")
    source.add_argument("--from-views", help="re-solve a previously saved .views.npz")
    source.add_argument("--preflight-host", help="check that a provisioned Pi is ready, then exit")
    parser.add_argument("--out", type=Path, help="output calibration .npz")
    parser.add_argument("--device", default="auto", help="remote V4L2 device (default: detect it)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--stride", type=int, default=15, help="offline video frame stride")
    parser.add_argument("--views", type=int, default=80, help="maximum accepted views")
    parser.add_argument("--grid", type=parse_grid, default=(8, 6), metavar="COLSxROWS")
    parser.add_argument("--coverage", type=float, default=0.72, help="live auto-solve threshold")
    parser.add_argument("--seconds", type=float, default=150.0, help="live capture time limit")
    parser.add_argument("--viewer-port", type=int, default=8898)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser viewer")
    args = parser.parse_args()

    if args.preflight_host:
        preflight_live_host(args.preflight_host)
        print(f"{args.preflight_host} is ready for camera calibration")
        return
    if args.out is None:
        parser.error("--out is required for calibration")
    if args.from_views:
        views, image_size = load_raw_views(args.from_views)
        print(f"loaded {len(views)} raw views at {image_size[0]}x{image_size[1]}")
        solve_and_save(views, image_size, args.out)
        return
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")
    if min(args.width, args.height, args.fps, args.stride, args.views) < 1 or args.seconds <= 0:
        parser.error("dimensions, fps, stride, views, and seconds must be positive")

    detector = cv2.aruco.CharucoDetector(make_board())
    if args.host:
        preflight_live_host(args.host)
        viewer = BrowserViewer(args.viewer_port)
        viewer.start(open_browser=not args.no_open)
        frames = frames_from_ssh(args.host, args.device, args.width, args.height, args.fps)
        try:
            views, image_size = collect_live(
                frames,
                detector,
                viewer,
                args.grid,
                args.coverage,
                args.views,
                args.seconds,
            )
        except KeyboardInterrupt:
            viewer.close("aborted")
            print("aborted")
            return
        except BaseException:
            viewer.close("capture failed; see terminal")
            raise
        finally:
            frames.close()
        viewer.close("capture complete; solving")
    else:
        frames = (
            frames_from_video(args.video, args.stride)
            if args.video
            else frames_from_images(args.images)
        )
        views, image_size = collect_offline(frames, detector, args.views)

    if len(views) < 8:
        raise SystemExit(f"only {len(views)} usable views captured; sweep more of the frame")
    views_path = args.out.with_name(args.out.name + ".views.npz")
    save_raw_views(views_path, views, image_size)
    print(f"saved {len(views)} raw views to {views_path}")
    solve_and_save(views, image_size, args.out)


if __name__ == "__main__":
    main()
