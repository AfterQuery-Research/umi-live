#!/usr/bin/env python3
"""Live gripper-width viewer: streams both Pis' cameras, overlays ArUco detections and
the computed width per arm, and serves a browser page. Run from the repo root:

    uv run --package umi-flow-server python tools/live_gripper_view.py
    # then open http://localhost:8899

Per-frame overlay: every detected tag (outline + id), the arm's configured pair joined
by a line with the width in mm (computed with THAT arm's camera geometry), and a
warning when the tags seen belong to the other arm (swapped finger assemblies).
Uses the same detector/unprojection/geometry as the processing pipeline, so what you
see is what the dataset would record. Cameras must be idle (not mid-recording).
"""

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from umi_flow_core.config import (
    GRIPPER_ARUCO_DICTIONARY,
    load_gripper_tag_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server" / "src"))
from umi_flow_server import gripper  # noqa: E402

TAG_PAIRS = load_gripper_tag_pairs(ROOT / "config/gripper_tags.yaml")

# Per-arm camera->tag geometry: the cameras were remounted at different forward
# offsets (2026-08-07), so the arms no longer share one GripperGeometry.
ARMS = [
    dict(
        name="umi1",
        host="umi1@umi1.local",
        calib="config/calib/umi1_calib.npz",
        tags=TAG_PAIRS["umi1"],
        geom=gripper.GripperGeometry(plane_mm=61.9, front_mm=30.1),
    ),
    dict(
        name="umi2",
        host="umi2@umi2.local",
        calib="config/calib/umi2_calib.npz",
        tags=TAG_PAIRS["umi2"],
        geom=gripper.GripperGeometry(plane_mm=61.9, front_mm=30.1),
    ),
]
W, H, PORT = 800, 600, 8899

STREAM_CMD = (
    "dev=$(ls /dev/v4l/by-path/*usb-*-video-index0 | head -1); "
    f'v4l2-ctl -d "$dev" --set-fmt-video=width={W},height={H},pixelformat=MJPG '
    "--stream-mmap --stream-to=- 2>/dev/null"
)

latest: dict[str, bytes] = {}  # arm name -> annotated JPEG
cond = threading.Condition()


def annotate(arm: dict, img: np.ndarray, det, calib) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    centers = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten(), strict=False):
            pts = c.reshape(4, 2).astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            ctr = pts.mean(0).astype(int)
            centers[int(i)] = c.reshape(4, 2).mean(0)
            cv2.putText(
                img, str(int(i)), tuple(ctr + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )
    a, b = arm["tags"]
    other = tuple(
        tag_id for pair in TAG_PAIRS.values() for tag_id in pair if tag_id not in arm["tags"]
    )
    if a in centers and b in centers:
        r0 = gripper.unproject_fisheye(centers[a], calib.K, calib.D)
        r1 = gripper.unproject_fisheye(centers[b], calib.K, calib.D)
        w_mm = gripper.width_from_rays(r0, r1, arm["geom"])
        p0, p1 = centers[a].astype(int), centers[b].astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (255, 0, 255), 2)
        cv2.putText(
            img,
            f"{arm['name']}  width = {w_mm:.1f} mm",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            6,
        )
        cv2.putText(
            img,
            f"{arm['name']}  width = {w_mm:.1f} mm",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (80, 255, 80),
            2,
        )
    else:
        missing = [t for t in arm["tags"] if t not in centers]
        msg = f"{arm['name']}: waiting for tags {missing}"
        if all(t in centers for t in other):
            msg = f"{arm['name']}: sees OTHER arm's tags {other} - fingers swapped?"
        cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2)
    return img


def worker(arm: dict) -> None:
    calib = gripper.scale_calib(gripper.load_calib(arm["calib"]), W, H)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, GRIPPER_ARUCO_DICTIONARY))
    det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    while True:
        proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", arm["host"], STREAM_CMD],
            stdout=subprocess.PIPE,
        )
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf = (buf + chunk)[-4_000_000:]
                end = buf.rfind(b"\xff\xd9")
                if end == -1:
                    continue
                start = buf.rfind(b"\xff\xd8", 0, end)
                if start == -1:
                    continue
                frame, buf = buf[start : end + 2], buf[end + 2 :]
                img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = annotate(arm, img, det, calib)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with cond:
                        latest[arm["name"]] = jpg.tobytes()
                        cond.notify_all()
        finally:
            proc.kill()
        print(f"{arm['name']}: stream ended; reconnecting", flush=True)


INDEX = f"""<html><head><title>UMI live gripper view</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h2>Live gripper width + tag detections</h2>
<div><img src="/umi1.mjpg" width="{W}"> <img src="/umi2.mjpg" width="{W}"></div>
<p>green = detected tag · magenta = the arm's pair · width uses per-arm geometry
(umi1 front_mm=30.1, umi2 front_mm=30.1)</p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/":
            body = INDEX.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return
        name = self.path.strip("/").removesuffix(".mjpg")
        if name not in {a["name"] for a in ARMS}:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with cond:
                    cond.wait(timeout=5)
                    jpg = latest.get(name)
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    for arm in ARMS:
        threading.Thread(target=worker, args=(arm,), daemon=True).start()
    print(f"serving http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
