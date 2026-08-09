#!/usr/bin/env python3
"""Produce a per-camera fisheye calibration .npz (the file CAMERA_CALIB points at).

Prints a checkerboard? Any rigid flat one works (a phone/tablet display is fine too).
Show it to the camera from many angles and distances, covering the WHOLE frame —
corners and edges especially; the fisheye model is worst exactly where you skip.

Two input modes:
  * --video   a recording from the camera — the natural flow on this stack: record a
              throwaway episode while waving the board, then point this at the episode's
              umiN_camera.mjpeg.avi (frames are sampled every --stride).
  * --images  a glob of still frames, e.g. 'calib_shots/*.jpg'.

    uv run tools/calibrate_camera.py --video umi1_camera.mjpeg.avi --board 9x6 \
        --out config/calib/umi1_calib.npz

Output keys (the contract server/src/umi_flow_server/gripper.py:load_calib reads):
K (3,3), dist (4,1) Kannala-Brandt k1..k4, image_size [w,h], model="fisheye".
Calibrate at the camera's RECORDING resolution — K is rescaled automatically at use,
but the distortion solve is only as good as the frames it saw. Square size does not
matter (intrinsics are scale-free). Aim for 20+ accepted views and RMS < 1 px.
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)


def frames_from_video(path: str, stride: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video {path}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"{path}#{i}", frame
        i += 1
    cap.release()


def frames_from_images(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no images match {pattern!r}")
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield p, img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video from the camera (e.g. an episode's umiN_camera.mjpeg.avi)")
    src.add_argument("--images", help="glob of still frames, e.g. 'shots/*.jpg' (quote it)")
    ap.add_argument("--board", default="9x6", help="INNER corner grid of the checkerboard, COLSxROWS (default 9x6)")
    ap.add_argument("--stride", type=int, default=15, help="--video: sample every Nth frame (default 15)")
    ap.add_argument("--out", required=True, help="output .npz, e.g. config/calib/umi1_calib.npz")
    a = ap.parse_args()

    cols, rows = (int(x) for x in a.board.lower().split("x"))
    pattern = (cols, rows)
    # Board model: unit squares (real size is irrelevant to intrinsics). cv2.fisheye wants
    # each view's points shaped (1, N, 3)/(1, N, 2) in float64.
    obj = np.zeros((1, cols * rows, 3), np.float64)
    obj[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    frames = frames_from_video(a.video, a.stride) if a.video else frames_from_images(a.images)
    objpoints, imgpoints, names, size = [], [], [], None
    seen = 0
    for name, frame in frames:
        seen += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]  # (w, h)
        elif gray.shape[::-1] != size:
            sys.exit(f"{name}: frame size {gray.shape[::-1]} != first frame {size} — one resolution per calibration")
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),
        )
        objpoints.append(obj.copy())
        imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
        names.append(name)
    print(f"board found in {len(names)}/{seen} frames at {size[0]}x{size[1]}")
    if len(names) < 10:
        sys.exit("fewer than 10 usable views — capture more angles/coverage and rerun")

    # CALIB_CHECK_COND rejects near-degenerate views by raising with the culprit's index;
    # the standard cure is to drop that view and re-solve until the set is clean.
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    while True:
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, size, K, D, None, None, FLAGS,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8),
            )
            break
        except cv2.error as e:
            msg = str(e)
            if "input array" not in msg:
                raise
            bad = int(msg.split("input array ")[1].split(" ")[0])
            print(f"  dropping ill-conditioned view {names[bad]}")
            for lst in (objpoints, imgpoints, names):
                del lst[bad]
            if len(names) < 10:
                sys.exit("too many views rejected — recapture with the board fully visible, less motion blur")

    print(f"solved over {len(names)} views: RMS {rms:.3f} px" + (" (aim < 1 px — consider recapturing)" if rms > 1.0 else ""))
    print(f"K = fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print(f"dist (Kannala-Brandt) = {D.ravel().round(5).tolist()}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=K, dist=D, image_size=np.array(size, np.int64), model="fisheye")
    print(f"wrote {out} — set CAMERA_CALIB=\"{out}\" in the rig env (or camera.calib in the pair yaml)")


if __name__ == "__main__":
    main()
