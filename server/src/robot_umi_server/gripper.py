"""ArUco gripper-width measurement (UMI-style), computed in post-processing.

Two ArUco tags (IDs 0 and 1) ride the gripper's two sliding jaws on a plane fixed
to the camera. We don't estimate depth from marker size (degenerate for small tags
at the fisheye edge); instead we use the FIXED camera->tag-plane geometry: unproject
each tag's center to a camera ray, take the angle between the rays, and combine with
the perpendicular distance L = hypot(plane_mm, front_mm) ->  W = 2*L*tan(theta/2),
then an optional linear calibration W_true = scale*W + offset. With the fisheye solved
over the full FOV (so the tags' ~70 deg off-axis position falls inside the model's
invertible range), the raw geometry is stable and accurate, so scale/offset default to
identity (1/0) and no temporal smoothing is applied — the per-frame width is used as-is.

This runs inside the pipeline (uv venv) where cv2 is 4.12 (opencv-python-headless),
so it uses the NEW aruco API (ArucoDetector), unlike tools/aruco_stream.py (4.6).
The geometry helpers are pure numpy; cv2 is imported lazily, only by the detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

ARUCO_DICT = "DICT_4X4_50"  # the gripper tags' dictionary
TARGET_IDS = (0, 1)


@dataclass(frozen=True)
class GripperGeometry:
    """Fixed camera<->tag-plane geometry + the linear calibration (mm)."""

    plane_mm: float = 61.9  # camera +Y (down) to the tag plane
    front_mm: float = 37.1  # camera +Z (forward) to the tags
    scale: float = 1.0  # W_true = scale * W_raw + offset; identity now that the fisheye
    offset: float = 0.0  # calib covers the tags' angular range -> raw geometry is used

    @property
    def perp_mm(self) -> float:
        """Perpendicular camera->tag-line distance L = hypot(down, forward)."""
        return float(np.hypot(self.plane_mm, self.front_mm))


@dataclass(frozen=True)
class FisheyeCalib:
    """Kannala-Brandt fisheye intrinsics (solved at image_size, native resolution)."""

    K: np.ndarray  # (3, 3)
    D: np.ndarray  # (4,)
    image_size: tuple[int, int] | None  # (width, height) the calibration was solved at
    model: str


@dataclass(frozen=True)
class GripperConfig:
    """Everything the pipeline needs to compute gripper width; None = disabled.

    ``calib_path`` is optional: without it the detector falls back to identity/pinhole
    intrinsics derived from the frame resolution (see ``identity_calib``), so the width is
    approximate/relative — not metric mm. Drop in a real fisheye calib later for true mm.
    """

    geom: GripperGeometry
    roi_top_frac: float = 0.0  # crop this fraction off the top before detection (0 = full frame)
    calib_path: Path | None = None  # None -> identity calib (uncalibrated) from the frame res


def identity_calib(width: int, height: int) -> FisheyeCalib:
    """Identity (uncalibrated) fisheye intrinsics from the frame resolution.

    A stand-in when no real calibration is available: focal f = 1.5*min(W, H), principal
    point at the image center, zero distortion. With D=0 this is the simplest *equidistant*
    fisheye (theta = r/f, not a rectilinear pinhole) and does not match the real lens, so
    the tag-to-tag width is APPROXIMATE/relative — no metric scale — but fine for a live LCD
    readout and to fill the dataset gripper dim. Swap in load_calib(<fisheye .npz>) for mm.
    """
    f = 1.5 * float(min(width, height))
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return FisheyeCalib(K=K, D=np.zeros(4), image_size=(width, height), model="fisheye")


def scale_calib(calib: FisheyeCalib, width: int, height: int) -> FisheyeCalib:
    """Rescale a calib's K to a different frame resolution (the KB dist is resolution-free).

    The detector unprojects raw pixel coords against K, so K must match the frame the tags
    are detected on. If the calib was solved at a different resolution than the frames come
    in at, scale fx/fy/cx/cy per-axis. No-op when sizes match or image_size is unknown.
    """
    if calib.image_size is None:
        return calib
    cw, ch = calib.image_size
    if cw <= 0 or ch <= 0 or (cw, ch) == (width, height):
        return calib
    sx, sy = width / cw, height / ch
    K = calib.K.astype(np.float64).copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return FisheyeCalib(K=K, D=calib.D, image_size=(width, height), model=calib.model)


def detector_from_config(
    cfg: GripperConfig,
    width: int,
    height: int,
    target_ids: tuple[int, int] = TARGET_IDS,
) -> GripperDetector:
    """Build a GripperDetector for a frame size, from cfg.calib_path or an identity calib.

    Centralizes the resolution-dependent construction shared by the live LCD monitor
    (scripts/run.py) and the post-processing pipeline. A real calib is rescaled to the
    frame resolution (see scale_calib) so it stays metric whatever resolution the camera
    publishes; with no calib, an identity calib is built at the frame resolution.
    ``target_ids`` are the two ArUco IDs on this arm's gripper (per-arm, so a camera that
    glimpses the other gripper can't confuse them): umi1 = (0, 1), umi2 = (2, 3).
    """
    if cfg.calib_path is not None:
        calib = scale_calib(load_calib(cfg.calib_path), width, height)
    else:
        calib = identity_calib(width, height)
    return GripperDetector(calib, cfg.geom, cfg.roi_top_frac, target_ids)


def load_calib(path: str | Path) -> FisheyeCalib:
    """Load the .npz written by tools/calibrate_camera.py (keys K, dist, image_size, model)."""
    data = np.load(str(path))
    model = str(data["model"]) if "model" in data.files else "fisheye"
    if model != "fisheye":
        raise ValueError(
            f"camera calibration model is {model!r}; gripper width needs a 'fisheye' "
            "calibration (the ray math + native-resolution K assume Kannala-Brandt)"
        )
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["dist"], dtype=np.float64).ravel()
    image_size: tuple[int, int] | None = None
    if "image_size" in data.files:
        wh = np.asarray(data["image_size"]).ravel()
        image_size = (int(wh[0]), int(wh[1]))
    return FisheyeCalib(K=K, D=D, image_size=image_size, model=model)


def unproject_fisheye(pix: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Pixel -> unit ray (camera frame) via the Kannala-Brandt fisheye model.

    Direct 1-D Newton inversion of r = theta*(1 + k1 th^2 + ... + k4 th^8) — robust at
    the periphery where cv2.fisheye.undistortPoints (iterative, 2-D) diverges. OpenCV
    camera axes: +x right, +y down, +z forward.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = (pix[0] - cx) / fx, (pix[1] - cy) / fy
    r = float(np.hypot(x, y))
    if r < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    th = r
    for _ in range(12):
        t2 = th * th
        poly = 1 + D[0] * t2 + D[1] * t2**2 + D[2] * t2**3 + D[3] * t2**4
        dpoly = 1 + 3 * D[0] * t2 + 5 * D[1] * t2**2 + 7 * D[2] * t2**3 + 9 * D[3] * t2**4
        th -= (th * poly - r) / dpoly
    s = np.sin(th)
    return np.array([s * x / r, s * y / r, np.cos(th)])


def width_from_rays(r0: np.ndarray, r1: np.ndarray, geom: GripperGeometry) -> float:
    """Calibrated tag-to-tag distance (mm) from two unit rays: W = 2*L*tan(theta/2)."""
    cosang = float(np.clip(np.dot(r0, r1), -1.0, 1.0))
    theta = float(np.arccos(cosang))
    w_raw = 2.0 * geom.perp_mm * float(np.tan(theta / 2.0))
    return geom.scale * w_raw + geom.offset


class GripperDetector:
    """Per-frame gripper width from the two ArUco tags (cv2 4.12 'new' aruco API).

    The tags ride a fixed band near the bottom of the frame (they're rigid to the camera,
    so their vertical position doesn't change as the rig moves — only the gripper opening
    slides them apart). ArUco on the full 2 MP native frame is the dominant cost
    (~0.5 s/frame on the Pi); cropping to that bottom band before detection cuts it
    several-fold. Detected corners are offset back to full-frame coordinates so the
    fisheye unprojection (which uses the native-resolution K) is unchanged.
    ``roi_top_frac`` is the fraction cropped off the top; 0 detects on the whole frame.
    """

    def __init__(
        self,
        calib: FisheyeCalib,
        geom: GripperGeometry,
        roi_top_frac: float = 0.0,
        target_ids: tuple[int, int] = TARGET_IDS,
    ):
        import cv2  # lazy: only the detector needs cv2; the math above is pure numpy

        self.calib = calib
        self.geom = geom
        self.roi_top_frac = min(max(roi_top_frac, 0.0), 0.95)
        self.target_ids = tuple(target_ids)  # this arm's two gripper tags (e.g. (0,1) or (2,3))
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self._cv2 = cv2

    def width_mm(self, rgb: np.ndarray) -> float | None:
        """Width (mm) for one NATIVE-resolution RGB frame, or None if both tags aren't seen."""
        cv2 = self._cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        y0 = int(gray.shape[0] * self.roi_top_frac)  # detect only the bottom band the tags sit in
        corners, ids, _ = self._detector.detectMarkers(gray[y0:])
        if ids is None:
            return None
        offset = np.array([0.0, float(y0)])  # ROI -> full-frame coords (native K unprojection)
        centers = {
            int(m): c.reshape(4, 2).mean(0) + offset
            for c, m in zip(corners, ids.flatten(), strict=False)
        }
        a, b = self.target_ids
        if a not in centers or b not in centers:  # only this arm's own tags count
            return None
        r0 = unproject_fisheye(centers[a], self.calib.K, self.calib.D)
        r1 = unproject_fisheye(centers[b], self.calib.K, self.calib.D)
        return width_from_rays(r0, r1, self.geom)


class WidthSource(Protocol):
    """Anything with a per-frame width_mm (GripperDetector, or a test fake)."""

    def width_mm(self, rgb: np.ndarray) -> float | None: ...


def gripper_widths(
    frames: list[Any],
    detector: WidthSource,
    decode: Callable[[Any], np.ndarray],
) -> np.ndarray | None:
    """Per-frame gripper width as (N, 1) float32, or None if no frame saw both tags.

    Detects on each frame's NATIVE decode and interpolates only over the frames that
    missed a tag (endpoints held, like filter_trajectory). No temporal smoothing: with
    the fisheye recalibrated so the tags fall inside the model's invertible range, the
    per-frame width is stable (~0.1 mm per pixel of center noise), so the raw value is
    written directly.
    """
    n = len(frames)
    raw: list[float | None] = [detector.width_mm(decode(f)) for f in frames]
    valid_idx = np.array([i for i, w in enumerate(raw) if w is not None], dtype=np.int64)
    if len(valid_idx) == 0:
        return None
    valid_w = np.array([raw[i] for i in valid_idx], dtype=np.float64)
    filled = np.interp(np.arange(n), valid_idx, valid_w)  # fill only the gaps, clamp ends
    return filled.reshape(n, 1).astype(np.float32)
