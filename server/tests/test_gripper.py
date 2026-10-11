import numpy as np
import pytest
from umi_live_server.gripper import (
    GripperGeometry,
    gripper_widths,
    identity_calib,
    load_calib,
    unproject_fisheye,
    width_from_rays,
)


def test_identity_calib_pinhole_from_resolution():
    cal = identity_calib(1920, 1080)
    assert cal.model == "fisheye"
    assert cal.image_size == (1920, 1080)
    np.testing.assert_allclose(cal.D, np.zeros(4))
    # principal point at center, focal = 1.5 * min(W, H)
    assert cal.K[0, 2] == pytest.approx(960.0)
    assert cal.K[1, 2] == pytest.approx(540.0)
    assert cal.K[0, 0] == pytest.approx(1.5 * 1080)
    # D=0 -> equidistant fisheye (theta = r/f, NOT a rectilinear pinhole): a pixel offset
    # of 100px in x from the center unprojects to [sin(theta), 0, cos(theta)], theta = 100/f
    f = cal.K[0, 0]
    theta = 100.0 / f
    ray = unproject_fisheye(np.array([960.0 + 100.0, 540.0]), cal.K, cal.D)
    expected = np.array([np.sin(theta), 0.0, np.cos(theta)])
    np.testing.assert_allclose(ray, expected, atol=1e-6)


def test_scale_calib_preserves_unprojection_across_resolutions():
    from umi_live_server.gripper import FisheyeCalib, scale_calib

    K = np.array([[541.4, 0, 950.5], [0, 541.4, 612.2], [0, 0, 1.0]])
    D = np.array([-0.043, -0.001, -0.002, 2.6e-5])
    calib = FisheyeCalib(K=K, D=D, image_size=(1920, 1200), model="fisheye")

    assert scale_calib(calib, 1920, 1200) is calib  # no-op when sizes match

    half = scale_calib(calib, 960, 600)  # half resolution
    assert half.image_size == (960, 600)
    np.testing.assert_allclose(half.K[0, 0], 541.4 * 0.5)  # fx scales
    np.testing.assert_allclose(half.K[0, 2], 950.5 * 0.5)  # cx scales
    np.testing.assert_allclose(half.D, D)  # KB dist is resolution-free
    # a pixel at the same FRACTIONAL position unprojects to the same camera ray
    full_ray = unproject_fisheye(np.array([1300.0, 800.0]), calib.K, calib.D)
    half_ray = unproject_fisheye(np.array([650.0, 400.0]), half.K, half.D)
    np.testing.assert_allclose(half_ray, full_ray, atol=1e-9)


def test_perp_mm():
    g = GripperGeometry(plane_mm=61.9, front_mm=37.1)
    assert g.perp_mm == pytest.approx(np.hypot(61.9, 37.1))


def test_width_from_rays_known_angle_and_calibration():
    # two rays symmetric about +z at half-angle a -> angle between = 2a -> W = 2*L*tan(a)
    a = np.radians(20.0)
    r0 = np.array([np.sin(a), 0.0, np.cos(a)])
    r1 = np.array([-np.sin(a), 0.0, np.cos(a)])
    g = GripperGeometry(plane_mm=61.9, front_mm=37.1, scale=1.0, offset=0.0)
    expected = 2.0 * g.perp_mm * np.tan(a)
    assert width_from_rays(r0, r1, g) == pytest.approx(expected, rel=1e-6)
    # the linear calibration is applied on top
    g2 = GripperGeometry(plane_mm=61.9, front_mm=37.1, scale=0.5, offset=3.0)
    assert width_from_rays(r0, r1, g2) == pytest.approx(0.5 * expected + 3.0, rel=1e-6)


def test_unproject_fisheye_matches_cv2_projection():
    cv2 = pytest.importorskip("cv2")
    K = np.array([[500.0, 0, 960.0], [0, 500.0, 600.0], [0, 0, 1.0]])
    D = np.array([0.05, -0.02, 0.01, -0.005])
    pts = np.array([[[0.1, 0.05, 1.0], [-0.3, 0.2, 1.0], [0.5, -0.4, 1.0]]], dtype=np.float64)
    px, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    for p3, p2 in zip(pts.reshape(-1, 3), px.reshape(-1, 2), strict=False):
        ray = unproject_fisheye(p2, K, D)
        expected = p3 / np.linalg.norm(p3)
        assert float(np.dot(ray, expected)) > 0.9999  # ray recovers the 3D direction


# --- gripper_widths: a fake detector keyed by frame index (decode = identity) ---


class FakeDetector:
    def __init__(self, widths):
        self.widths = widths

    def width_mm(self, rgb):  # rgb is the frame index here
        return self.widths[rgb]


def _widths(seq):
    return gripper_widths(list(range(len(seq))), FakeDetector(seq), lambda f: f)


def test_gripper_widths_interpolates_gaps():
    out = _widths([100.0, None, None, 130.0])
    assert out.shape == (4, 1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.ravel(), [100, 110, 120, 130], atol=1e-4)


def test_gripper_widths_clamps_endpoints():
    out = _widths([None, 100.0, 120.0, None])
    np.testing.assert_allclose(out.ravel(), [100, 100, 120, 120], atol=1e-4)  # held at ends


def test_gripper_widths_uses_raw_without_smoothing():
    # no temporal filtering: every detected frame's width passes through unchanged
    out = _widths([100.0, 100.0, 500.0, 100.0, 100.0])
    np.testing.assert_allclose(out.ravel(), [100, 100, 500, 100, 100], atol=1e-4)


def test_gripper_widths_single_valid_is_constant():
    out = _widths([None, 117.0, None])
    np.testing.assert_allclose(out.ravel(), [117, 117, 117], atol=1e-4)


def test_gripper_widths_all_missing_returns_none():
    assert _widths([None, None, None]) is None


def test_load_calib_rejects_non_fisheye(tmp_path):
    p = tmp_path / "cal.npz"
    np.savez(
        p, K=np.eye(3), dist=np.zeros((8, 1)), image_size=np.array([1920, 1200]), model="rational"
    )
    with pytest.raises(ValueError, match="fisheye"):
        load_calib(p)


def test_load_calib_reads_fisheye(tmp_path):
    p = tmp_path / "cal.npz"
    K = np.array([[542.0, 0, 950.0], [0, 542.0, 612.0], [0, 0, 1.0]])
    np.savez(
        p,
        K=K,
        dist=np.array([0.1, -0.03, 0.08, -0.07]),
        image_size=np.array([1920, 1200]),
        model="fisheye",
    )
    cal = load_calib(p)
    assert cal.model == "fisheye"
    assert cal.D.shape == (4,)
    assert cal.image_size == (1920, 1200)


def test_roi_detection_matches_full_frame():
    cv2 = pytest.importorskip("cv2")
    from umi_live_server.gripper import FisheyeCalib, GripperDetector

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[360:440, 100:180] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 0, 80), cv2.COLOR_GRAY2RGB
    )
    frame[360:440, 400:480] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 1, 80), cv2.COLOR_GRAY2RGB
    )
    calib = FisheyeCalib(
        K=np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
        D=np.zeros(4),
        image_size=(640, 480),
        model="fisheye",
    )
    full = GripperDetector(calib, GripperGeometry(), (0, 1), roi_top_frac=0.0).width_mm(frame)
    roi = GripperDetector(calib, GripperGeometry(), (0, 1), roi_top_frac=0.5).width_mm(frame)
    assert full is not None and roi is not None
    # the y-offset maps ROI corners back to full-frame coords -> identical width
    assert roi == pytest.approx(full, rel=1e-6)


def test_detector_returns_none_with_only_one_tag():
    cv2 = pytest.importorskip("cv2")
    from umi_live_server.gripper import FisheyeCalib, GripperDetector

    # render a single DICT_4X4_50 marker (id 0) on a white frame -> both tags not present
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 0, 200)
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[100:300, 100:300] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    calib = FisheyeCalib(
        K=np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
        D=np.zeros(4),
        image_size=(640, 480),
        model="fisheye",
    )
    det = GripperDetector(calib, GripperGeometry(), (0, 1))
    assert det.width_mm(frame) is None  # only tag 0 present, need both 0 and 1


def test_detector_per_arm_target_ids():
    cv2 = pytest.importorskip("cv2")
    from umi_live_server.gripper import GripperConfig, detector_from_config

    # The frame carries only one configured pair (2 and 3).
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((480, 640, 3), 255, np.uint8)
    frame[360:440, 100:180] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 2, 80), cv2.COLOR_GRAY2RGB
    )
    frame[360:440, 400:480] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 3, 80), cv2.COLOR_GRAY2RGB
    )
    cfg = GripperConfig(geom=GripperGeometry())
    d23 = detector_from_config(cfg, 640, 480, target_ids=(2, 3))
    d01 = detector_from_config(cfg, 640, 480, target_ids=(0, 1))
    assert d23.width_mm(frame) is not None  # sees its own tags
    assert d01.width_mm(frame) is None  # other configured IDs are absent -> no false read
