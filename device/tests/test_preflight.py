from __future__ import annotations

import numpy as np
from umi_flow_device.preflight import (
    CHECK_FRAME_COUNT,
    AiryTimeSyncCheck,
    ArucoTagDetector,
    CheckResult,
    GripperTagVisibilityCheck,
    PipelineResult,
    PreRecordingPipeline,
    ServerAdmissionCheck,
    StorageHeadroomCheck,
)
from umi_flow_device.safety import SafetyPolicy


class FakeFrames:
    def __init__(self, frame_ids: list[int], error: Exception | None = None):
        self.frame_ids = frame_ids
        self.error = error
        self.requests: list[int] = []

    def capture(self, frame_count: int) -> list[np.ndarray]:
        self.requests.append(frame_count)
        if self.error is not None:
            raise self.error
        return [np.full((1, 1, 1), frame_id, dtype=np.uint8) for frame_id in self.frame_ids]


class FakeDetector:
    def __init__(self, detections: dict[int, set[int]]):
        self.detections = detections

    def detect_ids(self, frame: np.ndarray) -> set[int]:
        return self.detections[int(frame[0, 0, 0])]


def make_check(frames: FakeFrames, detections: dict[int, set[int]]) -> GripperTagVisibilityCheck:
    return GripperTagVisibilityCheck(
        umi="umi1",
        target_tag_ids=(2, 3),
        frames=frames,
        detector=FakeDetector(detections),
    )


def test_tag_check_requests_five_frames_and_accepts_tags_seen_across_window() -> None:
    frames = FakeFrames([0, 1, 2, 3, 4])
    check = make_check(frames, {0: {2}, 1: set(), 2: {3}, 3: set(), 4: set()})

    result = check.run()

    assert frames.requests == [CHECK_FRAME_COUNT]
    assert result.ok
    assert result.missing_tag_ids == ()


def test_tag_check_reports_each_tag_never_seen_in_window() -> None:
    frames = FakeFrames([0, 1, 2, 3, 4])
    check = make_check(frames, {index: {2} for index in range(5)})

    result = check.run()

    assert not result.ok
    assert result.umi == "umi1"
    assert result.missing_tag_ids == (3,)


def test_camera_failure_blocks_with_both_configured_tags_missing() -> None:
    frames = FakeFrames([], error=RuntimeError("camera busy"))
    check = make_check(frames, {})

    result = check.run()

    assert not result.ok
    assert result.missing_tag_ids == (2, 3)
    assert "camera busy" in result.message


def test_short_frame_window_is_rejected() -> None:
    frames = FakeFrames([0, 1, 2, 3])
    check = make_check(frames, {index: {2, 3} for index in range(4)})

    result = check.run()

    assert not result.ok
    assert result.missing_tag_ids == (2, 3)
    assert "4 of 5" in result.message


def test_pipeline_collects_modular_check_failures() -> None:
    class BrokenCheck:
        name = "future_check"
        umi = "umi2"

        def run(self):
            raise RuntimeError("not ready")

    result = PreRecordingPipeline((BrokenCheck(),)).run()

    assert not result.ok
    assert result.failures[0].check == "future_check"
    assert "not ready" in result.failures[0].message


def test_pipeline_result_roundtrips_through_coordinator_payload() -> None:
    original = PipelineResult(
        (
            CheckResult(
                check="gripper_tags",
                umi="umi2",
                ok=False,
                missing_tag_ids=(0, 1),
                message="not visible",
            ),
        )
    )

    assert PipelineResult.from_dict(original.to_dict()) == original


def test_airy_time_sync_check_requires_sensor_reported_gptp_lock() -> None:
    responses = iter(
        [
            {"sync_mode": "GPTP", "sync_sts": "Lock"},
            {"sync_mode": "GPTP", "sync_sts": "UnLock"},
        ]
    )
    def fetch(_url: str, _timeout: float):
        return next(responses)

    check = AiryTimeSyncCheck("umi1", "192.168.1.200", fetch_json=fetch)

    assert check.run().ok
    failure = check.run()
    assert not failure.ok
    assert failure.check == "lidar_time_sync"
    assert "UnLock" in failure.message


def test_airy_time_sync_check_blocks_on_diagnostic_failure() -> None:
    def fail(_url: str, _timeout: float):
        raise TimeoutError("sensor did not respond")

    result = AiryTimeSyncCheck("umi2", "192.168.1.200", fetch_json=fail).run()

    assert not result.ok
    assert "sensor did not respond" in result.message


def test_real_aruco_detector_reports_generated_marker_ids() -> None:
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((300, 600, 3), 255, dtype=np.uint8)
    for marker_id, x in ((2, 50), (3, 350)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 180)
        frame[60:240, x : x + 180] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    assert {2, 3}.issubset(ArucoTagDetector().detect_ids(frame))


def test_storage_headroom_check_uses_projected_post_take_reserve(tmp_path, monkeypatch) -> None:
    class Usage:
        free = 199

    monkeypatch.setattr("umi_flow_device.safety.shutil.disk_usage", lambda _path: Usage())
    check = StorageHeadroomCheck(
        "umi1",
        tmp_path,
        SafetyPolicy(
            min_free_bytes_after_take=100,
            estimated_take_bytes=75,
            metadata_reserve_bytes=25,
        ),
    )

    result = check.run()

    assert not result.ok
    assert "free=199 required=200" in result.message


def test_server_admission_check_fails_closed() -> None:
    check = ServerAdmissionCheck(
        "umi1",
        123,
        lambda size: {
            "accepted": False,
            "reason": f"cannot accept {size}",
        },
    )

    result = check.run()

    assert not result.ok
    assert result.message == "cannot accept 123"
