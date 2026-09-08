"""Modular checks that must pass before either client begins recording."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol
from urllib.request import urlopen

import numpy as np
from umi_flow_core.config import GRIPPER_ARUCO_DICTIONARY, UmiConfig

from umi_flow_device.safety import SafetyPolicy, storage_headroom

CHECK_FRAME_COUNT = 5
GRIPPER_TAG_CHECK = "gripper_tags"
LIDAR_TIME_SYNC_CHECK = "lidar_time_sync"
STORAGE_HEADROOM_CHECK = "storage_headroom"
SERVER_ADMISSION_CHECK = "server_admission"


@dataclass(frozen=True)
class CheckResult:
    """The serializable result of one pre-recording check on one UMI."""

    check: str
    umi: str
    ok: bool
    missing_tag_ids: tuple[int, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "umi": self.umi,
            "ok": self.ok,
            "missing_tag_ids": list(self.missing_tag_ids),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> CheckResult:
        if not isinstance(raw, dict):
            raise ValueError("check result must be a mapping")
        missing = raw.get("missing_tag_ids", [])
        if not isinstance(missing, list) or any(type(tag_id) is not int for tag_id in missing):
            raise ValueError("missing_tag_ids must be a list of integers")
        if type(raw.get("ok")) is not bool:
            raise ValueError("check result ok must be a boolean")
        return cls(
            check=str(raw.get("check", "")),
            umi=str(raw.get("umi", "")),
            ok=raw["ok"],
            missing_tag_ids=tuple(missing),
            message=str(raw.get("message", "")),
        )


@dataclass(frozen=True)
class PipelineResult:
    """All check results from one client."""

    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.checks if not result.ok)

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [result.to_dict() for result in self.checks]}

    @classmethod
    def from_dict(cls, raw: Any) -> PipelineResult:
        if not isinstance(raw, dict) or not isinstance(raw.get("checks"), list):
            raise ValueError("pre-recording result must contain a checks list")
        result = cls(tuple(CheckResult.from_dict(item) for item in raw["checks"]))
        if type(raw.get("ok")) is not bool or raw["ok"] != result.ok:
            raise ValueError("pre-recording result has an inconsistent ok value")
        return result


class PreRecordingCheck(Protocol):
    """One independently runnable pre-recording check."""

    name: str
    umi: str

    def run(self) -> CheckResult: ...


class FrameSource(Protocol):
    def capture(self, frame_count: int) -> list[np.ndarray]: ...


class TagDetector(Protocol):
    def detect_ids(self, frame: np.ndarray) -> set[int]: ...


class PreRecordingPipeline:
    """Run modular checks and convert unexpected check errors into failures."""

    def __init__(self, checks: tuple[PreRecordingCheck, ...]):
        self.checks = checks

    def run(self) -> PipelineResult:
        results: list[CheckResult] = []
        for check in self.checks:
            try:
                results.append(check.run())
            except Exception as exc:
                results.append(
                    CheckResult(
                        check=check.name,
                        umi=check.umi,
                        ok=False,
                        message=f"check failed: {exc}",
                    )
                )
        return PipelineResult(tuple(results))


@dataclass(frozen=True)
class OpenCvCameraFrames:
    """Open a V4L2 camera briefly and return exactly the requested BGR frames."""

    device: str
    width: int
    height: int
    fps: int
    exposure: int | None = None

    def capture(self, frame_count: int) -> list[np.ndarray]:
        cv2 = import_module("cv2")
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"could not open camera {self.device}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if self.exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

            frames: list[np.ndarray] = []
            for index in range(frame_count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"camera returned only {index} of {frame_count} frames")
                frames.append(frame)
            return frames
        finally:
            cap.release()


class ArucoTagDetector:
    """Detect marker IDs only; gripper geometry is irrelevant to this visibility check."""

    def __init__(self, dictionary: str = GRIPPER_ARUCO_DICTIONARY):
        cv2 = import_module("cv2")
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self._detector = cv2.aruco.ArucoDetector(aruco_dictionary, cv2.aruco.DetectorParameters())
        self._cv2 = cv2

    def detect_ids(self, frame: np.ndarray) -> set[int]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        _corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None:
            return set()
        return {int(tag_id) for tag_id in ids.flatten()}


class GripperTagVisibilityCheck:
    """Require each configured gripper tag to appear in a five-frame window."""

    name = GRIPPER_TAG_CHECK

    def __init__(
        self,
        umi: str,
        target_tag_ids: tuple[int, int],
        frames: FrameSource,
        detector: TagDetector,
        frame_count: int = CHECK_FRAME_COUNT,
    ):
        self.umi = umi
        self.target_tag_ids = target_tag_ids
        self.frames = frames
        self.detector = detector
        self.frame_count = frame_count

    def run(self) -> CheckResult:
        try:
            frames = self.frames.capture(self.frame_count)
        except Exception as exc:
            return CheckResult(
                check=self.name,
                umi=self.umi,
                ok=False,
                missing_tag_ids=tuple(sorted(self.target_tag_ids)),
                message=f"camera check failed: {exc}",
            )

        if len(frames) != self.frame_count:
            return CheckResult(
                check=self.name,
                umi=self.umi,
                ok=False,
                missing_tag_ids=tuple(sorted(self.target_tag_ids)),
                message=f"camera returned {len(frames)} of {self.frame_count} frames",
            )

        seen: set[int] = set()
        for frame in frames:
            seen.update(self.detector.detect_ids(frame))
        missing = tuple(sorted(set(self.target_tag_ids) - seen))
        return CheckResult(
            check=self.name,
            umi=self.umi,
            ok=not missing,
            missing_tag_ids=missing,
            message="" if not missing else "configured tags were not visible in five frames",
        )


def _fetch_json(url: str, timeout: float) -> Any:
    import json

    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed local LiDAR URL
        return json.load(response)


class AiryTimeSyncCheck:
    """Require the sensor itself—not merely the Pi's ptp4l service—to report GPTP lock."""

    name = LIDAR_TIME_SYNC_CHECK

    def __init__(
        self,
        umi: str,
        lidar_ip: str,
        *,
        fetch_json: Callable[[str, float], Any] = _fetch_json,
        timeout: float = 2.0,
    ):
        self.umi = umi
        self.lidar_ip = lidar_ip
        self.fetch_json = fetch_json
        self.timeout = timeout

    def run(self) -> CheckResult:
        try:
            raw = self.fetch_json(f"http://{self.lidar_ip}/diag_data.json", self.timeout)
            if not isinstance(raw, dict):
                raise ValueError("diagnostic response is not a mapping")
            mode = str(raw.get("sync_mode", "")).strip()
            status = str(raw.get("sync_sts", "")).strip()
        except Exception as exc:
            return CheckResult(
                check=self.name,
                umi=self.umi,
                ok=False,
                message=f"Airy time-sync diagnostic failed: {exc}",
            )

        ok = mode.upper() == "GPTP" and status.lower() in {"lock", "locked"}
        return CheckResult(
            check=self.name,
            umi=self.umi,
            ok=ok,
            message="" if ok else f"Airy {mode or 'unknown mode'} is {status or 'unknown'}",
        )


class StorageHeadroomCheck:
    """Require space for a conservative take plus the post-take reserve."""

    name = STORAGE_HEADROOM_CHECK

    def __init__(self, umi: str, spool_dir: Path, policy: SafetyPolicy):
        self.umi = umi
        self.spool_dir = spool_dir
        self.policy = policy

    def run(self) -> CheckResult:
        try:
            result = storage_headroom(self.spool_dir, self.policy)
        except Exception as exc:
            return CheckResult(
                check=self.name,
                umi=self.umi,
                ok=False,
                message=f"storage check failed: {exc}",
            )
        return CheckResult(
            check=self.name,
            umi=self.umi,
            ok=result.ok,
            message=(
                ""
                if result.ok
                else "insufficient recording storage: "
                f"free={result.free_bytes} required={result.required_free_bytes}"
            ),
        )


class ServerAdmissionCheck:
    """Require the ingest server to reserve enough headroom for this arm's upload."""

    name = SERVER_ADMISSION_CHECK

    def __init__(
        self,
        umi: str,
        estimated_upload_bytes: int,
        probe: Callable[[int], dict[str, Any]],
    ):
        self.umi = umi
        self.estimated_upload_bytes = estimated_upload_bytes
        self.probe = probe

    def run(self) -> CheckResult:
        try:
            result = self.probe(self.estimated_upload_bytes)
            accepted = result.get("accepted")
            if type(accepted) is not bool:
                raise ValueError("response has no boolean accepted field")
        except Exception as exc:
            return CheckResult(
                check=self.name,
                umi=self.umi,
                ok=False,
                message=f"server admission check failed: {exc}",
            )
        return CheckResult(
            check=self.name,
            umi=self.umi,
            ok=accepted,
            message="" if accepted else str(result.get("reason") or "server rejected recording"),
        )


def build_pre_recording_pipeline(
    umi: UmiConfig,
    *,
    spool_dir: Path | None = None,
    safety_policy: SafetyPolicy | None = None,
    server_admission_probe: Callable[[int], dict[str, Any]] | None = None,
) -> PreRecordingPipeline:
    if umi.gripper is None:
        raise ValueError(f"{umi.name} has no gripper tag configuration")
    frames = OpenCvCameraFrames(
        device=umi.camera.device,
        width=umi.camera.width,
        height=umi.camera.height,
        fps=umi.camera.fps,
        exposure=umi.camera.exposure,
    )
    gripper_check = GripperTagVisibilityCheck(
        umi=umi.name,
        target_tag_ids=umi.gripper.tag_ids,
        frames=frames,
        detector=ArucoTagDetector(),
    )
    checks: list[PreRecordingCheck] = [
        AiryTimeSyncCheck(umi=umi.name, lidar_ip=umi.lidar.lidar_ip),
        gripper_check,
    ]
    if spool_dir is not None:
        policy = safety_policy or SafetyPolicy.from_env()
        checks.append(
            StorageHeadroomCheck(
                umi=umi.name,
                spool_dir=spool_dir,
                policy=policy,
            )
        )
        if server_admission_probe is not None:
            checks.append(
                ServerAdmissionCheck(
                    umi=umi.name,
                    # The fixed rig always commits two arms. Each independent
                    # check reserves against the whole pair so simultaneous
                    # reads cannot both approve insufficient aggregate space.
                    estimated_upload_bytes=2 * policy.estimated_take_bytes,
                    probe=server_admission_probe,
                )
            )
    return PreRecordingPipeline(tuple(checks))


def build_gripper_tag_pipeline(umi: UmiConfig) -> PreRecordingPipeline:
    """Backward-compatible alias for callers deployed before the clock preflight."""
    return build_pre_recording_pipeline(umi)
