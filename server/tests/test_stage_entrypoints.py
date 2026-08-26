"""Exercise stage-script failure handling without needing a ROS installation."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DECODE = (ROOT / "docker/decode/entrypoint.sh").read_text()


@pytest.mark.parametrize("counts", [None, {}, {"/rslidar_points": 0, "/rslidar_imu_data": 10}])
def test_decode_rejects_unusable_metadata(tmp_path: Path, counts) -> None:
    result = _validate_metadata(tmp_path, counts)
    assert result.returncode != 0
    assert "unusable decoded bag metadata" in result.stderr


def test_decode_accepts_both_required_topics(tmp_path: Path) -> None:
    result = _validate_metadata(tmp_path, {"/rslidar_points": 20, "/rslidar_imu_data": 400})
    assert result.returncode == 0, result.stderr


def _validate_metadata(tmp_path: Path, counts) -> subprocess.CompletedProcess:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        ""
        if counts is None
        else yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "topics_with_message_count": [
                        {"topic_metadata": {"name": name}, "message_count": count}
                        for name, count in counts.items()
                    ]
                }
            }
        )
    )
    validator = DECODE.split("python3 - \"$BAG_DIR/metadata.yaml\" <<'PY'\n", 1)[1].split(
        "\nPY", 1
    )[0]
    return subprocess.run(
        [sys.executable, "-", str(metadata)],
        input=validator,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_decode_shutdown_kills_child_ignoring_sigint() -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); "
            "print('ready'); time.sleep(60)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        functions = DECODE[DECODE.index("stop_group() {") : DECODE.index("trap cleanup EXIT")]
        result = subprocess.run(
            ["bash", "-c", functions + '\nstop_group "$1" 1', "test", str(child.pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert child.wait(timeout=3) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=3)


def test_both_stage_scripts_return_outputs_to_host_owner() -> None:
    for path in (ROOT / "docker/decode/entrypoint.sh", ROOT / "docker/slam-glim/entrypoint.sh"):
        script = path.read_text()
        assert "OUTPUT_OWNER=$(stat -c '%u:%g'" in script
        assert 'chown -R "$OUTPUT_OWNER" "$OUT_DIR"' in script
