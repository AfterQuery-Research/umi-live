"""Two arm uploads merge into the only processable bimanual episode."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from umi_flow_core import config as core_config
from umi_flow_core import layout, manifest
from umi_flow_device.episode import EpisodeRecorder
from umi_flow_server import pair_stage
from umi_flow_server.runner import Executor, Runner, default_stages
from umi_flow_test_support import MockLidarPcapRecorder

from tests.helpers import CvCameraRecorder


def _write_arm_config(
    config_root: Path, umi_name: str, tag_ids: tuple[int, int], alignment: dict | None = None
) -> Path:
    (config_root / "config" / "calib").mkdir(parents=True, exist_ok=True)
    (config_root / "config" / "calib" / f"{umi_name}_calib.npz").write_bytes(
        b"npz-" + umi_name.encode()
    )
    raw = {
        "version": 1,
        "task": "test task",
        "umis": {
            umi_name: {
                "camera": {
                    "device": f"/dev/{umi_name}",
                    "width": 64,
                    "height": 48,
                    "fps": 30,
                    "calib": f"config/calib/{umi_name}_calib.npz",
                },
                "lidar": {"iface": "eth0", "host_ip": "192.168.1.102"},
                "gripper": {"tag_ids": list(tag_ids)},
            }
        },
    }
    if alignment is not None:
        raw["alignment"] = alignment
    path = config_root / "config" / "umi.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _record_arm(
    tmp_path: Path, umi_name: str, *, session_id, role, tag_ids, clock_sync=None, alignment=None
) -> Path:
    config_root = tmp_path / f"repo_{umi_name}"
    config_path = _write_arm_config(config_root, umi_name, tag_ids, alignment=alignment)
    rig = core_config.load_rig_config(config_path)
    u = rig.umis[0]
    cam_key = layout.sensor_key(u.name, layout.KIND_CAMERA)
    lidar_key = layout.sensor_key(u.name, layout.KIND_LIDAR)
    recorder = EpisodeRecorder(
        rig,
        config_path,
        tmp_path / f"spool_{umi_name}",
        {cam_key: CvCameraRecorder(cam_key, frames=6), lidar_key: MockLidarPcapRecorder(lidar_key)},
        config_root=config_root,
        ptp_probe=lambda: {
            u.lidar.iface: manifest.PtpStatus(
                locked=True, timestamping_mode="hardware", residual_offset_ns=1_000
            )
        },
        clock_sync_probe=(lambda cs=clock_sync: cs),
    )
    recorder.start(session_id=session_id, role=role)
    return recorder.stop()


def _publish_pair(
    tmp_path: Path, session_id="s1", *, leader_cs=None, follower_cs=None, alignment=None
) -> Path:
    """Two single-arm episodes landed in raw_episodes, as two Pis' uploads would.

    ``alignment`` is a rig-global block carried by the LEADER config (the follower's own
    config has only umi2, so a reference like 'umi1' would fail its standalone validation);
    pair_stage's merge takes top-level keys from the leader, so it lands in the merged rig."""
    data_dir = tmp_path / "data"
    raw = data_dir / "raw_episodes"
    raw.mkdir(parents=True)
    leader = _record_arm(
        tmp_path,
        "umi1",
        session_id=session_id,
        role="umi1",
        tag_ids=(2, 3),
        clock_sync=leader_cs,
        alignment=alignment,
    )
    follower = _record_arm(
        tmp_path,
        "umi2",
        session_id=session_id,
        role="umi2",
        tag_ids=(0, 1),
        clock_sync=follower_cs,
    )
    for ep in (leader, follower):
        ep.rename(raw / ep.name)
    return data_dir


def _fake_runner(data_dir: Path, calls: list[tuple[str, str]]) -> Runner:
    def fake(name):
        def fn(episode_dir: Path, out_dir: Path, scope: str) -> None:
            calls.append((name, scope))
            if name == "glim":
                (out_dir / "trajectory.tum").write_text("# t x y z qx qy qz qw\n")
                # the map artifacts a real align stage would consume (unread by the fake)
                (out_dir / "map.ply").write_text("ply\n")
                (out_dir / "submaps").mkdir(exist_ok=True)
                (out_dir / "submaps" / "poses.txt").write_text("# id stamp x y z qx qy qz qw\n")

        return fn

    stages = default_stages(
        dataset_fn=fake("dataset"), gripper_fn=fake("gripper"), align_fn=fake("align")
    )
    for s in stages:
        if s.fn is None:
            s.fn = fake(s.name)
    return Runner(data_dir=data_dir, stages=stages, executor=Executor(data_dir))


def test_merge_produces_valid_bimanual_episode(tmp_path: Path):
    data_dir = _publish_pair(tmp_path)
    res = pair_stage.pair_once(data_dir)
    assert len(res.merged) == 1

    raw = data_dir / "raw_episodes"
    merged = raw / res.merged[0]
    # The two single-arm sources are relocated OUT of raw_episodes.
    assert list(raw.iterdir()) == [merged]
    assert len(list((data_dir / "paired_sources").iterdir())) == 2

    m = manifest.read_manifest(merged / layout.MANIFEST_FILE)
    assert manifest.present_streams(m, layout.KIND_LIDAR) == ["umi1/lidar", "umi2/lidar"]
    assert manifest.present_streams(m, layout.KIND_CAMERA) == ["umi1/camera", "umi2/camera"]
    assert m["episode_id"] == merged.name
    assert m["device_id"] == "paired" and m["session_id"] == "s1" and "role" not in m

    rig = core_config.load_rig_config(
        merged / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    )
    assert core_config.derive(rig).state_dim == 12
    for f in (
        "umi1_lidar.pcap",
        "umi2_lidar.pcap",
        "umi1_camera.mjpeg.avi",
        "umi2_camera.mjpeg.avi",
    ):
        assert (merged / f).is_file()
    # calib blobs for both arms rode along.
    assert (merged / layout.CONFIG_SNAPSHOT_DIR / "config" / "calib" / "umi2_calib.npz").is_file()


def test_runner_processes_merged_as_bimanual(tmp_path: Path):
    data_dir = _publish_pair(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = _fake_runner(data_dir, calls)

    # The runner pairs first; a role-tagged arm is never processed alone.
    runner.scan_once()
    assert ("glim", "umi1") in calls
    assert ("glim", "umi2") in calls
    assert ("gripper", "") in calls
    assert ("dataset", "") in calls
    # Non-joint by default: no alignment block => the align stage is never scheduled and
    # the dataset consumes the per-lidar GLIM trajectories directly.
    assert ("align", "") not in calls


def test_alignment_enabled_schedules_align_between_glim_and_dataset(tmp_path: Path):
    data_dir = _publish_pair(tmp_path, alignment={"enabled": True, "reference": "umi1"})
    calls: list[tuple[str, str]] = []
    runner = _fake_runner(data_dir, calls)

    pair_stage.pair_once(data_dir)
    runner.scan_once()

    assert ("align", "") in calls
    ai = calls.index(("align", ""))
    # align waits for BOTH lidars' glim; dataset waits for align.
    assert calls.index(("glim", "umi1")) < ai
    assert calls.index(("glim", "umi2")) < ai
    assert ai < calls.index(("dataset", ""))


def test_clock_sync_folds_into_merged(tmp_path: Path):
    leader_cs = manifest.ClockSyncStatus(synced=True, ref="10.0.0.1", offset_to_ref_ns=2_000)
    follower_cs = manifest.ClockSyncStatus(synced=True, ref="10.0.0.1", offset_to_ref_ns=1_500)
    data_dir = _publish_pair(tmp_path, leader_cs=leader_cs, follower_cs=follower_cs)
    res = pair_stage.pair_once(data_dir)
    m = manifest.read_manifest((data_dir / "raw_episodes" / res.merged[0]) / layout.MANIFEST_FILE)
    assert m["clock_sync"]["synced"] is True
    assert m["clock_sync"]["estimated_cross_pi_offset_ns"] == 500  # 2000 - 1500
    assert manifest.clock_sync_acceptable(m)


def test_idempotent_no_op_on_second_run(tmp_path: Path):
    data_dir = _publish_pair(tmp_path)
    raw = data_dir / "raw_episodes"
    assert pair_stage.pair_once(data_dir).merged == ["paired_s1"]
    # Sources are gone and the merged episode has no role -> nothing to re-pair.
    assert pair_stage.pair_once(data_dir).merged == []
    assert list(raw.iterdir()) == [raw / "paired_s1"]
    assert len(list((data_dir / "paired_sources").iterdir())) == 2


def test_crash_recovery_between_merge_and_relocate(tmp_path: Path):
    # A crash after writing the merged dir but before relocating the sources: the next
    # pass must NOT re-merge (deterministic dest exists) but must finish the relocation.
    data_dir = _publish_pair(tmp_path)
    raw = data_dir / "raw_episodes"
    ps = data_dir / "paired_sources"
    assert pair_stage.pair_once(data_dir).merged == ["paired_s1"]
    for src in list(ps.iterdir()):  # simulate: sources never got relocated
        src.rename(raw / src.name)
    res = pair_stage.pair_once(data_dir)
    assert res.merged == ["paired_s1"]  # confirms the existing merge, no duplicate
    assert list(raw.iterdir()) == [raw / "paired_s1"]
    assert len(list(ps.iterdir())) == 2


def test_orphan_dropped_after_timeout(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "raw_episodes").mkdir(parents=True)
    lone = _record_arm(tmp_path, "umi1", session_id="s9", role="umi1", tag_ids=(2, 3))
    lone.rename(data_dir / "raw_episodes" / lone.name)

    # not yet timed out -> pending, stays put.
    res = pair_stage.pair_once(data_dir, timeout_s=1e9)
    assert res.pending == ["s9"] and res.orphaned == []
    assert (data_dir / "raw_episodes" / lone.name).exists()

    # timed out -> dropped into orphans/.
    res = pair_stage.pair_once(data_dir, timeout_s=0.0, now_ns=2**62)
    assert res.orphaned == [lone.name]
    assert not (data_dir / "raw_episodes" / lone.name).exists()
    assert (data_dir / "orphans" / lone.name).exists()


def test_orphan_timeout_uses_server_arrival_not_recording_age(tmp_path: Path):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw_episodes"
    raw.mkdir(parents=True)
    lone = _record_arm(tmp_path, "umi1", session_id="delayed", role="umi1", tag_ids=(2, 3))
    lone.rename(raw / lone.name)
    arrival_ns = 10_000_000_000_000
    (raw / lone.name / pair_stage.INGESTED_FILE).write_text(f"{arrival_ns}\n")

    pending = pair_stage.pair_once(data_dir, timeout_s=600, now_ns=arrival_ns + 599_000_000_000)
    assert pending.pending == ["delayed"]

    expired = pair_stage.pair_once(data_dir, timeout_s=600, now_ns=arrival_ns + 601_000_000_000)
    assert expired.orphaned == [lone.name]


def test_merge_rejects_non_native_arm(tmp_path: Path):
    # A follower mistakenly provisioned as umi1 (role umi2 but config umi named umi1)
    # fails loud with PairError, not a silent collision.
    leader = _record_arm(tmp_path, "umi1", session_id="s1", role="umi1", tag_ids=(2, 3))
    bad = _record_arm(tmp_path, "umi2", session_id="s1", role="umi2", tag_ids=(0, 1))
    snapshot = bad / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    raw = yaml.safe_load(snapshot.read_text())
    raw["umis"] = {"umi1": raw["umis"].pop("umi2")}
    snapshot.write_text(yaml.safe_dump(raw))
    with pytest.raises(pair_stage.PairError):
        pair_stage.merge_pair(leader, bad, tmp_path / "dest", "s1")


def test_bad_pair_is_quarantined_not_crashed(tmp_path: Path):
    # Two independently-provisioned arms with COLLIDING gripper tags: the merged config
    # trips parse_rig_config -> ConfigError (a sibling of PairError, not caught by it).
    # pair_once must quarantine both arms, NOT crash the runner loop or wedge forever.
    data_dir = tmp_path / "data"
    raw = data_dir / "raw_episodes"
    raw.mkdir(parents=True)
    leader = _record_arm(tmp_path, "umi1", session_id="s1", role="umi1", tag_ids=(2, 3))
    follower = _record_arm(tmp_path, "umi2", session_id="s1", role="umi2", tag_ids=(0, 1))
    snapshot = follower / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    follower_raw = yaml.safe_load(snapshot.read_text())
    follower_raw["umis"]["umi2"]["gripper"]["tag_ids"] = [2, 3]
    snapshot.write_text(yaml.safe_dump(follower_raw))
    leader.rename(raw / leader.name)
    follower.rename(raw / follower.name)

    res = pair_stage.pair_once(data_dir)  # must not raise
    assert res.merged == [] and res.failed == ["s1"]
    assert list(raw.iterdir()) == []  # both arms moved out -> no wedge, no re-loop
    assert len(list((data_dir / "failed_pairs").iterdir())) == 2


def test_clock_sync_opposite_sign_offsets(tmp_path: Path):
    # Signed offsets: leader +4ms, follower -4ms => true cross-Pi skew 8ms (> 5ms gate).
    lead = manifest.ClockSyncStatus(synced=True, ref="r", offset_to_ref_ns=4_000_000)
    foll = manifest.ClockSyncStatus(synced=True, ref="r", offset_to_ref_ns=-4_000_000)
    data_dir = _publish_pair(tmp_path, leader_cs=lead, follower_cs=foll)
    res = pair_stage.pair_once(data_dir)
    m = manifest.read_manifest((data_dir / "raw_episodes" / res.merged[0]) / layout.MANIFEST_FILE)
    assert m["clock_sync"]["estimated_cross_pi_offset_ns"] == 8_000_000
    assert not manifest.clock_sync_acceptable(m)


def test_clock_sync_one_arm_missing_is_rejected(tmp_path: Path):
    # Half-instrumented pair (only the leader ran chrony) must not silently pass the gate.
    lead = manifest.ClockSyncStatus(synced=True, ref="r", offset_to_ref_ns=1_000)
    data_dir = _publish_pair(tmp_path, leader_cs=lead, follower_cs=None)
    res = pair_stage.pair_once(data_dir)
    m = manifest.read_manifest((data_dir / "raw_episodes" / res.merged[0]) / layout.MANIFEST_FILE)
    assert m["clock_sync"]["synced"] is False
    assert not manifest.clock_sync_acceptable(m)
