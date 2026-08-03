"""Tests for robot-umi-core: layout, config derivation, manifest, geometry, ports."""

from __future__ import annotations

import numpy as np
import pytest

from robot_umi_core import config as cfg
from robot_umi_core import geom, layout, manifest, ports

# --- fixtures ------------------------------------------------------------------------

SINGLE = {
    "version": 1,
    "motion_source": "lidar",
    "task": "test task",
    "umis": {
        "umi1": {
            "camera": {
                "device": "/dev/v4l/by-path/a",
                "width": 800,
                "height": 600,
                "fps": 30,
                "calib": "config/calib/umi1_calib.npz",
            },
            "lidar": {"type": "auto", "iface": "eth0"},
            "gripper": {"tag_ids": [0, 1]},
        }
    },
}

DUAL = {
    "version": 1,
    "motion_source": "lidar+vive",
    "umis": {
        "umi1": {
            "camera": {
                "device": "/dev/v4l/by-path/a",
                "width": 800,
                "height": 600,
                "fps": 30,
                "calib": "config/calib/umi1_calib.npz",
            },
            "lidar": {"iface": "eth0", "host_ip": "192.168.1.102"},
            "vive": {"serial": "LHR-A", "ee_calib": "config/ee_calib.umi1.json"},
            "gripper": {"tag_ids": [0, 1]},
        },
        "umi2": {
            "camera": {
                "device": "/dev/v4l/by-path/b",
                "width": 800,
                "height": 600,
                "fps": 30,
                "calib": "config/calib/umi2_calib.npz",
            },
            "lidar": {"iface": "eth1", "host_ip": "192.168.2.102"},
            "vive": {"serial": "LHR-B", "ee_calib": "config/ee_calib.umi2.json"},
            "gripper": {"tag_ids": [2, 3]},
        },
    },
}


# --- layout --------------------------------------------------------------------------


def test_sensor_key_roundtrip():
    key = layout.sensor_key("umi1", "lidar")
    assert key == "umi1/lidar"
    assert layout.sensor_stem(key) == "umi1_lidar"


def test_sensor_files_per_kind():
    assert layout.sensor_files("umi1/camera") == {
        "video": "umi1_camera.mjpeg.avi",
        "frames": "umi1_camera_frames.jsonl",
        "stats": "umi1_camera_stats.json",
    }
    assert layout.sensor_files("umi2/lidar")["pcap"] == "umi2_lidar.pcap"
    assert layout.sensor_files("umi1/vive") == {"poses": "umi1_vive.jsonl"}


def test_sensor_key_rejects_unknown_kind():
    with pytest.raises(ValueError):
        layout.sensor_key("umi1", "sonar")


def test_episode_id_shape():
    eid = layout.episode_id(now_ns=1_700_000_000_123_456_789, hostname="pi")
    assert eid.startswith("episode_") and "_pi_" in eid


# --- config --------------------------------------------------------------------------


def test_single_umi_schema():
    rig = cfg.parse_rig_config(SINGLE)
    d = cfg.derive(rig)
    assert d.n_umis == 1
    assert d.state_dim == 6
    assert d.state_names == ["umi1_x", "umi1_y", "umi1_z", "umi1_rx", "umi1_ry", "umi1_rz"]
    assert d.video_keys == {"umi1": "observation.images.umi1"}
    assert d.gripper_keys == {"umi1": "observation.gripper_width.umi1"}
    assert d.features["observation.state"]["shape"] == (6,)
    assert d.features["observation.images.umi1"]["shape"] == (600, 800, 3)


def test_dual_umi_schema():
    rig = cfg.parse_rig_config(DUAL)
    d = cfg.derive(rig)
    assert d.n_umis == 2
    assert d.state_dim == 12
    assert d.state_names[6] == "umi2_x"
    assert set(d.gripper_keys) == {"umi1", "umi2"}
    assert d.features["action"]["shape"] == (12,)


def test_capture_plan_enable_mask():
    rig = cfg.parse_rig_config(DUAL)
    lidar_plan = cfg.capture_plan(rig, cfg.MODE_LIDAR)
    keys = [k for k, _ in lidar_plan]
    assert "umi1/lidar" in keys and "umi1/camera" in keys
    assert "umi1/vive" not in keys  # not armed in lidar mode
    both = cfg.capture_plan(rig, cfg.MODE_LIDAR_VIVE)
    assert {"umi1/lidar", "umi1/vive", "umi2/lidar", "umi2/vive"} <= {k for k, _ in both}
    # Cameras are always armed.
    vive_plan = cfg.capture_plan(rig, cfg.MODE_VIVE)
    assert {"umi1/camera", "umi2/camera"} <= {k for k, _ in vive_plan}
    assert not any(kind == "lidar" for _, kind in vive_plan)


def test_mode_cycle():
    assert cfg.cycle_mode(cfg.MODE_LIDAR) == cfg.MODE_VIVE
    assert cfg.cycle_mode(cfg.MODE_LIDAR_VIVE) == cfg.MODE_LIDAR


def test_controls_defaults_when_omitted():
    # SINGLE has no `controls:` -> historical hardcoded pins, LCD on (back-compat).
    rig = cfg.parse_rig_config(SINGLE)
    b, d = rig.controls.buttons, rig.controls.display
    assert (b.record, b.quit, b.mode) == (17, 27, 22)
    assert (d.kind, d.dc, d.rst, d.bl) == ("lcd", 25, 4, 18)


def test_controls_custom_pins():
    import copy

    good = copy.deepcopy(SINGLE)
    good["controls"] = {
        "buttons": {"record": 12, "quit": 16, "mode": 22},
        "display": {"kind": "lcd", "dc": 25, "rst": 27, "bl": 18},
    }
    rig = cfg.parse_rig_config(good)
    assert (rig.controls.buttons.record, rig.controls.buttons.quit) == (12, 16)
    assert rig.controls.display.rst == 27


def test_controls_rejects_pin_collision():
    import copy

    bad = copy.deepcopy(SINGLE)
    bad["controls"] = {"buttons": {"record": 25}}  # collides with default display.dc=25
    with pytest.raises(cfg.ConfigError, match="GPIO pin 25"):
        cfg.parse_rig_config(bad)


def test_controls_pins_free_when_display_none():
    import copy

    # No LCD -> its pins aren't in use, so a button may take a former LCD pin.
    ok = copy.deepcopy(SINGLE)
    ok["controls"] = {"buttons": {"record": 4}, "display": {"kind": "none"}}
    rig = cfg.parse_rig_config(ok)
    assert rig.controls.display.kind == "none"
    assert rig.controls.buttons.record == 4


def test_controls_rejects_bad_display_kind():
    import copy

    bad = copy.deepcopy(SINGLE)
    bad["controls"] = {"display": {"kind": "oled"}}
    with pytest.raises(cfg.ConfigError, match="display.kind"):
        cfg.parse_rig_config(bad)


def test_rejects_unknown_top_level_key():
    import copy

    bad = copy.deepcopy(SINGLE)
    bad["hf_token"] = "secret"  # a stray/secret key must not ride into the snapshot
    with pytest.raises(cfg.ConfigError, match="unknown key.*hf_token"):
        cfg.parse_rig_config(bad)


def test_rejects_unknown_nested_keys():
    import copy

    for where, mutate in [
        ("camera", lambda u: u["umi1"]["camera"].update(exposre=100)),  # typo
        ("lidar", lambda u: u["umi1"].setdefault("lidar", {}).update(prot="udp")),
        ("umi", lambda u: u["umi1"].update(unknwn={})),
    ]:
        bad = copy.deepcopy(SINGLE)
        mutate(bad["umis"])
        with pytest.raises(cfg.ConfigError, match="unknown key"):
            cfg.parse_rig_config(bad)


def test_rejects_non_4x3_camera_with_gripper():
    bad = {
        "umis": {
            "umi1": {
                "camera": {
                    "device": "d",
                    "width": 1920,
                    "height": 1080,
                    "fps": 30,
                    "calib": "c.npz",
                },
                "gripper": {"tag_ids": [0, 1]},
            }
        },
        "motion_source": "vive",
    }
    with pytest.raises(cfg.ConfigError, match="4:3"):
        cfg.parse_rig_config(bad)


def test_rejects_shared_lidar_iface():
    import copy

    bad = copy.deepcopy(DUAL)
    bad["umis"]["umi2"]["lidar"]["iface"] = "eth0"
    with pytest.raises(cfg.ConfigError, match="share one interface"):
        cfg.parse_rig_config(bad)


def test_rejects_overlapping_tag_ids():
    import copy

    bad = copy.deepcopy(DUAL)
    bad["umis"]["umi2"]["gripper"]["tag_ids"] = [1, 3]
    with pytest.raises(cfg.ConfigError, match="tag_ids overlap"):
        cfg.parse_rig_config(bad)


def test_rejects_unrealizable_mode():
    bad = {
        "motion_source": "vive",
        "umis": {
            "umi1": {
                "camera": {
                    "device": "d",
                    "width": 800,
                    "height": 600,
                    "fps": 30,
                    "calib": "c.npz",
                }
            }
        },
    }
    with pytest.raises(cfg.ConfigError, match="no vive configured"):
        cfg.parse_rig_config(bad)


def test_committed_example_configs_load():
    # Guard the shipped examples (incl. the two-Pi pair configs) against schema drift.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    expected = {
        "umi.single.example.yaml": (["umi1"], 6),
        "umi.dual.example.yaml": (["umi1", "umi2"], 12),
        "umi.pair-leader.example.yaml": (["umi1"], 6),
        "umi.pair-follower.example.yaml": (["umi2"], 6),
    }
    for name, (names, state_dim) in expected.items():
        rig = cfg.load_rig_config(root / "config" / name)
        assert [u.name for u in rig.umis] == names
        assert cfg.derive(rig).state_dim == state_dim


def test_referenced_blobs():
    rig = cfg.parse_rig_config(DUAL)
    blobs = cfg.referenced_blobs(rig)
    assert "config/calib/umi1_calib.npz" in blobs
    assert "config/ee_calib.umi2.json" in blobs


def test_lidar_t_lidar_imu_absent_is_none():
    rig = cfg.parse_rig_config(SINGLE)
    assert rig.umis[0].lidar is not None
    assert rig.umis[0].lidar.t_lidar_imu is None


def test_lidar_t_lidar_imu_override_parses():
    import copy

    ext = [0.00425, 0.00418, -0.00446, 0.707999, -0.706198, 0.003362, 0.003284]
    good = copy.deepcopy(SINGLE)
    good["umis"]["umi1"]["lidar"]["t_lidar_imu"] = ext
    rig = cfg.parse_rig_config(good)
    assert rig.umis[0].lidar.t_lidar_imu == pytest.approx(tuple(ext))


def test_lidar_t_lidar_imu_wrong_length_rejected():
    import copy

    bad = copy.deepcopy(SINGLE)
    bad["umis"]["umi1"]["lidar"]["t_lidar_imu"] = [0, 0, 0, 0, 0, 1]  # 6, not 7
    with pytest.raises(cfg.ConfigError, match="7 values"):
        cfg.parse_rig_config(bad)


def test_lidar_t_lidar_imu_degenerate_quat_rejected():
    import copy

    bad = copy.deepcopy(SINGLE)
    bad["umis"]["umi1"]["lidar"]["t_lidar_imu"] = [0, 0, 0, 0, 0, 0, 0]  # zero quat
    with pytest.raises(cfg.ConfigError, match="degenerate"):
        cfg.parse_rig_config(bad)


# --- manifest ------------------------------------------------------------------------


def _mk_manifest(lidar_error=None, ptp_locked=True, residual=100_000):
    sensors = {
        "umi1/camera": manifest.SensorEntry(
            kind="camera", armed=True, files=layout.sensor_files("umi1/camera")
        ),
        "umi1/lidar": manifest.SensorEntry(
            kind="lidar",
            armed=True,
            files=layout.sensor_files("umi1/lidar"),
            error=lidar_error,
        ),
    }
    ptp = {
        "eth0": manifest.PtpStatus(
            locked=ptp_locked,
            timestamping_mode=manifest.PTP_MODE_HARDWARE,
            residual_offset_ns=residual,
        )
    }
    return manifest.build_manifest(
        episode_id="episode_x",
        device_id="pi",
        mode="lidar",
        task="t",
        sensors=sensors,
        ptp=ptp,
        config_sha256="deadbeef",
        start_unix_ns=1,
        stop_unix_ns=2,
        files={},
    )


def test_manifest_eligibility_follows_armed_errors():
    good = _mk_manifest()
    assert good["dataset_eligible"] is True
    bad = _mk_manifest(lidar_error="no packets")
    assert bad["dataset_eligible"] is False


def test_manifest_validates():
    m = _mk_manifest()
    assert manifest.validate_manifest(m) == []
    del m["ptp"]
    assert any("ptp" in p for p in manifest.validate_manifest(m))


def test_present_streams_skips_errored():
    m = _mk_manifest(lidar_error="died")
    assert manifest.present_streams(m, layout.KIND_LIDAR) == []
    assert manifest.present_streams(m, layout.KIND_CAMERA) == ["umi1/camera"]


def test_ptp_rejection_rule():
    assert manifest.lidar_ptp_acceptable(_mk_manifest())
    assert not manifest.lidar_ptp_acceptable(_mk_manifest(ptp_locked=False))
    assert not manifest.lidar_ptp_acceptable(_mk_manifest(residual=50_000_000))
    # No lidar stream -> PTP irrelevant.
    no_lidar = _mk_manifest(lidar_error="died")
    assert manifest.lidar_ptp_acceptable(no_lidar)


def test_coordination_fields_omitted_when_none():
    # Regression guard: a solo manifest must be byte-identical to before these fields
    # existed — the coordination keys are absent, not present-with-null.
    m = _mk_manifest()
    assert "session_id" not in m
    assert "role" not in m
    assert "clock_sync" not in m


def test_coordination_fields_present_when_set():
    sensors = {
        "umi1/camera": manifest.SensorEntry(
            kind="camera", armed=True, files=layout.sensor_files("umi1/camera")
        ),
    }
    m = manifest.build_manifest(
        episode_id="e",
        device_id="pi",
        mode="lidar",
        task="t",
        sensors=sensors,
        ptp={},
        config_sha256="d",
        start_unix_ns=1,
        stop_unix_ns=2,
        files={},
        session_id="sess_123",
        role="umi2",
        clock_sync=manifest.ClockSyncStatus(synced=True, ref="10.0.0.1", offset_to_ref_ns=1_000),
    )
    assert m["session_id"] == "sess_123"
    assert m["role"] == "umi2"
    assert m["clock_sync"]["synced"] is True
    assert m["clock_sync"]["offset_to_ref_ns"] == 1_000
    # Optional fields still validate (not required, extra keys tolerated).
    assert manifest.validate_manifest(m) == []


def test_clock_sync_acceptable():
    # No block -> always acceptable (single-Pi).
    assert manifest.clock_sync_acceptable(_mk_manifest())
    # Unsynced -> rejected.
    assert not manifest.clock_sync_acceptable({"clock_sync": {"synced": False}})
    # Synced, no cross-Pi estimate -> acceptable (a per-Pi block).
    assert manifest.clock_sync_acceptable({"clock_sync": {"synced": True}})
    # Synced, small cross-Pi offset -> acceptable; large -> rejected.
    assert manifest.clock_sync_acceptable(
        {"clock_sync": {"synced": True, "estimated_cross_pi_offset_ns": 1_000_000}}
    )
    assert not manifest.clock_sync_acceptable(
        {"clock_sync": {"synced": True, "estimated_cross_pi_offset_ns": 20_000_000}}
    )


# --- geometry ------------------------------------------------------------------------


def test_quat_roundtrip_matrix():
    q = geom.quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    m = geom.quat_wxyz_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
    q2 = geom.matrix_to_quat_xyzw(m)
    assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)


def test_pose_multiply_inverse_identity():
    p = np.array([1.0, -2.0, 0.5])
    q = geom.quat_normalize(np.array([0.3, -0.1, 0.2, 0.9]))
    pi, qi = geom.pose_inverse(p, q)
    p0, q0 = geom.pose_multiply(p, q, pi, qi)
    assert np.allclose(p0, 0.0, atol=1e-12)
    assert np.allclose(q0, [0, 0, 0, 1], atol=1e-12) or np.allclose(
        -q0, [0, 0, 0, 1], atol=1e-12
    )


def test_parse_trackers_skips_malformed():
    packet = {
        "trackers": [
            {"serial": "A", "valid": True, "pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
            {"serial": "B", "valid": True},  # malformed: no pos/quat
            {"serial": "C", "valid": False, "pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
        ]
    }
    poses = geom.parse_trackers(packet)
    assert set(poses) == {"A"}


def test_pose_zeroer_tare():
    z = geom.PoseZeroer()
    first = z.relative((1, 2, 3), (0, 0, 0, 1))
    assert first[:3] == (1, 2, 3)
    z.request_zero()
    z.relative((1, 2, 3), (0, 0, 0, 1))
    rel = z.relative((2, 2, 3), (0, 0, 0, 1))
    assert rel[:3] == pytest.approx((1, 0, 0))


def test_ee_from_vive_identity_calib_is_noop():
    poses = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=np.float64)
    out = geom.end_effector_poses_from_vive(
        poses, np.zeros(3), np.array([0, 0, 0, 1.0])
    )
    assert np.allclose(out, poses)


def test_load_ee_calib_strict(tmp_path):
    f = tmp_path / "ee.json"
    f.write_text('{"translation_mm": [100, 0, -200], "quat_xyzw": [0, 0, 0, 1]}')
    t, q = geom.load_ee_calib(f)
    assert np.allclose(t, [0.1, 0.0, -0.2])
    with pytest.raises(FileNotFoundError):
        geom.load_ee_calib(tmp_path / "missing.json")


# --- ports ---------------------------------------------------------------------------


def test_classify_port():
    assert ports.classify_port(6699) == "robosense"
    assert ports.classify_port(6688) == "robosense"
    assert ports.classify_port(56301) == "livox"
    assert ports.classify_port(1234) is None


def test_bpf_filter_covers_both_vendors():
    f = ports.bpf_filter("192.168.1.102")
    assert "port 6699" in f and "portrange 56100-56500" in f and "dst host" in f
