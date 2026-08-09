"""Tests for umi-flow-core: layout, config derivation, manifest, geometry, ports."""

from __future__ import annotations

import numpy as np
import pytest
from umi_flow_core import config as cfg
from umi_flow_core import geom, layout, manifest, ports

# --- fixtures ------------------------------------------------------------------------

CLIENT = {
    "version": 1,
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
            "lidar": {"iface": "eth0"},
            "gripper": {"tag_ids": [2, 3]},
        }
    },
}

DUAL = {
    "version": 1,
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
            "gripper": {"tag_ids": [2, 3]},
        },
        "umi2": {
            "camera": {
                "device": "/dev/v4l/by-path/b",
                "width": 800,
                "height": 600,
                "fps": 30,
                "calib": "config/calib/umi2_calib.npz",
            },
            "lidar": {"iface": "eth0", "host_ip": "192.168.1.102"},
            "gripper": {"tag_ids": [0, 1]},
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


def test_sensor_key_rejects_unknown_kind():
    with pytest.raises(ValueError):
        layout.sensor_key("umi1", "sonar")


def test_episode_id_shape():
    eid = layout.episode_id(now_ns=1_700_000_000_123_456_789, hostname="pi")
    assert eid.startswith("episode_") and "_pi_" in eid


# --- config --------------------------------------------------------------------------


def test_client_half_schema():
    rig = cfg.parse_rig_config(CLIENT)
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


def test_fixed_capture_plan():
    rig = cfg.parse_rig_config(DUAL)
    assert cfg.capture_plan(rig) == [
        ("umi1/lidar", "lidar"),
        ("umi1/camera", "camera"),
        ("umi2/lidar", "lidar"),
        ("umi2/camera", "camera"),
    ]


def test_controls_defaults_when_omitted():
    rig = cfg.parse_rig_config(CLIENT)
    b, d = rig.controls.buttons, rig.controls.display
    assert (b.record, b.quit) == (12, 16)
    assert (d.kind, d.dc, d.rst, d.bl) == ("lcd", 25, 27, 18)


def test_controls_custom_pins():
    import copy

    good = copy.deepcopy(CLIENT)
    good["controls"] = {
        "buttons": {"record": 12, "quit": 16},
        "display": {"kind": "lcd", "dc": 25, "rst": 27, "bl": 18},
    }
    rig = cfg.parse_rig_config(good)
    assert (rig.controls.buttons.record, rig.controls.buttons.quit) == (12, 16)
    assert rig.controls.display.rst == 27


def test_controls_rejects_pin_collision():
    import copy

    bad = copy.deepcopy(CLIENT)
    bad["controls"] = {"buttons": {"record": 25}}  # collides with default display.dc=25
    with pytest.raises(cfg.ConfigError, match="GPIO pin 25"):
        cfg.parse_rig_config(bad)


def test_controls_pins_free_when_display_none():
    import copy

    # No LCD -> its pins aren't in use, so a button may take a former LCD pin.
    ok = copy.deepcopy(CLIENT)
    ok["controls"] = {"buttons": {"record": 4}, "display": {"kind": "none"}}
    rig = cfg.parse_rig_config(ok)
    assert rig.controls.display.kind == "none"
    assert rig.controls.buttons.record == 4


def test_controls_rejects_bad_display_kind():
    import copy

    bad = copy.deepcopy(CLIENT)
    bad["controls"] = {"display": {"kind": "oled"}}
    with pytest.raises(cfg.ConfigError, match="display.kind"):
        cfg.parse_rig_config(bad)


def test_rejects_unknown_top_level_key():
    import copy

    bad = copy.deepcopy(CLIENT)
    bad["hf_token"] = "secret"  # a stray/secret key must not ride into the snapshot
    with pytest.raises(cfg.ConfigError, match="unknown key.*hf_token"):
        cfg.parse_rig_config(bad)


def test_rejects_unknown_nested_keys():
    import copy

    for _where, mutate in [
        ("camera", lambda u: u["umi1"]["camera"].update(exposre=100)),  # typo
        ("lidar", lambda u: u["umi1"].setdefault("lidar", {}).update(prot="udp")),
        ("umi", lambda u: u["umi1"].update(unknwn={})),
    ]:
        bad = copy.deepcopy(CLIENT)
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
                "lidar": {},
                "gripper": {"tag_ids": [2, 3]},
            }
        },
    }
    with pytest.raises(cfg.ConfigError, match="4:3"):
        cfg.parse_rig_config(bad)


def test_rejects_overlapping_tag_ids():
    import copy

    bad = copy.deepcopy(DUAL)
    bad["umis"]["umi2"]["gripper"]["tag_ids"] = [1, 3]
    with pytest.raises(cfg.ConfigError, match="tag_ids overlap"):
        cfg.parse_rig_config(bad)


@pytest.mark.parametrize("tag_ids", [[-1, 2], [2, 50], [2, 2], [2, 3.0], [True, 3]])
def test_rejects_invalid_gripper_tag_pair(tag_ids):
    import copy

    bad = copy.deepcopy(CLIENT)
    bad["umis"]["umi1"]["gripper"]["tag_ids"] = tag_ids
    with pytest.raises(cfg.ConfigError, match="tag_ids"):
        cfg.parse_rig_config(bad)


def test_loads_physical_gripper_tag_config(tmp_path):
    path = tmp_path / "gripper_tags.yaml"
    path.write_text(
        "version: 1\n"
        "dictionary: DICT_4X4_50\n"
        "umis:\n"
        "  umi1: {tag_ids: [10, 11]}\n"
        "  umi2: {tag_ids: [12, 13]}\n"
    )
    assert cfg.load_gripper_tag_pairs(path) == {"umi1": (10, 11), "umi2": (12, 13)}


def test_physical_gripper_tag_config_rejects_overlap(tmp_path):
    path = tmp_path / "gripper_tags.yaml"
    path.write_text(
        "version: 1\n"
        "dictionary: DICT_4X4_50\n"
        "umis:\n"
        "  umi1: {tag_ids: [10, 11]}\n"
        "  umi2: {tag_ids: [11, 12]}\n"
    )
    with pytest.raises(cfg.ConfigError, match="overlap"):
        cfg.load_gripper_tag_pairs(path)


def test_referenced_blobs():
    rig = cfg.parse_rig_config(DUAL)
    blobs = cfg.referenced_blobs(rig)
    assert "config/calib/umi1_calib.npz" in blobs
    assert "config/calib/umi2_calib.npz" in blobs


def test_lidar_t_lidar_imu_absent_is_none():
    rig = cfg.parse_rig_config(CLIENT)
    assert rig.umis[0].lidar is not None
    assert rig.umis[0].lidar.t_lidar_imu is None


def test_lidar_t_lidar_imu_override_parses():
    import copy

    ext = [0.00425, 0.00418, -0.00446, 0.707999, -0.706198, 0.003362, 0.003284]
    good = copy.deepcopy(CLIENT)
    good["umis"]["umi1"]["lidar"]["t_lidar_imu"] = ext
    rig = cfg.parse_rig_config(good)
    assert rig.umis[0].lidar.t_lidar_imu == pytest.approx(tuple(ext))


def test_lidar_t_lidar_imu_wrong_length_rejected():
    import copy

    bad = copy.deepcopy(CLIENT)
    bad["umis"]["umi1"]["lidar"]["t_lidar_imu"] = [0, 0, 0, 0, 0, 1]  # 6, not 7
    with pytest.raises(cfg.ConfigError, match="7 values"):
        cfg.parse_rig_config(bad)


def test_lidar_t_lidar_imu_degenerate_quat_rejected():
    import copy

    bad = copy.deepcopy(CLIENT)
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
    # Transport manifests omit optional coordination fields when callers pass None.
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
    # Missing pair synchronization evidence is rejected.
    assert not manifest.clock_sync_acceptable(_mk_manifest())
    # Unsynced -> rejected.
    assert not manifest.clock_sync_acceptable({"clock_sync": {"synced": False}})
    # A merged estimate is required.
    assert not manifest.clock_sync_acceptable({"clock_sync": {"synced": True}})
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
    assert np.allclose(q0, [0, 0, 0, 1], atol=1e-12) or np.allclose(-q0, [0, 0, 0, 1], atol=1e-12)


# --- ports ---------------------------------------------------------------------------


def test_classify_port():
    assert ports.classify_port(6699) == "robosense"
    assert ports.classify_port(6688) == "robosense"
    assert ports.classify_port(56301) is None
    assert ports.classify_port(1234) is None


def test_bpf_filter_covers_robosense_ports():
    f = ports.bpf_filter("192.168.1.102")
    assert "port 6699" in f and "port 7788" in f and "port 6688" in f
    assert "portrange" not in f and "dst host" in f
