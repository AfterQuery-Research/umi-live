from __future__ import annotations

import pytest
from umi_flow_core import config


def _umi(name: str, tags: list[int], calib: str) -> dict:
    return {
        "camera": {
            "device": f"/dev/{name}",
            "width": 800,
            "height": 600,
            "fps": 30,
            "calib": calib,
        },
        "lidar": {"iface": "eth0"},
        "gripper": {"tag_ids": tags},
    }


def test_client_role_is_derived_from_the_single_umi_name() -> None:
    leader = config.parse_rig_config(
        {"version": 1, "umis": {"umi1": _umi("umi1", [10, 11], "leader.npz")}}
    )
    follower = config.parse_rig_config(
        {"version": 1, "umis": {"umi2": _umi("umi2", [12, 13], "follower.npz")}}
    )

    assert config.client_role(leader) == "leader"
    assert config.client_role(follower) == "follower"


def test_client_role_rejects_solo_or_multi_umi_configs() -> None:
    unsupported = config.parse_rig_config(
        {"version": 1, "umis": {"arm": _umi("arm", [0, 1], "arm.npz")}}
    )
    with pytest.raises(config.ConfigError, match="umi1.*umi2"):
        config.client_role(unsupported)

    two_on_one_pi = config.parse_rig_config(
        {
            "version": 1,
            "umis": {
                "umi1": _umi("umi1", [10, 11], "leader.npz"),
                "umi2": _umi("umi2", [12, 13], "follower.npz"),
            },
        }
    )
    with pytest.raises(config.ConfigError, match="exactly one"):
        config.client_role(two_on_one_pi)


def test_bimanual_contract_requires_umi1_umi2_and_12_dimensions() -> None:
    rig = config.parse_rig_config(
        {
            "version": 1,
            "umis": {
                "umi1": _umi("umi1", [2, 3], "leader.npz"),
                "umi2": _umi("umi2", [0, 1], "follower.npz"),
            },
        }
    )

    derived = config.require_bimanual(rig)

    assert derived.state_dim == 12
    assert tuple(derived.video_keys) == ("umi1", "umi2")


def test_bimanual_contract_rejects_a_single_arm() -> None:
    rig = config.parse_rig_config(
        {"version": 1, "umis": {"umi1": _umi("umi1", [2, 3], "leader.npz")}}
    )
    with pytest.raises(config.ConfigError, match="paired umi1/umi2"):
        config.require_bimanual(rig)


def test_removed_sensor_and_mode_fields_are_rejected() -> None:
    with pytest.raises(config.ConfigError, match="motion_source"):
        config.parse_rig_config(
            {
                "version": 1,
                "motion_source": "vive",
                "umis": {"umi1": _umi("umi1", [2, 3], "leader.npz")},
            }
        )

    vive = _umi("umi1", [2, 3], "leader.npz")
    vive["vive"] = {"serial": "tracker", "ee_calib": "ee.json"}
    with pytest.raises(config.ConfigError, match="vive"):
        config.parse_rig_config({"version": 1, "umis": {"umi1": vive}})

    typed_lidar = _umi("umi1", [2, 3], "leader.npz")
    typed_lidar["lidar"]["type"] = "livox"
    with pytest.raises(config.ConfigError, match="type"):
        config.parse_rig_config({"version": 1, "umis": {"umi1": typed_lidar}})
