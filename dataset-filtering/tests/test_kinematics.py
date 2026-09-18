from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.kinematics import YAMKinematics


def test_yam_fk_home_pose_is_stable() -> None:
    fk = YAMKinematics().forward(np.zeros(6))
    # Fixture generated from I2RT 7ed46f4's assembled yam + linear_4310 MuJoCo model.
    np.testing.assert_allclose(
        fk.position,
        [0.3305972626871644, 1.7929304707971611e-6, 0.17350262012792556],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fk.rotation,
        [
            [-3.6732246653237155e-6, 3.6731855842123063e-6, 0.9999999999865088],
            [5.307166150059313e-6, -0.999999999979172, 3.6732050785989622e-6],
            [0.999999999979172, 5.307179642495201e-6, 3.6732051710286484e-6],
        ],
        atol=1e-12,
    )
    assert fk.points.shape == (8, 3)


def test_yam_fk_matches_pinned_i2rt_assembled_model_away_from_home() -> None:
    fk = YAMKinematics().forward(np.asarray([0.2, 1.0, 0.8, 0.2, -0.2, 0.3]))
    np.testing.assert_allclose(
        fk.position,
        [0.43476145912554093, 0.13994738212494914, 0.34578119131143775],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fk.rotation,
        [
            [0.11507831166788707, 0.37203207433455654, 0.92105869403095],
            [-0.2721876286813574, -0.879922162835685, 0.3894237822005652],
            [0.9553380956376081, -0.2955150131835957, 2.5712082432072236e-6],
        ],
        atol=1e-12,
    )


def test_home_grasp_axes_are_umi_compatible() -> None:
    rotation = YAMKinematics().forward(np.zeros(6)).rotation
    np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, 1.0], atol=6e-6)  # X up
    np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=6e-6)  # Y right
    np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=6e-6)  # Z forward


def test_ik_round_trip() -> None:
    kin = YAMKinematics()
    target_q = np.asarray([0.2, 1.0, 0.8, 0.2, -0.2, 0.3])
    target = kin.forward(target_q)
    result = kin.solve_ik(
        target.position,
        Rotation.from_matrix(target.rotation).as_quat(),
        initial_q=target_q + 0.05,
    )
    assert result.success
    assert result.position_error_m < 0.002
    assert result.orientation_error_rad < np.deg2rad(1)
    assert np.all(result.q >= kin.lower)
    assert np.all(result.q <= kin.upper)
