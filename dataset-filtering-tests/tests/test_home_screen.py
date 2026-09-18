import numpy as np
from dataset_filtering.kinematics import YAMKinematics

from dataset_filtering_tests.home_screen import HOME_CANDIDATES, _home_initializer


def test_balanced_candidates_preserve_zero_home_orientation() -> None:
    kin = YAMKinematics()
    reference = kin.forward(HOME_CANDIDATES["zero"]).rotation
    for name, q in HOME_CANDIDATES.items():
        np.testing.assert_allclose(kin.forward(q).rotation, reference, atol=1e-5, err_msg=name)


def test_home_initializer_changes_only_home() -> None:
    production_init = YAMKinematics.__init__
    kin = object.__new__(YAMKinematics)
    _home_initializer(production_init, HOME_CANDIDATES["balanced_1p2"])(kin)

    np.testing.assert_allclose(kin.home, HOME_CANDIDATES["balanced_1p2"])
    assert len(kin.joints) == 6
