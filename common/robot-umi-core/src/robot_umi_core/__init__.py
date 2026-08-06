"""robot-umi-core: the single home for truths shared by the device and the server.

If both sides need it, it lives here exactly once:

- :mod:`robot_umi_core.layout`   — episode directory layout + hashing
- :mod:`robot_umi_core.manifest` — the versioned episode manifest (v2)
- :mod:`robot_umi_core.config`   — fixed pair config and dataset-schema derivation
- :mod:`robot_umi_core.geom`     — pure-numpy pose geometry (quaternion and SE(3))
- :mod:`robot_umi_core.ports`    — lidar wire constants (ports, packet sizes, magic)

Dependency budget: numpy + pyyaml only. The Pi imports this package.
"""

__all__ = ["layout", "manifest", "config", "geom", "ports"]
