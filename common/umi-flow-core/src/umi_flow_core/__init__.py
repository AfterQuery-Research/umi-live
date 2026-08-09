"""umi-flow-core: the single home for truths shared by the device and the server.

If both sides need it, it lives here exactly once:

- :mod:`umi_flow_core.layout`   — episode directory layout + hashing
- :mod:`umi_flow_core.manifest` — the versioned episode manifest (v2)
- :mod:`umi_flow_core.config`   — fixed pair config and dataset-schema derivation
- :mod:`umi_flow_core.geom`     — pure-numpy pose geometry (quaternion and SE(3))
- :mod:`umi_flow_core.ports`    — lidar wire constants (ports, packet sizes, magic)

Dependency budget: numpy + pyyaml only. The Pi imports this package.
"""

__all__ = ["layout", "manifest", "config", "geom", "ports"]
