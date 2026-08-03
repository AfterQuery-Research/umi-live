# FAST-LIO2 fallback LI engine (not yet containerized)

GLIM is the primary lidar-inertial engine (`docker/slam-glim/`). If GLIM cannot
converge on the RoboSense Airy (see upstream glim issue #257 — mitigated in our
decode stage via per-point `t` + scan-start stamps), FAST-LIO2 with the native
RoboSense handler in `fastlio_robosense.patch` is the proven-on-hardware fallback:

- patch target: Ericsii/FAST_LIO_ROS2 @ 2fffc570a25d0df172720bac034fbdb6a13d2162
- adds `lidar_type: 5` (ROBOSENSE): per-point absolute `timestamp` -> relative ms
  for de-skew (validated live on an Airy: stable sub-cm stationary pose)
- build notes for the matching rslidar decode are in build_notes_rslidar.sh

To productionize: a `umi-slam-fastlio` stage image mirroring `docker/slam-glim/`
(ROS2 Jazzy + this patch + a TUM export shim reading /Odometry).
