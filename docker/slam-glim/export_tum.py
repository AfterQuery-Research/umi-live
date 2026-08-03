#!/usr/bin/env python3
"""Export the trajectory from a GLIM dump directory as a TUM file.

Usage: export_tum.py <glim_dump_dir> <out_trajectory.tum>

GLIM's ``save(dump_path)`` (glim/src/glim/mapping/global_mapping.cpp,
verified 2026-07-15) writes trajectories via ``write_tum_frame`` — i.e. the
files are ALREADY in TUM format::

    stamp x y z qx qy qz qw

so "conversion" here is selection + validation:

* prefer ``traj_lidar.txt`` (globally optimized, LiDAR frame — matches what
  we compare against the Vive EE trajectory) over ``odom_lidar.txt``
  (odometry-only, drifts), then the IMU-frame variants;
* drop malformed lines, sort by timestamp, de-duplicate stamps
  (global mapping can emit a frame twice around submap boundaries).

Runs on the stock python3 of the GLIM image (no third-party deps).
Needs on-GPU validation only insofar as the dump itself does.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Preference order: optimized before odometry-only, lidar frame before imu.
CANDIDATES = ["traj_lidar.txt", "odom_lidar.txt", "traj_imu.txt", "odom_imu.txt"]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 64
    dump_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    src = next(
        (
            dump_dir / name
            for name in CANDIDATES
            if (dump_dir / name).is_file() and (dump_dir / name).stat().st_size > 0
        ),
        None,
    )
    if src is None:
        print(
            f"error: no trajectory file ({', '.join(CANDIDATES)}) in {dump_dir}",
            file=sys.stderr,
        )
        return 66

    rows: dict[float, tuple[float, ...]] = {}
    bad = 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            bad += 1
            continue
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError:
            bad += 1
            continue
        rows[vals[0]] = vals  # keyed by stamp -> de-duplicates

    if not rows:
        print(f"error: {src} contained no valid TUM rows", file=sys.stderr)
        return 65
    if bad:
        print(f"warning: skipped {bad} malformed line(s) in {src}", file=sys.stderr)

    with out_path.open("w") as f:
        f.write(f"# TUM trajectory exported from GLIM dump ({src.name})\n")
        f.write("# stamp x y z qx qy qz qw\n")
        for stamp in sorted(rows):
            f.write(" ".join(f"{v:.9f}" for v in rows[stamp]) + "\n")

    print(f"export_tum: {src.name} -> {out_path} ({len(rows)} poses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
