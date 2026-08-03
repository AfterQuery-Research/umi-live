#!/usr/bin/env python3
"""Export the trajectory from a GLIM dump directory as a TUM file.

Vendored verbatim from docker/slam-glim/export_tum.py so the arm64 Mac image can
reuse it (mac/ is its own Docker build context). Keep the two in sync. See that
file's docstring for the rationale (GLIM already writes TUM; this selects the
globally-optimized lidar-frame trajectory, drops malformed lines, sorts + dedups).

Usage: export_tum.py <glim_dump_dir> <out_trajectory.tum>
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
