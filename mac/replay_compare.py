#!/usr/bin/env python3
"""Overlay TWO processings of the same episode in one Rerun window, to eyeball whether they
agree (e.g. GPU-box GLIM vs Mac CPU GLIM, or before/after an alignment fix). Source A draws
bright, source B dim, so any divergence — especially umi2, the arm the align stage places
relative to umi1 — is obvious without flipping between windows.

    uv run mac/replay_compare.py                                   # box:0  vs  test:18 (defaults)
    uv run mac/replay_compare.py --a brandonyang/umi-bimanual-box:0 --b brandonyang/umi-bimanual-test:18

Each --x is repo:episode. Both are zeroed to their own umi1 first-pose server-side, so they
share a frame and a timeline; identical processings overlap exactly.
"""
import argparse
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import rerun as rr
from scipy.spatial.transform import Rotation
from huggingface_hub import HfApi, hf_hub_download

MAX_WIDTH_M = 0.11  # clamp ArUco outliers to a physically sane gripper stroke
# base dim + RGB per arm, per source. A = bright, B = dim (same hue family, desaturated).
SOURCES = {
    "A": {"umi1": (0, (70, 150, 255)), "umi2": (6, (0, 220, 120))},   # blue / green
    "B": {"umi1": (0, (120, 120, 160)), "umi2": (6, (150, 120, 70))},  # dim blue / dim orange
}


def load(repo, episode):
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    pq = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    df = pd.concat(pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in pq)
    df = df[df.episode_index == episode]
    return df.sort_values("frame_index").reset_index(drop=True)


def claw_geometry(width_m):
    """Parallel-jaw claw in the EE-local frame: +Z approaches, fingers open along ±Y."""
    w = float(np.clip(width_m, 0.0, MAX_WIDTH_M))
    ft, fl, fd = 0.012, 0.060, 0.022  # finger thickness (Y), length (Z), depth (X)
    y = w / 2 + ft / 2
    centers = np.array([[0, 0, -0.028], [0, y, 0.030], [0, -y, 0.030]])
    half = np.array([[fd / 2, w / 2 + ft + 0.006, 0.010],
                     [fd / 2, ft / 2, fl / 2],
                     [fd / 2, ft / 2, fl / 2]])
    return centers, half


def parse_src(s):
    repo, ep = s.rsplit(":", 1)
    return repo, int(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="brandonyang/umi-bimanual-box:0", help="bright: repo:episode")
    ap.add_argument("--b", default="brandonyang/umi-bimanual-test:18", help="dim: repo:episode")
    ap.add_argument("--save", default=None, help="write an .rrd instead of spawning the viewer")
    a = ap.parse_args()

    (ra, ea), (rb, eb) = parse_src(a.a), parse_src(a.b)
    frames = {"A": load(ra, ea), "B": load(rb, eb)}
    out = a.save or "/tmp/umi_compare.rrd"
    rr.init("umi_compare")
    rr.save(out)
    rr.log("world", rr.ViewCoordinates.URF, static=True)  # system frame: Z fwd, X up, Y right

    for tag, arms in SOURCES.items():  # static full trajectory of each arm, per source
        states = np.stack(frames[tag]["observation.state"].to_numpy())
        for arm, (o, c) in arms.items():
            rr.log(f"world/{tag}_trail/{arm}", rr.LineStrips3D([states[:, o:o + 3]], colors=[c]),
                   static=True)

    n = min(len(frames["A"]), len(frames["B"]))
    for i in range(n):
        rr.set_time("frame", sequence=i)
        rr.set_time("time", duration=float(frames["A"].iloc[i]["timestamp"]))
        for tag, arms in SOURCES.items():
            row = frames[tag].iloc[i]
            st = np.asarray(row["observation.state"], float)
            for arm, (o, c) in arms.items():
                R = Rotation.from_rotvec(st[o + 3:o + 6])
                w = float(np.ravel(row[f"observation.gripper_width.{arm}"])[0]) / 1000.0  # mm -> m
                centers, half = claw_geometry(w)
                rr.log(f"world/{tag}/{arm}", rr.Transform3D(translation=st[o:o + 3],
                                                            rotation=rr.Quaternion(xyzw=R.as_quat())))
                rr.log(f"world/{tag}/{arm}/claw", rr.Boxes3D(centers=centers, half_sizes=half, colors=c))

    rr.disconnect()
    print(f"A (bright) = {ra}:{ea}   |   B (dim) = {rb}:{eb}   |   {n} frames -> {out}")

    if not a.save:
        viewer = os.path.join(os.path.dirname(sys.executable), "rerun")
        subprocess.Popen([viewer, out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("opening the Rerun viewer…")


if __name__ == "__main__":
    main()
