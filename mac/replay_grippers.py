#!/usr/bin/env python3
"""Replay the two UMI grippers in 3D over time (Rerun). No IK — each claw is drawn at its
recorded end-effector pose with the fingers opened to the recorded gripper width.

    uv run mac/replay_grippers.py --episode 4     # build + open the viewer (3/4 have the most claw motion)
    uv run mac/replay_grippers.py                 # all episodes
    uv run mac/replay_grippers.py -e 4 --save out.rrd   # write an .rrd only, don't open the viewer

State layout (12-D): dims 0-5 = umi1 (pos xyz + rotvec), 6-11 = umi2. Both arms are already
zeroed into umi1's first-pose frame server-side, so their relative geometry is faithful.
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

ARMS = {"umi1": (0, (70, 150, 255)), "umi2": (6, (255, 150, 50))}  # base dim, RGB color
MAX_WIDTH_M = 0.11  # clamp ArUco outliers to a physically sane gripper stroke


def load(repo, episode):
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    pq = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    df = pd.concat(pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in pq)
    if episode is not None:
        df = df[df.episode_index == episode]
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def claw_geometry(width_m):
    """A parallel-jaw claw in the EE-local frame: +Z approaches, fingers open along ±Y.
    Returns (centers, half_sizes) for a palm bar + two finger prongs a `width` gap apart."""
    w = float(np.clip(width_m, 0.0, MAX_WIDTH_M))
    ft, fl, fd = 0.012, 0.060, 0.022  # finger thickness (Y), length (Z), depth (X)
    y = w / 2 + ft / 2
    centers = np.array([[0, 0, -0.028], [0, y, 0.030], [0, -y, 0.030]])
    half = np.array([[fd / 2, w / 2 + ft + 0.006, 0.010],  # palm spans the full opening
                     [fd / 2, ft / 2, fl / 2],             # left finger
                     [fd / 2, ft / 2, fl / 2]])            # right finger
    return centers, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="brandonyang/umi-bimanual-test")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--save", default=None, help="write an .rrd instead of spawning the viewer")
    a = ap.parse_args()

    df = load(a.repo, a.episode)
    tag = a.episode if a.episode is not None else "all"
    out = a.save or f"/tmp/umi_grippers_ep{tag}.rrd"
    rr.init("umi_grippers")
    rr.save(out)  # rr.spawn's auto-viewer isn't on PATH here; write an .rrd and open it ourselves
    rr.log("world", rr.ViewCoordinates.URF, static=True)  # system frame: Z fwd, X up, Y right

    states = np.stack(df["observation.state"].to_numpy())
    for arm, (o, c) in ARMS.items():  # static full trajectory of each arm
        rr.log(f"world/trail/{arm}", rr.LineStrips3D([states[:, o:o + 3]], colors=[c]), static=True)

    for i, row in df.iterrows():
        rr.set_time("frame", sequence=int(i))
        rr.set_time("time", duration=float(row["timestamp"]))
        st = np.asarray(row["observation.state"], float)
        for arm, (o, c) in ARMS.items():
            R = Rotation.from_rotvec(st[o + 3:o + 6])
            w = float(np.ravel(row[f"observation.gripper_width.{arm}"])[0]) / 1000.0  # mm -> m
            centers, half = claw_geometry(w)
            rr.log(f"world/{arm}", rr.Transform3D(translation=st[o:o + 3],
                                                  rotation=rr.Quaternion(xyzw=R.as_quat())))
            rr.log(f"world/{arm}/claw", rr.Boxes3D(centers=centers, half_sizes=half, colors=c))

    rr.disconnect()  # flush the file sink so the .rrd is complete before we open it
    print(f"logged {len(df)} frames across {df.episode_index.nunique()} episode(s) -> {out}")

    if not a.save:  # open the bundled viewer (lives next to this venv's python)
        viewer = os.path.join(os.path.dirname(sys.executable), "rerun")
        subprocess.Popen([viewer, out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("opening the Rerun viewer…")


if __name__ == "__main__":
    main()
