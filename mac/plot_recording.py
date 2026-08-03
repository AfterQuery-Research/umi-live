"""Offline pose-plot video: render a PROCESSED episode's UMI EE trajectories as an MP4,
using the same two graphs as the live viewer (mac/live_pose_plot.py) — a top-down
right/forward overlay and an up-over-time overlay — but from recorded trajectories instead
of a live GLIM stream.

Data source: the runner's derived episode dir, i.e. ``data/derived/<id>/`` containing
``<umi>/glim/trajectory.tum`` and, when cross-lidar alignment ran, ``align/<umi>/trajectory.tum``.

Joint vs independent is picked automatically:
  * if ``align/<umi>/trajectory.tum`` exists for every UMI, those are already in ONE common
    frame -> both arms are zeroed to the REFERENCE arm's first pose (true cross-arm geometry);
  * otherwise each UMI's own ``glim`` trajectory is used and zeroed to ITS OWN first pose
    (independent SLAM frames — same caveat as the live viewer).
Force it with ``--joint on|off|auto`` (default auto).

Usage:
  python mac/plot_recording.py --episode data/derived/<id> [--out pose.mp4] [--fps 30]

The EE transform (GLIM-world -> UMI convention -> gripper tip) is kept IN SYNC with
mac/live_pose_plot.py so the video matches the live view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# --- EE transform (keep in sync with mac/live_pose_plot.py) ----------------------------
M_UMI = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # p_umi(up,right,fwd) = M @ p_glim
LIDAR_EE_LEVER_M = np.array([-104.79, 0.0, 222.44]) / 1000.0  # lidar bottom-face -> EE (TCP)
FWD_UMI = np.array([0.0, 0.0, 1.0])  # heading-arrow axis in the UMI frame
PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

RESERVED_DIRS = {"align", "dataset", "gripper"}


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ee_from_pose(p_glim: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One GLIM-world pose -> (ee, heading), both in UMI convention (up, right, fwd).

    ee = gripper tip = lidar position remapped + the lidar->EE lever rotated into place, so a
    pure rotation pivots about the tip. Matches live_pose_plot's glim_loop math exactly."""
    R = quat_to_R(*quat_xyzw)
    R_umi = M_UMI @ R @ M_UMI.T
    ee = M_UMI @ p_glim + R_umi @ LIDAR_EE_LEVER_M
    return ee, R_umi @ FWD_UMI


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a TUM file -> (t (N,), pos (N,3), quat_xyzw (N,4)), sorted by time."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"{path}: malformed TUM line: {line!r}")
        rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: empty trajectory")
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def ee_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A TUM trajectory -> (t (N,), ee (N,3) up/right/fwd, heading (N,3))."""
    t, pos, quat = read_tum(path)
    ee = np.empty((len(t), 3))
    head = np.empty((len(t), 3))
    for i in range(len(t)):
        ee[i], head[i] = ee_from_pose(pos[i], quat[i])
    return t, ee, head


# --- episode discovery -----------------------------------------------------------------


def discover_sources(episode_dir: Path, joint: str) -> tuple[list[tuple[str, Path]], bool]:
    """(sources, use_joint) where sources is [(umi, trajectory.tum), ...].

    ``joint`` is one of auto|on|off. auto uses the align outputs iff they exist for every UMI."""
    umis = sorted(
        d.name
        for d in episode_dir.iterdir()
        if d.is_dir() and d.name not in RESERVED_DIRS and (d / "glim" / "trajectory.tum").is_file()
    )
    if not umis:
        raise SystemExit(f"no <umi>/glim/trajectory.tum under {episode_dir}")
    align_present = len(umis) >= 2 and all(
        (episode_dir / "align" / u / "trajectory.tum").is_file() for u in umis
    )
    if joint == "on" and not align_present:
        raise SystemExit(
            f"--joint on but no align/<umi>/trajectory.tum for all of {umis} in {episode_dir} "
            "(run the pipeline with alignment enabled first)"
        )
    use_joint = joint == "on" or (joint == "auto" and align_present)
    srcs = [
        (u, episode_dir / ("align" if use_joint else u) / (u if use_joint else "glim") / "trajectory.tum")
        for u in umis
    ]
    return srcs, use_joint


# --- rendering -------------------------------------------------------------------------


def render(episode_dir: Path, out: Path, joint: str, fps: int, max_seconds: float) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: we save frames, never show a window
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    sources, use_joint = discover_sources(episode_dir, joint)
    tracks = [(umi, *ee_track(path)) for umi, path in sources]  # (umi, t, ee, head)

    # Shared origin (cm): joint -> the reference (first) arm's first EE; else each arm's own.
    ref0 = tracks[0][2][0].copy()
    # Per-arm ELAPSED time (each arm starts at 0). The two lidars stamp poses in their own
    # PTP clocks, which can differ by hours, so absolute timestamps don't line up; a paired
    # recording is simultaneous, so elapsed-from-own-start aligns the arms to the same real
    # motion (and matches the live viewer's per-arm-from-start behavior).
    arms = []
    span_max = 0.0
    for i, (umi, t, ee, head) in enumerate(tracks):
        origin = ref0 if use_joint else ee[0]
        rel = (ee - origin) * 100.0  # cm; columns = up, right, fwd
        tau = t - t[0]
        arms.append({
            "umi": umi, "tau": tau, "up": rel[:, 0], "right": rel[:, 1], "fwd": rel[:, 2],
            "head": head, "color": PALETTE[i % len(PALETTE)],
        })
        span_max = max(span_max, float(tau[-1]))

    n_frames = max(2, min(int(span_max * fps), int(max_seconds * fps)))
    grid = np.linspace(0.0, span_max, n_frames)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 9))
    fig.subplots_adjust(bottom=0.08, top=0.90, hspace=0.32)
    mode = "JOINT (common frame)" if use_joint else "independent (own start)"
    fig.suptitle(f"UMI EE pose — {episode_dir.name} — {mode}", fontsize=12, fontweight="bold")

    art = {}
    for a in arms:
        c, lab = a["color"], a["umi"]
        traj, = top.plot([], [], "-", lw=1.4, color=c, label=lab)
        headd, = top.plot([], [], "o", ms=7, color=c)
        arrow = top.annotate("", xytext=(0, 0), xy=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        hline, = bot.plot([], [], color=c, label=lab)
        art[lab] = {"traj": traj, "head": headd, "arrow": arrow, "hline": hline}
    top.set_xlabel("right (cm)")
    top.set_ylabel("forward (cm)")
    top.grid(True)
    top.set_aspect("equal", "datalim")
    bot.set_xlabel("t (s)")
    bot.set_ylabel("up (cm)")
    bot.grid(True)

    # fixed axis limits (whole trajectory) so the video doesn't jitter as it grows.
    all_r = np.concatenate([a["right"] for a in arms])
    all_f = np.concatenate([a["fwd"] for a in arms])
    all_u = np.concatenate([a["up"] for a in arms])
    _lim(top, all_r, all_f, equal=True)
    bot.set_xlim(0, max(1e-3, span_max))
    bot.set_ylim(*_pad(all_u))
    span = max(all_r.max() - all_r.min(), all_f.max() - all_f.min(), 1.0)
    top.legend(loc="upper right", fontsize=9)
    bot.legend(loc="upper right", fontsize=9)

    def update(k):
        tg = grid[k]
        titles = []
        for a in arms:
            m = a["tau"] <= tg
            g = art[a["umi"]]
            n = int(m.sum())
            if n:
                r, f, u = a["right"][:n], a["fwd"][:n], a["up"][:n]
                g["traj"].set_data(r, f)
                g["head"].set_data([r[-1]], [f[-1]])
                g["hline"].set_data(a["tau"][:n], u)
                hr, hf = a["head"][n - 1][1], a["head"][n - 1][2]
                nrm = (hr * hr + hf * hf) ** 0.5 or 1.0
                alen = 0.15 * span
                g["arrow"].set_position((r[-1], f[-1]))
                g["arrow"].xy = (r[-1] + alen * hr / nrm, f[-1] + alen * hf / nrm)
                titles.append(f"{a['umi']}: {n}pts up={u[-1]:+.1f} r={r[-1]:+.1f} f={f[-1]:+.1f}")
            else:
                titles.append(f"{a['umi']}: -")
        top.set_title(f"t={tg:5.1f}s    " + "   |   ".join(titles), fontsize=9)
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    _save(ani, out, fps)
    plt.close(fig)
    print(f"wrote {out}  ({n_frames} frames, {len(arms)} UMIs, {mode})")


def _pad(v: np.ndarray, frac: float = 0.1) -> tuple[float, float]:
    lo, hi = float(v.min()), float(v.max())
    pad = max((hi - lo) * frac, 1.0)
    return lo - pad, hi + pad


def _lim(ax, x: np.ndarray, y: np.ndarray, equal: bool) -> None:
    x0, x1 = _pad(x)
    y0, y1 = _pad(y)
    if equal:  # keep aspect square around the data's center
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half = 0.5 * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _save(ani, out: Path, fps: int) -> None:
    import matplotlib.animation as animation

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        ani.save(str(out), writer=animation.PillowWriter(fps=fps))
        return
    if animation.FFMpegWriter.isAvailable():
        ani.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        return
    alt = out.with_suffix(".gif")
    print(f"ffmpeg not found; writing {alt} instead (install ffmpeg for MP4)")
    ani.save(str(alt), writer=animation.PillowWriter(fps=fps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a processed episode's UMI EE trajectories to video.")
    ap.add_argument("--episode", type=Path, required=True,
                    help="derived episode dir, e.g. data/derived/<id> (holds <umi>/glim/trajectory.tum)")
    ap.add_argument("--out", type=Path, default=None, help="output video (default: <episode>/pose_plot.mp4)")
    ap.add_argument("--joint", choices=("auto", "on", "off"), default="auto",
                    help="auto: use align/ common-frame output if present; on: require it; off: per-arm glim")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-seconds", type=float, default=120.0, help="cap video length (frames = fps*this)")
    args = ap.parse_args()
    out = args.out or (args.episode / "pose_plot.mp4")
    render(args.episode.resolve(), out, args.joint, args.fps, args.max_seconds)


if __name__ == "__main__":
    main()
