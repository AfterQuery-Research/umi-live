"""Live EE-pose viewer: overlays N UMIs' GLIM end-effector trajectories on ONE graph.

Lidar-only (GLIM; no Vive). Each UMI's live pose is streamed from its own GLIM container
(``glim-<pi-ip>``) via docker-exec -> pose_sub.py, remapped to the UMI convention
(X up, Y right, Z forward), then moved to the end-effector by the lidar->EE lever so a
pure rotation pivots about the gripper tip. Each UMI is zeroed at its OWN first EE pose:
the two lidars run independent SLAM world frames, so this overlays each arm's motion FROM
ITS OWN START for side-by-side comparison -- it is NOT a rigid cross-arm alignment (that
needs a between-arm calibration; use the offline dataset pipeline for the true 12-D merge).

Usage:
  python live_pose_plot.py --pi-ip 10.103.0.166 10.103.3.174
    -> umi1 = first ip (blue), umi2 = second ip (orange), overlaid on shared axes.

TEMPORARY dev/debug view.
"""
import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np
import matplotlib
for _bk in ("macosx", "TkAgg", "QtAgg"):
    try:
        matplotlib.use(_bk); break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

_ap = argparse.ArgumentParser(description="Live GLIM EE-pose viewer, N UMIs overlaid (lidar-only).")
_ap.add_argument("--pi-ip", nargs="+", required=True,
                 help="one or more Pi IPs; labeled umi1, umi2, ... in order. Each selects "
                      "container glim-<ip>. e.g. --pi-ip 10.103.0.166 10.103.3.174")
_ap.add_argument("--container-prefix", default="glim-", help="GLIM container name prefix")
ARGS = _ap.parse_args()

# pose_sub.py streams JSON pose lines out of the GLIM container.
POSE_INNER = ("source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
              "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST && python3 /pose_sub.py")

# --- EE transform: GLIM-world -> UMI convention (X up, Y right, Z fwd) -> end-effector --
M_UMI = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # p_umi = M @ p_glim
# RoboSense lidar bottom-face -> EE (TCP) in the UMI frame: 104.79 mm down, 222.44 fwd.
LIDAR_EE_LEVER_M = np.array([-104.79, 0.0, 222.44]) / 1000.0
FWD_UMI = np.array([0.0, 0.0, 1.0])                            # heading-arrow axis
PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def quat_to_R(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Stream:
    """Dumb accumulator of display-frame EE poses (up, right, fwd cm) + heading. The
    producer loop owns `ref` (first-pose frame); `ref=None` re-captures at next sample."""

    def __init__(self, label):
        self.label = label
        self.lock = threading.Lock()
        self.up, self.right, self.fwd, self.t = [], [], [], []
        self.head = (0.0, 1.0)   # (right, fwd) unit heading of the latest pose
        self.ref = None
        self.msg = "waiting..."
        self.checkpoints = []    # frozen (up, right, fwd) marks (SPACE to add)

    def add(self, up, right, fwd, head, t):
        with self.lock:
            self.up.append(up); self.right.append(right); self.fwd.append(fwd)
            self.t.append(t); self.head = head

    def reset(self):
        with self.lock:
            self.up.clear(); self.right.clear(); self.fwd.clear(); self.t.clear()
            self.ref = None
            self.checkpoints.clear()

    def snapshot(self):
        with self.lock:
            return (list(self.right), list(self.fwd), list(self.up), list(self.t),
                    self.head, self.msg)

    def checkpoint(self):
        with self.lock:
            if self.up:
                self.checkpoints.append((self.up[-1], self.right[-1], self.fwd[-1]))

    def snapshot_checkpoints(self):
        with self.lock:
            return list(self.checkpoints)

    def clear_checkpoints(self):
        with self.lock:
            self.checkpoints.clear()


# --- one GLIM source per UMI ----------------------------------------------------------
SOURCES = []
for i, ip in enumerate(ARGS.pi_ip, start=1):
    SOURCES.append({
        "label": f"umi{i}",
        "ip": ip,
        "container": ARGS.container_prefix + ip.replace(".", "-"),
        "stream": Stream(f"umi{i}"),
        "color": PALETTE[(i - 1) % len(PALETTE)],
        "gen": 0,
        "proc": None,
    })
state = {"busy": False}


def glim_loop(src):
    gen = src["gen"]
    st = src["stream"]
    cmd = ["docker", "exec", "-i", src["container"], "bash", "-lc", POSE_INNER]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except Exception as exc:
        st.msg = f"cannot exec {src['container']}: {exc}"
        return
    src["proc"] = proc
    for line in proc.stdout:
        if src["gen"] != gen:
            break
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "info" in d:
            st.msg = d["info"]; continue
        if "x" not in d:
            continue
        p_glim = np.array([d["x"], d["y"], d["z"]])
        R = quat_to_R(d.get("qx", 0), d.get("qy", 0), d.get("qz", 0), d.get("qw", 1))
        R_umi = M_UMI @ R @ M_UMI.T
        ee = M_UMI @ p_glim + R_umi @ LIDAR_EE_LEVER_M   # (up, right, fwd), gravity-aligned
        with st.lock:
            if st.ref is None:
                st.ref = ee.copy()
            ee_ref = st.ref
        rel = (ee - ee_ref) * 100.0
        h = R_umi @ FWD_UMI
        st.msg = "streaming"
        st.add(float(rel[0]), float(rel[1]), float(rel[2]),
               (float(h[1]), float(h[2])), d["t"])
    if src["gen"] == gen:
        st.msg = "stream ended (press Restart SLAM)"


def restart_slam(_event):
    if state["busy"]:
        return
    state["busy"] = True

    def work():
        for src in SOURCES:
            src["gen"] += 1
            if src["proc"] is not None:
                try:
                    src["proc"].kill()
                except Exception:
                    pass
            src["stream"].reset()
            src["stream"].msg = "restarting GLIM..."
            subprocess.run(["docker", "restart", src["container"]], capture_output=True, timeout=60)
        time.sleep(6)
        for src in SOURCES:
            threading.Thread(target=glim_loop, args=(src,), daemon=True).start()
        state["busy"] = False

    threading.Thread(target=work, daemon=True).start()


for _src in SOURCES:
    threading.Thread(target=glim_loop, args=(_src,), daemon=True).start()

# --- figure: 2 rows (top-down overlay, height overlay), ALL UMIs on shared axes --------
fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 9))
fig.subplots_adjust(bottom=0.15, top=0.90, hspace=0.32)
_title = "UMI live GLIM EE pose — " + "  ".join(f"{s['label']}({s['ip']})" for s in SOURCES)
try:
    fig.canvas.manager.set_window_title("UMI live GLIM EE pose (bimanual)")
except Exception:
    pass
fig.suptitle(_title, fontsize=12, fontweight="bold")

ARROW_FRAC = 0.15
VIEW_MARGIN = 0.25
art = {}
for src in SOURCES:
    c, lab = src["color"], src["label"]
    traj, = top.plot([], [], "-", lw=1.4, color=c, label=lab)
    head, = top.plot([], [], "o", ms=7, color=c)
    arrow = top.annotate("", xytext=(0, 0), xy=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    cp_line, = top.plot([], [], "--", color=c, lw=1.0, alpha=0.5, zorder=5)
    cp_dots, = top.plot([], [], "o", color=c, ms=8, mec="k", zorder=6)
    hline, = bot.plot([], [], color=c, label=lab)
    art[lab] = {"traj": traj, "head": head, "arrow": arrow, "hline": hline,
                "cp_line": cp_line, "cp_dots": cp_dots, "cp_texts": [], "cp_n": -1}
top.set_xlabel("right (cm)"); top.set_ylabel("forward (cm)"); top.grid(True)
top.set_aspect("equal", "datalim"); top.margins(VIEW_MARGIN)
top.legend(loc="upper right", fontsize=9)
bot.set_xlabel("t (s)"); bot.set_ylabel("up (cm)"); bot.grid(True)
bot.legend(loc="upper right", fontsize=9)


def update(_):
    titles = []
    for src in SOURCES:
        st, a, c = src["stream"], art[src["label"]], src["color"]
        cps = st.snapshot_checkpoints()
        a["cp_dots"].set_data([p[1] for p in cps], [p[2] for p in cps])
        a["cp_line"].set_data([p[1] for p in cps], [p[2] for p in cps])
        if len(cps) != a["cp_n"]:
            for tx in a["cp_texts"]:
                tx.remove()
            a["cp_texts"] = []
            for u, v in zip(cps, cps[1:]):
                dist = ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2 + (u[2] - v[2]) ** 2) ** 0.5
                tx = top.text(0.5 * (u[1] + v[1]), 0.5 * (u[2] + v[2]), f"{dist:.1f}",
                              color=c, fontsize=8, fontweight="bold", zorder=7,
                              ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, lw=0.5, alpha=0.85))
                a["cp_texts"].append(tx)
            a["cp_n"] = len(cps)
        R, F, U, T, _head, msg = st.snapshot()
        if R:
            a["traj"].set_data(R, F); a["head"].set_data([R[-1]], [F[-1]])
            a["hline"].set_data([t - T[0] for t in T], U)
            titles.append(f"{src['label']}: {msg} {len(R)}pts "
                          f"up={U[-1]:+.1f} r={R[-1]:+.1f} f={F[-1]:+.1f}")
        else:
            titles.append(f"{src['label']}: {msg}")
    # autoscale over ALL streams (all lines on the shared axes), then lay heading arrows
    top.relim(); top.autoscale_view(); bot.relim(); bot.autoscale_view()
    x0, x1 = top.get_xlim(); y0, y1 = top.get_ylim(); span = max(x1 - x0, y1 - y0)
    for src in SOURCES:
        R, F, U, T, (hr, hf), _msg = src["stream"].snapshot()
        if R:
            nrm = (hr * hr + hf * hf) ** 0.5 or 1.0
            alen = ARROW_FRAC * span
            src_arrow = art[src["label"]]["arrow"]
            src_arrow.set_position((R[-1], F[-1]))
            src_arrow.xy = (R[-1] + alen * hr / nrm, F[-1] + alen * hf / nrm)
    top.set_title("   |   ".join(titles), fontsize=9)
    return []


# --- controls -------------------------------------------------------------------------
btn_ax = fig.add_axes([0.30, 0.03, 0.22, 0.06])
btn = Button(btn_ax, "Restart SLAM (all)", color="#e8a33d", hovercolor="#f0b95e")
btn.on_clicked(restart_slam)

CKPT_FILE = "/tmp/live/checkpoints_bimanual.json"


def save_checkpoints():
    try:
        os.makedirs("/tmp/live", exist_ok=True)
        with open(CKPT_FILE, "w") as f:
            json.dump({s["label"]: s["stream"].snapshot_checkpoints() for s in SOURCES}, f)
    except OSError:
        pass


btn3_ax = fig.add_axes([0.57, 0.03, 0.15, 0.06])
btn_clear = Button(btn3_ax, "Clear pts", color="#c0504d", hovercolor="#d47a77")
btn_clear.on_clicked(lambda _e: ([s["stream"].clear_checkpoints() for s in SOURCES], save_checkpoints()))

fig.text(0.02, 0.06, f"SPACE = drop a checkpoint on each arm (saved to {CKPT_FILE})",
         fontsize=9, color="#333", fontweight="bold")


def on_key(event):
    if event.key in (" ", "space"):
        for s in SOURCES:
            s["stream"].checkpoint()
        save_checkpoints()


fig.canvas.mpl_connect("key_press_event", on_key)

ani = animation.FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
plt.show()
