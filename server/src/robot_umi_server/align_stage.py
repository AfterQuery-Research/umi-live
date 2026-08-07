"""Cross-lidar map alignment stage (optional; bimanual rigs only).

Each lidar is SLAM'd INDEPENDENTLY by GLIM, so the two ``trajectory.tum`` land in two
unrelated GLIM world frames — subtracting one arm's origin from the other (as the dataset
stage does) only yields metric cross-arm geometry if both already share a frame. This
stage registers the two GLIM maps against the static structure both lidars observe and
rewrites every lidar UMI's trajectory into ONE common frame (the reference UMI's), so the
downstream shared-origin zeroing becomes correct.

It runs ONLY when ``rig.alignment.enabled`` (default False) on a >=2-lidar episode; the
runner never schedules it otherwise, and the dataset stage falls back to the per-lidar
GLIM trajectories (the historical non-joint path).

Inputs (per lidar UMI, from the GLIM stage's map export):
  derived/<id>/<umi>/glim/map.ply            merged map (reference-agnostic geometry)
  derived/<id>/<umi>/glim/submaps/poses.txt  one row: idx stamp x y z qx qy qz qw (optimized)
  derived/<id>/<umi>/glim/submaps/<idx:06>.ply   that submap's points in its LOCAL frame
Output:
  derived/<id>/align/<umi>/trajectory.tum    every lidar UMI's poses, in the common frame

Two tiers (``rig.alignment.tier``):
  1  a single rigid map-to-map registration T(ref<-other); the reference trajectory is
     copied, others are left-multiplied by T.
  2  a joint GTSAM pose graph (per-agent odometry + cross-agent submap GICP factors,
     robust kernel) seeded by the tier-1 transform; corrects each trajectory too.

Heavy deps (open3d, gtsam) are imported lazily so a server without them still runs the
non-joint pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from robot_umi_core import config as core_config
from robot_umi_core import geom, layout, manifest

from robot_umi_server.trajectory import PoseTrack, read_tum, write_tum

# A rigid transform as (translation (3,), quaternion xyzw (4,)).
SE3 = tuple[np.ndarray, np.ndarray]
IDENTITY: SE3 = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


# --- pure geometry (numpy + geom only; no heavy deps, fully unit-testable) -------------


def apply_transform_to_track(track: PoseTrack, t: SE3) -> PoseTrack:
    """Left-multiply the rigid transform ``t`` onto every pose: ``pose' = t . pose``.

    Used to carry a trajectory from its own GLIM frame into the common frame."""
    t_p, t_q = t
    pos = np.empty_like(track.pos)
    quat = np.empty_like(track.quat)
    for i in range(len(track)):
        p, q = geom.pose_multiply(t_p, t_q, track.pos[i], track.quat[i])
        pos[i], quat[i] = p, q
    return PoseTrack(t=track.t.copy(), pos=pos, quat=quat)


def crop_bbox(points: np.ndarray, bbox: tuple[float, ...] | None) -> np.ndarray:
    """Keep points inside ``[xmin,ymin,zmin,xmax,ymax,zmax]`` (excludes the moving arms)."""
    if bbox is None or len(points) == 0:
        return points
    lo = np.asarray(bbox[:3], dtype=float)
    hi = np.asarray(bbox[3:], dtype=float)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


@dataclass(frozen=True)
class Submaps:
    """One agent's submaps: parallel arrays of stamp + local-frame cloud + optimized pose."""

    stamps: np.ndarray  # (M,) unix seconds
    poses: list[SE3]  # M world<-submap transforms (GLIM-optimized, agent's own frame)
    clouds: list[np.ndarray]  # M (Ni,3) points in each submap's LOCAL frame


def read_submaps(glim_dir: Path, load_clouds: bool = True) -> Submaps:
    """Load ``submaps/poses.txt`` (+ per-submap PLYs when ``load_clouds``)."""
    poses_txt = glim_dir / "submaps" / "poses.txt"
    stamps: list[float] = []
    poses: list[SE3] = []
    clouds: list[np.ndarray] = []
    for line in poses_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{poses_txt}: malformed submap row: {line!r}")
        idx = int(float(parts[0]))
        stamps.append(float(parts[1]))
        vals = [float(x) for x in parts[2:]]
        poses.append((np.asarray(vals[:3]), np.asarray(vals[3:7])))
        if load_clouds:
            clouds.append(_read_ply(glim_dir / "submaps" / f"{idx:06d}.ply"))
    if not poses:
        raise ValueError(f"{poses_txt}: no submaps")
    return Submaps(stamps=np.asarray(stamps, dtype=np.float64), poses=poses, clouds=clouds)


def piecewise_correct(
    track: PoseTrack,
    init_common: list[SE3],
    optimized: list[SE3],
    submap_stamps: np.ndarray,
    init_to_common: SE3,
) -> PoseTrack:
    """Deform a per-frame trajectory by its submaps' pose-graph correction.

    Each frame is carried into the common frame by ``init_to_common`` (the tier-1
    transform; identity for the reference), then rigidly updated by its NEAREST submap's
    correction ``optimized[i] . inv(init_common[i])`` — mirroring how GLIM moves frames
    rigidly with their parent submap."""
    corrections = [
        geom.pose_multiply(op, oq, *geom.pose_inverse(ip, iq))
        for (ip, iq), (op, oq) in zip(init_common, optimized, strict=True)
    ]
    pos = np.empty_like(track.pos)
    quat = np.empty_like(track.quat)
    for k in range(len(track)):
        i = int(np.argmin(np.abs(submap_stamps - track.t[k])))
        # frame -> common (tier-1), then apply the submap's correction.
        cp, cq = geom.pose_multiply(
            init_to_common[0], init_to_common[1], track.pos[k], track.quat[k]
        )
        cp, cq = geom.pose_multiply(corrections[i][0], corrections[i][1], cp, cq)
        pos[k], quat[k] = cp, cq
    return PoseTrack(t=track.t.copy(), pos=pos, quat=quat)


# --- point-cloud IO + registration (open3d, lazy-imported) -----------------------------


def _read_ply(path: Path) -> np.ndarray:
    import open3d as o3d  # ty: ignore[unresolved-import]

    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def _o3d_cloud(points: np.ndarray, voxel: float):
    import open3d as o3d  # ty: ignore[unresolved-import]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    return pcd


def register_clouds(source: np.ndarray, target: np.ndarray, voxel: float) -> SE3:
    """Rigid ``T(target<-source)`` as (pos, quat_xyzw): FPFH+RANSAC coarse then GICP fine.

    ``source``/``target`` are (N,3) arrays already cropped to the workspace."""
    import open3d as o3d  # ty: ignore[unresolved-import]

    src, tgt = _o3d_cloud(source, voxel), _o3d_cloud(target, voxel)
    radius = voxel * 5.0
    fpfh_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    src_f = o3d.pipelines.registration.compute_fpfh_feature(src, fpfh_param)
    tgt_f = o3d.pipelines.registration.compute_fpfh_feature(tgt, fpfh_param)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        src_f,
        tgt_f,
        True,
        radius,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(radius)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    fine = o3d.pipelines.registration.registration_generalized_icp(
        src,
        tgt,
        voxel * 2.0,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    )
    return _mat_to_se3(fine.transformation)


def gicp_pair(source: np.ndarray, target: np.ndarray, init: SE3, voxel: float) -> SE3:
    """Fine GICP of two world-frame clouds from an initial guess -> ``T(target<-source)``."""
    import open3d as o3d  # ty: ignore[unresolved-import]

    src, tgt = _o3d_cloud(source, voxel), _o3d_cloud(target, voxel)
    res = o3d.pipelines.registration.registration_generalized_icp(
        src,
        tgt,
        voxel * 2.0,
        _se3_to_mat(init),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    )
    return _mat_to_se3(res.transformation)


def _mat_to_se3(m: np.ndarray) -> SE3:
    m = np.asarray(m)
    return m[:3, 3].copy(), geom.matrix_to_quat_xyzw(m[:3, :3])


def _se3_to_mat(t: SE3) -> np.ndarray:
    p, q = t
    m = np.eye(4)
    m[:3, :3] = geom.quat_wxyz_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
    m[:3, 3] = p
    return m


def transform_points(points: np.ndarray, t: SE3) -> np.ndarray:
    """Apply a rigid transform to (N,3) points."""
    if len(points) == 0:
        return points
    return (_se3_to_mat(t)[:3, :3] @ points.T).T + t[0]


# --- tier 2: joint pose graph (gtsam, lazy-imported) -----------------------------------


def _match_cross_agent(
    ref: Submaps, other: Submaps, offset_s: float, max_pairs: int
) -> list[tuple[int, int]]:
    """Pick up to ``max_pairs`` (ref_i, other_j) submap pairs nearest in time (after the
    cross-Pi clock offset), spread across the episode."""
    pairs: list[tuple[int, int]] = []
    for i, ts in enumerate(ref.stamps):
        j = int(np.argmin(np.abs((other.stamps + offset_s) - ts)))
        pairs.append((i, j))
    # spread evenly across the reference submaps
    if len(pairs) > max_pairs:
        idx = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[k] for k in sorted(set(idx))]
    return pairs


def optimize_pose_graph(
    ref: Submaps,
    other: Submaps,
    t_ref_other: SE3,
    offset_s: float,
    voxel: float,
    max_pairs: int,
    robust_k: float,
) -> tuple[list[SE3], list[SE3]]:
    """Joint optimization of both agents' submap poses in the common (reference) frame.

    Returns (optimized_ref_poses, optimized_other_poses). Seeded by the tier-1 transform;
    other agent's submaps are initialized ``t_ref_other . glim_pose``."""
    import gtsam  # ty: ignore[unresolved-import]

    def key(agent: int, i: int) -> int:
        return gtsam.symbol(chr(ord("a") + agent), i)

    def to_pose3(t: SE3):
        p, q = t
        return gtsam.Pose3(
            gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2]), gtsam.Point3(p[0], p[1], p[2])
        )

    def from_pose3(pose) -> SE3:
        q = pose.rotation().toQuaternion()
        return (
            np.asarray([pose.x(), pose.y(), pose.z()]),
            np.asarray([q.x(), q.y(), q.z(), q.w()]),
        )

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # common-frame initial poses
    ref_init = list(ref.poses)
    other_init = [geom.pose_multiply(*t_ref_other, *p) for p in other.poses]
    for agent, init in ((0, ref_init), (1, other_init)):
        for i, t in enumerate(init):
            values.insert(key(agent, i), to_pose3(t))

    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]))
    # anchor gauge to the reference's first submap (its GLIM pose)
    graph.add(
        gtsam.PriorFactorPose3(
            key(0, 0), to_pose3(ref_init[0]), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))
        )
    )
    # per-agent odometry (relative pose is frame-invariant -> use each agent's own poses)
    for agent, sm in ((0, ref), (1, other)):
        for i in range(len(sm.poses) - 1):
            rel = to_pose3(sm.poses[i]).between(to_pose3(sm.poses[i + 1]))
            graph.add(gtsam.BetweenFactorPose3(key(agent, i), key(agent, i + 1), rel, odom_noise))

    # cross-agent GICP factors (robust)
    base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(robust_k), base
    )
    for i, j in _match_cross_agent(ref, other, offset_s, max_pairs):
        p_ref = transform_points(ref.clouds[i], ref_init[i])
        p_other = transform_points(other.clouds[j], other_init[j])
        try:
            dt = gicp_pair(p_other, p_ref, IDENTITY, voxel)  # corrects other onto ref
        except Exception:
            continue
        other_j_corrected = geom.pose_multiply(*dt, *other_init[j])
        meas = to_pose3(ref_init[i]).between(to_pose3(other_j_corrected))
        graph.add(gtsam.BetweenFactorPose3(key(0, i), key(1, j), meas, robust))

    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    ref_opt = [from_pose3(result.atPose3(key(0, i))) for i in range(len(ref.poses))]
    other_opt = [from_pose3(result.atPose3(key(1, i))) for i in range(len(other.poses))]
    return ref_opt, other_opt


# --- stage entrypoint ------------------------------------------------------------------


def _lidar_umis(m: dict, rig: core_config.RigConfig) -> list[str]:
    present = {k.split("/")[0] for k in manifest.present_streams(m, layout.KIND_LIDAR)}
    return [u.name for u in rig.umis if u.name in present]


def _clock_offset_s(m: dict) -> float:
    """Signed cross-Pi offset (reference minus other) so the other agent's stamps map onto
    the reference axis. Zero for a single-Pi rig (no clock_sync block)."""
    cs = m.get("clock_sync") or {}
    est = cs.get("estimated_cross_pi_offset_ns")
    return float(est) / 1e9 if est is not None else 0.0


def run(episode_dir: Path, out_dir: Path, scope: str = "") -> None:
    del scope
    derived_episode_dir = out_dir.resolve().parent
    m = manifest.read_manifest(episode_dir / layout.MANIFEST_FILE)
    rig = core_config.load_rig_config(
        episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    )
    al = rig.alignment
    lidar_umis = _lidar_umis(m, rig)
    if len(lidar_umis) < 2:
        raise RuntimeError(f"align stage needs >=2 lidar UMIs, got {lidar_umis}")

    ref = al.reference or lidar_umis[0]
    if ref not in lidar_umis:
        raise RuntimeError(f"alignment reference {ref!r} has no lidar in this episode")
    others = [u for u in lidar_umis if u != ref]

    def glim_dir(umi: str) -> Path:
        return derived_episode_dir / umi / "glim"

    ref_map = crop_bbox(_read_ply(glim_dir(ref) / "map.ply"), al.workspace_bbox)

    receipt: dict[str, Any] = {"reference": ref, "tier": al.tier, "transforms": {}}
    # reference trajectory is the common frame by definition -> copy through.
    _write_track(out_dir, ref, read_tum(glim_dir(ref) / "trajectory.tum"))

    offset = _clock_offset_s(m)
    for other in others:
        other_map = crop_bbox(_read_ply(glim_dir(other) / "map.ply"), al.workspace_bbox)
        t_ref_other = register_clouds(other_map, ref_map, al.gicp_voxel)
        receipt["transforms"][other] = {
            "t": list(map(float, t_ref_other[0])),
            "q": list(map(float, t_ref_other[1])),
        }
        other_track = read_tum(glim_dir(other) / "trajectory.tum")

        if al.tier == 1:
            common = apply_transform_to_track(other_track, t_ref_other)
        else:
            ref_sm = read_submaps(glim_dir(ref))
            other_sm = read_submaps(glim_dir(other))
            ref_opt, other_opt = optimize_pose_graph(
                ref_sm,
                other_sm,
                t_ref_other,
                offset,
                al.gicp_voxel,
                al.cross_agent_constraints,
                al.robust_threshold,
            )
            other_init_common = [geom.pose_multiply(*t_ref_other, *p) for p in other_sm.poses]
            common = piecewise_correct(
                other_track,
                other_init_common,
                other_opt,
                other_sm.stamps,
                t_ref_other,
            )
            # tier 2 also refines the reference trajectory in-place (once).
            ref_track = read_tum(glim_dir(ref) / "trajectory.tum")
            ref_corrected = piecewise_correct(
                ref_track,
                list(ref_sm.poses),
                ref_opt,
                ref_sm.stamps,
                IDENTITY,
            )
            _write_track(out_dir, ref, ref_corrected)
        _write_track(out_dir, other, common)

    (out_dir / "receipt.json").write_text(_dumps(receipt))
    print(f"align: {ref} <- {others} (tier {al.tier}) -> {out_dir}", flush=True)


def _write_track(out_dir: Path, umi: str, track: PoseTrack) -> None:
    d = out_dir / umi
    d.mkdir(parents=True, exist_ok=True)
    write_tum(d / "trajectory.tum", track)


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2)
