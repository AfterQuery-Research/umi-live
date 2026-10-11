# UMI dataset filtering and continuous YAM retargeting

Local, read-only tooling for [`brandonyang/dual-lidar-umi`](https://huggingface.co/datasets/brandonyang/dual-lidar-umi). It measures and smooths LiDAR-SLAM pose noise, converts each bimanual UMI episode into one continuous 12-DoF YAM trajectory, validates the result, and provides an episode-selectable viewer. It never uploads or writes back to Hugging Face.

Dataset reads are cached under `.cache/`; reports and replay artifacts are written under `reports/` and `outputs/`. Those paths are git-ignored. Generated experiment reports and the separate `dataset-filtering-tests/` research workspace are kept locally rather than tracked in this repository.

On a fresh checkout, generate a reachability report using the command below before running `build-filtered-dataset`, or restore a matching report from your local research archive. Historical metrics below summarize archived experiments; the full reports are not bundled with the checkout.

## Quick start

```bash
uv sync --extra dev

# Linux only: install TOPP-RA for source-aligned dynamics retiming
uv sync --extra dev --extra retiming

# Viewer: http://127.0.0.1:8765
uv run umi-filter viewer

# Noise and smoothing experiment over all 115 episodes
uv run umi-filter analyze \
  --output reports/dual-lidar-umi-noise-and-smoothing.json

# Full continuous replayability analysis (expensive)
uv run umi-filter reachability --solver continuous --workers 8 \
  --output reports/dual-lidar-umi-tail-freeze-v2-replayability.json

# Read-only ablation: rerun after removing the final second in memory
uv run umi-filter reachability --solver continuous --workers 8 \
  --trim-end-seconds 1.0 \
  --output reports/dual-lidar-umi-trim-last-second-replayability.json

# Old framewise IK metric, retained only as a diagnostic baseline
uv run umi-filter reachability --solver greedy \
  --output reports/dual-lidar-umi-greedy-ik-reachability.json

# Export one local continuous replay artifact
uv run umi-filter retarget --episode 0 \
  --output outputs/episode-000-continuous.npz

# Build the filtered LeRobot dataset locally (never uploads)
uv run umi-filter build-filtered-dataset \
  --output outputs/dual-lidar-umi-filtered

# Upload is a separate, explicit command after local validation
uv sync --extra hub
uv run umi-filter push-filtered-dataset \
  --dataset outputs/dual-lidar-umi-filtered \
  --repo-id brandonyang/dual-lidar-umi-filtered

uv run pytest -q
uvx ruff check src tests
```

`--source` accepts the Hugging Face repository id or a local LeRobot v3 dataset root. The expected UMI schema is two 6-DoF poses in `observation.state` plus two gripper widths.

## Interactive gripper calibration for future exports

Calibrate one UMI at a time from the repository root while its camera is idle:

```bash
uv run --package umi-live-server python dataset-filtering/scripts/calibrate_gripper_range.py --gripper umi1
uv run --package umi-live-server python dataset-filtering/scripts/calibrate_gripper_range.py --gripper umi2
```

For each command, fully close the selected gripper and press Enter, then fully open it and press Enter. Each endpoint captures 30 live ArUco width measurements with the same fisheye detector as dataset processing, rejects an unstable hold, and stores the median endpoints in `config/calib/<umi>_gripper_range.json`.

Pass either or both files to a future combined build:

```bash
umi-filter build-combined-datasets \
  --report reports/combined/pipeline-report.json \
  --cartesian-output outputs/future-cartesian \
  --joint-output outputs/future-joints \
  --umi1-gripper-calibration config/calib/umi1_gripper_range.json \
  --umi2-gripper-calibration config/calib/umi2_gripper_range.json
```

Each calibration file applies only to the UMI named inside it; swapping the files is rejected. A missing arm-specific flag uses the built-in calibration for that arm.

## Noise and smoothing result

The completed report covers 115 episodes, 110,269 frames, and 3,675.63 seconds (61.26 minutes) at 30 Hz. The median high-frequency residual above 6 Hz is 0.130 mm translation and 0.0217° rotation.

Nine Savitzky–Golay, Gaussian, and zero-phase Butterworth configurations were compared. A third-order 4 Hz Butterworth filter gave the best distortion-aware result:

- 96.5% of translation and 95.5% of rotation high-frequency residual removed;
- 84.4% reduction in translation jerk;
- only 0.318 mm translation and 0.059° rotation RMS path change; and
- original poses restored exactly at detected gripper open/close events, with a short blended neighborhood.

The locally generated `reports/dual-lidar-umi-noise-and-smoothing.json` contains aggregate, candidate, and per-episode metrics.

The earlier greedy framewise baseline solved 95.95% of left-arm targets, 95.61% of right-arm targets, and 91.96% of simultaneous targets within its loose 10 mm/8° threshold. That is **not** a replayability fraction: independently projected frames can switch IK branches or jump even when adjacent frames both count as reachable.

The revised two-pass continuous optimizer completed all 115 episodes and all 110,269 frames:

- 98.061% of individual-arm targets and 96.995% of simultaneous bimanual frames are within 10 mm/8°;
- 1,919/1,922 gripper-event targets (99.844%) are strictly feasible at 1 mm/2°;
- all 115 solves satisfy joint, velocity, and configured acceleration limits;
- conditional per-arm tail freezing modifies 52 episodes and reduces failed bimanual frames from 6,878 to 3,314;
- 70/115 episodes pass strict gripper events plus the hard all-frame 10 mm/8° IK-trajectory gate;
- 42/115 episodes pass the complete conservative classification. The main additional rejection is collision screening: 52/115 pass arm-to-arm capsule clearance and 101/115 pass self-clearance. This is intentionally stricter than “IK solved” and still requires exact scene validation.

The locally generated `reports/dual-lidar-umi-tail-freeze-v2-replayability.json` contains every episode and failure reason.

## Continuous retargeting algorithm

The default solver operates on the smoothed Hugging Face Cartesian trajectory. Greedy IK is used only to initialize it and remains available in the viewer for comparison.

1. Detect gripper-width transitions and solve each event with multi-start, bounded IK at a strict 1 mm/2° tolerance. A dynamic program selects a consistent event branch.
2. Fit one provisional cubic B-spline over both arms' 12 joint coordinates. CasADi/IPOPT jointly minimizes robust Cartesian error, approach/retreat error, motion-flow mismatch, bimanual relative-pose mismatch, camera-pose deviation, and joint velocity/acceleration/jerk.
3. For each arm independently, inspect the provisional result from 0.5 s after that arm's last gripper event through the end. If any frame exceeds 10 mm/8°, hold that arm's smoothed 6-DoF cutoff pose through the remaining target frames.
4. Rerun the coupled bimanual optimizer when either arm is frozen. Enforce joint bounds, exact feasible event configurations, and sampled YAM capsule collision clearances during the nonlinear solve.
5. If validation finds a collision no deeper than 20 mm, add only the offending validation-resolution link pairs and frames to a sparse repair solve. Gripper events become hard 1 mm/2° Cartesian constraints during repair, while branch continuity, joint-limit margin, and collision proximity form the secondary objective. Keep the baseline unless the repair preserves all primary accuracy gates and strictly improves clearance.
6. Preserve the final geometric joint path and retime it for the YAM URDF velocity limits plus a configurable acceleration assumption. The retimer may slow the demonstration but can never play it faster than the recorded video.
7. Classify the complete episode. A retained result must solve, satisfy every strict event, keep both arms within 10 mm/8° at every frame, remain inside joint/dynamics limits, and pass the configured collision proxies.

The optimizer never substitutes a per-frame “nearest reachable” pose. Unreachable portions bend gradually through joint space; there is no hold-then-jump behavior in continuous mode.

## Coordinate and hardware assumptions

- `umi1` maps to the left YAM and `umi2` to the right YAM.
- Each UMI starts at its local position zero.
- The YAM bases are 18 in / 0.4572 m apart along the horizontal axis and initially face the same direction.
- UMI `(+X up, +Y right, +Z forward)` maps to YAM world `(+Z up, -Y left, +X forward)`.
- The public [YAM URDF](https://github.com/williamtsai726/YAM/blob/main/i2rt/i2rt/robot_models/yam/yam.urdf) supplies joint positions and 10 rad/s velocity limits.
- The URDF has no acceleration limits. The default 20 rad/s² value is an explicit planning assumption and must be replaced with measured/controller-approved limits.
- The collision model is a conservative sampled-capsule proxy based on public [YAM MuJoCo](https://github.com/williamtsai726/YAM/blob/main/i2rt/i2rt/robot_models/yam/yam.xml) radii. Exact self/arm/scene collision checking is still mandatory before hardware replay. Pass `--table-height` to include a horizontal table proxy.

“Continuously replayable” therefore means replayable under these configured model assumptions; it is not authorization to command hardware.

## Video/action consistency

The export preserves both `source_timestamps` and `robot_timestamps`. Corresponding array rows represent the same semantic pose, while the paired timelines define the monotonic video-to-robot time warp. Do not index the original video directly with robot timestamps.

The source UMI tool-to-camera and YAM tool-to-camera transforms are different quantities. Supply measured transforms when available:

```bash
uv run umi-filter retarget --episode 0 \
  --source-camera-translation SX SY SZ \
  --source-camera-rpy-deg SR SP SYAW \
  --yam-camera-translation YX YY YZ \
  --yam-camera-rpy-deg YR YP YYAW \
  --camera-extrinsics-calibrated \
  --output outputs/episode-000-calibrated.npz
```

Without `--camera-extrinsics-calibrated`, camera metrics are labeled an uncalibrated TCP proxy and `video_alignment_weight` stays at one—the tool will not silently filter training frames using guessed calibration. With calibration, the export contains per-frame camera translation/rotation error and a soft alignment weight. Large viewpoint corrections should be excluded or re-recorded in simulation/with the YAM camera.

## Viewer

The viewer has an episode menu, smoothing menu, and solver menu:

- left: raw and smoothed UMI trajectories (`umi1` is always left);
- right: the original UMI paths (dashed) overlaid with the achieved YAM FK paths (solid), both YAM chains, and a live target-to-achieved residual connector;
- one robot-time playback clock paired with source-time frames; and
- live pose errors, event markers, smoothing statistics, solver status, and replayability metadata.

Continuous mode always displays the optimized joint path. Greedy mode deliberately retains the old hold-last-valid behavior so discontinuities are visible as a diagnostic comparison.

## Artifact contents

A continuous `.npz` export contains smoothed targets, achieved FK poses, both arms' joint trajectory and link points, strict-event metadata, source and robot timestamps, position/orientation errors, camera errors and weights, original gripper widths, and complete JSON solver/validation metadata. The source video and dataset are never modified.

The filtered-dataset exporter selects only episodes marked `continuously_replayable` in
the complete report, recomputes each trajectory, and refuses stale selections that no
longer pass. It preserves the LeRobot-v3 UMI observation schema and source-frame video
alignment, replaces `observation.state` with the achieved optimized path mapped back to
UMI coordinates, and removes `action`. Build and upload are deliberately separate commands.

## Code map

- `noise.py`, `smoothing.py`: noise metrics, filter benchmark, and event anchoring.
- `kinematics.py`: YAM FK, DLS IK, and strict bounded multi-start IK.
- `trajectory_optimization.py`: contact branch selection, bimanual spline NLP, collision proxy, camera scoring, and validation.
- `retiming.py`: source-aligned TOPP-RA and deterministic conservative fallback.
- `retarget.py`: coordinate mapping and legacy greedy baseline.
- `hf_dataset.py`: read-only LeRobot v3 adapter.
- `viewer.py`, `web/index.html`: local API and interactive viewer.
- `analysis.py`, `reachability.py`, `cli.py`: experiments, reports, and commands.
