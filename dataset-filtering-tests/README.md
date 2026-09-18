# Dataset Filtering R&D

This directory is an isolated experiment workspace for improving the replayability of
`brandonyang/dual-lidar-umi-relative`. Production code remains in `../dataset-filtering`;
an experiment moves there only after a full-dataset result and regression audit justify it.

## Target and frozen baseline

- Baseline: 48/150 episodes retained (32.0%) by the v3 pipeline.
- Research target: at least 105/150 episodes retained (70%).
- The target is not permission to tune gates until the number is reached.
- Current gates remain 10 mm/8° at every frame, 1 mm/2° at gripper events, 5 mm
  inter-arm surface clearance, 1 mm self-clearance, plus solver, joint, dynamics, and
  optional table-clearance requirements.

Every experiment must report both retention and source-fidelity/safety effects. A method
is rejected if it makes the data easier only by deleting task-critical motion, moving
gripper events relative to the video, hiding failed frames, or changing thresholds after
seeing the acceptance count.

## Current evidence

- Failure attribution shows that tracking is the first bottleneck: only 65/150 baseline
  episodes pass all-frame tracking, while 65 pass inter-arm clearance and 125 pass
  self-clearance. Most rejection signatures overlap tracking and inter-arm failure.
- Event-safe head trimming/re-anchoring is rejected as a default. On the 85 baseline
  tracking failures, complete framewise IK reachability falls from 17 at zero trim to
  14, 3, and 0 after requested 1 s, 2 s, and 3 s trims. A 1 s trim helps only one
  episode and regresses four.
- Denser continuous Cartesian sampling is the first verified improvement. It adds 13
  tracking-failure episodes and 3 collision-only episodes without changing gates. A safe,
  stronger event-preserving smoother adds one more, raising the conservative
  same-registration portfolio from 48/150 to 65/150 (43.3%). Dense sampling costs about
  25 minutes wall time for the 85-episode residual sweep on eight CPU workers.
- Preserving the all-frame greedy IK warm start adds no episode beyond dense sampling.
- Raising the deep-collision repair budget adds 0/5 and takes a median 209 s per episode.
  Collision-clearance homotopy reaches 4.364 mm on its diagnostic versus the required
  5 mm, so the episode remains rejected.
- Enabling Cartesian event-branch adjustment and the collision secondary objective from
  the initial solve adds 0/5 and takes a median 849 s per episode; it is rejected.
- A literal all-frame hard-constraint NLP was stopped after more than 11 minutes on one
  episode and is computationally impractical in its current form.
- The source-code coordinate audit confirms UMI `+X up, +Y right, +Z forward`; no axis
  remapping change is justified. A different robot start posture must not be invented
  without the calibrated YAM deployment posture.

The latest verified same-registration portfolio is 65/150. A modest alternative global
YAM start registration (`q2=q3=0.6 rad`) improved a stratified framewise-IK cohort but
retained only 19/102 baseline-rejected episodes in the complete pipeline and regressed
contact reachability. Mixing zero-home and alternative-home results is not counted as a
deployment-consistent acceptance rate. See `RESULTS.md` for the full audit.

## Experiment order

1. Attribute overlapping failure cohorts and define fast screening metrics.
2. Test event-safe removal/re-anchoring of the initial setup segment.
3. Add active-set all-frame tracking repair under the unchanged 10 mm/8° envelope.
4. Test adaptive spline resolution, objective sampling, and global IK initializations.
5. Improve coupled contact-branch and collision optimization.
6. Test combinations on all 150 episodes and audit video alignment, event preservation,
   continuity, dynamics, collision clearance, and source-trajectory distortion.

Run the baseline summary with:

```bash
uv run filter-rd baseline
```
