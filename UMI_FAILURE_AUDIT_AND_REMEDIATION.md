# UMI Failure Audit and Code Remediation Plan

**Audit window:** 2026-08-28 through 2026-09-04 UTC  
**Active dataset:** `brandonyang/artem-pour-water`  
**Active server tree:** `/home/ubuntu/umi-flow/data/artem-pour-water`  
**Historical dataset excluded from active counters:** `artem-fold-towel`

## Purpose

This document consolidates the failure instances reported in the task history and maps them to code changes that should prevent recurrence. It distinguishes operational failures from ordinary dataset-quality failures.

The realistic reliability target is **zero preventable operational errors, zero lost or silently corrupted takes, and zero infinite retry loops**. A target of zero ordinary quality failures is not realistic: tracking loss, poor contact, self-collision, and failed task execution are properties of collected demonstrations and should remain visible rather than being hidden by looser thresholds.

## Executive summary

The incidents are not independent. Most came from five root design gaps:

1. A recording can start without proving that both Pis and the server have enough projected storage headroom.
2. A Pi upload first creates another full tar archive on the same constrained filesystem, so an almost-full Pi cannot upload the data that would free it.
3. Cross-UMI start, stop, precheck, and restart operations are not transactional or idempotent, allowing races and one-sided recordings.
4. Maintenance scripts depend on external safety checks but do not enforce their own singleton lock and recording lease.
5. The server retries deterministic, dataset-ineligible, or permanently damaged episodes indefinitely instead of assigning a terminal disposition.

These gaps produced storage exhaustion, capture corruption, HTTP 507 responses, stale-process mismatches, orphan arms, repeated decode/GLIM failures, and misleading readiness reports.

## Causal map

```text
No projected-capacity admission
  -> Pi/server disks approach zero
  -> capture writes fail or overrun
  -> uploader cannot build temporary tar
  -> Pi queue cannot drain
  -> server decode/GLIM fails with ENOSPC

Non-transactional coordinator + transient connectivity
  -> START/PRECHECK/restart races
  -> one arm records without the other
  -> unmatched raw episodes or incomplete pairs

No terminal failure classification
  -> known-ineligible or permanently damaged episodes retry forever
  -> workers and disk remain occupied
  -> retention cannot obtain a safe archive window

Cleanup without an internal lease/singleton
  -> cleanup competes with capture for I/O
  -> camera/LiDAR drops
  -> interrupted cleanup leaves ambiguous marker state
```

## Failure inventory

### 1. Pi storage exhaustion

#### Observed instances

- On 2026-09-01, UMI1 fell from roughly 25 GiB free to **0 bytes free**; UMI2 fell to roughly **647 MiB free**.
- On 2026-09-03/04, UMI1 again reached **0 bytes free** and UMI2 was around **0.52 GiB free**.
- Historical `artem-fold-towel` raw payloads remained on the Pis after the active dataset changed. The new `artem-pour-water` archive receipts could not validate those historical payloads, so the authorized cleanup path could not remove them.
- Sixteen current `pour water` sessions encountered UMI1 camera `ENOSPC`; seven of those were still being retried by the server pipeline at the audited snapshot.

#### Impact

- New takes were allowed to start with insufficient working space.
- Camera files, manifests, checksums, completion markers, and cleanup safety markers could no longer be written reliably.
- The uploader and cleanup path deadlocked exactly when they were most needed.

#### Code cause

- `device/src/umi_flow_device/preflight.py` checks camera/tag and LiDAR time state, but not storage capacity.
- `device/scripts/record.py::_preflight_before_start` does not obtain a projected-capacity report from both Pis or the server.
- `device/src/umi_flow_device/episode.py::EpisodeRecorder.start` has no storage admission check or emergency metadata reserve.
- The rig configuration in `common/umi-flow-core/src/umi_flow_core/config.py` has no storage policy fields.

#### Required code changes

1. Add a `StoragePolicy` configuration object with at least:
   - `min_free_bytes_after_take`
   - `estimated_take_bytes`
   - `upload_scratch_bytes`
   - `metadata_emergency_reserve_bytes`
   - `server_min_free_bytes_after_take`
2. Add a `StorageHeadroomCheck` to `device/src/umi_flow_device/preflight.py`. It must use `statvfs`, current unsent queue size, a conservative take-size estimate, and uploader scratch requirements.
3. Make `EpisodeRecorder.start` fail closed unless the check has passed recently for the same proposed session ID.
4. Exchange the capacity result between both Pis during coordinator prepare. The leader must reject the take if either side would fall below its post-take reserve.
5. Query a server admission endpoint before committing a take. The response should include current free bytes, reserved bytes, queued ingest bytes, and whether a new paired take is accepted.
6. Reserve a small amount of disk exclusively for writing manifests, terminal error receipts, and cleanup markers. The recorder must never count this reserve as capture capacity.

#### Tests to add

- `device/tests/test_preflight.py`: reject when projected post-take free space is below reserve.
- `device/tests/test_episode.py`: `start()` cannot bypass a stale or missing capacity token.
- `device/tests/test_record.py`: either Pi or server rejecting capacity prevents START on both arms.
- Fault-injection test: force `ENOSPC` during stop and verify an explicit recoverable failure state is preserved.

---

### 2. Uploader temporary-tar deadlock

#### Observed instances

- Complete episodes could not drain from the Pi because packaging required enough room for another archive-sized `*.tar.tmp` file.
- Cleanup policy correctly refused to clean while complete/staging queues were non-empty, while the uploader could not empty those queues because the disk was full.

#### Impact

- Recoverable completed takes remained stranded on the Pi.
- Queue size and disk use increased together.
- Operators were forced toward risky manual cleanup during an active collection window.

#### Code cause

- `device/src/umi_flow_device/uploader.py::create_episode_tar` creates a full tar in the local staging directory and renames it only after completion.
- `upload_episode` and `_upload_with_tar` require a local `Path` and fixed archive before the HTTP request begins.

#### Required code changes

1. Add a streaming upload path that emits the tar directly into the HTTP request without storing a full second copy on the Pi.
2. Use HTTP chunked transfer or a streaming request body, with a rolling byte counter and server-side maximum enforcement.
3. Keep the existing staged-tar path only as an optional fallback when the headroom check proves enough scratch space exists.
4. Preserve idempotency: the server should key the upload by episode/session ID plus manifest digest and return the same receipt for a safe retry.
5. Move an episode to `sent` only after a durable server acceptance receipt is returned.
6. If streaming stops partway through, leave the source episode unchanged and let the server atomically discard the partial upload.

#### Tests to add

- `device/tests/test_uploader.py`: upload succeeds when free space is smaller than the episode but sufficient for metadata only.
- Retry after a dropped connection sends no duplicate episode and does not move the source to `sent` prematurely.
- Server receiver test for a chunked request exceeding the declared maximum.
- Integration test for interrupted upload cleanup and idempotent retry.

---

### 3. Camera writer overruns, dropped frames, and LiDAR kernel drops

#### Observed instances

- Sixteen sessions reported camera writer-queue overruns.
- Ten of those sessions reported camera drops totaling **410 frames**.
- All sixteen also reported LiDAR kernel drops.
- Eight additional sessions reported LiDAR-only drop warnings.
- On 2026-08-28, hashing/cleanup work ran during capture and was shown to contribute to I/O contention. General storage and scheduling pressure remained a contributor in later incidents.

These categories overlap; they must not be summed as distinct episode totals.

#### Impact

- Captures were marked dataset-ineligible.
- Some damage was detected only after a take completed, wasting operator time and downstream compute.
- Paired takes could continue after one side was already irrecoverably unhealthy.

#### Code cause

- Recording and maintenance do not share a process-level lease.
- Systemd units do not provide strong resource isolation between capture, upload, hashing, and maintenance.
- The camera and LiDAR monitors record failures, but the paired coordinator does not reliably stop both sides as soon as a fatal capture condition appears.

#### Required code changes

1. Add a shared capture/maintenance lease using `fcntl.flock`:
   - recorder holds the capture lease for the entire take;
   - cleanup requires the mutually exclusive maintenance lease;
   - the cleanup helper validates recorder/capture state itself rather than trusting its caller.
2. Add systemd resource policy:
   - give capture predictable CPU and I/O priority;
   - run uploader and cleanup with lower I/O priority;
   - cap non-capture workers on the Pi while recording.
3. Treat camera queue overrun, camera write error, and excessive LiDAR kernel-drop delta as paired fatal events. The detecting arm should request a graceful paired stop and both manifests should reference the same fault.
4. Extend preflight to verify camera write throughput and LiDAR socket/kernel receive-buffer settings under a short bounded load.
5. Record maximum queue depth, drop deltas, write latency, disk latency, and competing-maintenance state in every manifest.

#### Tests to add

- `device/tests/test_camera.py`: threshold crossing produces a typed fatal capture event.
- `device/tests/test_record.py`: a fatal event on either arm gracefully stops both arms once.
- `device/tests/test_lidar_monitor.py`: kernel-drop delta is session-scoped and cannot inherit a stale system total.
- Hardware soak test with simultaneous capture and uploader traffic; cleanup must be rejected for the full take.

---

### 4. Server disk pressure and HTTP 507

#### Observed instances

- On 2026-08-28 the server had roughly **24.1 GiB free**, returned HTTP 507, retained 37 Pi uploads, and had 132 pairs blocked before decode.
- On 2026-08-29 the server had roughly **42 GiB free** and could not obtain a stage-free archive window.
- On 2026-09-02/03 server free space fell from roughly **131 GiB to 35.8 GiB**.
- On 2026-09-03/04 server free space was again below the 64 GiB readiness threshold while the Pis were also exhausted.

#### Impact

- Uploads remained queued on already-full Pis.
- Decode and GLIM failed with `ENOSPC`.
- Retention could not create headroom because active/retrying stages rarely left a safe archive window.

#### Code cause

- `server/src/umi_flow_server/server.py` stores the upload tar and then extracts it, requiring substantial temporary expansion headroom.
- `server/src/umi_flow_server/runner.py` schedules work without a stage-specific projected-space budget or low-disk drain mode.
- `server/src/umi_flow_server/retention.py` can prune only completed eligible material, while poison retries keep work active.
- `server/scripts/archive_published_raw.py` relies on an external no-active-stage check instead of a runner-owned pause/drain handshake.

#### Required code changes

1. Add `/health/admission` or equivalent to the ingest service. Return:
   - actual and reserved free bytes;
   - accepted maximum upload bytes;
   - active and queued stage bytes;
   - a typed rejection reason;
   - a retry-after value.
2. Estimate per-stage expansion before dispatch. Refuse decode/GLIM dispatch if projected completion would violate the stage reserve.
3. Add a low-disk drain mode that prioritizes already-near-complete pairs, publication, and retention rather than starting more high-expansion stages.
4. Add a coordinated runner pause-and-drain API for retention/archive. It should stop new dispatch, wait for current stages to finish, archive eligible raw, then resume.
5. Replace the fixed worst-case extraction reservation with a validated upload-specific budget where safe. Preserve atomic extraction and path validation.
6. Expose current HTTP 507 and admission rejection state as structured metrics rather than relying on journal text.

#### Tests to add

- `server/tests/test_receiver.py`: admission response and exact reserve behavior.
- `server/tests/test_runner.py`: stage-specific projected space prevents dispatch and enters drain mode.
- `server/tests/test_retention.py`: pause/drain/archive/resume sequence.
- End-to-end test: a full server returns backpressure before a take starts, not after the Pi has recorded it.

---

### 5. Coordinator races, stale loaded code, and one-sided recordings

#### Observed instances

- **2026-08-28:** UMI2 had stale code loaded in its long-running recorder. Follower precheck returned an invalid result, and UMI2 became orphaned after UMI1 was restarted.
- **2026-09-01 around 20:06 UTC:** a new START overlapped a precheck. The precheck returned an incomplete/invalid result and triggered an unnecessary restart.
- **2026-09-02 around 19:25 UTC:** a restart race interrupted both recorders and left `in_progress` state.
- **2026-09-02 around 19:36 UTC:** UMI1 continued capturing after UMI2 disappeared.
- **2026-09-02 around 20:10 UTC:** UMI1 was unmatched and UMI2 incomplete.
- **2026-09-03 around 23:54 UTC:** UMI2 became unmatched after UMI1 failed from storage exhaustion.

Current unmatched active-dataset raw episodes observed in the audit:

- `episode_20260901T225756Z_umi2_454899320`
- `episode_20260902T201046Z_umi1_659497215`
- `episode_20260903T235421Z_umi2_953862947`

#### Impact

- The two arms disagreed about whether a take had started or stopped.
- Valid raw data required manual disposition.
- Restarts intended to repair readiness created additional partial takes.

#### Code cause

- `device/scripts/record.py::_leader_start` tells the follower to start before the leader commits locally.
- `_remote_precheck` and the follower command loop operate against mutable recorder state without a generation or reservation token.
- `device/src/umi_flow_device/coordinator.py` has no protocol version, command ID, idempotency key, build ID, task/config digest, session lease, or two-phase start.
- A follower-detected fatal condition can stop only the follower; it does not reliably drive a paired state transition.

#### Required code changes

1. Version the coordinator protocol and add a typed status envelope containing:
   - `protocol_version`
   - `build_git_sha`
   - `config_sha256`
   - `task`
   - `recorder_generation`
   - `state`
   - `current_session_id`
   - capacity and preflight result
2. Replace one-step START with two-phase start:
   - `PREPARE_START(session_id, command_id)` runs preflight and reserves state on both arms;
   - leader verifies both prepare responses and starts local resources;
   - `COMMIT_START` begins capture for the same reserved session;
   - `ABORT_START` releases both reservations if either side fails.
3. Make every command idempotent by `(command_id, session_id, recorder_generation)`. Duplicate delivery must return the previous result.
4. Separate read-only status from sensor-opening precheck. A busy response is not an invalid protocol response and must never automatically cause a restart.
5. Add a mid-take paired lease/heartbeat. After a bounded grace interval, loss or fatal failure of one arm should trigger one graceful stop on its peer, preserving both halves.
6. Write a loaded-state JSON file at process startup containing the actual build/config/task in memory. Monitors should inspect this file or protocol response, not just on-disk source hashes.
7. Permit restart only after a compare-and-swap state transition from the same observed idle generation.

#### Tests to add

- `device/tests/test_coordinator.py`: duplicate commands, dropped replies, stale generation, incompatible protocol, and reconnect.
- `device/tests/test_record.py`: START racing PRECHECK, local failure after remote prepare, follower failure mid-take, and restart racing START.
- Fault-injection integration test proving that no single injected coordinator/network failure produces a one-sided active recording.

---

### 6. Gateway and `.local` connectivity failures

#### Observed instances

- Long periods of gateway timeout or `.local` resolution failure occurred during 2026-08-29 through 2026-09-01, again on 2026-09-02, and for much of 2026-09-03 from approximately 02:00 through 19:00 UTC.
- At least one connectivity loss occurred during an active take.

#### Impact

- UMI state could not be verified.
- A peer could disappear while the other continued recording.
- Monitoring could not distinguish gateway failure, DNS failure, and Pi failure.

#### Code/config cause

- Peer coordination and operations depend on `.local` name resolution behind the gateway.
- The protocol has no durable peer lease or structured connectivity diagnosis.

#### Required code changes

1. Configure peer and operations endpoints using stable Tailscale addresses or a verified static host mapping; retain `.local` only as a fallback.
2. Resolve and validate the peer endpoint at service startup and expose the selected address in loaded-state status.
3. Add separate health states for gateway unreachable, DNS failure, TCP refusal, protocol incompatibility, and recorder unavailable.
4. Combine the coordinator heartbeat with a grace period so a brief packet loss does not destroy a healthy take, while a sustained loss cannot leave the peer recording indefinitely.
5. Add safe `Restart=on-failure` behavior only for idle processes; recording state must never be restarted blindly by systemd or the monitor.

#### Tests to add

- DNS failure with stable-address fallback.
- Connection drop during PREPARE and during active capture.
- Gateway unreachable while server remains healthy; report UMI state as unknown, never healthy.

---

### 7. Cross-UMI clock synchronization failures

#### Observed instances

Two paired sessions were rejected for clock synchronization:

- `paired_sess_1788298084778965777`
- `paired_sess_1788468409501402448`

#### Impact

- Both arms could capture apparently valid data that was not safe to combine on a shared time axis.
- Compute was spent on episodes that should have been rejected before capture or terminalized immediately after pairing.

#### Code cause

- Preflight verifies each LiDAR's local time-sync status but does not prove a bounded cross-Pi offset immediately before start.
- The coordinator handshake does not expose uncertainty, multiple samples, or a shared reference-clock identity.

#### Required code changes

1. Add a cross-Pi clock check during `PREPARE_START` using multiple bounded RTT samples.
2. Use the minimum-RTT sample or another documented estimator and record offset plus uncertainty.
3. Require both Pis to report the same reference clock/domain and enforce the existing 5 ms dataset gate before COMMIT.
4. Store all clock evidence in both manifests so the server can independently verify it.
5. Keep the server pair-stage clock gate as defense in depth; a failure there should create a terminal receipt and skip decode/GLIM.

#### Tests to add

- Offset just below/above 5 ms.
- Asymmetric RTT and stale time sample.
- Reference-domain mismatch.
- Pair runner terminalizes an unacceptable clock without scheduling expensive stages.

---

### 8. Infinite or wasteful pipeline retries

#### Active-dataset instances at the audit snapshot

Decode failures caused by `ENOSPC`:

- `paired_sess_1788303131061808813`
- `paired_sess_1788303308584816022`
- `paired_sess_1788478694429174534`
- `paired_sess_1788478817590441904`

GLIM failures caused by `ENOSPC` or missing trajectory output:

- `paired_sess_1788477808484237176`
- `paired_sess_1788478199950011522`
- `paired_sess_1788478928614985001`

Genuine GLIM failure:

- `paired_sess_1787199470430852468`: return code 250, “Indeterminant linear system.”

Historical `artem-fold-towel` examples, excluded from active readiness counters but relevant to the bug:

- `paired_sess_1787945594771912043`: approximately 2.7 seconds, no trajectory.
- `paired_sess_1787948332695118080`: approximately 0.35 seconds, zero camera frames/no UMI1 PCAP, repeatedly sent to decode.

#### Impact

- Deterministic failures occupied worker slots forever.
- Restarts reset in-memory retry knowledge.
- Repeated work consumed disk and prevented retention windows.
- Queue counters mixed actionable work with episodes that could never progress.

#### Code cause

- `server/src/umi_flow_server/runner.py::FailureState` is in memory only.
- `_record_failure` applies capped exponential delay but no maximum attempts or terminal class.
- `ready_runs` does not short-circuit `manifest.dataset_eligible == false` before decode and GLIM.
- Stage failures do not produce a durable, typed failure receipt that excludes the run from future scheduling.

#### Required code changes

1. Before dispatch, terminalize pairs that are already dataset-ineligible or have unacceptable clock evidence. Write a durable terminal receipt and do not schedule decode/GLIM.
2. Persist per-stage retry state under the episode derived tree:
   - attempt count and first/last failure time;
   - normalized error code/class;
   - source manifest digest and software build;
   - next eligible retry time.
3. Classify failures:
   - **transient:** low disk, temporary service/network failure; retry after the blocking condition clears;
   - **deterministic input failure:** missing required stream, zero frames, known ineligible manifest; terminal immediately;
   - **algorithmic failure:** bounded retry count, then durable `FAILED.json`/dead-letter state;
   - **operator-approved requeue:** only after source/config/build changes are explicit.
4. Stop deleting all staging evidence blindly. Preserve the final failing log and a compact diagnostic receipt.
5. Exclude terminal states from `ready_runs` and from publication queue counts. Report them separately.

#### Tests to add

- `server/tests/test_runner.py`: ineligible pair schedules zero stages.
- Retry state survives runner restart.
- Deterministic failures run once; algorithmic failures stop at the configured attempt limit.
- A transient `ENOSPC` retry resumes only after capacity is restored.
- Terminal episodes never appear as publication-queued.

---

### 9. Cleanup and archiver safety defects

#### Observed instances

- Stale staging artifacts blocked the server archiver until the guard was corrected.
- Pi cleanup/hashing during recording caused or contributed to frame and packet drops.
- At zero free space, cleanup could not write its own safety marker.
- An interrupted SSH session left a remote cleanup process running; a second helper collided on the same temporary marker.

#### Impact

- A repair operation could worsen capture quality.
- Operators could not reliably tell whether cleanup was still active.
- Marker collisions made a safe, receipt-driven operation appear ambiguous.

#### Code cause

- `device/scripts/archive_sent_raw.py` and `server/scripts/archive_published_raw.py` have no process-level singleton lock.
- Their `_write_json` helpers use predictable sibling temporary files.
- Critical guards are implemented in the monitor procedure rather than owned by the destructive helper itself.

#### Required code changes

1. Add a nonblocking singleton `flock` to both archive helpers. Include PID, start time, build, and target data root in lock diagnostics.
2. The Pi helper must internally verify:
   - both recorder states are idle through a fresh coordinator status;
   - no capture children exist;
   - complete and staging queues are empty;
   - the capture lease is not held;
   - each candidate still matches its trusted receipt and fresh hash immediately before deletion.
3. Use unique temporary receipt names containing PID and a random/command ID, followed by file `fsync`, atomic rename, and parent-directory `fsync`.
4. Make durable intent possible at low space by maintaining a small preallocated maintenance metadata reserve. Release only the amount required to write the intent/receipt; never delete payload first.
5. Add a resumable state machine: `VERIFIED -> INTENT_DURABLE -> PAYLOAD_REMOVED -> RECEIPT_DURABLE`. Re-entry must be safe after SSH loss or power failure.
6. The server helper must acquire a runner pause/drain lease itself rather than relying on a caller's point-in-time process check.

#### Tests to add

- Concurrent helper invocation: exactly one proceeds.
- SSH/client disconnect does not permit a second process to collide.
- Power-loss simulation at every state transition is idempotently recoverable.
- Zero-space marker test using the metadata reserve.
- Cleanup invoked during capture is rejected before hashing begins.

---

### 10. Readiness-monitor and viewer reporting errors

#### Observed instances

- One readiness verdict ignored an invalid follower result and could have reported ready based on matching files alone.
- At times, a queue over a warning threshold was treated as an automatic `NO`, even though a controlled, draining backlog is allowed.
- An automation prompt was corrupted by `tomllib`-style parsing/quoting issues.
- The viewer displayed grey/white tiles because of a metadata/rendering issue.
- A sleeping viewer was sometimes at risk of being described as failed even though sleep is an expected state.

#### Impact

- Readiness could be both falsely positive and falsely negative.
- Operators could not trust counters or distinguish active work from queued work.
- UI defects looked like data corruption.

#### Code changes

1. Replace journal/string parsing with structured status JSON from recorder, uploader, ingest, runner, dataset publisher, quality watcher, and retention.
2. Define one shared status schema in `umi-flow-core` with explicit fields for:
   - active stage/process evidence;
   - queued-by-next-stage counts;
   - terminal skipped/ineligible counts;
   - upload queue count/bytes and observed growth;
   - current disk bytes and trend;
   - task/config/build identity;
   - follower precheck parse validity.
3. Make the readiness gate a pure function over that schema and unit-test every threshold and exception.
4. A malformed `PipelineResult.from_dict` response must always make an idle-online gate `NO`.
5. Warning/review queue thresholds must annotate risk, not force `NO` unless the documented stall/growth/storage conditions are also true.
6. Store automation configuration as validated structured data, not interpolated TOML/Python text.
7. Add a viewer metadata contract test covering empty datasets, missing optional metadata, tile color mapping, running, sleeping, and actual failure states.

#### Tests to add

- Table-driven readiness tests for every gate condition.
- Explicit counter tests: queued work is never labeled active, and terminal skipped work is never publication-queued.
- Viewer smoke test against zero-, one-, and many-episode datasets.

---

### 11. Dataset-quality failures

At the audited snapshot, active quality status reported:

- Evaluated: **321**
- Passed: **170**
- Failed: **151**
- Processing: **3**
- Operational errors: **0**

Failure-reason counts overlap:

- Tracking: **136**
- Contact: **135**
- Self-collision: **79**
- Inter-arm: **13**

Most common overlapping combinations included:

- Contact + self-collision + tracking: **72**
- Contact + tracking: **44**

For the historical fold-towel data, the newest 500 failure rate was approximately **64.4%**, versus approximately **27.1%** for the preceding comparison window, mainly tracking and contact. This is not part of `artem-pour-water` readiness, but it shows why quality trends should be sliced by configuration and collection period.

#### Interpretation

These are not service crashes. They should not be “fixed” by weakening quality thresholds. They need better capture conditions, calibration, operator feedback, and diagnostics that separate hardware/system causes from failed demonstrations.

#### Required code changes

1. Add per-episode quality analytics keyed by build, config digest, calibration digest, UMI, operator/session window, and capture-health metrics.
2. Correlate quality failures with frame drops, LiDAR drops, clock uncertainty, disk latency, and tracking confidence.
3. Surface reason combinations and latest-100 versus preceding-100 deltas in structured status.
4. Provide immediate post-take operator feedback for obvious visibility/contact/self-collision failures, without changing acceptance thresholds.
5. Keep operational errors separate from ordinary pass/fail in APIs and dashboards.

---

## Manifest-level affected session lists

### Camera writer-overrun sessions (16)

```text
1788292502532597077
1788295140224929937
1788296068056201046
1788296387170605187
1788297153502848688
1788299932322409337
1788302785749328338
1788311239552757360
1788312417599970793
1788376769892686684
1788377474967358571
1788380629970358941
1788464964195070378
1788465457184896842
1788469936787212610
1788477343873839244
```

### UMI1 camera `ENOSPC` sessions (16)

```text
1788303131061808813
1788303242537657311
1788303308584816022
1788303399997089691
1788477808484237176
1788478002532871610
1788478199950011522
1788478339503544022
1788478452407639401
1788478694429174534
1788478817590441904
1788478928614985001
1788479066127848040
1788479269843743920
1788479396381200456
1788479545118106034
```

At the snapshot, terminal disposition was:

- 25 dataset-ineligible/skipped receipts: 16 writer-overrun and 9 `ENOSPC`.
- 2 clock-synchronization skips.
- 7 `ENOSPC`-affected sessions still queued/retrying.
- 1 genuine GLIM failure still queued/retrying.
- 3 unmatched raw arms.

The writer-overrun, frame-drop, LiDAR-drop, and `ENOSPC` categories can overlap. The counts above are classifications, not a sum of unique failed takes.

## Prioritized implementation plan

### P0 — Stop creating new preventable failures

1. **Capacity admission:** add device/server projected-space checks and enforce them in `EpisodeRecorder.start` and coordinator prepare.
2. **Streaming uploads:** remove the requirement for a full temporary tar on the Pi.
3. **Capture/maintenance lease:** make cleanup impossible during a take and make archive helpers singleton/resumable.
4. **Terminal pipeline classification:** skip expensive stages for known-ineligible input and stop infinite retries.

These four changes directly break the largest failure cycle: low disk -> corrupted capture -> blocked upload -> more low disk -> infinite server retry.

### P1 — Make paired recording transactional

1. Version the coordinator protocol.
2. Add build/task/config identity to the live protocol.
3. Implement PREPARE/COMMIT/ABORT with idempotent command IDs.
4. Add a paired lease and typed fatal-stop propagation.
5. Add the cross-Pi clock gate before commit.

### P2 — Make throughput degrade safely

1. Add server admission/backpressure status.
2. Add projected-space stage scheduling and low-disk drain mode.
3. Add runner pause/drain ownership for retention.
4. Apply systemd/cgroup CPU and I/O isolation to capture, upload, pipeline, and maintenance.

### P3 — Make diagnosis and quality improvement systematic

1. Standardize status schemas and readiness evaluation.
2. Add structured quality correlations and trend alerts.
3. Add viewer contract tests and distinguish sleeping from failed.
4. Build a fault-injection suite around network loss, disk exhaustion, dropped acknowledgements, process restart, and power loss.

## Proposed new interfaces

### Coordinator status envelope

```json
{
  "protocol_version": 2,
  "build_git_sha": "...",
  "config_sha256": "...",
  "task": "pour water",
  "recorder_generation": 42,
  "state": "idle|prepared|recording|stopping|recovering|failed",
  "current_session_id": null,
  "free_bytes": 0,
  "projected_free_after_take_bytes": 0,
  "preflight": {"ok": true, "checks": []},
  "clock": {"offset_ns": 0, "uncertainty_ns": 0, "reference": "..."}
}
```

### Persistent pipeline failure receipt

```json
{
  "stage": "decode|glim|align|gripper|dataset|publication",
  "scope": "umi1|umi2|pair",
  "classification": "transient|deterministic_input|algorithmic|terminal",
  "attempts": 1,
  "first_failed_at": "...",
  "last_failed_at": "...",
  "next_retry_at": null,
  "source_manifest_sha256": "...",
  "build_git_sha": "...",
  "error_code": "...",
  "message": "..."
}
```

## Acceptance criteria

The remediation is complete only when all of the following are demonstrated:

1. A take cannot start if either Pi or the server would cross its post-take reserve.
2. A completed episode can upload without allocating an episode-sized temporary file on the Pi.
3. Cleanup cannot begin while either arm is recording, even if the caller omits the guard.
4. An interrupted cleanup is safely resumable and a concurrent invocation cannot collide.
5. A stale build/task/config mismatch is detected from the running process and blocks prepare.
6. A command retry or dropped acknowledgement cannot start or stop either arm twice.
7. Any one-arm fatal capture event produces one graceful paired stop and preserved evidence.
8. Dataset-ineligible and clock-invalid pairs never enter decode or GLIM.
9. Deterministic and algorithmic failures reach a durable terminal state after bounded work.
10. A runner restart does not reset retry history.
11. Low server space applies backpressure before recording, and the pipeline enters drain mode without HTTP 507.
12. Readiness counters distinguish active, queued-by-next-stage, terminal skipped, and publication state correctly.
13. A fault-injection run covering disk exhaustion, network loss, stale process, restart, duplicate command, and power loss produces zero silent data loss and zero indefinite retries.

## Files expected to change

Core implementation:

- `common/umi-flow-core/src/umi_flow_core/config.py`
- `common/umi-flow-core/src/umi_flow_core/manifest.py`
- a new shared status/protocol schema under `common/umi-flow-core/src/umi_flow_core/`
- `device/src/umi_flow_device/preflight.py`
- `device/src/umi_flow_device/episode.py`
- `device/src/umi_flow_device/uploader.py`
- `device/src/umi_flow_device/coordinator.py`
- `device/src/umi_flow_device/camera.py`
- `device/src/umi_flow_device/lidar_monitor.py`
- `device/scripts/record.py`
- `device/scripts/archive_sent_raw.py`
- `server/src/umi_flow_server/server.py`
- `server/src/umi_flow_server/runner.py`
- `server/src/umi_flow_server/pair_stage.py`
- `server/src/umi_flow_server/retention.py`
- `server/scripts/archive_published_raw.py`
- systemd units under `config/systemd/`

Primary tests:

- `device/tests/test_preflight.py`
- `device/tests/test_episode.py`
- `device/tests/test_uploader.py`
- `device/tests/test_coordinator.py`
- `device/tests/test_record.py`
- `device/tests/test_camera.py`
- `device/tests/test_lidar_monitor.py`
- `server/tests/test_receiver.py`
- `server/tests/test_runner.py`
- `server/tests/test_pair_stage.py`
- `server/tests/test_retention.py`
- `server/tests/test_device_integration.py`

## Guardrails during implementation

- Do not weaken calibration, IK, quality, or clock thresholds to make error counts look better.
- Do not automatically requeue already-appended dataset episodes.
- Do not run concurrent dataset writers.
- Do not delete unmatched, incomplete, unverified, or historical raw data as part of these code fixes.
- Keep `artem-fold-towel` data and counters isolated from `artem-pour-water` readiness.
- Preserve raw evidence and terminal receipts for every failed take.
- Deploy protocol changes with an explicit compatibility gate; an old and new recorder must refuse to record together rather than guessing.

## Implementation status: `dual-plus-long-term-use-patches`

The first long-term hardening implementation preserves the existing architecture: Pi recorder and
uploader processes, filesystem episode/receipt state, the newline-delimited coordinator socket,
HTTP ingest, and the filesystem pipeline runner all remain in place. The changes add explicit
admission, transactional state, durable failure records, and cross-process leases at those existing
boundaries.

Implemented and locally tested:

1. Recording storage admission is enforced both in preflight and again in
   `EpisodeRecorder.start`; policy values are validated environment settings with conservative
   defaults.
2. Both Pi preflights query a structured server admission endpoint. Production ingest reserves 64
   GiB plus the estimated incoming take before permitting a start or accepting a stream.
3. Production upload emits a bounded-memory, HTTP-chunked tar stream directly from the completed
   episode. It creates no episode-sized Pi staging copy and moves the source to `sent` only after
   successful tar production and server acceptance.
4. Capture and Pi maintenance share a nonblocking `flock`. Pi cleanup also checks local capture
   children and requires empty complete/staging queues before hashing.
5. Pipeline stages share a read lease; server archive maintenance requires the exclusive form. This
   makes stage drain and archive exclusion an owned invariant rather than a monitor convention.
6. Both archive helpers use unique temporary names, file and directory `fsync`, durable intent, and
   resumable post-delete receipt completion. Mixed partial-deletion states fail closed.
7. Coordinator protocol v2 includes command IDs, recorder-process generations, protocol
   compatibility rejection, live task/config/source identity, structured status, reply replay, and
   same-ID transport retry.
8. Paired start is `PREPARE_START -> local start -> COMMIT_START`, with `ABORT_START` on local
   failure and local cancellation on remote commit failure. Prepared reservations expire.
9. Runner retry state is persisted atomically with timestamps, classification, source-manifest
   digest, runner digest, attempt count, and next retry time. Known-ineligible/PTP/clock-invalid
   episodes skip expensive stages immediately. Retryable resource failures remain retryable;
   deterministic-input and algorithmic failures reach a terminal dataset receipt after bounded
   attempts.

Still required before deployment is considered complete:

1. Deploy both Pis together because protocol v2 intentionally rejects an old/new mixed pair.
2. Run the soft evaluation with idle precheck, prepare/abort, prepare/commit, paired stop/cancel,
   dropped-link recovery, low-space rejection, and a small streamed upload.
3. Run a hardware capture/upload soak test and verify camera writer depth, frame drops, and LiDAR
   kernel drops remain clean. Hardware-specific fatal-event propagation and peer lease tuning should
   be based on this evidence.
4. Re-enable the paused health automation only after the new server and both recorder processes are
   deployed and the soft evaluation passes.

## Live readiness incident: 2026-09-08 23:07 UTC

### Failure mode

Both Pis were reachable, idle, and running the intended `pour water` task with protocol v2 code,
but the parsed UMI1-to-UMI2 follower precheck returned `ok=false`. LiDAR time sync, gripper tags,
and Pi storage checks all passed. The only failed structured check was `server_admission` with
`insufficient server storage headroom`.

The ingest admission response supplied the exact cause:

- server free: `83,428,114,432` bytes (77.7 GiB);
- server reserve, including extraction headroom: `77,309,411,328` bytes (72 GiB);
- conservative paired upload estimate: `8,589,934,592` bytes (8 GiB);
- required free space: `85,899,345,920` bytes (80 GiB);
- deficit: `2,471,231,488` bytes (2.3 GiB).

This was a safe fail-closed event, not a malformed coordinator response. The precheck result parsed
successfully with `PipelineResult.from_dict`, and both required sensor results were valid. There
were no queued Pi uploads, no ingest temporary files, no active UMI pipeline stage, no current HTTP
507, no publication queue, no ledger mismatch, and no operational quality error.

### Snapshot counters before repair

- Pi state: UMI1 and UMI2 idle; recorder and upload-tunnel services active on both.
- Pi free: UMI1 `44,480,532,480` bytes (41.4 GiB); UMI2 `43,050,045,440` bytes (40.1 GiB).
- Pi upload queues: complete `0`, staging `0`, actively growing `0` on both.
- Pipeline active: `0` confirmed stage workers/containers.
- Queued by next stage: none.
- Dataset: 474 local episodes, 474 ledger entries, 510 completed dataset-stage receipts, including
  36 terminal dataset-ineligible/skipped receipts; publication queue `0`.
- Quality: 472 evaluated, 269 passed, 203 failed, 2 processing, 0 pending, 0 operational errors.
  Latest-100 failure rate was 28%, down from 41% in the preceding 100; no trend alert fired.
- Three preserved unmatched raw arms remain from previously audited incidents. They were not
  modified or counted as publication work.
- One stale decode staging directory remains preserved; it is not active and was not removed
  because unverified cleanup is outside the authorized repair policy.

### Repair and prevention

The immediate authorized repair is to run normal retention while the pipeline is idle, then rerun
the same structured admission and follower-precheck gates. Verified raw archival is not authorized
at this free-space level because the configured archive trigger is below 64 GiB.

The code should also eliminate the threshold mismatch exposed here: monitoring and proactive
retention should use the ingest endpoint's returned `reserved_bytes + estimated_upload_bytes`, not
a separate fixed 64 GiB readiness threshold. The structured precheck failure message should retain
the returned `free_bytes`, `reserved_bytes`, and `estimated_upload_bytes` so operators can diagnose
the deficit without a second probe. This does not change the architecture or lower any reserve; it
makes the existing admission contract the single source of truth for alerting and maintenance.

### Repair outcome

Normal retention completed successfully with exit status 0. It raised server free space from
`83,428,052,992` bytes to `93,952,876,544` bytes (87.5 GiB), recovering about 9.8 GiB without raw
archival or modification of historical/unmatched evidence.

The post-repair ingest response returned `accepted=true` for the full 8 GiB paired estimate with an
empty ingest queue. A fresh UMI1 local preflight and a fresh UMI1-to-UMI2 follower precheck both
passed `lidar_time_sync`, `gripper_tags`, `storage_headroom`, and `server_admission`; the follower
acknowledgement parsed successfully with `PipelineResult.from_dict`. The three latest completed
takes on both Pis were also dataset-eligible with zero camera sequence drops, zero camera writer
queue overruns, no camera errors, and zero LiDAR kernel drops.

**Resolved:** the system returned to `READY TO RECORD NOW: YES` without restarting a recorder,
interrupting a take, lowering a reserve, deleting raw evidence, or changing a quality threshold.

### Local prevention patch

Implemented on `dual-plus-long-term-use-patches` after the live repair; deployment is pending:

- `/health/admission` now returns `required_free_bytes`, calculated by the same shared server
  function used to decide acceptance.
- Pi preflight failures preserve `free_bytes` and `required_free_bytes` in the operator-visible
  message. They derive the requirement from `reserved_bytes + estimated_upload_bytes` when talking
  to an older server, retaining rolling-deployment compatibility.
- Normal retention reports `admission_ready`, `free_bytes`, and `required_free_bytes` using the
  server's admission calculation instead of a separate 64 GiB interpretation.
- The existing idempotent, lease-protected retention timer checks every five minutes instead of
  every fifteen, reducing the interval in which eligible derived data can keep recording admission
  unnecessarily blocked. Retention age, recording reserves, raw archival rules, and pipeline
  architecture are unchanged.

Validation: all 75 device tests and 127 server tests passed; two hardware-dependent server tests
were skipped. Ruff passed on every changed Python file.
