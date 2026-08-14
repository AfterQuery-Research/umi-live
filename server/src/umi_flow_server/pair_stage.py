"""Server-side pair merge for the fixed two-client bimanual system.

In the two-Pi topology each Pi records ONE arm (leader = role ``umi1``, follower =
role ``umi2``) as an independent arm transport episode; the two upload separately, each
tagged with a shared ``session_id``. This pre-pass buckets ingested episodes by
``session_id`` and, once both arms are present, fabricates ONE combined episode in
``raw_episodes/`` that the runner and bimanual dataset stage consume — the
follower is provisioned natively as ``umi2`` so this is a pure copy of both arms' streams
plus a synthesized 2-UMI config + merged manifest (no file renames).

The two source episodes are then relocated OUT of ``raw_episodes/`` (into
``paired_sources/``) so a lone arm is never processed alone into a stray 6-D dataset.
The runner always skips any still-unpaired arm until this merge consumes it or the
orphan timeout fires.

Idempotent: the merged dir is named deterministically from the session; a crash between
writing it and relocating the sources is repaired on the next pass.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from umi_flow_core import config as core_config
from umi_flow_core import layout, manifest

LEADER_ROLE = "umi1"
FOLLOWER_ROLE = "umi2"
PAIRED_SOURCES_DIR = "paired_sources"
ORPHANS_DIR = "orphans"
FAILED_PAIRS_DIR = "failed_pairs"
DEFAULT_TIMEOUT_S = 600.0


class PairError(ValueError):
    """A pair cannot be merged (misprovisioned arms, role/key mismatch)."""


@dataclass
class PairResult:
    merged: list[str] = field(default_factory=list)  # merged episode ids produced/confirmed
    orphaned: list[str] = field(default_factory=list)  # source ids timed out without a mate
    pending: list[str] = field(default_factory=list)  # session ids still waiting for a mate
    failed: list[str] = field(default_factory=list)  # session ids quarantined (bad merge)


# --- scanning -------------------------------------------------------------------------


def _read_manifest(episode_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((episode_dir / layout.MANIFEST_FILE).read_text())
    except (ValueError, OSError):
        return None


def scan_sources(raw_root: Path) -> dict[str, dict[str, Path]]:
    """``{session_id: {role: episode_dir}}`` over COMPLETE, role-tagged raw episodes."""
    out: dict[str, dict[str, Path]] = {}
    if not raw_root.exists():
        return out
    for ep in sorted(raw_root.iterdir()):
        if not ep.is_dir() or not (ep / layout.COMPLETE_FILE).exists():
            continue
        m = _read_manifest(ep)
        if m is None:
            continue
        sid, role = m.get("session_id"), m.get("role")
        if sid is None or role is None:
            continue
        out.setdefault(sid, {})[role] = ep
    return out


# --- merge ----------------------------------------------------------------------------


def _snapshot_yaml_path(episode_dir: Path) -> Path:
    return episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML


def _copy_streams(src_dir: Path, dest_dir: Path, m: dict[str, Any]) -> None:
    """Copy the stream files this arm's manifest lists (umiN_camera.*, umiN_lidar.*)."""
    for entry in m.get("sensors", {}).values():
        for fname in entry.get("files", {}).values():
            src = src_dir / fname
            if src.is_file():
                shutil.copy2(src, dest_dir / fname)


def _synthesize_config(leader_dir: Path, follower_dir: Path, dest_dir: Path) -> None:
    """Merge the two arm snapshots into one bimanual config and copy calibrations.

    Requires each arm to be provisioned NATIVELY (leader's only UMI is ``umi1``, the
    follower's is ``umi2``) so keys are disjoint and no stream renaming is needed."""
    leader_yaml = yaml.safe_load(_snapshot_yaml_path(leader_dir).read_text())
    follower_yaml = yaml.safe_load(_snapshot_yaml_path(follower_dir).read_text())
    lkeys, fkeys = list(leader_yaml.get("umis", {})), list(follower_yaml.get("umis", {}))
    if lkeys != [LEADER_ROLE]:
        raise PairError(f"leader must be a single '{LEADER_ROLE}' rig, got umis {lkeys}")
    if fkeys != [FOLLOWER_ROLE]:
        raise PairError(f"follower must be a single '{FOLLOWER_ROLE}' rig, got umis {fkeys}")

    leader_umi = leader_yaml["umis"][LEADER_ROLE]
    follower_umi = follower_yaml["umis"][FOLLOWER_ROLE]
    merged = dict(leader_yaml)
    merged["umis"] = {LEADER_ROLE: leader_umi, FOLLOWER_ROLE: follower_umi}
    # Validate before writing so a bad pair fails here, not in the dataset stage.
    core_config.parse_rig_config(merged, source="<pair-merge>")

    snap = dest_dir / layout.CONFIG_SNAPSHOT_DIR
    snap.mkdir(parents=True, exist_ok=True)
    (snap / layout.CONFIG_SNAPSHOT_YAML).write_text(yaml.safe_dump(merged, sort_keys=False))

    merged_rig = core_config.parse_rig_config(merged)
    for rel in core_config.referenced_blobs(merged_rig):
        for src_dir in (leader_dir, follower_dir):
            src = src_dir / layout.CONFIG_SNAPSHOT_DIR / rel
            if src.is_file():
                dst = snap / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                break


def _merge_clock_sync(lm: dict, fm: dict) -> dict[str, Any]:
    """Fold the two per-Pi clock_sync blocks into the merged one, deriving the cross-Pi
    offset estimate that ``clock_sync_acceptable`` gates on."""
    lcs, fcs = lm.get("clock_sync"), fm.get("clock_sync")
    if not lcs or not fcs:
        # Missing clock evidence from either arm must fail the paired dataset gate.
        return manifest.ClockSyncStatus(synced=False, ref="").to_dict()
    lo, fo = lcs.get("offset_to_ref_ns"), fcs.get("offset_to_ref_ns")
    # Both offsets are SIGNED (same convention on both Pis), so leader - follower is the
    # true signed cross-Pi skew (see manifest.ClockSyncStatus).
    est = (lo - fo) if (lo is not None and fo is not None) else None
    return manifest.ClockSyncStatus(
        synced=bool(lcs.get("synced") and fcs.get("synced")),
        ref=str(lcs.get("ref", "")),
        offset_to_ref_ns=None,  # per-Pi values live on the relocated sources
        handshake_offset_ns=lcs.get("handshake_offset_ns"),
        estimated_cross_pi_offset_ns=est,
    ).to_dict()


def _synthesize_manifest(leader_dir: Path, follower_dir: Path, dest_dir: Path, sid: str) -> None:
    lm, fm = _read_manifest(leader_dir), _read_manifest(follower_dir)
    assert lm is not None and fm is not None  # scan_sources already read them
    merged: dict[str, Any] = {
        "manifest_version": manifest.MANIFEST_VERSION,
        # Never leak the private atomic-staging name into the durable data contract.
        "episode_id": f"paired_{sid}",
        "device_id": "paired",
        "task": lm.get("task", ""),
        # umi1/* from the leader + umi2/* from the follower -> the runner sees both scopes.
        "sensors": {**lm.get("sensors", {}), **fm.get("sensors", {})},
        # re-key each Pi's ptp so the two eth0s don't collide; lidar_ptp_acceptable
        # then requires BOTH arms' intra-Pi PTP to be locked.
        "ptp": {
            **{f"{LEADER_ROLE}:{k}": v for k, v in lm.get("ptp", {}).items()},
            **{f"{FOLLOWER_ROLE}:{k}": v for k, v in fm.get("ptp", {}).items()},
        },
        "config_sha256": layout.sha256_file(_snapshot_yaml_path(dest_dir)),
        "start_unix_ns": min(lm.get("start_unix_ns", 0), fm.get("start_unix_ns", 0)),
        "stop_unix_ns": max(lm.get("stop_unix_ns", 0), fm.get("stop_unix_ns", 0)),
        "dataset_eligible": bool(lm.get("dataset_eligible") and fm.get("dataset_eligible")),
        "session_id": sid,
        "files": {},  # the runner never re-hashes; this internal artifact skips checksums
    }
    merged["clock_sync"] = _merge_clock_sync(lm, fm)
    problems = manifest.validate_manifest(merged)
    if problems:
        raise PairError(f"merged manifest invalid: {problems}")
    manifest.write_manifest(dest_dir / layout.MANIFEST_FILE, merged)


def merge_pair(leader_dir: Path, follower_dir: Path, dest_dir: Path, sid: str) -> None:
    """Fabricate the combined bimanual episode at ``dest_dir`` (which must not exist)."""
    dest_dir.mkdir(parents=True)
    _synthesize_config(leader_dir, follower_dir, dest_dir)
    _copy_streams(leader_dir, dest_dir, _read_manifest(leader_dir) or {})
    _copy_streams(follower_dir, dest_dir, _read_manifest(follower_dir) or {})
    _synthesize_manifest(leader_dir, follower_dir, dest_dir, sid)
    (dest_dir / layout.COMPLETE_FILE).write_text("complete\n")


# --- orchestration --------------------------------------------------------------------


def _relocate(ep: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / ep.name
    if target.exists():
        shutil.rmtree(ep, ignore_errors=True)  # already relocated on a prior pass
        return
    shutil.move(str(ep), str(target))


def _merge_into(leader: Path, follower: Path, dest: Path, sid: str) -> None:
    """Atomic merge: build in a staging dir, then rename — dest always has COMPLETE.
    A failure leaves no partial dest and no leftover staging dir."""
    staging = dest.parent / f".{dest.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        merge_pair(leader, follower, staging, sid)
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def pair_once(
    data_dir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    now_ns: int | None = None,
) -> PairResult:
    """One pairing pass over ``data_dir/raw_episodes``. Merges complete pairs and relocates
    their sources; quarantines a pair whose merge fails (so it neither crashes the loop nor
    wedges); drops a lone arm that times out without a mate into ``orphans/``."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    raw_root = data_dir / "raw_episodes"
    result = PairResult()
    for sid, arms in sorted(scan_sources(raw_root).items()):
        dest = raw_root / f"paired_{sid}"
        if LEADER_ROLE in arms and FOLLOWER_ROLE in arms:
            try:
                if not dest.exists():
                    _merge_into(arms[LEADER_ROLE], arms[FOLLOWER_ROLE], dest, sid)
            except (PairError, core_config.ConfigError, OSError, yaml.YAMLError) as exc:
                # A bad/corrupt/misprovisioned pair must neither crash the runner loop nor
                # wedge forever in raw_episodes (both arms present => the orphan-timeout
                # never fires). Quarantine both arms so processing continues; an operator
                # can inspect failed_pairs/. Deterministic failures won't retry-loop.
                print(f"pair: session {sid} merge FAILED ({exc}); quarantining sources", flush=True)
                _relocate(arms[LEADER_ROLE], data_dir / FAILED_PAIRS_DIR)
                _relocate(arms[FOLLOWER_ROLE], data_dir / FAILED_PAIRS_DIR)
                result.failed.append(sid)
                continue
            _relocate(arms[LEADER_ROLE], data_dir / PAIRED_SOURCES_DIR)
            _relocate(arms[FOLLOWER_ROLE], data_dir / PAIRED_SOURCES_DIR)
            result.merged.append(dest.name)
            print(f"pair: merged session {sid} -> {dest.name}", flush=True)
        else:
            arm = next(iter(arms.values()))
            am = _read_manifest(arm)
            if am is None:
                continue  # became unreadable since the scan; retry next pass
            age_s = (now_ns - int(am.get("stop_unix_ns", 0))) / 1e9
            if age_s > timeout_s:
                _relocate(arm, data_dir / ORPHANS_DIR)
                result.orphaned.append(arm.name)
                print(f"pair: session {sid} orphaned after {age_s:.0f}s -> {arm.name}", flush=True)
            else:
                result.pending.append(sid)
    return result
