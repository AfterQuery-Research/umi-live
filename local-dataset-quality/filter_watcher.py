#!/usr/bin/env python3
"""Continuously evaluate the active local UMI dataset with the latest IK gate.

This is deliberately a sidecar.  It only reads the production LeRobot dataset and
writes results below its own local results directory; it never changes the recording
pipeline or uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FILTERING_ROOT = ROOT / "filtering"
sys.path.insert(0, str(FILTERING_ROOT / "src"))

from dataset_filtering.combined_pipeline import (  # noqa: E402
    PIPELINE_VERSION,
    CombinedWorkItem,
    _process_work_item,
)
from dataset_filtering.hf_dataset import LeRobotSource  # noqa: E402

STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((FILTERING_ROOT / "src" / "dataset_filtering").rglob("*.py")):
        digest.update(str(path.relative_to(FILTERING_ROOT)).encode())
        digest.update(path.read_bytes())
    for distribution in ("casadi", "numpy", "scipy", "toppra"):
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = "missing"
        digest.update(f"{distribution}={installed}".encode())
    return digest.hexdigest()[:16]


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def active_dataset(repo_root: Path) -> tuple[Path, str]:
    env = _read_env(repo_root / ".env")
    data_dir = Path(env.get("DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    return (data_dir / "lerobot" / "local" / "umi-flow").resolve(), env.get(
        "HF_REPO_ID", ""
    )


def dataset_key(dataset: Path) -> str:
    stat = dataset.stat()
    identity = f"{dataset}|{stat.st_dev}|{stat.st_ino}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def result_directory(results_root: Path, dataset: Path, code: str) -> Path:
    return results_root / dataset_key(dataset) / code / "episodes"


def _run_work_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    # Every task gets fresh metadata.  A long-lived process-pool worker must not
    # retain LeRobotSource's episode index after a new recording is appended.
    import dataset_filtering.combined_pipeline as pipeline

    # The parent handles a graceful stop; workers use the default termination
    # behavior so a service restart does not wait for an in-progress IPOPT solve.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    pipeline._WORKER_SOURCES.clear()
    return _process_work_item(raw_item)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_rows(result_dir: Path, dataset: Path, code: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in result_dir.glob("input-*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            index = int(row["source_episode_index"])
            if (
                Path(row["source_dataset"]).resolve() != dataset
                or row.get("source_revision") != f"local-{code}"
                or row.get("pipeline_version") != PIPELINE_VERSION
            ):
                continue
            if row.get("accepted") and not path.with_suffix(".npz").is_file():
                continue
            rows[index] = row
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return rows


def _failure_reasons(row: dict[str, Any]) -> list[str]:
    selected = row.get("selected_result") or {}
    validation = selected.get("validation") or {}
    return sorted(
        key.removeprefix("passes_")
        for key, value in validation.items()
        if key.startswith("passes_") and not value
    )


def status_payload(
    *,
    dataset: Path,
    hub_repo: str,
    episode_indices: list[int],
    rows: dict[int, dict[str, Any]],
    processing: set[int],
    errors: dict[int, str],
    workers: int,
    code: str,
) -> dict[str, Any]:
    episodes = []
    for index in episode_indices:
        row = rows.get(index)
        if row is not None:
            state = "pass" if row.get("accepted") else "fail"
        elif index in processing:
            state = "processing"
        elif index in errors:
            state = "error"
        else:
            state = "pending"
        episodes.append(
            {
                "episode_index": index,
                "state": state,
                "selected_profile": row.get("selected_profile") if row else None,
                "failure_reasons": _failure_reasons(row) if row else [],
                "error": errors.get(index),
            }
        )
    passed = [item["episode_index"] for item in episodes if item["state"] == "pass"]
    failed = [item["episode_index"] for item in episodes if item["state"] == "fail"]
    return {
        "dataset": str(dataset),
        "hub_repo": hub_repo,
        "dataset_total": len(episode_indices),
        "evaluated": len(passed) + len(failed),
        "passed": len(passed),
        "failed": len(failed),
        "processing": sum(item["state"] == "processing" for item in episodes),
        "pending": sum(item["state"] == "pending" for item in episodes),
        "errors": sum(item["state"] == "error" for item in episodes),
        "passed_episode_indices": passed,
        "failed_episode_indices": failed,
        "episodes": episodes,
        "pipeline_version": PIPELINE_VERSION,
        "code_fingerprint": code,
        "workers": workers,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def waiting_payload(message: str, workers: int, code: str) -> dict[str, Any]:
    return {
        "dataset_total": 0,
        "evaluated": 0,
        "passed": 0,
        "failed": 0,
        "processing": 0,
        "pending": 0,
        "errors": 1,
        "episodes": [],
        "pipeline_version": PIPELINE_VERSION,
        "code_fingerprint": code,
        "workers": workers,
        "watcher_error": message,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def run(repo_root: Path, results_root: Path, workers: int, poll_seconds: float) -> None:
    code = _code_fingerprint()
    status_path = results_root / "status.json"
    inflight: dict[Future[dict[str, Any]], tuple[Path, int]] = {}
    retry_after: dict[tuple[Path, int], float] = {}
    errors: dict[tuple[Path, int], str] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while not STOP:
            for future, (future_dataset, episode_index) in list(inflight.items()):
                if not future.done():
                    continue
                del inflight[future]
                key = (future_dataset, episode_index)
                try:
                    future.result()
                    errors.pop(key, None)
                    retry_after.pop(key, None)
                    print(f"quality gate: episode {episode_index} complete", flush=True)
                except Exception as exc:
                    errors[key] = f"{type(exc).__name__}: {exc}"
                    retry_after[key] = time.monotonic() + 30
                    print(
                        f"quality gate: episode {episode_index} failed to evaluate: {exc}",
                        flush=True,
                    )

            try:
                dataset, hub_repo = active_dataset(repo_root)
                if not dataset.is_dir():
                    payload = status_payload(
                        dataset=dataset,
                        hub_repo=hub_repo,
                        episode_indices=[],
                        rows={},
                        processing=set(),
                        errors={},
                        workers=workers,
                        code=code,
                    )
                    payload["waiting_for_first_episode"] = True
                    _atomic_json(status_path, payload)
                    time.sleep(poll_seconds)
                    continue
                source = LeRobotSource(str(dataset), revision=f"local-{code}")
                info = source.info()
                summaries = source.episode_summaries()
                expected = int(info["total_episodes"])
                if len(summaries) != expected:
                    raise RuntimeError(
                        f"dataset append in progress: info has {expected} episodes but "
                        f"metadata has {len(summaries)}"
                    )
                indices = sorted(summary.episode_index for summary in summaries)
                result_dir = result_directory(results_root, dataset, code)
                result_dir.mkdir(parents=True, exist_ok=True)
                rows = _load_rows(result_dir, dataset, code)
                active_processing = {
                    index
                    for future, (item_dataset, index) in inflight.items()
                    if item_dataset == dataset and not future.done()
                }
                active_errors = {
                    index: message
                    for (item_dataset, index), message in errors.items()
                    if item_dataset == dataset
                }

                available_slots = max(workers * 2 - len(inflight), 0)
                pending = [
                    index
                    for index in reversed(indices)
                    if index not in rows
                    and index not in active_processing
                    and time.monotonic() >= retry_after.get((dataset, index), 0)
                ]
                for index in pending[:available_slots]:
                    item = CombinedWorkItem(
                        combined_input_index=index,
                        source_dataset=str(dataset),
                        source_revision=f"local-{code}",
                        source_episode_index=index,
                        cache_dir=str(results_root / "cache"),
                        result_dir=str(result_dir),
                    )
                    future = pool.submit(_run_work_item, vars(item))
                    inflight[future] = (dataset, index)
                    active_processing.add(index)
                    active_errors.pop(index, None)
                    print(f"quality gate: queued episode {index}", flush=True)

                rows = _load_rows(result_dir, dataset, code)
                payload = status_payload(
                    dataset=dataset,
                    hub_repo=hub_repo,
                    episode_indices=indices,
                    rows=rows,
                    processing=active_processing,
                    errors=active_errors,
                    workers=workers,
                    code=code,
                )
            except Exception as exc:
                payload = waiting_payload(f"{type(exc).__name__}: {exc}", workers, code)
            _atomic_json(status_path, payload)
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and score the active UMI dataset")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run(
        args.repo_root.expanduser().resolve(),
        args.results.expanduser().resolve(),
        args.workers,
        args.poll_seconds,
    )


if __name__ == "__main__":
    main()
