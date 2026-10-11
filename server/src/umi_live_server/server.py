"""FastAPI ingest for completed Pi episodes: extract + validate + publish. Nothing else.

The ingest is a durable inbox. Its only post-upload act is the atomic publish of a
validated episode into ``data/raw_episodes/<id>/`` — all processing (decode, SLAM,
gripper, dataset) belongs to the pipeline runner and its stage containers, which
watch that directory. This keeps ROS/CUDA/lerobot dependencies out of the always-up
ingest process entirely (deps: fastapi + stdlib + core).

Durability: upload tars land in ``data/uploads/`` under a tmp name and are renamed
only when fully streamed; a crash mid-anything leaves either a ``.tmp`` (ignored) or
a whole tar that startup recovery re-processes. An episode directory only appears in
``raw_episodes/`` after full extraction + checksum validation (single atomic rename).
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from io import BufferedWriter
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from umi_live_server.episode import (
    DEFAULT_MAX_EXTRACT_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    extract_episode,
    safe_name,
    validate_episode,
)

DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_SHUTDOWN_DRAIN_SECONDS = 60.0


@dataclass(frozen=True)
class ServerConfig:
    data_dir: Path = Path("data")
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_extract_bytes: int = DEFAULT_MAX_EXTRACT_BYTES
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    recording_reserve_bytes: int = 0
    shutdown_drain_seconds: float = DEFAULT_SHUTDOWN_DRAIN_SECONDS


def admission_reserved_bytes(cfg: ServerConfig) -> int:
    """Bytes that must remain available while an accepted upload is extracted."""
    return max(cfg.min_free_bytes, cfg.recording_reserve_bytes) + cfg.max_extract_bytes


def admission_required_bytes(cfg: ServerConfig, estimated_upload_bytes: int) -> int:
    """Single source of truth for the free-space admission threshold."""
    if estimated_upload_bytes < 0:
        raise ValueError("estimated upload bytes must be non-negative")
    return admission_reserved_bytes(cfg) + estimated_upload_bytes


@dataclass(frozen=True)
class IngestJob:
    episode_id: str
    tar_path: Path


@dataclass(frozen=True)
class IngestResult:
    episode_id: str
    success: bool
    error: str = ""


class IngestWorker:
    """Extract + validate + publish, off the event loop. One episode at a time."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.queue: queue.Queue[IngestJob | None] = queue.Queue()
        self.results: list[IngestResult] = []
        self.results_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="umi-ingest", daemon=True)
        self.thread.start()

    def submit(self, job: IngestJob) -> None:
        self.queue.put(job)

    def wait_idle(self) -> None:
        self.queue.join()

    def snapshot_results(self) -> list[IngestResult]:
        with self.results_lock:
            return list(self.results)

    def close(self, wait: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.queue.put(None)
        deadline = time.monotonic() + self.cfg.shutdown_drain_seconds
        self.thread.join(timeout=max(0.1, deadline - time.monotonic() if wait else 0.1))

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                if self.stop_event.is_set():
                    continue  # leave the tar for startup recovery
                result = self._ingest(job)
                with self.results_lock:
                    self.results.append(result)
            finally:
                self.queue.task_done()

    def _ingest(self, job: IngestJob) -> IngestResult:
        episode_id = job.episode_id
        try:
            # Validation runs inside extract_episode's quarantine staging dir, so a
            # checksum failure never publishes anything into raw_episodes/ (and the
            # runner never sees a half-validated episode). See episode.py.
            episode_dir = extract_episode(
                job.tar_path,
                self.cfg.data_dir / "raw_episodes",
                expected_root=job.episode_id,
                max_extract_bytes=self.cfg.max_extract_bytes,
                validate=validate_episode,
            )
            episode_id = episode_dir.name
            print(f"ingested episode {episode_id}", flush=True)
            return IngestResult(episode_id=episode_id, success=True)
        except Exception as exc:
            print(f"episode ingest failed: {episode_id}: {exc}", flush=True)
            return IngestResult(episode_id=episode_id, success=False, error=str(exc))
        finally:
            job.tar_path.unlink(missing_ok=True)


def create_app(cfg: ServerConfig) -> FastAPI:
    app = FastAPI(title="UMI-Live Ingest")
    worker = IngestWorker(cfg)
    app.state.worker = worker

    def recover_pending() -> None:
        """Re-enqueue upload tars left behind by a crash/hard shutdown."""
        uploads = cfg.data_dir / "uploads"
        if uploads.exists():
            for tar_path in sorted(uploads.glob("*.tar")):
                episode_id = safe_name(tar_path.stem.rsplit("_", 1)[0])
                worker.submit(IngestJob(episode_id=episode_id, tar_path=tar_path))

    app.router.add_event_handler("startup", recover_pending)
    app.router.add_event_handler("shutdown", lambda: worker.close(wait=True))

    @app.get("/health/admission")
    async def admission(estimated_upload_bytes: int = 0) -> dict[str, object]:
        if estimated_upload_bytes < 0 or estimated_upload_bytes > 2 * cfg.max_upload_bytes:
            raise HTTPException(status_code=400, detail="invalid estimated upload size")
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cfg.data_dir).free
        queued_bytes = 0
        uploads = cfg.data_dir / "uploads"
        if uploads.exists():
            queued_bytes = sum(
                path.stat().st_size
                for path in uploads.iterdir()
                if path.is_file() and (path.suffix == ".tar" or path.name.endswith(".tmp"))
            )
        reserve = admission_reserved_bytes(cfg)
        required = admission_required_bytes(cfg, estimated_upload_bytes)
        accepted = free >= required
        return {
            "ok": True,
            "accepted": accepted,
            "reason": "" if accepted else "insufficient server storage headroom",
            "free_bytes": free,
            "reserved_bytes": reserve,
            "estimated_upload_bytes": estimated_upload_bytes,
            "required_free_bytes": required,
            "queued_upload_bytes": queued_bytes,
            "ingest_queue_depth": worker.queue.qsize(),
            "max_upload_bytes": cfg.max_upload_bytes,
            "retry_after_seconds": 0 if accepted else 60,
        }

    @app.post("/upload", status_code=202)
    async def upload(
        request: Request,
        x_episode_id: str | None = Header(default=None),
    ) -> dict[str, object]:
        episode_id = safe_name(x_episode_id or "episode")
        upload_dir = cfg.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_name = f"{episode_id}_{uuid.uuid4().hex}.tar"
        tar_path = upload_dir / upload_name
        tmp_path = upload_dir / f".{upload_name}.tmp"
        bytes_written = 0
        renamed = False
        try:
            f: BufferedWriter = tmp_path.open("wb")
            try:
                async for chunk in request.stream():
                    bytes_written += len(chunk)
                    if bytes_written > cfg.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="upload exceeds maximum size")
                    # Extraction temporarily needs space for both the completed tar and
                    # its unpacked contents. Preserve that headroom plus an OS reserve.
                    required_free = (
                        max(cfg.min_free_bytes, cfg.recording_reserve_bytes)
                        + cfg.max_extract_bytes
                        + len(chunk)
                    )
                    if shutil.disk_usage(upload_dir).free < required_free:
                        raise HTTPException(
                            status_code=507,
                            detail="insufficient server disk space for upload and extraction",
                        )
                    await run_in_threadpool(f.write, chunk)
            finally:
                f.close()
            if bytes_written == 0:
                raise HTTPException(status_code=400, detail="empty upload")
            await run_in_threadpool(tmp_path.replace, tar_path)
            renamed = True
            worker.submit(IngestJob(episode_id=episode_id, tar_path=tar_path))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if not renamed:
                tmp_path.unlink(missing_ok=True)
        return {
            "ok": True,
            "status": "queued",
            "episode_id": episode_id,
            "upload_size_bytes": bytes_written,
            "queue_depth": worker.queue.qsize(),
        }

    return app
