"""HTTP upload and retry helpers for completed episodes."""

from __future__ import annotations

import http.client
import json
import queue
import shutil
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from umi_flow_core.layout import COMPLETE_FILE

# The client writes into a local SSH tunnel, so its socket can fill before the tunnel has
# delivered a large episode and received the server's response. Ten seconds caused valid
# uploads to time out and retry while their earlier tunnel streams were still draining.
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 300.0
REACHABILITY_TIMEOUT_SECONDS = 3.0
PostFile = Callable[[str, Path, str, float], dict[str, Any]]


@dataclass(frozen=True)
class UploadResult:
    episode_id: str
    success: bool
    size_bytes: int = 0
    error: str = ""
    server_response: dict[str, Any] = field(default_factory=dict)


def _parse_url(server_url: str) -> tuple[str, str, int]:
    parsed = urlparse(server_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"upload endpoint must use the local SSH tunnel: {server_url}")
    return parsed.scheme, parsed.hostname, parsed.port or 80


def _connect(server_url: str, timeout_seconds: float) -> http.client.HTTPConnection:
    """Open the loopback HTTP connection reached through the SSH tunnel."""
    scheme, host, port = _parse_url(server_url)
    return http.client.HTTPConnection(host, port, timeout=timeout_seconds)


def validate_server_url(server_url: str) -> None:
    """Reject unusable upload URLs up front, at program startup, instead of
    silently retrying them forever from the upload worker."""
    _parse_url(server_url)


def server_reachable(
    server_url: str, timeout_seconds: float = REACHABILITY_TIMEOUT_SECONDS
) -> bool:
    """Cheap TCP probe so retry loops do not rebuild multi-GB tars while the
    server is down."""
    conn = _connect(server_url, timeout_seconds)
    try:
        conn.connect()
        return True
    except OSError:
        return False
    finally:
        conn.close()


def post_tar_file(
    server_url: str,
    tar_path: Path,
    episode_id: str,
    timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    parsed = urlparse(server_url)
    conn = _connect(server_url, timeout_seconds)
    size = tar_path.stat().st_size
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    try:
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/x-tar")
        conn.putheader("Content-Length", str(size))
        conn.putheader("X-Episode-Id", episode_id)
        conn.endheaders()
        with tar_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                conn.send(chunk)
        response = conn.getresponse()
        payload = response.read()
        if response.status >= 300:
            detail = payload.decode("utf-8", errors="replace")
            raise RuntimeError(f"upload failed with HTTP {response.status}: {detail}")
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))
    finally:
        conn.close()


def upload_episode(
    episode_dir: Path,
    server_url: str,
    sent_dir: Path,
    post: PostFile = post_tar_file,
    upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
    staging_dir: Path | None = None,
) -> UploadResult:
    """Upload one episode. With a staging_dir, the tar is built once (atomic
    rename) and reused across retries instead of being rebuilt in the system
    temp dir every attempt."""
    if not (episode_dir / COMPLETE_FILE).exists():
        return UploadResult(
            episode_id=episode_dir.name,
            success=False,
            error="missing COMPLETE marker",
        )
    if staging_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / f"{episode_dir.name}.tar"
            return _upload_with_tar(
                episode_dir,
                tar_path,
                server_url,
                sent_dir,
                post,
                upload_timeout_seconds,
            )
    staging_dir.mkdir(parents=True, exist_ok=True)
    tar_path = staging_dir / f"{episode_dir.name}.tar"
    result = _upload_with_tar(
        episode_dir, tar_path, server_url, sent_dir, post, upload_timeout_seconds
    )
    if result.success:
        tar_path.unlink(missing_ok=True)
    return result


def _upload_with_tar(
    episode_dir: Path,
    tar_path: Path,
    server_url: str,
    sent_dir: Path,
    post: PostFile,
    upload_timeout_seconds: float,
) -> UploadResult:
    size_bytes = 0
    try:
        if not tar_path.exists():
            create_episode_tar(episode_dir, tar_path)
        size_bytes = tar_path.stat().st_size
        server_response = post(server_url, tar_path, episode_dir.name, upload_timeout_seconds)
    except Exception as exc:
        print(f"upload kept for retry: {episode_dir.name}: {exc}")
        return UploadResult(
            episode_id=episode_dir.name,
            success=False,
            size_bytes=size_bytes,
            error=str(exc),
        )
    sent_dir.mkdir(parents=True, exist_ok=True)
    dest = sent_dir / episode_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(episode_dir), dest)
    result = UploadResult(
        episode_id=episode_dir.name,
        success=True,
        size_bytes=size_bytes,
        server_response=server_response,
    )
    print(f"uploaded and moved to sent: {episode_dir.name} size={result.size_bytes}B")
    return result


def retry_completed(
    complete_dir: Path,
    server_url: str,
    sent_dir: Path,
    post: PostFile = post_tar_file,
    upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
) -> int:
    count = 0
    if not complete_dir.exists():
        return 0
    for episode_dir in sorted(p for p in complete_dir.iterdir() if p.is_dir()):
        if upload_episode(
            episode_dir,
            server_url,
            sent_dir,
            post=post,
            upload_timeout_seconds=upload_timeout_seconds,
        ).success:
            count += 1
    return count


def create_episode_tar(episode_dir: Path, tar_path: Path) -> None:
    """Build the tar atomically: a crash mid-build can never leave a partial
    tar that later gets reused for an upload."""
    tmp_path = tar_path.with_name(tar_path.name + ".tmp")
    try:
        with tarfile.open(tmp_path, "w") as tar:
            tar.add(episode_dir, arcname=episode_dir.name)
        tmp_path.replace(tar_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class AsyncUploader:
    def __init__(
        self,
        complete_dir: Path,
        server_url: str,
        sent_dir: Path,
        retry_seconds: float = 5.0,
        upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
        post: PostFile = post_tar_file,
    ):
        validate_server_url(server_url)
        self.complete_dir = complete_dir
        self.server_url = server_url
        self.sent_dir = sent_dir
        self.staging_dir = complete_dir.parent / "staging"
        self.retry_seconds = retry_seconds
        self.upload_timeout_seconds = upload_timeout_seconds
        self.post = post
        self.queue: queue.Queue[Path | None] = queue.Queue()
        self.results: list[UploadResult] = []
        self.results_lock = threading.Lock()
        # The episode whose tar is streaming to the server right now (None = idle).
        # Read by the recorder loop to drive the LCD "SENDING" indicator.
        self._uploading: str | None = None
        self._uploading_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, episode_dir: Path) -> None:
        self.queue.put(episode_dir)
        print(f"queued async upload: {episode_dir.name}")

    def retry_existing(self) -> None:
        if not self.complete_dir.exists():
            return
        for episode_dir in sorted(p for p in self.complete_dir.iterdir() if p.is_dir()):
            self.submit(episode_dir)

    def snapshot_results(self) -> list[UploadResult]:
        with self.results_lock:
            return list(self.results)

    def current_upload(self) -> str | None:
        """The episode id whose tar is streaming to the server right now (None = idle)."""
        with self._uploading_lock:
            return self._uploading

    def close(self, wait: bool = False) -> None:
        if wait:
            self.queue.join()
        self.stop_event.set()
        self.queue.put(None)
        self.thread.join(timeout=5)

    def _worker(self) -> None:
        next_retry = time.monotonic() + self.retry_seconds
        while not self.stop_event.is_set():
            timeout = max(0.1, next_retry - time.monotonic())
            try:
                episode_dir = self.queue.get(timeout=timeout)
            except queue.Empty:
                self._retry_scan()
                next_retry = time.monotonic() + self.retry_seconds
                continue
            try:
                if episode_dir is None:
                    return
                self._upload(episode_dir)
            finally:
                self.queue.task_done()

    def _retry_scan(self) -> None:
        if not self.complete_dir.exists():
            return
        self._clean_orphan_tars()
        pending = sorted(p for p in self.complete_dir.iterdir() if p.is_dir())
        if not pending:
            return
        if not server_reachable(self.server_url):
            return
        for episode_dir in pending:
            self._upload(episode_dir)

    def _clean_orphan_tars(self) -> None:
        if not self.staging_dir.exists():
            return
        for tar_path in self.staging_dir.glob("*.tar"):
            if not (self.complete_dir / tar_path.stem).exists():
                tar_path.unlink(missing_ok=True)

    def _upload(self, episode_dir: Path) -> None:
        if not episode_dir.exists():
            return
        with self._uploading_lock:
            self._uploading = episode_dir.name
        try:
            result = upload_episode(
                episode_dir,
                self.server_url,
                self.sent_dir,
                post=self.post,
                upload_timeout_seconds=self.upload_timeout_seconds,
                staging_dir=self.staging_dir,
            )
        except Exception as exc:
            print(f"upload kept for retry: {episode_dir.name}: {exc}")
            result = UploadResult(
                episode_id=episode_dir.name,
                success=False,
                error=str(exc),
            )
        finally:
            with self._uploading_lock:
                self._uploading = None
        with self.results_lock:
            self.results.append(result)
