from __future__ import annotations

import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

import pytest
import umi_flow_device.uploader as uploader_module
from umi_flow_core.layout import COMPLETE_FILE
from umi_flow_device.uploader import (
    DEFAULT_UPLOAD_TIMEOUT_SECONDS,
    AsyncUploader,
    post_tar_file,
    retry_completed,
    upload_episode,
    validate_server_url,
)


def make_episode(path: Path, name: str = "episode_1") -> Path:
    episode = path / name
    episode.mkdir()
    (episode / "manifest.json").write_text("{}\n")
    (episode / "camera_frames.jsonl").write_text("{}\n")
    (episode / "camera_stats.json").write_text("{}\n")
    (episode / COMPLETE_FILE).write_text("complete\n")
    return episode


def test_upload_failure_leaves_episode_in_complete(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)

    def fail(_url: str, _tar_path: Path, _episode_id: str, _timeout_seconds: float) -> dict:
        raise RuntimeError("down")

    assert not upload_episode(episode, "http://127.0.0.1:8080/upload", sent, post=fail).success
    assert episode.exists()
    assert not (sent / episode.name).exists()


def test_default_upload_streams_tar_without_staging_copy(tmp_path: Path, monkeypatch):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)
    captured = BytesIO()

    class Response:
        status = 202

        @staticmethod
        def read() -> bytes:
            return b'{"status":"queued"}'

    class Connection:
        def request(self, method, path, body, headers, encode_chunked):
            assert (method, path, encode_chunked) == ("POST", "/upload", True)
            assert "Content-Length" not in headers
            while chunk := body.read(8192):
                captured.write(chunk)

        @staticmethod
        def getresponse():
            return Response()

        @staticmethod
        def close() -> None:
            pass

    monkeypatch.setattr(uploader_module, "_connect", lambda *_args: Connection())

    result = upload_episode(episode, "http://127.0.0.1:8080/upload", sent)

    assert result.success
    assert result.size_bytes == len(captured.getvalue())
    assert not (tmp_path / "staging").exists()
    with tarfile.open(fileobj=BytesIO(captured.getvalue()), mode="r:") as tar:
        assert f"{episode.name}/manifest.json" in tar.getnames()
    assert (sent / episode.name).exists()


def test_tar_creation_failure_leaves_episode_in_complete(
    tmp_path: Path,
    monkeypatch,
):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)

    def fail_archive(_episode_dir: Path, _tar_path: Path) -> None:
        raise OSError("archive disk full")

    monkeypatch.setattr(uploader_module, "create_episode_tar", fail_archive)

    result = upload_episode(
        episode,
        "http://127.0.0.1:8080/upload",
        sent,
        post=lambda *_args: {"status": "queued"},
    )

    assert not result.success
    assert result.size_bytes == 0
    assert "archive disk full" in result.error
    assert episode.exists()
    assert not (sent / episode.name).exists()


def test_retry_success_moves_episode_to_sent(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)
    uploaded: list[str] = []

    def ok(_url: str, tar_path: Path, episode_id: str, _timeout_seconds: float) -> dict:
        with tarfile.open(tar_path, "r") as tar:
            assert any(member.name.startswith("episode_1/") for member in tar.getmembers())
        uploaded.append(episode_id)
        return {"status": "queued"}

    assert retry_completed(complete, "http://127.0.0.1:8080/upload", sent, post=ok) == 1
    assert uploaded == [episode.name]
    assert not episode.exists()
    assert (sent / episode.name).exists()


def test_async_uploader_does_not_block_submit(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)
    upload_started = threading.Event()
    release_upload = threading.Event()

    def slow_ok(_url: str, _tar_path: Path, _episode_id: str, _timeout_seconds: float) -> dict:
        upload_started.set()
        release_upload.wait(timeout=2)
        return {"status": "queued"}

    uploader = AsyncUploader(complete, "http://127.0.0.1:8080/upload", sent, post=slow_ok)
    uploader.submit(episode)
    assert episode.exists()
    assert upload_started.wait(timeout=2)
    release_upload.set()

    uploader.close(wait=True)
    assert not episode.exists()
    assert (sent / episode.name).exists()
    assert uploader.snapshot_results()[0].success


def test_async_uploader_survives_archive_failure(tmp_path: Path, monkeypatch):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    first = make_episode(complete)
    second = make_episode(complete, "episode_2")
    original_create_episode_tar = uploader_module.create_episode_tar
    attempts = 0

    def flaky_archive(episode_dir: Path, tar_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if episode_dir.name == first.name:
            raise OSError("archive disk full")
        original_create_episode_tar(episode_dir, tar_path)

    def ok(_url: str, _tar_path: Path, _episode_id: str, _timeout_seconds: float) -> dict:
        return {"status": "queued"}

    monkeypatch.setattr(uploader_module, "create_episode_tar", flaky_archive)

    uploader = AsyncUploader(complete, "http://127.0.0.1:8080/upload", sent, post=ok)
    uploader.submit(first)
    uploader.submit(second)
    uploader.close(wait=True)
    results = uploader.snapshot_results()

    assert attempts == 2
    assert [result.success for result in results] == [False, True]
    assert first.exists()
    assert (sent / second.name).exists()


def test_upload_timeout_is_passed_to_http_client(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)
    seen_timeout: list[float] = []

    def ok(_url: str, _tar_path: Path, _episode_id: str, timeout_seconds: float) -> dict:
        seen_timeout.append(timeout_seconds)
        return {"status": "queued"}

    result = upload_episode(
        episode,
        "http://127.0.0.1:8080/upload",
        sent,
        post=ok,
        upload_timeout_seconds=1.25,
    )

    assert result.success
    assert seen_timeout == [1.25]


def test_default_upload_timeout_allows_large_tunnel_uploads_to_drain():
    assert DEFAULT_UPLOAD_TIMEOUT_SECONDS == 300.0


def test_invalid_server_url_rejected_at_startup(tmp_path: Path):
    with pytest.raises(ValueError, match="SSH tunnel"):
        AsyncUploader(tmp_path / "complete", "ftp://server/upload", tmp_path / "sent")
    with pytest.raises(ValueError, match="SSH tunnel"):
        validate_server_url("server:8080/upload")  # no scheme
    with pytest.raises(ValueError, match="SSH tunnel"):
        validate_server_url("http://server:8080/upload")
    with pytest.raises(ValueError, match="SSH tunnel"):
        validate_server_url("https://127.0.0.1:8080/upload")
    validate_server_url("http://127.0.0.1:8080/upload")


def test_staging_tar_built_once_and_removed_after_success(tmp_path: Path, monkeypatch):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    staging = tmp_path / "staging"
    complete.mkdir()
    episode = make_episode(complete)
    tar_builds = 0
    original_create = uploader_module.create_episode_tar

    def counting_create(episode_dir: Path, tar_path: Path) -> None:
        nonlocal tar_builds
        tar_builds += 1
        original_create(episode_dir, tar_path)

    monkeypatch.setattr(uploader_module, "create_episode_tar", counting_create)
    attempts: list[str] = []

    def flaky(_url: str, _tar: Path, episode_id: str, _timeout: float) -> dict:
        attempts.append(episode_id)
        if len(attempts) == 1:
            raise RuntimeError("down")
        return {"status": "queued"}

    first = upload_episode(
        episode,
        "http://127.0.0.1:8080/upload",
        sent,
        post=flaky,
        staging_dir=staging,
    )
    assert not first.success
    assert (staging / f"{episode.name}.tar").exists()  # kept for reuse

    second = upload_episode(
        episode,
        "http://127.0.0.1:8080/upload",
        sent,
        post=flaky,
        staging_dir=staging,
    )
    assert second.success
    assert tar_builds == 1  # reused, not rebuilt
    assert not (staging / f"{episode.name}.tar").exists()
    assert (sent / episode.name).exists()


def test_zero_byte_staging_tar_is_rebuilt(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    staging = tmp_path / "staging"
    complete.mkdir()
    staging.mkdir()
    episode = make_episode(complete)
    tar_path = staging / f"{episode.name}.tar"
    tar_path.touch()

    def ok(_url: str, uploaded_tar: Path, _episode_id: str, _timeout: float) -> dict:
        assert uploaded_tar.stat().st_size > 0
        with tarfile.open(uploaded_tar, "r") as tar:
            assert any(member.name.startswith("episode_1/") for member in tar.getmembers())
        return {"status": "queued"}

    result = upload_episode(
        episode,
        "http://127.0.0.1:8080/upload",
        sent,
        post=ok,
        staging_dir=staging,
    )

    assert result.success
    assert (sent / episode.name).exists()


def test_retry_scan_skips_tarring_while_server_unreachable(tmp_path: Path, monkeypatch):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    complete.mkdir()
    episode = make_episode(complete)
    posts: list[str] = []

    def ok(_url: str, _tar: Path, episode_id: str, _timeout: float) -> dict:
        posts.append(episode_id)
        return {"status": "queued"}

    uploader = AsyncUploader(
        complete,
        "http://127.0.0.1:8080/upload",
        sent,
        retry_seconds=999.0,
        post=ok,
    )
    try:
        monkeypatch.setattr(uploader_module, "server_reachable", lambda *_a, **_k: False)
        uploader._retry_scan()
        assert posts == []
        assert not (tmp_path / "staging").exists()  # no tar built while down

        monkeypatch.setattr(uploader_module, "server_reachable", lambda *_a, **_k: True)
        uploader._retry_scan()
        assert posts == [episode.name]
        assert (sent / episode.name).exists()
    finally:
        uploader.close()


def test_orphan_staging_tars_are_cleaned(tmp_path: Path):
    complete = tmp_path / "complete"
    sent = tmp_path / "sent"
    staging = tmp_path / "staging"
    complete.mkdir()
    staging.mkdir()
    (staging / "episode_gone.tar").write_bytes(b"stale")

    uploader = AsyncUploader(
        complete,
        "http://127.0.0.1:8080/upload",
        sent,
        retry_seconds=999.0,
    )
    try:
        uploader._clean_orphan_tars()
        assert not (staging / "episode_gone.tar").exists()
    finally:
        uploader.close()


def test_create_episode_tar_never_leaves_partial_tar(tmp_path: Path, monkeypatch):
    complete = tmp_path / "complete"
    complete.mkdir()
    episode = make_episode(complete)
    tar_path = tmp_path / "staging" / "episode_1.tar"
    tar_path.parent.mkdir()

    class ExplodingTar:
        def __init__(self, path: Path):
            self.path = Path(path)

        def __enter__(self):
            self.path.write_bytes(b"partial")
            return self

        def __exit__(self, *args):
            return False

        def add(self, *_args, **_kwargs):
            raise OSError("disk full")

    monkeypatch.setattr(uploader_module.tarfile, "open", lambda path, mode: ExplodingTar(path))
    with pytest.raises(OSError, match="disk full"):
        uploader_module.create_episode_tar(episode, tar_path)

    assert not tar_path.exists()
    assert list(tar_path.parent.iterdir()) == []  # no .tmp leftovers either


class HeaderCaptureHandler(BaseHTTPRequestHandler):
    received_headers: dict[str, str] = {}

    def do_POST(self):
        type(self).received_headers = dict(self.headers)
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = b'{"status": "queued"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_post_tar_file_needs_no_authentication_header(tmp_path: Path):
    complete = tmp_path / "complete"
    complete.mkdir()
    episode = make_episode(complete)
    tar_path = tmp_path / "episode_1.tar"
    uploader_module.create_episode_tar(episode, tar_path)

    server = HTTPServer(("127.0.0.1", 0), HeaderCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/upload"
    try:
        response = post_tar_file(url, tar_path, episode.name, 5.0)
        assert response == {"status": "queued"}
        assert "X-Upload-Token" not in HeaderCaptureHandler.received_headers
        assert HeaderCaptureHandler.received_headers["X-Episode-Id"] == episode.name
    finally:
        server.shutdown()
        server.server_close()
