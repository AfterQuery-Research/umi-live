"""Ingest tests: upload -> extract -> validate -> publish to raw_episodes. Nothing else."""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import umi_flow_server.server as server_module
from fastapi.testclient import TestClient
from umi_flow_server.server import ServerConfig, create_app

from tests.helpers import make_episode_tar, make_v2_episode


def wait_results(app, n: int = 1, timeout_s: float = 10.0):
    worker = app.state.worker
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        results = worker.snapshot_results()
        if len(results) >= n:
            return results
        time.sleep(0.02)
    raise AssertionError("ingest worker produced no result in time")


def upload(client: TestClient, tar_path: Path, episode_id: str):
    headers = {"X-Episode-Id": episode_id}
    return client.post("/upload", content=tar_path.read_bytes(), headers=headers)


def test_valid_upload_lands_in_raw_episodes(tmp_path: Path):
    episode = make_v2_episode(tmp_path)
    tar_path = make_episode_tar(episode, tmp_path / "out")
    data_dir = tmp_path / "data"
    app = create_app(ServerConfig(data_dir=data_dir))
    with TestClient(app) as client:
        r = upload(client, tar_path, episode.name)
        assert r.status_code == 202
        results = wait_results(app, 1)
    assert results[0].success, results[0].error
    published = data_dir / "raw_episodes" / episode.name
    assert (published / "COMPLETE").exists()
    assert (published / "manifest.json").exists()
    # Terminal outcome: the upload tar is gone (only crash survivors remain).
    assert not list((data_dir / "uploads").glob("*.tar"))


def test_bad_checksum_rejected_and_cleaned(tmp_path: Path):
    episode = make_v2_episode(tmp_path)
    # Corrupt a data file after checksumming.
    (episode / "umi1_lidar.pcap").write_bytes(b"corrupted")
    tar_path = make_episode_tar(episode, tmp_path / "out")
    data_dir = tmp_path / "data"
    app = create_app(ServerConfig(data_dir=data_dir))
    with TestClient(app) as client:
        upload(client, tar_path, episode.name)
        results = wait_results(app, 1)
    assert not results[0].success
    assert "checksum" in results[0].error
    assert not (data_dir / "raw_episodes" / episode.name).exists()


def test_upload_requires_no_authentication_header(tmp_path: Path):
    episode = make_v2_episode(tmp_path)
    tar_path = make_episode_tar(episode, tmp_path / "out")
    app = create_app(ServerConfig(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        assert upload(client, tar_path, episode.name).status_code == 202
        wait_results(app, 1)


def test_empty_upload_rejected(tmp_path: Path):
    app = create_app(ServerConfig(data_dir=tmp_path / "data"))
    with TestClient(app) as client:
        r = client.post("/upload", content=b"", headers={"X-Episode-Id": "x"})
        assert r.status_code == 400


def test_upload_rejected_when_disk_headroom_is_exhausted(tmp_path: Path, monkeypatch):
    episode = make_v2_episode(tmp_path)
    tar_path = make_episode_tar(episode, tmp_path / "out")
    monkeypatch.setattr(
        server_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100),
    )
    app = create_app(
        ServerConfig(
            data_dir=tmp_path / "data",
            max_extract_bytes=100,
            min_free_bytes=100,
        )
    )
    with TestClient(app) as client:
        response = upload(client, tar_path, episode.name)

    assert response.status_code == 507
    assert "disk space" in response.json()["detail"]
    assert not list((tmp_path / "data" / "uploads").iterdir())


def test_admission_reports_capacity_and_queued_bytes(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True)
    (uploads / ".episode.tar.tmp").write_bytes(b"1234")
    monkeypatch.setattr(
        server_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_000),
    )
    app = create_app(
        ServerConfig(
            data_dir=data_dir,
            min_free_bytes=100,
            max_extract_bytes=200,
            recording_reserve_bytes=500,
        )
    )
    with TestClient(app) as client:
        accepted = client.get("/health/admission?estimated_upload_bytes=300").json()
        rejected = client.get("/health/admission?estimated_upload_bytes=301").json()

    assert accepted["accepted"] is True
    assert accepted["reserved_bytes"] == 700
    assert accepted["queued_upload_bytes"] == 4
    assert rejected["accepted"] is False
    assert rejected["reason"] == "insufficient server storage headroom"


def test_episode_id_mismatch_fails(tmp_path: Path):
    episode = make_v2_episode(tmp_path)
    tar_path = make_episode_tar(episode, tmp_path / "out")
    data_dir = tmp_path / "data"
    app = create_app(ServerConfig(data_dir=data_dir))
    with TestClient(app) as client:
        upload(client, tar_path, "episode_someone_else")
        results = wait_results(app, 1)
    assert not results[0].success
    assert "mismatch" in results[0].error


def test_startup_recovers_leftover_tar(tmp_path: Path):
    episode = make_v2_episode(tmp_path)
    data_dir = tmp_path / "data"
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True)
    # Simulate a crash: a fully-streamed tar left in uploads/.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(episode, arcname=episode.name)
    (uploads / f"{episode.name}_deadbeef.tar").write_bytes(buf.getvalue())

    app = create_app(ServerConfig(data_dir=data_dir))
    with TestClient(app):  # startup hook runs recovery
        results = wait_results(app, 1)
    assert results[0].success
    assert (data_dir / "raw_episodes" / episode.name / "COMPLETE").exists()
