from __future__ import annotations

import gzip
import io
import tarfile
import threading
from pathlib import Path

import pytest
from umi_live_server.episode import INGESTED_FILE, extract_episode, validate_episode


def make_episode(path: Path, episode_name: str = "episode_test") -> Path:
    """Extraction tests only need a named directory with some files, not a full
    v2 episode (format-level checks live in test_receiver/test_runner)."""
    episode = path / episode_name
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text("{}")
    (episode / "data.bin").write_bytes(b"x" * 128)
    (episode / "COMPLETE").write_text("complete\n")
    return episode


def tar_episode(source: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w") as tar:
        tar.add(source, arcname=source.name)


@pytest.mark.parametrize("contents", ["", "\n  \n"])
def test_validate_rejects_empty_checksums(tmp_path: Path, contents: str):
    source = make_episode(tmp_path / "source")
    (source / "checksums.sha256").write_text(contents)
    with pytest.raises(ValueError, match="empty checksums"):
        validate_episode(source)


def test_extract_rejects_gzip_compressed_tar(tmp_path: Path):
    # A gzip body must be rejected: transparent decompression would let a tiny
    # upload expand without bound at extraction time.
    source = make_episode(tmp_path / "source")
    plain = tmp_path / "episode.tar"
    tar_episode(source, plain)
    gz_path = tmp_path / "episode.tar.gz"
    gz_path.write_bytes(gzip.compress(plain.read_bytes()))

    with pytest.raises(tarfile.ReadError):
        extract_episode(gz_path, tmp_path / "raw")


def test_extract_rejects_oversized_members(tmp_path: Path):
    tar_path = tmp_path / "big.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("episode_big/data.bin")
        info.size = 4096
        tar.addfile(info, io.BytesIO(b"x" * 4096))
    with pytest.raises(ValueError, match="maximum extracted size"):
        extract_episode(
            tar_path,
            tmp_path / "raw",
            expected_root="episode_big",
            max_extract_bytes=1024,
        )


def test_extract_rejects_episode_id_mismatch(tmp_path: Path):
    source = make_episode(tmp_path / "source", episode_name="episode_real")
    body_path = tmp_path / "episode.tar"
    tar_episode(source, body_path)

    with pytest.raises(ValueError, match="episode id mismatch"):
        extract_episode(body_path, tmp_path / "raw", expected_root="episode_claimed")


def test_extract_leaves_no_partial_dir_on_failure(tmp_path: Path, monkeypatch):
    source = make_episode(tmp_path / "source", episode_name="episode_x")
    body_path = tmp_path / "episode.tar"
    tar_episode(source, body_path)
    raw = tmp_path / "raw"

    real_extractall = tarfile.TarFile.extractall

    def boom(self, *args, **kwargs):
        real_extractall(self, *args, **kwargs)
        raise OSError("disk full during extract")

    monkeypatch.setattr(tarfile.TarFile, "extractall", boom)

    with pytest.raises(OSError, match="disk full"):
        extract_episode(body_path, raw, expected_root="episode_x")

    # No partial episode dir and no leftover staging dir: a re-upload can retry.
    assert not (raw / "episode_x").exists()
    assert list(raw.iterdir()) == []


def test_validate_failure_never_publishes_episode(tmp_path: Path):
    # A validator that rejects the episode must leave raw_episodes empty: the
    # runner scans that dir for the COMPLETE marker, so a half-validated episode
    # appearing there (even briefly) could be picked up and processed.
    source = make_episode(tmp_path / "source", episode_name="episode_bad")
    body_path = tmp_path / "episode.tar"
    tar_episode(source, body_path)
    raw = tmp_path / "raw"

    def reject(_episode_dir: Path) -> None:
        raise ValueError("checksum mismatch")

    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_episode(body_path, raw, expected_root="episode_bad", validate=reject)

    assert not (raw / "episode_bad").exists()
    assert not raw.exists() or list(raw.iterdir()) == []


def test_validate_runs_before_publish_on_the_staged_dir(tmp_path: Path):
    # The validator must see the episode's contents (in quarantine) and, on
    # success, the episode is then published intact.
    source = make_episode(tmp_path / "source", episode_name="episode_ok")
    body_path = tmp_path / "episode.tar"
    tar_episode(source, body_path)
    raw = tmp_path / "raw"
    seen: list[bool] = []

    def check(episode_dir: Path) -> None:
        # validated in a staging dir, NOT yet under raw/episode_ok
        seen.append((episode_dir / "COMPLETE").exists())
        assert not (raw / "episode_ok").exists()

    dest = extract_episode(body_path, raw, expected_root="episode_ok", validate=check)
    assert seen == [True]
    assert dest == raw / "episode_ok"
    assert (dest / "COMPLETE").exists()
    assert int((dest / INGESTED_FILE).read_text()) > 0


def test_concurrent_extract_of_same_episode_yields_one_winner(tmp_path: Path):
    source = make_episode(tmp_path / "source", episode_name="episode_race")
    body_path = tmp_path / "episode.tar"
    tar_episode(source, body_path)
    raw = tmp_path / "raw"
    raw.mkdir()

    results: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            extract_episode(body_path, raw, expected_root="episode_race")
            with lock:
                results.append("ok")
        except ValueError as exc:
            with lock:
                results.append(f"err:{exc}")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one worker wins; the other loses cleanly with "already exists".
    assert results.count("ok") == 1
    losers = [r for r in results if r != "ok"]
    assert len(losers) == 1
    assert "already exists" in losers[0]
    assert (raw / "episode_race").exists()
    # No staging directories left behind.
    assert [p.name for p in raw.iterdir()] == ["episode_race"]
