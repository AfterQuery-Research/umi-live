"""Repository-level guards for the single supported deployment workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_pair_inventory_has_only_variable_connection_and_task_inputs() -> None:
    lines = [
        line.split("=", 1)[0]
        for line in (ROOT / "provisioning/pair.env.example").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == ["SERVER", "LEADER", "FOLLOWER", "TASK"]


def test_only_one_readme_and_no_legacy_inventory() -> None:
    readmes = [
        path
        for path in ROOT.rglob("README*")
        if not {".git", ".pytest_cache", ".venv", "target"}.intersection(path.parts)
    ]
    assert readmes == [ROOT / "README.md"]
    for relative in (
        ".env.example",
        "bin/switch-target",
        "provisioning/secrets.example",
        "provisioning/server.env.example",
        "provisioning/rigs/example.env",
        "config/umi.pair-leader.example.yaml",
        "config/umi.pair-follower.example.yaml",
    ):
        assert not (ROOT / relative).exists()


def test_provisioning_scripts_are_valid_and_have_no_removed_inputs() -> None:
    scripts = [
        ROOT / "bin/provision-server",
        ROOT / "bin/provision-pair",
        ROOT / "bin/provision-pi",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
    text = "\n".join(path.read_text() for path in scripts)
    for removed in (
        "UPLOAD_TOKEN",
        "RUNNER_ARGS",
        "RIG_TARGET",
        "TIME_REF",
        "ALLOW_SD",
        "SERVER_URL",
    ):
        assert removed not in text


def test_upload_service_is_loopback_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    assert '"127.0.0.1:8080:8080"' in compose
