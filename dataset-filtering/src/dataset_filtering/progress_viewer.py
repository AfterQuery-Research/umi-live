from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse


def summarize_progress(results_dir: str | Path, *, total_episodes: int = 298) -> dict[str, Any]:
    root = Path(results_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    valid_paths: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("input-*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append(row)
            valid_paths.append((path, row))
        except (OSError, json.JSONDecodeError):
            continue
    passed_rows = [row for row in rows if row.get("accepted")]
    failed_rows = [row for row in rows if not row.get("accepted")]
    passed_with_dense = [
        row for row in passed_rows if str(row.get("selected_profile", "")).startswith("dense")
    ]
    failed_with_dense = [
        row
        for row in failed_rows
        if any(
            str(attempt.get("profile", "")).startswith("dense")
            for attempt in row.get("attempts", [])
        )
    ]
    pipeline_errors = [row for row in failed_rows if row.get("pipeline_error")]
    latest = max(
        valid_paths,
        key=lambda value: value[0].stat().st_mtime,
        default=None,
    )
    return {
        "total": total_episodes,
        "completed": len(rows),
        "remaining": max(total_episodes - len(rows), 0),
        "passed": len(passed_rows),
        "passed_with_dense": len(passed_with_dense),
        "failed_with_dense": len(failed_with_dense),
        "failed": len(failed_rows),
        "pipeline_errors": len(pipeline_errors),
        "latest_combined_input": (
            int(latest[1]["combined_input_index"]) if latest is not None else None
        ),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def create_progress_app(
    results_dir: str | Path,
    *,
    total_episodes: int = 298,
) -> FastAPI:
    root = Path(results_dir).expanduser().resolve()
    page = Path(__file__).with_name("web") / "progress.html"
    app = FastAPI(title="Combined UMI filtering progress")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(page)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return summarize_progress(root, total_episodes=total_episodes)

    return app
