from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from dataset_filtering.models import Episode, EpisodeKind, EpisodeSummary, PoseEpisode

DEFAULT_REPO_ID = "brandonyang/dual-lidar-umi"
YAM_REFERENCE_REPO_ID = "brandonyang/yam-ultrawide-teleop"
EXPECTED_UMI_NAMES = (
    "umi1_x",
    "umi1_y",
    "umi1_z",
    "umi1_rx",
    "umi1_ry",
    "umi1_rz",
    "umi2_x",
    "umi2_y",
    "umi2_z",
    "umi2_rx",
    "umi2_ry",
    "umi2_rz",
)
EXPECTED_CALIBRATED_UMI_NAMES = (
    *EXPECTED_UMI_NAMES[:6],
    "umi1_gripper",
    *EXPECTED_UMI_NAMES[6:],
    "umi2_gripper",
)


class DatasetSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetLocation:
    source: str
    local_root: Path | None
    repo_id: str | None
    revision: str


class LeRobotSource:
    """Read-only LeRobot v3 access with an isolated local cache.

    A filesystem path is read in place. A Hub repo is downloaded only into ``cache_dir``.
    This class deliberately has no upload or mutation methods.
    """

    def __init__(
        self,
        source: str = DEFAULT_REPO_ID,
        *,
        revision: str = "main",
        cache_dir: str | Path | None = None,
    ) -> None:
        source_path = Path(source).expanduser()
        if source_path.exists():
            self.location = DatasetLocation(
                source=str(source_path.resolve()),
                local_root=source_path.resolve(),
                repo_id=None,
                revision=revision,
            )
        else:
            self.location = DatasetLocation(
                source=source,
                local_root=None,
                repo_id=source,
                revision=revision,
            )
        default_cache = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
        self.cache_dir = Path(cache_dir or default_cache).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._info: dict[str, Any] | None = None
        self._episodes: pd.DataFrame | None = None

    @property
    def source(self) -> str:
        return self.location.source

    def info(self) -> dict[str, Any]:
        if self._info is None:
            path = self._materialize("meta/info.json")
            self._info = json.loads(path.read_text(encoding="utf-8"))
        return self._info

    def episode_summaries(self) -> list[EpisodeSummary]:
        episodes = self._episode_metadata()
        result: list[EpisodeSummary] = []
        fps = float(self.info()["fps"])
        for row in episodes.to_dict(orient="records"):
            tasks = row.get("tasks", [])
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            task = str(tasks[0]) if isinstance(tasks, (list, tuple)) and tasks else ""
            length = int(row["length"])
            result.append(
                EpisodeSummary(
                    episode_index=int(row["episode_index"]),
                    length=length,
                    duration_s=length / fps,
                    task=task,
                    data_chunk_index=int(row["data/chunk_index"]),
                    data_file_index=int(row["data/file_index"]),
                )
            )
        return sorted(result, key=lambda item: item.episode_index)

    def load_episode(self, episode_index: int) -> Episode:
        meta = self._episode_metadata()
        match = meta.loc[meta["episode_index"] == episode_index]
        if match.empty:
            available = sorted(int(x) for x in meta["episode_index"].tolist())
            raise IndexError(f"episode {episode_index} not found; available={available}")
        row = match.iloc[0]
        chunk = int(row["data/chunk_index"])
        file_index = int(row["data/file_index"])
        data_path = self._materialize(f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet")
        table = pd.read_parquet(data_path)
        frame = table.loc[table["episode_index"] == episode_index].sort_values("frame_index")
        if len(frame) != int(row["length"]):
            raise RuntimeError(
                f"episode {episode_index}: metadata length {int(row['length'])} "
                f"!= data rows {len(frame)}"
            )

        info = self.info()
        names = tuple(info["features"]["observation.state"].get("names") or ())
        state = _stack_vector_column(frame["observation.state"])
        action = _stack_vector_column(frame["action"]) if "action" in frame else None
        kind = classify_schema(info, state.shape[1], names)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        grippers = _extract_grippers(frame, kind, state)
        tasks = row.get("tasks", [])
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        task = str(tasks[0]) if isinstance(tasks, (list, tuple)) and tasks else ""
        warnings: list[str] = []
        if kind is EpisodeKind.YAM_JOINT:
            warnings.append(
                "This source contains 14-D YAM joint states, not 12-D UMI Cartesian poses; "
                "LiDAR position noise and UMI-to-YAM IK cannot be inferred from it."
            )
        return Episode(
            episode_index=episode_index,
            fps=float(info["fps"]),
            timestamps=timestamps,
            state=state,
            action=action,
            grippers=grippers,
            task=task,
            kind=kind,
            state_names=names,
            source=self.source,
            warnings=tuple(warnings),
        )

    def prefetch_episode_data(self, episode_indices: list[int]) -> None:
        """Materialize selected parquet shards concurrently into the read-only cache."""
        meta = self._episode_metadata()
        selected = meta.loc[meta["episode_index"].isin(episode_indices)]
        paths = sorted(
            {
                f"data/chunk-{int(row['data/chunk_index']):03d}/"
                f"file-{int(row['data/file_index']):03d}.parquet"
                for row in selected.to_dict(orient="records")
            }
        )
        with ThreadPoolExecutor(max_workers=min(12, len(paths))) as pool:
            list(pool.map(self._materialize, paths))

    def materialize(self, relative_path: str) -> Path:
        """Return a local read-only copy of one source-repository file."""
        return self._materialize(relative_path)

    def episode_metadata_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the complete LeRobot episode index."""
        return self._episode_metadata().copy(deep=True)

    def _episode_metadata(self) -> pd.DataFrame:
        if self._episodes is not None:
            return self._episodes
        files = self._episode_metadata_paths()
        if not files:
            raise RuntimeError(f"{self.source}: no meta/episodes parquet files")
        with ThreadPoolExecutor(max_workers=min(12, len(files))) as pool:
            materialized = list(pool.map(self._materialize, files))
        self._episodes = pd.concat(
            [pd.read_parquet(path) for path in materialized], ignore_index=True
        )
        return self._episodes

    def _episode_metadata_paths(self) -> list[str]:
        if self.location.local_root is not None:
            root = self.location.local_root / "meta" / "episodes"
            return [
                str(path.relative_to(self.location.local_root))
                for path in sorted(root.rglob("*.parquet"))
            ]

        # A complete remote-repo cache must be usable without consulting the Hub.
        # Worker processes create fresh LeRobotSource objects for each episode, so
        # an unconditional tree-API lookup here would make an otherwise local run
        # fail as soon as Wi-Fi is disconnected.
        assert self.location.repo_id is not None
        repo_slug = self.location.repo_id.replace("/", "--")
        cached_revision = self.cache_dir / repo_slug / self.location.revision
        cached_root = cached_revision / "meta" / "episodes"
        cached_paths = sorted(cached_root.rglob("*.parquet"))
        cached_info = cached_revision / "meta" / "info.json"
        if cached_paths and cached_info.is_file():
            try:
                expected_episodes = int(
                    json.loads(cached_info.read_text(encoding="utf-8"))["total_episodes"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                expected_episodes = 0
            if expected_episodes > 0 and len(cached_paths) >= expected_episodes:
                return [str(path.relative_to(cached_revision)) for path in cached_paths]

        repo_id = urllib.parse.quote(self.location.repo_id or "", safe="/")
        quoted_revision = urllib.parse.quote(self.location.revision, safe="")
        url = (
            f"https://huggingface.co/api/datasets/{repo_id}/tree/"
            f"{quoted_revision}/meta/episodes?recursive=true&expand=false"
        )
        entries = _read_json_url(url)
        paths = sorted(
            str(entry["path"])
            for entry in entries
            if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".parquet")
        )
        if paths:
            return paths
        # Some Hub tree endpoints do not recurse through the chunk directory.
        chunk_url = url.replace("/meta/episodes?", "/meta/episodes/chunk-000?")
        entries = _read_json_url(chunk_url)
        return sorted(
            str(entry["path"])
            for entry in entries
            if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".parquet")
        )

    def _materialize(self, relative_path: str) -> Path:
        if self.location.local_root is not None:
            path = self.location.local_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        assert self.location.repo_id is not None
        repo_slug = self.location.repo_id.replace("/", "--")
        destination = self.cache_dir / repo_slug / self.location.revision / relative_path
        if destination.is_file() and destination.stat().st_size:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        repo_id = urllib.parse.quote(self.location.repo_id, safe="/")
        path = urllib.parse.quote(relative_path, safe="/")
        revision = urllib.parse.quote(self.location.revision, safe="")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "umi-dataset-filtering/0.1"})
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(destination)
        except (OSError, urllib.error.URLError) as exc:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(f"failed to download read-only dataset file {url}: {exc}") from exc
        return destination


def classify_schema(info: dict[str, Any], width: int, names: tuple[str, ...]) -> EpisodeKind:
    robot_type = str(info.get("robot_type", "")).lower()
    lowered = tuple(name.lower() for name in names)
    if width == 12 and (not names or lowered == EXPECTED_UMI_NAMES):
        return EpisodeKind.UMI_POSE
    if width == 12 and all(token in " ".join(lowered) for token in ("umi1", "umi2")):
        return EpisodeKind.UMI_POSE
    if width == 14 and lowered == EXPECTED_CALIBRATED_UMI_NAMES:
        return EpisodeKind.UMI_POSE
    if width == 14 and ("yam" in robot_type or any("joint" in name for name in lowered)):
        return EpisodeKind.YAM_JOINT
    raise DatasetSchemaError(
        f"unsupported observation.state schema: width={width}, robot_type={robot_type!r}, "
        f"names={list(names)}; expected 12-D UMI pose, 14-D calibrated UMI pose, or "
        "14-D [left 6 joints+gripper, right 6 joints+gripper]"
    )


def as_pose_episode(episode: Episode) -> PoseEpisode:
    if episode.kind is not EpisodeKind.UMI_POSE:
        raise DatasetSchemaError(
            f"{episode.source} episode {episode.episode_index} is "
            f"{episode.kind.value}, not UMI pose data"
        )
    if episode.state.shape[1] == 14:
        positions = np.stack((episode.state[:, 0:3], episode.state[:, 7:10]), axis=1)
        rotvecs = np.stack((episode.state[:, 3:6], episode.state[:, 10:13]), axis=1)
    else:
        positions = np.stack((episode.state[:, 0:3], episode.state[:, 6:9]), axis=1)
        rotvecs = np.stack((episode.state[:, 3:6], episode.state[:, 9:12]), axis=1)
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(-1, 2, 4)
    return PoseEpisode(
        episode_index=episode.episode_index,
        fps=episode.fps,
        timestamps=episode.timestamps,
        positions=positions,
        quaternions=quaternions,
        grippers=episode.grippers,
        task=episode.task,
        source=episode.source,
    )


def _stack_vector_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float64) for value in series.to_numpy()])


def _extract_grippers(
    frame: pd.DataFrame, kind: EpisodeKind, state: np.ndarray
) -> np.ndarray | None:
    if state.shape[1] == 14:
        return state[:, [6, 13]].copy()
    columns = ("observation.gripper_width.umi1", "observation.gripper_width.umi2")
    if all(column in frame for column in columns):
        values = []
        for column in columns:
            raw = frame[column].to_numpy()
            values.append(np.asarray([np.asarray(x).reshape(-1)[0] for x in raw], dtype=float))
        return np.stack(values, axis=1)
    return None


def _read_json_url(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "umi-dataset-filtering/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read {url}: {exc}") from exc
