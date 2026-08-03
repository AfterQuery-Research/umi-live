"""Run the pipeline runner: the filesystem state machine over stage containers.

Watches ``<data-dir>/raw_episodes`` for ingested episodes and drives the stage
graph to completion under ``<data-dir>/derived``.

--engine gpu (default): decode -> glim run via ``docker compose run --rm`` (the
amd64/CUDA images); gripper + dataset run in-process. --engine mac: a single
in-process ``glim`` stage replays each pcap through the arm64 CPU GLIM container
(mac/, no GPU, no compose), then the same in-process gripper + dataset.

Deployment knobs via env: ROBOT_UMI_REPO_ID (required for the dataset stage),
ROBOT_UMI_DATASET_ROOT, ROBOT_UMI_PUSH_TO_HUB=1, ROBOT_UMI_PUBLIC=1,
ROBOT_UMI_ENGINE (the trajectory subdir the dataset reads; default glim — leave it
as glim for both engines, since the mac SLAM stage is also named glim).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from robot_umi_server import align_stage, dataset_stage, gripper_stage, mac_glim, pair_stage
from robot_umi_server.runner import Executor, Runner, default_stages, mac_stages


@dataclass
class Args:
    data_dir: Path = Path("data")
    interval_s: float = 5.0
    once: bool = False  # single scan pass (cron-style), else run forever
    # gpu = compose decode+GLIM (amd64/CUDA); mac = in-process arm64 GLIM pcap replay.
    engine: Literal["gpu", "mac"] = "gpu"
    # Two-Pi bimanual: merge each session's two single-arm episodes into one before the
    # runner processes it. Off => single-Pi behavior unchanged (role tags, if any, ignored).
    pair: bool = False
    # Only meaningful with --pair: a session whose mate never arrives within this window has
    # its lone arm dropped to orphans/ (rather than processed alone into a wrong dataset).
    pair_timeout_s: float = pair_stage.DEFAULT_TIMEOUT_S


def main(args: Args) -> None:
    def run_gripper(episode_dir: Path, out_dir: Path, scope: str) -> None:
        del scope
        gripper_stage.run(episode_dir, out_dir)

    def run_dataset(episode_dir: Path, out_dir: Path, scope: str) -> None:
        dataset_stage.run(episode_dir, out_dir, scope)

    def run_align(episode_dir: Path, out_dir: Path, scope: str) -> None:
        del scope
        align_stage.run(episode_dir, out_dir)

    if args.engine == "mac":
        stages = mac_stages(
            glim_fn=mac_glim.make_glim_fn(),
            dataset_fn=run_dataset,
            gripper_fn=run_gripper,
            align_fn=run_align,
        )
    else:
        stages = default_stages(
            dataset_fn=run_dataset, gripper_fn=run_gripper, align_fn=run_align
        )

    runner = Runner(
        data_dir=args.data_dir,
        stages=stages,
        executor=Executor(args.data_dir),
        pair_sessions=args.pair,
    )

    def pair_pass() -> None:
        if not args.pair:
            return
        try:
            pair_stage.pair_once(args.data_dir, timeout_s=args.pair_timeout_s)
        except Exception as exc:  # pairing must never take down the whole runner loop
            print(f"pair: pass failed, continuing: {exc}", flush=True)

    if args.once:
        pair_pass()
        executed = runner.scan_once()
        print(f"runner: executed {executed} stage runs")
    else:
        print(f"pipeline runner watching {runner.raw_root} (pairing={args.pair})", flush=True)
        while True:
            pair_pass()
            runner.scan_once()
            time.sleep(args.interval_s)


if __name__ == "__main__":
    main(tyro.cli(Args))
