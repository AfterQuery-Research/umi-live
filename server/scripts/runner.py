"""Run the pipeline runner: the filesystem state machine over stage containers.

Watches ``<data-dir>/raw_episodes`` for ingested episodes and drives the stage
graph to completion under ``<data-dir>/derived``.

Every session is paired before the fixed decode -> GLIM -> gripper -> dataset graph.
"""

from __future__ import annotations

import os
from pathlib import Path

from umi_flow_server import align_stage, dataset_stage, gripper_stage
from umi_flow_server.runner import Executor, Runner, default_stages


def main() -> None:
    def run_gripper(episode_dir: Path, out_dir: Path, scope: str) -> None:
        del scope
        gripper_stage.run(episode_dir, out_dir)

    def run_dataset(episode_dir: Path, out_dir: Path, scope: str) -> None:
        dataset_stage.run(episode_dir, out_dir, scope)

    def run_align(episode_dir: Path, out_dir: Path, scope: str) -> None:
        del scope
        align_stage.run(episode_dir, out_dir)

    stages = default_stages(dataset_fn=run_dataset, gripper_fn=run_gripper, align_fn=run_align)

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    runner = Runner(data_dir=data_dir, stages=stages, executor=Executor(data_dir))
    runner.run_forever()


if __name__ == "__main__":
    main()
