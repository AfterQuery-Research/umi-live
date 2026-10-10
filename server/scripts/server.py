"""Run the ingest service: durable episode inbox (extract + validate + publish).

Processing (decode / SLAM / gripper / dataset) is the pipeline runner's job
(``server/scripts/runner.py``) — this process stays free of ROS/CUDA/lerobot.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from umi_flow_server.server import ServerConfig, create_app


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    recording_reserve_gib = int(os.environ.get("UMI_FLOW_RECORDING_RESERVE_GIB", "64"))
    app = create_app(
        ServerConfig(
            data_dir=data_dir,
            recording_reserve_bytes=recording_reserve_gib * 1024 * 1024 * 1024,
        )
    )
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
