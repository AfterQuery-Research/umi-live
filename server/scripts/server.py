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
    app = create_app(ServerConfig(data_dir=data_dir))
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
