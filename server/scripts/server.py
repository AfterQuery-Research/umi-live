"""Run the ingest service: durable episode inbox (extract + validate + publish).

Processing (decode / SLAM / gripper / dataset) is the pipeline runner's job
(``server/scripts/runner.py``) — this process stays free of ROS/CUDA/lerobot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tyro
import uvicorn

from robot_umi_server.server import ServerConfig, create_app

UPLOAD_TOKEN_ENV = "ROBOT_UMI_UPLOAD_TOKEN"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8080
    data_dir: Path = Path("data")


def main(args: Args) -> None:
    upload_token = os.environ.get(UPLOAD_TOKEN_ENV, "")
    if args.host not in LOOPBACK_HOSTS and not upload_token:
        raise SystemExit(
            f"refusing to bind {args.host} without an upload token: set "
            f"{UPLOAD_TOKEN_ENV} to a shared secret (matching the device), or "
            f"bind 127.0.0.1 for local-only use."
        )
    app = create_app(ServerConfig(data_dir=args.data_dir, upload_token=upload_token))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
