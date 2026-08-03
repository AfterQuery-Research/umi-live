# docker/ingest/Dockerfile — "umi-ingest" service image (CPU, always-up).
#
# Runs the FastAPI ingest server (server/scripts/server.py) on port 8080 inside
# the portable GPU processing box (Lambda instance / on-prem NVIDIA machine).
# Devices (Pi recorders) upload hermetic raw episodes here; the server writes
# them under /data/raw_episodes/<id>/ on the shared DATA_DIR volume.
#
# Build context is the REPO ROOT (see docker-compose.yml): we install the uv
# workspace's `robot-umi-server` package from the checked-in uv.lock, so the
# image is exactly as pinned as the repo.
#
# Server entry is server/scripts/server.py with --host/--port/--data-dir flags.
# Built + validated on the GPU box (2026-07-15). Note the image is large:
# lerobot[dataset] pulls torch per uv.lock (kept lock-faithful on purpose).

# Pinned to the 3.12 minor line (matches requires-python >=3.12 across the
# workspace); slim-bookworm variant verified present on Docker Hub 2026-07-15.
FROM python:3.12-slim-bookworm

# ffmpeg: lerobot's dataset video encoding shells out to the ffmpeg CLI
# (rgb_encoder_vcodec "h264" default in robot_umi_server.processing).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv pinned; must satisfy the workspace members' uv_build>=0.11.28,<0.12
# build-backend pins (0.11.29 was latest on PyPI as of 2026-07-15).
RUN pip install --no-cache-dir uv==0.11.29

WORKDIR /app

# COPY only what `uv sync` needs to resolve the workspace: root manifest +
# lock + all member packages (device is a workspace member, so its pyproject
# must be present even though the server doesn't depend on it).
COPY pyproject.toml uv.lock ./
COPY common ./common
COPY server ./server
COPY device ./device

# Install ONLY the server package + its locked deps (no dev group).
# UV_LINK_MODE=copy avoids hardlink warnings across Docker layer filesystems;
# Bytecode compile disabled: lerobot's tree trips buildkit's default nofile ulimit.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0
RUN uv sync --frozen --no-dev --package robot-umi-server

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

# Runtime env (set via docker-compose.yml / .env):
#   ROBOT_UMI_UPLOAD_TOKEN  required — server refuses to bind 0.0.0.0 without it
#   DATA_DIR                informational; the mount point is fixed at /data
# (Ingest is extract/validate/publish only — it takes no repo-id/push flags;
# those belong to the runner's dataset stage.)
ENTRYPOINT ["python", "server/scripts/server.py", \
            "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data"]
CMD []
