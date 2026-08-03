# docker/dataset/Dockerfile — "umi-dataset" stage-CLI image (CPU).
#
# Fuses a raw episode + its derived stage outputs (SLAM trajectory, decoded
# data) into the training dataset format via the (in-development)
# robot_umi_server.dataset_stage module. One-shot CLI:
#   docker compose run --rm umi-dataset /data/raw_episodes/<id> /data/derived/<id>/dataset_v1
#
# The install layers below INTENTIONALLY mirror docker/ingest/Dockerfile
# byte-for-byte (same base, same COPY set, same uv sync) so BuildKit shares
# the cache between the two images — see that file for the pin rationale.
#
# Status: `python -m robot_umi_server.dataset_stage` does not exist in the
# repo yet (module is being written); this image will build today but the
# stage will fail at runtime until the module lands. Entry contract
# (<episode_dir> <out_dir> + COMPLETE marker) is fixed by entrypoint.sh.

# Pinned to the 3.12 minor line; slim-bookworm verified on Docker Hub 2026-07-15.
FROM python:3.12-slim-bookworm

# ffmpeg: lerobot's dataset video encoding shells out to the ffmpeg CLI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv pinned; satisfies the workspace's uv_build>=0.11.28,<0.12 pins.
RUN pip install --no-cache-dir uv==0.11.29

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY common ./common
COPY server ./server
COPY device ./device

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0
RUN uv sync --frozen --no-dev --package robot-umi-server

ENV PATH="/app/.venv/bin:$PATH"

COPY docker/dataset/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runtime env (via docker-compose.yml): HF_TOKEN for Hub pushes, if the
# dataset stage publishes.
ENTRYPOINT ["/entrypoint.sh"]
