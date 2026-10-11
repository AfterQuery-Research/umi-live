FROM python:3.12-slim-bookworm

WORKDIR /app

# Ingest only validates and stores uploads. Keep LeRobot, Torch, CUDA, ROS, and
# processing dependencies out of this always-running image.
COPY docker/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY common/umi-live-core/src ./common/umi-live-core/src
COPY server/src ./server/src
COPY server/scripts/server.py ./server/scripts/server.py

ENV PYTHONPATH="/app/common/umi-live-core/src:/app/server/src"
EXPOSE 8080

ENTRYPOINT ["python", "server/scripts/server.py"]
