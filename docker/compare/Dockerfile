# docker/compare/Dockerfile — "umi-compare" stage-CLI image (CPU).
#
# Cross-engine trajectory comparison: for every pair of
# <derived_dir>/<engine>/trajectory.tum it runs evo's APE/RPE (TUM mode,
# Umeyama alignment) and merges the stats into one metrics.json, plus a
# combined evo_traj plot. One-shot CLI:
#   docker compose run --rm umi-compare /data/derived/<id> /data/derived/<id>/compare_v1
#
# evo==1.36.5 was the latest release on PyPI as of 2026-07-15 (verified).
# Pure-CPU, pure-python — nothing here needs GPU validation; only the
# end-to-end run against real stage outputs does.

# Pinned to the 3.12 minor line; slim-bookworm verified on Docker Hub 2026-07-15.
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir evo==1.36.5

# Headless matplotlib for evo's --save_plot (no display in the container).
ENV MPLBACKEND=Agg

COPY merge_metrics.py /merge_metrics.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
