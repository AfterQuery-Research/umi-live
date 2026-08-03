#!/usr/bin/env python3
"""Merge evo --save_results zips into a single metrics.json.

Usage: merge_metrics.py <out_dir>

Reads every ``<metric>__<ref>__<est>.zip`` that docker/compare/entrypoint.sh
produced in <out_dir> (metric = ape | rpe) and writes <out_dir>/metrics.json:

    {
      "pairs": [
        {"metric": "ape", "ref": "slam_glim_v1", "est": "vive_v1",
         "stats": {"rmse": ..., "mean": ..., "median": ..., "std": ...,
                   "min": ..., "max": ..., "sse": ...},
         "info": {...evo info.json...}},
        ...
      ],
      "engines": ["slam_glim_v1", "vive_v1", ...]
    }

evo result zips contain ``stats.json`` (the summary numbers) and
``info.json`` (title / ref & est names) — stable across evo 1.x. Uses only
the stdlib; runs on the image's python3.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


def load_result_zip(path: Path) -> tuple[dict, dict]:
    """Return (stats, info) from an evo --save_results zip."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        stats = json.loads(zf.read("stats.json")) if "stats.json" in names else {}
        info = json.loads(zf.read("info.json")) if "info.json" in names else {}
    return stats, info


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 64
    out_dir = Path(sys.argv[1])

    pairs = []
    engines: set[str] = set()
    for zip_path in sorted(out_dir.glob("*__*__*.zip")):
        metric, ref, est = zip_path.stem.split("__", 2)
        try:
            stats, info = load_result_zip(zip_path)
        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
            print(f"warning: skipping unreadable {zip_path.name}: {exc}", file=sys.stderr)
            continue
        engines.update((ref, est))
        pairs.append(
            {
                "metric": metric,
                "ref": ref,
                "est": est,
                "stats": stats,
                "info": info,
                "results_zip": zip_path.name,
            }
        )

    if not pairs:
        print("error: no evo result zips found to merge", file=sys.stderr)
        return 65

    out_path = out_dir / "metrics.json"
    out_path.write_text(
        json.dumps({"pairs": pairs, "engines": sorted(engines)}, indent=2) + "\n"
    )
    print(f"merge_metrics: {len(pairs)} results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
