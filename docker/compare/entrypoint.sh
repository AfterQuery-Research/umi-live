#!/usr/bin/env bash
# docker/compare/entrypoint.sh — umi-compare stage CLI.
#
#   entrypoint.sh <derived_dir> <out_dir>
#     <derived_dir>  the episode's derived root, e.g. /data/derived/<id>
#                    (engines = subdirs that contain a trajectory.tum,
#                     e.g. slam_glim_v1/, vive_v1/, ...)
#     <out_dir>      stage output, e.g. /data/derived/<id>/compare_v1
#
# For each UNORDERED pair of engines (ref = alphabetically first — arbitrary
# but deterministic; APE/RPE after -a alignment are near-symmetric anyway):
#   evo_ape tum <ref> <est> -a --save_results <out>/ape__<ref>__<est>.zip
#   evo_rpe tum <ref> <est> -a --save_results <out>/rpe__<ref>__<est>.zip
# then merge_metrics.py folds every zip's stats.json into <out>/metrics.json,
# evo_traj renders a combined plot, and COMPLETE is touched.
#
# <2 trajectories is NOT an error: single-engine episodes are normal early
# on. We write a metrics.json that says so and still mark COMPLETE, so the
# pipeline runner doesn't wedge on them.
#
# evo notes: `-a` = Umeyama alignment without scale (both engines are metric);
# `--no_warnings` keeps reruns non-interactive (evo prompts before
# overwriting result files otherwise); evo_rpe uses its default delta of
# 1 frame. MPLBACKEND=Agg is set in the image for headless --save_plot.

set -euo pipefail

usage() { echo "usage: $0 <derived_dir> <out_dir>" >&2; exit 64; }
[[ $# -eq 2 ]] || usage
DERIVED_DIR=$1
OUT_DIR=$2

[[ -d "$DERIVED_DIR" ]] || { echo "error: derived dir not found: $DERIVED_DIR" >&2; exit 66; }

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE" "$OUT_DIR"/*.zip "$OUT_DIR"/metrics.json 2>/dev/null || true

# --- discover engines (sorted -> deterministic pairing) ----------------------
engines=()
tums=()
while IFS= read -r tum; do
    dir=$(dirname "$tum")
    # Skip our own (possibly stale) output dir.
    [[ "$(realpath "$dir")" == "$(realpath "$OUT_DIR")" ]] && continue
    engines+=("$(basename "$dir")")
    tums+=("$tum")
done < <(find "$DERIVED_DIR" -mindepth 2 -maxdepth 2 -name trajectory.tum | sort)

echo "compare: found ${#engines[@]} trajectories: ${engines[*]:-none}"

if (( ${#engines[@]} < 2 )); then
    echo "compare: fewer than 2 trajectories — nothing to compare (not an error)"
    python3 - "$OUT_DIR" "${engines[@]:-}" <<'EOF'
import json, sys
out, engines = sys.argv[1], [e for e in sys.argv[2:] if e]
json.dump({"pairs": [], "engines": engines,
           "note": "fewer than 2 trajectories; nothing compared"},
          open(f"{out}/metrics.json", "w"), indent=2)
EOF
    touch "$OUT_DIR/COMPLETE"
    exit 0
fi

# --- pairwise APE / RPE -------------------------------------------------------
n=${#engines[@]}
for ((i = 0; i < n; i++)); do
    for ((j = i + 1; j < n; j++)); do
        ref_name=${engines[i]}; ref_tum=${tums[i]}
        est_name=${engines[j]}; est_tum=${tums[j]}
        base="${ref_name}__${est_name}"
        echo "compare: $ref_name vs $est_name"
        evo_ape tum "$ref_tum" "$est_tum" -a --no_warnings \
            --save_results "$OUT_DIR/ape__${base}.zip"
        evo_rpe tum "$ref_tum" "$est_tum" -a --no_warnings \
            --save_results "$OUT_DIR/rpe__${base}.zip"
    done
done

# --- merge every result zip into one metrics.json -----------------------------
python3 /merge_metrics.py "$OUT_DIR"

# --- combined trajectory plot (multi-page PDF; ref = first engine) ------------
evo_traj tum "${tums[@]}" --ref "${tums[0]}" -a --no_warnings \
    --plot_mode xyz --save_plot "$OUT_DIR/traj_plots.pdf" \
    || echo "warning: evo_traj plotting failed (non-fatal)" >&2

touch "$OUT_DIR/COMPLETE"
echo "compare: done -> $OUT_DIR/metrics.json"
