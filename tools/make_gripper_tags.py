"""Generate a printable PDF sheet of ArUco gripper tags, grouped per arm.

The gripper-width detector (server/src/umi_flow_server/gripper.py) reads two ArUco tags from
DICT_4X4_50 on each gripper's two jaws. To keep the two arms unambiguous (so a camera
that ever glimpses the other gripper can't confuse them), each arm uses its OWN pair:

    umi1 gripper -> IDs 0 and 1
    umi2 gripper -> IDs 2 and 3

This lays out one labeled block per arm, each a grid of that arm's two tags alternating,
at a fixed physical size INCLUDING the white quiet zone. Print, verify scale with a ruler,
then cut out 0/1 pairs for umi1 and 2/3 pairs for umi2.

    uv run --no-sync python tools/make_gripper_tags.py            # 20 mm, US Letter
    uv run --no-sync python tools/make_gripper_tags.py --tag-mm 20 --page a4 --out tags.pdf

IMPORTANT: print at 100% / "Actual Size" (NOT "Fit to page"), then check the 20 mm
reference square at the top with a ruler before trusting the tags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ARUCO_DICT = "DICT_4X4_50"  # must match server/src/umi_flow_server/gripper.py
# One (id_a, id_b) pair per arm; must match the per-arm target IDs the detector uses.
TAG_SETS = [("umi1", (0, 1)), ("umi2", (2, 3))]
PAGES_MM = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}


def module_grid(marker_id: int, dict_name: str = ARUCO_DICT) -> np.ndarray:
    """(M, M) bool grid for a marker, True = black module (incl the 1-module black border)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    modules = dictionary.markerSize + 2  # data bits + 1-module border each side
    px_per = 40
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, modules * px_per, borderBits=1)
    # sample each module's center -> black (<128) / white
    centers = img[px_per // 2 :: px_per, px_per // 2 :: px_per][:modules, :modules]
    return centers < 128


def draw_tag(ax, x: float, y: float, tag_mm: float, marker_id: int, quiet_frac: float) -> None:
    """Draw one tag at (x, y) top-left (mm): white cell, cut box, black marker, id label."""
    grid = module_grid(marker_id)  # True = black module (incl the 1-module black border)
    modules = grid.shape[0]
    marker_mm = tag_mm * (1.0 - 2.0 * quiet_frac)  # black marker inside the white quiet zone
    m0 = x + tag_mm * quiet_frac  # marker top-left within the cell
    n0 = y + tag_mm * quiet_frac
    step = marker_mm / modules
    eps = step * 0.04  # overlap so adjacent same-colour cells merge with no seam
    # white cell (the "20 mm incl. white area") + thin gray cut guide
    ax.add_patch(mpatches.Rectangle((x, y), tag_mm, tag_mm, facecolor="white", edgecolor="none"))
    ax.add_patch(
        mpatches.Rectangle((x, y), tag_mm, tag_mm, facecolor="none", edgecolor="0.7", linewidth=0.3)
    )
    # one solid black marker square (no black-black seams), then only the white modules on top
    ax.add_patch(
        mpatches.Rectangle((m0, n0), marker_mm, marker_mm, facecolor="black", edgecolor="none")
    )
    for r in range(modules):
        for c in range(modules):
            if not grid[r, c]:  # white data module (always interior, so eps can't reach the edge)
                ax.add_patch(
                    mpatches.Rectangle(
                        (m0 + c * step - eps, n0 + r * step - eps),
                        step + 2 * eps,
                        step + 2 * eps,
                        facecolor="white",
                        edgecolor="none",
                        antialiased=False,
                    )
                )
    ax.text(x + 0.4, y + 0.4, str(marker_id), va="top", ha="left", fontsize=3.5, color="0.5")


def build(out: Path, tag_mm: float, page: str, quiet_frac: float, margin_mm: float, gap_mm: float):
    pw, ph = PAGES_MM[page]
    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, pw)
    ax.set_ylim(0, ph)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()  # origin top-left, like a page

    sets_desc = ", ".join(f"{arm} {a}/{b}" for arm, (a, b) in TAG_SETS)
    ax.text(
        margin_mm,
        8,
        f"ArUco gripper tags — {ARUCO_DICT}, {sets_desc}, {tag_mm:g} mm incl. white border",
        fontsize=7,
        va="center",
    )
    ax.text(
        margin_mm,
        13,
        "PRINT AT 100% / ACTUAL SIZE (not 'fit to page'). "
        f"Verify the square below is exactly {tag_mm:g} mm with a ruler.",
        fontsize=5.5,
        va="center",
        color="0.3",
    )
    ax.add_patch(
        mpatches.Rectangle(
            (margin_mm, 16), tag_mm, tag_mm, facecolor="none", edgecolor="black", linewidth=0.4
        )
    )
    ax.text(
        margin_mm + tag_mm + 3,
        16 + tag_mm / 2,
        f"<- {tag_mm:g} mm reference",
        fontsize=5.5,
        va="center",
        color="0.3",
    )

    pitch = tag_mm + gap_mm
    ncols = int((pw - 2 * margin_mm + gap_mm) // pitch)
    top = 16 + tag_mm + 10  # first block starts below the header/reference
    sub_mm = 6.0  # height reserved for each block's subheader
    # split the remaining page height evenly across the arm blocks
    avail = ph - top - margin_mm
    block_h = (avail - (len(TAG_SETS) - 1) * gap_mm) / len(TAG_SETS)
    rows_per_block = max(1, int((block_h - sub_mm + gap_mm) // pitch))

    total = 0
    y_block = top
    for arm, ids in TAG_SETS:
        ax.text(
            margin_mm,
            y_block,
            f"{arm} gripper  —  IDs {ids[0]} / {ids[1]}",
            fontsize=7,
            va="center",
            fontweight="bold",
        )
        grid_top = y_block + sub_mm
        for row in range(rows_per_block):
            for col in range(ncols):
                x = margin_mm + col * pitch
                y = grid_top + row * pitch
                draw_tag(ax, x, y, tag_mm, ids[(row + col) % 2], quiet_frac)
                total += 1
        y_block = grid_top + rows_per_block * pitch + gap_mm

    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(
        f"wrote {out}  ({len(TAG_SETS)} arm blocks, {ncols}x{rows_per_block} tags each = "
        f"{total} total, {sets_desc}, {tag_mm:g} mm on {page})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("gripper_tags_4x4_20mm.pdf"))
    p.add_argument("--tag-mm", type=float, default=20.0, help="tag size incl. white border (mm)")
    p.add_argument("--page", choices=list(PAGES_MM), default="letter")
    p.add_argument(
        "--quiet-frac",
        type=float,
        default=0.10,
        help="white quiet-zone fraction per side (0.10 -> 16mm marker in a 20mm cell)",
    )
    p.add_argument("--margin-mm", type=float, default=12.0)
    p.add_argument("--gap-mm", type=float, default=4.0, help="gutter between tags for cutting")
    a = p.parse_args()
    build(a.out, a.tag_mm, a.page, a.quiet_frac, a.margin_mm, a.gap_mm)


if __name__ == "__main__":
    main()
