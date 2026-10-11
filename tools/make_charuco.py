#!/usr/bin/env python3
"""Generate the ChArUco board used by tools/calibrate_camera.py.

    uv run --group tools python tools/make_charuco.py --open

Display the image full-screen on a monitor. Its physical size is irrelevant for camera
intrinsics; the square and marker ratios and dictionary are what must match.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import cv2

SQUARES_X, SQUARES_Y = 8, 5
SQUARE_LEN, MARKER_LEN = 0.030, 0.022
BOARD_DICT = "DICT_5X5_100"


def generate_board(width: int, height: int, margin: int):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)
    return board.generateImage((width, height), marginSize=margin, borderBits=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/tmp/umi-live-charuco.png"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--margin", type=int, default=40)
    parser.add_argument("--open", action="store_true", help="open the generated image")
    args = parser.parse_args()
    if min(args.width, args.height) < 1 or args.margin < 0:
        parser.error("width and height must be positive; margin cannot be negative")
    image = generate_board(args.width, args.height, args.margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), image):
        raise SystemExit(f"could not write {args.out}")
    print(f"wrote {args.out} ({SQUARES_X}x{SQUARES_Y} ChArUco, {BOARD_DICT})")
    print("display it full-screen, then run tools/calibrate_camera.py")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
