from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline, summarize_baseline
from dataset_filtering_tests.head_trim import screen_head_trim
from dataset_filtering_tests.home_screen import screen_global_home_postures
from dataset_filtering_tests.portfolio import build_solver_portfolio
from dataset_filtering_tests.portfolio_export import (
    DEFAULT_PORTFOLIO_REPORT,
    build_portfolio_dataset,
)
from dataset_filtering_tests.smoothing_screen import (
    screen_smoothing_strength,
    select_safe_smoothing_improvements,
)
from dataset_filtering_tests.trajectory_sweep import (
    CONFIG_PROFILES,
    run_trajectory_sweep,
    select_experiment_episodes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset-filtering research harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="summarize the frozen v3 baseline")
    baseline.add_argument("--report", default=str(BASELINE_REPORT))
    baseline.add_argument("--output")
    head_trim = subparsers.add_parser(
        "screen-head-trim", help="rank event-safe head trims with framewise IK"
    )
    head_trim.add_argument("--report", default=str(BASELINE_REPORT))
    head_trim.add_argument("--source", default="brandonyang/dual-lidar-umi-relative")
    head_trim.add_argument("--candidate", action="append", type=float)
    head_trim.add_argument("--workers", type=int, default=8)
    head_trim.add_argument("--all-episodes", action="store_true")
    head_trim.add_argument("--output", required=True)
    trajectory = subparsers.add_parser(
        "sweep-trajectory", help="compare unchanged-gate continuous optimizer profiles"
    )
    trajectory.add_argument("--report", default=str(BASELINE_REPORT))
    trajectory.add_argument("--source", default="brandonyang/dual-lidar-umi-relative")
    trajectory.add_argument("--revision", default="main")
    trajectory.add_argument("--episode", action="append", type=int)
    trajectory.add_argument(
        "--tracking-failures",
        action="store_true",
        help="select every episode that fails baseline all-frame tracking",
    )
    trajectory.add_argument(
        "--baseline-collision-failures",
        action="store_true",
        help="select baseline tracking-valid episodes that fail either collision gate",
    )
    trajectory.add_argument("--candidates-from")
    trajectory.add_argument("--candidate-profile", choices=sorted(CONFIG_PROFILES))
    trajectory.add_argument(
        "--candidate-filter",
        choices=("not-replayable", "tracking-fail", "tracking-pass-collision-fail"),
    )
    trajectory.add_argument(
        "--profile", action="append", choices=sorted(CONFIG_PROFILES), required=True
    )
    trajectory.add_argument("--workers", type=int, default=4)
    trajectory.add_argument("--output", required=True)
    trajectory.add_argument(
        "--smoothing-screen",
        help="select safe framewise-reachability improvements from this smoothing report",
    )
    portfolio = subparsers.add_parser(
        "portfolio", help="combine unchanged-gate passing candidates without baseline regressions"
    )
    portfolio.add_argument("--report", default=str(BASELINE_REPORT))
    portfolio.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("REPORT", "PROFILE"),
        required=True,
    )
    portfolio.add_argument("--output", required=True)
    smoothing = subparsers.add_parser(
        "screen-smoothing", help="screen stronger event-preserving smoothing with IK"
    )
    smoothing.add_argument("--report", default=str(BASELINE_REPORT))
    smoothing.add_argument("--source", default="brandonyang/dual-lidar-umi-relative")
    smoothing.add_argument("--revision", default="main")
    smoothing.add_argument("--episode", action="append", type=int)
    smoothing.add_argument("--candidates-from")
    smoothing.add_argument("--candidate-profile", choices=sorted(CONFIG_PROFILES))
    smoothing.add_argument(
        "--candidate-filter",
        choices=("not-replayable", "tracking-fail", "tracking-pass-collision-fail"),
    )
    smoothing.add_argument("--workers", type=int, default=8)
    smoothing.add_argument("--output", required=True)
    home = subparsers.add_parser(
        "screen-home", help="screen global YAM start-posture registrations with IK"
    )
    home.add_argument("--report", default=str(BASELINE_REPORT))
    home.add_argument("--source", default="brandonyang/dual-lidar-umi-relative")
    home.add_argument("--episode", action="append", type=int)
    home.add_argument("--all-episodes", action="store_true")
    home.add_argument("--workers", type=int, default=8)
    home.add_argument("--output", required=True)
    export = subparsers.add_parser(
        "export-portfolio", help="build the final portfolio as an observation-only dataset"
    )
    export.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO_REPORT))
    export.add_argument("--source", default="brandonyang/dual-lidar-umi-relative")
    export.add_argument("--revision", default="main")
    export.add_argument("--workers", type=int, default=8)
    export.add_argument("--no-videos", action="store_true")
    export.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "baseline":
        summary = summarize_baseline(load_baseline(args.report))
        payload = json.dumps(summary, indent=2) + "\n"
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(f"wrote {output}")
        else:
            print(payload, end="")
    elif args.command == "screen-head-trim":
        candidates = tuple(args.candidate or (0.0, 1.0, 2.0, 3.0))
        report = screen_head_trim(
            source_name=args.source,
            baseline_report=args.report,
            candidates_s=candidates,
            workers=args.workers,
            tracking_failures_only=not args.all_episodes,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "sweep-trajectory":
        episode_indices = list(args.episode or ())
        smoothing_overrides = (
            select_safe_smoothing_improvements(args.smoothing_screen)
            if args.smoothing_screen
            else {}
        )
        episode_indices.extend(smoothing_overrides)
        if args.tracking_failures:
            baseline = load_baseline(args.report)
            episode_indices.extend(
                int(row["episode_index"])
                for row in baseline["per_episode"]
                if not row["validation"]["passes_tracking_accuracy"]
            )
        if args.baseline_collision_failures:
            baseline = load_baseline(args.report)
            episode_indices.extend(
                int(row["episode_index"])
                for row in baseline["per_episode"]
                if row["validation"]["passes_tracking_accuracy"]
                and (
                    not row["validation"]["passes_interarm_clearance"]
                    or not row["validation"]["passes_self_collision_clearance"]
                )
            )
        candidate_arguments = (
            args.candidates_from,
            args.candidate_profile,
            args.candidate_filter,
        )
        if any(candidate_arguments):
            if not all(candidate_arguments):
                raise SystemExit(
                    "--candidates-from, --candidate-profile, and --candidate-filter "
                    "must be provided together"
                )
            episode_indices.extend(
                select_experiment_episodes(
                    args.candidates_from,
                    profile=args.candidate_profile,
                    criterion=args.candidate_filter,
                )
            )
        if not episode_indices:
            raise SystemExit("provide --episode or --tracking-failures")
        report = run_trajectory_sweep(
            episode_indices=episode_indices,
            profiles=tuple(args.profile),
            source_name=args.source,
            source_revision=args.revision,
            baseline_report=args.report,
            workers=args.workers,
            smoothing_overrides=smoothing_overrides,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "portfolio":
        report = build_solver_portfolio(
            baseline_report=args.report,
            experiments=[(path, profile) for path, profile in args.candidate],
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "screen-smoothing":
        episode_indices = list(args.episode or ())
        candidate_arguments = (
            args.candidates_from,
            args.candidate_profile,
            args.candidate_filter,
        )
        if any(candidate_arguments):
            if not all(candidate_arguments):
                raise SystemExit(
                    "--candidates-from, --candidate-profile, and --candidate-filter "
                    "must be provided together"
                )
            episode_indices.extend(
                select_experiment_episodes(
                    args.candidates_from,
                    profile=args.candidate_profile,
                    criterion=args.candidate_filter,
                )
            )
        if not episode_indices:
            raise SystemExit("provide --episode or a complete --candidates-from selection")
        report = screen_smoothing_strength(
            episode_indices=episode_indices,
            source_name=args.source,
            source_revision=args.revision,
            baseline_report=args.report,
            workers=args.workers,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "screen-home":
        episode_indices = list(args.episode or ())
        if args.all_episodes:
            episode_indices.extend(
                int(row["episode_index"]) for row in load_baseline(args.report)["per_episode"]
            )
        if not episode_indices:
            raise SystemExit("provide --episode or --all-episodes")
        report = screen_global_home_postures(
            episode_indices=episode_indices,
            source_name=args.source,
            baseline_report=args.report,
            workers=args.workers,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    elif args.command == "export-portfolio":
        summary = build_portfolio_dataset(
            destination=args.output,
            portfolio_path=args.portfolio,
            source_name=args.source,
            source_revision=args.revision,
            include_videos=not args.no_videos,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
