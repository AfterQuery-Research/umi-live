from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_filtering.analysis import analyze_dataset, write_analysis
from dataset_filtering.combined_export import (
    CARTESIAN_REPO_ID,
    JOINT_REPO_ID,
    build_combined_datasets,
    push_combined_dataset,
    validate_combined_dataset,
    verify_remote_combined_dataset,
)
from dataset_filtering.combined_pipeline import (
    CombinedSource,
    process_combined_source_shard,
    process_combined_sources,
)
from dataset_filtering.export_dataset import (
    DEFAULT_FILTER_REPORT,
    DEFAULT_FILTERED_REPO_ID,
    build_filtered_dataset,
    push_filtered_dataset,
    validate_filtered_dataset,
)
from dataset_filtering.hf_dataset import DEFAULT_REPO_ID, LeRobotSource, as_pose_episode
from dataset_filtering.progress_viewer import create_progress_app
from dataset_filtering.reachability import (
    analyze_continuous_replayability,
    analyze_smoothed_reachability,
    write_reachability,
)
from dataset_filtering.retarget import playback_link_points, retarget_pose_episode
from dataset_filtering.smoothing import benchmark_smoothers, with_smoothed_pose
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    optimize_pose_episode,
)
from dataset_filtering.viewer import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="umi-filter", description="Local-only UMI smoothing and YAM IK retargeting"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_source_arguments(subparsers.add_parser("episodes", help="list dataset episodes"))

    analyze = subparsers.add_parser("analyze", help="quantify noise and compare smoothers")
    _add_source_arguments(analyze)
    analyze.add_argument("--episode", type=int, action="append", dest="episodes")
    analyze.add_argument("--no-smoothing-benchmark", action="store_true")
    analyze.add_argument(
        "--output", type=Path, default=Path("reports/dual-lidar-umi-noise-and-smoothing.json")
    )

    retarget = subparsers.add_parser("retarget", help="smooth and solve one episode for YAM")
    _add_source_arguments(retarget)
    retarget.add_argument("--episode", type=int, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--solver", choices=("continuous", "greedy"), default="continuous")
    retarget.add_argument("--no-toppra", action="store_true")
    retarget.add_argument("--velocity-scale", type=float, default=1.0)
    retarget.add_argument("--acceleration-limit", type=float, default=20.0)
    retarget.add_argument("--table-height", type=float)
    retarget.add_argument(
        "--tail-freeze-grace-seconds",
        type=float,
        default=0.5,
        help="freeze an invalid put-down tail this long after each arm's last gripper event",
    )
    retarget.add_argument(
        "--source-camera-translation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    retarget.add_argument(
        "--source-camera-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.0),
    )
    retarget.add_argument(
        "--yam-camera-translation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    retarget.add_argument(
        "--yam-camera-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.0),
    )
    retarget.add_argument(
        "--camera-extrinsics-calibrated",
        action="store_true",
        help="mark both supplied tool-to-camera transforms as measured calibrations",
    )

    reachability = subparsers.add_parser(
        "reachability", help="measure full-dataset smoothed YAM IK reachability"
    )
    _add_source_arguments(reachability)
    reachability.add_argument("--episode", type=int, action="append", dest="episodes")
    reachability.add_argument("--workers", type=int)
    reachability.add_argument("--solver", choices=("continuous", "greedy"), default="continuous")
    reachability.add_argument(
        "--trim-end-seconds",
        type=float,
        default=0.0,
        help="read-only experiment: remove this duration from every episode in memory",
    )
    reachability.add_argument(
        "--tail-freeze-grace-seconds",
        type=float,
        default=0.5,
        help="freeze an invalid put-down tail this long after each arm's last gripper event",
    )
    reachability.add_argument("--output", type=Path)

    viewer = subparsers.add_parser("viewer", help="start the episode-selectable local viewer")
    _add_source_arguments(viewer)
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.add_argument(
        "--precomputed-results",
        type=Path,
        help="load completed continuous solves from a quality-gate episodes directory",
    )

    build = subparsers.add_parser(
        "build-filtered-dataset",
        help="build a local observation-only dataset from complete continuous passes",
    )
    _add_source_arguments(build)
    build.add_argument("--report", type=Path, default=DEFAULT_FILTER_REPORT)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--no-videos", action="store_true")
    build.add_argument("--workers", type=int, default=8)

    validate = subparsers.add_parser(
        "validate-filtered-dataset", help="validate a local filtered LeRobot dataset"
    )
    validate.add_argument("--dataset", type=Path, required=True)

    push = subparsers.add_parser(
        "push-filtered-dataset", help="validate and upload a built filtered dataset"
    )
    push.add_argument("--dataset", type=Path, required=True)
    push.add_argument("--repo-id", default=DEFAULT_FILTERED_REPO_ID)
    push.add_argument("--private", action="store_true")

    combined = subparsers.add_parser(
        "process-combined",
        help="run the latest filtering cascade over original and relative UMI sources",
    )
    combined.add_argument("--original-revision", required=True)
    combined.add_argument("--relative-revision", required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--workers", type=int, default=8)

    combined_shard = subparsers.add_parser(
        "process-combined-shard",
        help="precompute a source episode range into a shared combined result cache",
    )
    combined_shard.add_argument("--source", required=True)
    combined_shard.add_argument("--revision", required=True)
    combined_shard.add_argument("--combined-offset", type=int, required=True)
    combined_shard.add_argument("--episode-from", type=int, required=True)
    combined_shard.add_argument("--episode-to", type=int, required=True)
    combined_shard.add_argument("--output", type=Path, required=True)
    combined_shard.add_argument("--workers", type=int, default=4)

    combined_build = subparsers.add_parser(
        "build-combined-datasets",
        help="build matching Cartesian and YAM-joint datasets from a combined report",
    )
    combined_build.add_argument("--report", type=Path, required=True)
    combined_build.add_argument("--cartesian-output", type=Path, required=True)
    combined_build.add_argument("--joint-output", type=Path, required=True)
    combined_build.add_argument(
        "--umi1-gripper-calibration",
        type=Path,
        help="interactive two-point calibration JSON for UMI1; built-in limits if omitted",
    )
    combined_build.add_argument(
        "--umi2-gripper-calibration",
        type=Path,
        help="interactive two-point calibration JSON for UMI2; built-in limits if omitted",
    )
    combined_build.add_argument("--workers", type=int, default=12)

    combined_validate = subparsers.add_parser(
        "validate-combined-dataset", help="validate one built combined dataset"
    )
    combined_validate.add_argument("--dataset", type=Path, required=True)
    combined_validate.add_argument("--variant", choices=("cartesian", "joint"), required=True)

    combined_push = subparsers.add_parser(
        "push-combined-dataset", help="validate and upload one combined dataset"
    )
    combined_push.add_argument("--dataset", type=Path, required=True)
    combined_push.add_argument("--variant", choices=("cartesian", "joint"), required=True)
    combined_push.add_argument("--repo-id")
    combined_push.add_argument("--private", action="store_true")
    combined_push.add_argument("--workers", type=int, default=12)

    combined_remote = subparsers.add_parser(
        "verify-remote-combined-dataset",
        help="verify a committed combined dataset on the Hugging Face Hub",
    )
    combined_remote.add_argument("--repo-id", required=True)
    combined_remote.add_argument("--variant", choices=("cartesian", "joint"), required=True)

    progress = subparsers.add_parser(
        "progress-viewer", help="serve a live dashboard for a combined filtering run"
    )
    progress.add_argument("--results", type=Path, required=True)
    progress.add_argument("--total", type=int, default=298)
    progress.add_argument("--host", default="127.0.0.1")
    progress.add_argument("--port", type=int, default=8766)

    args = parser.parse_args()
    if args.command == "progress-viewer":
        import uvicorn

        uvicorn.run(
            create_progress_app(args.results, total_episodes=args.total),
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "verify-remote-combined-dataset":
        print(
            json.dumps(verify_remote_combined_dataset(args.repo_id, variant=args.variant), indent=2)
        )
        return
    if args.command == "validate-combined-dataset":
        print(json.dumps(validate_combined_dataset(args.dataset, variant=args.variant), indent=2))
        return
    if args.command == "push-combined-dataset":
        default_repo = CARTESIAN_REPO_ID if args.variant == "cartesian" else JOINT_REPO_ID
        print(
            push_combined_dataset(
                args.dataset,
                variant=args.variant,
                repo_id=args.repo_id or default_repo,
                private=args.private,
                workers=args.workers,
            )
        )
        return
    if args.command == "build-combined-datasets":
        summary = build_combined_datasets(
            report_path=args.report,
            cartesian_destination=args.cartesian_output,
            joint_destination=args.joint_output,
            umi1_gripper_calibration=args.umi1_gripper_calibration,
            umi2_gripper_calibration=args.umi2_gripper_calibration,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "validate-filtered-dataset":
        print(json.dumps(validate_filtered_dataset(args.dataset), indent=2))
        return
    if args.command == "push-filtered-dataset":
        print(push_filtered_dataset(args.dataset, repo_id=args.repo_id, private=args.private))
        return
    if args.command == "process-combined":
        report = process_combined_sources(
            sources=(
                CombinedSource("brandonyang/dual-lidar-umi", args.original_revision),
                CombinedSource("brandonyang/dual-lidar-umi-relative", args.relative_revision),
            ),
            destination=args.output,
            workers=args.workers,
        )
        summary = {key: value for key, value in report.items() if key != "per_episode"}
        print(json.dumps(summary, indent=2))
        return
    if args.command == "process-combined-shard":
        summary = process_combined_source_shard(
            source_spec=CombinedSource(args.source, args.revision),
            combined_offset=args.combined_offset,
            source_episode_from=args.episode_from,
            source_episode_to=args.episode_to,
            destination=args.output,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
        return
    source = LeRobotSource(args.source, revision=args.revision, cache_dir=args.cache_dir)
    if args.command == "build-filtered-dataset":
        summary = build_filtered_dataset(
            source,
            report_path=args.report,
            destination=args.output,
            include_videos=not args.no_videos,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.command == "episodes":
        for summary in source.episode_summaries():
            print(
                f"{summary.episode_index:4d}  {summary.length:5d} frames  "
                f"{summary.duration_s:7.2f}s  {summary.task}"
            )
        return
    if args.command == "analyze":
        report = analyze_dataset(
            source,
            episode_indices=args.episodes,
            benchmark_smoothing=not args.no_smoothing_benchmark,
        )
        path = write_analysis(report, args.output)
        print(f"wrote {path}")
        return
    if args.command == "retarget":
        episode = as_pose_episode(source.load_episode(args.episode))
        smoothing = benchmark_smoothers(episode)[0]
        filtered = with_smoothed_pose(episode, smoothing)
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        smoothing_metadata = {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "metrics": smoothing.metrics,
        }
        grippers = (
            episode.grippers
            if episode.grippers is not None
            else np.zeros((len(episode.timestamps), 2))
        )
        if args.solver == "continuous":
            config = TrajectoryOptimizationConfig(
                planning_velocity_scale=args.velocity_scale,
                planning_acceleration_limit_rad_s2=args.acceleration_limit,
                table_height_m=args.table_height,
                tail_freeze_grace_s=args.tail_freeze_grace_seconds,
                source_tool_to_camera_translation_m=tuple(args.source_camera_translation),
                source_tool_to_camera_rpy_rad=tuple(np.deg2rad(args.source_camera_rpy_deg)),
                yam_tool_to_camera_translation_m=tuple(args.yam_camera_translation),
                yam_tool_to_camera_rpy_rad=tuple(np.deg2rad(args.yam_camera_rpy_deg)),
                camera_extrinsics_calibrated=args.camera_extrinsics_calibrated,
                prefer_toppra=not args.no_toppra,
            )
            result = optimize_pose_episode(filtered, config=config)
            np.savez_compressed(
                destination,
                timestamps=result.robot_timestamps,
                source_timestamps=result.source_timestamps,
                robot_timestamps=result.robot_timestamps,
                source_video_frame_index=np.arange(len(episode.timestamps)),
                raw_umi_positions=episode.positions,
                raw_umi_quaternions=episode.quaternions,
                smoothed_umi_positions=filtered.positions,
                smoothed_umi_quaternions=filtered.quaternions,
                target_positions=result.target_positions_world,
                target_quaternions=result.target_quaternions_world,
                achieved_positions=result.achieved_positions_world,
                achieved_quaternions=result.achieved_quaternions_world,
                left_q=result.q[:, 0],
                right_q=result.q[:, 1],
                left_valid=result.frame_valid[:, 0],
                right_valid=result.frame_valid[:, 1],
                position_error_m=result.position_error_m,
                orientation_error_rad=result.orientation_error_rad,
                camera_position_error_m=result.camera_position_error_m,
                camera_orientation_error_rad=result.camera_orientation_error_rad,
                video_alignment_weight=result.video_alignment_weight,
                left_gripper_event_frames=result.event_indices[0],
                right_gripper_event_frames=result.event_indices[1],
                robot_links=result.link_points_world,
                grippers=grippers,
                metadata_json=json.dumps(
                    {
                        "source": source.source,
                        "episode_index": episode.episode_index,
                        "smoothing": smoothing_metadata,
                        "retarget": result.metadata,
                    }
                ),
            )
            print(
                f"wrote {destination} (source dataset unchanged)\n"
                f"continuous={result.metadata['continuously_replayable']} "
                f"contacts={result.metadata['contact_events']['strictly_feasible']}/"
                f"{result.metadata['contact_events']['total']} "
                f"timing={result.metadata['timing']['method']}"
            )
            return

        result = retarget_pose_episode(filtered, max_iterations=75)
        links = playback_link_points(result)
        np.savez_compressed(
            destination,
            timestamps=episode.timestamps,
            target_positions=result.target_positions_world,
            target_quaternions=result.target_quaternions_world,
            left_q=result.arms[0].playback_q,
            right_q=result.arms[1].playback_q,
            left_reachable=result.arms[0].reachable,
            right_reachable=result.arms[1].reachable,
            robot_links=links,
            grippers=grippers,
            metadata_json=json.dumps(
                {
                    "source": source.source,
                    "episode_index": episode.episode_index,
                    "smoothing": smoothing_metadata,
                    "retarget": result.metadata,
                }
            ),
        )
        print(f"wrote {destination} (source dataset unchanged)")
        return
    if args.command == "reachability":
        if args.solver == "continuous":
            report = analyze_continuous_replayability(
                source,
                episode_indices=args.episodes,
                workers=args.workers,
                progress=True,
                trim_end_s=args.trim_end_seconds,
                tail_freeze_grace_s=args.tail_freeze_grace_seconds,
            )
            destination = args.output or Path(
                "reports/dual-lidar-umi-tail-freeze-v2-replayability.json"
            )
            path = write_reachability(report, destination)
            tracking = report["frame_weighted_tracking"]
            replayability = report["episode_replayability"]
            print(
                f"wrote {path} (source dataset unchanged)\n"
                f"ik-solvable={replayability['ik_trajectory_solvable']}/"
                f"{report['episodes']} "
                f"continuous={replayability['continuously_replayable']}/"
                f"{report['episodes']} "
                f"both-tracking={tracking['both_target_fraction_10mm_8deg']:.3%}"
            )
        else:
            report = analyze_smoothed_reachability(
                source, episode_indices=args.episodes, workers=args.workers
            )
            destination = args.output or Path("reports/dual-lidar-umi-greedy-ik-reachability.json")
            path = write_reachability(report, destination)
            fractions = report["frame_weighted_reachability"]
            print(
                f"wrote {path} (source dataset unchanged)\n"
                f"left={fractions['umi1_left_fraction']:.3%} "
                f"right={fractions['umi2_right_fraction']:.3%} "
                f"both={fractions['both_arms_same_frame_fraction']:.3%}"
            )
        return
    if args.command == "viewer":
        import uvicorn

        app = create_app(
            args.source,
            revision=args.revision,
            cache_dir=args.cache_dir,
            precomputed_results=args.precomputed_results,
        )
        uvicorn.run(app, host=args.host, port=args.port)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default=DEFAULT_REPO_ID, help="Hub repo id or local root")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)


if __name__ == "__main__":
    main()
