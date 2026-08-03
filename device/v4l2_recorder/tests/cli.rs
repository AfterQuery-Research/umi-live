use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn recorder_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_robot-umi-v4l2-recorder"))
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    env::temp_dir().join(format!(
        "robot_umi_v4l2_{name}_{}_{}",
        std::process::id(),
        nanos
    ))
}

#[test]
fn help_prints_usage_and_exits_successfully() {
    let output = Command::new(recorder_bin()).arg("--help").output().unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
}

#[test]
fn invalid_numeric_argument_fails_before_capture() {
    let dir = temp_dir("invalid_arg");
    fs::create_dir_all(&dir).unwrap();
    let output_path = dir.join("camera_front.mjpeg.avi");
    let timestamps_path = dir.join("camera_frames.jsonl");
    let stats_path = dir.join("camera_stats.json");

    let output = Command::new(recorder_bin())
        .arg("--output")
        .arg(&output_path)
        .arg("--timestamps")
        .arg(&timestamps_path)
        .arg("--stats")
        .arg(&stats_path)
        .arg("--fps")
        .arg("not-a-number")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value for --fps"));
    assert!(!output_path.exists());
    assert!(!timestamps_path.exists());
    assert!(!stats_path.exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg(target_os = "linux")]
fn nonexistent_v4l2_device_fails_before_artifacts_are_written() {
    let dir = temp_dir("missing_device");
    fs::create_dir_all(&dir).unwrap();
    let output_path = dir.join("camera_front.mjpeg.avi");
    let timestamps_path = dir.join("camera_frames.jsonl");
    let stats_path = dir.join("camera_stats.json");
    let missing_device = dir.join("missing-video-device");

    let output = Command::new(recorder_bin())
        .arg("--device")
        .arg(&missing_device)
        .arg("--output")
        .arg(&output_path)
        .arg("--timestamps")
        .arg(&timestamps_path)
        .arg("--stats")
        .arg(&stats_path)
        .arg("--seconds")
        .arg("1")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("recording failed: open"));
    assert!(!output_path.exists());
    assert!(!timestamps_path.exists());
    assert!(!stats_path.exists());

    let _ = fs::remove_dir_all(&dir);
}
