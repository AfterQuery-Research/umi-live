#![cfg(target_os = "linux")]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn recorder_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_umi-live-v4l2-recorder"))
}

fn temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    env::temp_dir().join(format!(
        "umi_live_v4l2_hardware_e2e_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn env_or_default(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn json_number(contents: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\": ");
    let start = contents
        .find(&needle)
        .unwrap_or_else(|| panic!("missing {key}"))
        + needle.len();
    let value = contents[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    value.parse().unwrap_or_else(|_| panic!("invalid {key}"))
}

fn json_bool(contents: &str, key: &str) -> bool {
    let needle = format!("\"{key}\": ");
    let start = contents
        .find(&needle)
        .unwrap_or_else(|| panic!("missing {key}"))
        + needle.len();
    if contents[start..].starts_with("true") {
        true
    } else if contents[start..].starts_with("false") {
        false
    } else {
        panic!("invalid {key}")
    }
}

#[test]
#[ignore = "requires a real V4L2 camera such as /dev/video0"]
fn records_real_camera_video_sidecar_and_stats() {
    let dir = temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let output_path = dir.join("camera_front.mjpeg.avi");
    let timestamps_path = dir.join("camera_frames.jsonl");
    let stats_path = dir.join("camera_stats.json");

    let output = Command::new(recorder_bin())
        .arg("--device")
        .arg(env_or_default("UMI_LIVE_V4L2_DEVICE", "/dev/video0"))
        .arg("--output")
        .arg(&output_path)
        .arg("--timestamps")
        .arg(&timestamps_path)
        .arg("--stats")
        .arg(&stats_path)
        .arg("--width")
        .arg(env_or_default("UMI_LIVE_V4L2_WIDTH", "1920"))
        .arg("--height")
        .arg(env_or_default("UMI_LIVE_V4L2_HEIGHT", "1200"))
        .arg("--fps")
        .arg(env_or_default("UMI_LIVE_V4L2_FPS", "30"))
        .arg("--seconds")
        .arg(env_or_default("UMI_LIVE_V4L2_SECONDS", "3"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "recorder failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.metadata().unwrap().len() > 0);
    assert!(timestamps_path.metadata().unwrap().len() > 0);
    assert!(stats_path.metadata().unwrap().len() > 0);

    let sidecar = fs::read_to_string(&timestamps_path).unwrap();
    let stats = fs::read_to_string(&stats_path).unwrap();
    let sidecar_rows = sidecar.lines().filter(|line| !line.is_empty()).count();
    let camera_frames = json_number(&stats, "camera_frame_count");
    let avi_frames = json_number(&stats, "avi_frame_count");

    assert!(camera_frames > 0);
    assert_eq!(sidecar_rows, camera_frames as usize);
    assert_eq!(avi_frames, camera_frames);
    assert_eq!(json_number(&stats, "camera_dropped_sequence_frames"), 0);
    assert_eq!(json_number(&stats, "camera_queue_full_count"), 0);
    assert!(json_bool(&stats, "camera_dataset_eligible"));

    let first_row = sidecar.lines().next().unwrap();
    assert!(first_row.contains("\"frame_index\":0"));
    assert!(first_row.contains("\"timestamp_clock\":\"monotonic\""));
    assert!(first_row.contains("\"timestamp_source\":\"start_of_exposure\""));

    let _ = fs::remove_dir_all(&dir);
}
