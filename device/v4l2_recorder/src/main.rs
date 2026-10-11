#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_WIDTH: u32 = 1920;
const DEFAULT_HEIGHT: u32 = 1200;
const DEFAULT_FPS: u32 = 30;
const DEFAULT_WARMUP_FRAMES: u32 = 8;
const DEFAULT_BUFFER_COUNT: u32 = 32;
const DEFAULT_WRITER_QUEUE_FRAMES: usize = 120;
const CAMERA_SYNC_MODE: &str = "v4l2_monotonic_start_of_exposure_v1";
const AVI_U32_MAX: u64 = u32::MAX as u64;
const AVI_INDEX_ENTRY_BYTES: u64 = 16;

#[derive(Debug, Clone)]
struct Args {
    device: PathBuf,
    output: PathBuf,
    timestamps: PathBuf,
    stats: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    exposure: Option<i64>,
    seconds: Option<u64>,
    warmup_frames: u32,
    buffer_count: u32,
    writer_queue_frames: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            device: PathBuf::from("/dev/video0"),
            output: PathBuf::from("camera_front.mjpeg.avi"),
            timestamps: PathBuf::from("camera_frames.jsonl"),
            stats: PathBuf::from("camera_stats.json"),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            fps: DEFAULT_FPS,
            exposure: None,
            seconds: None,
            warmup_frames: DEFAULT_WARMUP_FRAMES,
            buffer_count: DEFAULT_BUFFER_COUNT,
            writer_queue_frames: DEFAULT_WRITER_QUEUE_FRAMES,
        };
        let mut iter = env::args().skip(1);
        while let Some(flag) = iter.next() {
            if flag == "-h" || flag == "--help" {
                return Err(usage());
            }
            let value = iter
                .next()
                .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))?;
            match flag.as_str() {
                "--device" => args.device = PathBuf::from(value),
                "--output" => args.output = PathBuf::from(value),
                "--timestamps" => args.timestamps = PathBuf::from(value),
                "--stats" => args.stats = PathBuf::from(value),
                "--width" => args.width = parse_value(&flag, &value)?,
                "--height" => args.height = parse_value(&flag, &value)?,
                "--fps" => args.fps = parse_value(&flag, &value)?,
                "--exposure" => args.exposure = Some(parse_value(&flag, &value)?),
                "--seconds" => args.seconds = Some(parse_value(&flag, &value)?),
                "--warmup-frames" => args.warmup_frames = parse_value(&flag, &value)?,
                "--buffer-count" => args.buffer_count = parse_value(&flag, &value)?,
                "--writer-queue-frames" => args.writer_queue_frames = parse_value(&flag, &value)?,
                _ => return Err(format!("unknown argument: {flag}\n{}", usage())),
            }
        }
        Ok(args)
    }
}

fn parse_value<T>(flag: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn usage() -> String {
    "usage: umi-live-v4l2-recorder --device /dev/video0 --output camera_front.mjpeg.avi --timestamps camera_frames.jsonl --stats camera_stats.json [--width 1920] [--height 1200] [--fps 30] [--exposure 100] [--seconds N] [--warmup-frames 8] [--buffer-count 32] [--writer-queue-frames 120]".to_string()
}

#[derive(Debug, Clone, PartialEq)]
struct FrameMeta {
    frame_index: u32,
    v4l2_sequence: u32,
    time_monotonic_ns: u64,
    time_unix_ns: u64,
    bytesused: u32,
    buffer_index: u32,
    v4l2_flags: u32,
    timestamp_clock: &'static str,
    timestamp_source: &'static str,
}

impl FrameMeta {
    fn to_json_line(&self) -> String {
        format!(
            "{{\"frame_index\":{},\"v4l2_sequence\":{},\"time_monotonic_ns\":{},\"time_unix_ns\":{},\"bytesused\":{},\"buffer_index\":{},\"v4l2_flags\":{},\"timestamp_clock\":\"{}\",\"timestamp_source\":\"{}\"}}\n",
            self.frame_index,
            self.v4l2_sequence,
            self.time_monotonic_ns,
            self.time_unix_ns,
            self.bytesused,
            self.buffer_index,
            self.v4l2_flags,
            self.timestamp_clock,
            self.timestamp_source,
        )
    }
}

#[derive(Debug, Clone)]
struct CaptureStats {
    frame_count: u32,
    fps: u32,
    dropped_sequence_frames: u32,
    previous_sequence: Option<u32>,
    previous_timestamp_ns: Option<u64>,
    non_monotonic_timestamps: u32,
    bad_timestamp_metadata: u32,
    timestamp_ns: Vec<u64>,
}

impl Default for CaptureStats {
    fn default() -> Self {
        Self {
            frame_count: 0,
            fps: DEFAULT_FPS,
            dropped_sequence_frames: 0,
            previous_sequence: None,
            previous_timestamp_ns: None,
            non_monotonic_timestamps: 0,
            bad_timestamp_metadata: 0,
            timestamp_ns: Vec::new(),
        }
    }
}

impl CaptureStats {
    fn new(fps: u32) -> Self {
        Self {
            fps,
            ..Self::default()
        }
    }

    fn observe(&mut self, meta: &FrameMeta) {
        if let Some(previous) = self.previous_sequence
            && meta.v4l2_sequence > previous + 1
        {
            self.dropped_sequence_frames += meta.v4l2_sequence - previous - 1;
        }
        if let Some(previous) = self.previous_timestamp_ns
            && meta.time_monotonic_ns <= previous
        {
            self.non_monotonic_timestamps += 1;
        }
        if meta.timestamp_clock != "monotonic" || meta.timestamp_source != "start_of_exposure" {
            self.bad_timestamp_metadata += 1;
        }
        self.previous_sequence = Some(meta.v4l2_sequence);
        self.previous_timestamp_ns = Some(meta.time_monotonic_ns);
        self.timestamp_ns.push(meta.time_monotonic_ns);
        self.frame_count += 1;
    }

    fn delta_ms(&self) -> Vec<f64> {
        // Signed math: cameras can deliver a backwards timestamp (observe()
        // counts them), and u64 subtraction would wrap to garbage or panic.
        self.timestamp_ns
            .windows(2)
            .map(|pair| (pair[1] as i128 - pair[0] as i128) as f64 / 1_000_000.0)
            .collect()
    }

    fn delta_summary(&self) -> DeltaSummary {
        DeltaSummary::from_samples(&self.delta_ms(), self.fps)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct DeltaSummary {
    count: usize,
    min_ms: Option<f64>,
    max_ms: Option<f64>,
    mean_ms: Option<f64>,
    bad_delta_count: usize,
}

impl DeltaSummary {
    fn from_samples(samples: &[f64], fps: u32) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let expected = 1000.0 / fps as f64;
        let sum: f64 = samples.iter().sum();
        Self {
            count: samples.len(),
            min_ms: samples.iter().copied().reduce(f64::min),
            max_ms: samples.iter().copied().reduce(f64::max),
            mean_ms: Some(sum / samples.len() as f64),
            bad_delta_count: samples
                .iter()
                .filter(|sample| (*sample - expected).abs() > 5.0)
                .count(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct EnqueueStats {
    queue_full_count: u32,
    total_enqueue_block_ms: f64,
    max_enqueue_block_ms: f64,
    total_copy_ms: f64,
    max_copy_ms: f64,
}

impl EnqueueStats {
    fn observe_copy(&mut self, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_copy_ms += ms;
        self.max_copy_ms = self.max_copy_ms.max(ms);
    }

    fn observe_enqueue_block(&mut self, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.queue_full_count += 1;
        self.total_enqueue_block_ms += ms;
        self.max_enqueue_block_ms = self.max_enqueue_block_ms.max(ms);
    }

    fn mean_copy_ms(&self, frame_count: u32) -> f64 {
        if frame_count == 0 {
            0.0
        } else {
            self.total_copy_ms / frame_count as f64
        }
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn dataset_eligible(
    stats: &CaptureStats,
    enqueue_stats: &EnqueueStats,
    writer_outcome: &WriterOutcome,
    delta: &DeltaSummary,
    error: Option<&str>,
) -> bool {
    stats.frame_count > 0
        && error.is_none()
        && stats.dropped_sequence_frames == 0
        && enqueue_stats.queue_full_count == 0
        && stats.non_monotonic_timestamps == 0
        && stats.bad_timestamp_metadata == 0
        && delta.bad_delta_count == 0
        && writer_outcome.frames_written == stats.frame_count as usize
}

fn write_stats(
    path: &Path,
    args: &Args,
    stats: &CaptureStats,
    enqueue_stats: &EnqueueStats,
    writer_outcome: &WriterOutcome,
    delta: &DeltaSummary,
    error: Option<&str>,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let eligible = dataset_eligible(stats, enqueue_stats, writer_outcome, delta, error);
    let json = format!(
        concat!(
            "{{\n",
            "  \"camera_sync_mode\": \"{}\",\n",
            "  \"camera_width\": {},\n",
            "  \"camera_height\": {},\n",
            "  \"camera_fps\": {},\n",
            "  \"camera_frame_count\": {},\n",
            "  \"avi_frame_count\": {},\n",
            "  \"camera_dropped_sequence_frames\": {},\n",
            "  \"camera_non_monotonic_timestamps\": {},\n",
            "  \"camera_bad_timestamp_metadata\": {},\n",
            "  \"camera_queue_full_count\": {},\n",
            "  \"camera_dataset_eligible\": {},\n",
            "  \"camera_error\": \"{}\",\n",
            "  \"writer_bytes\": {},\n",
            "  \"delta_count\": {},\n",
            "  \"delta_ms_min\": {:.3},\n",
            "  \"delta_ms_max\": {:.3},\n",
            "  \"delta_ms_mean\": {:.3},\n",
            "  \"delta_bad_count\": {},\n",
            "  \"max_enqueue_block_ms\": {:.3},\n",
            "  \"max_copy_ms\": {:.3},\n",
            "  \"mean_copy_ms\": {:.3}\n",
            "}}\n"
        ),
        CAMERA_SYNC_MODE,
        args.width,
        args.height,
        args.fps,
        stats.frame_count,
        writer_outcome.frames_written,
        stats.dropped_sequence_frames,
        stats.non_monotonic_timestamps,
        stats.bad_timestamp_metadata,
        enqueue_stats.queue_full_count,
        eligible,
        json_escape(error.unwrap_or_default()),
        writer_outcome.bytes_written,
        delta.count,
        delta.min_ms.unwrap_or_default(),
        delta.max_ms.unwrap_or_default(),
        delta.mean_ms.unwrap_or_default(),
        delta.bad_delta_count,
        enqueue_stats.max_enqueue_block_ms,
        enqueue_stats.max_copy_ms,
        enqueue_stats.mean_copy_ms(stats.frame_count),
    );
    std::fs::write(path, json)
}

struct SidecarWriter {
    writer: BufWriter<File>,
}

impl SidecarWriter {
    fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    fn write_frame(&mut self, meta: &FrameMeta) -> io::Result<()> {
        self.writer.write_all(meta.to_json_line().as_bytes())
    }

    fn finish(mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AviIndexEntry {
    flags: u32,
    offset: u32,
    size: u32,
}

struct AviWriter {
    file: File,
    entries: Vec<AviIndexEntry>,
    riff_size_pos: u64,
    avih_frames_pos: u64,
    strh_length_pos: u64,
    movi_size_pos: u64,
    movi_data_start: u64,
    last_good_end: u64,
}

struct FramePacket {
    meta: FrameMeta,
    data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
struct WriterOutcome {
    frames_written: usize,
    bytes_written: u64,
    error: Option<String>,
}

fn drain_packets(rx: &Receiver<FramePacket>) {
    for _packet in rx.iter() {}
}

fn spawn_writer(
    output: PathBuf,
    timestamps: PathBuf,
    queue_capacity: usize,
    width: u32,
    height: u32,
    fps: u32,
) -> (SyncSender<FramePacket>, JoinHandle<WriterOutcome>) {
    let (tx, rx) = sync_channel::<FramePacket>(queue_capacity);
    let handle = thread::spawn(move || {
        let avi = match AviWriter::create(&output, width, height, fps) {
            Ok(avi) => avi,
            Err(err) => {
                drain_packets(&rx);
                return WriterOutcome {
                    frames_written: 0,
                    bytes_written: 0,
                    error: Some(format!("create avi: {err}")),
                };
            }
        };
        let sidecar = match SidecarWriter::create(&timestamps) {
            Ok(sidecar) => sidecar,
            Err(err) => {
                drain_packets(&rx);
                return WriterOutcome {
                    frames_written: 0,
                    bytes_written: 0,
                    error: Some(format!("create sidecar: {err}")),
                };
            }
        };
        run_writer(avi, sidecar, &rx)
    });
    (tx, handle)
}

fn run_writer(
    mut avi: AviWriter,
    mut sidecar: SidecarWriter,
    rx: &Receiver<FramePacket>,
) -> WriterOutcome {
    let mut bytes_written = 0_u64;
    let mut error: Option<String> = None;
    for packet in rx.iter() {
        if let Err(err) = avi.write_frame(&packet.data) {
            error = Some(format!("write avi frame: {err}"));
            break;
        }
        bytes_written += packet.data.len() as u64;
        if let Err(err) = sidecar.write_frame(&packet.meta) {
            error = Some(format!("write sidecar: {err}"));
            break;
        }
    }
    // Finalize even after an error: the frames written so far stay playable
    // (valid headers plus idx1 index) instead of becoming a dead AVI.
    if let Err(err) = sidecar.finish() {
        error.get_or_insert_with(|| format!("finish sidecar: {err}"));
    }
    let frames_written = match avi.finish_salvage() {
        Ok(frames) => frames,
        Err(err) => {
            error.get_or_insert_with(|| format!("finish avi: {err}"));
            0
        }
    };
    // Keep draining so the capture side never blocks on a full queue after an error.
    drain_packets(rx);
    WriterOutcome {
        frames_written,
        bytes_written,
        error,
    }
}

impl AviWriter {
    fn create(path: &Path, width: u32, height: u32, fps: u32) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path)?;

        write_fourcc(&mut file, b"RIFF")?;
        let riff_size_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        write_fourcc(&mut file, b"AVI ")?;

        write_fourcc(&mut file, b"LIST")?;
        let hdrl_size_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        let hdrl_start = stream_pos(&mut file)?;
        write_fourcc(&mut file, b"hdrl")?;

        write_fourcc(&mut file, b"avih")?;
        write_u32(&mut file, 56)?;
        write_u32(&mut file, 1_000_000 / fps)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0x10)?;
        let avih_frames_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 1)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, width)?;
        write_u32(&mut file, height)?;
        for _ in 0..4 {
            write_u32(&mut file, 0)?;
        }

        write_fourcc(&mut file, b"LIST")?;
        let strl_size_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        let strl_start = stream_pos(&mut file)?;
        write_fourcc(&mut file, b"strl")?;

        write_fourcc(&mut file, b"strh")?;
        write_u32(&mut file, 56)?;
        write_fourcc(&mut file, b"vids")?;
        write_fourcc(&mut file, b"MJPG")?;
        write_u32(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 1)?;
        write_u32(&mut file, fps)?;
        write_u32(&mut file, 0)?;
        let strh_length_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0xffff_ffff)?;
        write_u32(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, width as u16)?;
        write_u16(&mut file, height as u16)?;

        write_fourcc(&mut file, b"strf")?;
        write_u32(&mut file, 40)?;
        write_u32(&mut file, 40)?;
        write_i32(&mut file, width as i32)?;
        write_i32(&mut file, height as i32)?;
        write_u16(&mut file, 1)?;
        write_u16(&mut file, 24)?;
        write_fourcc(&mut file, b"MJPG")?;
        write_u32(&mut file, width * height * 3)?;
        write_i32(&mut file, 0)?;
        write_i32(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, 0)?;

        let strl_end = stream_pos(&mut file)?;
        patch_u32(&mut file, strl_size_pos, (strl_end - strl_start) as u32)?;
        let hdrl_end = stream_pos(&mut file)?;
        patch_u32(&mut file, hdrl_size_pos, (hdrl_end - hdrl_start) as u32)?;

        write_fourcc(&mut file, b"LIST")?;
        let movi_size_pos = stream_pos(&mut file)?;
        write_u32(&mut file, 0)?;
        let movi_data_start = stream_pos(&mut file)?;
        write_fourcc(&mut file, b"movi")?;
        let last_good_end = stream_pos(&mut file)?;

        Ok(Self {
            file,
            entries: Vec::new(),
            riff_size_pos,
            avih_frames_pos,
            strh_length_pos,
            movi_size_pos,
            movi_data_start,
            last_good_end,
        })
    }

    fn write_frame(&mut self, data: &[u8]) -> io::Result<()> {
        let chunk_start = stream_pos(&mut self.file)?;
        let data_len = checked_avi_u32(data.len() as u64, "frame chunk size")?;
        let padded_data_len = data.len() as u64 + if data.len() % 2 == 1 { 1 } else { 0 };
        let frame_end = chunk_start + 8 + padded_data_len;
        let movi_size = frame_end - self.movi_data_start;
        checked_avi_u32(chunk_start - self.movi_data_start, "frame index offset")?;
        checked_avi_u32(movi_size, "movi list size")?;
        let index_entries = self.entries.len() as u64 + 1;
        checked_avi_u32(index_entries, "frame count")?;
        let idx1_size = index_entries
            .checked_mul(AVI_INDEX_ENTRY_BYTES)
            .ok_or_else(|| avi_overflow("idx1 size"))?;
        checked_avi_u32(idx1_size, "idx1 size")?;
        let file_end_with_index = frame_end
            .checked_add(8)
            .and_then(|value| value.checked_add(idx1_size))
            .ok_or_else(|| avi_overflow("RIFF size"))?;
        checked_avi_u32(file_end_with_index - 8, "RIFF size")?;
        write_fourcc(&mut self.file, b"00dc")?;
        write_u32(&mut self.file, data_len)?;
        self.file.write_all(data)?;
        if data.len() % 2 == 1 {
            self.file.write_all(&[0])?;
        }
        self.entries.push(AviIndexEntry {
            flags: 0x10,
            offset: checked_avi_u32(chunk_start - self.movi_data_start, "frame index offset")?,
            size: data_len,
        });
        self.last_good_end = stream_pos(&mut self.file)?;
        Ok(())
    }

    /// Discard any partially written trailing frame, then finalize headers and
    /// index so the successfully written frames remain playable.
    fn finish_salvage(mut self) -> io::Result<usize> {
        use std::io::{Seek, SeekFrom};
        let end = self.last_good_end;
        self.file.set_len(end)?;
        self.file.seek(SeekFrom::Start(end))?;
        self.finish()
    }

    fn finish(mut self) -> io::Result<usize> {
        let movi_end = stream_pos(&mut self.file)?;
        write_fourcc(&mut self.file, b"idx1")?;
        let idx1_size = checked_avi_u32(
            self.entries.len() as u64 * AVI_INDEX_ENTRY_BYTES,
            "idx1 size",
        )?;
        write_u32(&mut self.file, idx1_size)?;
        for entry in &self.entries {
            write_fourcc(&mut self.file, b"00dc")?;
            write_u32(&mut self.file, entry.flags)?;
            write_u32(&mut self.file, entry.offset)?;
            write_u32(&mut self.file, entry.size)?;
        }
        let file_end = stream_pos(&mut self.file)?;
        patch_u32(
            &mut self.file,
            self.riff_size_pos,
            checked_avi_u32(file_end - 8, "RIFF size")?,
        )?;
        patch_u32(
            &mut self.file,
            self.avih_frames_pos,
            checked_avi_u32(self.entries.len() as u64, "frame count")?,
        )?;
        patch_u32(
            &mut self.file,
            self.strh_length_pos,
            checked_avi_u32(self.entries.len() as u64, "frame count")?,
        )?;
        patch_u32(
            &mut self.file,
            self.movi_size_pos,
            checked_avi_u32(movi_end - self.movi_data_start, "movi list size")?,
        )?;
        self.file.flush()?;
        Ok(self.entries.len())
    }
}

fn checked_avi_u32(value: u64, field: &str) -> io::Result<u32> {
    if value > AVI_U32_MAX {
        Err(avi_overflow(field))
    } else {
        Ok(value as u32)
    }
}

fn avi_overflow(field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("AVI32 {field} would exceed 4 GiB; stop earlier or segment recording"),
    )
}

fn write_fourcc<W: Write>(writer: &mut W, value: &[u8; 4]) -> io::Result<()> {
    writer.write_all(value)
}

fn write_u16<W: Write>(writer: &mut W, value: u16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_i32<W: Write>(writer: &mut W, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn stream_pos(file: &mut File) -> io::Result<u64> {
    use std::io::Seek;
    file.stream_position()
}

fn patch_u32(file: &mut File, pos: u64, value: u32) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let current = file.stream_position()?;
    file.seek(SeekFrom::Start(pos))?;
    write_u32(file, value)?;
    file.seek(SeekFrom::Start(current))?;
    Ok(())
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(if message.starts_with("usage:") { 0 } else { 2 });
        }
    };

    #[cfg(target_os = "linux")]
    {
        if let Err(err) = linux_v4l2::record(args) {
            eprintln!("recording failed: {err}");
            std::process::exit(1);
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = args;
        eprintln!("hardware recording is only supported on Linux");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
mod linux_v4l2 {
    use super::*;
    use libc::{c_int, c_ulong, c_void};
    use std::ffi::CString;
    use std::mem::{size_of, zeroed};
    use std::os::fd::RawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::ptr::null_mut;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::TrySendError;
    use std::time::Instant;

    static SHUTDOWN: AtomicBool = AtomicBool::new(false);

    const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
    const V4L2_CAP_STREAMING: u32 = 0x0400_0000;
    const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
    const V4L2_FIELD_NONE: u32 = 1;
    const V4L2_MEMORY_MMAP: u32 = 1;
    const V4L2_BUF_FLAG_TIMESTAMP_MASK: u32 = 0x0000_e000;
    const V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC: u32 = 0x0000_2000;
    const V4L2_BUF_FLAG_TIMESTAMP_COPY: u32 = 0x0000_4000;
    const V4L2_BUF_FLAG_TSTAMP_SRC_MASK: u32 = 0x0007_0000;
    const V4L2_BUF_FLAG_TSTAMP_SRC_EOF: u32 = 0x0000_0000;
    const V4L2_BUF_FLAG_TSTAMP_SRC_SOE: u32 = 0x0001_0000;
    const V4L2_PIX_FMT_MJPEG: u32 = fourcc(*b"MJPG");

    // Exposure controls (UVC user class). EXPOSURE_AUTO is a menu control whose
    // "Manual Mode" value is 1; EXPOSURE_ABSOLUTE is the shutter time in 100us
    // units. See linux/uapi/linux/v4l2-controls.h.
    const V4L2_CID_EXPOSURE_AUTO: u32 = 0x009a_0901;
    const V4L2_CID_EXPOSURE_ABSOLUTE: u32 = 0x009a_0902;
    const V4L2_EXPOSURE_MANUAL: i32 = 1;

    const fn fourcc(value: [u8; 4]) -> u32 {
        value[0] as u32
            | ((value[1] as u32) << 8)
            | ((value[2] as u32) << 16)
            | ((value[3] as u32) << 24)
    }

    const IOC_NRBITS: u64 = 8;
    const IOC_TYPEBITS: u64 = 8;
    const IOC_SIZEBITS: u64 = 14;
    const IOC_NRSHIFT: u64 = 0;
    const IOC_TYPESHIFT: u64 = IOC_NRSHIFT + IOC_NRBITS;
    const IOC_SIZESHIFT: u64 = IOC_TYPESHIFT + IOC_TYPEBITS;
    const IOC_DIRSHIFT: u64 = IOC_SIZESHIFT + IOC_SIZEBITS;
    const IOC_WRITE: u64 = 1;
    const IOC_READ: u64 = 2;

    const fn ioc(dir: u64, type_: u64, nr: u64, size: u64) -> c_ulong {
        ((dir << IOC_DIRSHIFT)
            | (type_ << IOC_TYPESHIFT)
            | (nr << IOC_NRSHIFT)
            | (size << IOC_SIZESHIFT)) as c_ulong
    }

    const fn ior<T>(type_: u64, nr: u64) -> c_ulong {
        ioc(IOC_READ, type_, nr, size_of::<T>() as u64)
    }

    const fn iow<T>(type_: u64, nr: u64) -> c_ulong {
        ioc(IOC_WRITE, type_, nr, size_of::<T>() as u64)
    }

    const fn iowr<T>(type_: u64, nr: u64) -> c_ulong {
        ioc(IOC_READ | IOC_WRITE, type_, nr, size_of::<T>() as u64)
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Capability {
        driver: [u8; 16],
        card: [u8; 32],
        bus_info: [u8; 32],
        version: u32,
        capabilities: u32,
        device_caps: u32,
        reserved: [u32; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2PixFormat {
        width: u32,
        height: u32,
        pixelformat: u32,
        field: u32,
        bytesperline: u32,
        sizeimage: u32,
        colorspace: u32,
        priv_: u32,
        flags: u32,
        ycbcr_enc: u32,
        quantization: u32,
        xfer_func: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Format {
        type_: u32,
        pad: u32,
        fmt: [u8; 200],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2RequestBuffers {
        count: u32,
        type_: u32,
        memory: u32,
        capabilities: u32,
        flags: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Fract {
        numerator: u32,
        denominator: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2CaptureParm {
        capability: u32,
        capturemode: u32,
        timeperframe: V4l2Fract,
        extendedmode: u32,
        readbuffers: u32,
        reserved: [u32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2StreamParm {
        type_: u32,
        parm: [u8; 200],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Timecode {
        type_: u32,
        flags: u32,
        frames: u8,
        seconds: u8,
        minutes: u8,
        hours: u8,
        userbits: [u8; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    union V4l2BufferM {
        offset: u32,
        userptr: c_ulong,
        planes: *mut c_void,
        fd: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    union V4l2BufferRequest {
        request_fd: i32,
        reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Buffer {
        index: u32,
        type_: u32,
        bytesused: u32,
        flags: u32,
        field: u32,
        timestamp: libc::timeval,
        timecode: V4l2Timecode,
        sequence: u32,
        memory: u32,
        m: V4l2BufferM,
        length: u32,
        reserved2: u32,
        request: V4l2BufferRequest,
    }

    struct MmapBuffer {
        start: *mut c_void,
        length: usize,
    }

    impl Drop for MmapBuffer {
        fn drop(&mut self) {
            if !self.start.is_null() && self.length > 0 {
                unsafe {
                    libc::munmap(self.start, self.length);
                }
            }
        }
    }

    struct Fd(RawFd);

    impl Drop for Fd {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.0);
            }
        }
    }

    pub fn record(args: Args) -> Result<(), String> {
        install_signal_handler()?;
        let fd = open_device(&args.device)?;
        configure_format(fd.0, args.width, args.height)?;
        configure_fps(fd.0, args.fps)?;
        if let Some(exposure) = args.exposure {
            // Best-effort: a camera that lacks these controls keeps auto-exposure and
            // recording proceeds. Never fail the capture over an exposure control.
            if let Err(err) = configure_exposure(fd.0, exposure) {
                eprintln!(
                    "warning: could not set exposure={exposure} \
                     (continuing with camera default): {err}"
                );
            }
        }
        let buffers = create_mmap_buffers(fd.0, args.buffer_count)?;
        queue_all_buffers(fd.0, buffers.len() as u32)?;
        let (writer_tx, writer_handle) = spawn_writer(
            args.output.clone(),
            args.timestamps.clone(),
            args.writer_queue_frames,
            args.width,
            args.height,
            args.fps,
        );

        // From here on, every failure funnels through the same shutdown path:
        // stop streaming, close the writer queue, join the writer so it can
        // finalize the AVI with the frames it has, and write stats so the
        // Python wrapper can always tell what happened.
        let mut stats = CaptureStats::new(args.fps);
        let mut enqueue_stats = EnqueueStats::default();
        let mut capture_error: Option<String> = None;
        let mut streaming = false;
        match stream_on(fd.0) {
            Ok(()) => streaming = true,
            Err(err) => capture_error = Some(err),
        }
        if capture_error.is_none()
            && let Err(err) = discard_warmup(fd.0, args.warmup_frames)
        {
            capture_error = Some(err);
        }
        if capture_error.is_none() {
            capture_error = capture_frames(
                fd.0,
                &buffers,
                &args,
                &writer_tx,
                &mut stats,
                &mut enqueue_stats,
            )
            .err();
        }
        if streaming && let Err(err) = stream_off(fd.0) {
            capture_error.get_or_insert(err);
        }

        drop(writer_tx);
        let writer_outcome = match writer_handle.join() {
            Ok(outcome) => outcome,
            Err(_) => WriterOutcome {
                frames_written: 0,
                bytes_written: 0,
                error: Some("writer thread panicked".to_string()),
            },
        };

        let mut error = capture_error;
        if let Some(writer_error) = &writer_outcome.error {
            error = Some(match error {
                Some(previous) => format!("{previous}; {writer_error}"),
                None => writer_error.clone(),
            });
        }
        let delta = stats.delta_summary();
        if let Err(stats_err) = write_stats(
            &args.stats,
            &args,
            &stats,
            &enqueue_stats,
            &writer_outcome,
            &delta,
            error.as_deref(),
        ) {
            let stats_err = format!("write stats: {stats_err}");
            error = Some(match error {
                Some(previous) => format!("{previous}; {stats_err}"),
                None => stats_err,
            });
        }
        eprintln!(
            "frames={} avi_frames={} dropped_sequence_frames={} delta_count={} delta_ms_min={:.3} delta_ms_max={:.3} delta_ms_mean={:.3} bad_delta_count={} non_monotonic_timestamps={} bad_timestamp_metadata={} writer_bytes={} queue_full_count={} max_enqueue_block_ms={:.3} max_copy_ms={:.3} mean_copy_ms={:.3}",
            stats.frame_count,
            writer_outcome.frames_written,
            stats.dropped_sequence_frames,
            delta.count,
            delta.min_ms.unwrap_or_default(),
            delta.max_ms.unwrap_or_default(),
            delta.mean_ms.unwrap_or_default(),
            delta.bad_delta_count,
            stats.non_monotonic_timestamps,
            stats.bad_timestamp_metadata,
            writer_outcome.bytes_written,
            enqueue_stats.queue_full_count,
            enqueue_stats.max_enqueue_block_ms,
            enqueue_stats.max_copy_ms,
            enqueue_stats.mean_copy_ms(stats.frame_count),
        );
        match error {
            None => Ok(()),
            Some(error) => Err(error),
        }
    }

    fn capture_frames(
        fd: RawFd,
        buffers: &[MmapBuffer],
        args: &Args,
        writer_tx: &SyncSender<FramePacket>,
        stats: &mut CaptureStats,
        enqueue_stats: &mut EnqueueStats,
    ) -> Result<(), String> {
        let unix_offset_ns = realtime_ns().wrapping_sub(monotonic_ns());
        let deadline = args
            .seconds
            .map(|seconds| Instant::now() + Duration::from_secs(seconds));
        while !SHUTDOWN.load(Ordering::SeqCst)
            && deadline.is_none_or(|value| Instant::now() < value)
        {
            wait_for_frame(fd)?;
            let mut buf = dequeue_buffer(fd)?;
            let data = unsafe {
                std::slice::from_raw_parts(
                    buffers[buf.index as usize].start.cast::<u8>(),
                    buf.bytesused as usize,
                )
            };
            let meta = FrameMeta {
                frame_index: stats.frame_count,
                v4l2_sequence: buf.sequence,
                time_monotonic_ns: timeval_to_ns(buf.timestamp),
                time_unix_ns: timeval_to_ns(buf.timestamp).wrapping_add(unix_offset_ns),
                bytesused: buf.bytesused,
                buffer_index: buf.index,
                v4l2_flags: buf.flags,
                timestamp_clock: timestamp_clock(buf.flags),
                timestamp_source: timestamp_source(buf.flags),
            };
            let copy_start = Instant::now();
            let packet = FramePacket {
                meta: meta.clone(),
                data: data.to_vec(),
            };
            enqueue_stats.observe_copy(copy_start.elapsed());
            match writer_tx.try_send(packet) {
                Ok(()) => {}
                Err(TrySendError::Full(packet)) => {
                    let enqueue_start = Instant::now();
                    if writer_tx.send(packet).is_err() {
                        // The writer's own error is reported after join.
                        return Err("writer thread stopped accepting frames".to_string());
                    }
                    enqueue_stats.observe_enqueue_block(enqueue_start.elapsed());
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err("writer thread stopped accepting frames".to_string());
                }
            }
            stats.observe(&meta);
            queue_buffer(fd, &mut buf)?;
        }
        Ok(())
    }

    extern "C" fn handle_signal(_signal: c_int) {
        SHUTDOWN.store(true, Ordering::SeqCst);
    }

    fn install_signal_handler() -> Result<(), String> {
        unsafe {
            if libc::signal(
                libc::SIGINT,
                handle_signal as *const () as libc::sighandler_t,
            ) == libc::SIG_ERR
            {
                return Err(format!(
                    "install SIGINT handler: {}",
                    io::Error::last_os_error()
                ));
            }
            if libc::signal(
                libc::SIGTERM,
                handle_signal as *const () as libc::sighandler_t,
            ) == libc::SIG_ERR
            {
                return Err(format!(
                    "install SIGTERM handler: {}",
                    io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }

    fn open_device(path: &Path) -> Result<Fd, String> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| format!("invalid device path: {}", path.display()))?;
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK, 0) };
        if fd < 0 {
            Err(format!(
                "open {}: {}",
                path.display(),
                io::Error::last_os_error()
            ))
        } else {
            Ok(Fd(fd))
        }
    }

    fn configure_format(fd: RawFd, width: u32, height: u32) -> Result<(), String> {
        let mut cap: V4l2Capability = unsafe { zeroed() };
        xioctl(fd, ior::<V4l2Capability>(b'V' as u64, 0), &mut cap)?;
        if cap.capabilities & V4L2_CAP_VIDEO_CAPTURE == 0
            || cap.capabilities & V4L2_CAP_STREAMING == 0
        {
            return Err("device does not support video capture + streaming".to_string());
        }

        let mut fmt: V4l2Format = unsafe { zeroed() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.pad = 0;
        let pix = V4l2PixFormat {
            width,
            height,
            pixelformat: V4L2_PIX_FMT_MJPEG,
            field: V4L2_FIELD_NONE,
            bytesperline: 0,
            sizeimage: 0,
            colorspace: 0,
            priv_: 0,
            flags: 0,
            ycbcr_enc: 0,
            quantization: 0,
            xfer_func: 0,
        };
        unsafe {
            (fmt.fmt.as_mut_ptr().cast::<V4l2PixFormat>()).write(pix);
        }
        xioctl(fd, iowr::<V4l2Format>(b'V' as u64, 5), &mut fmt)?;
        let accepted = unsafe { *(fmt.fmt.as_ptr().cast::<V4l2PixFormat>()) };
        if accepted.width != width
            || accepted.height != height
            || accepted.pixelformat != V4L2_PIX_FMT_MJPEG
        {
            return Err(format!(
                "device did not accept requested format: {}x{} fourcc=0x{:08x}",
                accepted.width, accepted.height, accepted.pixelformat
            ));
        }
        Ok(())
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct V4l2Control {
        id: u32,
        value: i32,
    }

    fn set_control(fd: RawFd, id: u32, value: i32) -> Result<(), String> {
        let mut ctrl = V4l2Control { id, value };
        // VIDIOC_S_CTRL = _IOWR('V', 28, struct v4l2_control)
        xioctl(fd, iowr::<V4l2Control>(b'V' as u64, 28), &mut ctrl)
    }

    fn configure_exposure(fd: RawFd, exposure: i64) -> Result<(), String> {
        // Disable auto-exposure, then pin the absolute shutter time. Fixed exposure
        // is load-bearing for downstream ArUco gripper detection under changing light.
        set_control(fd, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
            .map_err(|e| format!("set manual exposure mode: {e}"))?;
        let value =
            i32::try_from(exposure).map_err(|_| format!("exposure {exposure} out of range"))?;
        set_control(fd, V4L2_CID_EXPOSURE_ABSOLUTE, value)
            .map_err(|e| format!("set exposure_absolute={value}: {e}"))?;
        Ok(())
    }

    fn configure_fps(fd: RawFd, fps: u32) -> Result<(), String> {
        if fps == 0 {
            return Err("fps must be greater than zero".to_string());
        }
        let mut parm: V4l2StreamParm = unsafe { zeroed() };
        parm.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let capture = V4l2CaptureParm {
            capability: 0,
            capturemode: 0,
            timeperframe: V4l2Fract {
                numerator: 1,
                denominator: fps,
            },
            extendedmode: 0,
            readbuffers: 0,
            reserved: [0; 4],
        };
        unsafe {
            (parm.parm.as_mut_ptr().cast::<V4l2CaptureParm>()).write(capture);
        }
        xioctl(fd, iowr::<V4l2StreamParm>(b'V' as u64, 22), &mut parm)?;
        let accepted = unsafe { *(parm.parm.as_ptr().cast::<V4l2CaptureParm>()) };
        validate_timeperframe(accepted.timeperframe, fps)?;

        let mut verified: V4l2StreamParm = unsafe { zeroed() };
        verified.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd, iowr::<V4l2StreamParm>(b'V' as u64, 21), &mut verified)?;
        let verified_capture = unsafe { *(verified.parm.as_ptr().cast::<V4l2CaptureParm>()) };
        validate_timeperframe(verified_capture.timeperframe, fps)
    }

    /// Relative fps tolerance: accepts driver rounding such as NTSC
    /// 30000/1001 (~29.97) for a requested 30, rejects grossly different
    /// rates (per-frame sidecar timestamps stay authoritative either way).
    const FPS_RELATIVE_TOLERANCE: f64 = 0.02;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum TimeperframeCheck {
        Match,
        Unknown,
        Mismatch(f64),
    }

    fn check_timeperframe(timeperframe: V4l2Fract, requested_fps: u32) -> TimeperframeCheck {
        if timeperframe.numerator == 0 || timeperframe.denominator == 0 {
            return TimeperframeCheck::Unknown;
        }
        let actual = f64::from(timeperframe.denominator) / f64::from(timeperframe.numerator);
        let requested = f64::from(requested_fps);
        if ((actual - requested) / requested).abs() <= FPS_RELATIVE_TOLERANCE {
            TimeperframeCheck::Match
        } else {
            TimeperframeCheck::Mismatch(actual)
        }
    }

    fn validate_timeperframe(timeperframe: V4l2Fract, requested_fps: u32) -> Result<(), String> {
        match check_timeperframe(timeperframe, requested_fps) {
            TimeperframeCheck::Match => Ok(()),
            TimeperframeCheck::Unknown => {
                eprintln!(
                    "warning: device did not report timeperframe; cannot verify fps {requested_fps}"
                );
                Ok(())
            }
            TimeperframeCheck::Mismatch(actual) => Err(format!(
                "device runs at {actual:.3} fps for requested fps {requested_fps} (timeperframe={}/{}); request a rate the camera supports",
                timeperframe.numerator, timeperframe.denominator
            )),
        }
    }

    fn create_mmap_buffers(fd: RawFd, requested_count: u32) -> Result<Vec<MmapBuffer>, String> {
        let mut req = V4l2RequestBuffers {
            count: requested_count,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            capabilities: 0,
            flags: 0,
        };
        xioctl(fd, iowr::<V4l2RequestBuffers>(b'V' as u64, 8), &mut req)?;
        if req.count < 2 {
            return Err("insufficient V4L2 buffers".to_string());
        }
        let mut buffers = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let mut buf = empty_buffer(index);
            xioctl(fd, iowr::<V4l2Buffer>(b'V' as u64, 9), &mut buf)?;
            let offset = unsafe { buf.m.offset };
            let start = unsafe {
                libc::mmap(
                    null_mut(),
                    buf.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    offset as libc::off_t,
                )
            };
            if start == libc::MAP_FAILED {
                return Err(format!(
                    "mmap buffer {index}: {}",
                    io::Error::last_os_error()
                ));
            }
            buffers.push(MmapBuffer {
                start,
                length: buf.length as usize,
            });
        }
        Ok(buffers)
    }

    fn queue_all_buffers(fd: RawFd, count: u32) -> Result<(), String> {
        for index in 0..count {
            let mut buf = empty_buffer(index);
            queue_buffer(fd, &mut buf)?;
        }
        Ok(())
    }

    fn discard_warmup(fd: RawFd, warmup_frames: u32) -> Result<(), String> {
        for _ in 0..warmup_frames {
            // Honor shutdown signals during warmup so a quick start/stop
            // exits promptly instead of ignoring the signal.
            if SHUTDOWN.load(Ordering::SeqCst) {
                return Ok(());
            }
            wait_for_frame(fd)?;
            let mut buf = dequeue_buffer(fd)?;
            queue_buffer(fd, &mut buf)?;
        }
        Ok(())
    }

    fn empty_buffer(index: u32) -> V4l2Buffer {
        V4l2Buffer {
            index,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            bytesused: 0,
            flags: 0,
            field: 0,
            timestamp: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            timecode: V4l2Timecode {
                type_: 0,
                flags: 0,
                frames: 0,
                seconds: 0,
                minutes: 0,
                hours: 0,
                userbits: [0; 4],
            },
            sequence: 0,
            memory: V4L2_MEMORY_MMAP,
            m: V4l2BufferM { offset: 0 },
            length: 0,
            reserved2: 0,
            request: V4l2BufferRequest { reserved: 0 },
        }
    }

    fn stream_on(fd: RawFd) -> Result<(), String> {
        let mut type_: c_int = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
        xioctl(fd, iow::<c_int>(b'V' as u64, 18), &mut type_)
    }

    fn stream_off(fd: RawFd) -> Result<(), String> {
        let mut type_: c_int = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
        xioctl(fd, iow::<c_int>(b'V' as u64, 19), &mut type_)
    }

    fn wait_for_frame(fd: RawFd) -> Result<(), String> {
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let result = unsafe { libc::poll(&mut pollfd, 1, 2000) };
            if result > 0 {
                return Ok(());
            }
            if result == 0 {
                return Err("poll timeout waiting for camera frame".to_string());
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(format!("poll: {err}"));
        }
    }

    fn dequeue_buffer(fd: RawFd) -> Result<V4l2Buffer, String> {
        loop {
            let mut buf = empty_buffer(0);
            match xioctl(fd, iowr::<V4l2Buffer>(b'V' as u64, 17), &mut buf) {
                Ok(()) => return Ok(buf),
                Err(err) if err.contains("Resource temporarily unavailable") => continue,
                Err(err) => return Err(err),
            }
        }
    }

    fn queue_buffer(fd: RawFd, buf: &mut V4l2Buffer) -> Result<(), String> {
        xioctl(fd, iowr::<V4l2Buffer>(b'V' as u64, 15), buf)
    }

    fn xioctl<T>(fd: RawFd, request: c_ulong, arg: &mut T) -> Result<(), String> {
        loop {
            let result = unsafe { libc::ioctl(fd, request, arg as *mut T) };
            if result == 0 {
                return Ok(());
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(format!("ioctl 0x{request:x}: {err}"));
        }
    }

    fn timeval_to_ns(tv: libc::timeval) -> u64 {
        tv.tv_sec as u64 * 1_000_000_000 + tv.tv_usec as u64 * 1_000
    }

    fn monotonic_ns() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
        }
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }

    fn realtime_ns() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts);
        }
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }

    fn timestamp_clock(flags: u32) -> &'static str {
        match flags & V4L2_BUF_FLAG_TIMESTAMP_MASK {
            V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC => "monotonic",
            V4L2_BUF_FLAG_TIMESTAMP_COPY => "copy",
            _ => "unknown",
        }
    }

    fn timestamp_source(flags: u32) -> &'static str {
        match flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK {
            V4L2_BUF_FLAG_TSTAMP_SRC_SOE => "start_of_exposure",
            V4L2_BUF_FLAG_TSTAMP_SRC_EOF => "end_of_frame",
            _ => "unknown",
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fourcc_is_little_endian() {
            assert_eq!(fourcc(*b"MJPG"), 0x4750_4a4d);
        }

        #[test]
        fn timestamp_flags_decode_monotonic_start_of_exposure() {
            let flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_SOE;
            assert_eq!(timestamp_clock(flags), "monotonic");
            assert_eq!(timestamp_source(flags), "start_of_exposure");
        }

        #[test]
        fn v4l2_struct_sizes_match_linux_headers() {
            assert_eq!(size_of::<V4l2Capability>(), 104);
            assert_eq!(size_of::<V4l2Format>(), 208);
            assert_eq!(size_of::<V4l2StreamParm>(), 204);
            assert_eq!(size_of::<V4l2RequestBuffers>(), 20);
            assert_eq!(size_of::<V4l2Buffer>(), 88);
            assert_eq!(size_of::<V4l2PixFormat>(), 48);
            assert_eq!(size_of::<V4l2CaptureParm>(), 40);
        }

        #[test]
        fn v4l2_struct_offsets_match_linux_headers() {
            assert_eq!(std::mem::offset_of!(V4l2Format, type_), 0);
            assert_eq!(std::mem::offset_of!(V4l2Format, fmt), 8);
            assert_eq!(std::mem::offset_of!(V4l2Buffer, timestamp), 24);
            assert_eq!(std::mem::offset_of!(V4l2Buffer, sequence), 56);
            assert_eq!(std::mem::offset_of!(V4l2Buffer, m), 64);
            assert_eq!(std::mem::offset_of!(V4l2Buffer, length), 72);
            assert_eq!(std::mem::offset_of!(V4l2Buffer, request), 80);
        }

        #[test]
        fn v4l2_ioctl_numbers_match_linux_headers() {
            assert_eq!(ior::<V4l2Capability>(b'V' as u64, 0), 0x8068_5600);
            assert_eq!(iowr::<V4l2Format>(b'V' as u64, 5), 0xc0d0_5605);
            assert_eq!(iowr::<V4l2StreamParm>(b'V' as u64, 21), 0xc0cc_5615);
            assert_eq!(iowr::<V4l2StreamParm>(b'V' as u64, 22), 0xc0cc_5616);
            assert_eq!(iowr::<V4l2RequestBuffers>(b'V' as u64, 8), 0xc014_5608);
            assert_eq!(iowr::<V4l2Buffer>(b'V' as u64, 9), 0xc058_5609);
            assert_eq!(iowr::<V4l2Buffer>(b'V' as u64, 15), 0xc058_560f);
            assert_eq!(iowr::<V4l2Buffer>(b'V' as u64, 17), 0xc058_5611);
        }

        #[test]
        fn timeperframe_accepts_exact_and_close_fps() {
            assert_eq!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 1,
                        denominator: 30,
                    },
                    30,
                ),
                TimeperframeCheck::Match
            );
            assert_eq!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 2,
                        denominator: 60,
                    },
                    30,
                ),
                TimeperframeCheck::Match
            );
            // NTSC-style driver rounding (~29.97 fps) must not be fatal.
            assert_eq!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 1001,
                        denominator: 30_000,
                    },
                    30,
                ),
                TimeperframeCheck::Match
            );
        }

        #[test]
        fn timeperframe_rejects_grossly_different_fps() {
            assert!(matches!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 1,
                        denominator: 25,
                    },
                    30,
                ),
                TimeperframeCheck::Mismatch(_)
            ));
        }

        #[test]
        fn timeperframe_without_driver_support_is_unknown_not_fatal() {
            assert_eq!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 0,
                        denominator: 30,
                    },
                    30,
                ),
                TimeperframeCheck::Unknown
            );
            assert_eq!(
                check_timeperframe(
                    V4l2Fract {
                        numerator: 1,
                        denominator: 0,
                    },
                    30,
                ),
                TimeperframeCheck::Unknown
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_json_line_is_stable() {
        let meta = FrameMeta {
            frame_index: 2,
            v4l2_sequence: 10,
            time_monotonic_ns: 123,
            time_unix_ns: 456,
            bytesused: 789,
            buffer_index: 3,
            v4l2_flags: 73_729,
            timestamp_clock: "monotonic",
            timestamp_source: "start_of_exposure",
        };

        assert_eq!(
            meta.to_json_line(),
            "{\"frame_index\":2,\"v4l2_sequence\":10,\"time_monotonic_ns\":123,\"time_unix_ns\":456,\"bytesused\":789,\"buffer_index\":3,\"v4l2_flags\":73729,\"timestamp_clock\":\"monotonic\",\"timestamp_source\":\"start_of_exposure\"}\n"
        );
    }

    #[test]
    fn capture_stats_detects_sequence_gaps() {
        let mut stats = CaptureStats::default();
        for (frame_index, sequence) in [(0, 8), (1, 9), (2, 12)] {
            stats.observe(&FrameMeta {
                frame_index,
                v4l2_sequence: sequence,
                time_monotonic_ns: 1_000_000_000 + frame_index as u64 * 33_333_333,
                time_unix_ns: 2_000_000_000,
                bytesused: 10,
                buffer_index: 0,
                v4l2_flags: 0,
                timestamp_clock: "monotonic",
                timestamp_source: "start_of_exposure",
            });
        }

        assert_eq!(stats.frame_count, 3);
        assert_eq!(stats.dropped_sequence_frames, 2);
    }

    #[test]
    fn delta_summary_flags_large_jitter() {
        let mut stats = CaptureStats::default();
        for (frame_index, time_monotonic_ns) in [
            (0, 1_000_000_000),
            (1, 1_033_333_333),
            (2, 1_066_666_666),
            (3, 1_200_000_000),
        ] {
            stats.observe(&FrameMeta {
                frame_index,
                v4l2_sequence: frame_index,
                time_monotonic_ns,
                time_unix_ns: time_monotonic_ns,
                bytesused: 10,
                buffer_index: 0,
                v4l2_flags: 0,
                timestamp_clock: "monotonic",
                timestamp_source: "start_of_exposure",
            });
        }

        let summary = stats.delta_summary();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.bad_delta_count, 1);
        assert!(summary.max_ms.unwrap() > 100.0);
    }

    #[test]
    fn avi_writer_patches_frame_count_and_index() {
        let temp_dir = env::temp_dir().join(format!("umi_live_avi_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("test.avi");

        let mut writer = AviWriter::create(&path, 64, 48, 30).unwrap();
        writer.write_frame(&[0xff, 0xd8, 0xff, 0xd9]).unwrap();
        writer.write_frame(&[0xff, 0xd8, 0, 0xff, 0xd9]).unwrap();
        assert_eq!(writer.finish().unwrap(), 2);

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"AVI ");
        assert!(bytes.windows(4).any(|window| window == b"idx1"));
        let idx1_pos = bytes
            .windows(4)
            .position(|window| window == b"idx1")
            .unwrap();
        let idx1_size = u32::from_le_bytes(bytes[idx1_pos + 4..idx1_pos + 8].try_into().unwrap());
        assert_eq!(idx1_size, 32);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn avi_writer_rejects_before_avi32_overflow() {
        use std::io::{Seek, SeekFrom};

        let temp_dir = env::temp_dir().join(format!(
            "umi_live_avi_overflow_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("test.avi");

        let mut writer = AviWriter::create(&path, 64, 48, 30).unwrap();
        writer
            .file
            .seek(SeekFrom::Start(u64::from(u32::MAX) + 1))
            .unwrap();
        let error = writer.write_frame(&[0xff, 0xd8, 0xff, 0xd9]).unwrap_err();

        assert!(error.to_string().contains("AVI32"));
        assert!(error.to_string().contains("4 GiB"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn writer_thread_writes_sidecar_and_video() {
        let temp_dir =
            env::temp_dir().join(format!("umi_live_writer_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let output = temp_dir.join("camera_front.mjpeg.avi");
        let timestamps = temp_dir.join("camera_frames.jsonl");

        let (tx, handle) = spawn_writer(output.clone(), timestamps.clone(), 1, 64, 48, 30);
        for frame_index in 0..2 {
            tx.send(FramePacket {
                meta: FrameMeta {
                    frame_index,
                    v4l2_sequence: 8 + frame_index,
                    time_monotonic_ns: 1_000_000_000 + frame_index as u64 * 33_333_333,
                    time_unix_ns: 2_000_000_000 + frame_index as u64 * 33_333_333,
                    bytesused: 4,
                    buffer_index: frame_index,
                    v4l2_flags: 73_728,
                    timestamp_clock: "monotonic",
                    timestamp_source: "start_of_exposure",
                },
                data: vec![0xff, 0xd8, 0xff, 0xd9],
            })
            .unwrap();
        }
        drop(tx);

        let outcome = handle.join().unwrap();
        assert_eq!(
            outcome,
            WriterOutcome {
                frames_written: 2,
                bytes_written: 8,
                error: None,
            }
        );
        assert!(
            std::fs::read(&output)
                .unwrap()
                .windows(4)
                .any(|w| w == b"idx1")
        );
        assert_eq!(
            std::fs::read_to_string(&timestamps)
                .unwrap()
                .lines()
                .count(),
            2
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    fn frame_meta(frame_index: u32, time_monotonic_ns: u64) -> FrameMeta {
        FrameMeta {
            frame_index,
            v4l2_sequence: frame_index,
            time_monotonic_ns,
            time_unix_ns: time_monotonic_ns,
            bytesused: 4,
            buffer_index: 0,
            v4l2_flags: 73_728,
            timestamp_clock: "monotonic",
            timestamp_source: "start_of_exposure",
        }
    }

    fn test_args() -> Args {
        Args {
            device: PathBuf::from("/dev/video0"),
            output: PathBuf::from("out.avi"),
            timestamps: PathBuf::from("ts.jsonl"),
            stats: PathBuf::from("stats.json"),
            width: 64,
            height: 48,
            fps: 30,
            exposure: None,
            seconds: None,
            warmup_frames: 0,
            buffer_count: 4,
            writer_queue_frames: 4,
        }
    }

    #[test]
    fn writer_salvages_partial_avi_after_write_error() {
        use std::io::{Read, Seek, SeekFrom};

        let temp_dir = env::temp_dir().join(format!(
            "umi_live_writer_salvage_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let output = temp_dir.join("camera_front.mjpeg.avi");
        let timestamps = temp_dir.join("camera_frames.jsonl");

        let mut avi = AviWriter::create(&output, 64, 48, 30).unwrap();
        let sidecar = SidecarWriter::create(&timestamps).unwrap();
        // Position the file so the first frame still fits under the AVI32
        // 4 GiB cap but the second one does not.
        avi.file
            .seek(SeekFrom::Start(u64::from(u32::MAX) - 40))
            .unwrap();

        let (tx, rx) = sync_channel::<FramePacket>(4);
        for frame_index in 0..2 {
            tx.send(FramePacket {
                meta: frame_meta(
                    frame_index,
                    1_000_000_000 + u64::from(frame_index) * 33_333_333,
                ),
                data: vec![0xff, 0xd8, 0xff, 0xd9],
            })
            .unwrap();
        }
        drop(tx);

        let outcome = run_writer(avi, sidecar, &rx);

        assert_eq!(outcome.frames_written, 1);
        assert!(outcome.error.as_deref().unwrap().contains("AVI32"));
        // The salvaged file must be finalized: idx1 written at the end and
        // the RIFF size patched to match the truncated file.
        let mut file = File::open(&output).unwrap();
        let len = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(len - 24)).unwrap();
        let mut fourcc = [0_u8; 4];
        file.read_exact(&mut fourcc).unwrap();
        assert_eq!(&fourcc, b"idx1");
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut header = [0_u8; 8];
        file.read_exact(&mut header).unwrap();
        assert_eq!(&header[0..4], b"RIFF");
        let riff_size = u32::from_le_bytes(header[4..8].try_into().unwrap());
        assert_eq!(u64::from(riff_size), len - 8);
        assert_eq!(
            std::fs::read_to_string(&timestamps)
                .unwrap()
                .lines()
                .count(),
            1
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn dataset_eligibility_requires_frames_and_no_errors() {
        let enqueue = EnqueueStats::default();

        let empty = CaptureStats::new(30);
        let empty_outcome = WriterOutcome {
            frames_written: 0,
            bytes_written: 0,
            error: None,
        };
        assert!(!dataset_eligible(
            &empty,
            &enqueue,
            &empty_outcome,
            &empty.delta_summary(),
            None,
        ));

        let mut one_frame = CaptureStats::new(30);
        one_frame.observe(&frame_meta(0, 1_000_000_000));
        let one_outcome = WriterOutcome {
            frames_written: 1,
            bytes_written: 4,
            error: None,
        };
        let delta = one_frame.delta_summary();
        assert!(dataset_eligible(
            &one_frame,
            &enqueue,
            &one_outcome,
            &delta,
            None,
        ));
        assert!(!dataset_eligible(
            &one_frame,
            &enqueue,
            &one_outcome,
            &delta,
            Some("write avi frame: boom"),
        ));
    }

    #[test]
    fn stats_json_reports_escaped_error_and_ineligibility() {
        let temp_dir =
            env::temp_dir().join(format!("umi_live_stats_error_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let stats_path = temp_dir.join("camera_stats.json");

        let stats = CaptureStats::new(30);
        write_stats(
            &stats_path,
            &test_args(),
            &stats,
            &EnqueueStats::default(),
            &WriterOutcome {
                frames_written: 0,
                bytes_written: 0,
                error: Some("write avi frame: \"disk\" full".to_string()),
            },
            &stats.delta_summary(),
            Some("write avi frame: \"disk\" full"),
        )
        .unwrap();

        let contents = std::fs::read_to_string(&stats_path).unwrap();
        assert!(contents.contains("\"camera_dataset_eligible\": false"));
        assert!(contents.contains("\"camera_error\": \"write avi frame: \\\"disk\\\" full\""));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn delta_ms_handles_backwards_timestamps_without_wrapping() {
        let mut stats = CaptureStats::default();
        stats.observe(&frame_meta(0, 2_000_000_000));
        stats.observe(&frame_meta(1, 1_000_000_000));

        let deltas = stats.delta_ms();
        assert_eq!(deltas.len(), 1);
        assert!((deltas[0] + 1000.0).abs() < 1e-6);
        assert_eq!(stats.non_monotonic_timestamps, 1);
        assert_eq!(stats.delta_summary().bad_delta_count, 1);
    }

    #[test]
    fn json_escape_handles_quotes_and_control_characters() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
        assert_eq!(json_escape("\u{1}"), "\\u0001");
    }
}
