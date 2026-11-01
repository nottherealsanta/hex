//! The training-example store behind Labs › AirPods training data: one folder
//! per example under `<support>/Training/<local day>/<example>/`, holding the
//! motion window (`motion.csv`) and its label (`meta.json`).
//!
//! A folder is the unit that is saved, counted, discarded, and expired. No
//! audio is ever written; the text is the one the user already pasted.
//!
//! The format matches the Speechless trainer (`speechless/training`): motion.csv
//! is `t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz` with `t` in
//! seconds from the piece's start, and meta.json carries the text, the word
//! timings relative to the piece, and the AirPods session.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{Result, WrapErr, eyre};
use serde::{Deserialize, Serialize};

use crate::app_paths;
use crate::motion::MotionSample;

pub const SCHEMA: u32 = 1;
/// Default retention: 90 days and 1 GB, oldest first (see the Labs row).
pub const DEFAULT_RETENTION_DAYS: u32 = 90;
pub const DEFAULT_MAX_BYTES: u64 = 1_000_000_000;

pub const MOTION_CSV: &str = "motion.csv";
pub const META_JSON: &str = "meta.json";

/// One word of the label, timed from the piece's first sample.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Parakeet reports no per-word confidence; the field is absent then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// One example's label and provenance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    pub schema: u32,
    /// The training target: lowercase, without punctuation.
    pub text: String,
    /// The transcript as Hex produced it.
    pub raw_text: String,
    pub words: Vec<Word>,
    /// Every piece of one dictation shares its id.
    pub dictation: String,
    pub piece: u32,
    pub pieces: u32,
    pub duration_ms: u64,
    pub motion_hz: f64,
    /// The share of the window with motion samples.
    pub motion_coverage: f64,
    /// The largest attitude change in the piece, in radians.
    pub max_head_turn_rad: f64,
    /// Speech onset against jaw motion, as the once-per-session self-check
    /// measured it.
    pub clock_offset_ms: i64,
    pub model: String,
    pub airpods: String,
    /// A new id each time the AirPods reconnect; the trainer splits on it.
    pub session: String,
    pub created_at: String,
    /// The microphone the dictation heard (never the AirPods, which would
    /// switch them to call audio), so the trainer can weigh examples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub microphone: Option<String>,
}

/// One stored example, as `hex training list` reads it.
#[derive(Clone, Debug)]
pub struct Stored {
    pub path: PathBuf,
    pub meta: Meta,
    pub bytes: u64,
}

pub fn root() -> Result<PathBuf> {
    Ok(app_paths::support_dir()?.join("Training"))
}

/// The local-time stamp of an instant, split as the store names folders.
struct LocalTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    millis: u32,
}

impl LocalTime {
    fn now() -> Self {
        Self::from_unix(SystemTime::now())
    }

    fn from_unix(time: SystemTime) -> Self {
        let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        let seconds = since.as_secs() as libc::time_t;
        // SAFETY: `localtime_r` writes only into the provided zeroed `tm`.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::localtime_r(&seconds, &mut tm) };
        Self {
            year: tm.tm_year + 1_900,
            month: (tm.tm_mon + 1) as u32,
            day: tm.tm_mday as u32,
            hour: tm.tm_hour as u32,
            minute: tm.tm_min as u32,
            second: tm.tm_sec as u32,
            millis: since.subsec_millis(),
        }
    }

    fn day_folder(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    fn stamp(&self) -> String {
        format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}.{:03}",
            self.year, self.month, self.day, self.hour, self.minute, self.second, self.millis
        )
    }

    fn iso(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            self.year, self.month, self.day, self.hour, self.minute, self.second, self.millis
        )
    }
}

/// The local day, as `Training`'s first-level folders and an AirPods
/// session's label use it.
pub fn day_stamp() -> String {
    LocalTime::now().day_folder()
}

/// A short stable id from a dictation's identifier, as example folders name
/// it; every piece of one dictation shares it.
fn short_id(dictation: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in dictation.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{:04x}", hash & 0xffff)
}

/// Writes one example and returns its folder. The label is written last, so
/// an interrupted save leaves a folder the store treats as incomplete.
pub fn save(meta: &Meta, samples: &[MotionSample]) -> Result<PathBuf> {
    let root = root()?;
    let time = LocalTime::now();
    let day = root.join(time.day_folder());
    let id = format!(
        "{}-{}-{}",
        time.stamp(),
        short_id(&meta.dictation),
        meta.piece
    );
    let mut folder = day.join(&id);
    let mut attempt = 1;
    while folder.exists() {
        folder = day.join(format!("{id}-{attempt}"));
        attempt += 1;
    }
    fs::create_dir_all(&folder)
        .wrap_err_with(|| format!("could not create {}", folder.display()))?;
    // Motion times are written relative to the piece's first sample.
    let base = samples
        .first()
        .map_or(0.0, |sample| sample.t_ns as f64 / 1_000_000_000.0);
    let mut csv = String::from("t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz\n");
    for sample in samples {
        let t = sample.t_ns as f64 / 1_000_000_000.0 - base;
        let values = [
            t,
            sample.acceleration[0],
            sample.acceleration[1],
            sample.acceleration[2],
            sample.rotation_rate[0],
            sample.rotation_rate[1],
            sample.rotation_rate[2],
            sample.attitude[0],
            sample.attitude[1],
            sample.attitude[2],
            sample.gravity[0],
            sample.gravity[1],
            sample.gravity[2],
            sample.quaternion[0],
            sample.quaternion[1],
            sample.quaternion[2],
            sample.quaternion[3],
        ];
        let row = values
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        csv.push_str(&row);
        csv.push('\n');
    }
    let mut meta = meta.clone();
    if meta.created_at.is_empty() {
        meta.created_at = time.iso();
    }
    let json = serde_json::to_vec_pretty(&meta)?;
    crate::durable::write(&folder.join(MOTION_CSV), csv.as_bytes())
        .wrap_err("could not write motion.csv")?;
    crate::durable::write(&folder.join(META_JSON), &json).wrap_err("could not write meta.json")?;
    owner_only(&folder);
    owner_only(&day);
    Ok(folder)
}

#[cfg(unix)]
fn owner_only(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = fs::metadata(path) {
        let mut permissions = metadata.permissions();
        permissions.set_mode(if metadata.is_dir() { 0o700 } else { 0o600 });
        let _ = fs::set_permissions(path, permissions);
    }
}

#[cfg(not(unix))]
fn owner_only(_path: &Path) {}

/// Every complete example, oldest first.
pub fn examples() -> Result<Vec<Stored>> {
    let root = root()?;
    let mut stored = Vec::new();
    let Ok(days) = fs::read_dir(&root) else {
        return Ok(stored);
    };
    for day in days.flatten() {
        if !day.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let Ok(folders) = fs::read_dir(day.path()) else {
            continue;
        };
        for folder in folders.flatten() {
            let path = folder.path();
            if !path.is_dir() {
                continue;
            }
            let Some(meta) = read_meta(&path) else {
                continue;
            };
            stored.push(Stored {
                bytes: folder_bytes(&path),
                path,
                meta,
            });
        }
    }
    stored.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(stored)
}

fn read_meta(folder: &Path) -> Option<Meta> {
    let bytes = fs::read(folder.join(META_JSON)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn folder_bytes(folder: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(folder) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .map(|metadata| metadata.len())
        .sum()
}

/// What the menu bar shows: today's examples and their motion time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Today {
    pub examples: usize,
    pub seconds: f64,
}

pub fn today() -> Today {
    let day = LocalTime::now().day_folder();
    let Ok(stored) = examples() else {
        return Today::default();
    };
    let mut today = Today::default();
    for example in stored {
        let in_day = example
            .path
            .parent()
            .and_then(|parent| parent.file_name())
            .is_some_and(|name| name == day.as_str());
        if in_day {
            today.examples += 1;
            today.seconds += example.meta.duration_ms as f64 / 1000.0;
        }
    }
    today
}

/// Removes the newest example (the one just dictated).
pub fn discard_last() -> Result<Option<Stored>> {
    let Some(last) = examples()?.pop() else {
        return Ok(None);
    };
    remove(&last.path)?;
    Ok(Some(last))
}

/// Removes every example, returning how many folders went.
pub fn purge() -> Result<usize> {
    let root = root()?;
    let Ok(days) = fs::read_dir(&root) else {
        return Ok(0);
    };
    let mut removed = 0;
    for day in days.flatten() {
        let path = day.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(folders) = fs::read_dir(&path) else {
            continue;
        };
        for folder in folders.flatten() {
            if folder.path().is_dir() && remove(&folder.path()).is_ok() {
                removed += 1;
            }
        }
        let _ = fs::remove_dir(&path);
    }
    Ok(removed)
}

/// Expires examples past `days` or over `max_bytes`, oldest first, and
/// returns how many went.
pub fn prune(days: u32, max_bytes: u64) -> Result<usize> {
    let stored = examples()?;
    let cutoff = SystemTime::now().checked_sub(Duration::from_secs(u64::from(days) * 24 * 60 * 60));
    let mut removed = 0;
    let mut total: u64 = stored.iter().map(|example| example.bytes).sum();
    for example in &stored {
        let expired = cutoff.is_some_and(|cutoff| {
            fs::metadata(&example.path)
                .and_then(|metadata| metadata.modified())
                .is_ok_and(|modified| modified < cutoff)
        });
        if (expired || total > max_bytes) && remove(&example.path).is_ok() {
            removed += 1;
            total = total.saturating_sub(example.bytes);
        }
    }
    Ok(removed)
}

/// Runs the default retention at launch and hourly while Hex runs, so a
/// long-lived Hex expires old examples without waiting for a restart.
pub fn start_retention() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("training-retention".into())
            .spawn(|| {
                loop {
                    match prune(DEFAULT_RETENTION_DAYS, DEFAULT_MAX_BYTES) {
                        Ok(0) => {}
                        Ok(removed) => tracing::info!(removed, "expired training examples"),
                        Err(error) => {
                            tracing::warn!(%error, "could not expire training examples")
                        }
                    }
                    std::thread::sleep(Duration::from_secs(60 * 60));
                }
            });
    });
}

fn remove(folder: &Path) -> Result<()> {
    if !folder.starts_with(root()?) {
        return Err(eyre!("{} is outside the training folder", folder.display()));
    }
    fs::remove_dir_all(folder)
        .wrap_err_with(|| format!("could not remove {}", folder.display()))?;
    if let Some(day) = folder.parent()
        && fs::read_dir(day).is_ok_and(|mut entries| entries.next().is_none())
    {
        let _ = fs::remove_dir(day);
    }
    Ok(())
}

/// A scratch store for this crate's tests. The store reads its root from the
/// process environment, so one test holds the turn at a time.
#[cfg(test)]
pub(crate) mod scratch {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    static TURN: Mutex<()> = Mutex::new(());

    pub(crate) struct Store {
        path: PathBuf,
        _turn: MutexGuard<'static, ()>,
    }

    pub(crate) fn store() -> Store {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let turn = TURN.lock().unwrap_or_else(|error| error.into_inner());
        let path = std::env::temp_dir().join(format!(
            "hex-training-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        unsafe { std::env::set_var("HEX_APPLICATION_SUPPORT_DIR", &path) };
        Store { path, _turn: turn }
    }

    impl Drop for Store {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_ns: u64) -> MotionSample {
        MotionSample {
            t_ns,
            acceleration: [0.1, 0.2, 0.3],
            rotation_rate: [0.0; 3],
            attitude: [0.05, 0.0, 0.0],
            gravity: [0.0, 0.0, -1.0],
            quaternion: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn meta() -> Meta {
        Meta {
            schema: SCHEMA,
            text: "send the report".into(),
            raw_text: "Send the report.".into(),
            words: vec![Word {
                text: "send".into(),
                start: 0.1,
                end: 0.4,
                confidence: None,
            }],
            dictation: "d1".into(),
            piece: 0,
            pieces: 1,
            duration_ms: 2_310,
            motion_hz: 24.8,
            motion_coverage: 0.99,
            max_head_turn_rad: 0.12,
            clock_offset_ms: 3,
            model: "parakeet".into(),
            airpods: "AirPods Pro 2".into(),
            session: "2026-11-01-1".into(),
            created_at: String::new(),
            microphone: None,
        }
    }

    /// A store in a scratch Application Support folder (see [`scratch`]).
    fn with_store() -> super::scratch::Store {
        super::scratch::store()
    }

    #[test]
    fn an_example_round_trips_through_its_folder() {
        let _store = with_store();
        let samples = [sample(1_000_000_000), sample(1_040_000_000)];
        let folder = save(&meta(), &samples).unwrap();
        let stored = examples().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].path, folder);
        assert_eq!(stored[0].meta.text, "send the report");
        assert!(!stored[0].meta.created_at.is_empty());
        let csv = fs::read_to_string(folder.join(MOTION_CSV)).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap().split(',').count(), 17);
        let first: Vec<&str> = lines.next().unwrap().split(',').collect();
        assert_eq!(first[0], "0.000000");
        let second: Vec<&str> = lines.next().unwrap().split(',').collect();
        assert_eq!(second[0], "0.040000");
        assert!(stored[0].bytes > 0);
    }

    #[test]
    fn discard_last_takes_the_newest_example_only() {
        let _store = with_store();
        save(&meta(), &[sample(1)]).unwrap();
        let mut second = meta();
        second.text = "second".into();
        save(&second, &[sample(2)]).unwrap();
        let discarded = discard_last().unwrap().unwrap();
        assert_eq!(discarded.meta.text, "second");
        let left = examples().unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].meta.text, "send the report");
        assert!(discard_last().is_ok());
    }

    #[test]
    fn purging_removes_every_example_and_its_day() {
        let _store = with_store();
        save(&meta(), &[sample(1)]).unwrap();
        save(&meta(), &[sample(2)]).unwrap();
        assert_eq!(purge().unwrap(), 2);
        assert!(examples().unwrap().is_empty());
        let root = root().unwrap();
        let days: Vec<_> = fs::read_dir(&root).unwrap().flatten().collect();
        assert!(days.is_empty(), "the day folder goes with its examples");
    }

    #[test]
    fn pruning_keeps_what_fits_the_size_limit() {
        let _store = with_store();
        for _ in 0..3 {
            save(&meta(), &[sample(1), sample(2)]).unwrap();
        }
        let stored = examples().unwrap();
        let smallest = stored.iter().map(|example| example.bytes).max().unwrap();
        let removed = prune(90, smallest).unwrap();
        assert_eq!(removed, 2, "the oldest examples go first");
        assert_eq!(examples().unwrap().len(), 1);
    }

    #[test]
    fn an_incomplete_folder_is_not_an_example() {
        let _store = with_store();
        let folder = root().unwrap().join("2026-11-01").join("half");
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join(MOTION_CSV), "t,ax\n").unwrap();
        assert!(examples().unwrap().is_empty());
    }
}
