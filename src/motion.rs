//! AirPods head motion for the labs' AirPods training data (macOS).
//!
//! Motion updates run only while a dictation records: [`begin`] at the
//! shortcut press and [`window`] (or [`abort`]) at the release. The Swift
//! bridge in `motion_bridge.swift` owns CoreMotion and buffers the samples;
//! this module reads one capture's window out of that buffer and stops the
//! updates, so nothing is kept between dictations.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::audio::CaptureInstant;

/// How far before the clip's first audio sample the window starts, so the
/// start of the first word is never cut off even when the mouth moves before
/// the microphone has audio.
pub const LOOK_BACK: Duration = Duration::from_millis(500);

/// No capture may keep motion updates running longer than this; a stuck
/// capture is stopped rather than left recording.
const WATCHDOG: Duration = Duration::from_secs(300);

/// A sample this recent means the stream is delivering. CoreMotion sends at
/// about 25 Hz, so a connected pair that sends nothing for this long is
/// quiet, not late.
const FLOWING_WITHIN: Duration = Duration::from_millis(500);

/// One CoreMotion sample. Times are `mach_absolute_time` nanoseconds, the
/// same clock as [`CaptureInstant`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSample {
    pub t_ns: u64,
    /// User acceleration in g, gravity removed.
    pub acceleration: [f64; 3],
    /// Rotation rate in radians per second.
    pub rotation_rate: [f64; 3],
    /// Attitude in radians.
    pub attitude: [f64; 3],
    /// Gravity in g.
    pub gravity: [f64; 3],
    /// Attitude quaternion.
    pub quaternion: [f64; 4],
}

impl MotionSample {
    /// The largest change of attitude, in radians, from any other sample.
    pub fn attitude_spread(samples: &[Self]) -> f64 {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for sample in samples {
            for (index, value) in sample.attitude.into_iter().enumerate() {
                min[index] = min[index].min(value);
                max[index] = max[index].max(value);
            }
        }
        if samples.is_empty() {
            return 0.0;
        }
        (0..3)
            .map(|index| max[index] - min[index])
            .fold(0.0, f64::max)
    }
}

/// One capture's motion: the samples between its audio's start and end.
#[derive(Clone, Debug)]
pub struct MotionWindow {
    pub samples: Vec<MotionSample>,
    /// The AirPods session the samples came from; a new session means the
    /// AirPods were re-seated.
    pub session: u64,
}

impl MotionWindow {
    pub fn duration(&self) -> Duration {
        match (self.samples.first(), self.samples.last()) {
            (Some(first), Some(last)) => Duration::from_nanos(last.t_ns.saturating_sub(first.t_ns)),
            _ => Duration::ZERO,
        }
    }

    /// The measured sample rate.
    pub fn hertz(&self) -> f64 {
        let seconds = self.duration().as_secs_f64();
        if self.samples.len() < 2 || seconds <= 0.0 {
            return 0.0;
        }
        (self.samples.len() - 1) as f64 / seconds
    }
}

/// The AirPods motion state `hex training status` prints.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub available: bool,
    pub authorization: Authorization,
    pub connected: bool,
    pub session: u64,
    /// Measured over the samples observed while the status was taken.
    pub hertz: f64,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Authorization {
    #[default]
    NotDetermined,
    Restricted,
    Denied,
    Authorized,
}

impl Authorization {
    fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Restricted,
            2 => Self::Denied,
            3 => Self::Authorized,
            _ => Self::NotDetermined,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NotDetermined => "not asked yet",
            Self::Restricted => "restricted",
            Self::Denied => "denied",
            Self::Authorized => "allowed",
        }
    }
}

/// Whether a capture is open now, and which one, so a late release from an
/// older capture cannot end a newer one's motion.
static CAPTURE: AtomicU64 = AtomicU64::new(0);

struct Watching {
    capture: u64,
    session: u64,
}

static WATCHING: Mutex<Option<Watching>> = Mutex::new(None);

fn next_capture() -> u64 {
    CAPTURE.fetch_add(1, Ordering::AcqRel) + 1
}

/// Starts keeping motion for a capture whose shortcut was pressed now. Does
/// nothing unless Labs › AirPods training data is on.
pub fn begin() {
    if !crate::labs::airpods_training() {
        return;
    }
    let capture = next_capture();
    drop_replaced(capture);
    if let Err(error) = platform::start() {
        tracing::info!(%error, "AirPods motion is unavailable for this dictation");
        return;
    }
    let session = platform::session();
    {
        let mut watching = WATCHING.lock().unwrap_or_else(|error| error.into_inner());
        *watching = Some(Watching { capture, session });
    }
    spawn_watchdog(capture);
}

/// Stops a replaced capture's updates when a new press begins before the old
/// one ended (a cancelled dictation whose release never reached us).
fn drop_replaced(_capture: u64) {
    if WATCHING
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_some()
    {
        platform::stop();
        *WATCHING.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }
}

/// Stops a capture's updates without keeping anything (cancelled, discarded,
/// or failed dictations).
pub fn abort() {
    if WATCHING
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
        .is_some()
    {
        platform::stop();
    }
}

/// Whether CoreMotion is delivering samples for the open capture, for the
/// HUD's motion indicator. `None` when no capture is open (the feature is off,
/// or the dictation is over), so the HUD clears the indicator.
pub fn live() -> Option<bool> {
    WATCHING
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_ref()?;
    Some(
        platform::newest_sample()
            .is_some_and(|newest| flowing(CaptureInstant::now().as_nanos(), newest)),
    )
}

/// A sample this recent means the stream is delivering now.
fn flowing(now_ns: u64, newest_ns: u64) -> bool {
    now_ns.saturating_sub(newest_ns) <= FLOWING_WITHIN.as_nanos() as u64
}

/// Ends the open capture. With the clip's real audio window, returns the
/// motion covering it, padded by [`LOOK_BACK`] at the front; without one,
/// drops whatever was kept.
pub fn end(clip: Option<(CaptureInstant, CaptureInstant)>) -> Option<MotionWindow> {
    let watching = WATCHING
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()?;
    let Some((start, end)) = clip else {
        platform::stop();
        return None;
    };
    let from = start.checked_sub(LOOK_BACK).unwrap_or(start).as_nanos();
    let to = end.as_nanos();
    let samples = platform::samples(from, to);
    platform::stop();
    (!samples.is_empty()).then_some(MotionWindow {
        samples,
        session: watching.session,
    })
}

fn spawn_watchdog(capture: u64) {
    let _ = std::thread::Builder::new()
        .name("motion-watchdog".into())
        .spawn(move || {
            std::thread::sleep(WATCHDOG);
            let mut watching = WATCHING.lock().unwrap_or_else(|error| error.into_inner());
            if watching
                .as_ref()
                .is_some_and(|open| open.capture == capture)
            {
                tracing::warn!("AirPods motion ran past its limit; stopping it");
                *watching = None;
                drop(watching);
                platform::stop();
            }
        });
}

/// The AirPods' state for `hex training status`, measured over about a
/// second of updates.
pub fn measure(duration: Duration) -> Status {
    let mut status = Status {
        available: platform::available(),
        authorization: platform::authorization(),
        ..Status::default()
    };
    if !status.available && status.authorization == Authorization::NotDetermined {
        // CoreMotion still answers for a pair that is not worn; ask about the
        // hardware only, with no updates.
        return status;
    }
    if let Err(error) = platform::start() {
        status.error = Some(error);
        return status;
    }
    std::thread::sleep(duration);
    let now = CaptureInstant::now();
    let from = now
        .checked_sub(duration + Duration::from_secs(1))
        .unwrap_or(CaptureInstant::ZERO);
    let mut window = MotionWindow {
        samples: platform::samples(from.as_nanos(), now.as_nanos()),
        session: platform::session(),
    };
    status.connected = platform::connected();
    status.session = window.session;
    status.hertz = window.hertz();
    status.error = platform::error();
    window.samples.clear();
    platform::stop();
    status
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{CStr, CString};

    use super::{Authorization, MotionSample};

    #[repr(C)]
    struct RawSample {
        t: f64,
        ax: f64,
        ay: f64,
        az: f64,
        rx: f64,
        ry: f64,
        rz: f64,
        pitch: f64,
        roll: f64,
        yaw: f64,
        gx: f64,
        gy: f64,
        gz: f64,
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
    }

    #[derive(serde::Deserialize)]
    struct Sample {
        t: f64,
        ax: f64,
        ay: f64,
        az: f64,
        rx: f64,
        ry: f64,
        rz: f64,
        pitch: f64,
        roll: f64,
        yaw: f64,
        gx: f64,
        gy: f64,
        gz: f64,
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
    }

    // A compile-time guard against the bridge's field order drifting from the
    // JSON keys above.
    const _: () = assert!(std::mem::size_of::<RawSample>() == 17 * 8);

    unsafe extern "C" {
        fn hex_motion_authorization() -> i32;
        fn hex_motion_available() -> bool;
        fn hex_motion_start() -> i32;
        fn hex_motion_stop();
        fn hex_motion_connected() -> bool;
        fn hex_motion_session() -> u64;
        fn hex_motion_error() -> *mut std::ffi::c_char;
        fn hex_motion_samples(from: f64, to: f64) -> *mut std::ffi::c_char;
        fn hex_motion_newest_sample() -> f64;
        fn hex_motion_free_string(string: *mut std::ffi::c_char);
    }

    /// Takes ownership of a bridge string.
    fn take_string(pointer: *mut std::ffi::c_char) -> Option<String> {
        if pointer.is_null() {
            return None;
        }
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { hex_motion_free_string(pointer) };
        Some(value)
    }

    pub(super) fn authorization() -> Authorization {
        Authorization::from_code(unsafe { hex_motion_authorization() })
    }

    pub(super) fn available() -> bool {
        unsafe { hex_motion_available() }
    }

    pub(super) fn connected() -> bool {
        unsafe { hex_motion_connected() }
    }

    pub(super) fn session() -> u64 {
        unsafe { hex_motion_session() }
    }

    pub(super) fn error() -> Option<String> {
        take_string(unsafe { hex_motion_error() })
    }

    pub(super) fn start() -> Result<(), String> {
        match unsafe { hex_motion_start() } {
            0 => Ok(()),
            1 => Err("these headphones do not report motion".into()),
            2 => Err("CoreMotion has no access to head motion".into()),
            _ => Err("CoreMotion could not start motion updates".into()),
        }
    }

    pub(super) fn stop() {
        unsafe { hex_motion_stop() }
    }

    /// The newest buffered sample's time, ns: the buffer holds only the open
    /// capture, because `start` clears it.
    pub(super) fn newest_sample() -> Option<u64> {
        let newest = unsafe { hex_motion_newest_sample() };
        (newest > 0.0).then_some(newest as u64)
    }

    pub(super) fn samples(from: u64, to: u64) -> Vec<MotionSample> {
        let Some(json) = take_string(unsafe { hex_motion_samples(from as f64, to as f64) }) else {
            return Vec::new();
        };
        // The bridge writes only what it read from CoreMotion, so a malformed
        // answer means no samples rather than a failed capture.
        let raw: Vec<Sample> = serde_json::from_str(&json).unwrap_or_default();
        raw.into_iter()
            .filter(|sample| sample.t >= 0.0)
            .map(|sample| MotionSample {
                t_ns: sample.t as u64,
                acceleration: [sample.ax, sample.ay, sample.az],
                rotation_rate: [sample.rx, sample.ry, sample.rz],
                attitude: [sample.pitch, sample.roll, sample.yaw],
                gravity: [sample.gx, sample.gy, sample.gz],
                quaternion: [sample.qw, sample.qx, sample.qy, sample.qz],
            })
            .collect()
    }

    #[allow(dead_code)]
    fn unused(_: CString) {}
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{Authorization, MotionSample};

    pub(super) fn authorization() -> Authorization {
        Authorization::NotDetermined
    }

    pub(super) fn available() -> bool {
        false
    }

    pub(super) fn connected() -> bool {
        false
    }

    pub(super) fn session() -> u64 {
        0
    }

    pub(super) fn error() -> Option<String> {
        None
    }

    pub(super) fn start() -> Result<(), String> {
        Err("AirPods motion is only available on macOS".into())
    }

    pub(super) fn stop() {}

    pub(super) fn newest_sample() -> Option<u64> {
        None
    }

    pub(super) fn samples(_from: u64, _to: u64) -> Vec<MotionSample> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_ns: u64, pitch: f64) -> MotionSample {
        MotionSample {
            t_ns,
            acceleration: [0.0; 3],
            rotation_rate: [0.0; 3],
            attitude: [pitch, 0.0, 0.0],
            gravity: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn hertz_comes_from_the_samples_own_times() {
        let samples = (0..25)
            .map(|index| sample(index * 40_000_000, 0.0))
            .collect();
        let window = MotionWindow {
            samples,
            session: 1,
        };
        assert!((window.hertz() - 25.0).abs() < 0.01);
        assert_eq!(window.duration(), Duration::from_millis(960));
    }

    #[test]
    fn a_quiet_stream_is_not_flowing() {
        let newest = 10_000_000_000;
        assert!(flowing(newest + 80_000_000, newest));
        // One CoreMotion update late is late; half a second of nothing is a
        // pair that stopped sending.
        assert!(flowing(newest + 500_000_000, newest));
        assert!(!flowing(newest + 501_000_000, newest));
        // A sample stamped after now (the clock moved under us) still counts.
        assert!(flowing(newest - 1_000_000, newest));
    }

    #[test]
    fn attitude_spread_reads_the_largest_axis_turn() {
        let samples = vec![sample(0, 0.1), sample(1, 0.4), sample(2, -0.2)];
        assert!((MotionSample::attitude_spread(&samples) - 0.6).abs() < 1e-9);
        assert_eq!(MotionSample::attitude_spread(&[]), 0.0);
    }
}
