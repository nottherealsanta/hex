//! Turns each successful dictation into training examples: the AirPods'
//! head motion of the clip, the text that was pasted, and the word timings
//! Parakeet produced for it (Labs › AirPods training data).
//!
//! Everything here runs after the paste, on its own thread, so it can never
//! slow or break a dictation. One dictation is `begin` at submit (where the
//! clip's real audio window is known and motion updates stop) and `finish`
//! once the paste succeeded, which is the only place the final text exists.
//! A dictation that is cancelled, fails, is near-silent, or has no AirPods
//! contributes nothing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio::CaptureInstant;
use crate::motion::{MotionSample, MotionWindow};
use crate::training_store::{Meta, Word};
use crate::transcript::TranscriptSegment;
use crate::transcription::Transcript;
use crate::transcription_engine::TranscriptionEngine;

/// The motion model reads at most six seconds at a time.
pub const MAX_PIECE_MS: f64 = 5_800.0;
/// Shorter pieces are fragments, not examples.
pub const MIN_PIECE_MS: f64 = 400.0;
/// Breathing room around each piece's words.
const PAD_MS: f64 = 150.0;
/// A pause this long ends a sentence.
const PAUSE_MS: f64 = 450.0;
const MIN_COVERAGE: f64 = 0.90;
const MIN_HZ: f64 = 20.0;
const MAX_HEAD_TURN_RAD: f64 = 0.5;
/// Parakeet reports no per-word confidence, so this check runs only when a
/// model does.
const MIN_CONFIDENCE: f64 = 0.6;
/// A speech-to-motion offset beyond this means the clocks disagree and the
/// labels are misaligned; nothing from that session is saved.
const MAX_CLOCK_OFFSET_MS: i64 = 80;
/// The correlation a self-check's peak needs before its offset is trusted.
const MIN_CLOCK_CORRELATION: f64 = 0.3;
/// A gap this long in the motion stream counts as missing.
const MOTION_GAP_MS: u64 = 100;
/// How long a finished capture waits for its paste.
const LABEL_WAIT: Duration = Duration::from_secs(60);
/// How long the word pass waits for the engine to be free.
const WORDS_WAIT: Duration = Duration::from_secs(30);

/// Why captures were not saved, for `hex training status`.
#[derive(Default)]
struct Counters {
    saved: AtomicU64,
    /// Reason and count; a list rather than a map so it can live in a
    /// `static` (a `HashMap` cannot be built there).
    rejections: Mutex<Vec<(&'static str, u64)>>,
}

static COUNTERS: Counters = Counters {
    saved: AtomicU64::new(0),
    rejections: Mutex::new(Vec::new()),
};

impl Counters {
    fn reject(&self, reason: &'static str) {
        tracing::info!(reason, "training example not saved");
        let mut rejections = self
            .rejections
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match rejections.iter_mut().find(|(name, _)| *name == reason) {
            Some((_, count)) => *count += 1,
            None => rejections.push((reason, 1)),
        }
    }
}

/// This process's running totals.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub saved: u64,
    /// Reason and count, most frequent first.
    pub rejections: Vec<(&'static str, u64)>,
}

pub fn report() -> Report {
    let saved = COUNTERS.saved.load(Ordering::Acquire);
    let mut rejections = COUNTERS
        .rejections
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    rejections.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(right.0)));
    Report { saved, rejections }
}

/// The text of one completed paste, handed over by the output worker.
struct Label {
    text: String,
    raw: String,
}

struct Job {
    id: u64,
    start: CaptureInstant,
    duration_ms: u64,
    window: MotionWindow,
    samples: Vec<f32>,
    engine: Option<Arc<TranscriptionEngine>>,
    model: String,
    microphone: Option<String>,
    session: String,
    airpods: String,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);
static PENDING: Mutex<Option<HashMap<u64, Sender<Label>>>> = Mutex::new(None);

fn pending() -> std::sync::MutexGuard<'static, Option<HashMap<u64, Sender<Label>>>> {
    PENDING.lock().unwrap_or_else(|error| error.into_inner())
}

/// Starts one dictation's capture. `window` is its motion, already read from
/// the AirPods at the release; `None` means no motion was kept, for any
/// reason. Returns the id [`finish`] takes, or `None` when this dictation
/// cannot be an example.
pub fn begin(
    window: Option<MotionWindow>,
    start: Option<CaptureInstant>,
    duration_ms: u64,
    samples: &[f32],
    engine: Option<Arc<TranscriptionEngine>>,
) -> Option<u64> {
    if !crate::labs::airpods_training() {
        return None;
    }
    crate::training_store::start_retention();
    let (Some(window), Some(start)) = (window, start) else {
        COUNTERS.reject("no AirPods motion");
        return None;
    };
    let hz = window.hertz();
    if hz < MIN_HZ {
        COUNTERS.reject("motion too slow");
        return None;
    }
    let coverage = coverage(&window, start, duration_ms);
    if coverage < MIN_COVERAGE {
        COUNTERS.reject("motion gaps");
        return None;
    }
    let (_, selection) = crate::app_settings::transcription_selection();
    let model = crate::transcription_models::validate(&selection)
        .map(|definition| match definition.runtime {
            crate::transcription_models::ModelRuntime::Gguf(artifact) => {
                artifact.filename.trim_end_matches(".gguf").to_owned()
            }
            crate::transcription_models::ModelRuntime::AppleSpeech => definition.name.to_owned(),
        })
        .unwrap_or_else(|_| selection.model.as_str().to_owned());
    let (_, device) = crate::app_settings::microphone_selection();
    let id = NEXT_ID.fetch_add(1, Ordering::AcqRel) + 1;
    let (send, receive) = channel();
    pending().get_or_insert_with(HashMap::new).insert(id, send);
    let job = Job {
        id,
        start,
        duration_ms,
        session: format!("{}-s{}", crate::training_store::day_stamp(), window.session),
        window,
        samples: samples.to_vec(),
        engine,
        model,
        microphone: device,
        airpods: "AirPods".to_owned(),
    };
    let spawn = std::thread::Builder::new()
        .name("training-capture".into())
        .spawn(move || run(job, receive));
    if let Err(error) = spawn {
        tracing::warn!(%error, "could not start the training capture");
        pending().get_or_insert_with(HashMap::new).remove(&id);
        return None;
    }
    Some(id)
}

/// Hands the pasted text to a capture. Called only after a successful paste.
pub fn finish(id: u64, text: &str, raw: &str) {
    let sender = pending().get_or_insert_with(HashMap::new).remove(&id);
    if let Some(sender) = sender {
        let _ = sender.send(Label {
            text: text.to_owned(),
            raw: raw.to_owned(),
        });
    }
}

/// Forgets a capture whose dictation never reached a paste.
pub fn cancel(id: u64) {
    if let Some(senders) = pending().as_mut() {
        senders.remove(&id);
    }
}

fn run(job: Job, receive: Receiver<Label>) {
    let words = match &job.engine {
        Some(engine) => transcribe_words(engine, job.samples.clone()),
        None => None,
    };
    let Ok(label) = receive.recv_timeout(LABEL_WAIT) else {
        return;
    };
    if let Some(senders) = pending().as_mut() {
        senders.remove(&job.id);
    }
    let Some(words) = words else {
        COUNTERS.reject("no word timings");
        return;
    };
    match write(&job, &label, &words) {
        Ok(saved) => {
            COUNTERS.saved.fetch_add(saved as u64, Ordering::AcqRel);
            tracing::info!(saved, dictation = job.id, "saved training examples");
        }
        Err(error) => {
            tracing::warn!(%error, "could not save a training example");
        }
    }
}

/// The word pass, on the engine's loaded model as background work: it yields
/// to any dictation sent before it starts, and never loads a second model.
fn transcribe_words(engine: &Arc<TranscriptionEngine>, samples: Vec<f32>) -> Option<Transcript> {
    let deadline = Instant::now() + WORDS_WAIT;
    loop {
        let clip = samples.clone();
        match engine.background(move |model| match model.active() {
            Some(transcriber) => transcriber.transcribe_segments(clip),
            None => Err(color_eyre::eyre::eyre!("no transcription model is loaded")),
        }) {
            Ok(Some(Ok(transcript))) => return Some(transcript),
            Ok(Some(Err(error))) => {
                tracing::info!(%error, "training word pass failed");
                return None;
            }
            // A dictation request or a live stream came first; try again.
            Ok(None) => {}
            Err(error) => {
                tracing::info!(%error, "training word pass could not start");
                return None;
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// One piece of a dictation: its words and its padded audio window.
#[derive(Clone, Debug, PartialEq)]
struct Piece {
    /// Milliseconds from the clip's first sample.
    start_ms: f64,
    end_ms: f64,
    words: Vec<TimedWord>,
}

/// A word with its raw text (punctuation included) and its normalized one.
#[derive(Clone, Debug, PartialEq)]
struct TimedWord {
    text: String,
    raw: String,
    start_ms: f64,
    end_ms: f64,
    confidence: Option<f64>,
}

impl TimedWord {
    fn ends_sentence(&self) -> bool {
        let Some(last) = self.raw.chars().last() else {
            return false;
        };
        matches!(last, '.' | '?' | '!')
    }

    fn ends_clause(&self) -> bool {
        self.raw.ends_with(',') || self.raw.ends_with(';') || self.raw.ends_with(':')
    }
}

fn timed(words: &[TranscriptSegment]) -> Vec<TimedWord> {
    words
        .iter()
        .map(|word| TimedWord {
            text: normalize_word(&word.text),
            raw: word.text.trim().to_owned(),
            start_ms: word.start_ms as f64,
            end_ms: word.end_ms as f64,
            confidence: None,
        })
        .filter(|word| !word.text.is_empty())
        .collect()
}

/// The training target's form of one word: lowercase, no punctuation.
fn normalize_word(text: &str) -> String {
    let lowered = text.trim().to_lowercase();
    let mut normalized = String::with_capacity(lowered.len());
    for character in lowered.chars() {
        if character.is_alphanumeric() || character == '\'' || character == '’' {
            normalized.push(if character == '’' { '\'' } else { character });
        }
    }
    normalized
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .map(normalize_word)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Groups words into sentences, splits sentences longer than the model's
/// window at a comma or the largest pause near the middle, and pads each
/// piece without overlapping its neighbours.
fn slice(words: &[TimedWord], clip_ms: f64) -> Vec<Piece> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (index, word) in words.iter().enumerate() {
        current.push(index);
        let pause = words
            .get(index + 1)
            .map_or(f64::INFINITY, |next| next.start_ms - word.end_ms);
        if word.ends_sentence() || pause >= PAUSE_MS {
            groups.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }

    let mut pieces: Vec<Vec<usize>> = Vec::new();
    for group in groups {
        split_long(&group, words, &mut pieces);
    }

    let spans: Vec<(f64, f64)> = pieces
        .iter()
        .map(|piece| {
            (
                words[piece[0]].start_ms,
                words[piece[piece.len() - 1]].end_ms,
            )
        })
        .collect();
    let padded: Vec<Piece> = pieces
        .into_iter()
        .zip(spans.clone())
        .enumerate()
        .map(|(index, (group, (start, end)))| {
            let floor = if index == 0 { 0.0 } else { spans[index - 1].1 };
            let ceiling = spans.get(index + 1).map_or(clip_ms, |next| next.0);
            Piece {
                start_ms: (start - PAD_MS).max(floor).max(0.0),
                end_ms: (end + PAD_MS).min(ceiling).min(clip_ms),
                words: group
                    .into_iter()
                    .map(|index| words[index].clone())
                    .collect(),
            }
        })
        .collect();
    padded
}

fn split_long(group: &[usize], words: &[TimedWord], out: &mut Vec<Vec<usize>>) {
    let span = words[group[group.len() - 1]].end_ms - words[group[0]].start_ms;
    if span <= MAX_PIECE_MS || group.len() < 2 {
        out.push(group.to_vec());
        return;
    }
    let middle = words[group[0]].start_ms + span / 2.0;
    let mut best: Option<(usize, f64)> = None;
    for (offset, &index) in group.iter().enumerate().take(group.len() - 1) {
        let boundary = words[group[offset + 1]].start_ms;
        let distance = (boundary - middle).abs();
        let clause = words[index].ends_clause();
        let score = if clause { distance - 1_000.0 } else { distance };
        if best.is_none_or(|(_, best_score)| score < best_score) {
            best = Some((offset + 1, score));
        }
    }
    let split = best.map_or(1, |(offset, _)| offset);
    split_long(&group[..split], words, out);
    split_long(&group[split..], words, out);
}

/// The share of `start..start + duration` covered by motion samples, with
/// gaps up to [`MOTION_GAP_MS`] counted as present.
fn coverage(window: &MotionWindow, start: CaptureInstant, duration_ms: u64) -> f64 {
    let from = start.as_nanos();
    let to = from + duration_ms * 1_000_000;
    covered_fraction(&window.samples, from, to)
}

fn covered_fraction(samples: &[MotionSample], from: u64, to: u64) -> f64 {
    let duration = to.saturating_sub(from);
    if duration == 0 {
        return 0.0;
    }
    let gap = MOTION_GAP_MS * 1_000_000;
    let mut covered = 0u64;
    let mut previous: Option<u64> = None;
    for sample in samples {
        if sample.t_ns < from || sample.t_ns > to {
            continue;
        }
        if let Some(previous) = previous {
            let step = sample.t_ns.saturating_sub(previous);
            covered += step.min(gap);
        } else {
            covered += (sample.t_ns - from).min(gap);
        }
        previous = Some(sample.t_ns);
    }
    if let Some(previous) = previous {
        covered += (to - previous).min(gap);
    }
    (covered as f64 / duration as f64).min(1.0)
}

fn samples_between(samples: &[MotionSample], from: u64, to: u64) -> Vec<MotionSample> {
    samples
        .iter()
        .filter(|sample| sample.t_ns >= from && sample.t_ns <= to)
        .copied()
        .collect()
}

/// Speech against jaw motion, measured once per AirPods session: the lag, in
/// milliseconds, that best lines the words up with the motion's energy. The
/// sign is positive when the words say the speech came after the mouth moved.
fn clock_offset_ms(words: &[TimedWord], samples: &[MotionSample], start: u64) -> Option<i64> {
    let last = samples.last()?.t_ns;
    let duration_ms = last.saturating_sub(start) / 1_000_000;
    if duration_ms < 1_000 || words.len() < 3 {
        return None;
    }
    let speech_ms = words
        .iter()
        .map(|word| word.end_ms - word.start_ms)
        .sum::<f64>();
    if speech_ms < 800.0 {
        return None;
    }
    let step_ms = 10u64;
    let buckets = (duration_ms / step_ms) as usize + 1;
    if buckets > 60_000 {
        return None;
    }
    // Motion energy per 10 ms bucket, on the clip's clock (the words' clock).
    // Samples from the look-back before the clip are left out.
    let mut energy = vec![0.0f64; buckets];
    let mut counts = vec![0u32; buckets];
    for sample in samples {
        if sample.t_ns < start {
            continue;
        }
        let offset_ms = (sample.t_ns - start) / 1_000_000;
        let bucket = (offset_ms / step_ms) as usize;
        if bucket >= buckets {
            continue;
        }
        let acceleration = sample
            .acceleration
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let rotation = sample
            .rotation_rate
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        energy[bucket] += acceleration.sqrt() + 0.2 * rotation.sqrt();
        counts[bucket] += 1;
    }
    for (bucket, count) in counts.iter().enumerate() {
        if *count > 0 {
            energy[bucket] /= f64::from(*count);
        }
    }
    // Speech activity per bucket, on the clip's clock.
    let mut activity = vec![0.0f64; buckets];
    for word in words {
        let from = word.start_ms / step_ms as f64;
        let to = from + (word.end_ms - word.start_ms) / step_ms as f64;
        let from = from.max(0.0) as usize;
        let to = (to as usize).min(buckets);
        for bucket in activity.iter_mut().take(to).skip(from) {
            *bucket = 1.0;
        }
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let energy_mean = mean(&energy);
    let activity_mean = mean(&activity);
    let mut best: Option<(i64, f64)> = None;
    for lag in (-200..=200).step_by(step_ms as usize) {
        let mut numerator = 0.0;
        let mut left = 0.0;
        let mut right = 0.0;
        for (bucket, activity) in activity.iter().enumerate() {
            // A positive lag lines speech at this bucket up with motion that
            // much earlier: the words say the speech came after the mouth
            // moved.
            let moved = bucket as i64 - lag / step_ms as i64;
            if moved < 0 || moved as usize >= buckets {
                continue;
            }
            let a = activity - activity_mean;
            let b = energy[moved as usize] - energy_mean;
            numerator += a * b;
            left += a * a;
            right += b * b;
        }
        if left <= 0.0 || right <= 0.0 {
            continue;
        }
        let score = numerator / (left * right).sqrt();
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((lag, score));
        }
    }
    // A weak peak means this dictation cannot settle the question; the next
    // one tries again rather than condemning the session to nothing.
    best.filter(|(_, score)| *score >= MIN_CLOCK_CORRELATION)
        .map(|(lag, _)| lag)
}

/// Sessions whose measured offset is known, and whose examples therefore may
/// or may not be saved.
static SESSION_OFFSETS: Mutex<Option<HashMap<u64, Option<i64>>>> = Mutex::new(None);

fn session_offset(session: u64, measure: impl FnOnce() -> Option<i64>) -> Option<i64> {
    let mut offsets = SESSION_OFFSETS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let offsets = offsets.get_or_insert_with(HashMap::new);
    *offsets.entry(session).or_insert_with(measure)
}

fn write(job: &Job, label: &Label, transcript: &Transcript) -> color_eyre::Result<usize> {
    let words = timed(&transcript.words);
    if words.is_empty() {
        COUNTERS.reject("no words");
        return Ok(0);
    }
    let clip_ms = job.duration_ms as f64;
    let pieces = slice(&words, clip_ms);
    let whole = pieces.len() == 1;
    let start_ns = job.start.as_nanos();
    let dictation = training_id(start_ns);
    let mut saved = 0;
    for (index, piece) in pieces.iter().enumerate() {
        let from = start_ns + (piece.start_ms.max(0.0) * 1_000_000.0) as u64;
        let to = start_ns + (piece.end_ms.max(0.0) * 1_000_000.0) as u64;
        let duration_ms = to.saturating_sub(from) / 1_000_000;
        if !(MIN_PIECE_MS as u64..=MAX_PIECE_MS as u64).contains(&duration_ms) {
            COUNTERS.reject(if duration_ms < MIN_PIECE_MS as u64 {
                "piece too short"
            } else {
                "piece too long"
            });
            continue;
        }
        let samples = samples_between(&job.window.samples, from, to);
        let hz = MotionWindow {
            samples: samples.clone(),
            session: job.window.session,
        }
        .hertz();
        if hz < MIN_HZ {
            COUNTERS.reject("motion too slow");
            continue;
        }
        let coverage = covered_fraction(&job.window.samples, from, to);
        if coverage < MIN_COVERAGE {
            COUNTERS.reject("motion gaps");
            continue;
        }
        let head_turn = MotionSample::attitude_spread(&samples);
        if head_turn >= MAX_HEAD_TURN_RAD {
            COUNTERS.reject("head turned");
            continue;
        }
        let from_words = piece
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        // A dictation saved whole is labelled with the text the user accepted
        // (mode corrections included); a piece keeps its own words, whose
        // timings it must match.
        let text = if whole {
            let accepted = normalize_text(&label.text);
            if accepted.is_empty() {
                from_words
            } else {
                accepted
            }
        } else {
            from_words
        };
        if text.is_empty() {
            COUNTERS.reject("no words");
            continue;
        }
        let confidences: Vec<f64> = piece
            .words
            .iter()
            .filter_map(|word| word.confidence)
            .collect();
        let mean_confidence = confidences.iter().sum::<f64>() / confidences.len().max(1) as f64;
        if !confidences.is_empty() && mean_confidence < MIN_CONFIDENCE {
            COUNTERS.reject("low confidence");
            continue;
        }
        let offset = session_offset(job.window.session, || {
            clock_offset_ms(&piece.words, &job.window.samples, start_ns)
        });
        if let Some(offset) = offset
            && offset.abs() > MAX_CLOCK_OFFSET_MS
        {
            COUNTERS.reject("clock offset");
            continue;
        }
        let word_offset_ms = piece.start_ms;
        let meta = Meta {
            schema: crate::training_store::SCHEMA,
            text: text.clone(),
            raw_text: if whole {
                label.raw.clone()
            } else {
                piece
                    .words
                    .iter()
                    .map(|word| word.raw.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            },
            words: piece
                .words
                .iter()
                .map(|word| Word {
                    text: word.text.clone(),
                    start: ((word.start_ms - word_offset_ms) / 1000.0).max(0.0),
                    end: ((word.end_ms - word_offset_ms) / 1000.0).max(0.0),
                    confidence: word.confidence,
                })
                .collect(),
            dictation: dictation.clone(),
            piece: index as u32,
            pieces: pieces.len() as u32,
            duration_ms,
            motion_hz: hz,
            motion_coverage: coverage,
            max_head_turn_rad: head_turn,
            clock_offset_ms: offset.unwrap_or(0),
            model: job.model.clone(),
            airpods: job.airpods.clone(),
            session: job.session.clone(),
            created_at: String::new(),
            microphone: job.microphone.clone(),
        };
        crate::training_store::save(&meta, &samples)?;
        saved += 1;
    }
    if saved == 0 {
        COUNTERS.reject("no piece passed");
    }
    Ok(saved)
}

fn training_id(start_ns: u64) -> String {
    format!("{start_ns:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: f64, end_ms: f64) -> TimedWord {
        TimedWord {
            text: normalize_word(text),
            raw: text.to_owned(),
            start_ms,
            end_ms,
            confidence: None,
        }
    }

    #[test]
    fn words_normalize_to_the_training_target() {
        assert_eq!(normalize_word("Maria,"), "maria");
        assert_eq!(normalize_word("don’t"), "don't");
        assert_eq!(
            normalize_text("Send the report to Maria."),
            "send the report to maria"
        );
        assert_eq!(normalize_word("—"), "");
    }

    #[test]
    fn short_dictations_stay_one_piece() {
        let words = [word("send", 200.0, 500.0), word("it.", 600.0, 900.0)];
        let pieces = slice(&words, 2_000.0);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].start_ms, 50.0);
        assert_eq!(pieces[0].end_ms, 1_050.0);
        assert_eq!(pieces[0].words.len(), 2);
    }

    #[test]
    fn sentences_break_at_punctuation_and_pauses() {
        let words = [
            word("one.", 100.0, 300.0),
            word("two", 350.0, 500.0),
            word("three", 1_100.0, 1_300.0),
        ];
        let pieces = slice(&words, 1_400.0);
        assert_eq!(pieces.len(), 3);
    }

    #[test]
    fn long_sentences_split_near_the_middle_without_overlapping() {
        let words: Vec<TimedWord> = (0..10)
            .map(|index| {
                let start = index as f64 * 1_000.0;
                word("word,", start, start + 600.0)
            })
            .collect();
        let pieces = slice(&words, 10_000.0);
        assert!(pieces.len() >= 2);
        for piece in &pieces {
            let span = piece.end_ms - piece.start_ms;
            assert!(span <= MAX_PIECE_MS + PAD_MS, "piece is {span} ms");
        }
        for pair in pieces.windows(2) {
            assert!(pair[0].end_ms <= pair[1].start_ms, "pieces overlap");
        }
    }

    #[test]
    fn pieces_never_run_past_the_clip() {
        let words = [word("send", 100.0, 500.0), word("it.", 500.0, 900.0)];
        let pieces = slice(&words, 950.0);
        assert_eq!(pieces[0].end_ms, 950.0);
        assert!(pieces[0].start_ms >= 0.0);
    }

    #[test]
    fn a_gap_of_more_than_the_allowance_lowers_coverage() {
        // One sample every 200 ms: each 100 ms allowance of the gap counts,
        // so a two-second window at 5 Hz is mostly uncovered.
        let samples: Vec<MotionSample> = (0..10)
            .map(|index| MotionSample {
                t_ns: index * 200_000_000,
                acceleration: [0.0; 3],
                rotation_rate: [0.0; 3],
                attitude: [0.0; 3],
                gravity: [0.0; 3],
                quaternion: [1.0, 0.0, 0.0, 0.0],
            })
            .collect();
        let covered = covered_fraction(&samples, 0, 2_000_000_000);
        assert!(covered < MIN_COVERAGE, "covered {covered}");
        // The same samples at 25 Hz cover it.
        let dense: Vec<MotionSample> = (0..50)
            .map(|index| MotionSample {
                t_ns: index * 40_000_000,
                ..samples[0]
            })
            .collect();
        let continuous = covered_fraction(&dense, 0, 2_000_000_000);
        assert!(continuous > 0.95, "covered {continuous}");
    }

    #[test]
    fn clock_offset_finds_a_shifted_speech_onset() {
        // The mouth moves from 1.0 s to 1.9 s; the words say the speech began
        // 60 ms later, so the offset reads positive.
        let samples: Vec<MotionSample> = (0..250)
            .map(|index| {
                let seconds = index as f64 * 0.01;
                let loud = (1.0..1.9).contains(&seconds);
                MotionSample {
                    t_ns: 1_000_000_000 + index * 10_000_000,
                    acceleration: if loud {
                        [0.2, 0.0, 0.0]
                    } else {
                        [0.001, 0.0, 0.0]
                    },
                    rotation_rate: [0.0; 3],
                    attitude: [0.0; 3],
                    gravity: [0.0; 3],
                    quaternion: [1.0, 0.0, 0.0, 0.0],
                }
            })
            .collect();
        let words = [
            word("one", 1_060.0, 1_360.0),
            word("two", 1_400.0, 1_700.0),
            word("three", 1_740.0, 1_960.0),
        ];
        let offset = clock_offset_ms(&words, &samples, 1_000_000_000).unwrap();
        assert!((offset - 60).abs() <= 20, "offset {offset}");
    }

    /// A dictation of `text` with one word per `words`, and motion at 25 Hz
    /// that moves while a word sounds.
    fn dictation(
        start_ms: u64,
        duration_ms: u64,
        sentences: &[(u64, u64, &str)],
    ) -> (Job, Label, Transcript) {
        let session = 7;
        let origin = start_ms * 1_000_000;
        let samples: Vec<MotionSample> = (0..duration_ms / 40 + 1)
            .map(|index| {
                let t_ms = index * 40;
                let speaking = sentences
                    .iter()
                    .any(|(from, to, _)| t_ms >= *from && t_ms < *to);
                MotionSample {
                    t_ns: origin + t_ms * 1_000_000,
                    acceleration: if speaking {
                        [0.05, 0.0, 0.0]
                    } else {
                        [0.002, 0.0, 0.0]
                    },
                    rotation_rate: [0.0; 3],
                    attitude: [0.01, 0.0, 0.0],
                    gravity: [0.0, 0.0, -1.0],
                    quaternion: [1.0, 0.0, 0.0, 0.0],
                }
            })
            .collect();
        let mut words = Vec::new();
        let mut text = Vec::new();
        for (from, to, sentence) in sentences {
            let tokens: Vec<&str> = sentence.split_whitespace().collect();
            let step = (to - from) / tokens.len() as u64;
            for (index, token) in tokens.iter().enumerate() {
                let word_start = from + index as u64 * step;
                words.push(TranscriptSegment {
                    start_ms: word_start as i64,
                    end_ms: (word_start + step - 20) as i64,
                    text: (*token).to_owned(),
                });
                text.push(*token);
            }
        }
        let transcript = Transcript {
            text: text.join(" "),
            segments: Vec::new(),
            words,
        };
        let job = Job {
            id: 1,
            start: CaptureInstant::from_nanos(origin),
            duration_ms,
            window: MotionWindow { samples, session },
            samples: Vec::new(),
            engine: None,
            model: "parakeet-unified-en-0.6b-Q8_0".into(),
            microphone: Some("MacBook Pro Microphone".into()),
            session: format!("2026-11-01-s{session}"),
            airpods: "AirPods".into(),
        };
        let label = Label {
            text: transcript.text.clone(),
            raw: transcript.text.clone(),
        };
        (job, label, transcript)
    }

    #[test]
    fn one_dictation_writes_one_example() {
        let _store = crate::training_store::scratch::store();
        let (job, label, transcript) =
            dictation(1_000, 3_000, &[(400, 2_600, "Send the report to Maria.")]);
        assert_eq!(write(&job, &label, &transcript).unwrap(), 1);
        let stored = crate::training_store::examples().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].meta.text, "send the report to maria");
        assert_eq!(stored[0].meta.pieces, 1);
        assert!(stored[0].meta.motion_hz > 20.0);
        assert!(stored[0].meta.motion_coverage > 0.9);
        assert_eq!(stored[0].meta.clock_offset_ms, 0);
        assert_eq!(
            stored[0].meta.microphone.as_deref(),
            Some("MacBook Pro Microphone")
        );
        assert!(
            stored[0]
                .path
                .join(crate::training_store::MOTION_CSV)
                .exists()
        );
    }

    #[test]
    fn a_long_dictation_becomes_pieces_that_share_its_id() {
        let _store = crate::training_store::scratch::store();
        let (job, label, transcript) = dictation(
            1_000,
            16_000,
            &[
                (500, 4_500, "Send the report to Maria tomorrow."),
                (5_000, 9_000, "Then call the bank about the invoice."),
                (9_500, 13_500, "And book the flight for Friday."),
                (14_000, 15_500, "Thanks."),
            ],
        );
        let saved = write(&job, &label, &transcript).unwrap();
        assert_eq!(saved, 4, "one example per sentence");
        let stored = crate::training_store::examples().unwrap();
        assert_eq!(stored.len(), 4);
        assert!(stored.iter().all(|example| example.meta.pieces == 4));
        let ids: std::collections::HashSet<_> = stored
            .iter()
            .map(|example| example.meta.dictation.clone())
            .collect();
        assert_eq!(ids.len(), 1, "pieces share one dictation id");
        let pieces: Vec<u32> = stored.iter().map(|example| example.meta.piece).collect();
        assert_eq!(pieces, vec![0, 1, 2, 3]);
        assert!(stored[0].meta.text.starts_with("send the report"));
        for example in &stored {
            assert!(example.meta.duration_ms <= MAX_PIECE_MS as u64);
            assert!(example.meta.duration_ms >= MIN_PIECE_MS as u64);
            let first = example.meta.words.first().unwrap();
            assert!(first.start >= 0.0);
        }
    }

    #[test]
    fn a_turned_head_saves_nothing() {
        let _store = crate::training_store::scratch::store();
        let (mut job, label, transcript) =
            dictation(1_000, 3_000, &[(400, 2_600, "Send the report to Maria.")]);
        // A head turn across the piece: the attitude ranges over more than
        // half a radian.
        for (index, sample) in job.window.samples.iter_mut().enumerate() {
            sample.attitude = [if index % 2 == 0 { -0.6 } else { 0.6 }, 0.0, 0.0];
        }
        assert_eq!(write(&job, &label, &transcript).unwrap(), 0);
        assert!(crate::training_store::examples().unwrap().is_empty());
        assert!(
            report()
                .rejections
                .iter()
                .any(|(reason, _)| *reason == "head turned")
        );
    }

    #[test]
    fn a_gap_in_the_motion_saves_nothing() {
        let _store = crate::training_store::scratch::store();
        let (mut job, label, transcript) =
            dictation(1_000, 3_000, &[(400, 2_600, "Send the report to Maria.")]);
        // Keep the ends of the clip and drop everything between them.
        let samples = job.window.samples.clone();
        job.window.samples = samples
            .into_iter()
            .filter(|sample| {
                let ms = (sample.t_ns - job.start.as_nanos()) / 1_000_000;
                !(200..=2_800).contains(&ms)
            })
            .collect();
        assert_eq!(write(&job, &label, &transcript).unwrap(), 0);
        assert!(crate::training_store::examples().unwrap().is_empty());
    }
}
