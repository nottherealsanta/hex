//! `hex training`: the Labs › AirPods training data folder, from the command
//! line, for scripts and the trainer. Read-only except for discarding or
//! deleting examples.

use std::time::Duration;

use clap::Subcommand;
use color_eyre::Result;
use color_eyre::eyre::eyre;

use crate::app_settings::AppSettings;
use crate::labs::Feature;
use crate::training_store;

#[derive(Debug, Subcommand)]
pub enum TrainingCommand {
    /// Show the AirPods' state, today's count, and this run's rejections (the
    /// default).
    Status {
        /// Print the versioned JSON reply instead of lines.
        #[arg(long)]
        json: bool,
    },
    /// List saved examples, oldest first.
    List {
        /// Show at most this many, newest last.
        #[arg(long)]
        limit: Option<usize>,
        /// Print the versioned JSON reply instead of lines.
        #[arg(long)]
        json: bool,
    },
    /// Discard the newest example (for example, after dictating a password).
    Delete {
        /// Print the versioned JSON reply instead of lines.
        #[arg(long)]
        json: bool,
    },
    /// Delete every saved example.
    Purge {
        /// Required: purging removes every example.
        #[arg(long)]
        yes: bool,
    },
}

pub fn run(command: Option<&TrainingCommand>) -> Result<()> {
    match command {
        None | Some(TrainingCommand::Status { .. }) => {
            let json = matches!(command, Some(TrainingCommand::Status { json: true }));
            status(json)
        }
        Some(TrainingCommand::List { json, limit }) => list(*json, *limit),
        Some(TrainingCommand::Delete { json }) => delete(*json),
        Some(TrainingCommand::Purge { yes }) => purge(*yes),
    }
}

fn status(json: bool) -> Result<()> {
    let settings = AppSettings::load()?;
    let on = Feature::AirPodsTraining.on(&settings);
    let motion = crate::motion::measure(Duration::from_millis(1_200));
    let today = training_store::today();
    let stored = training_store::examples()?;
    let total_seconds: f64 = stored
        .iter()
        .map(|example| example.meta.duration_ms as f64 / 1000.0)
        .sum();
    let bytes: u64 = stored.iter().map(|example| example.bytes).sum();
    let report = crate::training_capture::report();
    let folder = training_store::root()?;
    if json {
        let reply = serde_json::json!({
            "v": 1,
            "enabled": on,
            "settingsKey": Feature::AirPodsTraining.settings_key(),
            "airpods": {
                "available": motion.available,
                "connected": motion.connected,
                "authorization": motion.authorization.label(),
                "hertz": motion.hertz,
                "session": motion.session,
                "error": motion.error,
            },
            "today": { "examples": today.examples, "seconds": today.seconds },
            "total": { "examples": stored.len(), "seconds": total_seconds, "bytes": bytes },
            "thisRun": {
                "saved": report.saved,
                "rejections": report.rejections
                    .iter()
                    .map(|(reason, count)| serde_json::json!({ "reason": reason, "count": count }))
                    .collect::<Vec<_>>(),
            },
            "folder": folder,
        });
        println!("{}", serde_json::to_string_pretty(&reply)?);
        return Ok(());
    }
    println!(
        "AirPods training data   {} (hex labs {} airpods-training)",
        if on { "on" } else { "off" },
        if on { "off" } else { "on" }
    );
    let airpods = if motion.connected {
        format!("connected · {:.0} Hz", motion.hertz)
    } else if motion.available {
        "not connected".to_owned()
    } else {
        "no headphones report motion".to_owned()
    };
    println!(
        "AirPods                 {airpods} (CoreMotion: {})",
        motion.authorization.label()
    );
    if let Some(error) = &motion.error {
        println!("CoreMotion error        {error}");
    }
    println!(
        "Collected today         {} · {}",
        examples(today.examples),
        duration(today.seconds)
    );
    println!(
        "Collected in all        {} · {} · {}",
        examples(stored.len()),
        duration(total_seconds),
        bytes_label(bytes)
    );
    let rejections = report
        .rejections
        .iter()
        .map(|(reason, count)| format!("{reason} {count}"))
        .collect::<Vec<_>>()
        .join(" · ");
    println!(
        "This run                saved {}{}",
        report.saved,
        if rejections.is_empty() {
            String::new()
        } else {
            format!(" · not saved: {rejections}")
        }
    );
    println!("Folder                  {}", folder.display());
    Ok(())
}

fn list(json: bool, limit: Option<usize>) -> Result<()> {
    let mut stored = training_store::examples()?;
    if let Some(limit) = limit
        && stored.len() > limit
    {
        stored.drain(..stored.len() - limit);
    }
    if json {
        let examples: Vec<_> = stored
            .iter()
            .map(|example| {
                serde_json::json!({
                    "folder": example.path,
                    "text": example.meta.text,
                    "rawText": example.meta.raw_text,
                    "durationMs": example.meta.duration_ms,
                    "motionHz": example.meta.motion_hz,
                    "motionCoverage": example.meta.motion_coverage,
                    "session": example.meta.session,
                    "piece": example.meta.piece,
                    "pieces": example.meta.pieces,
                    "createdAt": example.meta.created_at,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&examples)?);
        return Ok(());
    }
    for example in &stored {
        println!(
            "{:<19} {:>6} ms  {:>4.0} Hz  {:.2}  {}",
            example.meta.created_at,
            example.meta.duration_ms,
            example.meta.motion_hz,
            example.meta.motion_coverage,
            example.meta.text,
        );
    }
    println!("{}", examples(stored.len()));
    Ok(())
}

fn delete(json: bool) -> Result<()> {
    match training_store::discard_last()? {
        Some(example) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "v": 1, "deleted": example.path, "text": example.meta.text })
                );
            } else {
                println!(
                    "Discarded {} ({}).",
                    example.meta.text,
                    example.path.display()
                );
            }
            Ok(())
        }
        None => {
            if json {
                println!("{}", serde_json::json!({ "v": 1, "deleted": null }));
            } else {
                println!("There is no training example to discard.");
            }
            Ok(())
        }
    }
}

fn purge(yes: bool) -> Result<()> {
    if !yes {
        return Err(eyre!(
            "deleting every example needs --yes: hex training purge --yes"
        ));
    }
    let removed = training_store::purge()?;
    println!("Deleted {}.", examples(removed));
    Ok(())
}

fn examples(count: usize) -> String {
    if count == 1 {
        "1 example".to_owned()
    } else {
        format!("{count} examples")
    }
}

fn duration(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    if seconds < 60 {
        format!("{seconds} s")
    } else {
        format!("{} min", (seconds + 30) / 60)
    }
}

fn bytes_label(bytes: u64) -> String {
    if bytes < 1_000_000 {
        format!("{:.0} kB", bytes as f64 / 1_000.0)
    } else {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    }
}
