# AirPods training data

Labs › AirPods training data (macOS, off by default) keeps each dictation's
AirPods head motion together with the text that was pasted, as training data
for a motion-to-text model. It is a data-collection experiment for Hex's
owner: nothing is uploaded, no audio is kept, and dictation behaves exactly as
before.

```ts
Labs > AirPods training data (off)          // labs.md: labs.airpods_training
└── on -> each dictation may become one or more training examples

Hold Option                                  // the ordinary dictation capture
├── motion_bridge.swift starts CoreMotion headphone motion
│   └── no AirPods, or no motion permission -> nothing kept (dictation unchanged)
├── the HUD shows a dot under Dot scope's plate while samples arrive
│   └── motion.rs::live -> a fading mark; a connected pair that sends nothing
│       leaves it dark, so a capture that is not collecting shows at a glance
└── Capture, transcription, paste run exactly as before

Release Option
├── parakeet::submit reads the clip's real audio window out of the motion buffer
│   └── motion.rs stops updates at once; nothing is kept between dictations
└── training_capture::begin decides whether this clip can be an example
    └── no AirPods motion, too few samples, or gaps >= 10% -> rejected, counted

Paste succeeds                               // record_history: the only label
├── the word pass runs on the engine's loaded model as background work
│   └── yields to any dictation started before it; never loads a second model
├── one piece (<= 5.8 s) -> one folder
└── longer -> sentences split at . ? ! or a >= 450 ms pause, then at a comma
             or the largest pause near the middle; 150 ms pad, no overlap

Quality checks per piece                     // counted, never saved
├── motion >= 90% covered and >= 20 Hz
├── 400 ms <= piece <= 5.8 s
├── attitude range < 0.5 rad
├── mean word confidence >= 0.6 (only when the model reports one)
└── speech-to-motion offset within 80 ms (measured once per AirPods session)

Saved folder                                 // ~/Library/Application Support/hex/Training
└── <local day>/<20261101T143205.123-7f3a-0>/
    ├── motion.csv   t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz
    └── meta.json    text, rawText, words, dictation, piece, pieces, durationMs,
                     motionHz, motionCoverage, maxHeadTurnRad, clockOffsetMs,
                     model, airpods, session, createdAt, microphone

Menu bar > Training Data                     // status_item.rs
├── "Collected today: 214 examples · 18 min"
├── Discard Last Example -> removes the newest folder
├── Open Training Folder -> Finder
└── Delete All Training Data -> removes every example

hex training [status|list|delete|purge]      // training_cli.rs, no running Hex
├── status --json -> the AirPods' CoreMotion state, today's count, this run's rejections
├── list --limit N --json -> every example, oldest first
├── delete -> discards the newest example
└── purge --yes -> deletes everything

Retention                                    // training_store::start_retention
└── 90 days and 1 GB, oldest first; runs when the first dictation is captured
    and hourly while Hex runs
```

The text is the training target: lowercase, no punctuation. A dictation saved
whole is labelled with the text the user accepted (mode corrections applied);
each piece of a long dictation keeps its own words, so its word timings always
match its text. Each AirPods connection is one `session`, which the trainer
uses to split train and test data, because re-seating the AirPods changes the
signal.

## What this does not do

- No audio is written, ever; the word pass runs from the clip already in
  memory and the samples are dropped with the job.
- Nothing records between dictations: motion updates run from the shortcut
  press to the release, with a five-minute watchdog.
- No Hex Meetings, Annotate, or voice notes contribute data.
- The HUD's motion mark is the only visible sign: the menu bar count, the
  folder, and `hex training status` say nothing while a dictation records.

## Checks

- `src/motion.rs` and `src/training_capture.rs` unit tests: normalizing
  words, sentence and long-piece slicing without overlap, coverage, the
  speech-to-motion offset, and a whole synthetic dictation saved as one
  example or as four pieces sharing a dictation id.
- `src/motion.rs` and `src/dictation_indicator.rs` tests: a stream counts as
  flowing within half a second of its newest sample, the HUD mark fades in
  and out, Reduce Motion switches it, and a release clears it.
- `src/training_store.rs` tests: folder round trip, discard last, purge,
  retention by size, and an incomplete folder never counting as an example.
- `scripts/check training` runs them all.
- Hardware is unverified here: connection status, real 25 Hz delivery, the
  CoreMotion permission prompt, and the measured offset need a Mac wearing
  AirPods; `hex training status` is the manual check.
- The Labs page adds one row, so the Linux window goldens must be regenerated
  (dispatch `check-linux.yml` and commit its `golden-actual` artifact).

## Verification limits

A saved example proves the pipeline, not the model: spoken dictation is not
silent mouthing, which is why the project's go/no-go point evaluates on the
Speechless app's silent clips (see `../docs/HEX_FORK_PLAN.md` in the Speechless
checkout this fork sits inside).
