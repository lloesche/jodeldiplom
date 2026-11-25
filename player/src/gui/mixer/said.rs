//! What each strip of the Mixer tab says, from the core's desk state, as
//! lines of pieces ([`Piece`]: text, lamps, bars, numbers) that
//! [`super::strips`] lays out: the player's format and heads, the
//! preview's two inputs, a stem's state, the microphone's port and echo
//! canceller, the master's device, lamps and clock. [`text`] gives them as
//! plain text.

use iced::Color;
use jodel_output::lanes::Parts;
use jodel_output::mixer::{InputView, MicPort, Role};
use jodel_output::StreamFormat;

use crate::core::desk::{self, taper, Channel, StripKind};
use crate::core::text::{self as core_text, khz_short};
use crate::core::{karaoke, State};
use crate::gui::style::{ACCENT, DIM, FAIL, GREEN, SOURCE, TEXT, WARN};
/// A piece of a line under a strip's name.
pub(super) enum Piece {
    Text(String, Color),
    /// A lamp: lit in its colour or not.
    Lamp(bool, Color),
    /// A bar `share` full, `width` wide.
    Bar(f32, f32, Color),
    /// A number that changes (fixed-width digits).
    Digits(String, Color),
}

pub(super) type Line = Vec<Piece>;

fn t(s: impl Into<String>, c: Color) -> Piece {
    Piece::Text(s.into(), c)
}

/// Whether the stems are heard, and the word for how they stand.
pub(super) fn stems_said(state: &State) -> (bool, &'static str) {
    let Some(parts) = state.desk.parts() else {
        return (false, "—");
    };
    if parts.state != Parts::None {
        return if parts.share > 0.0 {
            (true, "heard")
        } else {
            (false, "ready")
        };
    }
    let track = state.playing.track.as_ref();
    if track.is_some_and(|t| state.desk.parts_failed.contains(&t.index)) {
        return (false, "no stems");
    }
    match state.gpu {
        _ if track.is_none() => (false, "—"),
        crate::core::Gpu::Present(_) => (false, "separating…"),
        _ => {
            let kept = track
                .and_then(|t| state.facts.get(&t.info.path))
                .and_then(|f| f.stems)
                .unwrap_or(false);
            (false, if kept { "loading…" } else { "no stems" })
        }
    }
}

/// A format, short: `96k 24b`.
fn format_short(f: &StreamFormat) -> String {
    format!("{}k {}b", khz_short(f.rate), f.sample.valid_bits())
}

/// What a stem's strip says: only what its fader and buttons do not.
fn part_lines(state: &State, kind: StripKind) -> Vec<Line> {
    let setting = state.desk.setting(kind);
    let lane = kind
        .lane()
        .map_or(1.0, |k| crate::core::stems::lanes_due(state)[k]);
    if setting.mute {
        vec![vec![t("muted", FAIL)]]
    } else if lane < 1.0 {
        let db = taper::number(taper::db_of_gain(lane), 0, false);
        vec![vec![t(format!("voice {db}"), WARN)]]
    } else {
        Vec::new()
    }
}

/// What a fader strip carries, under its name.
pub(super) fn source_lines(state: &State, kind: StripKind) -> Vec<Line> {
    let inputs: Vec<&InputView> = state
        .desk
        .view(kind)
        .map_or(Vec::new(), |s| s.inputs.iter().collect());
    match kind {
        StripKind::Drums | StripKind::Bass | StripKind::Vocals | StripKind::Other => {
            part_lines(state, kind)
        }
        StripKind::Player => {
            let lead = inputs.iter().find(|i| i.role == Role::Lead).copied();
            let mut lines = vec![format_line(lead.and_then(|i| i.format.as_ref()))];
            lines.push(head_line(lead, state.playing.key));
            let other = state
                .desk
                .open_heads()
                .into_iter()
                .find(|i| i.role != Role::Lead);
            if let Some(i) = other {
                lines.push(head_line(Some(i), 0));
            }
            let fill = lead.map_or(0.0, |i| i.queued as f64 / i.capacity.max(1) as f64);
            let dropped: u64 = inputs.iter().map(|i| i.dropped).sum();
            let mut ring = vec![t("ring", DIM), Piece::Bar(fill as f32, 48.0, SOURCE)];
            if dropped > 0 {
                ring.push(Piece::Digits(format!("↓{dropped}"), WARN));
            }
            lines.push(ring);
            lines
        }
        StripKind::Preview => {
            let format = inputs.iter().find_map(|i| i.format.as_ref());
            let mut lines = vec![format_line(format)];
            for (k, name) in ["A", "B"].iter().enumerate() {
                lines.push(match inputs.get(k) {
                    Some(i) => vec![
                        t(*name, DIM),
                        Piece::Lamp(i.heard, GREEN),
                        Piece::Digits(taper::number(taper::db_of_gain(i.gain), 1, false), TEXT),
                    ],
                    None => vec![t(*name, DIM), t("—", DIM)],
                });
            }
            lines
        }
        // No player: the karaoke game's sounds.
        StripKind::Effects => vec![vec![t("game sounds", DIM)]],
    }
}

/// The format a strip carries, short (`—`: none).
fn format_line(format: Option<&StreamFormat>) -> Line {
    vec![t(
        format.map_or_else(|| "—".to_owned(), format_short),
        SOURCE,
    )]
}

/// A head of the player: lit while heard, LEAD or NEXT, its key, a
/// transition's gain on it.
fn head_line(input: Option<&InputView>, key: i8) -> Line {
    let lead = input.is_none_or(|i| i.role == Role::Lead);
    let mut line = vec![
        Piece::Lamp(input.is_some_and(|i| i.heard), GREEN),
        t(if lead { "LEAD" } else { "NEXT" }, TEXT),
    ];
    if key != 0 {
        line.push(t(crate::core::key::text(key), ACCENT));
    }
    if let Some(a) = input
        .and_then(|i| i.automation.as_ref())
        .filter(|a| !a.settled)
    {
        let db = taper::db_of_gain(a.input * a.main);
        line.push(Piece::Digits(taper::number(db, 1, false), WARN));
    }
    line
}

/// What the microphone's strip says: closed, opening, its name, its port,
/// the echo canceller, the clock bridge.
pub(super) fn mic_lines(state: &State) -> Vec<Line> {
    let shown = &state.karaoke.mic;
    if !desk::listens(state) {
        return vec![
            vec![Piece::Lamp(false, GREEN), t("closed", DIM)],
            vec![t("select it", DIM)],
            vec![t("to listen", DIM)],
        ];
    }
    let Some(name) = &shown.name else {
        return match &shown.error {
            Some(why) => vec![vec![t("no microphone", TEXT)], vec![t(why.clone(), FAIL)]],
            None => vec![vec![t(
                format!("{} opening…", core_text::spinner(state)),
                DIM,
            )]],
        };
    };
    let reading = state.mic;
    let mut lines = vec![vec![t(name.clone(), TEXT)]];
    let rate = reading.map_or(String::new(), |m| format!("{}k", khz_short(m.rate)));
    let mut port = vec![t(rate, DIM)];
    port.extend(match &state.desk.mic_port.port {
        MicPort::Hearing => vec![Piece::Lamp(true, GREEN), t("hearing", TEXT)],
        MicPort::Idle => vec![Piece::Lamp(false, GREEN), t("no output", DIM)],
        MicPort::CannotHear(why) => vec![t(format!("✕ {why}"), FAIL)],
        MicPort::None => vec![Piece::Lamp(false, GREEN), t("not on the desk", DIM)],
    });
    lines.push(port);
    let echo_on = state.config.karaoke_echo;
    lines.push(match (echo_on, reading.and_then(|m| m.echo())) {
        (false, _) => vec![t("echo off", DIM)],
        (true, None) => vec![t("echo waiting", DIM)],
        (true, Some((_, false, _))) => vec![Piece::Lamp(false, GREEN), t("learning", DIM)],
        (true, Some((removed, true, delay))) => {
            let delay = delay.map_or(String::new(), |d| format!(" {:.0}ms", d * 1000.0));
            vec![
                Piece::Lamp(true, GREEN),
                Piece::Digits(
                    format!(
                        "{} dB{delay}",
                        taper::number(-f64::from(removed.abs()), 0, false)
                    ),
                    TEXT,
                ),
            ]
        }
    });
    if let Some(m) = reading {
        lines.push(vec![
            t("drift", DIM),
            Piece::Digits(format!("{:+.0}ppm", m.drift_ppm), TEXT),
        ]);
        lines.push(vec![
            t("fill", DIM),
            Piece::Digits(format!("{:.0}ms", m.fill_ms), TEXT),
            t("xrun", DIM),
            Piece::Digits(m.xruns.to_string(), TEXT),
        ]);
    }
    lines
}

/// The desk's clock: minutes, seconds and tenths.
fn desk_clock(seconds: f64) -> String {
    let tenths = (seconds.max(0.0) * 10.0).floor() as u64;
    format!("{}.{}", core_text::clock((tenths / 10) as f64), tenths % 10)
}

/// The master's lines: the device's backend and format, the LIMIT lamp,
/// the clips and the desk's clock.
pub(super) fn master_lines(state: &State) -> Vec<Line> {
    let master = state.desk.master.as_ref();
    let stream = master.and_then(|m| m.stream.as_ref());
    let clips = master.map_or(0, |m| m.clipped);
    let clock = master.filter(|m| m.timebase.rate > 0).map_or_else(
        || "—".to_owned(),
        |m| desk_clock(m.timebase.heard as f64 / f64::from(m.timebase.rate)),
    );
    vec![
        match stream {
            Some(s) => vec![t(s.backend, TEXT)],
            None => vec![t("closed", DIM)],
        },
        match stream {
            Some(s) => vec![t(
                format!(
                    "{} kHz · {} bit · {} ch",
                    khz_short(s.format.rate),
                    s.format.sample.valid_bits(),
                    s.format.channels
                ),
                SOURCE,
            )],
            None => vec![t("—", DIM)],
        },
        vec![
            Piece::Lamp(state.desk.limiting(state.now_ms), FAIL),
            t("LIMIT", TEXT),
        ],
        vec![
            t("clips", DIM),
            Piece::Digits(clips.to_string(), if clips > 0 { FAIL } else { TEXT }),
            t("·", DIM),
            Piece::Digits(clock, TEXT),
        ],
    ]
}

/// The player's own volume, under the master's meter.
pub(super) fn volume_line(state: &State) -> Line {
    let db = state.desk.volume;
    vec![
        t("VOLUME", TEXT),
        Piece::Bar((taper::position_of(db) / taper::UNITY) as f32, 60.0, ACCENT),
        Piece::Digits(taper::label(db), TEXT),
    ]
}

/// The microphone's PITCH and REC lamps (the game, the singing kept):
/// yellow while set, green while they hear (in Karaoke mode, playing).
pub(super) fn taps_line(state: &State) -> Line {
    let playing = state.playing.state == jodel_playback::PlayState::Playing;
    let tap = |set: bool| {
        if set && karaoke::in_mode(state) && playing {
            Piece::Lamp(true, GREEN)
        } else {
            Piece::Lamp(set, ACCENT)
        }
    };
    vec![
        t("PITCH", DIM),
        tap(state.config.karaoke_game),
        t("REC", DIM),
        tap(state.config.karaoke_keep),
    ]
}

/// What `channel`'s strip says, line by line, as plain text: a lamp lit
/// `●`, unlit `○`, a bar six `▮` `▯`, the pieces a space apart.
pub fn text(state: &State, channel: Channel) -> Vec<String> {
    let lines = match channel {
        Channel::Strip(kind) => source_lines(state, kind),
        Channel::Mic => {
            let mut lines = mic_lines(state);
            lines.push(taps_line(state));
            lines
        }
        Channel::Main => {
            let mut lines = master_lines(state);
            lines.push(volume_line(state));
            lines
        }
    };
    lines.iter().map(|line| line_text(line)).collect()
}

/// What the bracket over the stems says, as plain text: `● STEMS heard`.
pub fn stems(state: &State) -> String {
    let (lit, said) = stems_said(state);
    format!("{} STEMS {said}", if lit { "●" } else { "○" })
}

fn line_text(line: &[Piece]) -> String {
    let pieces: Vec<String> = line
        .iter()
        .map(|piece| match piece {
            Piece::Text(s, _) | Piece::Digits(s, _) => s.clone(),
            Piece::Lamp(lit, _) => if *lit { "●" } else { "○" }.to_owned(),
            Piece::Bar(share, _, _) => {
                let lit = (share.clamp(0.0, 1.0) * 6.0).round() as usize;
                "▮".repeat(lit) + &"▯".repeat(6 - lit)
            }
        })
        .collect();
    pieces.join(" ")
}
