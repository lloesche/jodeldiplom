//! What each strip of the Mixer tab says, from the core's desk state: its
//! name, lines of pieces ([`Piece`]: text, lamps, bars, numbers) in the
//! look's tones ([`Tone`]), its level and its buttons ([`Said`]), which
//! [`super::strips`] lays out: the player's format and heads, the
//! preview's two inputs, a stem's state, the microphone's port and echo
//! canceller, the master's device, lamps and clock. [`text`] gives the
//! lines as plain text; [`Said`]'s outline gives the whole strip with its
//! tones.

use std::fmt;

use jodel_output::lanes::Parts;
use jodel_output::mixer::{InputView, MicPort, Role};
use jodel_output::StreamFormat;

use crate::core::desk::{self, taper, Channel, DeskButton, StripKind};
use crate::core::text::{self as core_text, khz_short};
use crate::core::{karaoke, State};

/// The look's tone a piece is in ([`crate::gui::look`]'s colours).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// Primary text.
    Text,
    /// Secondary text and labels.
    Dim,
    /// Hints.
    Faint,
    /// What is set or active.
    Accent,
    /// Heard, hearing, good.
    Good,
    /// A level off unity, a gain on a head.
    Warn,
    /// Muted, clipped, failed.
    Fail,
    /// A format, the ring.
    Source,
}

impl Tone {
    /// Its name in an outline.
    fn name(self) -> &'static str {
        match self {
            Tone::Text => "text",
            Tone::Dim => "dim",
            Tone::Faint => "faint",
            Tone::Accent => "accent",
            Tone::Good => "good",
            Tone::Warn => "warn",
            Tone::Fail => "fail",
            Tone::Source => "source",
        }
    }
}

/// A piece of a line under a strip's name.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    /// Words.
    Text(String, Tone),
    /// A lamp: lit in its tone or not.
    Lamp(bool, Tone),
    /// A bar `share` full, `width` points wide.
    Bar(f32, f32, Tone),
    /// A number that changes (tabular figures).
    Digits(String, Tone),
}

/// A line under a strip's name, its pieces left to right.
pub type Line = Vec<Piece>;

fn t(s: impl Into<String>, c: Tone) -> Piece {
    Piece::Text(s.into(), c)
}

/// A push button of a strip: mute, solo, the echo canceller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Push {
    /// What it says.
    pub label: &'static str,
    /// Lit: its setting is on.
    pub on: bool,
    /// Its tone while lit.
    pub tone: Tone,
    /// What it presses.
    pub button: DeskButton,
}

/// Everything a strip says, top to bottom: its name (lit while selected),
/// the lines under it, its level, its buttons and the lines at its foot.
/// Its tall section (the fader, the scale and the meters) is drawn
/// between the lines and the level (the microphone's ECHO above it).
#[derive(Clone, Debug, PartialEq)]
pub struct Said {
    /// Whose.
    pub channel: Channel,
    /// The name on its scribble strip.
    pub name: String,
    /// The strip selected (the keys act on it).
    pub selected: bool,
    /// The lines under the name.
    pub lines: Vec<Line>,
    /// The level's readout, and its tone (none on the master: its VOLUME
    /// line says it).
    pub level: Option<(String, Tone)>,
    /// Its push buttons.
    pub pushes: Vec<Push>,
    /// The lines at its foot.
    pub foot: Vec<Line>,
}

/// What `channel`'s strip says.
pub fn said(state: &State, channel: Channel) -> Said {
    let selected = state.desk.selected == channel;
    match channel {
        Channel::Strip(kind) => {
            let setting = state.desk.setting(kind);
            // A stem's strip is narrow: its short name.
            let name = if kind.is_part() {
                kind.short_label()
            } else {
                kind.label()
            };
            let level = taper::label(setting.db);
            let tone = if setting.db == 0.0 {
                Tone::Text
            } else {
                Tone::Warn
            };
            Said {
                channel,
                name: name.to_owned(),
                selected,
                lines: source_lines(state, kind),
                level: Some((level, tone)),
                pushes: vec![
                    Push {
                        label: "M",
                        on: setting.mute,
                        tone: Tone::Fail,
                        button: DeskButton::Mute(kind),
                    },
                    Push {
                        label: "S",
                        on: setting.solo,
                        tone: Tone::Accent,
                        button: DeskButton::Solo(kind),
                    },
                ],
                foot: Vec::new(),
            }
        }
        Channel::Mic => {
            let open = desk::listens(state) && state.karaoke.mic.name.is_some();
            Said {
                channel,
                name: "MIC".to_owned(),
                selected,
                lines: mic_lines(state),
                level: Some(("LISTEN ONLY".to_owned(), Tone::Dim)),
                pushes: if open {
                    vec![Push {
                        label: "ECHO",
                        on: state.config.karaoke_echo,
                        tone: Tone::Good,
                        button: DeskButton::Echo,
                    }]
                } else {
                    Vec::new()
                },
                foot: vec![taps_line(state)],
            }
        }
        Channel::Main => {
            let name = match state.desk.master.as_ref() {
                Some(m) if !m.device.is_empty() => format!("MAIN · {}", m.device),
                _ => "MAIN".to_owned(),
            };
            Said {
                channel,
                name,
                selected,
                lines: master_lines(state),
                level: None,
                pushes: Vec::new(),
                foot: vec![
                    volume_line(state),
                    vec![t("the player's volume", Tone::Faint)],
                ],
            }
        }
    }
}

impl fmt::Display for Said {
    /// The strip's outline: its name, then a line each, each piece with
    /// its tone in brackets.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lit = if self.selected { " selected" } else { "" };
        writeln!(f, "[name{lit}] {}", self.name)?;
        for line in &self.lines {
            writeln!(f, "  {}", line_outline(line))?;
        }
        if let Some((level, tone)) = &self.level {
            writeln!(f, "  [level {}] {level}", tone.name())?;
        }
        if !self.pushes.is_empty() {
            let pushes: Vec<String> = self
                .pushes
                .iter()
                .map(|p| {
                    let lit = if p.on { p.tone.name() } else { "off" };
                    format!("[button {lit}] {}", p.label)
                })
                .collect();
            writeln!(f, "  {}", pushes.join(" "))?;
        }
        for line in &self.foot {
            writeln!(f, "  {}", line_outline(line))?;
        }
        Ok(())
    }
}

/// A line with each piece's tone: `[good] ● [text] LEAD`.
fn line_outline(line: &[Piece]) -> String {
    line.iter()
        .map(|piece| match piece {
            Piece::Text(_, c) | Piece::Digits(_, c) | Piece::Bar(_, _, c) => {
                format!("[{}] {}", c.name(), piece_text(piece))
            }
            Piece::Lamp(lit, c) => {
                let tone = if *lit { c.name() } else { "off" };
                format!("[{tone}] {}", piece_text(piece))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
        vec![vec![t("muted", Tone::Fail)]]
    } else if lane < 1.0 {
        let db = taper::number(taper::db_of_gain(lane), 0, false);
        vec![vec![t(format!("voice {db}"), Tone::Warn)]]
    } else {
        Vec::new()
    }
}

/// What a fader strip carries, under its name.
fn source_lines(state: &State, kind: StripKind) -> Vec<Line> {
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
            let mut ring = vec![
                t("ring", Tone::Dim),
                Piece::Bar(fill as f32, 48.0, Tone::Source),
            ];
            if dropped > 0 {
                ring.push(Piece::Digits(format!("↓{dropped}"), Tone::Warn));
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
                        t(*name, Tone::Dim),
                        Piece::Lamp(i.heard, Tone::Good),
                        Piece::Digits(
                            taper::number(taper::db_of_gain(i.gain), 1, false),
                            Tone::Text,
                        ),
                    ],
                    None => vec![t(*name, Tone::Dim), t("—", Tone::Dim)],
                });
            }
            lines
        }
        // No player: the karaoke game's sounds.
        StripKind::Effects => vec![vec![t("game sounds", Tone::Dim)]],
    }
}

/// The format a strip carries, short (`—`: none).
fn format_line(format: Option<&StreamFormat>) -> Line {
    vec![t(
        format.map_or_else(|| "—".to_owned(), format_short),
        Tone::Source,
    )]
}

/// A head of the player: lit while heard, LEAD or NEXT, its key, a
/// transition's gain on it.
fn head_line(input: Option<&InputView>, key: i8) -> Line {
    let lead = input.is_none_or(|i| i.role == Role::Lead);
    let mut line = vec![
        Piece::Lamp(input.is_some_and(|i| i.heard), Tone::Good),
        t(if lead { "LEAD" } else { "NEXT" }, Tone::Text),
    ];
    if key != 0 {
        line.push(t(crate::core::key::text(key), Tone::Accent));
    }
    if let Some(a) = input
        .and_then(|i| i.automation.as_ref())
        .filter(|a| !a.settled)
    {
        let db = taper::db_of_gain(a.input * a.main);
        line.push(Piece::Digits(taper::number(db, 1, false), Tone::Warn));
    }
    line
}

/// What the microphone's strip says: closed, opening, its name (`▾`: a
/// click opens the microphones' list), its port, the echo canceller, the
/// clock bridge.
fn mic_lines(state: &State) -> Vec<Line> {
    let shown = &state.karaoke.mic;
    if !desk::listens(state) {
        return vec![
            vec![Piece::Lamp(false, Tone::Good), t("closed", Tone::Dim)],
            vec![t("select it", Tone::Dim)],
            vec![t("to listen", Tone::Dim)],
        ];
    }
    let Some(name) = &shown.name else {
        return match &shown.error {
            Some(why) => vec![
                vec![t("▾", Tone::Dim), t("no microphone", Tone::Text)],
                vec![t(why.clone(), Tone::Fail)],
            ],
            None => vec![vec![t(
                format!("{} opening…", core_text::spinner(state)),
                Tone::Dim,
            )]],
        };
    };
    let reading = state.mic;
    let mut lines = vec![vec![t("▾", Tone::Dim), t(name.clone(), Tone::Text)]];
    let rate = reading.map_or(String::new(), |m| format!("{}k", khz_short(m.rate)));
    let mut port = vec![t(rate, Tone::Dim)];
    port.extend(match &state.desk.mic_port.port {
        MicPort::Hearing => vec![Piece::Lamp(true, Tone::Good), t("hearing", Tone::Text)],
        MicPort::Idle => vec![Piece::Lamp(false, Tone::Good), t("no output", Tone::Dim)],
        MicPort::CannotHear(why) => vec![t(format!("✕ {why}"), Tone::Fail)],
        MicPort::None => vec![
            Piece::Lamp(false, Tone::Good),
            t("not on the desk", Tone::Dim),
        ],
    });
    lines.push(port);
    let echo_on = state.config.karaoke_echo;
    lines.push(match (echo_on, reading.and_then(|m| m.echo())) {
        (false, _) => vec![t("echo off", Tone::Dim)],
        (true, None) => vec![t("echo waiting", Tone::Dim)],
        (true, Some((_, false, _))) => {
            vec![Piece::Lamp(false, Tone::Good), t("learning", Tone::Dim)]
        }
        (true, Some((removed, true, delay))) => {
            let delay = delay.map_or(String::new(), |d| format!(" {:.0}ms", d * 1000.0));
            vec![
                Piece::Lamp(true, Tone::Good),
                Piece::Digits(
                    format!(
                        "{} dB{delay}",
                        taper::number(-f64::from(removed.abs()), 0, false)
                    ),
                    Tone::Text,
                ),
            ]
        }
    });
    if let Some(m) = reading {
        lines.push(vec![
            t("drift", Tone::Dim),
            Piece::Digits(format!("{:+.0}ppm", m.drift_ppm), Tone::Text),
        ]);
        lines.push(vec![
            t("fill", Tone::Dim),
            Piece::Digits(format!("{:.0}ms", m.fill_ms), Tone::Text),
            t("xrun", Tone::Dim),
            Piece::Digits(m.xruns.to_string(), Tone::Text),
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
fn master_lines(state: &State) -> Vec<Line> {
    let master = state.desk.master.as_ref();
    let stream = master.and_then(|m| m.stream.as_ref());
    let clips = master.map_or(0, |m| m.clipped);
    let clock = master.filter(|m| m.timebase.rate > 0).map_or_else(
        || "—".to_owned(),
        |m| desk_clock(m.timebase.heard as f64 / f64::from(m.timebase.rate)),
    );
    vec![
        match stream {
            Some(s) => vec![t(s.backend, Tone::Text)],
            None => vec![t("closed", Tone::Dim)],
        },
        match stream {
            Some(s) => vec![t(
                format!(
                    "{} kHz · {} bit · {} ch",
                    khz_short(s.format.rate),
                    s.format.sample.valid_bits(),
                    s.format.channels
                ),
                Tone::Source,
            )],
            None => vec![t("—", Tone::Dim)],
        },
        vec![
            Piece::Lamp(state.desk.limiting(state.now_ms), Tone::Fail),
            t("LIMIT", Tone::Text),
        ],
        vec![
            t("clips", Tone::Dim),
            Piece::Digits(
                clips.to_string(),
                if clips > 0 { Tone::Fail } else { Tone::Text },
            ),
            t("·", Tone::Dim),
            Piece::Digits(clock, Tone::Text),
        ],
    ]
}

/// The player's own volume, under the master's meter.
fn volume_line(state: &State) -> Line {
    let db = state.desk.volume;
    vec![
        t("VOLUME", Tone::Text),
        Piece::Bar(
            (taper::position_of(db) / taper::UNITY) as f32,
            60.0,
            Tone::Accent,
        ),
        Piece::Digits(taper::label(db), Tone::Text),
    ]
}

/// The microphone's PITCH and REC lamps (the game, the singing kept):
/// yellow while set, green while they hear (in Karaoke mode, playing).
fn taps_line(state: &State) -> Line {
    let playing = state.playing.state == jodel_playback::PlayState::Playing;
    let tap = |set: bool| {
        if set && karaoke::in_mode(state) && playing {
            Piece::Lamp(true, Tone::Good)
        } else {
            Piece::Lamp(set, Tone::Accent)
        }
    };
    vec![
        t("PITCH", Tone::Dim),
        tap(state.config.karaoke_game),
        t("REC", Tone::Dim),
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
    let pieces: Vec<String> = line.iter().map(piece_text).collect();
    pieces.join(" ")
}

/// A piece as plain text.
fn piece_text(piece: &Piece) -> String {
    match piece {
        Piece::Text(s, _) | Piece::Digits(s, _) => s.clone(),
        Piece::Lamp(lit, _) => if *lit { "●" } else { "○" }.to_owned(),
        Piece::Bar(share, _, _) => {
            let lit = (share.clamp(0.0, 1.0) * 6.0).round() as usize;
            "▮".repeat(lit) + &"▯".repeat(6 - lit)
        }
    }
}
