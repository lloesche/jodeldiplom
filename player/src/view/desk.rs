//! The Mix tab: the mixer drawn as a desk, in cells only (the window
//! rasterises the same cells). Each strip is a box: its name on a scribble
//! strip, what it carries, a tall section with the fader beside its
//! meters (one ruler for both), the level and its buttons. The
//! microphone's strip has meters and no fader; the master shows the
//! device, its lamps and the device's own volume.

use jodel_output::mixer::{InputView, MicPort, Role};
use jodel_output::StreamFormat;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Widget};

use super::colors::*;
use crate::core::desk::meter::{Meter, FLOOR_DB, ZONES};
use crate::core::desk::mouse::cap_row;
use crate::core::desk::{self, taper, Channel, DeskButton, StripKind};
use crate::core::layout::desk::{ChannelLayout, DeskLayout};
use crate::core::text::{self, khz_short, width};
use crate::core::{karaoke, State};

/// The tab's bottom border: its keys.
pub const FOOTER: &str = "← → strip · ↑ ↓ fader · home 0 dB · m mute · s solo · e echo";

/// An unlit meter cell's background.
const METER_BG: Color = Color::Rgb(30, 30, 38);
/// Text on a lit scribble strip or button.
const ON_LIT: Color = Color::Rgb(20, 20, 24);

/// The eighths of a cell, from the bottom: none to full.
const EIGHTHS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// The marks of the microphone's and the master's meters, in dBFS.
const DBFS_MARKS: [f32; 5] = [0.0, -6.0, -18.0, -40.0, -60.0];

/// The ruler a meter is drawn on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeterScale {
    /// −60 to 0 dBFS, linear in dB, from the bottom to the top.
    Dbfs60,
    /// The fader's taper ([`taper`]): the strip's meter beside its fader,
    /// read on the fader's marks, up to +6 dB at the top (above 0 dBFS,
    /// reached only with the fader above 0 dB: the overs).
    Taper,
}

impl MeterScale {
    /// The rows of a meter `h` rows high that `db` (dBFS) lights, from the
    /// bottom.
    fn lit(self, db: f32, h: usize) -> f32 {
        match self {
            MeterScale::Dbfs60 => Meter::share(db) * h as f32,
            MeterScale::Taper => {
                if db == f32::NEG_INFINITY {
                    return 0.0;
                }
                let p = taper::position_of(f64::from(db)) as f32;
                (p * (h as f32 - 1.0) + 0.5).clamp(0.0, h as f32)
            }
        }
    }

    /// The level (dBFS) at the middle of row `k` from the bottom.
    fn db_of_row(self, k: usize, h: usize) -> f32 {
        match self {
            MeterScale::Dbfs60 => FLOOR_DB + (k as f32 + 0.5) / h as f32 * -FLOOR_DB,
            MeterScale::Taper => {
                taper::db_at(k as f64 / (h.saturating_sub(1).max(1)) as f64) as f32
            }
        }
    }
}

/// Draws the desk.
pub fn draw(buf: &mut Buffer, state: &State, layout: &DeskLayout) {
    for c in &layout.channels {
        let selected = state.desk.selected == c.channel;
        frame(buf, c.frame, selected);
        match c.channel {
            Channel::Strip(kind) => fader_strip(buf, state, c, kind, selected),
            Channel::Mic => mic_strip(buf, state, c, selected),
            Channel::Main => master(buf, state, c, selected),
        }
    }
}

fn frame(buf: &mut Buffer, rect: Rect, selected: bool) {
    if rect.width < 2 || rect.height < 2 {
        return;
    }
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(if selected { ACCENT } else { BORDER }))
        .render(rect, buf);
}

/// Writes `line` from `x` in `area`'s row `y`, cut at its right edge.
fn put(buf: &mut Buffer, area: Rect, x: u16, y: u16, line: Line) {
    if y < area.y || y >= area.bottom() || x >= area.right() {
        return;
    }
    buf.set_line(x, y, &line, area.right() - x);
}

/// `line` in the middle of `area`'s first row.
fn centred(buf: &mut Buffer, area: Rect, line: Line) {
    if area.height == 0 {
        return;
    }
    let w = line.width() as u16;
    let x = area.x + area.width.saturating_sub(w) / 2;
    put(buf, area, x, area.y, line);
}

/// The lines of `lines` from the top of `area`, as many as fit.
fn lines_in(buf: &mut Buffer, area: Rect, lines: Vec<Line>) {
    for (k, line) in lines.into_iter().take(usize::from(area.height)).enumerate() {
        put(buf, area, area.x, area.y + k as u16, line);
    }
}

/// The scribble strip: the name, lit in the accent colour when selected.
fn scribble(buf: &mut Buffer, area: Rect, name: &str, selected: bool) {
    let style = if selected {
        Style::new().fg(ON_LIT).bg(ACCENT)
    } else {
        Style::new().fg(TEXT).bg(BUTTON)
    }
    .add_modifier(Modifier::BOLD);
    buf.set_style(area, style);
    let name = text::shorten(name, usize::from(area.width.saturating_sub(2)));
    centred(buf, area, Line::styled(name, style));
}

/// A lamp: `●` lit in `color`, `○` unlit.
fn lamp(lit: bool, color: Color) -> Span<'static> {
    if lit {
        Span::styled("●", Style::new().fg(color))
    } else {
        Span::styled("○", Style::new().fg(DIM))
    }
}

fn dim(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::new().fg(DIM))
}

fn plain(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::new().fg(TEXT))
}

/// A format, short: `96k 24b`.
fn format_short(f: &StreamFormat) -> String {
    format!("{}k {}b", khz_short(f.rate), f.sample.valid_bits())
}

/// The colour of a meter at `db`: red above −6 dBFS, the overs too.
fn zone(db: f32) -> Color {
    if db < ZONES.0 {
        GREEN
    } else if db < ZONES.1 {
        ACCENT
    } else {
        FAIL
    }
}

/// A meter's bar in `rect` on `scale`, lit from the bottom to its level
/// in eighths of a cell, its peak-hold tick above it.
fn bar(buf: &mut Buffer, rect: Rect, m: &Meter, scale: MeterScale) {
    let h = usize::from(rect.height);
    if h == 0 || rect.width == 0 {
        return;
    }
    let eighths = (scale.lit(m.db, h) * 8.0).round() as usize;
    let tick = (m.peak_db > m.db)
        .then(|| scale.lit(m.peak_db, h))
        .filter(|&rows| rows > 0.0)
        .map(|rows| (rows.ceil() as usize).saturating_sub(1).min(h - 1));
    for k in 0..h {
        let y = rect.bottom() - 1 - k as u16;
        let lit = eighths.saturating_sub(k * 8).min(8);
        let symbol = if lit == 0 && tick == Some(k) {
            "─"
        } else {
            EIGHTHS[lit]
        };
        let style = Style::new().fg(zone(scale.db_of_row(k, h))).bg(METER_BG);
        for x in rect.x..rect.right() {
            buf.set_string(x, y, symbol, style);
        }
    }
}

/// The dBFS scale left of a meter: its labels at the rows they top, each
/// with a tick after it.
fn meter_scale(buf: &mut Buffer, scale: Rect) {
    let mut used = Vec::new();
    for db in DBFS_MARKS {
        if scale.height == 0 {
            return;
        }
        let k = ((Meter::share(db) * f32::from(scale.height)).ceil() as u16)
            .saturating_sub(1)
            .min(scale.height - 1);
        let y = scale.bottom() - 1 - k;
        if used.contains(&y) {
            continue;
        }
        used.push(y);
        let label = taper::number(f64::from(db), 0, true);
        let w = width(&label) as u16;
        buf.set_string(
            scale.right().saturating_sub(w),
            y,
            &label,
            Style::new().fg(DIM),
        );
        buf.set_string(scale.right(), y, "┤", Style::new().fg(BORDER));
    }
}

/// The fader: the scale's marks, the track and the cap.
fn fader(buf: &mut Buffer, c: &ChannelLayout, db: f64, selected: bool) {
    let Some(t) = c.track else {
        return;
    };
    if t.height < 2 {
        return;
    }
    for y in t.y..t.bottom() {
        buf.set_string(t.x + 1, y, "│", Style::new().fg(TRACK));
    }
    // The marks, the ends and unity first, where they do not meet.
    let mut marks: Vec<(f64, f64)> =
        vec![taper::MARKS[0], taper::MARKS[1], (0.0, f64::NEG_INFINITY)];
    marks.extend(taper::MARKS[2..].iter().copied());
    let mut used = Vec::new();
    for (p, mark) in marks {
        let y = cap_row(t, p);
        if used.contains(&y) {
            continue;
        }
        used.push(y);
        let label = taper::number(mark, 0, true);
        let style = if mark == 0.0 {
            Style::new().fg(TEXT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(DIM)
        };
        let w = width(&label) as u16;
        buf.set_string(c.scale.right().saturating_sub(w), y, &label, style);
        buf.set_string(t.x + 1, y, "┼", Style::new().fg(BORDER));
    }
    let y = cap_row(t, taper::position_of(db));
    let cap = Style::new()
        .fg(if selected { ACCENT } else { TEXT })
        .add_modifier(Modifier::BOLD);
    buf.set_string(t.x, y, "▐█▌", cap);
}

/// What a fader strip carries, under its name.
fn source_lines(state: &State, kind: StripKind, iw: usize) -> Vec<Line<'static>> {
    let inputs: Vec<&InputView> = state
        .desk
        .view(kind)
        .map_or(Vec::new(), |s| s.inputs.iter().collect());
    let format = inputs
        .iter()
        .find_map(|i| i.format.as_ref())
        .map_or_else(|| "—".to_owned(), format_short);
    let mut lines = vec![Line::from(plain(format))];
    match kind {
        StripKind::Player => {
            let lead = inputs.iter().find(|i| i.role == Role::Lead);
            lines.push(Line::from(vec![
                lamp(lead.is_some_and(|i| i.heard), GREEN),
                plain(" LEAD"),
            ]));
            let fill = lead.map_or(0.0, |i| i.queued as f64 / i.capacity.max(1) as f64);
            let [lit, unlit] = super::blocks(fill, iw.saturating_sub(5).max(1), SOURCE);
            lines.push(Line::from(vec![dim("ring "), lit, unlit]));
        }
        StripKind::Preview => {
            for (k, name) in ["A ", "B "].iter().enumerate() {
                let line = match inputs.get(k) {
                    Some(i) => Line::from(vec![
                        dim(*name),
                        lamp(i.heard, GREEN),
                        plain(format!(
                            " {}",
                            taper::number(taper::db_of_gain(i.gain), 1, false)
                        )),
                    ]),
                    None => Line::from(vec![dim(*name), dim("—")]),
                };
                lines.push(line);
            }
        }
    }
    lines
}

fn fader_strip(
    buf: &mut Buffer,
    state: &State,
    c: &ChannelLayout,
    kind: StripKind,
    selected: bool,
) {
    let name = match kind {
        StripKind::Player => "PLAYER",
        StripKind::Preview => "PREVIEW",
    };
    scribble(buf, c.name, name, selected);
    lines_in(
        buf,
        c.info,
        source_lines(state, kind, usize::from(c.info.width)),
    );
    let setting = state.desk.setting(kind);
    fader(buf, c, setting.db, selected);
    // On the fader's ruler: the red above 0 dBFS is the overs.
    for (r, m) in c.meters.iter().zip(state.desk.meters(kind)) {
        bar(buf, *r, m, MeterScale::Taper);
    }
    // The level: anything but 0 dB adds gain (or takes it away).
    let style = if setting.db == 0.0 {
        Style::new().fg(TEXT)
    } else {
        Style::new().fg(WARN)
    }
    .add_modifier(Modifier::BOLD);
    centred(buf, c.value, Line::styled(taper::label(setting.db), style));
    for &(rect, button) in &c.buttons {
        let (label, on, color) = match button {
            DeskButton::Mute(_) => ("M", setting.mute, FAIL),
            DeskButton::Solo(_) => ("S", setting.solo, ACCENT),
            DeskButton::Echo => continue,
        };
        button_cells(buf, rect, label, on, color);
    }
}

/// A button `[M]`: lit in `color`, else dim.
fn button_cells(buf: &mut Buffer, rect: Rect, label: &str, on: bool, color: Color) {
    let style = if on {
        Style::new()
            .fg(ON_LIT)
            .bg(color)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(DIM).bg(BUTTON)
    };
    let text = format!("[{label}]");
    buf.set_stringn(rect.x, rect.y, &text, usize::from(rect.width), style);
}

/// What the microphone's strip says under its name. While a microphone
/// is open the third line is left for the echo canceller's button.
fn mic_lines(state: &State, iw: usize) -> Vec<Line<'static>> {
    let shown = &state.karaoke.mic;
    if !desk::listens(state) {
        return vec![
            Line::from(vec![lamp(false, GREEN), dim(" closed")]),
            Line::from(dim("select it")),
            Line::from(dim("to listen")),
        ];
    }
    let Some(name) = &shown.name else {
        return match &shown.error {
            Some(why) => vec![
                Line::from(plain("no microphone")),
                Line::from(Span::styled(text::shorten(why, iw), Style::new().fg(FAIL))),
            ],
            None => vec![Line::from(dim(format!(
                "{} opening…",
                text::spinner(state)
            )))],
        };
    };
    let reading = state.mic;
    let mut lines = vec![Line::from(plain(text::shorten(name, iw)))];
    let rate = reading.map_or(String::new(), |m| format!("{}k ", khz_short(m.rate)));
    let port = match &state.desk.mic_port.port {
        MicPort::Hearing => vec![lamp(true, GREEN), plain(" hearing")],
        MicPort::Idle => vec![lamp(false, GREEN), dim(" no output")],
        MicPort::CannotHear(why) => vec![
            Span::styled("✕ ", Style::new().fg(FAIL)),
            Span::styled(why.clone(), Style::new().fg(FAIL)),
        ],
        MicPort::None => vec![lamp(false, GREEN), dim(" not on the desk")],
    };
    let mut row = vec![dim(rate)];
    row.extend(port);
    lines.push(Line::from(row));
    // The echo canceller's button goes here.
    lines.push(Line::default());
    let echo_on = state.config.karaoke_echo;
    lines.push(match (echo_on, reading.and_then(|m| m.echo())) {
        (false, _) => Line::from(dim("off")),
        (true, None) => Line::from(dim("waiting")),
        (true, Some((_, false, _))) => Line::from(vec![lamp(false, GREEN), dim(" learning")]),
        (true, Some((removed, true, delay))) => {
            let delay = delay.map_or(String::new(), |d| format!(" {:.0}ms", d * 1000.0));
            Line::from(vec![
                lamp(true, GREEN),
                plain(format!(
                    " {} dB{delay}",
                    taper::number(-f64::from(removed.abs()), 0, false)
                )),
            ])
        }
    });
    if let Some(m) = reading {
        lines.push(Line::from(vec![
            dim("drift "),
            plain(format!("{:+.0}ppm", m.drift_ppm)),
        ]));
        lines.push(Line::from(vec![
            dim("fill "),
            plain(format!("{:.0}ms", m.fill_ms)),
            dim(" xrun "),
            plain(m.xruns.to_string()),
        ]));
    }
    lines
}

fn mic_strip(buf: &mut Buffer, state: &State, c: &ChannelLayout, selected: bool) {
    scribble(buf, c.name, "MIC", selected);
    lines_in(buf, c.info, mic_lines(state, usize::from(c.info.width)));
    let open = desk::listens(state) && state.karaoke.mic.name.is_some();
    for &(rect, button) in &c.buttons {
        if button == DeskButton::Echo && open {
            button_cells(buf, rect, "ECHO", state.config.karaoke_echo, GREEN);
        }
    }
    if let Some(first) = c.meters.first() {
        if c.section.height > 1 {
            put(
                buf,
                c.section,
                first.x,
                c.section.y,
                Line::from(dim("IN VOICE")),
            );
        }
    }
    meter_scale(buf, c.scale);
    let r = &state.desk.readings;
    for (rect, m) in c.meters.iter().zip([&r.mic_in, &r.mic_voice]) {
        bar(buf, *rect, m, MeterScale::Dbfs60);
    }
    centred(
        buf,
        c.value,
        Line::styled(
            "LISTEN ONLY",
            Style::new().fg(DIM).add_modifier(Modifier::BOLD),
        ),
    );
    // The taps: the game's pitch and the singing kept, lit while they
    // hear (in Karaoke mode), yellow while set to.
    let playing = state.playing.state == jodel_playback::PlayState::Playing;
    let tap = |set: bool| {
        if set && karaoke::in_mode(state) && playing {
            Span::styled("●", Style::new().fg(GREEN))
        } else if set {
            Span::styled("●", Style::new().fg(ACCENT))
        } else {
            Span::styled("○", Style::new().fg(DIM))
        }
    };
    centred(
        buf,
        c.foot,
        Line::from(vec![
            dim("PITCH"),
            tap(state.config.karaoke_game),
            dim(" REC"),
            tap(state.config.karaoke_keep),
        ]),
    );
}

/// Seconds as `m:ss.d`.
fn desk_clock(seconds: f64) -> String {
    let tenths = (seconds.max(0.0) * 10.0).floor() as u64;
    format!("{}.{}", text::clock((tenths / 10) as f64), tenths % 10)
}

fn master(buf: &mut Buffer, state: &State, c: &ChannelLayout, selected: bool) {
    let master = state.desk.master.as_ref();
    let name = match master {
        Some(m) if !m.device.is_empty() => format!("MAIN · {}", m.device),
        _ => "MAIN".to_owned(),
    };
    scribble(buf, c.name, &name, selected);
    let stream = master.and_then(|m| m.stream.as_ref());
    let clips = master.map_or(0, |m| m.clipped);
    let clock = master.filter(|m| m.timebase.rate > 0).map_or_else(
        || "—".to_owned(),
        |m| desk_clock(m.timebase.heard as f64 / f64::from(m.timebase.rate)),
    );
    lines_in(
        buf,
        c.info,
        vec![
            match stream {
                Some(s) => Line::from(vec![
                    plain(s.backend),
                    dim(if s.exclusive { " exclusive" } else { " shared" }),
                ]),
                None => Line::from(dim("closed")),
            },
            match stream {
                Some(s) => Line::from(plain(format!(
                    "{} kHz · {} bit · {} ch",
                    khz_short(s.format.rate),
                    s.format.sample.valid_bits(),
                    s.format.channels
                ))),
                None => Line::from(dim("—")),
            },
            Line::from(vec![
                lamp(
                    master.is_some_and(|m| m.passthrough) && stream.is_some(),
                    GREEN,
                ),
                plain(" BIT-PERFECT  "),
                lamp(state.desk.limiting(state.now_ms), FAIL),
                plain(" LIMIT"),
            ]),
            Line::from(vec![
                dim("clips "),
                Span::styled(
                    clips.to_string(),
                    Style::new().fg(if clips > 0 { FAIL } else { TEXT }),
                ),
            ]),
        ],
    );
    if c.info.height > 3 {
        let w = width(&clock) as u16;
        put(
            buf,
            c.info,
            c.info.right().saturating_sub(w),
            c.info.y + 3,
            Line::from(plain(clock)),
        );
    }
    meter_scale(buf, c.scale);
    let r = &state.desk.readings.master;
    for (rect, m) in c.meters.iter().zip(r) {
        bar(buf, *rect, m, MeterScale::Dbfs60);
        if rect.y > 0 && rect.width > 0 {
            // The clip lamp over it, its channel's name under it.
            let x = rect.x + rect.width.saturating_sub(1) / 2;
            buf.set_line(
                x,
                rect.y - 1,
                &Line::from(lamp(m.clipped(state.now_ms), FAIL)),
                1,
            );
        }
    }
    for (rect, name) in c.meters.iter().zip(["L", "R"]) {
        if rect.height > 0 {
            let x = rect.x + rect.width.saturating_sub(1) / 2;
            buf.set_string(x, rect.bottom(), name, Style::new().fg(DIM));
        }
    }
    // The device's own volume: after the samples, so nothing changes in
    // them.
    let volume = match state.device_volume {
        Some(v) => {
            let [lit, unlit] = super::blocks(f64::from(v.level), 8, ACCENT);
            let said = match (v.muted, v.db) {
                (Some(true), _) => "muted".to_owned(),
                (_, Some(db)) if v.level < 1.0 => taper::label(f64::from(db)),
                _ => format!("{:.0} %", v.level * 100.0),
            };
            Line::from(vec![
                plain("VOLUME "),
                lit,
                unlit,
                plain(format!(" {said}")),
            ])
        }
        None => Line::from(vec![plain("VOLUME "), dim("—")]),
    };
    centred(buf, c.value, volume);
    centred(buf, c.foot, Line::from(dim("adds no gain")));
}
