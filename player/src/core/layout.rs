//! The layout, in cell units: which panes each screen has at a size and
//! how they collapse when it gets small.
//!
//! Main screen, wide:
//!
//! ```text
//! ╭ Browser │ Queue (3) ───────────╮╭─────────────╮
//! │ ..                             ││   cover     │
//! │ ▸ Album/                       │╰─────────────╯
//! │ ♪ Song             FLAC 44.1   │ title, artist
//! │ / filter                       │ ○ PCM
//! ╰ C:\Music ──────────────────────╯ signal path
//!  spectrum of what is heard
//!  20   100   1k   10k   22k   44k
//!  0:12 ━━━━━━━━━━━━━━━━━━──────────────── 4:01
//!  ◀◀  ▶   ▶▶  ■   message   ≡ settings  ? help
//! ```
//!
//! Collapse rules (columns × rows):
//! - below 72 columns the right column goes; a two-row summary of the
//!   track sits above the spectrum instead,
//! - the spectrum takes 10 rows from 30 rows, 7 from 24, 4 from 18 and
//!   none below,
//! - the cover needs 4 rows; on a short screen it goes before the text,
//! - below 40 × 12 only a "too small" note is shown.

pub mod desk;
mod track;

pub use track::Track;

use ratatui::layout::Rect;

use crate::core::karaoke;
use crate::core::keymap::Op;
use crate::core::singing::Level;
use crate::core::sliders::Slider;
use crate::core::text::{self, width};
use crate::core::{Gpu, Screen, Side, State};

/// The smallest size anything but a note is drawn at.
pub const MIN_SIZE: (u16, u16) = (40, 12);

/// A clickable button.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Button {
    /// Where it is.
    pub rect: Rect,
    /// What it does.
    pub op: Op,
    /// Its text.
    pub label: String,
}

/// The bottom bar: spectrum, progress, transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bottom {
    /// The spectrum picture.
    pub spectrum: Option<Rect>,
    /// The frequency labels under it.
    pub labels: Option<Rect>,
    /// The time heard, left of the bar.
    pub elapsed: Rect,
    /// The progress bar (click or drag to seek).
    pub bar: Rect,
    /// The length, right of the bar.
    pub total: Rect,
    /// The transport row.
    pub transport: Rect,
    /// The buttons on it.
    pub buttons: Vec<Button>,
    /// The space between the buttons, for messages.
    pub message: Rect,
}

/// The main screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainLayout {
    /// The bordered list pane.
    pub list: Rect,
    /// The Browser / Queue toggle pieces in its top border.
    pub toggle: Vec<(Rect, Side)>,
    /// The rows inside it.
    pub rows: Rect,
    /// The filter line, when a filter is set.
    pub filter: Option<Rect>,
    /// The right column.
    pub right: Option<Rect>,
    /// The cover, in the right column.
    pub cover: Option<Rect>,
    /// Title, status light and signal path, in the right column.
    pub info: Option<Rect>,
    /// The two-row track summary when there is no right column.
    pub mini: Option<Rect>,
    /// The Transition tab, when it is the side shown (it then takes the
    /// whole width, with no right column and no spectrum).
    pub transition: Option<TransitionLayout>,
    /// The Karaoke tab, when it is the side shown (the whole width too).
    pub karaoke: Option<KaraokeTabLayout>,
    /// The Mixer tab's desk, when it is the side shown (the whole width too).
    pub desk: Option<desk::DeskLayout>,
    /// The bottom bar.
    pub bottom: Bottom,
}

/// A Karaoke slider's cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliderLayout {
    /// Which.
    pub slider: Slider,
    /// Its row: the label, the track and the value.
    pub row: Rect,
    /// The track its knob moves along (a click or drag sets it).
    pub track: Rect,
    /// The value, right of the track.
    pub value: Rect,
}

/// The Karaoke tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeTabLayout {
    /// The playing track's cover.
    pub cover: Rect,
    /// The title and artist, right of it.
    pub title: Rect,
    /// The button into Karaoke mode.
    pub enter: Rect,
    /// The sliders.
    pub sliders: Vec<SliderLayout>,
    /// The game on or off: its row, a click turns it.
    pub game: Rect,
    /// The levels, side by side: a click picks one.
    pub levels: Vec<(Rect, Level)>,
    /// The notes shown or hidden: its row.
    pub show_notes: Rect,
    /// The singing kept or not: its row.
    pub keep: Rect,
    /// The sync's row: its button (a click starts one), then the delay in
    /// force.
    pub sync: Rect,
    /// The sync's panel over the tab, while one shows.
    pub panel: Option<Rect>,
    /// Its buttons (keep, again, cancel, as the sync stands), and what
    /// each does.
    pub panel_buttons: Vec<(Rect, Op)>,
    /// What karaoke will do, below.
    pub notes: Rect,
    /// The voice's lane with the words on it, under the pane (where the
    /// other tabs have the spectrum).
    pub voice: Option<Rect>,
}

/// Karaoke mode: the whole screen is its picture, and its controls (while
/// they show) a bar over its bottom (the picture stays where it is).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeLayout {
    /// The whole screen.
    pub screen: Rect,
    /// The controls, while they show.
    pub controls: Option<KaraokeControls>,
}

/// Karaoke mode's controls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KaraokeControls {
    /// The bar they are in.
    pub bar: Rect,
    /// The sliders.
    pub sliders: Vec<SliderLayout>,
    /// The START button, while nothing plays.
    pub start: Option<Rect>,
    /// The time heard, the seek bar and the length, under the sliders.
    pub elapsed: Rect,
    /// The seek bar.
    pub seek: Rect,
    /// The length.
    pub total: Rect,
    /// The keys, in a line.
    pub hint: Rect,
}

/// A lane of a block: the whole song or one stem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lane {
    /// The whole song.
    Song,
    /// The drum stem.
    Drums,
    /// The bass stem.
    Bass,
    /// The other instruments.
    Other,
    /// The vocals.
    Vocals,
}

impl Lane {
    /// The lanes with stems, top first.
    pub const ALL: [Lane; 5] = [
        Lane::Song,
        Lane::Drums,
        Lane::Bass,
        Lane::Other,
        Lane::Vocals,
    ];

    /// Its label.
    pub fn label(self) -> &'static str {
        match self {
            Lane::Song => "song",
            Lane::Drums => "drums",
            Lane::Bass => "bass",
            Lane::Other => "other",
            Lane::Vocals => "vocals",
        }
    }

    /// Its place among a waveform's stems (bass, drums, other, vocals);
    /// `None` for the whole song.
    pub fn stem(self) -> Option<usize> {
        match self {
            Lane::Song => None,
            Lane::Bass => Some(0),
            Lane::Drums => Some(1),
            Lane::Other => Some(2),
            Lane::Vocals => Some(3),
        }
    }
}

/// Cells left of a block's lanes for their labels.
pub const TRANSITION_LANE_LABELS: u16 = 7;
/// The most rows a lane of the Transition tab takes.
pub const TRANSITION_LANE_ROWS: u16 = 4;

/// One block of the Transition tab: a header line, the whole track in one row,
/// the bar numbers, and the lanes (all from [`TRANSITION_LANE_LABELS`] in, so their
/// columns are the same times).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionBlockLayout {
    /// The track it shows.
    pub block: crate::core::lanes::Block,
    /// Title, tempo and bar.
    pub header: Rect,
    /// The whole track.
    pub overview: Rect,
    /// Bar numbers and the cursor.
    pub ruler: Rect,
    /// The lanes, top first, each with its label's cell (empty where that
    /// row is out of view) and the part of it in view.
    pub lanes: Vec<(Lane, Rect, Rect)>,
    /// The rows a lane takes, in view or not.
    pub lane_rows: u16,
    /// The rows of the first lane listed that are above the view (while
    /// the block slides out of it).
    pub top_cut: u16,
}

/// The Transition tab: the playing track's block, then the next one's when
/// there is room.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionLayout {
    /// The blocks, top first.
    pub blocks: Vec<TransitionBlockLayout>,
    /// The button that zooms to fit the lyrics ([`FIT_LYRICS`]), in the
    /// pane's bottom border.
    pub fit: Option<Rect>,
}

/// The Now playing screen: the cover filling the screen and a caption
/// (title, artist · album, status light) at its bottom left. A frontend
/// that can draw text over the cover (the window, with a cover) draws it
/// in `cover` and the caption over it; one that cannot (a terminal's
/// graphics, or a placeholder) draws it in `cover_above`, which leaves the
/// caption's rows free. The caption is in the same place either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NowLayout {
    /// The whole screen (a click anywhere returns).
    pub screen: Rect,
    /// The cover, as large as the screen allows (square, in cells of
    /// twice the height of their width).
    pub cover: Rect,
    /// The cover with the caption's rows free under it.
    pub cover_above: Rect,
    /// The caption: under `cover_above` at its left edge, a row above the
    /// bottom (so over `cover`'s bottom left).
    pub caption: Rect,
}

/// One settings row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsLine {
    /// The whole row.
    pub rect: Rect,
    /// The `‹` that steps back.
    pub less: Rect,
    /// The value.
    pub value: Rect,
    /// The `›` that steps on.
    pub more: Rect,
}

/// The settings screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsLayout {
    /// The bordered pane.
    pub pane: Rect,
    /// One line per [`crate::core::SettingsRow`].
    pub lines: Vec<SettingsLine>,
    /// The device list, or the data folder's list, when open.
    pub devices: Option<Rect>,
    /// Notes under the rows (the config file).
    pub notes: Rect,
    /// The bottom bar (no spectrum).
    pub bottom: Bottom,
}

/// The help screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelpLayout {
    /// The bordered pane.
    pub pane: Rect,
    /// The text inside it.
    pub text: Rect,
    /// The search line under the text, while the help is searched (as the
    /// browser's filter line).
    pub search: Option<Rect>,
    /// The bottom bar (no spectrum).
    pub bottom: Bottom,
}

/// The layout of the screen shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Too small to draw anything else.
    TooSmall(Rect),
    /// The main screen.
    Main(Box<MainLayout>),
    /// Now playing.
    NowPlaying(NowLayout),
    /// Settings.
    Settings(SettingsLayout),
    /// Help.
    Help(HelpLayout),
    /// Karaoke mode.
    Karaoke(KaraokeLayout),
}

impl Layout {
    /// The bottom bar, on every screen but the note and Now playing.
    pub fn bottom(&self) -> Option<&Bottom> {
        match self {
            Layout::TooSmall(_) => None,
            Layout::Main(m) => Some(&m.bottom),
            Layout::NowPlaying(_) => None,
            Layout::Settings(s) => Some(&s.bottom),
            Layout::Help(h) => Some(&h.bottom),
            Layout::Karaoke(_) => None,
        }
    }
}

/// The layout for `state` at its size.
pub fn compute(state: &State) -> Layout {
    let (cols, rows) = state.size;
    let area = Rect::new(0, 0, cols, rows);
    if cols < MIN_SIZE.0 || rows < MIN_SIZE.1 {
        return Layout::TooSmall(area);
    }
    match state.screen {
        Screen::Main => Layout::Main(Box::new(main(state, area))),
        Screen::NowPlaying => Layout::NowPlaying(now_playing(state, area)),
        Screen::Settings => Layout::Settings(settings(state, area)),
        Screen::Help => Layout::Help(help(state, area)),
        Screen::Karaoke => Layout::Karaoke(karaoke_mode(state, area)),
    }
}

/// The main screen's layout whatever screen is shown (the list keeps its
/// cursor and scroll behind the other screens), unless too small.
pub fn main_for(state: &State) -> Option<MainLayout> {
    let (cols, rows) = state.size;
    if cols < MIN_SIZE.0 || rows < MIN_SIZE.1 {
        return None;
    }
    Some(main(state, Rect::new(0, 0, cols, rows)))
}

/// Spectrum rows on the main screen for a height.
pub fn spectrum_rows(rows: u16) -> u16 {
    match rows {
        30.. => 10,
        24..=29 => 7,
        18..=23 => 4,
        _ => 0,
    }
}

/// The right column's width for a width and the track shown: as wide as
/// its widest line needs (so the signal path does not wrap), or the widest
/// seen since the last resize ([`State::info_width_held`]), from
/// [`right_width`] up to half the screen; 0 (none) where [`right_width`]
/// has none.
pub fn right_width_for(state: &State, cols: u16) -> u16 {
    let base = right_width(cols);
    if base == 0 {
        return 0;
    }
    // The text is one cell in from the column's edge.
    let widest = text::info_width(state).max(state.info_width_held);
    let need = (widest + 1).min(usize::from(cols)) as u16;
    need.clamp(base, base.max(cols / 2))
}

/// The right column's least width for a width (0: none).
pub fn right_width(cols: u16) -> u16 {
    match cols {
        120.. => (cols * 34 / 100).min(60),
        96..=119 => 34,
        72..=95 => 28,
        _ => 0,
    }
}

/// The cells the stem buttons leave for messages, at least.
const STEMS_ROOM: u16 = 16;

fn bottom(state: &State, area: Rect, spectrum: u16) -> (Rect, Bottom) {
    let used = 2 + spectrum;
    let top = Rect {
        height: area.height.saturating_sub(used),
        ..area
    };
    let y = area.y + top.height;
    let (spectrum_rect, labels) = if spectrum >= 3 {
        (
            Some(Rect::new(
                area.x + 1,
                y,
                area.width.saturating_sub(2),
                spectrum - 1,
            )),
            Some(Rect::new(
                area.x + 1,
                y + spectrum - 1,
                area.width.saturating_sub(2),
                1,
            )),
        )
    } else {
        (None, None)
    };
    let progress_y = y + spectrum;
    let transport_y = progress_y + 1;
    let time_w = 8u16;
    let elapsed = Rect::new(area.x + 1, progress_y, time_w, 1);
    let total = Rect::new(
        area.right().saturating_sub(time_w + 1),
        progress_y,
        time_w,
        1,
    );
    let bar = Rect::new(
        elapsed.right() + 1,
        progress_y,
        total.x.saturating_sub(elapsed.right() + 2),
        1,
    );
    let transport = Rect::new(area.x, transport_y, area.width, 1);
    let play = match state.playing.state {
        jodel_playback::PlayState::Playing => " ▮▮ ",
        _ => " ▶  ",
    };
    let wide = area.width >= 70;
    let mut buttons = Vec::new();
    let mut x = area.x + 1;
    for (label, op) in [
        (" ◀◀ ", Op::Previous),
        (play, Op::PlayPause),
        (" ▶▶ ", Op::Next),
        (" ■ ", Op::Stop),
    ] {
        let w = width(label) as u16;
        buttons.push(Button {
            rect: Rect::new(x, transport_y, w, 1),
            op,
            label: label.into(),
        });
        x += w + 1;
    }
    let right: [(&str, Op); 2] = if wide {
        [(" ≡ settings ", Op::Settings), (" ? help ", Op::Help)]
    } else {
        [(" ≡ ", Op::Settings), (" ? ", Op::Help)]
    };
    let right_w: u16 = right.iter().map(|(l, _)| width(l) as u16 + 1).sum();
    // The stems, with a GPU, where they leave room for messages: by name,
    // else by initial.
    if matches!(state.gpu, Gpu::Present(_)) {
        let room = area.right().saturating_sub(right_w).saturating_sub(x);
        let fitting = [true, false].into_iter().map(|named| {
            let stems: Vec<(String, Op)> = jodel_playback::stems::LANES
                .iter()
                .map(|&stem| (text::stem_button(stem, named), Op::for_stem(stem)))
                .collect();
            let stems_w: u16 = stems.iter().map(|(l, _)| width(l) as u16 + 1).sum();
            (stems, stems_w)
        });
        let chosen = fitting
            .filter(|(_, w)| w + STEMS_ROOM <= room)
            .map(|(stems, _)| stems)
            .next();
        if let Some(stems) = chosen {
            for (label, op) in stems {
                let w = width(&label) as u16;
                buttons.push(Button {
                    rect: Rect::new(x, transport_y, w, 1),
                    op,
                    label,
                });
                x += w + 1;
            }
        }
    }
    let left_end = x;
    let mut x = area.right().saturating_sub(right_w);
    let right_start = x;
    for (label, op) in right {
        let w = width(label) as u16;
        buttons.push(Button {
            rect: Rect::new(x, transport_y, w, 1),
            op,
            label: label.into(),
        });
        x += w + 1;
    }
    let message = Rect::new(
        left_end + 1,
        transport_y,
        right_start.saturating_sub(left_end + 2),
        1,
    );
    (
        top,
        Bottom {
            spectrum: spectrum_rect,
            labels,
            elapsed,
            bar,
            total,
            transport,
            buttons,
            message,
        },
    )
}

/// The lanes a Transition block shows: every stem's where its track has them
/// (or, before its analysis is in, where stems are asked for), else the whole
/// song's.
fn block_lanes(state: &State, block: crate::core::lanes::Block) -> &'static [Lane] {
    let stems = match crate::core::lanes::rhythmed(state, block) {
        Some(r) => r.waveform.as_ref().is_some_and(|w| w.stems.is_some()),
        None => state.rhythm_recipe.stems,
    };
    if stems {
        &Lane::ALL
    } else {
        &Lane::ALL[..1]
    }
}

/// The Transition tab's blocks in `inner`: each a header, an overview and a
/// ruler row and its lanes, as many rows a lane (up to [`TRANSITION_LANE_ROWS`]) as both
/// blocks fit with; the next track's block only where it fits.
fn transition(state: &State, inner: Rect, list: Rect) -> TransitionLayout {
    use crate::core::lanes::Block;
    let lanes = [
        block_lanes(state, Block::Now),
        block_lanes(state, Block::Next),
    ];
    let need = |which: usize, h: u16| 3 + lanes[which].len() as u16 * h;
    let rows = inner.height;
    // The tallest lanes (up to four rows) both blocks fit with.
    let both = (1..=TRANSITION_LANE_ROWS)
        .rev()
        .find(|&h| need(0, h) + 1 + need(1, h) <= rows);
    let plan: Vec<(usize, u16)> = match both {
        Some(h) => vec![(0, h), (1, h)],
        None if need(0, 1) <= rows => vec![(0, 1)],
        None => Vec::new(),
    };
    let mut order: Vec<(Block, &'static [Lane], u16)> = plan
        .into_iter()
        .map(|(which, h)| ([Block::Now, Block::Next][which], lanes[which], h))
        .collect();
    let (top, bottom) = (i32::from(inner.y), i32::from(inner.bottom()));
    let mut y = top;
    // After a hand-over, the track left slides up out of view and the
    // playing one up into its place, the next one in under it.
    if let (Some(done), Some(h)) = (crate::core::transition::viewport::slide(state), both) {
        let prev = block_lanes(state, Block::Prev);
        let pitch = 3 + prev.len() as i32 * i32::from(h) + 1;
        y -= (done * f64::from(pitch)).round() as i32;
        order.insert(0, (Block::Prev, prev, h));
    }
    let x = inner.x + TRANSITION_LANE_LABELS.min(inner.width);
    let w = inner.width.saturating_sub(TRANSITION_LANE_LABELS);
    let shown = |y: i32| y >= top && y < bottom;
    let row = |y: i32, x: u16, w: u16| {
        if shown(y) {
            Rect::new(x, y as u16, w, 1)
        } else {
            Rect::default()
        }
    };
    let mut blocks = Vec::new();
    for (block, block_lanes, h) in order {
        let header = row(y, inner.x, inner.width);
        let overview = row(y + 1, x, w);
        let ruler = row(y + 2, x, w);
        let first = y;
        y += 3;
        let mut placed = Vec::new();
        let mut top_cut = 0;
        for &lane in block_lanes {
            let (from, to) = (y.max(top), (y + i32::from(h)).min(bottom));
            if to > from {
                if placed.is_empty() {
                    top_cut = (from - y) as u16;
                }
                placed.push((
                    lane,
                    row(y, inner.x, TRANSITION_LANE_LABELS.min(inner.width)),
                    Rect::new(x, from as u16, w, (to - from) as u16),
                ));
            }
            y += i32::from(h);
        }
        y += 1;
        if first >= bottom || y <= top {
            continue;
        }
        blocks.push(TransitionBlockLayout {
            block,
            header,
            overview,
            ruler,
            lanes: placed,
            lane_rows: h,
            top_cut,
        });
    }
    // The zoom that fits the lyrics, at the right of the pane's bottom
    // border.
    let w = width(FIT_LYRICS) as u16;
    let fit = (list.width >= w + 24 && list.height >= 2)
        .then(|| Rect::new(list.right().saturating_sub(w + 2), list.bottom() - 1, w, 1));
    TransitionLayout { blocks, fit }
}

/// The button that zooms the Transition tab to fit the lyrics.
pub const FIT_LYRICS: &str = " z fit lyrics ";

/// The main screen with a wide tab (Transition, Karaoke or Mixer): the list pane's
/// border over the whole width, no right column, no spectrum (the Karaoke
/// tab has its voice's lane there).
fn main_wide(state: &State, area: Rect) -> MainLayout {
    // The Karaoke tab has the voice's lane where the spectrum goes.
    let karaoke = state.side == Side::Karaoke;
    let strip = if karaoke {
        spectrum_rows(area.height)
    } else {
        0
    };
    let (top, mut bottom) = bottom(state, area, strip);
    let voice = bottom.spectrum.zip(bottom.labels).map(|(s, l)| s.union(l));
    bottom.spectrum = None;
    bottom.labels = None;
    let list = top;
    let inner = Rect::new(
        list.x + 1,
        list.y + 1,
        list.width.saturating_sub(2),
        list.height.saturating_sub(2),
    );
    MainLayout {
        list,
        toggle: toggle_rects(state, list),
        rows: inner,
        filter: None,
        right: None,
        cover: None,
        info: None,
        mini: None,
        transition: (state.side == Side::Transition).then(|| transition(state, inner, list)),
        karaoke: karaoke.then(|| KaraokeTabLayout {
            voice,
            ..karaoke_tab(inner, state.karaoke.sync.as_ref())
        }),
        desk: (state.side == Side::Mixer)
            .then(|| desk::compute(inner, state.desk.has_stems(), state.desk.heads_open())),
        bottom,
    }
}

/// The list toggle's pieces in the top border of `list`.
fn toggle_rects(state: &State, list: Rect) -> Vec<(Rect, Side)> {
    let mut toggle = Vec::new();
    let mut x = list.x + 1;
    for (label, side, _) in text::toggle(state) {
        let w = width(&label) as u16;
        if let Some(side) = side {
            toggle.push((Rect::new(x, list.y, w, 1), side));
        }
        x += w;
    }
    toggle
}

fn main(state: &State, area: Rect) -> MainLayout {
    if state.side.wide() {
        return main_wide(state, area);
    }
    let (top, bottom) = bottom(state, area, spectrum_rows(area.height));
    let rw = right_width_for(state, area.width);
    let (left, right) = if rw > 0 {
        (
            Rect {
                width: top.width - rw,
                ..top
            },
            Some(Rect::new(top.x + top.width - rw, top.y, rw, top.height)),
        )
    } else {
        (top, None)
    };
    // Without a right column, a summary of the track sits under the list.
    let (list, mini) = if right.is_none() && left.height >= 10 {
        (
            Rect {
                height: left.height - 2,
                ..left
            },
            Some(Rect::new(
                left.x + 1,
                left.y + left.height - 2,
                left.width - 2,
                2,
            )),
        )
    } else {
        (left, None)
    };
    let inner = Rect::new(
        list.x + 1,
        list.y + 1,
        list.width.saturating_sub(2),
        list.height.saturating_sub(2),
    );
    let (rows, filter) = if state.browser.filter.is_some() && state.side == Side::Browser {
        (
            Rect {
                height: inner.height.saturating_sub(1),
                ..inner
            },
            Some(Rect::new(
                inner.x,
                inner.bottom().saturating_sub(1),
                inner.width,
                1,
            )),
        )
    } else {
        (inner, None)
    };
    let toggle = toggle_rects(state, list);
    let (cover, info) = match right {
        Some(r) => {
            // The cover, a free row, then the text (at least 7 rows).
            let text_rows = text::info_lines(state, usize::from(r.width.saturating_sub(1)))
                .len()
                .max(7) as u16;
            let cover_h = (r.width / 2).min(r.height.saturating_sub(text_rows + 1));
            if cover_h >= 4 {
                (
                    Some(Rect::new(r.x, r.y, r.width, cover_h)),
                    Some(Rect::new(
                        r.x + 1,
                        r.y + cover_h + 1,
                        r.width - 1,
                        r.height - cover_h - 1,
                    )),
                )
            } else {
                (None, Some(Rect::new(r.x + 1, r.y, r.width - 1, r.height)))
            }
        }
        None => (None, None),
    };
    MainLayout {
        list,
        toggle,
        rows,
        filter,
        right,
        cover,
        info,
        mini,
        transition: None,
        karaoke: None,
        desk: None,
        bottom,
    }
}

/// The largest square (in cells of twice the height of their width)
/// centred in `area`.
fn square_in(area: Rect) -> Rect {
    let width = area.width.min(area.height.saturating_mul(2));
    let height = area.height.min(width.div_ceil(2));
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}

/// The widest a caption wraps to.
fn caption_wrap(cover_above: Rect, x: u16, area: Rect) -> usize {
    let room = area.right().saturating_sub(x + 2);
    usize::from(cover_above.width.max(24).min(room).max(1))
}

fn now_playing(state: &State, area: Rect) -> NowLayout {
    let cover = square_in(area);
    // The caption's rows and a free row each side of it come off the
    // cover above it; wrapping to that cover's width may take more rows.
    let mut rows = text::caption_lines(state, usize::from(area.width.saturating_sub(4))).len();
    let (mut cover_above, mut lines);
    loop {
        let above = Rect {
            height: area.height.saturating_sub(rows as u16 + 2),
            ..area
        };
        cover_above = square_in(above);
        let x = cover_above.x.max(area.x + 2);
        lines = text::caption_lines(state, caption_wrap(cover_above, x, area));
        if lines.len() <= rows || cover_above.height == 0 {
            break;
        }
        rows = lines.len();
    }
    let x = cover_above.x.max(area.x + 2);
    let height = (lines.len() as u16).min(area.height.saturating_sub(1));
    let width = lines
        .iter()
        .map(|l| l.iter().map(|s| width(&s.text)).sum::<usize>())
        .max()
        .unwrap_or(0) as u16;
    let caption = Rect::new(
        x,
        area.bottom().saturating_sub(1 + height),
        width.min(area.right().saturating_sub(x)),
        height,
    );
    NowLayout {
        screen: area,
        cover,
        cover_above,
        caption,
    }
}

/// The width of a settings label column.
pub const SETTINGS_LABEL: u16 = 24;
/// The widest a settings value is shown.
pub const SETTINGS_VALUE: u16 = 56;

fn settings(state: &State, area: Rect) -> SettingsLayout {
    let (top, bottom) = bottom(state, area, 0);
    let pane = top;
    let inner = Rect::new(
        pane.x + 2,
        pane.y + 2,
        pane.width.saturating_sub(4),
        pane.height.saturating_sub(3),
    );
    let rows = crate::core::SettingsRow::ALL.len() as u16;
    let notes_h = 3u16;
    // A blank line between rows when they fit (with the device list's
    // room when it is open), else none.
    let listing = state.settings.picking || state.settings.folders.is_some();
    let list_h = if listing { 8 } else { 0 };
    let step = if inner.height >= rows * 2 + notes_h + list_h {
        2
    } else {
        1
    };
    let mut lines = Vec::new();
    for (i, _) in crate::core::SettingsRow::ALL.iter().enumerate() {
        let y = inner.y + i as u16 * step;
        let rect = Rect::new(inner.x, y, inner.width, 1);
        let value_x = inner.x + SETTINGS_LABEL;
        let value_w = inner
            .width
            .saturating_sub(SETTINGS_LABEL + 4)
            .min(SETTINGS_VALUE);
        lines.push(SettingsLine {
            rect,
            less: Rect::new(value_x, y, 2, 1),
            value: Rect::new(value_x + 2, y, value_w, 1),
            more: Rect::new(value_x + 2 + value_w, y, 2, 1),
        });
    }
    let rows_end = inner.y + rows * step;
    let devices = listing.then(|| {
        Rect::new(
            inner.x + 2,
            rows_end,
            inner.width.saturating_sub(2),
            inner.bottom().saturating_sub(rows_end + notes_h),
        )
    });
    let notes = Rect::new(
        inner.x,
        inner.bottom().saturating_sub(notes_h),
        inner.width,
        notes_h.min(inner.height),
    );
    SettingsLayout {
        pane,
        lines,
        devices,
        notes,
        bottom,
    }
}

fn help(state: &State, area: Rect) -> HelpLayout {
    let (top, bottom) = bottom(state, area, 0);
    let inner = Rect::new(
        top.x + 2,
        top.y + 1,
        top.width.saturating_sub(4),
        top.height.saturating_sub(2),
    );
    let (text, search) = if state.help_search.is_some() {
        (
            Rect {
                height: inner.height.saturating_sub(1),
                ..inner
            },
            Some(Rect::new(
                inner.x,
                inner.bottom().saturating_sub(1),
                inner.width,
                1,
            )),
        )
    } else {
        (inner, None)
    };
    HelpLayout {
        pane: top,
        text,
        search,
        bottom,
    }
}

/// The label of the button into Karaoke mode.
pub const KARAOKE_ENTER: &str = "▶  Enter Karaoke Mode";
/// The START button's label.
pub const KARAOKE_START: &str = "START";
/// A slider's label column, in cells.
pub const SLIDER_LABEL: u16 = 12;
/// How wide a switch on the Karaoke tab is (" on " or " off ").
pub const SWITCH: u16 = 5;
/// A slider's value column (`100 %`, `−200 ms`), in cells.
const SLIDER_VALUE: u16 = 7;
/// How wide the sync's button is (` sync `).
pub const SYNC_BUTTON: u16 = 6;
/// The sync's panel's height: over the button into the mode and every
/// row of settings.
const PANEL: u16 = 12;
/// The longest a slider's track gets, in cells.
const SLIDER_MOST: u16 = 40;

/// A slider in the row at `x`, `y`, `width` cells wide: its label, the
/// track (up to [`SLIDER_MOST`]) and the value after it.
fn slider_at(slider: Slider, x: u16, y: u16, width: u16) -> SliderLayout {
    let track = width
        .saturating_sub(SLIDER_LABEL + SLIDER_VALUE + 1)
        .min(SLIDER_MOST);
    let row = Rect::new(x, y, SLIDER_LABEL + track + 1 + SLIDER_VALUE, 1);
    SliderLayout {
        slider,
        row,
        track: Rect::new(x + SLIDER_LABEL, y, track, 1),
        value: Rect::new(x + SLIDER_LABEL + track + 1, y, SLIDER_VALUE, 1),
    }
}

/// The Karaoke tab inside the list pane's border: the cover at the left,
/// the title, the button into the mode, the sliders and the sync's row
/// right of it, notes below; while a sync shows (`session`), its panel
/// over the button and the rows.
fn karaoke_tab(inner: Rect, session: Option<&crate::core::sync::Session>) -> KaraokeTabLayout {
    // The cover as tall as the pane allows, a third of its width at most.
    let side = Rect::new(
        inner.x + 1,
        inner.y + 1,
        (inner.width / 3).min(inner.height.saturating_sub(2).saturating_mul(2)),
        inner.height.saturating_sub(2),
    );
    let cover = square_in(side);
    let cover = Rect { y: side.y, ..cover };
    let x = if cover.width > 0 {
        cover.right() + 3
    } else {
        inner.x + 2
    };
    let room = inner.right().saturating_sub(x + 1);
    let row = |k: u16| (inner.y + 1 + k).min(inner.bottom().saturating_sub(1));
    let title = Rect::new(x, row(0), room, 2.min(inner.height));
    let enter_w = (width(KARAOKE_ENTER) as u16 + 4).min(room);
    let enter = Rect::new(x, row(3), enter_w, 3.min(inner.height.saturating_sub(4)));
    // A row each from the seventh, the sync's row right after the delay's.
    let (mut at, mut sync_row, mut sliders) = (7, 0, Vec::new());
    for &s in &Slider::ALL {
        sliders.push(slider_at(s, x, row(at), room));
        at += 1;
        if s == Slider::Delay {
            (sync_row, at) = (at, at + 1);
        }
    }
    let sync = Rect::new(x, row(sync_row), (SLIDER_LABEL + SYNC_BUTTON).min(room), 1);
    // Over the button into the mode and the rows below it.
    let panel =
        session.map(|_| Rect::new(x, row(3), room, PANEL.min(inner.height.saturating_sub(4))));
    // The panel's buttons, side by side on its eighth row inside.
    let mut panel_buttons = Vec::new();
    if let (Some(p), Some(session)) = (panel, session) {
        let mut at = p.x + 2;
        for (label, op) in session.buttons() {
            let w = width(label) as u16 + 2;
            let r = Rect::new(at, p.y + 8, w, 1);
            if r.right() < p.right() && r.y < p.bottom().saturating_sub(1) {
                panel_buttons.push((r, op));
            }
            at += w + 2;
        }
    }
    // The game's settings, a row each: a switch, the levels, a switch.
    let game = Rect::new(x, row(11), (SLIDER_LABEL + SWITCH).min(room), 1);
    let mut at = x + SLIDER_LABEL;
    let levels = Level::ALL
        .iter()
        .map(|&level| {
            let w = width(level.name()) as u16 + 2;
            let r = Rect::new(at, row(12), w, 1);
            at += w + 1;
            (r, level)
        })
        .filter(|(r, _)| r.right() <= x + room)
        .collect();
    let show_notes = Rect::new(x, row(13), (SLIDER_LABEL + SWITCH).min(room), 1);
    let keep = Rect::new(x, row(14), (SLIDER_LABEL + SWITCH).min(room), 1);
    let notes_y = row(16);
    let notes = Rect::new(x, notes_y, room, inner.bottom().saturating_sub(notes_y));
    KaraokeTabLayout {
        cover,
        title,
        enter,
        sliders,
        game,
        levels,
        show_notes,
        keep,
        sync,
        panel,
        panel_buttons,
        notes,
        voice: None,
    }
}

/// Karaoke mode: the whole screen, and the controls' bar over its bottom
/// while they show (the sliders, the seek bar and the keys at the left,
/// START at the right). The game's settings are not here but on the tab:
/// a song is sung at one level from its start to its end.
fn karaoke_mode(state: &State, area: Rect) -> KaraokeLayout {
    let controls = karaoke::controls_shown(state).then(|| {
        let bar = Rect::new(
            area.x,
            area.bottom().saturating_sub(5),
            area.width,
            5.min(area.height),
        );
        let start_w = width(KARAOKE_START) as u16 + 8;
        let start = karaoke::waiting(state).then(|| {
            Rect::new(
                bar.right().saturating_sub(start_w + 2),
                bar.y + 1,
                start_w,
                3,
            )
        });
        let room = bar
            .width
            .saturating_sub(start.map_or(0, |s| s.width + 4))
            .saturating_sub(4);
        let sliders = Slider::MODE
            .iter()
            .enumerate()
            .map(|(k, &s)| slider_at(s, bar.x + 2, bar.y + 1 + k as u16, room))
            .collect();
        // The seek bar as on the other screens: the time, the bar, the
        // length.
        let seek_y = bar.y + 1 + Slider::MODE.len() as u16;
        let time_w = 8u16;
        let elapsed = Rect::new(bar.x + 1, seek_y, time_w.min(room), 1);
        let total = Rect::new(
            (bar.x + 2 + room).saturating_sub(time_w),
            seek_y,
            time_w.min(room),
            1,
        );
        let seek = Rect::new(
            elapsed.right() + 1,
            seek_y,
            total.x.saturating_sub(elapsed.right() + 2),
            1,
        );
        KaraokeControls {
            bar,
            sliders,
            start,
            elapsed,
            seek,
            total,
            hint: Rect::new(bar.x + 2, seek_y + 1, room, 1),
        }
    });
    KaraokeLayout {
        screen: area,
        controls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn state(cols: u16, rows: u16) -> State {
        State::new("/music".into(), Config::default(), (cols, rows))
    }

    fn inside(inner: Rect, outer: Rect) -> bool {
        inner.x >= outer.x
            && inner.y >= outer.y
            && inner.right() <= outer.right()
            && inner.bottom() <= outer.bottom()
    }

    #[test]
    fn wide_main_screen_has_every_pane_inside_the_area() {
        let s = state(160, 48);
        let Layout::Main(m) = compute(&s) else {
            panic!()
        };
        let area = Rect::new(0, 0, 160, 48);
        let right = m.right.unwrap();
        let cover = m.cover.unwrap();
        assert!(inside(m.list, area) && inside(right, area) && inside(cover, right));
        assert!(!m.list.intersects(right));
        assert_eq!(m.list.width + right.width, 160);
        assert!(m.mini.is_none());
        // The cover is about square in pixels: twice as wide as tall.
        assert_eq!(cover.height, right.width / 2);
        let b = &m.bottom;
        assert_eq!(b.spectrum.unwrap().height, 9);
        assert_eq!(b.transport.y, 47);
        assert_eq!(b.bar.y, 46);
        assert!(b.bar.width > 100);
        assert_eq!(m.list.bottom(), b.spectrum.unwrap().y);
        // Buttons do not overlap and sit on the transport row.
        for (i, a) in b.buttons.iter().enumerate() {
            assert_eq!(a.rect.y, 47);
            for c in &b.buttons[i + 1..] {
                assert!(!a.rect.intersects(c.rect), "{a:?} {c:?}");
            }
        }
        assert_eq!(m.toggle.len(), 5);
        assert_eq!(m.toggle[0].0.y, m.list.y);
    }

    #[test]
    fn narrow_screens_drop_the_right_column_and_spectrum() {
        let s = state(60, 20);
        let Layout::Main(m) = compute(&s) else {
            panic!()
        };
        assert!(m.right.is_none() && m.cover.is_none());
        assert_eq!(m.mini.unwrap().height, 2);
        assert_eq!(m.bottom.spectrum.unwrap().height, 3);
        let s = state(60, 15);
        let Layout::Main(m) = compute(&s) else {
            panic!()
        };
        assert!(m.bottom.spectrum.is_none());
        // The list, the two-row summary and the bar fill the height.
        assert_eq!(m.list.height + 2 + 2, 15);
        assert!(matches!(compute(&state(39, 30)), Layout::TooSmall(_)));
        assert!(matches!(compute(&state(80, 11)), Layout::TooSmall(_)));
        // Short buttons on narrow screens.
        assert!(m.bottom.buttons.iter().all(|b| width(&b.label) <= 8));
    }

    #[test]
    fn medium_sizes_keep_a_cover_when_there_is_room() {
        let s = state(80, 24);
        let Layout::Main(m) = compute(&s) else {
            panic!()
        };
        assert_eq!(m.right.unwrap().width, 28);
        let cover = m.cover.unwrap();
        assert!(cover.height >= 4 && m.info.unwrap().height >= 7);
        let s = state(100, 18);
        let Layout::Main(m) = compute(&s) else {
            panic!()
        };
        // 18 rows: 4 of spectrum and 2 of bar leave 12; the cover gets 4,
        // then a free row and 7 of text.
        assert_eq!(m.cover.unwrap().height, 4);
        assert_eq!(m.info.unwrap().y, m.cover.unwrap().bottom() + 1);
    }

    #[test]
    fn now_playing_is_the_cover_with_a_caption() {
        let mut s = state(160, 48);
        s.screen = Screen::NowPlaying;
        let Layout::NowPlaying(n) = compute(&s) else {
            panic!()
        };
        let screen = Rect::new(0, 0, 160, 48);
        assert_eq!(n.screen, screen);
        // The cover is as tall as the screen, square and centred.
        assert_eq!(n.cover, Rect::new(32, 0, 96, 48));
        // The caption: under the smaller cover at its left edge, a row
        // above the bottom, and so over the big cover's bottom left.
        assert!(n.cover_above.bottom() < n.caption.y);
        assert_eq!(n.caption.bottom(), 47);
        assert_eq!(n.caption.x, n.cover_above.x);
        assert!(n.caption.height >= 1 && inside(n.caption, n.cover));
        // No bottom bar.
        assert!(compute(&s).bottom().is_none());
        // Tall and narrow: the cover takes the width.
        s.size = (70, 40);
        let Layout::NowPlaying(n) = compute(&s) else {
            panic!()
        };
        assert_eq!(n.cover.width, 70);
        assert!(inside(n.cover, Rect::new(0, 0, 70, 40)));
        assert!(inside(n.caption, Rect::new(0, 0, 70, 40)));
        assert!(n.cover_above.bottom() < n.caption.y);
    }

    #[test]
    fn settings_rows_and_device_list() {
        let mut s = state(100, 30);
        s.screen = Screen::Settings;
        s.settings.picking = true;
        let Layout::Settings(l) = compute(&s) else {
            panic!()
        };
        let rows = crate::core::SettingsRow::ALL.len();
        assert_eq!(l.lines.len(), rows);
        let devices = l.devices.unwrap();
        assert!(devices.y > l.lines[rows - 1].rect.y && devices.height > 4);
        for line in &l.lines {
            assert!(line.less.right() <= line.value.x && line.value.right() <= line.more.x);
        }
        // Tall enough: a blank line between rows; short: none, all inside.
        for (height, step) in [(40, 2), (24, 1)] {
            let mut s = state(100, height);
            s.screen = Screen::Settings;
            let Layout::Settings(l) = compute(&s) else {
                panic!()
            };
            assert_eq!(l.lines[1].rect.y - l.lines[0].rect.y, step, "{height} rows");
            assert!(l.lines[rows - 1].rect.y < l.notes.y, "{height} rows");
        }
    }
}
