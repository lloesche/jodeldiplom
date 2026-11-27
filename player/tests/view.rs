//! The screens as the outlines of what the window draws from their
//! models: the list pane's (its tabs, the browser's or the queue's rows)
//! and the right column's text (or the summary under the list) on the
//! main screen, the Transition tab's and the Karaoke tab's while they
//! show, the command palette's while it is open, and the bottom bar's
//! where the screen has one; compared with the snapshots in
//! `tests/snapshots/` (set `UPDATE_SNAPSHOTS=1` to rewrite them after an
//! intended change). The spectrum's model in points: its labels and where
//! a tone peaks.

mod common;
#[path = "common/setup.rs"]
mod setup_web;

use std::path::PathBuf;

use jodeldiplom::core::desk::{Channel, StripKind};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{update, Action, DeviceChoice, Prepared, Screen, Side, State};
use jodeldiplom::gui::metrics;
use jodeldiplom::gui::mixer::{self, said};
use jodeldiplom::spectrum::{sine, tap_frames_for, Analyzer, SpectrumFrame};
use jodeldiplom::view;
use jodeldiplom::view::bottom::Bar;
use jodeldiplom::view::command_palette::Palette;
use jodeldiplom::view::help::Page as HelpPage;
use jodeldiplom::view::info::Info;
use jodeldiplom::view::lanes::WavePlot;
use jodeldiplom::view::list::Pane;
use jodeldiplom::view::now::NowPlaying;
use jodeldiplom::view::plot::{LabelTone, Plot};
use jodeldiplom::view::settings::Settings;

fn spectrum() -> SpectrumFrame {
    let mut analyzer = Analyzer::new();
    let mut frames = sine(176_400, 1_000.0, 0.5, tap_frames_for(176_400));
    for (f, s) in frames
        .iter_mut()
        .zip(sine(176_400, 60_000.0, 0.001, tap_frames_for(176_400)))
    {
        f[0] += s[0];
        f[1] += s[1];
    }
    analyzer.analyse(176_400, &frames, 0.0);
    analyzer.frame()
}

/// `state` drawn in a window of `cells` of the old cells.
fn draw(state: &State, cells: (u16, u16)) -> String {
    draw_heard(state, cells, 0.0)
}

/// A cell's size in points, as the window has it with its own font (the
/// tests' fallback font measures otherwise).
fn cell() -> (f32, f32) {
    (8.0, 19.0)
}

/// The bottom bar's outline as the window draws it `width` points wide,
/// its top `heard` points the spectrum: its spectrum's labels where it
/// has one, then the bar's model.
fn bar_outline(state: &State, width: f32, heard: f32) -> String {
    let mut out = String::new();
    if heard > 0.0 {
        let size = iced::Size::new(width, heard);
        let plot = jodeldiplom::gui::widgets::spectrum::plot(size, &spectrum());
        out.push_str(&format!("{plot}\n"));
    }
    out.push_str(&Bar::of(state, width).to_string());
    out
}

/// [`draw`] with `heard` seconds of the playing track heard: in a window
/// of `cells` of the old cells ([`common::window`]) as the window lays it out
/// ([`metrics::main`]).
fn draw_heard(state: &State, cells: (u16, u16), heard: f64) -> String {
    let mut text = String::new();
    let size = common::window(cells.0, cells.1);
    let main = (state.screen == Screen::Main).then(|| metrics::main(size, state.side));
    if let Some(m) = &main {
        text.push_str(&Pane::of(state, m.pane.width).to_string());
        // The right column: a song's preparation while it shows, else the
        // track's text.
        match (&m.right, view::prep::PrepView::of(state)) {
            (Some(_), Some(prep)) => text.push_str(&prep.to_string()),
            (Some(_), None) => text.push_str(&Info::of(state).to_string()),
            (None, _) => {}
        }
        if m.summary.is_some() {
            text.push_str(&Info::of(state).summary());
        }
        if state.side == Side::Transition {
            text.push_str(&view::transition::Tab::of(state, heard).to_string());
        }
        if state.side == Side::Karaoke {
            text.push_str(&view::karaoke::Tab::of(state).to_string());
        }
    }
    if let Some(palette) = Palette::of(state, 8) {
        text.push_str(&palette.to_string());
    }
    if let Some(m) = &main {
        text.push_str(&bar_outline(state, size.width, m.spectrum));
    }
    text
}

fn snapshot(name: &str, text: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        durable::write_atomic(&path, text.as_bytes()).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| {
            panic!(
                "no snapshot {}; run with UPDATE_SNAPSHOTS=1",
                path.display()
            )
        })
        .replace("\r\n", "\n");
    if expected != text {
        let first = expected
            .lines()
            .zip(text.lines())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{name} differs from its snapshot from line {}:\n{text}",
            first + 1
        );
    }
}

fn has(text: &str, parts: &[&str]) {
    for part in parts {
        assert!(text.contains(part), "missing {part:?} in\n{text}");
    }
}

#[test]
fn main_screen() {
    let state = common::playing_state(120, 36);
    let text = draw(&state, (120, 36));
    has(
        &text,
        &[
            "Browser │ Queue (5)",
            "▸ Bonus/",
            "Morning Light",
            "FLAC 44.1",
            "WAV 96",
            "The Ramps",
            "○ PCM",
            "file 44.1 → USB Audio DAC",
            "♩ 128 BPM",
            "1:23",
            "4:05",
            "[icon] pause",
            "[button] Settings",
            "[button] Help",
            "1k",
        ],
    );
    snapshot("main", &text);
}

#[test]
fn queue_side() {
    let mut state = common::playing_state(120, 36);
    // As the app: lyrics are got ready too, the track playing's done.
    state.preparing.push(jodeldiplom::core::Job::Lyrics);
    let playing = state.queue.paths[1].clone();
    state
        .ready
        .insert(playing, vec![jodeldiplom::core::Job::Lyrics]);
    update(&mut state, Action::ShowSide(Side::Queue));
    let text = draw(&state, (120, 36));
    has(&text, &["▶ 2 Morning Light", "playing 2 of 5"]);
    snapshot("queue", &text);
}

/// The bottom bar's outline under a screen the window lays out, as wide
/// as the window of 120 cells.
fn bar_of(state: &State) -> String {
    Bar::of(state, 120.0 * cell().0).to_string()
}

#[test]
fn now_playing() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::NowPlaying;
    let text = NowPlaying::of(&state, 0.0).to_string();
    // The caption: title, artist and album (links), the file's format and
    // rate, the status light's word, the tempo; no path, details,
    // spectrum, bar or buttons, and no hint how to return.
    has(
        &text,
        &[
            "Morning Light",
            "The Ramps · Synthetic Sessions",
            "[chip source] FLAC",
            "[chip] PCM",
            "♩ 128 BPM",
            "Morning Light → GoTrack",
            "The Ramps → GoArtist",
            "Synthetic Sessions → GoAlbum",
        ],
    );
    for gone in ["file 44.1", "20 Hz", "4:05", "settings", "esc", "click"] {
        assert!(!text.contains(gone), "{gone:?} in\n{text}");
    }
    snapshot("nowplaying", &text);
    // Nothing playing: says so, no links.
    state.playing.track = None;
    let text = NowPlaying::of(&state, 0.0).to_string();
    has(&text, &["[title] Nothing playing"]);
    assert!(!text.contains("[links]"), "{text}");
}

#[test]
fn settings() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Settings;
    update(
        &mut state,
        Action::Devices(Ok(vec![DeviceChoice {
            id: Some("usb".into()),
            name: "Speakers (USB Audio DAC)".into(),
            summary: "44.1 48 88.2 96 176.4 192 352.8 384 kHz · S32/S24in32 · 2 ch".into(),
        }])),
    );
    state.settings.picking = true;
    state.config_file = Some("/home/me/.config/jodeldiplom/config.toml".into());
    let text = format!("{}{}", Settings::of(&state), bar_of(&state));
    has(
        &text,
        &[
            "Settings",
            "Output device",
            "system default",
            "● System default",
            "○ Speakers (USB Audio DAC)",
            "Saved to /home/me/.config/jodeldiplom/config.toml",
            "[button] Settings",
        ],
    );
    snapshot("settings", &text);
}

/// The settings' rows as their controls show them: every row in its
/// section, the one selected marked, switches, checks (boxes while the
/// methods vote, radio buttons while one decides) and choices with the one
/// set; the data folder's list open in its place.
#[test]
fn settings_rows_show_their_controls() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Settings;
    state.settings.row = 7;
    let text = Settings::of(&state).to_string();
    has(
        &text,
        &[
            "[section] Output",
            "[section] Tempo and beats",
            "[section] Stems",
            "[section] Next track",
            "[section] Data",
            "[row] Headroom [choice] <none> | 0.5 dB | 1 dB | 2 dB | 3 dB",
            "[row] Tempo and beats [switch on]",
            "  [row] Spectral flux [x]",
            "[row] Stems [switch unusable]",
            "[row selected] Stem switching [choice] <on the next bar> | instantly",
            "[row] Next track (N) [choice] <cut> | mix in",
            "  [row] Its transition [choice] <the tab's> | fade | drums first",
            "[row] Data folder [open]",
        ],
    );
    state.config.beat_voting = false;
    state.config.bpm = false;
    let text = Settings::of(&state).to_string();
    has(
        &text,
        &[
            "[row] Tempo and beats [switch off]",
            "no tempo, no beat pulse, nothing analysed",
            "  [row] Spectral flux ( )",
            "  [row] Beat This! network (●)",
        ],
    );
    let dir = std::path::PathBuf::from("/home/me");
    state.settings.folders = Some(jodeldiplom::core::FolderPick {
        dir: Some(dir.clone()),
        folders: Some(vec![dir.join("Music")]),
        cursor: 2,
    });
    let text = Settings::of(&state).to_string();
    has(
        &text,
        &[
            "[entry] ✓ move the data folder here:",
            "[entry] ↑ ..",
            "[entry selected] ▸ Music",
        ],
    );
    assert!(!text.contains("[row selected]"), "the list has the keys");
}

/// The help on a window of 120 × 50 cells: its first page.
#[test]
fn help() {
    let mut state = common::playing_state(120, 50);
    state.screen = Screen::Help;
    let text = format!("{}{}", HelpPage::of(&state), bar_of(&state));
    has(
        &text,
        &[
            "Keys",
            "Anywhere",
            "space",
            "play / pause",
            "Q / ctrl+c",
            "Browser and queue",
        ],
    );
    snapshot("help", &text);
    // As many lines as the window shows.
    let lines = HelpPage::of(&state).lines.len();
    assert_eq!(lines, usize::from(state.fit.help_rows));
    // Scrolled to the end, the mouse section shows.
    update(&mut state, Action::HelpScroll(1_000));
    has(
        &HelpPage::of(&state).to_string(),
        &["Mouse", "double click"],
    );
}

/// `/` on the help, then "volume": the search, and only the keys it
/// matches under their titles.
#[test]
fn help_searched() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Help;
    update(&mut state, Action::Op(Op::Filter));
    for c in "volume".chars() {
        update(&mut state, Action::Type(c));
    }
    let text = format!("{}{}", HelpPage::of(&state), bar_of(&state));
    has(
        &text,
        &[
            "/ volume▏",
            "shown",
            "Anywhere",
            "alt+↑",
            "device volume a step up",
            "alt+↓",
            "device volume a step down",
            "/ search",
        ],
    );
    for gone in ["play / pause", "Browser and queue", "Mouse"] {
        assert!(!text.contains(gone), "{gone:?} in\n{text}");
    }
    snapshot("help_search", &text);
    // Enter keeps it (no caret); nothing matching says so.
    update(&mut state, Action::Op(Op::Accept));
    let text = HelpPage::of(&state).to_string();
    has(&text, &["/ volume", "device volume a step up"]);
    assert!(!text.contains("/ volume▏"));
    update(&mut state, Action::Op(Op::Filter));
    update(&mut state, Action::Type('#'));
    update(&mut state, Action::Type('#'));
    has(
        &HelpPage::of(&state).to_string(),
        &["/ volume##▏", "0 shown", "No key or text has “volume##”"],
    );
}

#[test]
fn filter_and_palette() {
    let mut state = common::playing_state(120, 36);
    update(&mut state, Action::Op(Op::Filter));
    for c in "light".chars() {
        update(&mut state, Action::Type(c));
    }
    let text = draw(&state, (120, 36));
    has(
        &text,
        &["/ light", "1 shown", "» search all subfolders for “light”"],
    );
    snapshot("filter", &text);
    let mut state = common::playing_state(120, 36);
    update(&mut state, Action::Op(Op::Palette));
    update(&mut state, Action::Type('n'));
    update(&mut state, Action::Type('e'));
    let text = draw(&state, (120, 36));
    has(&text, &[": ne", "Next track", "n"]);
    snapshot("palette", &text);
}

#[test]
fn narrow_and_tiny_screens() {
    let state = common::playing_state(64, 22);
    let text = draw(&state, (64, 22));
    has(&text, &["▶ Morning Light — The Ramps", "○ PCM"]);
    assert!(
        text.contains("[icon] Settings") && !text.contains("[button] Settings"),
        "icons alone when narrow:\n{text}"
    );
    snapshot("narrow", &text);
}

/// The spectrum's model at `width` × `height` points, a 1 kHz tone (and a
/// faint 60 kHz one) heard at 176.4 kHz, labelled as the window does.
fn plot(width: f32, height: f32) -> Plot {
    jodeldiplom::gui::widgets::spectrum::plot(iced::Size::new(width, height), &spectrum())
}

/// The spectrum is labelled round its curve: levels on the left every 40
/// dB on a plot of nine lines, the unit on top; frequencies under it with
/// units, the Nyquist frequency in the curve's colour; and a 1 kHz tone
/// peaks at the "1k" label.
#[test]
fn the_spectrum_is_labelled_and_a_tone_peaks_at_its_label() {
    let (cw, ch) = cell();
    let p = plot(120.0 * cw, 10.0 * ch);
    let texts = |labels: &[jodeldiplom::view::plot::AxisLabel]| -> Vec<String> {
        labels.iter().map(|l| l.text.clone()).collect()
    };
    assert_eq!(texts(&p.db_labels), ["0 dB", "-40", "-80", "-120"]);
    assert_eq!(
        texts(&p.hz_labels),
        ["20 Hz", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k", "50k", "88k"]
    );
    let nyquist = p.hz_labels.last().unwrap();
    assert_eq!(nyquist.tone, LabelTone::Source);
    // The labels are under the curve, the levels left of it.
    for l in &p.hz_labels {
        assert!(l.y > p.area.y + p.area.height, "{l:?}");
    }
    for l in &p.db_labels {
        assert!(l.x + l.width <= p.area.x, "{l:?}");
    }
    // The tone's peak, the curve's highest point, is over its label.
    let curve = p.curve(3.0);
    let peak = curve
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .copied()
        .unwrap();
    let k = p.hz_labels.iter().find(|l| l.text == "1k").unwrap();
    assert!(
        (k.x..k.x + k.width).contains(&peak.0),
        "the peak at {peak:?}, 1k at {k:?}"
    );
    assert!(peak.1 < p.area.y + p.area.height / 2.0, "{peak:?}");
}

/// However wide, the labels fit and never touch: the Nyquist frequency
/// and the decades always there, the levels only where there is room.
#[test]
fn the_axis_labels_fit_and_never_overlap() {
    let (cw, ch) = cell();
    for cols in [40u16, 52, 64, 80, 100, 160, 240] {
        let width = f32::from(cols) * cw;
        let p = plot(width, 10.0 * ch);
        let words: Vec<&str> = p.hz_labels.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(words[0], "20 Hz", "{cols}: {words:?}");
        for word in &words[1..] {
            assert!(
                word.ends_with('k') || word.parse::<u32>().is_ok(),
                "{cols}: {word:?}"
            );
        }
        assert!(words.contains(&"88k"), "{cols}: the Nyquist frequency wins");
        assert!(words.contains(&"1k"), "{cols}: decades win");
        for pair in p.hz_labels.windows(2) {
            assert!(pair[0].x + pair[0].width < pair[1].x, "{cols}: {pair:?}");
        }
        for l in &p.hz_labels {
            assert!(l.x >= 0.0 && l.x + l.width <= width, "{cols}: {l:?}");
        }
        // Level labels only where there is room.
        assert_eq!(
            !p.db_labels.is_empty(),
            width >= jodeldiplom::view::plot::DB_AXIS_MIN_W + 24.0,
            "{cols}"
        );
    }
}

/// Before anything is heard there is nothing to draw: no curve, no grid,
/// no labels.
#[test]
fn an_empty_spectrum_draws_nothing() {
    let p = Plot::new(
        960.0,
        190.0,
        &SpectrumFrame::default(),
        jodeldiplom::gui::widgets::spectrum::TYPE,
    );
    assert!(p.curve(3.0).is_empty());
    assert!(p.db_lines.is_empty() && p.hz_lines.is_empty());
    assert!(p.db_labels.is_empty() && p.hz_labels.is_empty());
}

#[test]
fn the_right_column_keeps_its_width_between_tracks() {
    use jodel_playback::{Event, PlayState};
    let size = common::window(160, 40);
    let column = |state: &State| metrics::main(size, state.side).right.map(|r| r.width);
    let mut state = common::playing_state(160, 40);
    let playing = column(&state);
    assert_eq!(playing, Some(metrics::right_width(1280.0)));
    // A device name as long as they come: the column stays, its text
    // wraps.
    let mut long = state.clone();
    long.playing.track.as_mut().unwrap().device =
        "Digital Output (High Definition Audio Device 2)".into();
    assert_eq!(column(&long), playing);
    // Between two tracks nothing is shown for a moment; the column stays.
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(state.playing.track.is_none());
    assert_eq!(column(&state), playing);
}

/// The beat pulse is lit on each beat where the beat is clear (the bar's
/// first one in the accent), fading over a moment; between beats and
/// while paused it is dim.
#[test]
fn the_beat_pulse_lights_on_each_beat() {
    use jodeldiplom::view::info::{pulse, PULSE_SECONDS};
    let mut state = common::playing_state(160, 40);
    // The fixture's beats: from 0.4 s, 128 a minute, a bar of four.
    let beat = |i: f64| 0.4 + i * 60.0 / 128.0;
    let on = pulse(&state, beat(4.0)).unwrap();
    assert!(on.downbeat && on.lit > 0.99, "{on:?}");
    let next = pulse(&state, beat(5.0) + PULSE_SECONDS / 2.0).unwrap();
    assert!(!next.downbeat && (next.lit - 0.5).abs() < 0.01, "{next:?}");
    assert_eq!(pulse(&state, beat(5.0) + PULSE_SECONDS * 1.5), None);
    state.playing.state = jodel_playback::PlayState::Paused;
    assert_eq!(pulse(&state, beat(4.0)), None);
}

#[test]
fn transition_tab() {
    let state = common::transition_state(120, 40);
    let text = draw(&state, (120, 40));
    assert!(text.contains(" Transition "), "{text}");
    assert!(text.contains("drums") && text.contains("vocals"), "{text}");
    snapshot("transition", &text);
}

/// The Karaoke tab has the voice's lane where the other tabs have the
/// spectrum, the song's words on it at the zoom that leaves none out, each
/// where it is sung: a wider view would leave some out; every word sung in
/// view is on it.
#[test]
fn the_karaoke_tab_shows_the_voice_with_every_word() {
    use jodeldiplom::core::lanes::{self, Block};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    common::sung_every(&mut state, 0.5);
    let m = metrics::main(common::window(120, 40), Side::Karaoke);
    let voice = m.voice.expect("the voice's lane");
    assert_eq!(m.spectrum, 0.0, "no spectrum on the tab");
    assert_eq!(
        voice.y + voice.height + metrics::MARGIN,
        m.bar.y,
        "over the bar"
    );
    let lyrics = lanes::lyrics(&state, Block::Now).unwrap();
    let words = lanes::timed_words(&lyrics);
    let per = lanes::lyrics_per_second(&words, 1).unwrap();
    assert_eq!(lanes::words_left_out(&words, per, 1), 0);
    assert!(
        lanes::words_left_out(&words, per / 1.5, 1) > 0,
        "a wider view leaves some out"
    );
    let heard = 30.0;
    let plot = voice_lane(&state, (120, 40), heard).expect("the voice's lane");
    let letters = (voice.width / cell().0) as u16;
    let scale = plot.text_scale;
    let (start, end) = view::karaoke::voice_window(&state, letters, heard, scale);
    assert_eq!((plot.start, plot.end), (start, end));
    assert!(((end - start) - f64::from(letters) / (per * f64::from(scale))).abs() < 1e-9);
    let text = draw_heard(&state, (120, 40), heard);
    // The lane's first line: every word sung in view (but the last, which
    // may run past its edge) is there, in order.
    let row: String = lane_words(&plot, letters)
        .iter()
        .filter(|l| l.row == 0)
        .map(|l| format!("{} ", l.text))
        .collect();
    let shown: Vec<&str> = row
        .split(|c: char| !c.is_alphabetic() && c != ',')
        .filter(|w| !w.is_empty())
        .collect();
    let sung: Vec<&str> = words
        .iter()
        .filter(|w| w.start.unwrap() >= start && w.start.unwrap() < end - 1.0)
        .map(|w| w.text.as_str())
        .collect();
    // At the window's size of the words: two cells a character.
    assert_eq!(scale, view::karaoke::VOICE_TEXT);
    assert!(sung.len() >= 6, "{sung:?}");
    assert!(
        shown.windows(sung.len()).any(|s| s == sung.as_slice()),
        "{shown:?} against {sung:?}\n{text}"
    );
    snapshot("karaoke_tab", &text);
}
/// The Karaoke tab's Delay row (the delay in force) and sync row (where
/// it comes from), and the sync's panel over the tab: what to do, the
/// clicks as dots (played, answered), then the delay measured and its
/// buttons.
#[test]
fn the_sync_panel_shows_the_clicks_and_the_delay_measured() {
    use jodeldiplom::core::sync::{outcome, Session, CLICKS};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    state.karaoke.mic.name = Some("Headset".into());
    jodeldiplom::core::lag::set_pair(&mut state, "out", "headset");
    let text = draw(&state, (120, 40));
    has(
        &text,
        &["Delay", "150 ms", "Sync", "Measure (s)", "default"],
    );
    // Three clicks played, the first and third answered.
    let mut session = Session::new(("out".into(), "headset".into()));
    session.played = 3;
    session.answers[0] = Some(0.15);
    session.answers[2] = Some(0.16);
    state.karaoke.sync = Some(session);
    let text = draw(&state, (120, 40));
    has(
        &text,
        &[
            "Sync: how late your voice comes",
            "Say “ta” (or clap) with each click.",
            "2 of 3 heard",
            "[button] Cancel (esc)",
            "say “ta” with each click · esc cancel",
        ],
    );
    assert!(!text.contains("Keep (enter)"), "nothing to keep yet");
    snapshot("karaoke_sync", &text);
    // Every click answered: the delay measured.
    let session = state.karaoke.sync.as_mut().unwrap();
    session.played = CLICKS;
    session.answers = vec![Some(0.152); CLICKS];
    session.outcome = Some(outcome(&session.answers));
    let text = draw(&state, (120, 40));
    has(
        &text,
        &[
            "8 of 8 heard",
            "Your delay: 152 ms",
            "[primary] Keep (enter)",
            "[button] Again (r)",
            "enter keep · r again · esc cancel",
        ],
    );
    snapshot("karaoke_sync_done", &text);
}

/// The Transition tab asks for its tracks' lyrics; once timed, the words go
/// along the bottom of the voice's lane, each where it is sung, a long one
/// spread over its syllables.
#[test]
fn the_words_go_on_the_voice_lane_where_they_are_sung() {
    use jodel_lyrics::{Lyrics, Source};
    use jodel_playback::Stem;
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::lanes::Lane;
    use jodeldiplom::core::transition::viewport;
    use jodeldiplom::core::Effect;
    use jodeldiplom::work::lyrics::Answer;
    let mut state = common::transition_state(120, 40);
    state.karaoke.lyrics_wanted.clear();
    let fx = update(&mut state, Action::Tick(1));
    let index = state.playing.track.as_ref().unwrap().index;
    let path = state.queue.paths[index].clone();
    let asked = path.clone();
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::WantLyrics(p) if p.contains(&path))),
        "{fx:?}"
    );
    let (start, end) = viewport::window(&state, Block::Now, 0.0);
    let at = start + (end - start) * 0.25;
    let mut timed = Lyrics::parse_text("Mister sunshine", Source::Lrclib);
    let words = &mut timed.lines[0].words;
    // "Mister" held over a tenth of the span, "sunshine" after it.
    let span = end - start;
    words[0].start = Some(at);
    words[0].end = Some(at + span * 0.1);
    words[1].start = Some(at + span * 0.15);
    words[1].end = Some(at + span * 0.2);
    timed.timing.by = Some(jodel_lyrics::Maker::new("ctc:test", 1));
    timed.timing.detail = jodel_lyrics::Detail::Words;
    update(
        &mut state,
        Action::Lyrics(
            path,
            Answer::Found {
                lyrics: Lyrics::parse_text("Mister sunshine", Source::Lrclib),
                timed: Some(timed),
            },
        ),
    );
    // The words, on the voice's lane only.
    let worded = |state: &State| -> Vec<(Lane, String)> {
        lanes(state, (120, 40))
            .iter()
            .flat_map(|p| {
                lane_words(p, state.fit.lane_letters)
                    .into_iter()
                    .map(|l| (p.lane, l.text))
            })
            .collect()
    };
    let words = worded(&state);
    assert!(
        words
            .iter()
            .all(|(lane, _)| *lane == Lane::Stem(Stem::Vocals)),
        "{words:?}"
    );
    let line: String = words.iter().map(|(_, w)| format!("{w} ")).collect();
    assert!(line.contains("Mis─"), "no words on the lane: {line}");
    assert!(line.contains("─ter") && line.contains("sun"), "{line}");
    // `L`: off (kept in the settings), the lane without words, and no
    // lyrics asked for; again: on.
    use jodeldiplom::core::keymap::Op;
    let fx = update(&mut state, Action::Op(Op::TransitionLyrics));
    assert!(!state.config.transition_lyrics);
    assert!(fx.contains(&Effect::SaveConfig), "{fx:?}");
    assert!(worded(&state).is_empty());
    assert!(jodeldiplom::core::karaoke::lyrics_wanted(&state).is_empty());
    // On again: asked for again (from the cache, at once in the app).
    let fx = update(&mut state, Action::Op(Op::TransitionLyrics));
    assert!(state.config.transition_lyrics);
    assert!(
        fx.iter()
            .any(|e| matches!(e, Effect::WantLyrics(p) if p.contains(&asked))),
        "{fx:?}"
    );
}

#[test]
fn transition_tab_in_a_short_window() {
    // One row a lane, both tracks.
    let state = common::transition_state(100, 26);
    let text = draw(&state, (100, 26));
    assert!(text.contains("Next"), "{text}");
    snapshot("transition_short", &text);
}

#[test]
fn transition_tab_with_a_transition_being_edited() {
    let mut state = common::transition_state(120, 40);
    for o in [
        Op::Accept,
        Op::Down,
        Op::Accept,
        Op::Up,
        Op::TransitionPreset,
    ] {
        update(&mut state, Action::Op(o));
    }
    update(&mut state, Action::Op(Op::TransitionEdit));
    let text = draw(&state, (120, 40));
    assert!(text.contains("transition drums first, varispeed"), "{text}");
    assert!(text.contains("editing now bass at bar 0"), "{text}");
    snapshot("transition_edit", &text);
}

#[test]
fn the_transition_header_shows_the_pairing_and_the_glide() {
    // The next track at 66 BPM: its beat on two of the playing track's
    // (128 BPM), so the tempo glides from 128 to 132.
    let mut state = common::transition_state(190, 40);
    common::next_at(&mut state, 66.0);
    update(&mut state, Action::Op(Op::TransitionPreset));
    let text = draw(&state, (190, 40));
    assert!(
        text.contains("next 2:1 +3.1 % · tempo 128 → 132 across the window"),
        "{text}"
    );
    update(&mut state, Action::Op(Op::TransitionPairing));
    let text = draw(&state, (190, 40));
    assert!(
        text.contains("next 1:1 (set) -48.4 % · tempo 128 → 98 across the window, 66 after it"),
        "{text}"
    );
}

/// The Karaoke tab's voice lane in a window of `cells` of the old cells, with
/// `heard` seconds of the playing track heard, as the window lays it out:
/// its letters and its lines a cell each.
fn voice_lane(state: &State, cells: (u16, u16), heard: f64) -> Option<WavePlot> {
    let voice = metrics::main(common::window(cells.0, cells.1), state.side).voice?;
    let (cw, ch) = cell();
    let (letters, rows) = ((voice.width / cw) as u16, (voice.height / ch) as u16);
    view::karaoke::voice_plot(state, letters, rows, heard)
}

/// The lanes' waveforms a frame of `state` in a window of `cells` hands
/// the window:
/// the Karaoke tab's voice lane, and the Transition tab's blocks', the
/// playing track's and the next one's.
fn lanes(state: &State, cells: (u16, u16)) -> Vec<WavePlot> {
    let mut lanes: Vec<WavePlot> = voice_lane(state, cells, 0.0).into_iter().collect();
    if state.side == Side::Transition {
        use jodeldiplom::core::lanes::Block;
        for block in [Block::Now, Block::Next] {
            if let Some(plots) = view::transition::plots(state, block, 0.0, true) {
                lanes.extend(plots.lanes);
            }
        }
    }
    lanes
}

/// The words the window lays along `plot` (its lyrics, as it places them,
/// in letters: a character [`WavePlot::text_scale`] letters wide) on a
/// lane `width` letters wide, by where they start.
fn lane_words(plot: &WavePlot, width: u16) -> Vec<view::lanes::Label> {
    let Some(lyrics) = plot.lyrics.as_deref() else {
        return Vec::new();
    };
    let words = jodeldiplom::core::lanes::timed_words(lyrics);
    let width = f64::from(width);
    let per = width / (plot.end - plot.start).max(1e-9);
    let lo = (plot.start * per).floor();
    let mut labels = view::lanes::lane_labels(
        &words,
        &|t| (t * per).floor(),
        per,
        f64::from(plot.text_scale.max(1)),
        plot.lines,
        plot.playhead,
        (lo, lo + width),
    );
    labels.sort_by(|a, b| a.at.total_cmp(&b.at));
    labels
}

#[test]
fn stems_turned_off_are_grey_on_the_transition_tab() {
    use jodel_playback::stems::Stem;
    use jodeldiplom::core::desk::StripKind;
    use jodeldiplom::core::lanes::Lane;
    use view::lanes::{dimmed, lane_color, OFF};
    let mut state = common::transition_state(120, 40);
    state.desk.strips[StripKind::for_stem(Stem::Drums).index()].mute = true;
    let lanes = lanes(&state, (120, 40));
    let color = |lane: Lane| {
        lanes
            .iter()
            .find(|p| p.lane == lane)
            .unwrap_or_else(|| panic!("no {lane:?} lane"))
            .color
    };
    assert!(
        [OFF, dimmed(OFF)].contains(&color(Lane::Stem(Stem::Drums))),
        "{:?}",
        color(Lane::Stem(Stem::Drums))
    );
    let bass_color = lane_color(Lane::Stem(Stem::Bass));
    assert!(
        [bass_color, dimmed(bass_color)].contains(&color(Lane::Stem(Stem::Bass))),
        "{:?}",
        color(Lane::Stem(Stem::Bass))
    );
}

#[test]
fn the_fade_curves_go_to_the_lanes_of_a_transition() {
    // The lanes with a fade to draw over their transition.
    let curves = |state: &State| {
        lanes(state, (120, 40))
            .iter()
            .filter(|p| p.transition.is_some() && !p.gain.is_empty())
            .count()
    };
    let mut state = common::transition_state(120, 40);
    assert_eq!(curves(&state), 0, "no transition, no curve");
    // A preset alone only names itself in the header.
    update(&mut state, Action::Op(Op::TransitionPreset));
    assert_eq!(curves(&state), 0, "no cue, no curve");
    // A cue on the next track: its fades in, over the cue.
    update(&mut state, Action::Op(Op::Down));
    update(&mut state, Action::Op(Op::Accept));
    let n = curves(&state);
    assert!(n >= 2, "the curves of the next track's lanes: {n} lanes");
}

/// The whole queue is got ready ahead, from the track playing on, once per
/// change; each track's icons light up one after the other as its work
/// finds each thing.
#[test]
fn the_queue_is_got_ready_ahead_and_its_icons_light_up_in_turn() {
    use jodeldiplom::core::{Effect, Job};
    let mut state = common::playing_state(120, 40);
    // As the app: lyrics are got ready too.
    state.preparing.push(Job::Lyrics);
    let paths = state.queue.paths.clone();
    let current = state.playing.track.as_ref().unwrap().index;
    state.prepared = Prepared::default();
    let fx = update(&mut state, Action::Tick(1));
    let order: Vec<PathBuf> = paths[current..]
        .iter()
        .chain(&paths[..current])
        .cloned()
        .collect();
    let prepared = fx.iter().find_map(|e| match e {
        Effect::Prepare(p, recipe) => Some((p.clone(), recipe.is_some())),
        _ => None,
    });
    assert_eq!(prepared, Some((order, true)));
    // Nothing changed: not again.
    let fx = update(&mut state, Action::Tick(2));
    assert!(!fx.iter().any(|e| matches!(e, Effect::Prepare(..))));
    // A worker at work on a track, then done.
    let busy = paths[current + 1].clone();
    for job in [Job::Rhythm, Job::Lyrics] {
        update(
            &mut state,
            Action::Working {
                path: busy.clone(),
                job,
                on: true,
            },
        );
    }
    assert_eq!(state.working[&busy].len(), 2);
    state.side = Side::Queue;
    let title = state.details[&busy].title.clone().unwrap();
    let row = |state: &State| {
        let text = draw(state, (120, 40));
        text.lines()
            .find(|l| l.contains(&format!("{} {title}", current + 2)))
            .map(str::to_owned)
            .unwrap_or_default()
    };
    assert!(row(&state).contains("♯ ▥ ♩ ¶"), "{}", row(&state));
    for job in [Job::Rhythm, Job::Lyrics] {
        update(
            &mut state,
            Action::Working {
                path: busy.clone(),
                job,
                on: false,
            },
        );
    }
    assert!(state.working.is_empty());
    // Grey until the track has each thing, lit once it has, in the order
    // the work finds them (the key, the stems, the tempo, the lyrics).
    use jodeldiplom::core::{Facts, LyricsFacts, RhythmFacts};
    use jodeldiplom::view::list::{Body, Fact};
    const ACCENT: bool = true;
    const DIM: bool = false;
    let colors = |state: &State| -> Vec<bool> {
        let Body::Queue(q) = Pane::of(state, 960.0).body else {
            panic!("the queue");
        };
        let row = q.rows.iter().find(|r| r.index == current + 1).unwrap();
        let facts: Vec<Fact> = row.facts.iter().map(|s| s.fact).collect();
        assert_eq!(facts, [Fact::Key, Fact::Stems, Fact::Tempo, Fact::Lyrics]);
        row.facts.iter().map(|s| s.lit).collect()
    };
    assert_eq!(colors(&state), [DIM, DIM, DIM, DIM]);
    // One after the other as the analysis goes: the key once decoded, the
    // stems once separated, the tempo with the whole analysis.
    use jodeldiplom::core::Heard;
    update(
        &mut state,
        Action::Heard {
            path: busy.clone(),
            part: Heard::Key(Some(jodel_rhythm::key::Key::on(9, false))),
        },
    );
    assert_eq!(colors(&state), [ACCENT, DIM, DIM, DIM]);
    update(
        &mut state,
        Action::Heard {
            path: busy.clone(),
            part: Heard::Stems,
        },
    );
    assert_eq!(colors(&state), [ACCENT, ACCENT, DIM, DIM]);
    update(
        &mut state,
        Action::Ready {
            path: busy.clone(),
            job: Job::Rhythm,
            facts: Some(Facts::Rhythm(RhythmFacts {
                bpm: Some(124.4),
                key: Some("10B".into()),
                stems: true,
            })),
        },
    );
    assert_eq!(colors(&state), [ACCENT, ACCENT, ACCENT, DIM]);
    update(
        &mut state,
        Action::Ready {
            path: busy.clone(),
            job: Job::Lyrics,
            facts: Some(Facts::Lyrics(LyricsFacts::Timed)),
        },
    );
    assert_eq!(colors(&state), [ACCENT; 4]);
    // None found: grey again.
    update(
        &mut state,
        Action::Ready {
            path: busy.clone(),
            job: Job::Lyrics,
            facts: Some(Facts::Lyrics(LyricsFacts::NotFound)),
        },
    );
    assert_eq!(colors(&state)[3], DIM);
}

/// A queue row's icons are all lit only when nothing is left to do for
/// its track: with every step done they are, and with any one step still
/// to come, or under way again, they are not (nor is the track ready).
#[test]
fn a_queue_rows_icons_are_all_lit_only_when_nothing_is_left_for_it() {
    use jodeldiplom::core::karaoke::{ModelState, SongLyrics};
    use jodeldiplom::core::{readiness, Facts, Heard, Job, LyricsFacts, Readiness, RhythmFacts};
    use jodeldiplom::view::list::Body;
    use std::sync::Arc;
    let mut base = common::playing_state(120, 40);
    base.preparing.extend([Job::Lyrics, Job::Cover]);
    base.side = Side::Queue;
    base.karaoke.model = ModelState::Ready;
    // Got ready with the tempo on, as the app.
    base.prepared = Prepared::default();
    update(&mut base, Action::Tick(1));
    assert!(base.prepared.recipe.is_some());
    let current = base.playing.track.as_ref().unwrap().index;
    let path = base.queue.paths[current + 1].clone();
    let ready = |job| match job {
        Job::Rhythm => Facts::Rhythm(RhythmFacts {
            bpm: Some(124.4),
            key: Some("10B".into()),
            stems: true,
        }),
        Job::Lyrics => Facts::Lyrics(LyricsFacts::Timed),
        Job::Cover => Facts::Cover(true),
    };
    // Every step done but `skip`'s.
    let done = |skip: Option<Job>| {
        let mut state = base.clone();
        for job in [Job::Rhythm, Job::Lyrics, Job::Cover] {
            if Some(job) == skip {
                continue;
            }
            update(
                &mut state,
                Action::Ready {
                    path: path.clone(),
                    job,
                    facts: Some(ready(job)),
                },
            );
        }
        state
    };
    let lit = |state: &State| {
        let Body::Queue(q) = Pane::of(state, 960.0).body else {
            panic!("the queue");
        };
        let row = q.rows.iter().find(|r| r.index == current + 1).unwrap();
        assert_eq!(row.facts.len(), 5);
        row.facts.iter().all(|s| s.lit)
    };
    let state = done(None);
    assert!(lit(&state));
    assert_eq!(readiness(&state, &path), Readiness::Done);
    for job in [Job::Rhythm, Job::Lyrics, Job::Cover] {
        // Still to come, what it found so far known.
        let mut state = done(Some(job));
        match job {
            Job::Rhythm => {
                for part in [
                    Heard::Key(Some(jodel_rhythm::key::Key::on(9, false))),
                    Heard::Stems,
                ] {
                    update(
                        &mut state,
                        Action::Heard {
                            path: path.clone(),
                            part,
                        },
                    );
                }
            }
            // Wanted, none found and the voice still to be heard (or being
            // heard); then found and shown, their words still to be timed.
            Job::Lyrics => {
                state
                    .karaoke
                    .lyrics
                    .insert(path.clone(), SongLyrics::Pending);
                assert!(!lit(&state), "to be heard");
                assert_eq!(readiness(&state, &path), Readiness::Waiting);
                let text = Arc::new(jodel_lyrics::Lyrics::parse(
                    "One line\n",
                    jodel_lyrics::Source::Lrclib,
                ));
                state.karaoke.lyrics.insert(
                    path.clone(),
                    SongLyrics::Found {
                        lyrics: Arc::clone(&text),
                        shown: text,
                    },
                );
            }
            Job::Cover => {}
        }
        assert!(!lit(&state), "{job:?} to come");
        assert_eq!(readiness(&state, &path), Readiness::Waiting, "{job:?}");
        // Done, and under way again.
        let mut state = done(None);
        update(
            &mut state,
            Action::Working {
                path: path.clone(),
                job,
                on: true,
            },
        );
        assert!(!lit(&state), "{job:?} under way");
        assert_eq!(readiness(&state, &path), Readiness::Working, "{job:?}");
    }
}

/// The Mixer tab: the window's desk, which lays itself out in points with
/// the core's rule (the keys go through the same strips), its keys under
/// it: every strip at the old 120-column width, the microphone gone at
/// the old 80. The snapshots are the desk's outline: the strips shown,
/// where, and what each says ([`said`]), and its keys.
#[test]
fn the_mixer_tab_is_a_desk() {
    use iced::Size;
    let state = common::desk_state(120, 36);
    let text = draw(&state, (120, 36));
    has(&text, &["[shown] Mixer"]);
    let wide = metrics::main(common::window(120, 36), Side::Mixer)
        .pane
        .width;
    assert_eq!(wide, 944.0);
    assert_eq!(state.fit.desk_width, wide);
    let outline = mixer::outline(&state, Size::new(wide, 560.0));
    snapshot("mixer", &outline);
    has(
        &outline,
        &[
            "[bracket] ● STEMS heard",
            "[name selected] MIC",
            "[button fail] M",
            "[keys] ← → strip · ↑ ↓ fader · home 0 dB · m mute · s solo · e echo",
        ],
    );
    let said = |channel| said::text(&state, channel).join(" | ");
    assert_eq!(said::stems(&state), "● STEMS heard");
    let player = said(Channel::Strip(StripKind::Player));
    has(&player, &["96k 24b", "● LEAD"]);
    assert!(said(Channel::Strip(StripKind::Vocals)).contains("muted"));
    let mic = said(Channel::Mic);
    has(&mic, &["● −31 dB 42ms", "fill 10ms xrun 0", "PITCH"]);
    let main = said(Channel::Main);
    has(
        &main,
        &[
            "WASAPI",
            "96 kHz · 24 bit · 2 ch",
            "LIMIT",
            "12:04.3",
            "VOLUME ▮▮▮▮▮▮ 0.0 dB",
        ],
    );
    let shown = |w| mixer::layout::compute(Size::new(w, 560.0), true).channels();
    // The strips the arrow keys go through, from the first.
    let walked = |state: &State| {
        let mut s = state.clone();
        s.desk.selected = Channel::Strip(StripKind::Player);
        let mut seen = vec![s.desk.selected];
        for _ in 0..Channel::ALL.len() {
            update(&mut s, Action::Op(Op::DeskNextStrip));
            seen.push(s.desk.selected);
        }
        seen.dedup();
        seen
    };
    assert_eq!(shown(wide), Channel::ALL, "every strip at 120 columns");
    assert_eq!(walked(&state), Channel::ALL);
    // An 80-column window: the microphone goes first.
    let narrow = metrics::main(common::window(80, 24), Side::Mixer)
        .pane
        .width;
    let state = common::desk_state(80, 24);
    snapshot(
        "mixer_small",
        &mixer::outline(&state, Size::new(narrow, 360.0)),
    );
    let strips = shown(narrow);
    assert!(strips.contains(&Channel::Strip(StripKind::Preview)));
    assert!(strips.contains(&Channel::Strip(StripKind::Vocals)));
    assert!(!strips.contains(&Channel::Mic));
    assert_eq!(walked(&state), strips);
}

/// The setup screen as it reads ([`view::setup::Screen`]): under way, the
/// step's share, why and what it has done under its line; stopped, what
/// went wrong, what to do, Try again and Quit.
#[test]
fn setup_screen() {
    use jodeldiplom::view::setup::Screen;
    use setup_web::{start, when, Web};
    let dir = common::temp_dir("setup-screen");
    let stalls = Web::Stalls {
        share: 0.43,
        after: std::time::Duration::ZERO,
    };
    let (setup, _) = start(stalls, &dir.join("stalls"));
    let status = when(&setup, |s| s.current == Some(1) && s.done > 0);
    let text = Screen::of(&status).to_string();
    has(
        &text,
        &[
            "[title] Getting the player ready",
            "[done] Downloading the word-timing model",
            "[working] Downloading the speech recogniser (",
            "  [bar accent] 43 %",
            "  [dim] 807 MB of 1.88 GB",
            "[to do] Packing the speech recogniser",
            "[folded to do] ",
            "[label] All of it: ",
            "[button] Quit (Q)",
        ],
    );
    assert!(!text.contains("Try again"));
    snapshot("setup", &text);
    let (setup, _) = start(Web::Offline, &dir.join("offline"));
    let status = when(&setup, |s| s.failure.is_some());
    let text = Screen::of(&status).to_string();
    has(
        &text,
        &[
            "[failed] Downloading the speech recogniser (",
            "[fail] Downloading the speech recogniser failed: the connection timed out",
            "[text] Check that this computer is online",
            "[primary] Try again (Enter)",
            "[button] Quit (Q)",
        ],
    );
    assert!(!text.contains("[bar accent]"), "no step under way:\n{text}");
    snapshot("setup_failed", &text);
    // No steps yet: nothing shown, nothing folded.
    let empty = Screen::of(&jodeldiplom::setup::Status::default());
    assert!(empty.steps.is_empty());
    assert_eq!((empty.folded_done, empty.folded_to_come), (0, 0));
    let _ = std::fs::remove_dir_all(&dir);
}
