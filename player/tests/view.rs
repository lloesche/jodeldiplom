//! The screens drawn into a ratatui test buffer, compared with the
//! snapshots in `tests/snapshots/` (set `UPDATE_SNAPSHOTS=1` to rewrite
//! them after an intended change).

mod common;

use std::path::PathBuf;

use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{update, Action, DeviceChoice, Screen, Side, State};
use jodeldiplom::spectrum::{sine, tap_frames_for, Analyzer, SpectrumFrame};
use jodeldiplom::tui::{render_text, TermPictures};
use jodeldiplom::view::{self, CellPictures, Links, Scene};
use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;

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

fn draw(state: &State) -> String {
    draw_heard(state, 0.0)
}

/// [`draw`] with `heard` seconds of the playing track heard.
fn draw_heard(state: &State, heard: f64) -> String {
    let (cols, rows) = state.size;
    let mut terminal = Terminal::new(TestBackend::new(cols, rows)).unwrap();
    let spectrum = spectrum();
    let scene = Scene {
        state,
        spectrum: &spectrum,
        background: Color::Reset,
        links: Links::Underlined,
        heard,
    };
    terminal
        .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
        .unwrap();
    view::buffer_text(terminal.backend().buffer())
}

fn snapshot(name: &str, text: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
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
    let text = draw(&state);
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
            "file 44.1 → USB Audio DAC (exclusive), bit-perfect",
            "♩ 128 BPM",
            "1:23",
            "4:05",
            "≡ settings",
            "? help",
            "1k",
        ],
    );
    snapshot("main", &text);
}

#[test]
fn queue_side() {
    let mut state = common::playing_state(120, 36);
    update(&mut state, Action::ShowSide(Side::Queue));
    let text = draw(&state);
    has(&text, &["▶ 2 Morning Light", "playing 2 of 5"]);
    snapshot("queue", &text);
}

#[test]
fn now_playing() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::NowPlaying;
    let text = draw(&state);
    // The cover (a placeholder here) and the caption under it; no path,
    // details, spectrum, bar or buttons, and no hint how to return.
    has(
        &text,
        &[
            "♪",
            "Morning Light",
            "The Ramps · Synthetic Sessions",
            "○ PCM",
            "♩ 128 BPM",
        ],
    );
    for gone in [
        "file 44.1",
        "FLAC",
        "20 Hz",
        "4:05",
        "settings",
        "esc",
        "click",
    ] {
        assert!(!text.contains(gone), "{gone:?} in\n{text}");
    }
    let lines: Vec<&str> = text.lines().collect();
    let title = lines
        .iter()
        .position(|l| l.contains("Morning Light"))
        .unwrap();
    // Title, artist and album, status, tempo: the caption ends a row
    // above the bottom.
    assert_eq!(title, 36 - 5, "the caption ends a row above the bottom");
    assert!(lines[title - 2].contains('╰'), "a gap under the cover");
    snapshot("nowplaying", &text);
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
    let text = draw(&state);
    has(
        &text,
        &[
            "Settings",
            "Output device",
            "system default",
            "Rate cap",
            "none",
            "● System default",
            "○ Speakers (USB Audio DAC)",
            "Saved to /home/me/.config/jodeldiplom/config.toml",
        ],
    );
    snapshot("settings", &text);
}

/// A rate cap the device cannot satisfy, as the settings show it.
#[test]
fn settings_show_the_cap_in_force() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Settings;
    state.config.max_rate = Some(44_100);
    update(
        &mut state,
        Action::DeviceRates(Some(vec![48_000, 96_000, 192_000])),
    );
    let text = draw(&state);
    has(&text, &["48 kHz (saved: 44.1 kHz, below this device)"]);
}

#[test]
fn help() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Help;
    let text = draw(&state);
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
    // Scrolled to the end, the mouse section shows.
    update(&mut state, Action::HelpScroll(1_000));
    has(&draw(&state), &["Mouse", "double click"]);
}

/// `/` on the help, then "volume": the search line as the browser's
/// filter line, and only the keys it matches under their titles.
#[test]
fn help_searched() {
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Help;
    update(&mut state, Action::Op(Op::Filter));
    for c in "volume".chars() {
        update(&mut state, Action::Type(c));
    }
    let text = draw(&state);
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
    // Enter keeps it (no cursor); nothing matching says so.
    update(&mut state, Action::Op(Op::Accept));
    has(&draw(&state), &["/ volume", "device volume a step up"]);
    assert!(!draw(&state).contains("/ volume▏"));
    update(&mut state, Action::Op(Op::Filter));
    update(&mut state, Action::Type('#'));
    update(&mut state, Action::Type('#'));
    has(
        &draw(&state),
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
    let text = draw(&state);
    has(
        &text,
        &["/ light", "1 shown", "» search all subfolders for “light”"],
    );
    snapshot("filter", &text);
    let mut state = common::playing_state(120, 36);
    update(&mut state, Action::Op(Op::Palette));
    update(&mut state, Action::Type('n'));
    update(&mut state, Action::Type('e'));
    let text = draw(&state);
    has(&text, &[": ne", "Next track", "n"]);
    snapshot("palette", &text);
}

#[test]
fn narrow_and_tiny_screens() {
    let state = common::playing_state(64, 22);
    let text = draw(&state);
    has(&text, &["▶ Morning Light — The Ramps", "○ PCM"]);
    assert!(!text.contains("≡ settings"), "short buttons when narrow");
    snapshot("narrow", &text);
    let state = common::playing_state(30, 10);
    has(&draw(&state), &["Make the window larger"]);
}

#[test]
fn the_terminal_draws_covers_in_half_blocks() {
    let state = common::playing_state(120, 36);
    let mut pictures = TermPictures::halfblocks();
    let mut app_art = jodeldiplom::art::Art {
        id: 1,
        image: std::sync::Arc::new(jodeldiplom::art::sample_cover(128)),
    };
    pictures.set_art(Some(&app_art));
    let text = render_text(&state, &spectrum(), &mut pictures, 120, 36);
    let blocks = text.matches('▀').count();
    assert!(blocks > 200, "{blocks} half blocks");
    // The cover starts on the lower half of the first row, under the list
    // pane's top border line: lower half blocks there, at the left.
    let first: String = text
        .lines()
        .next()
        .unwrap()
        .chars()
        .skip_while(|&c| c != '╮')
        .skip(1)
        .collect();
    assert!(first.starts_with("▄▄▄▄▄▄▄▄"), "{first}");
    assert!(!first.contains('▀'), "{first}");
    // A new cover replaces the old one.
    app_art.id = 2;
    pictures.set_art(Some(&app_art));
    let again = render_text(&state, &spectrum(), &mut pictures, 120, 36);
    assert_eq!(again.matches('▀').count(), blocks);
    // Without a cover, the placeholder.
    pictures.set_art(None);
    has(
        &render_text(&state, &spectrum(), &mut pictures, 120, 36),
        &["♪"],
    );
}

/// The spectrum's rows on the main screen (the plot and the frequency
/// labels, above the progress bar and the transport).
fn spectrum_rows(text: &str, plot_rows: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let end = lines.len() - 2;
    lines[end - plot_rows - 1..end]
        .iter()
        .map(|l| (*l).to_owned())
        .collect()
}

fn is_braille(c: char) -> bool {
    ('\u{2800}'..='\u{28ff}').contains(&c)
}

#[test]
fn the_spectrum_is_a_dotted_curve_with_labelled_axes() {
    let state = common::playing_state(120, 36);
    let text = draw(&state);
    let rows = spectrum_rows(&text, 9);
    // Levels on the left, every 40 dB on nine rows, the unit on top.
    assert!(rows[0].starts_with(" 0 dB "), "{:?}", rows[0]);
    assert!(rows[2].starts_with("  -40 "), "{:?}", rows[2]);
    assert!(rows[4].starts_with("  -80 "), "{:?}", rows[4]);
    assert!(rows[6].starts_with(" -120 "), "{:?}", rows[6]);
    // Frequencies under it, with units.
    let labels = &rows[9];
    let words: Vec<&str> = labels.split_whitespace().collect();
    assert_eq!(
        words,
        ["20", "Hz", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k", "50k", "88k"],
        "{labels}"
    );
    // The plot is braille only: no block characters.
    for row in &rows[..9] {
        let plot: String = row.chars().skip(6).collect();
        assert!(plot.chars().all(|c| c == ' ' || is_braille(c)), "{row}");
    }
    assert!(!text.contains('█'));
    // The tone at 1 kHz (-6 dB) has dots in the top row at its column,
    // more than the grid's.
    let column = labels.find("1k").unwrap();
    let top: Vec<char> = rows[0].chars().skip(column - 2).take(5).collect();
    let dots = |c: &char| (*c as u32).saturating_sub(0x2800).count_ones();
    assert!(top.iter().map(dots).max().unwrap() >= 2, "{top:?}");
}

#[test]
fn the_axis_labels_fit_and_never_overlap() {
    for cols in [40u16, 52, 64, 80, 100, 160, 240] {
        let state = common::playing_state(cols, 36);
        let text = draw(&state);
        let rows = spectrum_rows(&text, 9);
        let labels = &rows[9];
        assert!(labels.contains("20 Hz"), "{cols}: {labels}");
        let words: Vec<&str> = labels.split_whitespace().collect();
        for word in &words[2..] {
            assert!(
                word.ends_with('k') || word.parse::<u32>().is_ok(),
                "{cols}: {word:?} in {labels}"
            );
        }
        assert!(words.contains(&"88k"), "{cols}: the Nyquist edge wins");
        assert!(words.contains(&"1k"), "{cols}: decades win");
        // Level labels only where there is room.
        assert_eq!(rows[0].contains("0 dB"), cols >= 42, "{cols}: {}", rows[0]);
    }
}

#[test]
fn an_empty_spectrum_draws_nothing() {
    let state = common::playing_state(120, 36);
    let (cols, rows) = state.size;
    let mut terminal = Terminal::new(TestBackend::new(cols, rows)).unwrap();
    let spectrum = SpectrumFrame::default();
    let scene = Scene {
        state: &state,
        spectrum: &spectrum,
        background: Color::Reset,
        links: Links::Underlined,
        heard: 0.0,
    };
    terminal
        .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
        .unwrap();
    let text = view::buffer_text(terminal.backend().buffer());
    for row in spectrum_rows(&text, 9) {
        assert!(row.trim().is_empty(), "{row:?}");
    }
}

#[test]
fn the_right_column_keeps_its_width_between_tracks() {
    use jodel_playback::{Event, PlayState};
    use jodeldiplom::core::layout::right_width_for;
    let mut state = common::playing_state(160, 40);
    update(&mut state, Action::Tick(1));
    let playing = right_width_for(&state, 160);
    // Between two tracks nothing is shown for a moment; the column stays.
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    assert!(state.playing.track.is_none());
    assert_eq!(right_width_for(&state, 160), playing);
    // A new window size measures again.
    update(&mut state, Action::Resize(150, 40));
    assert!(right_width_for(&state, 150) <= playing);
}

#[test]
fn a_frame_of_another_size_than_the_state_is_drawn_blank() {
    // The terminal was resized and the state has not heard of it yet: the
    // layout (for 120 x 36) would not fit, so nothing is drawn until it has.
    let spectrum = spectrum();
    for screen in [Screen::Main, Screen::NowPlaying, Screen::Settings] {
        let mut state = common::playing_state(120, 36);
        state.screen = screen;
        for (cols, rows) in [(118, 35), (60, 20), (160, 50)] {
            let mut terminal = Terminal::new(TestBackend::new(cols, rows)).unwrap();
            let scene = Scene {
                state: &state,
                spectrum: &spectrum,
                background: Color::Reset,
                links: Links::Underlined,
                heard: 0.0,
            };
            terminal
                .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
                .unwrap();
            let text = view::buffer_text(terminal.backend().buffer());
            assert!(
                text.trim().is_empty(),
                "{screen:?} at {cols}x{rows}:\n{text}"
            );
        }
    }
}

#[test]
fn transition_tab() {
    let state = common::transition_state(120, 40);
    let text = draw(&state);
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
    use jodeldiplom::core::layout::{self, Layout};
    let mut state = common::transition_state(120, 40);
    state.side = Side::Karaoke;
    common::sung_every(&mut state, 0.5);
    let Layout::Main(m) = layout::compute(&state) else {
        panic!()
    };
    let voice = m.karaoke.as_ref().unwrap().voice.expect("the voice's lane");
    assert!(m.bottom.spectrum.is_none(), "no spectrum on the tab");
    assert_eq!(voice.bottom(), m.bottom.elapsed.y, "over the progress bar");
    let lyrics = lanes::lyrics(&state, Block::Now).unwrap();
    let words = lanes::timed_words(&lyrics);
    let per = lanes::lyrics_per_second(&words, 1).unwrap();
    assert_eq!(lanes::words_left_out(&words, per, 1), 0);
    assert!(
        lanes::words_left_out(&words, per / 1.5, 1) > 0,
        "a wider view leaves some out"
    );
    let heard = 30.0;
    let (start, end) = view::karaoke::voice_window(&state, voice.width, heard, 1);
    assert!(((end - start) - f64::from(voice.width) / per).abs() < 1e-9);
    let text = draw_heard(&state, heard);
    // The lane's middle row: every word sung in view (but the last, which
    // may run past its edge) is there, in order.
    let row = text
        .lines()
        .nth(usize::from(voice.y + (voice.height - 1) / 2))
        .unwrap();
    let shown: Vec<&str> = row
        .split(|c: char| !c.is_alphabetic() && c != ',')
        .filter(|w| !w.is_empty())
        .collect();
    let sung: Vec<&str> = words
        .iter()
        .filter(|w| w.start.unwrap() >= start && w.start.unwrap() < end - 1.0)
        .map(|w| w.text.as_str())
        .collect();
    assert!(sung.len() >= 10, "{sung:?}");
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
    let text = draw(&state);
    has(
        &text,
        &["Delay", "150 ms", "Sync  (s)", " sync ", "default"],
    );
    // Three clicks played, the first and third answered.
    let mut session = Session::new(("out".into(), "headset".into()));
    session.played = 3;
    session.answers[0] = Some(0.15);
    session.answers[2] = Some(0.16);
    state.karaoke.sync = Some(session);
    let text = draw(&state);
    has(
        &text,
        &[
            "Sync: how late your voice comes",
            "Say “ta” (or clap) with each click.",
            "2 of 3 heard",
            " cancel (esc) ",
            "say “ta” with each click · esc cancel",
        ],
    );
    assert!(!text.contains("keep"), "nothing to keep yet");
    snapshot("karaoke_sync", &text);
    // Every click answered: the delay measured.
    let session = state.karaoke.sync.as_mut().unwrap();
    session.played = CLICKS;
    session.answers = vec![Some(0.152); CLICKS];
    session.outcome = Some(outcome(&session.answers));
    let text = draw(&state);
    has(
        &text,
        &[
            "8 of 8 heard",
            "Your delay: 152 ms",
            " keep (enter) ",
            " again (r) ",
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
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::viewport;
    use jodeldiplom::core::Effect;
    use jodeldiplom::lyrics::Answer;
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
    let text = draw(&state);
    let line = text
        .lines()
        .find(|l| l.contains("Mis─"))
        .unwrap_or_else(|| panic!("no words on the lane:\n{text}"));
    assert!(line.contains("─ter") && line.contains("sun"), "{line}");
    // On the voice's lane: the line is one of its rows.
    let rows: Vec<&str> = text.lines().collect();
    let k = rows.iter().position(|l| *l == line).unwrap();
    assert!(
        rows[..=k]
            .iter()
            .rev()
            .take(6)
            .any(|l| l.contains("vocals")),
        "{text}"
    );
    // `L`: off (kept in the settings), the lane without words, and no
    // lyrics asked for; again: on.
    use jodeldiplom::core::keymap::Op;
    let fx = update(&mut state, Action::Op(Op::TransitionLyrics));
    assert!(!state.config.transition_lyrics);
    assert!(fx.contains(&Effect::SaveConfig), "{fx:?}");
    assert!(!draw(&state).contains("Mis─"));
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
    let text = draw(&state);
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
    let text = draw(&state);
    assert!(
        text.contains("transition: drums first, varispeed"),
        "{text}"
    );
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
    let text = draw(&state);
    assert!(
        text.contains("next 2:1 +3.1 % · tempo 128 → 132 across the window"),
        "{text}"
    );
    update(&mut state, Action::Op(Op::TransitionPairing));
    let text = draw(&state);
    assert!(
        text.contains("next 1:1 (set) -48.4 % · tempo 128 → 98 across the window, 66 after it"),
        "{text}"
    );
}

#[test]
fn stems_turned_off_are_grey_on_the_transition_tab() {
    use jodel_playback::stems::Stem;
    use jodeldiplom::core::layout::Lane;
    use view::lanes::{dimmed, lane_color, OFF};
    let mut state = common::transition_state(120, 40);
    state.stems = state.stems.toggle(Stem::Drums);
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    let spectrum = spectrum();
    let scene = Scene {
        state: &state,
        spectrum: &spectrum,
        background: Color::Reset,
        links: Links::Underlined,
        heard: 0.0,
    };
    terminal
        .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
        .unwrap();
    let buf = terminal.backend().buffer();
    let row_of = |label: &str| {
        (0..40)
            .find(|&y| {
                let text: String = (1..8).map(|x| buf[(x, y)].symbol().to_owned()).collect();
                text.trim() == label
            })
            .unwrap_or_else(|| panic!("no {label} lane"))
    };
    let (drums, bass) = (row_of("drums"), row_of("bass"));
    let fg = |y: u16| buf[(60, y)].fg;
    assert!([OFF, dimmed(OFF)].contains(&fg(drums)), "{:?}", fg(drums));
    let bass_color = lane_color(Lane::Bass);
    assert!(
        [bass_color, dimmed(bass_color)].contains(&fg(bass)),
        "{:?}",
        fg(bass)
    );
}

#[test]
fn the_fade_curves_are_drawn_in_the_terminal_lanes() {
    use view::lanes::CURVE;
    let curve_cells = |state: &State| {
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let spectrum = spectrum();
        let scene = Scene {
            state,
            spectrum: &spectrum,
            background: Color::Reset,
            links: Links::Underlined,
            heard: 0.0,
        };
        terminal
            .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
            .unwrap();
        let buf = terminal.backend().buffer();
        buf.content().iter().filter(|c| c.fg == CURVE).count()
    };
    let mut state = common::transition_state(120, 40);
    assert_eq!(curve_cells(&state), 0, "no transition, no curve");
    // A preset alone only names itself in the header.
    update(&mut state, Action::Op(Op::TransitionPreset));
    assert_eq!(curve_cells(&state), 0, "no cue, no curve");
    // A cue on the next track: its fades in, over the cue.
    update(&mut state, Action::Op(Op::Down));
    update(&mut state, Action::Op(Op::Accept));
    let n = curve_cells(&state);
    assert!(n > 40, "the curves of the next track's lanes: {n} cells");
}

/// The whole queue is got ready ahead, from the track playing on, once per
/// change; each track's icons light up one after the other as its work
/// finds each thing.
#[test]
fn the_queue_is_got_ready_ahead_and_its_icons_light_up_in_turn() {
    use jodeldiplom::core::{Effect, Job};
    let mut state = common::playing_state(120, 40);
    let paths = state.queue.paths.clone();
    let current = state.playing.track.as_ref().unwrap().index;
    state.prepared = (Vec::new(), None);
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
        let text = draw(state);
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
    // Grey until the track has each thing, lit once it has, left of its
    // format.
    use jodeldiplom::core::{Facts, LyricsFacts, RhythmFacts};
    use jodeldiplom::view::colors::{ACCENT, DIM};
    let icons = |state: &State| -> Vec<(String, Color)> {
        let (cols, rows) = state.size;
        let mut terminal = Terminal::new(TestBackend::new(cols, rows)).unwrap();
        let spectrum = spectrum();
        let scene = Scene {
            state,
            spectrum: &spectrum,
            background: Color::Reset,
            links: Links::Underlined,
            heard: 0.0,
        };
        terminal
            .draw(|frame| view::draw(frame, &scene, &mut CellPictures))
            .unwrap();
        let buf = terminal.backend().buffer();
        let text = view::buffer_text(buf);
        let y = text
            .lines()
            .position(|l| l.contains(&format!("{} {title}", current + 2)))
            .unwrap() as u16;
        let line: Vec<&str> = (0..cols).map(|x| buf[(x, y)].symbol()).collect();
        let chip = state.details[&busy]
            .chip
            .split(' ')
            .next()
            .unwrap()
            .to_owned();
        let chip_x = (0..cols as usize)
            .find(|&x| line[x..].concat().starts_with(&chip))
            .unwrap();
        ["♯", "▥", "♩", "¶"]
            .iter()
            .map(|icon| {
                let x = (0..chip_x).rev().find(|&x| line[x] == *icon).expect(icon);
                ((*icon).to_owned(), buf[(x as u16, y)].fg)
            })
            .collect()
    };
    let colors =
        |state: &State| -> Vec<Color> { icons(state).into_iter().map(|(_, c)| c).collect() };
    assert_eq!(colors(&state), [DIM, DIM, DIM, DIM]);
    // One after the other as the analysis goes: the key once decoded, the
    // stems once separated, the tempo with the whole analysis.
    use jodeldiplom::core::Heard;
    update(
        &mut state,
        Action::Heard {
            path: busy.clone(),
            part: Heard::Key(true),
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

#[test]
fn the_mix_tab_is_a_desk() {
    let state = common::desk_state(120, 36);
    let text = draw(&state);
    snapshot("mix", &text);
    has(
        &text,
        &[
            " Mix ",
            "PLAYER",
            "PREVIEW",
            "MIC",
            "MAIN · USB Audio DAC",
            "96k 24b",
            "LEAD",
            "0.0 dB",
            "[M]",
            "[S]",
            "LISTEN ONLY",
            "[ECHO]",
            "● −31 dB 42ms",
            "fill 10ms xrun 0",
            "IN VOICE",
            "WASAPI exclusive",
            "96 kHz · 24 bit · 2 ch",
            "BIT-PERFECT",
            "LIMIT",
            "12:04.3",
            "VOLUME —",
        ],
    );
    // A smaller window: the same strips, shorter faders.
    let state = common::desk_state(80, 24);
    let text = draw(&state);
    snapshot("mix_small", &text);
    has(&text, &["PLAYER", "PREVIEW", "MIC", "MAIN", "0.0 dB"]);
}
