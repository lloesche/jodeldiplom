//! Renders the player's screens to files, with a fixture track playing
//! through the null output (no device) and a generated cover:
//! `gui-*.png` from the window's drawing path and `tui-*.txt` from the
//! terminal's.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example shots -- OUT_DIR [--real-devices]
//! ```
//!
//! `--real-devices` lists this machine's output devices in the settings
//! shot (nothing is opened or played); otherwise a made-up list is shown.

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, DeviceChoice, Effect, Screen, Side, State};
use jodeldiplom::gui::{font_px, Canvas};
use jodeldiplom::spectrum::SpectrumFrame;
use jodeldiplom::tui::{render_text, TermPictures};
use jodeldiplom::view::{colors, Links, Scene};

fn fake_devices() -> Result<Vec<DeviceChoice>, String> {
    Ok(vec![
        DeviceChoice {
            id: Some("usb-dac".into()),
            name: "Speakers (USB Audio DAC) (default)".into(),
            summary: "44.1 48 88.2 96 176.4 192 352.8 384 kHz · S32/S24in32 · 2 ch".into(),
        },
        DeviceChoice {
            id: Some("hdmi".into()),
            name: "Monitor (HDMI)".into(),
            summary: "48 96 192 kHz · S24_3/S16 · 2/8 ch".into(),
        },
    ])
}

fn gui_shot(
    out: &Path,
    name: &str,
    state: &State,
    spectrum: &SpectrumFrame,
    app: &App,
    scale: f64,
) {
    let (w, h) = ((1280.0 * scale) as u32, (800.0 * scale) as u32);
    let mut canvas = Canvas::new(80, 24, font_px(scale));
    let grid = canvas.fit(w, h);
    let mut state = state.clone();
    state.size = grid;
    let scene = Scene {
        state: &state,
        spectrum,
        background: colors::BG,
        links: Links::OnHover(None),
        // As the terminal's: the position.
        heard: state.playing.position,
    };
    canvas.draw(&scene, app.art());
    let path = out.join(format!("gui-{name}.png"));
    canvas.image().save(&path).expect("write the PNG");
    println!("{} ({grid:?} cells)", path.display());
}

fn tui_shot(
    out: &Path,
    name: &str,
    state: &State,
    spectrum: &SpectrumFrame,
    app: &App,
    size: (u16, u16),
) {
    let mut state = state.clone();
    state.size = size;
    let mut pictures = TermPictures::halfblocks();
    pictures.set_art(app.art());
    let text = render_text(&state, spectrum, &mut pictures, size.0, size.1);
    let path = out.join(format!("tui-{name}.txt"));
    std::fs::write(&path, text).expect("write the text");
    println!("{}", path.display());
}

/// The playing track's words timed over the Transition tab's span: a line sung
/// from a quarter of the way in, its words a little apart.
fn sung_words(state: &mut State) {
    use jodel_lyrics::{Lyrics, Source};
    use jodeldiplom::core::lanes::Block;
    use jodeldiplom::core::transition::viewport;
    let Some(index) = state.playing.track.as_ref().map(|t| t.index) else {
        return;
    };
    let path = state.queue.paths[index].clone();
    let text = "Mister rock and roll, take me home tonight";
    let (start, end) = viewport::window(state, Block::Now, state.playing.position);
    let mut timed = Lyrics::parse_text(text, Source::Lrclib);
    let words = &mut timed.lines[0].words;
    let n = words.len() as f64;
    for (k, w) in words.iter_mut().enumerate() {
        let at = start + (end - start) * (0.3 + 0.6 * k as f64 / n);
        w.start = Some(at);
        w.end = Some(at + (end - start) * 0.6 / n * if k == 0 { 0.9 } else { 0.5 });
    }
    timed.timing.by = Some(jodel_lyrics::Maker::new("ctc:shots", 1));
    timed.timing.detail = jodel_lyrics::Detail::Words;
    state.karaoke.lyrics.insert(
        path,
        jodeldiplom::core::karaoke::SongLyrics::Found {
            lyrics: std::sync::Arc::new(Lyrics::parse_text(text, Source::Lrclib)),
            shown: std::sync::Arc::new(timed),
        },
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("give an output folder"));
    let real = args.any(|a| a == "--real-devices");
    std::fs::create_dir_all(&out).unwrap();

    let tree = common::music_tree("shots");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.saved = options.config.clone();
    options.size = (120, 36);
    options.devices = if real {
        Arc::new(jodeldiplom::app::list_devices)
    } else {
        Arc::new(fake_devices)
    };
    let mut app = App::new(options);
    let ok = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        !app.state.browser.loading && app.state.pending.is_empty() && !app.state.details.is_empty()
    });
    assert!(ok, "the folder was not listed");
    app.run(Effect::ListDevices);
    common::wait_for(&mut app, Duration::from_secs(20), |app| {
        app.state.settings.devices.is_some()
    });

    // Play "02 Ramp 48" (the third row, after the parent row and the
    // first file): the whole folder is queued and playing starts there.
    app.dispatch(Action::SelectRow(2));
    app.dispatch(Action::Op(Op::OpenReplace));
    let playing = common::wait_for(&mut app, Duration::from_secs(10), |app| {
        app.state.playing.track.is_some()
            && app.state.playing.position >= 0.05
            && app.art().is_some()
    });
    assert!(playing, "the track did not start");
    let spectrum = app.spectrum();
    let mut state = app.state.clone();
    state.message = None;
    let screens: Vec<(&str, State)> = {
        let mut queue = state.clone();
        queue.side = Side::Queue;
        let mut now = state.clone();
        now.screen = Screen::NowPlaying;
        let mut settings = state.clone();
        settings.screen = Screen::Settings;
        settings.settings.picking = true;
        let mut help = state.clone();
        help.screen = Screen::Help;
        let mut filter = state.clone();
        jodeldiplom::core::update(&mut filter, Action::Op(Op::Filter));
        for c in "carr".chars() {
            jodeldiplom::core::update(&mut filter, Action::Type(c));
        }
        let mut palette = state.clone();
        jodeldiplom::core::update(&mut palette, Action::Op(Op::Palette));
        for c in "se".chars() {
            jodeldiplom::core::update(&mut palette, Action::Type(c));
        }
        vec![
            ("main", state.clone()),
            ("queue", queue),
            ("nowplaying", now),
            ("settings", settings),
            ("help", help),
            ("filter", filter),
            ("palette", palette),
            // The Transition tab, on generated tracks with stems and waveforms,
            // the playing one's words timed.
            ("transition", {
                let mut m = common::transition_state(state.size.0, state.size.1);
                sung_words(&mut m);
                m
            }),
            ("transition-set-up", {
                // A transition set up from the bar heard, drums first.
                let mut m = common::transition_state(state.size.0, state.size.1);
                m.drawn_heard = Some(m.playing.position);
                jodeldiplom::core::update(&mut m, Action::Op(Op::TransitionPreset));
                jodeldiplom::core::update(&mut m, Action::Op(Op::TransitionZoomOut));
                m
            }),
            ("mix", {
                // The Mix tab: the player at 96 kHz, a preview crossfading
                // over it (so the master sums: not bit-perfect), the
                // microphone's strip selected and hearing.
                let mut d = state.clone();
                d.side = Side::Mix;
                d.desk.selected = jodeldiplom::core::desk::Channel::Mic;
                common::fill_desk(&mut d, 96_000);
                d
            }),
            ("karaoke-tab", {
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.config.karaoke_voice = 20;
                k
            }),
            ("karaoke-sync", {
                // A sync under way: five clicks played, four answered.
                use jodeldiplom::core::sync::Session;
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.karaoke.mic.name = Some("USB Headset".into());
                let mut s = Session::new(("out".into(), "usb-headset".into()));
                s.played = 5;
                for (i, a) in s.answers.iter_mut().take(5).enumerate() {
                    *a = (i != 2).then_some(0.15);
                }
                k.karaoke.sync = Some(s);
                k
            }),
            ("karaoke-sync-done", {
                // A sync measured: the delay and the keys.
                use jodeldiplom::core::sync::{outcome, Session, CLICKS};
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.karaoke.mic.name = Some("USB Headset".into());
                let mut s = Session::new(("out".into(), "usb-headset".into()));
                s.played = CLICKS;
                s.answers = vec![Some(0.152); CLICKS];
                s.answers[6] = None;
                s.outcome = Some(outcome(&s.answers));
                k.karaoke.sync = Some(s);
                k
            }),
            ("karaoke", {
                // The mode, the song paused: START shows.
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.screen = Screen::Karaoke;
                k.playing.state = jodel_playback::PlayState::Paused;
                k
            }),
            ("karaoke-entered", {
                // Into the mode from the tab, nothing playing: START shows.
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.playing.state = jodel_playback::PlayState::Stopped;
                jodeldiplom::core::update(&mut k, Action::Op(Op::KaraokeEnter));
                k
            }),
            ("karaoke-count", {
                // Half-way through "2" of a count-in (heard at 0 s).
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.screen = Screen::Karaoke;
                k.karaoke.countdown =
                    k.playing
                        .track
                        .as_ref()
                        .map(|t| jodeldiplom::core::karaoke::Countdown {
                            index: t.index,
                            beat: 0.5,
                            on: 0.75,
                            title: 0.0,
                        });
                k
            }),
        ]
    };
    for (name, s) in &screens {
        gui_shot(&out, name, s, &spectrum, &app, 1.0);
        tui_shot(&out, name, s, &spectrum, &app, (120, 36));
    }
    gui_shot(&out, "main-hidpi", &screens[0].1, &spectrum, &app, 2.0);
    tui_shot(
        &out,
        "main-narrow",
        &screens[0].1,
        &spectrum,
        &app,
        (64, 22),
    );
    // The Mix tab in an 80-column terminal, the default one's width.
    if let Some((_, mix)) = screens.iter().find(|(name, _)| *name == "mix") {
        tui_shot(&out, "mix-80", mix, &spectrum, &app, (80, 24));
    }
    app.shutdown();
    let _ = std::fs::remove_dir_all(&tree.root);
}
