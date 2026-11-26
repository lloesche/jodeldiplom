//! Renders the player's screens to files, with a fixture track playing
//! through the null output (no device) and a generated cover: `gui-*.png`
//! as the window draws them ([`Gui::view`], with iced's headless renderer:
//! the old canvas, the widgets iced draws over it, and what the GPU draws
//! itself, Karaoke mode's stage among it), and for each a `gui-*.txt` with
//! the text of its cells (the pictures' cells blank). Each screen at
//! 1280 × 800 at scale 1, and as `gui-*-hidpi` at 1512 × 982 at scale 2.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example shots -- OUT_DIR [--real-devices]
//! ```
//!
//! `--real-devices` lists this machine's output devices in the settings
//! shot (nothing is opened or played); otherwise a made-up list is shown.

#[path = "../tests/common/mod.rs"]
mod common;
mod headless;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::{window, Size};
use jodel_output::SimClock;
use jodeldiplom::app::{App, AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, DeviceChoice, Effect, Screen, Side, State};
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::gui::{font_px, Canvas};

use headless::Headless;

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

/// The window the shots are taken in, and where they go.
struct Shots {
    gui: Gui,
    headless: Headless,
    id: window::Id,
    out: PathBuf,
}

impl Shots {
    /// `state` in a window of `size` (logical pixels) at `scale`, as
    /// `gui-{name}.png` and `.txt`: told the window's size and scale as a
    /// window would, ticked once (the frame drawn whole), then the view
    /// drawn and rendered.
    fn shot(&mut self, name: &str, state: &State, size: Size, scale: f32) {
        let event = |e| Message::Event(iced::Event::Window(e), self.id);
        self.gui.show(state.clone());
        let _ = self.gui.update(event(window::Event::Resized(size)));
        let _ = self.gui.update(event(window::Event::Rescaled(scale)));
        let _ = self.gui.update(Message::Frame(Instant::now()));
        self.headless.frame(&self.gui, size);
        let path = self.out.join(format!("gui-{name}.png"));
        let mut png = durable::AtomicFile::create(&path).expect("write the PNG");
        self.headless
            .screenshot(scale)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("a PNG");
        png.commit().expect("write the PNG");
        durable::write_atomic(
            &self.out.join(format!("gui-{name}.txt")),
            self.gui.text().as_bytes(),
        )
        .expect("write the text");
        println!(
            "{} ({} × {} at {scale})",
            path.display(),
            size.width,
            size.height
        );
    }

    /// `state` in a window of exactly `cells`, at a scale of 1.
    fn cells_shot(&mut self, name: &str, state: &State, cells: (u16, u16)) {
        let (cw, ch) = Canvas::new(cells.0, cells.1, font_px(1.0)).cell();
        let size = Size::new(
            (cw * usize::from(cells.0)) as f32,
            (ch * usize::from(cells.1)) as f32,
        );
        self.shot(name, state, size, 1.0);
    }
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
    // The app runs under the shots: no microphone of this machine opened
    // on the Karaoke and Mixer tabs.
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
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
            ("mixer", {
                // The Mixer tab: the player at 96 kHz, a preview crossfading
                // over it (so the master sums: not bit-perfect), the
                // microphone's strip selected and hearing.
                let mut d = state.clone();
                d.side = Side::Mixer;
                d.desk.selected = jodeldiplom::core::desk::Channel::Mic;
                common::fill_desk(&mut d, 96_000);
                d
            }),
            ("karaoke-tab", {
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.config.karaoke_voice = 44;
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
    let mut shots = Shots {
        gui: Gui::headless(app),
        headless: Headless::new(),
        id: window::Id::unique(),
        out,
    };
    println!("drawn by {}", shots.headless.name());
    // Which renderer draws is known to the canvas once a frame was drawn:
    // the first shot paints what the GPU does not draw.
    shots.headless.frame(&shots.gui, Size::new(1280.0, 800.0));
    for (name, s) in &screens {
        shots.shot(name, s, Size::new(1280.0, 800.0), 1.0);
        shots.shot(&format!("{name}-hidpi"), s, Size::new(1512.0, 982.0), 2.0);
    }
    shots.cells_shot("main-narrow", &screens[0].1, (64, 22));
    // The Mixer tab in the narrowest usual window, 80 columns.
    if let Some((_, mixer)) = screens.iter().find(|(name, _)| *name == "mixer") {
        shots.cells_shot("mixer-80", mixer, (80, 24));
    }
    // The player stops with the window's state.
    drop(shots);
    let _ = std::fs::remove_dir_all(&tree.root);
}
