//! Renders the player's screens to files, with a fixture track playing
//! through the null output (no device) and a generated cover: `gui-*.png`
//! as the window draws them ([`Gui::view`], with iced's headless renderer:
//! the old canvas, the widgets iced draws over it, and what the GPU draws
//! itself, Karaoke mode's stage among it), and for each a `gui-*.txt` with
//! the text of its cells (the pictures' cells blank), or the outline of
//! its model where the window draws it whole in iced's widgets (Now
//! playing, the settings, the help). Each screen at
//! 1280 × 800 at scale 1, and as `gui-*-hidpi` at 1512 × 982 at scale 2.
//! The main screen also with tags and a device name too long for any
//! column (`gui-main-long`), with tags in Japanese (`gui-main-cjk`),
//! narrow (`gui-main-narrow`, `gui-main-narrow-long`), and last without
//! a cover (`gui-main-noart`, the Japanese tags).
//! The setup screen comes first, on a made-up web: under way
//! (`gui-setup`) and stopped by a failure (`gui-setup-failed`), each with
//! its outline as the text. The fonts are the player's own, as built in.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example shots -- OUT_DIR [--real-devices] [--only NAME]
//! ```
//!
//! `--real-devices` lists this machine's output devices in the settings
//! shot (nothing is opened or played); otherwise a made-up list is shown.
//! `--only NAME`: only the shots whose names start with `NAME`.

#[path = "../tests/common/mod.rs"]
mod common;
mod headless;
#[path = "../tests/common/setup.rs"]
mod setup_web;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::{window, Size};
use jodel_output::SimClock;
use jodeldiplom::app::{AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::{Action, DeviceChoice, Effect, Screen, Side, State};
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::gui::{font_px, Canvas};

use headless::Headless;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// The sizes every screen is shot at: a name's suffix, the window's size
/// in logical pixels, its scale.
const SIZES: [(&str, Size, f32); 2] = [
    ("", Size::new(1280.0, 800.0), 1.0),
    ("-hidpi", Size::new(1512.0, 982.0), 2.0),
];

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
    /// drawn, and drawn again once what eases in has settled, and
    /// rendered.
    fn shot(&mut self, name: &str, state: &State, size: Size, scale: f32) {
        let event = |e| Message::Event(iced::Event::Window(e), self.id);
        self.gui.show(state.clone());
        let _ = self.gui.update(event(window::Event::Resized(size)));
        let _ = self.gui.update(event(window::Event::Rescaled(scale)));
        let _ = self.gui.update(Message::Frame(Instant::now()));
        self.headless.frame(&self.gui, size);
        // What eases (the tabs' pill, a row's selection, the palette
        // coming in) settled, as a person sees it a moment later.
        self.headless.settled(&self.gui, size);
        let path = self.out.join(format!("gui-{name}.png"));
        write_png(&path, &self.headless.screenshot(scale));
        durable::write_atomic(
            &self.out.join(format!("gui-{name}.txt")),
            self.text().as_bytes(),
        )
        .expect("write the text");
        println!(
            "{} ({} × {} at {scale})",
            path.display(),
            size.width,
            size.height
        );
    }

    /// The text of what shows: a screen's outline where the window lays
    /// it out in iced's widgets, the old canvas's cells otherwise.
    fn text(&self) -> String {
        use jodeldiplom::view::{help, now, settings};
        match self.gui.state() {
            Some(s) if s.screen == Screen::NowPlaying => {
                now::NowPlaying::of(s, s.heard()).to_string()
            }
            Some(s) if s.screen == Screen::Settings => settings::Settings::of(s).to_string(),
            Some(s) if s.screen == Screen::Help => help::Page::of(s).to_string(),
            _ => self.gui.text(),
        }
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

/// `image` into the PNG file `path`.
fn write_png(path: &Path, image: &image::RgbaImage) {
    let mut png = durable::AtomicFile::create(path).expect("write the PNG");
    image
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("a PNG");
    png.commit().expect("write the PNG");
}

/// The setup screen under way (a download stalled part-way, its speed
/// known) and stopped (offline), as `gui-setup*.png` with the screen's
/// outline as `gui-setup*.txt`.
fn setup_shots(headless: &mut Headless, out: &Path, scratch: &Path) {
    use jodeldiplom::view::setup::Screen;
    use setup_web::{start, when, Web};
    let id = window::Id::unique();
    let stalls = Web::Stalls {
        share: 0.43,
        after: Duration::from_secs(4),
    };
    for (name, web) in [("setup", stalls), ("setup-failed", Web::Offline)] {
        let (setup, _made) = start(web, &scratch.join(name));
        let status = when(&setup, |s| s.rate.is_some() || s.failure.is_some());
        let mut gui = Gui::headless_setup(setup, AppOptions::new(scratch.to_path_buf()));
        for (suffix, size, scale) in SIZES {
            let event = |e| Message::Event(iced::Event::Window(e), id);
            let _ = gui.update(event(window::Event::Resized(size)));
            let _ = gui.update(event(window::Event::Rescaled(scale)));
            headless.frame(&gui, size);
            let path = out.join(format!("gui-{name}{suffix}.png"));
            write_png(&path, &headless.screenshot(scale));
            durable::write_atomic(
                &out.join(format!("gui-{name}{suffix}.txt")),
                Screen::of(&status).to_string().as_bytes(),
            )
            .expect("write the text");
            println!(
                "{} ({} × {} at {scale})",
                path.display(),
                size.width,
                size.height
            );
        }
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

/// `state` with the playing track's title, artist and album as given,
/// and its device where one is given.
fn retagged(state: &State, [title, artist, album]: [&str; 3], device: Option<&str>) -> State {
    let mut state = state.clone();
    if let Some(track) = state.playing.track.as_mut() {
        let mut info = (*track.info).clone();
        info.tags.title = Some(title.into());
        info.tags.artist = Some(artist.into());
        info.tags.album = Some(album.into());
        track.info = Arc::new(info);
        if let Some(device) = device {
            track.device = device.into();
        }
    }
    state
}

/// Tags too long for any column, and a device name as long as they come.
const LONG: [&str; 3] = [
    "Everything You Always Wanted to Hear (Extended Twelve-Inch Remix, Remastered 2026)",
    "The Very Long Named Orchestra and Choir of Somewhere Far Away",
    "Collected Recordings, Volume Seventeen: The Late Night Sessions",
];
const LONG_DEVICE: &str = "Digital Output (High Definition Audio Device 2)";

/// Tags in Japanese.
const CJK: [&str; 3] = [
    "春の海辺で聴いた歌",
    "山田花子と星の楽団",
    "夏の終わりのメロディー 第二集",
];

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("give an output folder"));
    let rest: Vec<String> = args.collect();
    let real = rest.iter().any(|a| a == "--real-devices");
    let only = rest
        .iter()
        .position(|a| a == "--only")
        .map(|i| rest.get(i + 1).cloned().expect("--only NAME"));
    let wanted = |name: &str| only.as_ref().is_none_or(|o| name.starts_with(o.as_str()));
    std::fs::create_dir_all(&out).unwrap();
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let mut headless = Headless::new();
    println!("drawn by {}", headless.name());
    let scratch = common::temp_dir("shots-setup");
    if wanted("setup") {
        setup_shots(&mut headless, &out, &scratch);
    }

    let tree = common::music_tree("shots");
    let mut options = AppOptions::new(tree.album.clone());
    options.output = Output::Null(SimClock::Realtime);
    options.saved = options.config.clone();
    options.devices = if real {
        Arc::new(jodeldiplom::app::list_devices)
    } else {
        Arc::new(fake_devices)
    };
    // The app runs under the shots: no microphone of this machine opened
    // on the Karaoke and Mixer tabs.
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    let mut app = common::fitted_app(options, 120, 36);
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
        // The rows alone, the stems' row selected, stems on a GPU.
        let mut settings_rows = state.clone();
        settings_rows.screen = Screen::Settings;
        settings_rows.settings.row = 6;
        settings_rows.gpu = jodeldiplom::core::Gpu::Present("a GPU".into());
        settings_rows.config.stems = true;
        // The data folder's list open on a folder of four.
        let mut settings_folders = state.clone();
        settings_folders.screen = Screen::Settings;
        settings_folders.settings.row = 10;
        let home = PathBuf::from(if cfg!(windows) {
            r"D:\Music"
        } else {
            "/home/me"
        });
        settings_folders.settings.folders = Some(jodeldiplom::core::FolderPick {
            dir: Some(home.clone()),
            folders: Some(
                ["Albums", "Downloads", "Karaoke", "Players"]
                    .iter()
                    .map(|n| home.join(n))
                    .collect(),
            ),
            cursor: 3,
        });
        let mut help = state.clone();
        help.screen = Screen::Help;
        let mut help_search = help.clone();
        jodeldiplom::core::update(&mut help_search, Action::Op(Op::Filter));
        for c in "volume".chars() {
            jodeldiplom::core::update(&mut help_search, Action::Type(c));
        }
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
            ("main-long", retagged(&state, LONG, Some(LONG_DEVICE))),
            ("main-cjk", retagged(&state, CJK, None)),
            ("queue", queue),
            ("nowplaying", now),
            ("settings", settings),
            ("settings-rows", settings_rows),
            ("settings-folders", settings_folders),
            ("help", help),
            ("help-search", help_search),
            ("filter", filter),
            ("palette", palette),
            // The Transition tab, on generated tracks with stems and waveforms,
            // the playing one's words timed.
            ("transition", {
                let mut m = common::transition_state(120, 36);
                sung_words(&mut m);
                m
            }),
            ("transition-palette", {
                // The command palette over the Transition tab's lanes.
                let mut m = common::transition_state(120, 36);
                sung_words(&mut m);
                jodeldiplom::core::update(&mut m, Action::Op(Op::Palette));
                for c in "tra".chars() {
                    jodeldiplom::core::update(&mut m, Action::Type(c));
                }
                m
            }),
            ("transition-set-up", {
                // A transition set up from the bar heard, drums first.
                let mut m = common::transition_state(120, 36);
                m.drawn_heard = Some(m.playing.position);
                jodeldiplom::core::update(&mut m, Action::Op(Op::TransitionPreset));
                jodeldiplom::core::update(&mut m, Action::Op(Op::TransitionZoomOut));
                m
            }),
            ("transition-held", {
                // The playing track's view held (its cursor moved on, bars
                // selected), the playhead going across it.
                let mut m = common::transition_state(120, 36);
                sung_words(&mut m);
                for o in [Op::TransitionBarForward, Op::TransitionBarForward] {
                    jodeldiplom::core::update(&mut m, Action::Op(o));
                }
                for _ in 0..3 {
                    jodeldiplom::core::update(&mut m, Action::Op(Op::TransitionSelectForward));
                }
                m
            }),
            ("transition-edit", {
                // Both cues set, drums first, the next track's bass fade
                // being edited.
                let mut m = common::transition_state(120, 36);
                for o in [
                    Op::Accept,
                    Op::Down,
                    Op::Accept,
                    Op::Up,
                    Op::TransitionPreset,
                    Op::TransitionEdit,
                ] {
                    jodeldiplom::core::update(&mut m, Action::Op(o));
                }
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
            ("karaoke-lane", {
                // The tab on a generated song with its stems in: the
                // voice's lane, its words timed, going by.
                let mut k = common::transition_state(120, 36);
                k.side = Side::Karaoke;
                sung_words(&mut k);
                k
            }),
            ("karaoke-game-off", {
                // The same with the game and the notes off, the singing
                // kept, the Delay row at hand.
                let mut k = common::transition_state(120, 36);
                k.side = Side::Karaoke;
                sung_words(&mut k);
                k.config.karaoke_game = false;
                k.config.karaoke_notes = false;
                k.config.karaoke_keep = true;
                k.karaoke.focus = jodeldiplom::core::sliders::Slider::Delay;
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
            ("karaoke-controls", {
                // The song playing, the pointer come to the bottom: the
                // controls' bar faded in over the stage.
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.screen = Screen::Karaoke;
                k.config.karaoke_voice = 44;
                jodeldiplom::core::update(&mut k, Action::PointerMoved { from_bottom: 0.0 });
                let shown = k.now_ms + jodeldiplom::core::karaoke::controls::FADE_MS;
                jodeldiplom::core::update(&mut k, Action::Tick(shown));
                k
            }),
        ]
    };
    let mut shots = Shots {
        gui: Gui::headless(app),
        headless,
        id: window::Id::unique(),
        out,
    };
    // Which renderer draws is known to the canvas once a frame was drawn:
    // the first shot paints what the GPU does not draw.
    shots.headless.frame(&shots.gui, Size::new(1280.0, 800.0));
    for (name, s) in screens.iter().filter(|(name, _)| wanted(name)) {
        for (suffix, size, scale) in SIZES {
            shots.shot(&format!("{name}{suffix}"), s, size, scale);
        }
    }
    if wanted("main-narrow") {
        shots.cells_shot("main-narrow", &screens[0].1, (64, 22));
        let long = retagged(&screens[0].1, LONG, Some(LONG_DEVICE));
        shots.cells_shot("main-narrow-long", &long, (64, 22));
    }
    // The Transition tab in a narrow window, set up, at both scales.
    if let Some((_, tab)) = screens
        .iter()
        .find(|(name, _)| *name == "transition-set-up")
    {
        if wanted("transition-narrow") {
            let size = Size::new(800.0, 560.0);
            shots.shot("transition-narrow", tab, size, 1.0);
            shots.shot("transition-narrow-hidpi", tab, size, 2.0);
        }
    }
    // The Karaoke tab in a window half as wide, its lyrics ready.
    if let Some((_, lane)) = screens.iter().find(|(name, _)| *name == "karaoke-lane") {
        if wanted("karaoke-narrow") {
            shots.shot("karaoke-narrow", lane, Size::new(640.0, 800.0), 1.0);
            shots.shot("karaoke-narrow-hidpi", lane, Size::new(756.0, 982.0), 2.0);
        }
    }
    // The Mixer tab in the narrowest usual window, 80 columns, and in a
    // window 800 wide.
    if let Some((_, mixer)) = screens.iter().find(|(name, _)| *name == "mixer") {
        if wanted("mixer-80") {
            shots.cells_shot("mixer-80", mixer, (80, 24));
        }
        if wanted("mixer-800") {
            shots.shot("mixer-800", mixer, Size::new(800.0, 600.0), 1.0);
        }
    }
    // Last, as the cover stays gone: the main screen without art, the
    // CJK tags.
    if wanted("main-noart") {
        shots.gui.set_art(None);
        let cjk = retagged(&screens[0].1, CJK, None);
        for (suffix, size, scale) in SIZES {
            shots.shot(&format!("main-noart{suffix}"), &cjk, size, scale);
        }
    }
    // The player stops with the window's state.
    drop(shots);
    let _ = std::fs::remove_dir_all(&tree.root);
    let _ = std::fs::remove_dir_all(&scratch);
}
