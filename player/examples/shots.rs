//! Renders the player's screens to files, with a fixture track playing
//! through the null output (no device) and a generated cover: `gui-*.png`
//! as the window draws them ([`Gui::view`], with iced's headless renderer:
//! the widgets iced draws and what the GPU draws itself, Karaoke mode's
//! stage among it), and for each a `gui-*.txt` with the outline of its
//! models (none for Karaoke mode); the lyrics editor on a made-up song
//! whose voice is made from its words. Each screen at 1280 × 800 at scale
//! 1, and as `gui-*-hidpi` at 1512 × 982 at scale 2.
//! The main screen also with tags and a device name too long for any
//! column (`gui-main-long`), with tags in Japanese (`gui-main-cjk`),
//! narrow (`gui-main-narrow`, `gui-main-narrow-long`), and last without
//! a cover (`gui-main-noart`, the Japanese tags).
//! The queue with a song being got ready, its preparation in the right
//! column at each step: the stems being separated (`gui-prep-stems`), the
//! key found (`gui-prep-key`), the beat found (`gui-prep-beat`), the words
//! timed (`gui-prep-words`), its story planned, its words typed over the
//! cover (`gui-prep-plan`), and its scenes drawn over the cover's tiles
//! (`gui-prep-story`); made-up results, as the workers would tell them.
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
mod prep;
#[path = "../tests/common/setup.rs"]
mod setup_web;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::{window, Size};
use jodel_output::SimClock;
use jodeldiplom::app::{AppOptions, Output};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::timeview::TimeView;
use jodeldiplom::core::{Action, DeviceChoice, Effect, Screen, Side, State};
use jodeldiplom::gui::window::{Gui, Message};

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
            self.text(size).as_bytes(),
        )
        .expect("write the text");
        println!(
            "{} ({} × {} at {scale})",
            path.display(),
            size.width,
            size.height
        );
    }

    /// The outline of what shows in a window of `size`: its screen's
    /// models (none for Karaoke mode, a picture).
    fn text(&self, size: Size) -> String {
        use jodeldiplom::view::{help, lyrics, now, settings};
        match self.gui.state() {
            Some(s) if s.screen == Screen::NowPlaying => {
                now::NowPlaying::of(s, s.heard()).to_string()
            }
            Some(s) if s.screen == Screen::Settings => settings::Settings::of(s).to_string(),
            Some(s) if s.screen == Screen::Help => help::Page::of(s).to_string(),
            Some(s) if s.screen == Screen::Main => main_outline(s, size),
            Some(s) if matches!(s.screen, Screen::Lyrics(_)) => lyrics::Editor::of(s)
                .map(|e| e.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        }
    }

    /// `state` in a window of `cells` of the old cells, at a scale of 1.
    fn cells_shot(&mut self, name: &str, state: &State, cells: (u16, u16)) {
        self.shot(name, state, common::window(cells.0, cells.1), 1.0);
    }
}

/// The main screen's models in a window of `size`: the list pane (its
/// tabs, the browser's or the queue's rows), the right column's text (or
/// the summary under the list), the side's tab, the command palette, the
/// bottom bar.
fn main_outline(state: &State, size: Size) -> String {
    use jodeldiplom::view::{bottom, command_palette, info, karaoke, list, transition};
    let m = jodeldiplom::gui::metrics::main(size, state.side);
    let mut out = list::Pane::of(state, m.pane.width).to_string();
    match state.side {
        Side::Transition => out.push_str(&transition::Tab::of(state, state.heard()).to_string()),
        Side::Karaoke => out.push_str(&karaoke::Tab::of(state).to_string()),
        Side::Synth => out.push_str(&jodeldiplom::view::synth::Synth::of(state).to_string()),
        Side::Shader => out.push_str(&jodeldiplom::gui::shader::outline(state)),
        Side::Browser | Side::Queue | Side::Mixer => {}
    }
    // The right column: a song's preparation while it shows, else the
    // track's text.
    match (&m.right, jodeldiplom::view::prep::PrepView::of(state)) {
        (Some(_), Some(prep)) => out.push_str(&prep.to_string()),
        (Some(_), None) => out.push_str(&info::Info::of(state).to_string()),
        (None, _) => {}
    }
    if m.summary.is_some() {
        out.push_str(&info::Info::of(state).summary());
    }
    if let Some(palette) = command_palette::Palette::of(state, 8) {
        out.push_str(&palette.to_string());
    }
    out.push_str(&bottom::Bar::of(state, size.width).to_string());
    out
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
/// known), stopped (offline), and as every start shows it first (the
/// files looked for, the player's own start to come), as `gui-setup*.png`
/// with the screen's outline as `gui-setup*.txt`.
fn setup_shots(headless: &mut Headless, out: &Path, scratch: &Path) {
    use jodeldiplom::view::setup::Screen;
    use setup_web::{start, when, Web};
    let id = window::Id::unique();
    let stalls = Web::Stalls {
        share: 0.43,
        after: Duration::from_secs(4),
    };
    let mut screens = Vec::new();
    for (name, web) in [("setup", stalls), ("setup-failed", Web::Offline)] {
        let (setup, _made) = start(web, &scratch.join(name));
        when(&setup, |s| s.rate.is_some() || s.failure.is_some());
        screens.push((name, Gui::headless_setup(setup)));
    }
    // The start: the files still looked for while it is taken (the
    // manifest waits until these shots are done).
    let (_found, finding) = std::sync::mpsc::channel::<()>();
    let mut options = AppOptions::new(scratch.to_path_buf());
    options.output = Output::Null(SimClock::Realtime);
    let manifest = move || {
        let _ = finding.recv();
        Vec::new()
    };
    let work = Arc::new(jodeldiplom::setup::Real);
    screens.push((
        "setup-start",
        Gui::headless_starting(options, manifest, work),
    ));
    for (name, mut gui) in screens {
        let status = gui.setup_status().expect("the setup screen");
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

/// A made-up song for the lyrics editor on the Transition tab's tracks:
/// two lines sung from 62 s on the playing track, its voice's stem made
/// from them (each word's attack, its held note with a little vibrato,
/// its release), and the words as the aligner timed them a little off
/// the voice in places, the second line's first two a backing vocal; the
/// editor open on them, the song paused in the first line.
fn lyrics_song() -> State {
    use jodel_lyrics::{Lyrics, Source};
    use jodel_rhythm::waveform::FPS;
    use jodeldiplom::core::{Rhythm, Rhythmed};
    let mut state = common::transition_state(120, 36);
    state.side = Side::Karaoke;
    let index = state.playing.track.as_ref().map_or(1, |t| t.index);
    let path = state.queue.paths[index].clone();
    let lines: [&[(&str, f64)]; 2] = [
        &[
            ("Mister", 0.55),
            ("rock", 0.35),
            ("and", 0.2),
            ("roll,", 0.7),
            ("take", 0.3),
            ("me", 0.25),
            ("home", 0.45),
            ("tonight", 0.95),
        ],
        &[
            ("(oh", 0.3),
            ("yeah)", 0.5),
            ("the", 0.2),
            ("lights", 0.45),
            ("are", 0.25),
            ("low", 0.6),
            ("along", 0.4),
            ("the", 0.2),
            ("river", 1.0),
        ],
    ];
    // Where each word is sung.
    let mut sung = Vec::new();
    let mut t = 62.0;
    for line in lines {
        for &(_, length) in line {
            sung.push((t, t + length));
            t += length + 0.09;
        }
        t += 1.1;
    }
    let text: String = lines
        .iter()
        .map(|l| l.iter().map(|w| w.0).collect::<Vec<_>>().join(" ") + "\n")
        .collect();
    let mut timed = Lyrics::parse_text(&text, Source::Lrclib);
    // The aligner a little off in places: "roll," early, "lights" short.
    let off = |k: usize| match k {
        3 => (-0.12, -0.18),
        11 => (0.0, -0.2),
        _ => (0.0, 0.0),
    };
    for (k, w) in timed
        .lines
        .iter_mut()
        .flat_map(|l| &mut l.words)
        .enumerate()
    {
        let (a, b) = sung[k];
        let (da, db) = off(k);
        (w.start, w.end, w.confidence) = (Some(a + da), Some(b + db), 0.8);
    }
    timed.timing.by = Some(jodel_lyrics::Maker::new("ctc:shots", 1));
    timed.timing.detail = jodel_lyrics::Detail::Words;
    // The voice's stem from what is sung.
    let Some(Rhythm::Ready(r)) = state.rhythm.get(&path).cloned() else {
        panic!("the playing track is analysed");
    };
    let mut waveform = common::waveform(245.0, 128.0, 0.4, 1e9);
    let level = |t: f64| -> f64 {
        sung.iter()
            .map(|&(a, b)| {
                if t < a - 0.01 || t > b + 0.08 {
                    return 0.0;
                }
                let attack = ((t - a) / 0.03).clamp(0.0, 1.0);
                let release = (1.0 - (t - b) / 0.08).clamp(0.0, 1.0);
                let held =
                    0.82 + 0.1 * ((t - a) * 34.0).sin() - 0.25 * ((t - a) / (b - a)).min(1.0);
                attack * release * held.max(0.3)
            })
            .fold(0.0, f64::max)
    };
    if let Some(stems) = waveform.stems.as_mut() {
        let vocals = &mut stems[3];
        for (i, (peak, rms)) in vocals.peak.iter_mut().zip(&mut vocals.rms).enumerate() {
            let breath = 0.03 + 0.02 * ((i as f64) * 0.37).sin().abs();
            let v = level(i as f64 / FPS).max(breath);
            *peak = (v * 235.0) as u8;
            *rms = (v * 170.0) as u8;
        }
    }
    state.rhythm.insert(
        path.clone(),
        Rhythm::Ready(Arc::new(Rhythmed::new(
            (*r.analysis).clone(),
            Some(waveform),
        ))),
    );
    state.karaoke.lyrics.insert(
        path,
        jodeldiplom::core::karaoke::SongLyrics::Found {
            lyrics: Arc::new(Lyrics::parse_text(&text, Source::Lrclib)),
            shown: Arc::new(timed),
        },
    );
    state.karaoke.lyrics_wanted = jodeldiplom::core::karaoke::lyrics_wanted(&state);
    state.playing.state = jodel_playback::PlayState::Paused;
    state.playing.position = 63.1;
    state.drawn_heard = Some(63.1);
    jodeldiplom::core::update(&mut state, Action::Op(Op::LyricsEditor));
    if let Screen::Lyrics(editor) = &mut state.screen {
        (editor.screen.view, editor.screen.span) = (TimeView::held(61.4), 12.0);
    }
    state
}

/// The lyrics editor's shots: a word set by hand, the backing vocal not to
/// be sung with the voice raised over it, "roll," selected (`lyrics-editor`)
/// and its end being dragged later onto its voice (`lyrics-editor-drag`);
/// the second line being written (`lyrics-editor-write`).
fn lyrics_shots() -> Vec<(&'static str, State)> {
    use jodel_lyrics::WordAt;
    use jodeldiplom::core::lyric_edit::{Change, Edit, LyricsEdit, Part, Selection};
    let act = |s: &mut State, what: LyricsEdit| {
        jodeldiplom::core::update(s, Action::LyricsEdit(what));
    };
    let mut base = lyrics_song();
    for (line, word) in [(1, 0), (1, 1)] {
        act(
            &mut base,
            LyricsEdit::Edit(Edit::Select(Selection::Word(WordAt { line, word }))),
        );
        act(
            &mut base,
            LyricsEdit::Edit(Edit::Change(Change::NotSung(true))),
        );
    }
    act(
        &mut base,
        LyricsEdit::Edit(Edit::Change(Change::Voice(Some(-3.0)))),
    );
    act(
        &mut base,
        LyricsEdit::Edit(Edit::Select(Selection::Word(WordAt { line: 0, word: 6 }))),
    );
    act(
        &mut base,
        LyricsEdit::Edit(Edit::Change(Change::Move(0.04))),
    );
    act(
        &mut base,
        LyricsEdit::Edit(Edit::Select(Selection::Word(WordAt { line: 0, word: 3 }))),
    );
    let mut drag = base.clone();
    let end = jodeldiplom::core::lyric_edit::editor(&drag)
        .and_then(|e| e.word())
        .and_then(|w| w.end)
        .unwrap_or(0.0);
    act(
        &mut drag,
        LyricsEdit::Hold {
            word: WordAt { line: 0, word: 3 },
            part: Part::End,
            time: end,
            slop: 0.01,
        },
    );
    act(&mut drag, LyricsEdit::Drag(end + 0.16));
    let mut write = base.clone();
    act(
        &mut write,
        LyricsEdit::Edit(Edit::Select(Selection::Word(WordAt { line: 1, word: 3 }))),
    );
    act(&mut write, LyricsEdit::Write { line: true });
    act(
        &mut write,
        LyricsEdit::Draft("(oh yeah) the lights are low along the river bend".into()),
    );
    vec![
        ("lyrics-editor", base),
        ("lyrics-editor-drag", drag),
        ("lyrics-editor-write", write),
    ]
}

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
        Arc::new(jodeldiplom::desk_wiring::list_devices)
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
            ("prep-stems", prep::preparing(&state, "stems")),
            ("prep-key", prep::preparing(&state, "key")),
            ("prep-beat", prep::preparing(&state, "beat")),
            ("prep-words", prep::preparing(&state, "words")),
            ("prep-plan", prep::preparing(&state, "plan")),
            ("prep-story", prep::preparing(&state, "story")),
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
            ("synth", {
                // The Synth tab on the top rung's stinger (its layers, its
                // many patches), the song's key and tempo.
                use jodeldiplom::sfx::bench::{What, SOUNDS};
                let mut s = state.clone();
                s.side = Side::Synth;
                let top = jodeldiplom::tiers::Tier::TOP;
                s.synth.sound = SOUNDS
                    .iter()
                    .position(|x| x.what == What::Rung(top))
                    .unwrap_or(0);
                s
            }),
            ("shader", {
                // The Shader tab on the night drive, its halation selected.
                let mut s = state.clone();
                s.side = Side::Shader;
                s.shader.look = jodel_looks::presets::named("35mm-night-drive").cloned();
                s.shader.layer = 2;
                s
            }),
            ("synth-edited", {
                // A word's pop edited (a longer decay, more reverb) and not
                // kept yet, in A minor a little slower, a chord held on the
                // keyboard.
                use jodeldiplom::core::synth_tab::SynthAction;
                let mut s = state.clone();
                s.side = Side::Synth;
                let pop = jodeldiplom::theme::CLASSIC.patch(jodeldiplom::sfx::conductor::Atom::Pop);
                let knob = |key: &str| pop.find(key).unwrap().k;
                for a in [
                    SynthAction::KeyStep(22),
                    SynthAction::TempoStep(-5),
                    SynthAction::Slide {
                        knob: knob("envelope decay"),
                        share: 0.4,
                    },
                    SynthAction::Slide {
                        knob: knob("sends reverb"),
                        share: 0.5,
                    },
                    SynthAction::SlideEnd,
                ] {
                    jodeldiplom::core::update(&mut s, Action::Synth(a));
                }
                for k in [0, 4, 7] {
                    jodeldiplom::core::update(&mut s, Action::Op(Op::SynthNote(k)));
                }
                s
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
                // The mode, the song paused: Next shows.
                let mut k = state.clone();
                k.side = Side::Karaoke;
                k.screen = Screen::Karaoke;
                k.playing.state = jodel_playback::PlayState::Paused;
                k
            }),
            ("karaoke-entered", {
                // Into the mode from the tab, nothing playing: Next shows.
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
        .into_iter()
        .chain(lyrics_shots())
        .collect()
    };
    let mut shots = Shots {
        gui: Gui::headless(app),
        headless,
        id: window::Id::unique(),
        out,
    };
    // Which renderer draws is known to the lanes and the stage once a frame
    // was drawn:
    // the first shot paints what the GPU does not draw.
    shots.headless.frame(&shots.gui, Size::new(1280.0, 800.0));
    // What the player still had to tell (its queue, its track) taken in
    // now, not over the first shot's state.
    let _ = shots.gui.update(Message::Frame(Instant::now()));
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
