//! Frames of Karaoke mode as the window draws it ([`Gui::view`] with
//! iced's headless renderer: the stage over the cover's ground on the GPU,
//! the controls' bar in the window's own widgets over it; without a GPU,
//! `ICED_TEST_BACKEND=vello-cpu`, the lines as text), to look at Next's
//! button and the controls' bar: the mode just entered
//! (Next alone), a song being sung (its lane, its progress along the
//! bottom) and a song's result (the diploma, Next beside it), each with
//! the controls hidden, fading in a quarter, half and three quarters of
//! the way (the stage's progress going in the first half, the bar coming
//! in the second) and shown (the pointer near the bottom). Saved as PNGs
//! into `OUT_DIR` (`<scene>-<controls>.png`, written whole). `W H`: the
//! window in pixels (1600 × 900 by default), at `SCALE` pixels to a point
//! (2 by default).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example controls_frames -- OUT_DIR [W H [SCALE]] [--theme NAME]
//! ```
//!
//! `--theme NAME`: the game in that theme (`classic`, the default, or
//! `spooky`).

mod headless;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use iced::{window, Size};
use jodel_output::{SampleFormat, SimClock, StreamFormat, StreamInfo};
use jodel_playback::{
    choose_route, Codec, Container, Event, FileFacts, PlayState, Tags, TrackInfo,
};
use jodeldiplom::app::{App, AppClock, AppOptions, Output};
use jodeldiplom::config::Config;
use jodeldiplom::core::karaoke::controls::FADE_MS;
use jodeldiplom::core::karaoke::SongLyrics;
use jodeldiplom::core::{update, Action, NowTrack, Screen, State};
use jodeldiplom::gui::window::{Gui, Message};
use jodeldiplom::work::lyrics::Answer;

use headless::Headless;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// How long the song is, in seconds.
const LENGTH: u64 = 245;
/// Where it is heard while sung, in seconds.
const HEARD: f64 = 31.0;

/// The song's lines: a word a second from 26 s, a line of four, on C4.
fn lyrics() -> jodel_lyrics::Lyrics {
    let line = |from: f64, words: &[&str]| jodel_lyrics::Line {
        words: words
            .iter()
            .enumerate()
            .map(|(k, w)| jodel_lyrics::Word {
                start: Some(from + k as f64),
                end: Some(from + k as f64 + 0.9),
                confidence: 0.9,
                ..jodel_lyrics::Word::new(*w)
            })
            .collect(),
        ..jodel_lyrics::Line::default()
    };
    jodel_lyrics::Lyrics {
        lines: vec![
            line(26.0, &["Take", "on", "me,"]),
            line(30.0, &["take", "me", "on,"]),
            line(34.0, &["I'll", "be", "gone"]),
            line(38.0, &["in", "a", "day", "or", "two"]),
        ],
        source: jodel_lyrics::Source::Lrclib,
        timing: jodel_lyrics::Timing::default(),
    }
}

/// Its melody: each word on its own note, a hundred frames a second.
fn melody() -> jodel_rhythm::waveform::Melody {
    let n = LENGTH as usize * 100;
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; n],
        sure: vec![0; n],
    };
    for (k, w) in lyrics().lines.iter().flat_map(|l| &l.words).enumerate() {
        let (a, b) = (w.start.unwrap_or(0.0), w.end.unwrap_or(0.0));
        for i in (a * 100.0) as usize..(b * 100.0) as usize {
            melody.cents[i] = 6000 + 200 * (k % 5) as u16;
            melody.sure[i] = 255;
        }
    }
    melody
}

/// Track `index` of `path` as the engine opened it.
fn now_track(state: &State, path: PathBuf, index: usize) -> NowTrack {
    let rate = 44_100;
    let info = TrackInfo {
        path,
        container: Container::Flac,
        codec: Codec::Flac,
        rate,
        channels: 2,
        bits: 24,
        float: false,
        bitrate: None,
        frames: Some(u64::from(rate) * LENGTH),
        tags: Tags {
            title: Some("Take On Me".into()),
            artist: Some("a-ha".into()),
            ..Tags::default()
        },
        cover: None,
    };
    let settings = state.config.settings();
    let format = StreamFormat::new(info.rate, 2, SampleFormat::S24In32);
    let route = choose_route(&FileFacts::from(&info), format, &settings).unwrap();
    let output = StreamInfo {
        format,
        backend: "null",
        device_buffer_frames: 4_096,
        ring_frames: 65_536,
        realtime: true,
    };
    NowTrack {
        index,
        info: Arc::new(info),
        route,
        output,
        device: "null".into(),
    }
}

/// A made-up cover: a gradient of two colours with a disc of a third.
fn cover() -> image::RgbImage {
    image::RgbImage::from_fn(256, 256, |x, y| {
        let t = (x + y) as f32 / 510.0;
        let (dx, dy) = (x as f32 - 150.0, y as f32 - 110.0);
        if dx * dx + dy * dy < 60.0 * 60.0 {
            return image::Rgb([250, 230, 200]);
        }
        let (a, b) = ([120.0, 30.0, 30.0], [200.0, 120.0, 40.0]);
        image::Rgb([0, 1, 2].map(|k| (a[k] * (1.0 - t) + b[k] * t) as u8))
    })
}

/// Karaoke mode, two songs queued, the first held for Next
/// (just entered), its lyrics and its notes ready.
fn entered() -> State {
    let mut state = State::new("/music".into(), Config::default());
    // Stems separated: the voice slider takes effect.
    state.gpu = jodeldiplom::core::Gpu::Present("a GPU".into());
    state.browser.loading = false;
    state.queue.paths = ["Take On Me.flac", "The Sun Always Shines on TV.flac"]
        .map(|f| PathBuf::from("/music").join(f))
        .to_vec();
    state.queue.current = Some(0);
    let lyrics = lyrics();
    for path in state.queue.paths.clone() {
        let found = Answer::Found {
            lyrics: lyrics.clone(),
            timed: None,
        };
        state
            .karaoke
            .lyrics
            .insert(path, SongLyrics::from_answer(found));
    }
    let path = state.queue.paths[0].clone();
    let notes = jodeldiplom::core::singing::melodic(&melody());
    state
        .karaoke
        .singing
        .prepare(&path, &Arc::new(lyrics), &notes);
    state.screen = Screen::Karaoke;
    state.karaoke.held = Some(0);
    update(&mut state, Action::Tick(1_000));
    state
}

/// The first song sung at [`HEARD`] seconds, its singer on the notes.
fn singing() -> State {
    let mut state = entered();
    let path = state.queue.paths[0].clone();
    state.playing.track = Some(now_track(&state, path, 0));
    state.playing.state = PlayState::Playing;
    state.playing.position = HEARD;
    state.karaoke.held = None;
    for i in 2_600..(HEARD * 100.0) as u32 {
        let t = f64::from(i) / 100.0;
        let on = state.karaoke.singing.word_at(t).map(|(_, c)| c as f32);
        state.karaoke.singing.hear(t, on);
    }
    state
}

/// The first song played to its end (sung as [`singing`]), the next held:
/// its result, its reveal settled (a minute on).
fn result() -> State {
    let mut state = singing();
    let ended = Event::TrackEnded {
        index: 0,
        finished: true,
    };
    update(&mut state, Action::Engine(ended));
    update(&mut state, Action::Engine(Event::Held { index: 1 }));
    update(&mut state, Action::Engine(Event::State(PlayState::Stopped)));
    let settled = state.now_ms + 60_000;
    update(&mut state, Action::Tick(settled));
    state
}

/// The controls in `state` faded in `quarters` quarters of the way (none:
/// not brought up), the pointer having come to the bottom.
fn controls(state: &mut State, quarters: Option<u64>) {
    let Some(quarters) = quarters else {
        return;
    };
    update(state, Action::PointerMoved { from_bottom: 0.0 });
    let at = state.now_ms + FADE_MS * quarters / 4;
    update(state, Action::Tick(at));
}

/// A scene's state.
type Make = fn() -> State;

/// A player with no output, no devices and no microphone, its files in
/// `dir`, timing by `clock`.
fn player(dir: PathBuf, clock: &AppClock) -> App {
    let mut options = AppOptions::new(dir);
    options.output = Output::Null(SimClock::Realtime);
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.clock = clock.clone();
    App::new(options)
}

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().expect("OUT_DIR"));
    let number = |k: usize| args.get(k).and_then(|a| a.parse::<f32>().ok());
    let (w, h) = (number(1).unwrap_or(1600.0), number(2).unwrap_or(900.0));
    let scale = number(3).unwrap_or(2.0);
    let theme = args
        .iter()
        .position(|a| a == "--theme")
        .and_then(|k| args.get(k + 1))
        .map(|name| jodeldiplom::theme::Theme::named(name).expect("a theme"));
    let size = Size::new(w / scale, h / scale);
    let dir = std::env::temp_dir().join(format!("jodeldiplom-controls-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a folder for the player");
    let mut headless = Headless::new();
    println!("drawn by {}", headless.name());
    let id = window::Id::unique();
    let event = |e| Message::Event(iced::Event::Window(e), id);
    let scenes: [(&str, Make); 3] = [
        ("entered", entered),
        ("singing", singing),
        ("result", result),
    ];
    for (name, make) in scenes {
        let fades = [
            ("hidden", None),
            ("quarter", Some(1)),
            ("half", Some(2)),
            ("three-quarters", Some(3)),
            ("shown", Some(4)),
        ];
        for (how, quarters) in fades {
            // A window of its own each: the mode just come in it, as when
            // the window first shows it (its score at once, no bursts
            // from another scene); its clock at the scene's time.
            let mut state = make();
            if let Some(theme) = theme {
                state.config.karaoke_theme = theme;
            }
            controls(&mut state, quarters);
            // Playing, the app hears the song up to a quarter second on
            // from the position last told (no engine tells it again): told
            // that much before, so it is heard where the scene is.
            if state.playing.state == PlayState::Playing {
                state.playing.position -= 0.25;
            }
            let clock = AppClock::stepped();
            let mut gui = Gui::headless(player(dir.clone(), &clock));
            clock.step(std::time::Duration::from_millis(state.now_ms));
            gui.set_art(Some(cover()));
            let _ = gui.update(event(window::Event::Resized(size)));
            let _ = gui.update(event(window::Event::Rescaled(scale)));
            gui.show(state);
            // Two frames: the first tells the renderer it has, the second
            // draws what it draws.
            for _ in 0..2 {
                let _ = gui.update(Message::Frame(Instant::now()));
                headless.frame(&gui, size);
            }
            let shot = headless.screenshot(scale);
            let file = out.join(format!("{name}-{how}.png"));
            let mut png = std::io::Cursor::new(Vec::new());
            shot.write_to(&mut png, image::ImageFormat::Png).unwrap();
            durable::write_atomic(&file, png.get_ref()).unwrap();
            println!("{}", file.display());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
