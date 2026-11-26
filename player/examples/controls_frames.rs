//! Frames of Karaoke mode as the window draws it on the GPU (the canvas
//! with its cells, the stage over it, the cover's ground under it), to
//! look at START's pill and the controls' bar: the mode just entered
//! (START alone), a song being sung (its lane, its progress along the
//! bottom) and a song's result (the diploma, START under it), each with
//! the controls hidden, fading in a quarter, half and three quarters of
//! the way (the stage's progress going in the first half, the bar coming
//! in the second) and shown (the pointer near the bottom). Saved as PNGs
//! into `OUT_DIR` (`<scene>-<controls>.png`, written whole).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example controls_frames -- OUT_DIR [W H]
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use jodel_output::{SampleFormat, StreamFormat, StreamInfo};
use jodel_playback::{
    choose_route, Codec, Container, Event, FileFacts, PlayState, Tags, TrackInfo,
};
use jodeldiplom::art::Art;
use jodeldiplom::config::Config;
use jodeldiplom::core::karaoke::controls::FADE_MS;
use jodeldiplom::core::karaoke::SongLyrics;
use jodeldiplom::core::{update, Action, NowTrack, Screen, State};
use jodeldiplom::gui::backdrop::{Cue, Visualiser};
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::Tint;
use jodeldiplom::gui::{font_px, Canvas};
use jodeldiplom::lyrics::Answer;
use jodeldiplom::spectrum::SpectrumFrame;
use jodeldiplom::view::Scene;

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
    let settings = state.engine_settings();
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
fn cover() -> Art {
    let image = image::RgbImage::from_fn(256, 256, |x, y| {
        let t = (x + y) as f32 / 510.0;
        let (dx, dy) = (x as f32 - 150.0, y as f32 - 110.0);
        if dx * dx + dy * dy < 60.0 * 60.0 {
            return image::Rgb([250, 230, 200]);
        }
        let (a, b) = ([120.0, 30.0, 30.0], [200.0, 120.0, 40.0]);
        image::Rgb([0, 1, 2].map(|k| (a[k] * (1.0 - t) + b[k] * t) as u8))
    });
    Art {
        id: 7,
        image: Arc::new(image),
    }
}

/// Karaoke mode at `size` cells, two songs queued, the first held for
/// START (just entered), its lyrics and its notes ready.
fn entered(size: (u16, u16)) -> State {
    let mut state = State::new("/music".into(), Config::default(), size);
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
fn singing(size: (u16, u16)) -> State {
    let mut state = entered(size);
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
fn result(size: (u16, u16)) -> State {
    let mut state = singing(size);
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
/// not brought up), the pointer having come near the bottom.
fn controls(state: &mut State, quarters: Option<u64>) {
    let Some(quarters) = quarters else {
        return;
    };
    let row = state.size.1 - 1;
    update(state, Action::PointerMoved { row });
    let at = state.now_ms + FADE_MS * quarters / 4;
    update(state, Action::Tick(at));
}

/// A scene's state, made for a size in cells.
type Make = fn((u16, u16)) -> State;

fn main() {
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(args.first().expect("OUT_DIR"));
    let (w, h) = match (
        args.get(1).and_then(|a| a.parse().ok()),
        args.get(2).and_then(|a| a.parse().ok()),
    ) {
        (Some(w), Some(h)) => (w, h),
        _ => (1600u32, 900u32),
    };
    let mut gpu = Offscreen::new(false).expect("a GPU");
    let art = cover();
    let scenes: [(&str, Make, f64); 3] = [
        ("entered", entered, 0.0),
        ("singing", singing, HEARD),
        ("result", result, HEARD),
    ];
    for (name, make, heard) in scenes {
        let fades = [
            ("hidden", None),
            ("quarter", Some(1)),
            ("half", Some(2)),
            ("three-quarters", Some(3)),
            ("shown", Some(4)),
        ];
        for (how, quarters) in fades {
            let mut canvas = Canvas::new(80, 24, font_px(1.0));
            canvas.set_gpu(true);
            let grid = canvas.fit(w, h);
            let mut state = make(grid);
            controls(&mut state, quarters);
            let spectrum = SpectrumFrame::default();
            let scene = Scene {
                state: &state,
                spectrum: &spectrum,
                pointer: None,
                heard,
            };
            canvas.draw(&scene, Some(&art));
            let stage = canvas.stage_frame((&state, None), heard, 1.0 / 60.0, &Tint::STAGE);
            let (pixels, pw, ph) = canvas.pixels();
            let (pw, ph) = (pw as u32, ph as u32);
            let cue = Cue {
                art: Some(art.clone()),
                layers: None,
                heard: Default::default(),
            };
            let ground = Visualiser::default().step(&cue, (pw, ph), 1.0 / 60.0);
            gpu.set_backdrop(Some(&ground), pw, ph);
            gpu.set_stage(stage.as_ref(), pw, ph);
            let shown = gpu.show_pixels(pixels, pw, ph).expect("a frame");
            let img = image::RgbImage::from_fn(pw, ph, |x, y| {
                let p = shown[(y * pw + x) as usize];
                image::Rgb([(p >> 16) as u8, (p >> 8) as u8, p as u8])
            });
            let file = out.join(format!("{name}-{how}.png"));
            let mut png = std::io::Cursor::new(Vec::new());
            img.write_to(&mut png, image::ImageFormat::Png).unwrap();
            durable::write_atomic(&file, png.get_ref()).unwrap();
            println!("{}", file.display());
        }
    }
}
