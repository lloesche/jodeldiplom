//! Test support: temporary music folders built from the PCM fixtures, a
//! generated WAV, a generated cover, and states with a track
//! "playing" for the view tests.

#![allow(
    clippy::disallowed_methods,
    reason = "tests: a music folder of fixture tracks, a cover and a stray file"
)]
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::{SampleFormat, StreamFormat, StreamInfo};
use jodel_playback::{
    any_capabilities, choose_route, Codec, Container, Cover, CoverOrigin, FileFacts, PlayState,
    Settings, Tags, TrackInfo,
};
use jodeldiplom::app::App;
use jodeldiplom::config::Config;
use jodeldiplom::core::{NowTrack, State};

/// A fixture from the repository's PCM fixtures (`tests/fixtures/pcm`).
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/pcm")
        .join(name)
}

/// The 44.1 kHz, 16-bit ramp (8,193 frames).
pub const F44: &str = "ramp-44100-16bit.flac";
/// The 48 kHz, 24-bit ramp (8,193 frames).
pub const F48: &str = "ramp-48000-24bit.flac";

/// A new empty folder under the system's temporary folder.
pub fn temp_dir(tag: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let dir = std::env::temp_dir().join(format!(
        "jodeldiplom-{tag}-{}-{n}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes a 24-bit stereo WAV of `seconds` at `rate`: a quiet chord.
pub fn write_wav(path: &Path, rate: u32, seconds: f64) {
    let frames = (f64::from(rate) * seconds) as u32;
    let data_len = frames * 6;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 6).to_le_bytes());
    out.extend_from_slice(&6u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..frames {
        let t = f64::from(i) / f64::from(rate);
        let x = [220.0, 277.18, 329.63, 3_000.0]
            .iter()
            .zip([0.20, 0.15, 0.12, 0.02])
            .map(|(hz, a)| a * (std::f64::consts::TAU * hz * t).sin())
            .sum::<f64>();
        let v = (x * 8_388_607.0 * 0.5) as i32;
        let b = v.to_le_bytes();
        out.extend_from_slice(&b[..3]);
        out.extend_from_slice(&b[..3]);
    }
    fs::write(path, out).unwrap();
}

/// A generated cover as PNG bytes.
pub fn cover_png(size: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(jodeldiplom::art::sample_cover(size))
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    bytes
}

/// A music folder:
///
/// ```text
/// root/
///   Archive/
///     Deep/Hidden Track.flac      (the 44.1 kHz ramp)
///   Synthetic Ramps/
///     01 Ramp 44.1.flac           (ramp)
///     02 Ramp 48.flac             (ramp)
///     03 Chord 96k.wav            (plain PCM, 3 s)
///     10 Ramp again.flac          (ramp)
///     cover.png
///     notes.txt                   (not playable, not listed)
///   Empty/
///   .hidden/
/// ```
pub struct Tree {
    /// The root.
    pub root: PathBuf,
    /// `Synthetic Ramps`.
    pub album: PathBuf,
    /// `Archive`.
    pub archive: PathBuf,
}

/// Builds the music folder under a new temporary folder.
pub fn music_tree(tag: &str) -> Tree {
    let root = temp_dir(tag);
    let album = root.join("Synthetic Ramps");
    let archive = root.join("Archive");
    fs::create_dir_all(archive.join("Deep")).unwrap();
    fs::create_dir_all(&album).unwrap();
    fs::create_dir_all(root.join("Empty")).unwrap();
    fs::create_dir_all(root.join(".hidden")).unwrap();
    fs::copy(fixture(F44), album.join("01 Ramp 44.1.flac")).unwrap();
    fs::copy(fixture(F48), album.join("02 Ramp 48.flac")).unwrap();
    write_wav(&album.join("03 Chord 96k.wav"), 96_000, 3.0);
    fs::copy(fixture(F44), album.join("10 Ramp again.flac")).unwrap();
    fs::write(album.join("cover.png"), cover_png(300)).unwrap();
    fs::write(album.join("notes.txt"), "not music").unwrap();
    fs::copy(fixture(F44), archive.join("Deep").join("Hidden Track.flac")).unwrap();
    Tree {
        root,
        album,
        archive,
    }
}

/// A track "playing" through the null output: a FLAC file at `rate` on
/// the route the policy picks with `settings` for a device that takes
/// everything.
pub fn now_track(rate: u32, settings: Settings, index: usize) -> NowTrack {
    let info = TrackInfo {
        path: PathBuf::from(format!("/music/Album/0{}.flac", index + 1)),
        container: Container::Flac,
        codec: Codec::Flac,
        rate,
        channels: 2,
        bits: 24,
        float: false,
        bitrate: None,
        frames: Some(u64::from(rate) * 245),
        tags: Tags {
            title: Some("Morning Light".into()),
            artist: Some("The Ramps".into()),
            album: Some("Synthetic Sessions".into()),
            track_number: Some("2".into()),
            ..Tags::default()
        },
        cover: Some(Cover {
            data: Arc::from(cover_png(64)),
            mime: "image/png".into(),
            origin: CoverOrigin::Embedded,
        }),
    };
    let route = choose_route(
        &FileFacts::from(&info),
        &any_capabilities(info.rate),
        &settings,
    )
    .unwrap();
    let output = StreamInfo {
        format: StreamFormat::new(route.output_rate, 2, SampleFormat::S24In32),
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
        device: "USB Audio DAC".into(),
    }
}

/// A state with a browser listing, a queue and a track playing at 1:23.
pub fn playing_state(cols: u16, rows: u16) -> State {
    use jodeldiplom::browser::{Details, Entry, EntryKind};
    let mut state = State::new("/music/Album".into(), Config::default(), (cols, rows));
    // No GPU here: the cues listen to the mix.
    state.gpu = jodeldiplom::core::Gpu::Absent;
    state.browser.loading = false;
    let mut entries = vec![Entry {
        name: "..".into(),
        path: "/music".into(),
        kind: EntryKind::Parent,
        relative: None,
    }];
    entries.push(Entry {
        name: "Bonus".into(),
        path: "/music/Album/Bonus".into(),
        kind: EntryKind::Dir,
        relative: None,
    });
    let titles = [
        "Opening",
        "Morning Light",
        "Undertow",
        "Glass Rooms",
        "Coda",
    ];
    for (i, title) in titles.iter().enumerate() {
        let path = PathBuf::from(format!("/music/Album/0{}.flac", i + 1));
        entries.push(Entry {
            name: format!("0{} {title}.flac", i + 1),
            path: path.clone(),
            kind: EntryKind::File,
            relative: None,
        });
        state.details.insert(
            path,
            Details {
                chip: if i == 4 {
                    "WAV 96".into()
                } else {
                    "FLAC 44.1".into()
                },
                title: Some(title.to_string()),
                artist: Some("The Ramps".into()),
                seconds: Some(180.0 + 37.0 * i as f64),
                error: None,
            },
        );
    }
    state.browser.entries = entries;
    state.browser.cursor = 3;
    state.queue.paths = (1..=5)
        .map(|i| PathBuf::from(format!("/music/Album/0{i}.flac")))
        .collect();
    state.queue.current = Some(1);
    state.playing.track = Some(now_track(44_100, state.config.settings(), 1));
    state.playing.state = PlayState::Playing;
    state.playing.position = 83.0;
    // Its beats: a steady 128 BPM, clear throughout.
    let beats: Vec<f64> = (0..520)
        .map(|i| 0.4 + f64::from(i) * 60.0 / 128.0)
        .collect();
    let mut analysis = jodel_rhythm::Analysis {
        duration: 245.0,
        regions: beats
            .chunks(16)
            .map(|c| jodel_rhythm::Region {
                start: c[0],
                end: c[c.len() - 1],
                confidence: 0.9,
            })
            .collect(),
        downbeats: (0..beats.len()).step_by(4).collect(),
        beats,
        confidence: 0.9,
        ..Default::default()
    };
    analysis.derive();
    let path = state.queue.paths[1].clone();
    state.rhythm_wanted = state.queue.paths[1..4].to_vec();
    state.rhythm_recipe = jodeldiplom::core::RhythmRecipe {
        methods: state.config.methods_in_force(),
        stems: false,
    };
    // The queue already asked to be got ready, from the playing track.
    let p = &state.queue.paths;
    let order = p[1..].iter().chain(&p[..1]).cloned().collect();
    state.prepared = (order, Some(state.rhythm_recipe.clone()));
    state.rhythm.insert(
        path,
        jodeldiplom::core::Rhythm::Ready(std::sync::Arc::new(jodeldiplom::core::Rhythmed::new(
            analysis, None,
        ))),
    );
    state
}

/// A generated waveform for `seconds` with beats at `bpm` from `first`:
/// the drums hit on each beat, the bass and the other instruments hold,
/// and the vocals come in at `vocals_from` seconds.
#[allow(dead_code)]
pub fn waveform(
    seconds: f64,
    bpm: f64,
    first: f64,
    vocals_from: f64,
) -> jodel_rhythm::waveform::Waveform {
    use jodel_rhythm::waveform::{Lane, Waveform, FPS};
    let frames = (seconds * FPS) as usize;
    let beat = 60.0 / bpm;
    let lane = |level: &dyn Fn(f64) -> u8| {
        let peak: Vec<u8> = (0..frames).map(|i| level(i as f64 / FPS)).collect();
        let rms = peak.iter().map(|&p| p.saturating_sub(30)).collect();
        Lane { peak, rms }
    };
    let hit = |t: f64| {
        let since = (t - first).rem_euclid(beat);
        if t < first {
            0
        } else {
            (230.0 - 900.0 * since).max(60.0) as u8
        }
    };
    let drums = lane(&hit);
    let bass = lane(&|t| if t < first { 0 } else { 170 });
    let other = lane(&|t| 120 + ((t * 0.7).sin() * 30.0) as u8);
    let vocals = lane(&|t| if t < vocals_from { 0 } else { 190 });
    let mix = lane(&|t| hit(t).max(200));
    Waveform {
        mix,
        stems: Some([bass, drums, other, vocals]),
        // The voice from `vocals_from`: a note a half second, up and down
        // a fifth from middle C, a breath between.
        melody: Some(jodel_rhythm::waveform::Melody {
            cents: (0..frames)
                .map(|i| {
                    let t = i as f64 / FPS;
                    let k = ((t - vocals_from) * 2.0).floor() as i64;
                    if t < vocals_from || (t * 2.0).fract() > 0.85 {
                        0
                    } else {
                        (6000 + 100 * [0, 2, 4, 5, 7, 5, 4, 2][k.rem_euclid(8) as usize]) as u16
                    }
                })
                .collect(),
            sure: vec![230; frames],
        }),
    }
}

/// [`playing_state`] on the Transition tab: the playing track and the next one
/// (100 BPM) analysed with stems and their waveforms.
#[allow(dead_code)]
pub fn transition_state(cols: u16, rows: u16) -> State {
    use jodeldiplom::core::{Rhythm, Rhythmed, Side};
    let mut state = playing_state(cols, rows);
    state.side = Side::Transition;
    state.gpu = jodeldiplom::core::Gpu::Present("Test GPU".into());
    state.config.stems = true;
    state.rhythm_recipe.stems = true;
    state.prepared.1 = Some(state.rhythm_recipe.clone());
    let now = state.queue.paths[1].clone();
    let Some(Rhythm::Ready(r)) = state.rhythm.get(&now).cloned() else {
        panic!("the playing track is analysed");
    };
    let mut a = (*r.analysis).clone();
    a.stems = true;
    state.rhythm.insert(
        now,
        Rhythm::Ready(std::sync::Arc::new(Rhythmed::new(
            a,
            Some(waveform(245.0, 128.0, 0.4, 60.0)),
        ))),
    );
    let beats: Vec<f64> = (0..400).map(|i| 2.0 + f64::from(i) * 0.6).collect();
    let mut next = jodel_rhythm::Analysis {
        duration: 252.0,
        downbeats: (0..beats.len()).step_by(4).collect(),
        beats,
        confidence: 0.9,
        stems: true,
        ..Default::default()
    };
    next.derive();
    state.rhythm.insert(
        state.queue.paths[2].clone(),
        Rhythm::Ready(std::sync::Arc::new(Rhythmed::new(
            next,
            Some(waveform(252.0, 100.0, 2.0, 30.0)),
        ))),
    );
    // The tab's lyrics asked for already (as the first update would).
    state.karaoke.lyrics_wanted = jodeldiplom::core::karaoke::lyrics_wanted(&state);
    // The engine told of the stems for the tab (as the first update would).
    state.stems_sent.make = jodeldiplom::core::stems::make_wanted(&state);
    state
}

/// Words timed on the playing and the next track, one every `gap` seconds
/// from 5 s to 120 s, each sung for 0.3 s ("Mister rock and roll, take me
/// home" over and over).
#[allow(dead_code)]
pub fn sung_every(state: &mut State, gap: f64) {
    use jodel_lyrics::{Line, Lyrics, Source, Timing, Word};
    use jodeldiplom::core::karaoke::SongLyrics;
    let texts = ["Mister", "rock", "and", "roll,", "take", "me", "home"];
    let count = (115.0 / gap) as usize;
    let words: Vec<Word> = (0..count)
        .map(|k| {
            let start = 5.0 + gap * k as f64;
            Word {
                start: Some(start),
                end: Some(start + 0.3),
                ..Word::new(texts[k % texts.len()])
            }
        })
        .collect();
    let lyrics = std::sync::Arc::new(Lyrics {
        lines: words
            .chunks(7)
            .map(|w| Line {
                words: w.to_vec(),
                ..Line::default()
            })
            .collect(),
        source: Source::Unknown,
        timing: Timing {
            detail: jodel_lyrics::Detail::Words,
            by: Some(jodel_lyrics::Maker::new("ctc:test", 2)),
            fit: None,
        },
    });
    for index in [1, 2] {
        let path = state.queue.paths[index].clone();
        state.karaoke.lyrics.insert(
            path,
            SongLyrics::Found {
                lyrics: std::sync::Arc::clone(&lyrics),
                shown: std::sync::Arc::clone(&lyrics),
            },
        );
    }
}

/// The next track replaced by one at `bpm` from 2 s, clear throughout.
#[allow(dead_code)]
pub fn next_at(state: &mut State, bpm: f64) {
    use jodeldiplom::core::{Rhythm, Rhythmed};
    let beats: Vec<f64> = (0..600).map(|i| 2.0 + f64::from(i) * 60.0 / bpm).collect();
    let mut next = jodel_rhythm::Analysis {
        duration: beats[beats.len() - 1] + 1.0,
        regions: beats
            .chunks(16)
            .map(|c| jodel_rhythm::Region {
                start: c[0],
                end: c[c.len() - 1],
                confidence: 0.9,
            })
            .collect(),
        downbeats: (0..beats.len()).step_by(4).collect(),
        beats,
        confidence: 0.9,
        stems: true,
        ..Default::default()
    };
    next.derive();
    state.rhythm.insert(
        state.queue.paths[2].clone(),
        Rhythm::Ready(std::sync::Arc::new(Rhythmed::new(next, None))),
    );
}

/// Pumps `app` until `done` holds or `timeout` passes; returns whether it
/// held.
pub fn wait_for(app: &mut App, timeout: Duration, mut done: impl FnMut(&App) -> bool) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if done(app) {
            return true;
        }
        app.pump_wait(Duration::from_millis(20));
    }
    done(app)
}

/// A linear peak for `db` dBFS.
fn lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Fills the Mixer tab's desk as a frame's look would (at the state's
/// clock): the player at `rate` with its meters at −12 / −9 dBFS, its
/// stems there and heard with the vocals muted, a preview crossfading (A
/// going out, B coming in), a microphone open and hearing with its echo
/// canceller learnt, and the master summing (a preview mixes and the stems
/// play, so it is not bit-perfect).
pub fn fill_desk(state: &mut State, rate: u32) {
    use jodel_output::lanes::Parts;
    use jodel_output::mixer::{
        DeskView, InputView, LaneView, MasterView, Meters, MicPort, MicView, PartsView, Role,
        StripView, Timebase,
    };
    use jodeldiplom::core::desk::{self, Look, StripKind};
    use jodeldiplom::core::mic::MicReading;
    let format = StreamFormat::new(rate, 2, SampleFormat::S24In32);
    let input = |role, gain, queued| InputView {
        role,
        gain,
        heard: gain > 0.0,
        lanes: Vec::new(),
        parts: None,
        format: Some(format),
        queued,
        capacity: 65_536,
        dropped: 0,
        automation: None,
    };
    // The player's lanes: main (switched out), then the stems, the vocals
    // muted.
    let lane = |strip, gain: f64| LaneView {
        strip,
        gain,
        heard: gain > 0.0,
    };
    let lead = InputView {
        lanes: vec![
            lane(1, 0.0),
            lane(3, 1.0),
            lane(4, 1.0),
            lane(5, 0.0),
            lane(6, 1.0),
        ],
        parts: Some(PartsView {
            state: Parts::Present,
            share: 1.0,
        }),
        ..input(Role::Lead, 1.0, 40_000)
    };
    let stem = |id, name: &str, mute| StripView {
        id,
        name: name.into(),
        parent: Some(1),
        level: 1.0,
        mute,
        solo: false,
        inputs: Vec::new(),
    };
    state.desk.strips[StripKind::Vocals.index()].mute = true;
    let view = DeskView {
        master: MasterView {
            device: "USB Audio DAC".into(),
            stream: Some(StreamInfo {
                format,
                backend: "WASAPI",
                device_buffer_frames: 960,
                ring_frames: 65_536,
                realtime: true,
            }),
            timebase: Timebase {
                epoch: 1,
                rate,
                frame: u64::from(rate) * 724,
                heard: u64::from(rate) * 7243 / 10,
            },
            passthrough: false,
            clipped: 0,
        },
        strips: vec![
            StripView {
                id: 1,
                name: "Player".into(),
                parent: None,
                level: 1.0,
                mute: false,
                solo: false,
                inputs: vec![lead],
            },
            stem(3, "Drums", false),
            stem(4, "Bass", false),
            stem(5, "Vocals", true),
            stem(6, "Other", false),
            StripView {
                id: 2,
                name: "Preview".into(),
                parent: None,
                level: 1.0,
                mute: false,
                solo: false,
                inputs: vec![
                    input(Role::Guest, 0.35, 20_000),
                    input(Role::Guest, 0.65, 52_000),
                ],
            },
        ],
        mic: MicView {
            port: MicPort::Hearing,
            refused: 0,
        },
    };
    let meters = Meters {
        master: vec![lin(-10.0), lin(-7.5)],
        strips: vec![
            (1, vec![lin(-12.0), lin(-9.0)]),
            (3, vec![lin(-16.0), lin(-14.0)]),
            (4, vec![lin(-19.0), lin(-21.0)]),
            (6, vec![lin(-18.0), lin(-15.0)]),
            (2, vec![lin(-20.0), lin(-17.0)]),
        ],
        limited: false,
    };
    let mic = MicReading {
        rate: 48_000,
        peak: lin(-20.0),
        voice: Some(lin(-27.0)),
        echo_on: true,
        removed_db: 31.0,
        learnt: true,
        delay: Some(0.042),
        drift_ppm: 18.0,
        fill_ms: 10.0,
        xruns: 0,
    };
    // Two frames: the first look after none takes no meters.
    state.mic = Some(mic);
    state.karaoke.mic.name = Some("Yeti Stereo Microphone".into());
    let now = state.now_ms;
    for at in [now.saturating_sub(16), now] {
        let ids = [1, 3, 4, 5, 6, 2, 7];
        desk::feed(state, Look::of(view.clone(), meters.clone(), ids), at);
    }
}

/// [`playing_state`] on the Mixer tab, its desk filled ([`fill_desk`]), the
/// microphone's strip selected (so it listens).
pub fn desk_state(cols: u16, rows: u16) -> State {
    let mut state = playing_state(cols, rows);
    state.side = jodeldiplom::core::Side::Mixer;
    state.desk.selected = jodeldiplom::core::desk::Channel::Mic;
    state.now_ms = 10_000;
    fill_desk(&mut state, 96_000);
    // The engine told of the stems as the desk has them (as the first
    // update would).
    state.stems_sent.make = jodeldiplom::core::stems::make_wanted(&state);
    state
}

/// Held by a test that plays on a device in real time (a simulated clock
/// that plays a period every period of wall-clock time) for as long as it
/// runs: one such test of a binary at a time. Side by side on a small,
/// loaded machine (a debug build, every test of a binary at once), their
/// threads would starve each other of the periods they must keep.
pub fn real_time() -> std::sync::RwLockWriteGuard<'static, ()> {
    REAL_TIME.write().unwrap_or_else(|e| e.into_inner())
}

static REAL_TIME: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// A line "la lo" to sing in Karaoke mode: "la" a second on C4, "lo" the
/// next on G4, and its notes.
pub fn bubble_song() -> (
    jodeldiplom::core::singing::Singing,
    Arc<jodel_lyrics::Lyrics>,
) {
    let word = |text: &str, s: f64| jodel_lyrics::Word {
        start: Some(s),
        end: Some(s + 1.0),
        confidence: 0.9,
        ..jodel_lyrics::Word::new(text)
    };
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![jodel_lyrics::Line {
            words: vec![word("la", 0.0), word("lo", 1.0)],
            ..jodel_lyrics::Line::default()
        }],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let mut melody = jodel_rhythm::waveform::Melody {
        cents: vec![0; 300],
        sure: vec![0; 300],
    };
    for i in 0..200 {
        melody.cents[i] = if i < 100 { 6000 } else { 6700 };
        melody.sure[i] = 255;
    }
    let mut singing = jodeldiplom::core::singing::Singing::default();
    singing.prepare(
        Path::new("a"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    (singing, lyrics)
}

/// `singing` of [`bubble_song`] hears "la" sung on its note for its first
/// 15 % (it counts, at 0.10 s), then nothing, a frame every 10 ms
/// up to `to` seconds (from `from`).
pub fn sing_bubbles(singing: &mut jodeldiplom::core::singing::Singing, from: f64, to: f64) {
    let mut i = (from * 100.0).round() as u32;
    while f64::from(i) / 100.0 <= to {
        let t = f64::from(i) / 100.0;
        singing.hear(t, (t < 0.15 - 1e-9).then_some(6000.0));
        i += 1;
    }
}

/// Karaoke mode's picture of [`bubble_song`] at `heard` (the game on, the
/// voice coming `late`) in `area`: its line being sung with its bubbles as
/// `singing` filled them.
pub fn bubble_picture(
    singing: &jodeldiplom::core::singing::Singing,
    lyrics: &jodel_lyrics::Lyrics,
    (heard, late): (f64, f64),
    area: ratatui::layout::Rect,
) -> jodeldiplom::view::karaoke::KaraokePicture {
    use jodeldiplom::view::karaoke::{sung_on, timed_lines, KaraokePicture, LineKind};
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut lines = timed_lines(&sung, heard).expect("timed lines");
    for l in lines.iter_mut().filter(|l| l.kind == LineKind::Current) {
        let notes = singing.notes.lines[0].clone();
        sung_on(singing, (true, late), 0, &notes, heard, l);
        l.notes = notes.notes;
    }
    KaraokePicture {
        area,
        bar: jodeldiplom::core::karaoke::controls::bar(area),
        lines,
        count: None,
        cue: None,
        controls: None,
        start: None,
        before: Vec::new(),
        moved: 1.0,
        score: None,
        result: None,
    }
}

/// The Karaoke tab shown, a track playing, a GPU for the stems, the voice
/// at 20 % of its travel (−35 dB), and every song's lyrics found (their words not timed: there is
/// no speech model), so each can be sung.
pub fn karaoke_tab() -> State {
    let mut state = transition_state(120, 40);
    state.side = jodeldiplom::core::Side::Karaoke;
    state.config.karaoke_voice = 20;
    singable(&mut state);
    state
}

/// Every song in the queue has its lyrics: found, as they are (kept for
/// whichever karaoke comes to want).
pub fn singable(state: &mut State) {
    use jodel_lyrics::{Lyrics, Source};
    use jodeldiplom::core::karaoke::SongLyrics;
    use jodeldiplom::lyrics::Answer;
    for path in state.queue.paths.clone() {
        let lyrics = Lyrics::parse_text("la la la\nla la", Source::Lrclib);
        state.karaoke.lyrics.insert(
            path,
            SongLyrics::from_answer(Answer::Found {
                lyrics,
                timed: None,
            }),
        );
    }
}
