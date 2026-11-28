//! The karaoke game's sounds: the song's frames a time of it is heard on
//! ([`Playing::frame_at`]), what they are cued on (a moment at a time of
//! the singing, plus how late the voice comes) and a stem's change on a
//! bar, through the last position told and the output's speed through the
//! track; the conductor's game played live by the desk's synth, heard on
//! the device where it landed, called off by a seek and silent with the
//! notes hidden; and the result's
//! reveal heard after the song on the device kept open, through the whole
//! player on a memory device.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::mixer::{Effects, Fader, Mixer, MixerOptions, Role, Start, Timebase};
use jodel_output::{
    Capabilities, OpenRequest, OutputTarget, SampleFormat, SimClock, SimTime, StreamFormat,
};
use jodel_playback::{Captures, MemoryTarget, PlayState};
use jodel_synth::Synth;
use jodeldiplom::app::{App, AppClock, AppOptions, MemoryOutput, Output};
use jodeldiplom::core::karaoke::result::Beat;
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::shine::Shone;
use jodeldiplom::core::{karaoke, Action, Rhythm, Screen, Side, State};
use jodeldiplom::core::{Anchor, HeadFrame, Playing};
use jodeldiplom::sfx::conductor::{Clock, Conductor, Diploma, Event, Sheet, Tempo};
use jodeldiplom::sfx::{Desk, Sfx};
use jodeldiplom::tiers::Tier;

/// The longest a step of a test on the test's time waits for the player's
/// threads (a guard: it fails the test only where they hang).
const GUARD: Duration = Duration::from_secs(60);

/// Returns once the mixer whose clock `timebase` reads waits for `device`
/// ([`Timebase::waits_for`]): nothing more is mixed until its time passes.
fn mixer_waits(device: &SimTime, timebase: impl Fn() -> Timebase) {
    let guard = Instant::now() + GUARD;
    loop {
        let t = timebase();
        if t.waits_for(device) {
            return;
        }
        let queued = (device.position(), device.queued(), device.capacity());
        assert!(
            Instant::now() < guard,
            "the mixer fills the device: {t:?}, (played, queued, of) {queued:?}"
        );
        std::thread::yield_now();
    }
}

/// The time of the device `captures` opened first (on the mixer's thread,
/// as the mixer opens).
fn device_time(captures: &Captures) -> SimTime {
    let guard = Instant::now() + GUARD;
    loop {
        if let Some(c) = captures.list().first() {
            return c.time.clone().expect("a device on the test's time");
        }
        assert!(Instant::now() < guard, "the device opened");
        std::thread::yield_now();
    }
}

/// A device on the test's time, a mixer on it, ticked a period at a time.
struct Ticks<'a> {
    mixer: &'a Mixer,
    time: SimTime,
}

impl Ticks<'_> {
    /// A period of the device's time passes; the mixer fills it again.
    fn tick(&self) {
        mixer_waits(&self.time, || self.mixer.timebase());
        self.time.advance(u64::from(RATE) / 100).unwrap();
        mixer_waits(&self.time, || self.mixer.timebase());
    }

    /// Writes `samples` to `out` and drains it, ticking as the ring of
    /// `out` (on the mixer) has room.
    fn play(&self, out: &mut dyn jodel_output::Output, samples: &[i32]) {
        let mut done = 0;
        while done < samples.len() {
            done += 2 * out.write(&samples[done..], Some(Duration::ZERO)).unwrap();
            self.tick();
        }
        let written = out.written();
        out.drain(Some(Duration::ZERO)).unwrap();
        let guard = Instant::now() + GUARD;
        while out.position().unwrap() < written {
            assert!(Instant::now() < guard, "drained");
            self.tick();
        }
    }
}

/// The window's frames on the test's time: each one 1/60 s of the app's
/// clock and as much played by the device ([`SimClock::Manual`]), the
/// mixer let fill it again before the frame is drawn. However busy the
/// machine, what the device plays is what the frames cued, where they
/// cued it.
struct Window {
    clock: AppClock,
    time: SimTime,
}

impl Window {
    /// The device frames of a window's frame (1/60 s).
    const FRAME: u64 = RATE as u64 / 60;

    /// A frame: its time passes (the device plays it), the mixer fills
    /// the device again, the app takes what came and draws (cues the
    /// game's sounds); the desk's frame they were cued on.
    fn frame(&mut self, app: &mut App) -> usize {
        let timebase = |a: &App| a.desk().unwrap().master.timebase;
        mixer_waits(&self.time, || timebase(app));
        self.clock.step(Duration::from_nanos(
            Self::FRAME * 1_000_000_000 / u64::from(RATE),
        ));
        self.time.advance(Self::FRAME).unwrap();
        mixer_waits(&self.time, || timebase(app));
        app.pump();
        let cued = timebase(app).frame as usize;
        app.heard_for_frame();
        cued
    }

    /// Frames until `done`; whether it came (within [`GUARD`]).
    fn frames_until(&mut self, app: &mut App, mut done: impl FnMut(&App) -> bool) -> bool {
        let guard = Instant::now() + GUARD;
        while !done(app) {
            if Instant::now() >= guard {
                return false;
            }
            self.frame(app);
        }
        true
    }
}

/// `sound` of the sheet alone, as the desk's synth makes it (on a
/// conductor of its own, offline), its first `span` frames: its left side.
/// It is cued `phase` frames into a block of the synth's
/// ([`jodel_synth::BLOCK`]): the synth takes its modulation once a block,
/// so a sound sounds a little otherwise from each place in one.
fn made(sound: Sheet, phase: usize, span: usize) -> Vec<f32> {
    let (mut c, mut synth) = Conductor::new(RATE, &jodeldiplom::theme::CLASSIC);
    let sheet = Diploma::Sheet {
        sound,
        gain: 1.0,
        pan: 0.0,
    };
    synth.render_to(&mut vec![0.0f32; 2 * phase]);
    c.event(&mut synth, phase as u64, Event::Diploma(sheet));
    let mut out = vec![0.0f32; 2 * span];
    synth.render_to(&mut out);
    out.into_iter().step_by(2).collect()
}

/// How well `made` lines up with `left` from `from` (normalized
/// correlation, 1: the same up to a gain).
fn fit_at(left: &[f32], from: usize, made: &[f32]) -> f64 {
    let got = &left[from..from + made.len()];
    let dot: f64 = got
        .iter()
        .zip(made)
        .map(|(&g, &m)| f64::from(g) * f64::from(m))
        .sum();
    let energy = |x: &[f32]| x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    dot / (energy(made) * energy(got)).sqrt().max(1e-12)
}

/// Where `sound` (its left side) is heard in `left` (the device's left
/// channel, full scale 1) near the device frames `near` (from the first
/// to the last, within 60 ms): the frame its first `span` frames line up
/// best from, and how well (normalized correlation, 1: alone and whole,
/// against the sound as the synth makes it from the same place in its
/// block), against how well at its best 200 ms after (where it is not).
fn heard_near(left: &[f32], sound: Sheet, near: (usize, usize), span: usize) -> (usize, f64, f64) {
    let reach = RATE as usize * 6 / 100;
    let best = |(first, last): (usize, usize), made: &[f32]| {
        (first.saturating_sub(reach)..last + reach)
            .map(|f| (f, fit_at(left, f, made)))
            .fold((0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b })
    };
    // Found with the sound from a block's start, then made from where in
    // a block it was heard to start.
    let (found, _) = best(near, &made(sound, 0, span));
    let (frame, fit, made) = (0..jodel_synth::BLOCK)
        .map(|phase| {
            let m = made(sound, phase, span);
            let (f, fit) = (found.saturating_sub(1)..=found + 1)
                .map(|f| (f, fit_at(left, f, &m)))
                .fold((0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b });
            (f, fit, m)
        })
        .fold(
            (0, f64::MIN, Vec::new()),
            |b, x| if x.1 > b.1 { x } else { b },
        );
    let after = near.1 + RATE as usize / 5;
    let (_, away) = best((after, after), &made);
    (frame, fit, away)
}

const RATE: u32 = 48_000;
/// A beat at 120 a minute, in frames.
const BEAT: u64 = 24_000;
/// The desk's samples.
const FORMAT: SampleFormat = SampleFormat::S24In32Low;

/// The left and the right side of what `captures`' first stream played,
/// full scale 1.
fn sides(captures: &Captures) -> (Vec<f32>, Vec<f32>) {
    let streams = captures.list();
    let bytes = streams[0].sink.bytes();
    let words: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
        .map(|b| ((i32::from_le_bytes(*b) << 8) >> 8) as f32 / 8_388_608.0)
        .collect();
    let left = words.iter().step_by(2).copied().collect();
    let right = words.iter().skip(1).step_by(2).copied().collect();
    (left, right)
}

/// A song at 120 in 4/4 from 0 s, sure: its beat grid.
fn steady() -> jodel_rhythm::Analysis {
    let beats: Vec<f64> = (0..400).map(|k| 0.5 * f64::from(k)).collect();
    jodel_rhythm::Analysis {
        downbeats: (0..400).step_by(4).collect(),
        local_bpm: vec![120.0; beats.len()],
        regions: vec![jodel_rhythm::Region {
            start: 0.0,
            end: 200.0,
            confidence: 0.9,
        }],
        beats,
        ..jodel_rhythm::Analysis::default()
    }
}

/// A conductor on that grid (from stream frame 0) in A minor, and its
/// synth.
fn conductor() -> (Conductor, Synth) {
    let (mut c, mut synth) = Conductor::new(RATE, &jodeldiplom::theme::CLASSIC);
    let clock = Clock::from_analysis(&steady(), 0.0, 0, f64::from(RATE), [None; Clock::LINES]);
    Tempo::default().time(&mut c, &mut synth, clock);
    c.set_key(Some(&jodel_rhythm::key::Key::on(9, true)), 0);
    (c, synth)
}

/// A desk on a memory device playing in real time (on `clock`: the wall
/// clock's, or the test's): the player's strip with a Lead input, the
/// effects input playing `synth`; what the device plays.
fn live_desk(
    synth: Synth,
    clock: SimClock,
) -> (
    Mixer,
    Fader,
    jodel_output::mixer::InputTarget,
    Effects,
    Captures,
) {
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, FORMAT));
    let target = MemoryTarget::new(caps.clone(), clock);
    let captures = target.captures();
    let options = MixerOptions {
        capabilities: Some(caps),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(target), options);
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    effects.live(synth);
    let (lead, fader) = mixer.input(&player, Role::Lead, 1.0);
    (mixer, fader, lead, effects, captures)
}

/// The song's frames from `from`, `n` of them: its left side counting them
/// (from 1 000 on, so none is silent), its right silent.
fn counting(from: u64, n: u64) -> Vec<i32> {
    (from..from + n)
        .flat_map(|k| [(k + 1_000) as i32, 0])
        .collect()
}

/// How well `got` from `at` lines up with `want` from `from` over `span`
/// frames (normalized correlation, 1: the same up to a gain).
fn fit(got: &[f32], at: usize, want: &[f32], from: usize, span: usize) -> f64 {
    let (g, w) = (&got[at..at + span], &want[from..from + span]);
    let dot: f64 = g
        .iter()
        .zip(w)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let energy = |x: &[f32]| x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    dot / (energy(g) * energy(w)).sqrt().max(1e-12)
}

/// Where in `got` (near `near`, within 10 ms) `want`'s `span` frames from
/// `from` line up best: the frame, and how well; and how well at their
/// best 200 ms away (where they are not).
fn found(got: &[f32], near: usize, want: &[f32], from: usize, span: usize) -> (usize, f64, f64) {
    let best = |c: usize, reach: usize| {
        (c.saturating_sub(reach)..c + reach)
            .map(|k| (k, fit(got, k, want, from, span)))
            .fold((0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b })
    };
    let (k, f) = best(near, 480);
    let (_, away) = best(near + RATE as usize / 5, 480);
    (k, f, away)
}

/// The game as the conductor plays it live through the desk, on a memory
/// device in real time on the test's time ([`Ticks`]): words popping up
/// the ladder to the first rung (the last reaching it, its stinger
/// landing on the next beat), a phrase of the top tier (its chime on the
/// beat) and its jackpot (on the downbeat after). The same events played
/// offline on a twin conductor are what each must sound like: each moment
/// is heard on the device where it landed, on the device's frame for the
/// stream frame it was planned on (and the synth's look-ahead after it),
/// to the frame.
#[test]
fn the_game_is_heard_where_it_lands() {
    let (mut live, synth) = conductor();
    let latency = synth.latency();
    let (mixer, fader, mut lead, effects, captures) = live_desk(synth, SimClock::Manual);
    let ticks = Ticks {
        mixer: &mixer,
        time: device_time(&captures),
    };
    let (mut twin, mut offline) = conductor();
    // The words a sixteenth apart, the rung's on frame 66 000.
    let n = Tier::ALL[0].spec().streak;
    let mut events: Vec<(u64, Event)> = (0..n)
        .map(|k| {
            let pan = -0.5 + k as f32 / n as f32;
            (
                66_000 - 6_000 * u64::from(n - 1 - k),
                Event::Word {
                    streak: k + 1,
                    pan,
                    size: 0.0,
                },
            )
        })
        .collect();
    events.push((110_000, Event::Phrase(Tier::TOP)));
    events.push((110_000, Event::Jackpot));
    // The song's output open (no frames given yet: the notes wait for its
    // first).
    let request = OpenRequest::new(RATE, 2)
        .formats(&[FORMAT])
        .prefill(Duration::ZERO);
    let mut out = lead.open(&request).unwrap();
    let mut landed = Vec::new();
    {
        let mut out = effects.commands(Some(&fader));
        for &(at, e) in &events {
            landed.push(live.event(&mut out, at, e));
            twin.event(&mut offline, at, e);
        }
    }
    assert_eq!(live.dropped(), 0);
    // The pop at once, the rung's hit on the beat after its word, the
    // phrase's chime on the beat, the jackpot on the downbeat (bars of four
    // from frame 0).
    let n = n as usize;
    assert_eq!(landed[0], events[0].0);
    assert_eq!(landed[n - 1], 3 * BEAT);
    assert_eq!(landed[n], 5 * BEAT);
    assert_eq!(landed[n + 1], 8 * BEAT);
    let total = 13 * BEAT;
    ticks.play(&mut *out, &counting(0, total));
    // Where stream frame 0 was mixed: the device's frame for it.
    let Start::At { time, .. } = fader.at_stream(0) else {
        panic!("the song was mixed");
    };
    let zero = (time.as_secs_f64() * f64::from(RATE)).round() as usize;
    let mut want = vec![0.0f32; 2 * total as usize];
    offline.render_to(&mut want);
    let want: Vec<f32> = want.iter().skip(1).step_by(2).copied().collect();
    let (left, right) = sides(&captures);
    assert!(
        left[zero..].iter().take(1_000).all(|&v| v > 0.0),
        "the song from {zero}"
    );
    // Each moment's first 50 ms, as it sounds where it landed (the synth's
    // look-ahead after).
    assert!(mixer.desk().master.stream.is_some(), "the device open");
    for (k, &at) in landed.iter().enumerate().skip(n - 1) {
        let from = at as usize + latency;
        let (heard, how, away) = found(&right, zero + from, &want, from, 2_400);
        let late = heard as i64 - (zero + from) as i64;
        assert!(
            how > 0.99 && away < how - 0.4,
            "moment {k}: {how} ({away} away)"
        );
        assert!(late.abs() <= 2, "moment {k}: {late} frames late");
        println!("moment {k} at {at}: heard {late} frames after it, fit {how:.3} ({away:.3} away)");
    }
    let pop = 42_000 + latency;
    let (heard, how, _) = found(&right, zero + pop, &want, pop, 1_200);
    assert!(how > 0.99, "the first pop: {how}");
    assert!(heard.abs_diff(zero + pop) <= 2, "{heard}");
    let view = effects.view();
    assert_eq!((view.missed, view.live), (0, RATE), "{view:?}");
    assert!(view.commands > 50, "{view:?}");
}

/// The Synth tab's sounds in the theme it compares, through the effects
/// as the window plays them on the desk (a frame drawn, then the click):
/// a word and a phrase's chime (atoms a theme overrides) heard first in
/// Classic, then, the tab switched to Spooky, again: each time as a synth
/// of the theme's own makes it, not as the other's, on a memory device on
/// the test's time ([`Ticks`]).
#[test]
fn the_synth_tab_plays_the_theme_it_compares() {
    use jodeldiplom::core::synth_tab::Bench;
    use jodeldiplom::sfx::bench::{self, What, SOUNDS};
    use jodeldiplom::sfx::conductor::Atom;
    use jodeldiplom::theme::{Theme, CLASSIC, SPOOKY};
    let (_, idle) = Conductor::new(RATE, &CLASSIC);
    let (mixer, fader, _lead, effects, captures) = live_desk(idle, SimClock::Manual);
    let ticks = Ticks {
        mixer: &mixer,
        time: device_time(&captures),
    };
    let key = jodel_rhythm::key::Key::on(9, true);
    let mut state = common::playing_state(120, 36);
    state.side = Side::Synth;
    state.screen = Screen::Main;
    let mut sfx = Sfx::default();
    let desk = || Desk {
        effects: &effects,
        heads: [&fader, &fader],
        timebase: mixer.timebase(),
    };
    // A word (its pop and its body) and a phrase's chime: each of atoms
    // Spooky overrides alone. (A sound that mixes them with others, the
    // payout's coins under its brass, tells the two apart less by its
    // samples: the synth's shared hall and chorus go on from the sounds
    // before, so no synth made anew sounds quite the same; the
    // conductor's test holds that every atom plays its theme's patch.)
    let overridden = |a: &Atom| SPOOKY.sounds.iter().any(|(b, _)| b == a);
    let sounds: Vec<What> = ["A word", "Good"]
        .map(|name| {
            let s = (SOUNDS.iter()).find(|s| s.name == name && s.atoms.iter().all(overridden));
            s.expect(name).what
        })
        .into();
    // Each once the last has rung out and the synth rests.
    const GAP: usize = 3 * RATE as usize;
    let mut played: Vec<(&'static Theme, What, usize)> = Vec::new();
    for theme in [&CLASSIC, &SPOOKY] {
        state.synth.theme = Some(theme);
        for &what in &sounds {
            sfx.frame(&state, desk());
            ticks.tick();
            let at = mixer.timebase().frame as usize;
            let ask = Bench::Play {
                what,
                key: Some(key),
                bpm: 120.0,
                theme,
            };
            sfx.bench(&state, desk(), ask);
            for _ in 0..GAP / 480 {
                ticks.tick();
            }
            played.push((theme, what, at));
        }
    }
    // Played on past the last.
    for _ in 0..50 {
        ticks.tick();
    }
    let other = |theme: &Theme| -> &'static Theme {
        if *theme == CLASSIC {
            &SPOOKY
        } else {
            &CLASSIC
        }
    };
    let latency = Conductor::new(RATE, &CLASSIC).1.latency();
    // Sound `k` of those played as the bench plays it in `theme` (its
    // player made anew for each sound) on a synth of its own, `phase`
    // frames into a block of its own (the synth takes its modulation once
    // a block, so a sound sounds a little otherwise from each place in
    // one): its left side, its first frame the one heard as it is cued.
    let made = |k: usize, theme: &'static Theme, phase: usize| {
        let (_, what, at) = played[k];
        let (_, mut synth) = Conductor::new(RATE, &CLASSIC);
        synth.command(jodel_synth::Command::Tempo { bpm: 120.0 });
        let mut out = Shifted {
            synth: &mut synth,
            by: (at - phase) as u64,
        };
        let play = (theme, Some(&key), 120.0);
        bench::Bench::new(RATE).play(&mut out, at as u64, what, play);
        let mut both = vec![0.0f32; 2 * (phase + latency + SPAN)];
        synth.render_to(&mut both);
        both.into_iter()
            .step_by(2)
            .skip(phase + latency)
            .collect::<Vec<f32>>()
    };
    let (left, _) = sides(&captures);
    let left = &left[..];
    for (k, &(theme, what, at)) in played.iter().enumerate() {
        let at = at + latency;
        // Found with the sound from a block's start, then made from where
        // in a block it was heard to start.
        let heard = |theme: &'static Theme| {
            let (near, _, _) = found(left, at, &made(k, theme, 0), 0, SPAN / 8);
            let mut best = f64::MIN;
            for phase in 0..jodel_synth::BLOCK {
                let m = made(k, theme, phase);
                for f in near - 1..=near + 1 {
                    best = best.max(fit(left, f, &m, 0, SPAN));
                }
            }
            best
        };
        let (own, others) = (heard(theme), heard(other(theme)));
        let other = other(theme).name;
        println!("{} {what:?}: {own:.3}, as {other}: {others:.3}", theme.name);
        assert!(
            own > 0.7 && others < own - 0.4,
            "{} {what:?}: {own} as itself, {others} as {other}",
            theme.name
        );
    }
}

/// How long of each sound the Synth tab's test compares, frames.
const SPAN: usize = RATE as usize / 2;

/// The synth is made with every theme's patches as the state's book has
/// them: with nothing edited nothing is told it (the game sounds as it is
/// made, bit for bit), an edit is told it once as it plays, the patch
/// alone, and so is one set back; a theme changed in the settings is told
/// it nothing (its patches are in the synth already).
#[test]
fn the_synth_is_told_a_patch_once_when_it_changes() {
    use jodeldiplom::sfx::conductor::Atom;
    use jodeldiplom::theme::{CLASSIC, SPOOKY};
    let (_, idle) = Conductor::new(RATE, &CLASSIC);
    let (mixer, fader, _lead, effects, captures) = live_desk(idle, SimClock::Manual);
    let ticks = Ticks {
        mixer: &mixer,
        time: device_time(&captures),
    };
    let mut state = common::playing_state(120, 36);
    let mut sfx = Sfx::default();
    let desk = || Desk {
        effects: &effects,
        heads: [&fader, &fader],
        timebase: mixer.timebase(),
    };
    // Each frame drawn, then the mixer let take what it was sent.
    let mut frame = |state: &State| {
        sfx.frame(state, desk());
        for _ in 0..3 {
            ticks.tick();
        }
        effects.view().commands
    };
    frame(&state);
    let made = frame(&state);
    assert_eq!(frame(&state), made, "nothing edited: nothing told");
    state.config.karaoke_theme = &SPOOKY;
    assert_eq!(frame(&state), made, "another theme: nothing told");
    state.patches.edit(&CLASSIC, Atom::Pop, |p| p.level += 6.0);
    assert_eq!(frame(&state), made + 1, "an edit: once");
    assert_eq!(frame(&state), made + 1);
    state.patches.revert(&CLASSIC, Atom::Pop);
    assert_eq!(frame(&state), made + 2, "set back: once again");
    assert_eq!(effects.view().missed, 0);
}

/// Commands to `synth`, their frames `by` earlier.
struct Shifted<'a> {
    synth: &'a mut Synth,
    by: u64,
}

impl jodel_synth::Commands for Shifted<'_> {
    fn send(&mut self, c: jodel_synth::Command) -> bool {
        let c = match c.frame() {
            Some(f) => c.on(f - self.by),
            None => c,
        };
        self.synth.send(c)
    }
}

/// A seek: what the game planned and the desk's synth waits to play (a
/// phrase's chime and its jackpot, a second ahead) never sounds once the
/// song's output is flushed; what the player plans after it does.
#[test]
fn a_seek_calls_off_what_the_game_planned() {
    let _guard = jodel_testkit::real_time();
    let (mut c, synth) = conductor();
    // In real time on the wall clock's: the seek's flush gives the device
    // what is mixed first, which a device on the test's time would take
    // only as the test let its time pass.
    let (_mixer, fader, mut lead, effects, captures) = live_desk(synth, SimClock::Realtime);
    let request = OpenRequest::new(RATE, 2)
        .formats(&[FORMAT])
        .prefill(Duration::ZERO);
    let mut out = lead.open(&request).unwrap();
    out.write(&counting(0, 2 * BEAT), None).unwrap();
    let mixed = fader.mixed_to().expect("mixed");
    {
        let mut cmds = effects.commands(Some(&fader));
        c.event(&mut cmds, mixed + BEAT, Event::Phrase(Tier::TOP));
        c.event(&mut cmds, mixed + BEAT, Event::Jackpot);
    }
    let taken = || effects.view().commands > 40;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !taken() {
        assert!(Instant::now() < deadline, "taken into the synth");
        std::thread::sleep(Duration::from_millis(1));
    }
    // The seek, and the player's own call to call it off, as the game's
    // sounds do when the song's place changes.
    out.flush().unwrap();
    c.cancel(&mut effects.commands(Some(&fader)), 0);
    out.write(&counting(0, 6 * BEAT), None).unwrap();
    out.drain(None).unwrap();
    let (_, right) = sides(&captures);
    let loud = right.iter().filter(|v| v.abs() > 1e-4).count();
    assert_eq!(loud, 0, "nothing of it heard");
    // After the seek, a word: heard.
    c.event(
        &mut effects.commands(Some(&fader)),
        fader.mixed_to().unwrap() + 4_800,
        Event::Word {
            streak: 1,
            pan: 0.0,
            size: 0.0,
        },
    );
    out.write(&counting(6 * BEAT, BEAT), None).unwrap();
    out.drain(None).unwrap();
    let (_, right) = sides(&captures);
    assert!(right.iter().any(|v| v.abs() > 1e-3), "the word after");
}

/// A song sung to its end in Karaoke mode, the game on, through the whole
/// player on a memory device playing in real time, on the test's time
/// ([`Window`]): the device is opened once, as the player starts, and
/// stays open; the song over and its output closed, the result's reveal
/// is heard on it, its paper's rustle from its first beat and the
/// examiner's pen from its last, each where it was cued.
#[test]
fn the_result_is_heard_after_the_song() {
    let dir = jodel_testkit::TempDir::new("reveal-heard");
    let song = dir.join("a.wav");
    common::write_wav(&song, RATE, 3.0);
    durable::write_atomic(&dir.join("a.lrc"), b"la la la\nla la\n").unwrap();
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Manual);
    let mut options = AppOptions::new(dir.to_path_buf());
    let clock = AppClock::stepped();
    options.clock = clock.clone();
    options.output = Output::Memory(memory.clone());
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Ok(jodel_output::input::Input::feed(48_000, 240).0));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    // The beats by the flux alone, of the mix: the networks' inference
    // (the beat network's, and the separator's where there is a GPU, as on
    // the macOS runners), unoptimised, would take the processors from the
    // mixer as it plays in real time on a small machine.
    options.config.beat_voting = false;
    options.config.beat_method = jodel_rhythm::Method::Flux;
    options.config.stems = false;
    options.autoplay = Some(song.clone());
    let mut app = common::fitted_app(options, 120, 36);
    // The device, opened as the player starts.
    let engine = memory.captures().last().expect("an engine").clone();
    let mut window = Window {
        clock,
        time: device_time(&engine),
    };
    let playing = |a: &App| a.state.playing.state == PlayState::Playing;
    assert!(window.frames_until(&mut app, playing));
    // Its frames drawn as it plays, the song analysed before it is sung.
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(
        window.frames_until(&mut app, |a| karaoke::not_singable(&a.state, 0).is_none()
            && matches!(a.state.rhythm.get(&song), Some(Rhythm::Ready(_))))
    );
    // Played out before it is sung: Next then plays it from its start,
    // not by a seek. (A seek's flush gives the device what the mixer
    // holds first, waiting for room in its ring; this device makes room
    // only as the frames go by, and this thread is in the app then.)
    let stopped = |a: &App| a.state.playing.state == PlayState::Stopped;
    assert!(window.frames_until(&mut app, stopped), "played out");
    app.dispatch(Action::Op(Op::KaraokeEnter));
    app.dispatch(Action::Op(Op::KaraokeNext));
    let started = |a: &App| a.state.karaoke.countdown.is_none() && playing(a);
    assert!(window.frames_until(&mut app, started), "Next");
    // To its end, and the reveal after it drawn frame by frame, where the
    // mixer is as its first beat and its last are cued (the mixer waits
    // for the device, which waits for the next frame: nothing is mixed
    // meanwhile, so before and after the frame's sounds are cued are the
    // same frame).
    let mixing = |a: &App| a.desk().unwrap().master.timebase.frame as usize;
    // The signature's beat is reached on the frame whose reveal time
    // passes its end from the frame before's, as the sounds read it.
    let (mut paper, mut signed, mut until, mut before) = (None, None, None, None);
    let mut frame = 0;
    let guard = Instant::now() + GUARD;
    while until.is_none_or(|u| frame < u) {
        assert!(Instant::now() < guard, "the song ends");
        frame += 1;
        let cueing = window.frame(&mut app);
        let Some((shown, after)) = karaoke::result_now(&app.state) else {
            continue;
        };
        let cued = (cueing, mixing(&app));
        let (end, settled) = (
            shown.reveal.when(Beat::End).unwrap(),
            shown.reveal.settled(),
        );
        if paper.is_none() {
            paper = Some(cued);
            until = Some(frame + ((settled + 1.0) * 60.0) as u64);
        }
        if signed.is_none() && after >= end && before.is_some_and(|t| t < end) {
            signed = Some(cued);
        }
        before = Some(after);
    }
    let (paper, signed) = (paper.unwrap(), signed.expect("signed"));
    assert_eq!(paper.0, paper.1, "nothing mixed as the paper was cued");
    let desk = app.desk().unwrap();
    assert_eq!(desk.master.timebase.epoch, 1, "never opened again");
    app.shutdown();
    let streams = memory.captures().last().expect("a device").list();
    assert_eq!(streams.len(), 1, "the device opened once");
    let left: Vec<f32> = streams[0]
        .sink
        .bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .step_by(2)
        .map(|b| ((i32::from_le_bytes(*b) << 8) >> 8) as f32 / 8_388_608.0)
        .collect();
    // The paper's rustle, alone before the title lands (its first 0.1 s),
    // and the pen's scratch (its first 0.3 s, a lullaby over it): each
    // heard where it was cued, and not elsewhere.
    let rustle = heard_near(&left, Sheet::Unroll, paper, 4_000);
    println!("the paper cued {paper:?}: {rustle:?}");
    assert!(rustle.1 > 0.8 && rustle.2 < 0.3, "the paper: {rustle:?}");
    let pen = heard_near(&left, Sheet::Signature, signed, 14_400);
    println!("the signature cued {signed:?}: {pen:?}");
    assert!(
        pen.1 > 0.12 && pen.1 > 8.0 * pen.2,
        "the signature: {pen:?}"
    );
}

/// With the notes hidden the game's bubbles are not shown, so its sounds
/// are not played: a phrase sung to the top tier (its chime and its
/// jackpot) sends nothing to the desk's synth, the device mixing on; the
/// notes shown again, it sounds.
#[test]
fn the_game_is_silent_with_its_notes_hidden() {
    let _guard = jodel_testkit::real_time();
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, FORMAT));
    let target = MemoryTarget::new(caps.clone(), SimClock::Realtime);
    let options = MixerOptions {
        capabilities: Some(caps),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(target), options);
    let effects = mixer.effects(&mixer.strip("Effects"));
    let (_lead, fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let until = |done: &dyn Fn() -> bool, what: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    until(&|| mixer.timebase().rate == RATE, "the device open");
    let mut state = common::playing_state(120, 36);
    state.screen = Screen::Karaoke;
    state.config.karaoke_game = true;
    state.config.karaoke_notes = false;
    state.karaoke.singing = common::bubble_song().0;
    // A phrase sung now, to the top tier.
    let sung = |state: &mut State| {
        let at = state.karaoke.lag.sung_clock(state.heard());
        let shone = Shone {
            line: 0,
            end: at,
            value: 1.0,
            top: 0.9,
            tier: Some(Tier::TOP),
        };
        state.karaoke.singing.decided(at, shone);
    };
    let mut sfx = Sfx::default();
    let mut frame = |state: &State| {
        let desk = Desk {
            effects: &effects,
            heads: [&fader, &fader],
            timebase: mixer.timebase(),
        };
        sfx.frame(state, desk);
    };
    // A frame drawn (the game's moments read from now on), the phrase
    // sung, then frames for a fifth of a second, the desk mixing them.
    frame(&state);
    sung(&mut state);
    let start = mixer.timebase().frame;
    for _ in 0..12 {
        frame(&state);
        std::thread::sleep(Duration::from_millis(16));
    }
    until(
        &|| mixer.timebase().frame > start + u64::from(RATE) / 5,
        "mixed on",
    );
    let view = effects.view();
    assert_eq!(view.commands, 0, "{view:?}");
    assert_eq!(view.live, RATE, "the synth there to play: {view:?}");
    // The notes shown: the next phrase's chime and its jackpot.
    state.config.karaoke_notes = true;
    frame(&state);
    sung(&mut state);
    frame(&state);
    until(&|| effects.view().commands > 0, "the phrase played");
}

/// A result of a song's preparation found sounds a soft pluck on the
/// desk while no music plays (the track stopped or paused); with a track
/// playing, or a preview on, none at all.
#[test]
fn a_found_result_sounds_only_with_no_music() {
    use jodeldiplom::core::{update, Heard};
    let _guard = jodel_testkit::real_time();
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, FORMAT));
    let target = MemoryTarget::new(caps.clone(), SimClock::Realtime);
    let options = MixerOptions {
        capabilities: Some(caps),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(target), options);
    let effects = mixer.effects(&mixer.strip("Effects"));
    let (_lead, fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let until = |done: &dyn Fn() -> bool, what: &str| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    until(&|| mixer.timebase().rate == RATE, "the device open");
    let mut state = common::playing_state(120, 36);
    let song = state.queue.paths[1].clone();
    let mut sfx = Sfx::default();
    let mut frame = |state: &State| {
        let desk = Desk {
            effects: &effects,
            heads: [&fader, &fader],
            timebase: mixer.timebase(),
        };
        sfx.frame(state, desk);
    };
    // A key found now, as its worker tells it.
    let found = |state: &mut State| {
        update(
            state,
            Action::Heard {
                path: song.clone(),
                part: Heard::Key(Some(jodel_rhythm::key::Key::on(9, true))),
            },
        );
    };
    // Frames for a fifth of a second: what the effects took by then.
    let frames = |frame: &mut dyn FnMut(&State), state: &State| {
        let start = mixer.timebase().frame;
        for _ in 0..12 {
            frame(state);
            std::thread::sleep(Duration::from_millis(16));
        }
        until(
            &|| mixer.timebase().frame > start + u64::from(RATE) / 5,
            "mixed on",
        );
        effects.view().commands
    };
    // A track playing: none.
    assert_eq!(state.playing.state, PlayState::Playing);
    frame(&state);
    found(&mut state);
    assert_eq!(frames(&mut frame, &state), 0, "a track playing");
    // A preview on, the track stopped: none.
    state.playing.state = PlayState::Stopped;
    state.preview.on = Some(jodeldiplom::core::preview::On { path: None });
    found(&mut state);
    assert_eq!(frames(&mut frame, &state), 0, "a preview on");
    // No music: the pluck.
    state.preview.on = None;
    found(&mut state);
    frame(&state);
    until(&|| effects.view().commands > 0, "the found sound played");
    // Paused counts as no music too.
    let before = effects.view().commands;
    state.playing.state = PlayState::Paused;
    found(&mut state);
    assert!(frames(&mut frame, &state) > before, "paused");
}

#[test]
fn a_time_lands_on_the_frame_it_is_heard_on() {
    let mut state = common::playing_state(120, 36);
    // None before a position is told.
    assert_eq!(state.playing.frame_at(84.0), None);
    let index = state.playing.track.as_ref().unwrap().index;
    state.playing.anchor = Some(Anchor {
        index,
        at: HeadFrame {
            head: 1,
            frame: 1_000_000,
        },
        seconds: 83.0,
    });
    // Nor with one: the output's speed through the track is not known
    // before two.
    assert_eq!(state.playing.frame_at(84.0), None);
    // At its speed through the track once it is (a varispeed).
    state.playing.speed = Some(44_100.0 * 1.25);
    let at = |frame| Some(HeadFrame { head: 1, frame });
    assert_eq!(state.playing.frame_at(84.0), at(1_055_125));
    // Before the output's first frame: none.
    assert_eq!(state.playing.frame_at(0.0), None);
    // A position of another track: none.
    state.playing.anchor.as_mut().unwrap().index = index + 1;
    assert_eq!(state.playing.frame_at(84.0), None);
    // Forgotten (a seek, a jump): none, and counted, so what was worked
    // out from it is known to be off.
    let before = state.playing.anchors_forgotten;
    state.playing.forget_anchor();
    assert_eq!(state.playing.frame_at(84.0), None);
    assert_eq!(state.playing.anchors_forgotten, before + 1);
    assert_eq!(Playing::default().anchors_forgotten, 0);
}

/// The game's synth (its hall, its delay and its chorus) comes to rest
/// after its sounds: a phrase of the top tier and its jackpot played, and
/// their tails over, the device gets the song bit for bit again.
#[test]
fn the_desk_is_bit_for_bit_once_the_game_s_synth_rests() {
    let (mut c, synth) = conductor();
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, FORMAT));
    let target = MemoryTarget::new(caps.clone(), SimClock::Instant);
    let captures = target.captures();
    let options = MixerOptions {
        capabilities: Some(caps),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(target), options);
    let effects = mixer.effects(&mixer.strip("Effects"));
    effects.live(synth);
    let (mut lead, fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let request = OpenRequest::new(RATE, 2)
        .formats(&[FORMAT])
        .prefill(Duration::ZERO);
    let mut out = lead.open(&request).unwrap();
    {
        let mut cmds = effects.commands(Some(&fader));
        c.event(&mut cmds, BEAT, Event::Phrase(Tier::TOP));
        c.event(&mut cmds, BEAT, Event::Jackpot);
    }
    // The jackpot's three beats from its downbeat, then the synth's tail.
    let total = 30 * BEAT;
    out.write(&counting(0, total), None).unwrap();
    out.drain(None).unwrap();
    let Start::At { time, .. } = fader.at_stream(0) else {
        panic!("the song was mixed");
    };
    let zero = (time.as_secs_f64() * f64::from(RATE)).round() as usize;
    let words: Vec<i32> = {
        let bytes = captures.list()[0].sink.bytes();
        let all: Vec<i32> = (bytes.as_chunks::<4>().0.iter())
            .map(|b| (i32::from_le_bytes(*b) << 8) >> 8)
            .collect();
        all
    };
    let heard = |k: usize| (words[2 * (zero + k)], words[2 * (zero + k) + 1]);
    assert!(
        (BEAT as usize..2 * BEAT as usize).any(|k| heard(k).1 != 0),
        "it sounded"
    );
    let rest = 26 * BEAT as usize;
    let off = (rest..total as usize).find(|&k| heard(k) != ((k + 1_000) as i32, 0));
    assert_eq!(off, None, "{:?}", off.map(heard));
    assert!(mixer.passthrough(), "at rest");
}
