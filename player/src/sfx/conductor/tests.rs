//! The conductor's rules on the song's grid (in frames: the beat, an
//! unsure grid, a downbeat with room before the singing; the rule itself
//! is the grid's, tested in `jodel_rhythm::grid`), the key from the
//! analysis's and its shift, the ladder landing on the tonic at every rung
//! (a jump in the run missing none), the tier table's rows growing their
//! sounds up the tiers, the held sounds and the payout stopping with the
//! streak, held sounds ending by themselves when their release is lost, no
//! key meaning no pitched layer, the most commands an event sends, the
//! bus never clipping, the sounds' levels (over a song at the reference
//! loudness, under its ceiling), the bus's level following the song's
//! loudness, and the announcer's call (landing with the slam, in key,
//! under the stinger, ducked under a sung line, only at the rows that
//! slam).

use jodel_rhythm::grid::When;
use jodel_rhythm::{Analysis, Region};
use jodel_synth::{Command, Commands, PatchId, Synth, Tag};

use super::palette::{Atom, BRASS, COINS, PADS};
use super::*;
use crate::core::karaoke::result::{Medal, SIGNING, STAMP_DOWN, UNROLL};
use crate::tiers::Tier;

const RATE: u32 = 48_000;
/// A second of the song in frames.
const SECOND: f64 = 48_000.0;
/// 120 beats a minute at 48 kHz.
const BEAT: u64 = 24_000;
const NONE: [Option<(u64, u64)>; Clock::LINES] = [None; Clock::LINES];

/// What was sent, kept.
#[derive(Default)]
struct Sent(Vec<Command>);

impl Commands for Sent {
    fn send(&mut self, c: Command) -> bool {
        self.0.push(c);
        true
    }
}

impl Sent {
    /// The notes sent: frame, patch, pitch, tag.
    fn notes(&self) -> Vec<(u64, PatchId, f32, Tag)> {
        self.0
            .iter()
            .filter_map(|c| match *c {
                Command::Note {
                    frame,
                    patch,
                    note,
                    tag,
                    ..
                } => Some((frame, patch, note.note, tag)),
                _ => None,
            })
            .collect()
    }

    /// The frames of the notes on `patch`.
    fn on(&self, patch: PatchId) -> Vec<u64> {
        self.notes()
            .iter()
            .filter(|n| n.1 == patch)
            .map(|n| n.0)
            .collect()
    }
}

/// A synth that loses every release and every cancel (a full queue).
struct Lossy<'a>(&'a mut Synth);

impl Commands for Lossy<'_> {
    fn send(&mut self, c: Command) -> bool {
        match c {
            Command::Release { .. } | Command::Cancel { .. } => false,
            c => self.0.command(c),
        }
    }
}

/// A queue that is full: it takes nothing.
struct Full;

impl Commands for Full {
    fn send(&mut self, _: Command) -> bool {
        false
    }
}

/// A song at 120 in 4/4 from 0 s, as sure as `sure` (its region's
/// confidence).
fn song(sure: f64) -> Analysis {
    let beats: Vec<f64> = (0..400).map(|k| 0.5 * f64::from(k)).collect();
    Analysis {
        downbeats: (0..400).step_by(4).collect(),
        local_bpm: vec![120.0; beats.len()],
        regions: vec![Region {
            start: 0.0,
            end: 200.0,
            confidence: sure,
        }],
        beats,
        ..Analysis::default()
    }
}

/// Its grid from frame 0 (a downbeat), the lines `lines`.
fn grid(lines: [Option<(u64, u64)>; Clock::LINES]) -> Clock {
    Clock::from_analysis(&song(0.9), 0.0, 0, SECOND, lines)
}

/// The same, unsure.
fn unsure(lines: [Option<(u64, u64)>; Clock::LINES]) -> Clock {
    Clock::from_analysis(&song(0.1), 0.0, 0, SECOND, lines)
}

/// A minor, as the analysis has it.
fn a_minor() -> jodel_rhythm::key::Key {
    jodel_rhythm::key::Key::on(9, true)
}

/// A conductor on the grid in A minor, and where its notes go.
fn conductor() -> (Conductor, Sent) {
    let (mut c, _) = Conductor::new(RATE, &crate::theme::CLASSIC);
    let mut out = Sent::default();
    c.set_clock(&mut out, grid(NONE));
    c.set_key(Some(&a_minor()), 0);
    out.0.clear();
    (c, out)
}

/// The same, playing on its synth.
fn playing() -> (Conductor, Synth) {
    let (mut c, mut s) = Conductor::new(RATE, &crate::theme::CLASSIC);
    c.set_clock(&mut s, grid(NONE));
    c.set_key(Some(&a_minor()), 0);
    (c, s)
}

/// The words in a row rung `r` (1 up) takes, the tier table's.
fn rung(r: usize) -> u32 {
    layers::rungs()[r - 1]
}

fn word(streak: u32) -> Event {
    Event::Word {
        streak,
        pan: 0.0,
        size: 0.0,
    }
}

fn line(a: u64, b: u64) -> Option<(u64, u64)> {
    Some((a * BEAT, b * BEAT))
}

#[test]
fn a_hit_lands_on_the_song_s_grid_in_frames() {
    let c = grid(NONE);
    assert_eq!(c.land(15_000, When::Beat), BEAT);
    assert_eq!(
        c.land(11_000, When::Beat),
        11_000,
        "more than half a beat: now"
    );
    let length = (2 * BEAT) as f64;
    assert_eq!(c.land(30_000, When::Downbeat { length }), 4 * BEAT);
    let sung = grid([line(3, 6), line(10, 20), None, None]);
    assert_eq!(sung.land(30_000, When::Downbeat { length }), 8 * BEAT);
    let busy = grid([line(0, 40), line(40, 41), None, None]);
    assert_eq!(busy.land(30_000, When::Downbeat { length }), 44 * BEAT);
    let c = unsure(NONE);
    assert_eq!(c.land(15_000, When::Beat), 15_000);
    assert_eq!(c.land(15_000, When::Downbeat { length }), 15_000);
    // Heard from frame 100 000 at 0.25 s in: the same grid moved on.
    let later = Clock::from_analysis(&song(0.9), 0.25, 100_000, SECOND, NONE);
    assert_eq!(later.land(100_000, When::Beat), 112_000);
}

#[test]
fn the_key_is_the_analysis_s_shifted_as_heard() {
    let a = Pentatonic::of(&a_minor(), 0);
    assert_eq!(a.home(60), 69);
    let notes: Vec<f32> = (-1..=6).map(|d| a.note(d, 69)).collect();
    assert_eq!(notes, [67.0, 69.0, 72.0, 74.0, 76.0, 79.0, 81.0, 84.0]);
    assert_eq!((0..4).map(|k| a.chord(k)).collect::<Vec<_>>(), [0, 1, 3, 5]);
    assert_eq!(Pentatonic::of(&a_minor(), 3).home(60), 60, "C minor");
    let c_major = jodel_rhythm::key::Key::on(0, false);
    assert_eq!(Pentatonic::of(&c_major, -2).home(60), 70, "B♭ major");
    let c = Pentatonic::of(&c_major, 0);
    let chord: Vec<f32> = (0..4).map(|k| c.note(c.chord(k), 60)).collect();
    assert_eq!(chord, [60.0, 64.0, 67.0, 72.0]);
}

#[test]
fn the_ladder_climbs_and_lands_on_the_tonic_at_every_rung() {
    let table = layers::rungs();
    for rungs in [&table[..], &[3, 7], &[4, 8, 12, 16, 20, 24, 30]] {
        let mut last = 0;
        for streak in 1..=rungs[rungs.len() - 1] + 40 {
            let step = ladder::step(streak, rungs);
            if rungs.contains(&streak) {
                assert_eq!(step.degree, key::DEGREES, "{rungs:?} {streak}");
            } else if last != key::DEGREES {
                assert!(step.degree >= last, "climbs: {rungs:?} {streak}");
            }
            let reached = rungs.iter().filter(|&&r| r <= streak).count();
            assert_eq!(step.rung as usize, reached);
            last = step.degree;
        }
    }
    // Up to the first rung: the pops climbing, its word on the octave.
    let first = rung(1);
    let (mut c, mut out) = conductor();
    for streak in 1..=first {
        c.event(&mut out, u64::from(streak), word(streak));
    }
    let pop = c.palette[Atom::Pop];
    let pops: Vec<f32> = out
        .notes()
        .iter()
        .filter(|n| n.1 == pop)
        .map(|n| n.2)
        .collect();
    assert_eq!(pops.len(), first as usize);
    assert_eq!(
        pops[pops.len() - 1],
        93.0,
        "A6, an octave over the climb's A5"
    );
    assert!(pops.windows(2).all(|w| w[1] >= w[0]) && pops[0] < 93.0);
}

/// A word told a step lower than the rung reached (a word rescued after
/// those past it) keeps the rung: nothing is let go or called off, and the
/// next word on reaches no rung again.
#[test]
fn a_lower_step_keeps_the_rung_only_the_streak_lost_lets_it_go() {
    let (mut c, mut out) = conductor();
    c.event(&mut out, 15_000, word(rung(1)));
    assert_eq!(c.rung, 1);
    out.0.clear();
    c.event(&mut out, 30_000, word(rung(1) - 1));
    c.event(&mut out, 36_000, word(rung(1) + 1));
    assert_eq!(c.rung, 1);
    let called_off = out.0.iter().any(|c| matches!(c, Command::Cancel { .. }));
    assert!(!called_off, "{:?}", out.0);
    let plucks = out.on(c.palette[Atom::Pluck]);
    assert!(plucks.is_empty(), "no stinger again: {plucks:?}");
    c.event(&mut out, 40_000, Event::Lost(Tier::ALL[0]));
    assert_eq!(c.rung, 0);
}

#[test]
fn a_stinger_picks_up_now_and_hits_on_the_beat_and_a_jump_misses_none() {
    let (mut c, mut out) = conductor();
    assert_eq!(c.event(&mut out, 15_000, word(rung(1))), BEAT);
    let plucks = out.on(c.palette[Atom::Pluck]);
    assert_eq!(plucks[0], 15_000, "the pickup at once");
    assert!(plucks.contains(&BEAT), "the hit on the beat");
    assert!(
        plucks.contains(&(BEAT + BEAT / 4)),
        "the arpeggio in sixteenths"
    );
    let tonic = out.notes().into_iter().find(|n| n.0 == BEAT).unwrap();
    assert_eq!(tonic.2 as i32 % 12, 9);
    // The hit under 80 ms away: no pickup.
    let (mut c, mut out) = conductor();
    c.event(&mut out, BEAT - 2_000, word(rung(1)));
    assert_eq!(out.on(c.palette[Atom::Pluck])[0], BEAT);
    // Unsure: the hit at once.
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, unsure(NONE));
    assert_eq!(c.event(&mut out, 15_000, word(rung(1))), 15_000);
    // A run that jumps past two rungs: the higher one's stinger.
    let (mut c, mut out) = conductor();
    c.event(&mut out, 1_000, word(rung(1) - 1));
    assert!(out.on(c.palette[Atom::Chime]).is_empty());
    c.event(&mut out, 15_000, word(rung(2) + 1));
    assert_eq!(c.rung, 2);
    assert!(
        out.on(c.palette[Atom::Chime]).contains(&BEAT),
        "rung 2's bell"
    );
}

#[test]
fn the_payout_and_the_jackpot_land_with_room_their_whole_length() {
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, grid([line(0, 7), line(14, 30), None, None]));
    let d = c.event(&mut out, BEAT / 2, Event::Jackpot);
    assert_eq!(
        d,
        8 * BEAT,
        "three beats of room from the downbeat in the gap"
    );
    assert!(out.on(c.palette[Atom::SlamKick]).contains(&d));
    // The payout after the top rung's stinger: its lead would be sung
    // over, so none; the rest of it all in the gap after the lines.
    let lines = [line(0, 3), line(5, 7), line(14, 40), None];
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, grid(lines));
    c.event(&mut out, BEAT / 2, word(rung(5)));
    assert!(out.on(c.palette[Atom::Whoosh]).is_empty());
    let payout: Vec<u64> = out
        .notes()
        .iter()
        .filter(|n| n.3 == sounds::PAYOUT)
        .map(|n| n.0)
        .collect();
    assert!(!payout.is_empty());
    let sung = |f: u64| lines.iter().flatten().any(|&(a, b)| f >= a && f < b);
    assert!(payout.iter().all(|&f| !sung(f) && f >= 40 * BEAT));
    // With the beat before it clear: led into by the whoosh.
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, grid([line(0, 3), line(5, 7), line(14, 30), None]));
    c.event(&mut out, BEAT / 2, word(rung(5)));
    assert_eq!(out.on(c.palette[Atom::Whoosh]), [32 * BEAT - BEAT]);
}

/// The voices of `atoms` sounding in `s`.
fn sounding(c: &Conductor, s: &Synth, atoms: &[Atom]) -> usize {
    let patches: Vec<PatchId> = atoms.iter().map(|&a| c.palette[a]).collect();
    s.voices().filter(|v| patches.contains(&v.patch)).count()
}

/// `frames` of `s` rendered.
fn render(s: &mut Synth, frames: usize) {
    let mut out = vec![0.0; 2 * frames];
    s.render_to(&mut out);
}

#[test]
fn a_streak_lost_lets_the_held_sounds_go_and_calls_off_the_payout() {
    let held = [Atom::Riser, PADS[0], PADS[1]];
    let (mut c, mut s) = playing();
    c.event(&mut s, 1_000, word(rung(4)));
    c.event(&mut s, 2_000, word(rung(5) - 1));
    render(&mut s, 24_000);
    assert_eq!(sounding(&c, &s, &held), 3, "the riser and the shimmer");
    let now = s.frame();
    c.event(&mut s, now, Event::Lost(Tier::named("Amazing")));
    render(&mut s, 9_600);
    assert_eq!(sounding(&c, &s, &held), 0, "gone in 0.2 s");
    // ×32 reached, then lost before its payout: none of it sounds.
    let (mut c, mut s) = playing();
    c.event(&mut s, 1_000, word(rung(5)));
    c.event(&mut s, 2_000, Event::Lost(Tier::TOP));
    for _ in 0..80 {
        render(&mut s, 4_800);
        assert_eq!(sounding(&c, &s, &[Atom::Whoosh, Atom::Low]), 0);
    }
}

#[test]
fn the_riser_ends_by_itself_when_no_rung_comes() {
    let (mut c, mut s) = playing();
    c.event(&mut s, 0, word(rung(4)));
    c.event(&mut s, 1_000, word(rung(5) - 1));
    render(&mut s, (2 * BEAT) as usize);
    assert_eq!(sounding(&c, &s, &[Atom::Riser]), 1);
    render(&mut s, (BEAT + 9_600) as usize);
    assert_eq!(
        sounding(&c, &s, &[Atom::Riser]),
        0,
        "three beats and its release"
    );
}

#[test]
fn lost_releases_cost_only_the_held_sounds_own_time() {
    let (mut c, mut s) = playing();
    let mut lossy = Lossy(&mut s);
    c.event(&mut lossy, 0, word(rung(4)));
    for k in 1..=6 {
        c.event(&mut lossy, k * BEAT / 2, word(rung(4) + k as u32));
    }
    c.event(
        &mut lossy,
        3 * BEAT + 1_000,
        Event::Lost(Tier::named("Amazing")),
    );
    assert!(
        c.dropped() >= 4,
        "the cancels and releases: {}",
        c.dropped()
    );
    render(&mut s, (4 * BEAT) as usize);
    assert_eq!(sounding(&c, &s, &PADS), 2, "held on past the lost release");
    // Two bars from the last word that held it on (on beat 3), and its
    // release.
    render(&mut s, (8 * BEAT) as usize);
    assert_eq!(sounding(&c, &s, &PADS), 0, "ended by itself");
}

#[test]
fn the_rows_grow_their_sounds_up_the_tiers() {
    let count = |event: Event| {
        let (mut c, mut out) = conductor();
        c.event(&mut out, 15_000, event);
        out.notes().len()
    };
    let chimes: Vec<usize> = Tier::ALL.map(|t| count(Event::Phrase(t))).to_vec();
    assert!(chimes.windows(2).all(|w| w[1] > w[0]), "{chimes:?}");
    let stings: Vec<usize> = layers::rungs().map(|r| count(word(r))).to_vec();
    assert!(stings.windows(2).all(|w| w[1] > w[0]), "{stings:?}");
    // The points under the first tier: the smallest chime there is.
    let none = count(Event::Diploma(Diploma::Points(None)));
    assert!(none <= chimes[0], "{none} {chimes:?}");
}

#[test]
fn no_event_sends_more_than_the_most_commands() {
    let events = [
        word(rung(5)),
        word(rung(4)),
        Event::Lost(Tier::TOP),
        Event::Phrase(Tier::TOP),
        Event::Jackpot,
        Event::Zap { pan: 0.0 },
        Event::Diploma(Diploma::Tick { k: 11, of: 12 }),
        Event::Diploma(Diploma::Gem {
            tier: Some(Tier::TOP),
            pan: 0.3,
        }),
        Event::Diploma(Diploma::Points(Some(Tier::TOP))),
        Event::Diploma(Diploma::Rung(Tier::TOP)),
    ]
    .into_iter()
    .chain(Sheet::ALL.map(|sound| {
        Event::Diploma(Diploma::Sheet {
            sound,
            gain: 1.0,
            pan: 0.0,
        })
    }));
    let mut most = 0;
    for e in events {
        let (mut c, mut out) = conductor();
        // At the top already, the riser sounding: the most a word does.
        c.event(&mut out, 0, word(rung(5) - 1));
        out.0.clear();
        c.event(&mut out, 15_000, e);
        most = most.max(out.0.len());
    }
    assert!(most <= MOST_COMMANDS, "{most}");
    assert!(
        most > MOST_COMMANDS / 2,
        "the bound is near the most: {most}"
    );
}

#[test]
fn clinks_come_on_high_rungs_softer_when_thick_and_never_over_a_stinger() {
    let (mut c, mut out) = conductor();
    let clinks = |c: &Conductor, out: &Sent| {
        out.0
            .iter()
            .filter_map(|cmd| match *cmd {
                Command::Note { patch, note, .. }
                    if COINS.iter().any(|&a| c.palette[a] == patch) =>
                {
                    Some(note.velocity)
                }
                _ => None,
            })
            .collect::<Vec<f32>>()
    };
    c.event(&mut out, 0, word(rung(2)));
    assert!(clinks(&c, &out).is_empty(), "×4: none");
    // ×8 reached (its stinger's ka-ching rattling coins of its own).
    c.event(&mut out, BEAT, word(rung(3)));
    out.0.clear();
    c.event(&mut out, 10 * BEAT, word(rung(3) + 2));
    c.event(&mut out, 10 * BEAT + 2_400, word(rung(3) + 3));
    let v = clinks(&c, &out);
    assert_eq!(v.len(), 4, "two clinks and their bounces");
    assert!(v[2] < v[0], "the second, close after, softer");
    out.0.clear();
    c.event(&mut out, 20 * BEAT, word(rung(4)));
    let before = clinks(&c, &out).len();
    c.event(&mut out, 20 * BEAT + 1_000, word(rung(4) + 1));
    assert_eq!(clinks(&c, &out).len(), before, "none over the stinger");
}

#[test]
fn with_no_key_nothing_pitched_low_plays_and_the_pops_sit_high() {
    let (mut c, _) = Conductor::new(RATE, &crate::theme::CLASSIC);
    let mut out = Sent::default();
    c.set_clock(&mut out, grid(NONE));
    let top = rung(Tier::COUNT);
    for streak in 1..=top {
        c.event(&mut out, u64::from(streak) * 6_000, word(streak));
    }
    let after = u64::from(top + 10) * 6_000;
    for e in [
        Event::Phrase(Tier::TOP),
        Event::Jackpot,
        Event::Lost(Tier::TOP),
    ] {
        c.event(&mut out, after, e);
    }
    c.event(
        &mut out,
        after,
        Event::Diploma(Diploma::Tick { k: 3, of: 12 }),
    );
    let pitched = [
        Atom::Pluck,
        BRASS[0],
        BRASS[1],
        Atom::Choir,
        PADS[0],
        PADS[1],
        Atom::Low,
        Atom::Pip,
        Atom::Wah,
        Atom::PopBody,
        Atom::Singer,
        Atom::Burst,
        Atom::Hiss,
    ]
    .map(|a| c.palette[a]);
    assert!(out.notes().iter().all(|n| !pitched.contains(&n.1)));
    let pop = c.palette[Atom::Pop];
    assert!(out
        .notes()
        .iter()
        .filter(|n| n.1 == pop)
        .all(|n| n.2 >= 96.0));
}

#[test]
fn a_seek_calls_everything_off_and_lets_go() {
    let (mut c, mut out) = conductor();
    c.event(&mut out, 1_000, word(rung(4)));
    c.event(&mut out, 2_000, word(rung(5) - 1));
    out.0.clear();
    c.cancel(&mut out, 3_000);
    assert_eq!(out.0[0], Command::Cancel { tag: None });
    let released = |a: Atom| {
        out.0.contains(&Command::Release {
            frame: 3_000,
            patch: c.palette[a],
            note: None,
            tag: Tag::NONE,
        })
    };
    assert!(released(Atom::Riser) && released(PADS[0]) && released(PADS[1]));
    assert!(!c.riser && c.rung == 0);
}

#[test]
fn the_tempo_goes_to_the_synth_with_the_clock() {
    let (mut c, _) = Conductor::new(RATE, &crate::theme::CLASSIC);
    let mut out = Sent::default();
    let mut fast = song(0.9);
    fast.local_bpm = vec![150.0; fast.beats.len()];
    c.set_clock(&mut out, Clock::from_analysis(&fast, 0.0, 0, SECOND, NONE));
    assert_eq!(out.0, [Command::Tempo { bpm: 150.0 }]);
    c.set_clock(
        &mut out,
        Clock::from_analysis(&fast, 1.0, 48_000, SECOND, NONE),
    );
    assert_eq!(out.0.len(), 1, "told once");
}

#[test]
fn a_pile_of_everything_never_clips_through_the_bus() {
    let (mut c, mut s) = playing();
    let mut peak = 0.0f32;
    let mut out = vec![0.0; 2 * 960];
    let ceiling = 10f32.powf(-1.0 / 20.0);
    for block in 0..500u64 {
        let now = s.frame();
        if block % 25 == 0 {
            for r in layers::rungs() {
                c.event(&mut s, now, word(r));
            }
            for t in Tier::ALL {
                c.event(&mut s, now, Event::Phrase(t));
            }
            c.event(&mut s, now, Event::Jackpot);
            for k in 0..30 {
                c.event(&mut s, now, Event::Diploma(Diploma::Tick { k, of: 30 }));
            }
            c.event(&mut s, now, Event::Zap { pan: 0.5 });
            c.event(&mut s, now, Event::Lost(Tier::TOP));
        }
        c.event(&mut s, now, word(30 + block as u32 % 4));
        s.render_to(&mut out);
        assert!(out.iter().all(|x| x.is_finite()));
        peak = out.iter().fold(peak, |m, x| m.max(x.abs()));
    }
    assert!(peak > 0.3, "heard: {peak}");
    assert!(peak <= ceiling + 1e-3, "peak {peak} over -1 dBFS");
}

/// The peak of `events` on frame 1 000 through a fresh synth, dBFS.
fn peak_db(events: &[Event]) -> f32 {
    let (mut c, mut s) = playing();
    for &e in events {
        c.event(&mut s, 1_000, e);
    }
    let mut out = vec![0.0; 2 * RATE as usize * 6];
    s.render_to(&mut out);
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    20.0 * peak.max(1e-9).log10()
}

#[test]
fn the_sounds_sit_on_top_of_a_song_under_the_ceiling() {
    // Over a song at the reference loudness: a word's pop clearly heard
    // yet short; the big ones bold, a few dB under full scale; none over
    // the limiter's ceiling.
    for streak in 1..=4 {
        let pop = peak_db(&[word(streak)]);
        assert!((-13.0..=-10.0).contains(&pop), "a pop at {pop:.1} dBFS");
    }
    let top = rung(Tier::COUNT);
    let big = [
        ("the top phrase's chime", vec![Event::Phrase(Tier::TOP)]),
        (
            "the jackpot",
            vec![Event::Phrase(Tier::TOP), Event::Jackpot],
        ),
        ("the top rung's stinger", vec![word(top - 1), word(top)]),
    ];
    for (what, events) in big {
        let peak = peak_db(&events);
        assert!(
            (-8.0..=-1.0 + 1e-3).contains(&peak),
            "{what} at {peak:.1} dBFS"
        );
    }
}

/// The song's loudness sets the synth's bus level, before its limiter: a
/// loud song's higher than a quiet one's, told once each, none as at the
/// reference; a pile-up at the highest level still under the ceiling.
#[test]
fn the_bus_level_follows_the_song_s_loudness() {
    let (mut c, mut out) = conductor();
    c.set_loudness(&mut out, None);
    assert!(out.0.is_empty(), "as at the reference: nothing to tell");
    c.set_loudness(&mut out, Some(-8.0));
    c.set_loudness(&mut out, Some(-8.0));
    let loud = calibration::trim(Some(-8.0));
    assert_eq!(out.0, [Command::Level { db: loud }]);
    assert_eq!(c.trim(), loud);
    c.set_loudness(&mut out, Some(-22.0));
    let quiet = calibration::trim(Some(-22.0));
    assert_eq!(out.0[1], Command::Level { db: quiet });
    assert!(quiet < 0.0 && 0.0 < loud);
    // On the synth: a pile at the highest level held at the ceiling.
    let (mut c, mut s) = playing();
    c.set_loudness(&mut s, Some(10.0));
    assert_eq!(s.level(), calibration::MOST as f32);
    let ceiling = 10f32.powf(-1.0 / 20.0);
    let mut out = vec![0.0; 2 * 960];
    let mut peak = 0.0f32;
    for block in 0..200u64 {
        let now = s.frame();
        if block % 25 == 0 {
            for r in layers::rungs() {
                c.event(&mut s, now, word(r));
            }
            c.event(&mut s, now, Event::Jackpot);
        }
        s.render_to(&mut out);
        peak = out.iter().fold(peak, |m, x| m.max(x.abs()));
    }
    assert!(peak > 0.3 && peak <= ceiling + 1e-3, "{peak}");
}

/// A word held full long pops bigger: louder, a body an octave under it
/// too, still under the limiter's ceiling.
#[test]
fn a_word_held_full_long_pops_bigger() {
    let pop = |size| Event::Word {
        streak: 3,
        pan: 0.0,
        size,
    };
    let (plain, held) = (peak_db(&[pop(0.0)]), peak_db(&[pop(1.0)]));
    assert!(
        held > plain + 1.0 && held <= -1.0 + 1e-3,
        "{plain:.1} then {held:.1} dBFS"
    );
    let (mut c, mut out) = conductor();
    c.event(&mut out, 1, pop(1.0));
    let body = c.palette[Atom::PopBody];
    let bodies: Vec<f32> = (out.notes().iter())
        .filter(|n| n.1 == body)
        .map(|n| n.2)
        .collect();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[1], bodies[0] - 12.0);
}

/// `sound` of the sheet alone at its level through a fresh synth for
/// `seconds`: its left side from the frame it is cued on (the limiter's
/// look-ahead taken off).
fn sheet_alone(sound: Sheet, seconds: f64) -> Vec<f32> {
    let (mut c, mut s) = playing();
    let sheet = Diploma::Sheet {
        sound,
        gain: 1.0,
        pan: 0.0,
    };
    c.event(&mut s, 0, Event::Diploma(sheet));
    let mut out = vec![0.0; 2 * (seconds * SECOND) as usize];
    s.render_to(&mut out);
    out.iter().step_by(2).skip(s.latency()).copied().collect()
}

/// The loudness of `x` in windows of 10 ms from `from` to `to` seconds.
fn loudness(x: &[f32], from: f64, to: f64) -> Vec<f64> {
    let w = (0.01 * SECOND) as usize;
    let (a, z) = ((from * SECOND) as usize, (to * SECOND) as usize);
    x[a..z.min(x.len())]
        .chunks(w)
        .map(|c| (c.iter().map(|&v| f64::from(v * v)).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

/// Every sound of the sheet: heard, well under the ceiling (the softest
/// paper under the medals, the bigger moments louder over their first
/// second), the same every time, and over (its reverb's tail too) a few
/// seconds after.
#[test]
fn every_sheet_sound_is_heard_and_ends() {
    let mut louds = Vec::new();
    for sound in Sheet::ALL {
        let x = sheet_alone(sound, 7.0);
        assert_eq!(x, sheet_alone(sound, 7.0), "{sound:?} the same every time");
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((0.01..0.6).contains(&peak), "{sound:?}: {peak}");
        let end = &x[x.len() - SECOND as usize / 2..];
        let tail = end.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(tail < 1e-3 * peak, "{sound:?} over: {tail} of {peak}");
        let first = &x[..SECOND as usize];
        let loud = first.iter().map(|&v| f64::from(v * v)).sum::<f64>();
        louds.push((sound, loud));
    }
    let of = |s: Sheet| louds.iter().find(|p| p.0 == s).unwrap().1;
    assert!(of(Sheet::Tap) < of(Sheet::Medal(Medal::Bronze)));
    assert!(of(Sheet::Scribbled(Entry::Words)) < of(Sheet::Written(Entry::Words)));
    assert!(of(Sheet::Medal(Medal::Bronze)) < of(Sheet::Medal(Medal::Gold)));
}

/// The sheet's rustle follows its front: loudest as it starts to unroll,
/// quieter as it slows, the front's way
/// ([`crate::core::karaoke::result::unrolled`]).
#[test]
fn the_rustle_follows_the_sheet() {
    let x = sheet_alone(Sheet::Unroll, 1.0);
    let heard = loudness(&x, 0.0, UNROLL * 0.8);
    let early = heard[1..6].iter().sum::<f64>() / 5.0;
    let late = heard[heard.len() - 5..].iter().sum::<f64>() / 5.0;
    assert!(early > 3.0 * late, "{early} {late}");
}

/// The signature's scratch follows its stroke: loud where the pen goes
/// fast, quiet where it slows (its loops' turns, its swash coming to a
/// stop), silent once it is off but for the flick.
#[test]
fn the_signature_follows_its_stroke() {
    let x = sheet_alone(Sheet::Signature, 1.5);
    let heard = loudness(&x, 0.0, SIGNING - 0.03);
    let most = super::pen::fastest();
    let pace: Vec<f64> = (0..heard.len())
        .map(|k| super::pen::signature_speed(0.01 * k as f64, most))
        .collect();
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let (mh, mp) = (mean(&heard), mean(&pace));
    let cov: f64 = (heard.iter().zip(&pace))
        .map(|(h, p)| (h - mh) * (p - mp))
        .sum();
    let var = |x: &[f64], m: f64| x.iter().map(|v| (v - m) * (v - m)).sum::<f64>();
    let r = cov / (var(&heard, mh) * var(&pace, mp)).sqrt();
    assert!(r > 0.7, "the scratch follows the pen's speed: {r}");
    // Quick at first (its loops), slowing to a stop.
    let early = mean(&heard[..20]);
    let late = mean(&heard[heard.len() - 15..]);
    assert!(early > 4.0 * late, "{early} {late}");
    // The pen off with a flick: its glint rings out, then nothing.
    let flick = loudness(&x, SIGNING - 0.02, SIGNING + 0.01);
    let loudest = flick.iter().copied().fold(0.0, f64::max);
    assert!(loudest > 2.0 * late, "{flick:?}");
    let after = loudness(&x, SIGNING + 0.3, SIGNING + 0.5);
    assert!(after.iter().all(|&v| v < 0.05 * early), "{after:?}");
}

/// A rubber stamp lands when the stage brings it down: its loudest moment
/// just after [`STAMP_DOWN`].
#[test]
fn a_stamp_lands_on_the_sheet_when_drawn() {
    let x = sheet_alone(Sheet::Stamp, 0.6);
    let heard = loudness(&x, 0.0, 0.6);
    let loudest = (0..heard.len())
        .max_by(|&a, &z| heard[a].total_cmp(&heard[z]))
        .unwrap();
    let at = loudest as f64 * 0.01;
    assert!((STAMP_DOWN..STAMP_DOWN + 0.04).contains(&at), "{at}");
}

/// The notes on the announcer's voice: frame, pitch, velocity.
fn sung(c: &Conductor, out: &Sent) -> Vec<(u64, f32, f32)> {
    let singer = c.palette[Atom::Singer];
    out.0
        .iter()
        .filter_map(|cmd| match *cmd {
            Command::Note {
                frame, patch, note, ..
            } if patch == singer => Some((frame, note.note, note.velocity)),
            _ => None,
        })
        .collect()
}

/// Whether `pitch` is a tone of A minor's pentatonic (A C D E G).
fn in_a_minor(pitch: f32) -> bool {
    let class = (pitch.round() as i32).rem_euclid(12);
    [9, 0, 2, 4, 7].contains(&class) && pitch.fract() == 0.0
}

/// The call lands where the slam does: the top rung's multiplier from its
/// stinger's hit on the next beat, the top phrase's tier's name on the
/// jackpot's downbeat in the gap between the lines; its syllables on the
/// eighths after it, nothing before the moment it was called on; every
/// note it sings a tone of the key's pentatonic, at the call's own height.
#[test]
fn the_call_lands_on_the_slam_s_beat_in_key() {
    let (mut c, mut out) = conductor();
    let hit = c.event(&mut out, 15_000, word(rung(Tier::COUNT)));
    assert_eq!(hit, BEAT);
    let notes = sung(&c, &out);
    assert_eq!(notes[0].0, hit, "the first vowel on the stinger's hit");
    let mut eighths = 0;
    for syllable in call::times(Tier::TOP.multiplier()).unwrap() {
        let on = hit + u64::from(eighths) * BEAT / 2;
        assert!(
            notes.iter().any(|n| n.0 == on),
            "a vowel on eighth {eighths}"
        );
        eighths += syllable.eighths;
    }
    for n in &notes {
        assert!(in_a_minor(n.1), "{n:?} in key");
        assert!((48.0..=80.0).contains(&n.1), "{n:?} where a voice sings");
    }
    let (burst, hiss) = (c.palette[Atom::Burst], c.palette[Atom::Hiss]);
    let noises = out
        .notes()
        .into_iter()
        .filter(|n| n.1 == burst || n.1 == hiss);
    assert!(noises.map(|n| n.0).all(|f| f >= 15_000), "none before now");
    // A phrase reaching the top: on the downbeat with room before the
    // next line, where the jackpot's slam lands.
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, grid([line(0, 7), line(14, 30), None, None]));
    c.event(&mut out, BEAT / 2, Event::Phrase(Tier::TOP));
    let notes = sung(&c, &out);
    let mut jackpot = Sent::default();
    let d = c.event(&mut jackpot, BEAT / 2, Event::Jackpot);
    assert_eq!((notes[0].0, d), (8 * BEAT, 8 * BEAT));
    assert!(notes.iter().all(|n| in_a_minor(n.1)), "{notes:?}");
    // Unsure: at once, its first consonant from the moment it was called.
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, unsure(NONE));
    c.event(&mut out, 15_000, word(rung(Tier::COUNT)));
    assert_eq!(sung(&c, &out)[0].0, 15_000);
    assert!(out.notes().iter().all(|n| n.0 >= 15_000));
}

/// Only the rows that slam call out (the call goes with the slam), and
/// every row has its words to be called with.
#[test]
fn the_call_goes_with_the_slam() {
    use crate::gui::fx::{strength, Fx};
    for t in Tier::ALL {
        let fx = &t.spec().fx;
        let slams = |list| strength(list, Fx::Slam) > 0.0;
        let chimes = layers::chime_has(Some(t), Chime::Call);
        assert_eq!(chimes, slams(fx.on_phrase), "{}", t.name());
        let rungs = rung_has(rung_of(t), Streak::Call);
        assert_eq!(rungs, slams(fx.on_rung), "{}", t.name());
        assert!(call::name(t).is_some() && call::times(t.multiplier()).is_some());
        let (mut c, mut out) = conductor();
        c.event(&mut out, 15_000, word(t.spec().streak));
        assert_eq!(!sung(&c, &out).is_empty(), rungs, "{}", t.name());
        let (mut c, mut out) = conductor();
        c.event(&mut out, 15_000, Event::Phrase(t));
        assert_eq!(!sung(&c, &out).is_empty(), chimes, "{}", t.name());
    }
}

/// The left side of what `called` sounds through a fresh synth from frame
/// 1 000 (its first vowel there), as the clock has the lines `lines`.
fn called(called: Called, lines: [Option<(u64, u64)>; Clock::LINES]) -> Vec<f32> {
    let (mut c, mut s) = playing();
    c.set_clock(&mut s, grid(lines));
    c.announce(&mut s, 1_000, 1_000, called);
    let mut out = vec![0.0; 2 * RATE as usize * 3];
    s.render_to(&mut out);
    out.iter().step_by(2).copied().collect()
}

/// The loudest 0.4 s of `x` (its mean square's root), and its peak, dB.
fn levels(x: &[f32]) -> (f64, f64) {
    let w = (0.4 * SECOND) as usize;
    let squares: Vec<f64> = x.iter().map(|&v| f64::from(v * v)).collect();
    let mut sum: f64 = squares[..w].iter().sum();
    let mut most = sum;
    for k in w..squares.len() {
        sum += squares[k] - squares[k - w];
        most = most.max(sum);
    }
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let db = |v: f64| 20.0 * v.max(1e-12).log10();
    (db((most / w as f64).sqrt()), db(f64::from(peak)))
}

/// The call sits under the stinger it comes with (the top rung's, as the
/// diploma plays it: no call), its loudest 0.4 s 2 dB down at least and
/// its peak 3, yet heard; sung over a line it is softer still.
#[test]
fn the_call_stays_under_the_stinger_and_ducks_under_singing() {
    let (mut c, mut s) = playing();
    c.event(&mut s, 1_000, Event::Diploma(Diploma::Rung(Tier::TOP)));
    let mut out = vec![0.0; 2 * RATE as usize * 3];
    s.render_to(&mut out);
    let stinger: Vec<f32> = out.iter().step_by(2).copied().collect();
    let (sting, sting_peak) = levels(&stinger);
    for word in [Called::Times(Tier::TOP), Called::Name(Tier::TOP)] {
        let (loud, peak) = levels(&called(word, NONE));
        println!("{word:?}: {loud:.1} dB, peak {peak:.1}; the stinger {sting:.1}, {sting_peak:.1}");
        assert!(loud < sting - 2.0 && loud > sting - 12.0, "{word:?}");
        assert!(peak < sting_peak - 3.0, "{word:?}");
        let (ducked, _) = levels(&called(word, [line(0, 8), None, None, None]));
        assert!(ducked < loud - 6.0, "{word:?} {ducked:.1} {loud:.1}");
    }
}

/// A theme gives the announcer its own voice (Spooky's ghost) and nothing
/// else: the same notes on the same frames, on a patch of its own.
#[test]
fn a_theme_s_announcer_calls_the_same_in_its_own_voice() {
    let calls = |theme: &crate::theme::Theme| {
        let (mut c, _) = Conductor::new(RATE, theme);
        let mut out = Sent::default();
        c.set_clock(&mut out, grid(NONE));
        c.set_key(Some(&a_minor()), 0);
        c.announce(&mut out, 15_000, BEAT, Called::Times(Tier::TOP));
        c.announce(&mut out, 15_000, 4 * BEAT, Called::Name(Tier::TOP));
        sung(&c, &out)
    };
    let spooky = &crate::theme::SPOOKY;
    assert_eq!(calls(spooky), calls(&crate::theme::CLASSIC));
    assert_ne!(spooky.patch(Atom::Singer), palette::patch(Atom::Singer));
}

/// A palette as built is told the synth nothing (so the game sounds as it
/// is made, bit for bit, while nothing is edited); a patch edited is told
/// once, as a command, its next notes the new patch's, and set back it is
/// told again; one not taken is told on the next call.
#[test]
fn a_patch_is_told_the_synth_once_when_it_changes() {
    for theme in crate::theme::THEMES {
        let (mut c, _) = Conductor::new(RATE, theme);
        let mut out = Sent::default();
        for atom in Atom::ALL {
            c.set_patch(&mut out, atom, theme.patch(atom));
        }
        assert!(out.0.is_empty(), "{}", theme.name);
    }
    let (mut c, mut synth) = Conductor::new(RATE, &crate::theme::CLASSIC);
    let built = palette::patch(Atom::Pop);
    let louder = jodel_synth::Patch {
        level: built.level + 6.0,
        ..built
    };
    c.set_patch(&mut Full, Atom::Pop, louder);
    assert_eq!(c.dropped(), 1);
    let mut out = Sent::default();
    c.set_patch(&mut out, Atom::Pop, louder);
    c.set_patch(&mut out, Atom::Pop, louder);
    let pop = c.palette[Atom::Pop];
    assert_eq!(
        out.0,
        [Command::Patch {
            patch: pop,
            to: louder
        }]
    );
    synth.command(out.0[0]);
    assert_eq!(synth.patch(pop), Some(&louder));
    out.0.clear();
    c.set_patch(&mut out, Atom::Pop, built);
    assert_eq!(out.0.len(), 1);
}

/// A key of the Synth tab's keyboard holds its note on the atom's patch
/// until it is let go of, alone: the other keys held hold on.
#[test]
fn a_key_holds_its_note_until_it_is_let_go_of() {
    let (mut c, mut synth) = Conductor::new(RATE, &crate::theme::CLASSIC);
    c.key_down(&mut synth, 0, Atom::Choir, 60.0);
    c.key_down(&mut synth, 0, Atom::Choir, 64.0);
    let mut out = vec![0.0f32; 2 * 24_000];
    synth.render_to(&mut out);
    assert_eq!(synth.voices().filter(|v| v.held).count(), 2);
    let at = synth.frame();
    c.key_up(&mut synth, at, Atom::Choir, 60.0);
    synth.render_to(&mut out);
    let held: Vec<f32> = synth.voices().filter(|v| v.held).map(|v| v.note).collect();
    assert_eq!(held, [64.0]);
    assert_eq!(c.dropped(), 0);
}
