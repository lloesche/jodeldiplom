//! The conductor's rules on the song's grid (in frames: the beat, an
//! unsure grid, a downbeat with room before the singing; the rule itself
//! is the grid's, tested in `jodel_rhythm::grid`), the key from the
//! analysis's and its shift, the ladder landing on the tonic at every rung
//! (a jump in the run missing none), the tier table's rows growing their
//! sounds up the tiers, the held sounds and the payout stopping with the
//! streak, held sounds ending by themselves when their release is lost, no
//! key meaning no pitched layer, the most commands an event sends, the
//! bus never clipping, the sounds' levels (over a song at the reference
//! loudness, under its ceiling) and the bus's level following the song's
//! loudness.

use jodel_rhythm::grid::When;
use jodel_rhythm::{Analysis, Region};
use jodel_synth::{Command, Commands, PatchId, Synth, Tag};

use super::palette::{Atom, BRASS, COINS, PADS};
use super::*;
use crate::tiers::Tier;

const RATE: u32 = 48_000;
/// A second of the song in frames.
const SECOND: f64 = 48_000.0;
/// 120 beats a minute at 48 kHz.
const BEAT: u64 = 24_000;
/// The tier table's rungs.
const RUNGS: [u32; 5] = [5, 10, 20, 35, 55];
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
    let (mut c, _) = Conductor::new(RATE);
    let mut out = Sent::default();
    c.set_clock(&mut out, grid(NONE));
    c.set_key(Some(&a_minor()), 0);
    out.0.clear();
    (c, out)
}

/// The same, playing on its synth.
fn playing() -> (Conductor, Synth) {
    let (mut c, mut s) = Conductor::new(RATE);
    c.set_clock(&mut s, grid(NONE));
    c.set_key(Some(&a_minor()), 0);
    (c, s)
}

fn word(streak: u32) -> Event {
    Event::Word { streak, pan: 0.0 }
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
    for rungs in [&RUNGS[..], &[3, 7], &[4, 8, 12, 16, 20, 24, 30]] {
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
    let (mut c, mut out) = conductor();
    for streak in 1..=5 {
        c.event(&mut out, u64::from(streak), word(streak));
    }
    let pop = c.palette[Atom::Pop];
    let pops: Vec<f32> = out
        .notes()
        .iter()
        .filter(|n| n.1 == pop)
        .map(|n| n.2)
        .collect();
    assert_eq!(pops.len(), 5);
    assert_eq!(pops[4], 93.0, "A6, an octave over the climb's A5");
    assert!(pops.windows(2).all(|w| w[1] > w[0]));
}

/// A word told a step lower than the rung reached (a word rescued after
/// those past it) keeps the rung: nothing is let go or called off, and the
/// next word on reaches no rung again.
#[test]
fn a_lower_step_keeps_the_rung_only_the_streak_lost_lets_it_go() {
    let (mut c, mut out) = conductor();
    c.event(&mut out, 15_000, word(5));
    assert_eq!(c.rung, 1);
    out.0.clear();
    c.event(&mut out, 30_000, word(4));
    c.event(&mut out, 36_000, word(6));
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
    assert_eq!(c.event(&mut out, 15_000, word(5)), BEAT);
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
    c.event(&mut out, BEAT - 2_000, word(5));
    assert_eq!(out.on(c.palette[Atom::Pluck])[0], BEAT);
    // Unsure: the hit at once.
    let (mut c, mut out) = conductor();
    c.set_clock(&mut out, unsure(NONE));
    assert_eq!(c.event(&mut out, 15_000, word(5)), 15_000);
    // A run that jumps past two rungs: the higher one's stinger.
    let (mut c, mut out) = conductor();
    c.event(&mut out, 1_000, word(3));
    assert!(out.on(c.palette[Atom::Chime]).is_empty());
    c.event(&mut out, 15_000, word(12));
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
    c.event(&mut out, BEAT / 2, word(55));
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
    c.event(&mut out, BEAT / 2, word(55));
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
    c.event(&mut s, 1_000, word(35));
    c.event(&mut s, 2_000, word(54));
    render(&mut s, 24_000);
    assert_eq!(sounding(&c, &s, &held), 3, "the riser and the shimmer");
    let now = s.frame();
    c.event(&mut s, now, Event::Lost(Tier::named("Amazing")));
    render(&mut s, 9_600);
    assert_eq!(sounding(&c, &s, &held), 0, "gone in 0.2 s");
    // ×32 reached, then lost before its payout: none of it sounds.
    let (mut c, mut s) = playing();
    c.event(&mut s, 1_000, word(55));
    c.event(&mut s, 2_000, Event::Lost(Tier::TOP));
    for _ in 0..80 {
        render(&mut s, 4_800);
        assert_eq!(sounding(&c, &s, &[Atom::Whoosh, Atom::Low]), 0);
    }
}

#[test]
fn the_riser_ends_by_itself_when_no_rung_comes() {
    let (mut c, mut s) = playing();
    c.event(&mut s, 0, word(35));
    c.event(&mut s, 1_000, word(54));
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
    c.event(&mut lossy, 0, word(35));
    for k in 1..=6 {
        c.event(&mut lossy, k * BEAT / 2, word(35 + k as u32));
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
    let stings: Vec<usize> = RUNGS.map(|r| count(word(r))).to_vec();
    assert!(stings.windows(2).all(|w| w[1] > w[0]), "{stings:?}");
    assert_eq!(RUNGS, layers::rungs(), "the table's");
    // The points under the first tier: the smallest chime there is.
    let none = count(Event::Diploma(Diploma::Points(None)));
    assert!(none <= chimes[0], "{none} {chimes:?}");
}

#[test]
fn no_event_sends_more_than_the_most_commands() {
    let events = [
        word(55),
        word(35),
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
    ];
    let mut most = 0;
    for e in events {
        let (mut c, mut out) = conductor();
        // At the top already, the riser sounding: the most a word does.
        c.event(&mut out, 0, word(54));
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
    c.event(&mut out, 0, word(12));
    assert!(clinks(&c, &out).is_empty(), "×4: none");
    // ×8 reached (its stinger's ka-ching rattling coins of its own).
    c.event(&mut out, BEAT, word(20));
    out.0.clear();
    c.event(&mut out, 10 * BEAT, word(22));
    c.event(&mut out, 10 * BEAT + 2_400, word(23));
    let v = clinks(&c, &out);
    assert_eq!(v.len(), 4, "two clinks and their bounces");
    assert!(v[2] < v[0], "the second, close after, softer");
    out.0.clear();
    c.event(&mut out, 20 * BEAT, word(35));
    let before = clinks(&c, &out).len();
    c.event(&mut out, 20 * BEAT + 1_000, word(36));
    assert_eq!(clinks(&c, &out).len(), before, "none over the stinger");
}

#[test]
fn with_no_key_nothing_pitched_low_plays_and_the_pops_sit_high() {
    let (mut c, _) = Conductor::new(RATE);
    let mut out = Sent::default();
    c.set_clock(&mut out, grid(NONE));
    for streak in 1..=55 {
        c.event(&mut out, u64::from(streak) * 6_000, word(streak));
    }
    for e in [
        Event::Phrase(Tier::TOP),
        Event::Jackpot,
        Event::Lost(Tier::TOP),
    ] {
        c.event(&mut out, 400_000, e);
    }
    c.event(
        &mut out,
        400_000,
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
    c.event(&mut out, 1_000, word(35));
    c.event(&mut out, 2_000, word(54));
    out.0.clear();
    c.cancel(&mut out, 3_000);
    assert_eq!(out.0[0], Command::Cancel { tag: None });
    let released = |a: Atom| {
        out.0.contains(&Command::Release {
            frame: 3_000,
            patch: c.palette[a],
            tag: Tag::NONE,
        })
    };
    assert!(released(Atom::Riser) && released(PADS[0]) && released(PADS[1]));
    assert!(!c.riser && c.rung == 0);
}

#[test]
fn the_tempo_goes_to_the_synth_with_the_clock() {
    let (mut c, _) = Conductor::new(RATE);
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
            for &r in &RUNGS {
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
    let top = RUNGS[RUNGS.len() - 1];
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
            for &r in &RUNGS {
                c.event(&mut s, now, word(r));
            }
            c.event(&mut s, now, Event::Jackpot);
        }
        s.render_to(&mut out);
        peak = out.iter().fold(peak, |m, x| m.max(x.abs()));
    }
    assert!(peak > 0.3 && peak <= ceiling + 1e-3, "{peak}");
}
