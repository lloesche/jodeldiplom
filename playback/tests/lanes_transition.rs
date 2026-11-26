//! A transition while the desk plays the stems at other levels: both
//! tracks' stems are on the player's strips, each track on a head of its
//! own, so a stem muted there is out of both, on the desk's frames, and
//! nothing jumps where the next track's map ends and it goes back onto its
//! file. The separator gives each stem a quarter of the mix; the drums
//! muted, what is heard throughout is three quarters of what the
//! transition plays; a stem soloed, a quarter of it; and a stem muted on a
//! bar inside the window, half of it from there.

mod common;

use common::transition::*;
use common::*;
use jodel_dsp::FULL_SCALE;
use jodel_output::lanes::Parts;
use jodel_output::SampleFormat::S24In32Low;
use jodel_playback::stems::STEMS;
use jodel_playback::{Envelope, Event, MakeStems, TempoMode, TransitionPlan, TransitionState};
use jodel_testkit::TempDir;

fn plan() -> TransitionPlan {
    TransitionPlan {
        from: 0,
        into: 1,
        start: 5.0,
        length: 2.0,
        entry: 1.0,
        beats: beats(5.0, 0.5, 1.0, 0.5, 5),
        lanes: lanes(
            Envelope::fade(1.0, 0.0, 0.0, 1.0),
            Envelope::fade(0.0, 1.0, 0.0, 1.0),
        ),
        tempo: TempoMode::Varispeed,
        settle: 1.0,
        glide: false,
    }
}

/// How the stems stand on the desk.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stems {
    /// The drums muted throughout.
    DrumsMuted,
    /// The vocals soloed throughout.
    VocalsSoloed,
    /// The drums muted, and the vocals too from 6 s of the playing track,
    /// a bar inside the window.
    VocalsOnABar,
}

/// The second of the playing track the vocals go off at in
/// [`Stems::VocalsOnABar`].
const BAR: f64 = 6.0;

/// The share of the transition heard at output frame `f`.
fn share_heard(stems: Stems, f: usize) -> f64 {
    match stems {
        Stems::DrumsMuted => 0.75,
        Stems::VocalsSoloed => 0.25,
        Stems::VocalsOnABar if (f as f64) < BAR * f64::from(RATE) => 0.75,
        Stems::VocalsOnABar => 0.5,
    }
}

/// What a run of the transition gave: the device's samples, the events,
/// and A's and B's samples.
struct Ran {
    got: Vec<i32>,
    seen: Vec<Event>,
    a: Vec<i32>,
    b: Vec<i32>,
}

/// Plays the transition with the stems as `stems` has them.
fn run(stems: Stems) -> Ran {
    let dir = TempDir::new(&format!("lanes-transition-{stems:?}"));
    let (a_path, a) = wav(&dir, "a.wav", 12.0, tone([1_000.0, 5_000.0]));
    let (b_path, b) = wav(&dir, "b.wav", 12.0, tone([2_000.0, 3_000.0]));
    let desk = Desk::new(caps(&[RATE], &[S24In32Low]), options());
    desk.player.make_stems(MakeStems::Separate).unwrap();
    let strips = &desk.desk.stems;
    match stems {
        Stems::DrumsMuted | Stems::VocalsOnABar => strips[0].set_mute(true),
        Stems::VocalsSoloed => strips[2].set_solo(true),
    }
    desk.play(vec![a_path, b_path]);
    desk.player.transition(plan()).unwrap();
    let mut seen = Vec::new();
    if stems == Stems::VocalsOnABar {
        // In the window, before the bar is written: the vocals off from
        // the bar, on the playing track's frame.
        seen = step_until(&desk.gate, &desk.events, |e| {
            matches!(e, Event::Transition(TransitionState::Started { .. }))
        });
        let bar = desk.lead.input.at_stream((BAR * f64::from(RATE)) as u64);
        strips[2].set_mute_at(true, bar);
    }
    seen.extend(step_to_end(&desk.gate, &desk.events));
    // The next track's stems are parts at the end, and heard.
    let view = desk.mixer.desk();
    let heard: Vec<_> = view
        .strips
        .iter()
        .flat_map(|s| &s.inputs)
        .filter_map(|i| i.parts)
        .filter(|p| p.state == Parts::Present)
        .collect();
    assert!(heard.iter().any(|p| p.share == 1.0), "{stems:?}: {heard:?}");
    let streams = desk.close();
    assert!(transitions(&seen).contains(&&TransitionState::Finished));
    assert_eq!(streams.len(), 1, "one stream throughout");
    Ran {
        got: samples(&streams[0].1),
        seen,
        a,
        b,
    }
}

/// The worst difference of `ran`'s frames `range` from the share of the
/// transition heard there, and where.
fn worst(ran: &Ran, stems: Stems, range: impl Iterator<Item = usize>) -> (f64, usize) {
    let plan = plan();
    let offset = 4 * RATE as usize;
    let (mut worst, mut at) = (0.0f64, 0);
    for f in range {
        let share = plan.share(f as f64 / f64::from(RATE));
        let (ga, gb) = (plan.lanes[0].gain(share), plan.lanes[STEMS].gain(share));
        for c in 0..2 {
            let a = ran.a.get(2 * f + c).map_or(0.0, |&s| f64::from(s));
            let b = (f >= offset).then(|| f64::from(ran.b[2 * (f - offset) + c]));
            let want = share_heard(stems, f) * (ga * a + gb * b.unwrap_or(0.0));
            let off = (f64::from(ran.got[2 * f + c]) - want).abs();
            if off > worst {
                (worst, at) = (off, f);
            }
        }
    }
    (worst, at)
}

/// Plays the transition with the stems as `stems` has them, and checks
/// every frame against that share of what it plays.
fn never_jumps(stems: Stems) {
    let ran = run(stems);
    let rate = RATE as usize;
    assert_eq!(ran.got.len() / 2, 16 * rate, "A to 7 s, then B from 3 s");
    // (Not where the window starts and ends: its steps there are as
    // without stems muted, `tests/transition.rs`; nor over the 20 ms a
    // strip is muted over on its bar.)
    let bar = (BAR * f64::from(RATE)) as usize;
    let edge = |f: usize| {
        [5 * rate, 7 * rate].iter().any(|&w| f.abs_diff(w) < 200)
            || (stems == Stems::VocalsOnABar && f.abs_diff(bar) < 1_000)
    };
    let frames = (200..ran.got.len() / 2 - 200).filter(|&f| !edge(f));
    let (worst, at) = worst(&ran, stems, frames);
    assert!(
        worst < 0.005 * FULL_SCALE,
        "{stems:?}: off by {:.4} of full scale at {:.4} s",
        worst / FULL_SCALE,
        at as f64 / f64::from(RATE)
    );
}

#[test]
fn a_transition_with_a_stem_muted_never_jumps() {
    let _serial = serial();
    never_jumps(Stems::DrumsMuted);
}

#[test]
fn a_transition_with_a_stem_soloed_never_goes_silent() {
    let _serial = serial();
    never_jumps(Stems::VocalsSoloed);
}

#[test]
fn a_stem_muted_on_a_bar_in_the_window_goes_off_there() {
    let _serial = serial();
    never_jumps(Stems::VocalsOnABar);
}

/// The next track goes back onto its file after its map, its parts heard
/// on the desk across that: nothing is heard of the move, the drums kept
/// out on either side of it.
#[test]
fn the_next_tracks_return_to_its_file_is_not_heard() {
    let _serial = serial();
    let ran = run(Stems::DrumsMuted);
    let back = ran
        .seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) if s.index == 1 => Some(s.output_frame as usize),
            _ => None,
        })
        .next_back()
        .expect("B back on its file");
    let rate = RATE as usize;
    let at = back + origin(5.0) as usize;
    assert!(at > 7 * rate, "B back on its file at {at}");
    let (worst, _) = worst(&ran, Stems::DrumsMuted, at - rate / 5..at + rate / 5);
    assert!(
        worst < 0.002 * FULL_SCALE,
        "off by {:.5} around {at}",
        worst / FULL_SCALE
    );
}
