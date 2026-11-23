//! Transitions on the Transition tab: a preset or hand-made fades per stem, the
//! window on the playing track and where the next one comes in, made into
//! the engine's [`TransitionPlan`].
//!
//! The window starts at the playing track's cue and the next track comes
//! in at its own (see [`draft`]). The beats of the two are paired
//! from there to the window's end, one to one or, where one track is at
//! about half the other's tempo, two to one ([`Pairing`]), so the engine
//! lays the next track's beats onto the playing track's; the tempo glides
//! from one to the other across the window.
//!
//! The beats paired are on one level ([`level_beats`]): where the analysis
//! counts a stretch at another level of the pulse (a half-time bridge, a
//! ballad's verse counted in whole bars), its beats are divided or thinned
//! to the track's usual beat, so the pairs never jump between speeds.

use jodel_playback::stems::{Stem, STEMS};
use jodel_playback::warp::TimeMap;
use jodel_playback::{Envelope, Point, TempoMode, TransitionPlan};
use jodel_rhythm::Analysis;

pub mod draft;
pub mod jump;
pub mod tab;
pub mod viewport;

/// Bars a window lasts without a selection.
pub const DEFAULT_BARS: usize = 8;
/// A lane's gain changes by this much a step while editing.
pub const GAIN_STEP: f64 = 0.25;

/// A ready-made set of fades.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Preset {
    /// Everything crosses over evenly.
    #[default]
    Fade,
    /// The new drums come in over the first quarter, the bass swaps on the
    /// middle bar, the other instruments cross over in the second half,
    /// and the vocals swap on the last bar.
    DrumsFirst,
    /// Everything crosses over, but the bass swaps on the middle bar.
    BassSwap,
    /// The old vocals stay to the end, the new ones come in after it;
    /// the rest crosses over in the first three quarters.
    VocalsLast,
}

impl Preset {
    /// Every preset, in the order `p` goes through them.
    pub const ALL: [Preset; 4] = [
        Preset::Fade,
        Preset::DrumsFirst,
        Preset::BassSwap,
        Preset::VocalsLast,
    ];

    /// Its name in the config file.
    pub fn id(self) -> &'static str {
        match self {
            Preset::Fade => "fade",
            Preset::DrumsFirst => "drums-first",
            Preset::BassSwap => "bass-swap",
            Preset::VocalsLast => "vocals-last",
        }
    }

    /// The preset of a config file name.
    pub fn from_id(id: &str) -> Option<Preset> {
        Preset::ALL.into_iter().find(|p| p.id() == id)
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Fade => "fade",
            Preset::DrumsFirst => "drums first",
            Preset::BassSwap => "bass swap",
            Preset::VocalsLast => "vocals last",
        }
    }

    /// The next preset.
    pub fn next(self) -> Preset {
        let i = Preset::ALL.iter().position(|&p| p == self).unwrap_or(0);
        Preset::ALL[(i + 1) % Preset::ALL.len()]
    }

    /// Its lanes over a window of `bars` bars: the playing track's four
    /// stems then the next's, in [`Stem::ALL`] order. Cuts fall on bars.
    pub fn lanes(self, bars: usize) -> [Envelope; 2 * STEMS] {
        let bars = bars.max(1) as f64;
        // A share of the window, on a bar.
        let on_bar = |share: f64| (share * bars).round() / bars;
        let fade = |from: f64, to: f64| {
            (
                Envelope::fade(1.0, 0.0, on_bar(from), on_bar(to)),
                Envelope::fade(0.0, 1.0, on_bar(from), on_bar(to)),
            )
        };
        let cut = |at: f64| {
            (
                Envelope::cut(1.0, 0.0, on_bar(at)),
                Envelope::cut(0.0, 1.0, on_bar(at)),
            )
        };
        let per_stem = |stem: Stem| -> (Envelope, Envelope) {
            match (self, stem) {
                (Preset::Fade, _) => fade(0.0, 1.0),
                (Preset::DrumsFirst, Stem::Drums) => fade(0.0, 0.25),
                (Preset::DrumsFirst, Stem::Bass) | (Preset::BassSwap, Stem::Bass) => cut(0.5),
                (Preset::DrumsFirst, Stem::Other) => fade(0.5, 1.0),
                (Preset::DrumsFirst, Stem::Vocals) => cut(1.0 - 1.0 / bars),
                (Preset::BassSwap, _) => fade(0.0, 1.0),
                (Preset::VocalsLast, Stem::Vocals) => cut(1.0),
                (Preset::VocalsLast, _) => fade(0.0, 0.75),
            }
        };
        let pairs: Vec<(Envelope, Envelope)> = Stem::ALL.iter().map(|&s| per_stem(s)).collect();
        std::array::from_fn(|i| {
            if i < STEMS {
                pairs[i].0.clone()
            } else {
                pairs[i - STEMS].1.clone()
            }
        })
    }
}

/// The tempo mode's name (as [`Preset::name`] is a preset's).
pub fn tempo_name(tempo: TempoMode) -> &'static str {
    match tempo {
        TempoMode::Varispeed => "varispeed",
        TempoMode::Stretch => "time-stretch",
    }
}

/// How the two tracks' beats pair: how many of the playing track's go
/// with one of the next track's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pairing {
    /// One to one.
    #[default]
    One,
    /// Two of the playing track's to one of the next's: the next at half
    /// the tempo, its bar over two of the playing track's.
    Two,
    /// One of the playing track's to two of the next's: the next at double
    /// the tempo, two of its bars in one of the playing track's.
    Half,
}

impl Pairing {
    /// In the order `r` goes through them (after automatic).
    pub const ALL: [Pairing; 3] = [Pairing::One, Pairing::Two, Pairing::Half];

    /// The playing track's beats to one of the next's.
    pub fn ratio(self) -> f64 {
        match self {
            Pairing::One => 1.0,
            Pairing::Two => 2.0,
            Pairing::Half => 0.5,
        }
    }

    /// How it is shown.
    pub fn label(self) -> &'static str {
        match self {
            Pairing::One => "1:1",
            Pairing::Two => "2:1",
            Pairing::Half => "1:2",
        }
    }

    /// The speed the next track plays at, locked to the playing one, with
    /// the tempi `bpm_a` and `bpm_b`.
    pub fn speed(self, bpm_a: f64, bpm_b: f64) -> f64 {
        bpm_a / (self.ratio() * bpm_b)
    }

    /// The playing track's bars that one bar of the slower track spans:
    /// what its cursor and window move by.
    pub fn bars_per_step(self) -> usize {
        match self {
            Pairing::Two => 2,
            _ => 1,
        }
    }

    /// The next track's bars that one bar of the slower track spans.
    pub fn next_bars_per_step(self) -> usize {
        match self {
            Pairing::Half => 2,
            _ => 1,
        }
    }
}

/// The tempo of `a`'s beats, the ones a plan pairs: a minute over their
/// median spacing (its summary's tempo may count another level, half or
/// double these). `None` with fewer than two.
pub fn beat_tempo(a: &Analysis) -> Option<f64> {
    tempo_of(&a.beats)
}

/// The tempo of `a`'s beats from `from` for `span` seconds, where a
/// transition pairs them, on the track's usual level ([`level_beats`]: a
/// stretch counted at another level is not another tempo); the whole
/// track's where there are too few there.
pub fn beat_tempo_near(a: &Analysis, from: f64, span: f64) -> Option<f64> {
    let lo = a.beats.partition_point(|&t| t < from);
    let leveled = level_beats(a, lo);
    let hi = leveled.partition_point(|&t| t <= from + span);
    if hi >= 5 {
        tempo_of(&leveled[..hi])
    } else {
        beat_tempo(a)
    }
}

/// A gap this far (as a share) from a whole number of usual beats is
/// divided into them.
const LEVEL_SLACK: f64 = 0.2;

/// `a`'s beats from beat `from` on, on the track's usual level (its beats'
/// median spacing, [`beat_tempo`]): a gap of about two, three or four
/// usual beats gets the beats between put in, evenly; beats closer than
/// about half a usual beat to the last one kept are left out (a stretch
/// counted twice as fast keeps every other beat, from the first).
pub fn level_beats(a: &Analysis, from: usize) -> Vec<f64> {
    let beats = a.beats.get(from..).unwrap_or(&[]);
    let Some(usual) = beat_tempo(a).map(|bpm| 60.0 / bpm) else {
        return beats.to_vec();
    };
    let mut out: Vec<f64> = Vec::with_capacity(beats.len());
    for &t in beats {
        let Some(&last) = out.last() else {
            out.push(t);
            continue;
        };
        let gap = t - last;
        let steps = (gap / usual).round();
        if gap < 0.6 * usual {
            // Faster than the usual level: not a beat of it.
            continue;
        }
        if (2.0..=8.0).contains(&steps) && (gap / steps / usual - 1.0).abs() <= LEVEL_SLACK {
            let n = steps as usize;
            out.extend((1..n).map(|k| last + gap * k as f64 / n as f64));
        }
        out.push(t);
    }
    out
}

fn tempo_of(beats: &[f64]) -> Option<f64> {
    let mut gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(f64::total_cmp);
    let median = gaps[gaps.len() / 2];
    (median > 0.0).then(|| 60.0 / median)
}

/// Seconds of each track around where they meet that their tempi are
/// measured over.
pub const TEMPO_SPAN: f64 = 16.0;

/// The pairing that changes the next track's speed least, for the tempi
/// of their beats `bpm_a` and `bpm_b` ([`beat_tempo`]; one to one without
/// them).
pub fn pairing_for(bpm_a: Option<f64>, bpm_b: Option<f64>) -> Pairing {
    let (Some(a), Some(b)) = (bpm_a, bpm_b) else {
        return Pairing::One;
    };
    if !(a > 0.0 && b > 0.0) {
        return Pairing::One;
    }
    Pairing::ALL
        .into_iter()
        .min_by(|x, y| {
            x.speed(a, b)
                .ln()
                .abs()
                .total_cmp(&y.speed(a, b).ln().abs())
        })
        .unwrap_or_default()
}

/// Where a transition stands, as the engine reports it.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Status {
    /// Being set up (or nothing yet).
    #[default]
    Draft,
    /// Taken by the engine: it starts at the window.
    Armed,
    /// The window is being heard (as
    /// [`jodel_playback::TransitionState::Started`] tells it), through the
    /// hand-over and the next track's settling.
    Mixing(MixStart),
}

/// Where a mix started: the window's first frame, of the playing track's
/// head, and that head's frame the next track's head's first is linked to
/// (the positions told from the hand-over on count the next track's
/// head's frames, these fewer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixStart {
    /// The window's first frame.
    pub start: crate::core::HeadFrame,
    /// The next track's head's frame 0, on the playing head's frames.
    pub origin: u64,
}

/// The editing cursor on the lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edit {
    /// The lane: 0..4 the playing track's stems, 4..8 the next's (in
    /// [`Stem::ALL`] order).
    pub lane: usize,
    /// The bar of the window.
    pub bar: usize,
}

/// The transition being set up on the Transition tab.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Draft {
    /// The preset the lanes start from.
    pub preset: Preset,
    /// The lanes as edited; `None`: the preset's.
    pub lanes: Option<[Envelope; 2 * STEMS]>,
    /// How the tempi meet.
    pub tempo: TempoMode,
    /// How the beats pair, as set with `r`; `None`: by the tempi
    /// ([`pairing_for`]).
    pub pairing: Option<Pairing>,
    /// Editing the lanes, and where.
    pub edit: Option<Edit>,
    /// Where it stands.
    pub status: Status,
    /// The plan last sent to the engine, while it is armed or mixing:
    /// what is shown then, and what edits change.
    pub sent: Option<Sent>,
}

/// A transition sent to the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Sent {
    /// The plan.
    pub plan: TransitionPlan,
    /// Its window's bars.
    pub bars: usize,
    /// The preset it was made from.
    pub preset: Preset,
    /// The next track's time at the playing track's ([`TransitionPlan::time_map`]).
    pub map: TimeMap,
    /// How its beats pair.
    pub pairing: Pairing,
}

impl Sent {
    /// `plan` over `bars` bars, made from `preset`, its beats paired by
    /// `pairing`.
    pub fn new(plan: TransitionPlan, bars: usize, preset: Preset, pairing: Pairing) -> Sent {
        let map = plan.time_map();
        Sent {
            plan,
            bars,
            preset,
            map,
            pairing,
        }
    }
}

impl Draft {
    /// The lanes over a window of `bars`: as edited, else the preset's.
    pub fn lanes(&self, bars: usize) -> [Envelope; 2 * STEMS] {
        self.lanes
            .clone()
            .unwrap_or_else(|| self.preset.lanes(bars))
    }

    /// The transition in force: the one sent, while it is armed or mixing.
    pub fn in_force(&self) -> Option<&Sent> {
        match self.status {
            Status::Armed | Status::Mixing(_) => self.sent.as_ref(),
            Status::Draft => None,
        }
    }
}

/// The window on the playing track, in its seconds, and its bars: from
/// the bar of `first` (a beat) for `bars` bars (fewer where the track has
/// fewer). `None` without bars.
pub fn window(a: &Analysis, first: usize, bars: usize) -> Option<(f64, f64, usize)> {
    let k = a
        .downbeats
        .partition_point(|&d| d <= first)
        .checked_sub(1)?;
    let start = *a.beats.get(a.downbeats[k])?;
    let last = (k + bars.max(1)).min(a.downbeats.len() - 1);
    let (end, bars) = if last > k {
        (a.beats[a.downbeats[last]], last - k)
    } else {
        (a.duration.max(start), 1)
    };
    Some((start, end, bars))
}

/// The engine's plan: from queue index `from` (with analysis `a`) into
/// the next (`b`), over the window `start..end` of the playing track
/// ([`window`]), the next one coming in at its beat `entry`, the beats
/// paired by `per`.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    from: usize,
    a: &Analysis,
    b: &Analysis,
    (start, end, _bars): (f64, f64, usize),
    entry: usize,
    lanes: [Envelope; 2 * STEMS],
    tempo: TempoMode,
    per: Pairing,
) -> TransitionPlan {
    let i0 = a.beats.partition_point(|&t| t < start - 1e-6);
    let entry_time = b.beats.get(entry).copied().unwrap_or(0.0);
    // Both on their usual level, so the pairs keep one speed.
    let (beats_a, beats_b) = (level_beats(a, i0), level_beats(b, entry.min(b.beats.len())));
    // Through the window, and the beat that ends it: every beat of each
    // side, or every other beat of the faster one.
    let (every_a, every_b) = match per {
        Pairing::One => (1, 1),
        Pairing::Two => (2, 1),
        Pairing::Half => (1, 2),
    };
    let beats: Vec<(f64, f64)> = beats_a
        .iter()
        .step_by(every_a)
        .zip(beats_b.iter().step_by(every_b))
        .take_while(|(&ta, _)| ta <= end + 1e-6)
        .map(|(&ta, &tb)| (ta, tb))
        .collect();
    let length = end - start;
    // What the glide cannot take (the engine keeps it within 30 %), the
    // next track settles after the window, gently.
    let pairs = TimeMap::new(&beats, 1.0);
    let speed = if beats.len() >= 2 {
        pairs.typical_speed()
    } else {
        1.0
    };
    let settle = if speed.ln().abs() > 1.3f64.ln() {
        4.0
    } else {
        1.0
    };
    TransitionPlan {
        from,
        into: from + 1,
        start,
        length,
        entry: entry_time,
        beats,
        lanes,
        tempo,
        // The tempo glides across the window: what is left after it is a
        // residue.
        settle,
        glide: true,
    }
}

/// Sets lane `lane`'s gain at bar `bar` of `bars` (a point there, a cut or
/// a fade from the point before as `cut` says), keeping its other points.
pub fn set_point(
    lanes: &mut [Envelope; 2 * STEMS],
    lane: usize,
    bar: usize,
    bars: usize,
    gain: f64,
    cut: bool,
) {
    let at = bar as f64 / bars.max(1) as f64;
    let points = &mut lanes[lane].points;
    let gain = gain.clamp(0.0, 1.0);
    match points.iter_mut().find(|p| (p.at - at).abs() < 1e-9) {
        Some(p) => {
            p.gain = gain;
            p.cut = cut;
        }
        None => {
            let i = points.partition_point(|p| p.at < at);
            points.insert(i, Point { at, gain, cut });
        }
    }
}

/// Lane `lane`'s point at bar `bar`, if it has one there.
pub fn point_at(
    lanes: &[Envelope; 2 * STEMS],
    lane: usize,
    bar: usize,
    bars: usize,
) -> Option<Point> {
    let at = bar as f64 / bars.max(1) as f64;
    lanes[lane]
        .points
        .iter()
        .find(|p| (p.at - at).abs() < 1e-9)
        .copied()
}

/// A lane's name: the side and the stem.
pub fn lane_name(lane: usize) -> String {
    let side = if lane < STEMS { "now" } else { "next" };
    let stem = match Stem::ALL[lane % STEMS] {
        Stem::Bass => "bass",
        Stem::Drums => "drums",
        Stem::Other => "other",
        Stem::Vocals => "vocals",
    };
    format!("{side} {stem}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_playback::transition::Side;

    /// 64 beats at 120 BPM, bars of four from beat 0.
    fn analysis(bpm: f64, first: f64) -> Analysis {
        let beats: Vec<f64> = (0..64).map(|i| first + f64::from(i) * 60.0 / bpm).collect();
        let mut a = Analysis {
            duration: 40.0,
            downbeats: (0..64).step_by(4).collect(),
            beats,
            ..Analysis::default()
        };
        a.derive();
        a
    }

    #[test]
    fn presets_put_their_cuts_on_bars_and_cross_over() {
        let lanes = Preset::DrumsFirst.lanes(8);
        let p = TransitionPlan {
            from: 0,
            into: 1,
            start: 0.0,
            length: 8.0,
            entry: 0.0,
            beats: Vec::new(),
            lanes: lanes.clone(),
            tempo: TempoMode::Varispeed,
            settle: 2.0,
            glide: false,
        };
        // Bass swaps on bar 4 of 8, vocals on bar 7.
        let bass = Stem::ALL.iter().position(|&s| s == Stem::Bass).unwrap();
        let vocals = Stem::ALL.iter().position(|&s| s == Stem::Vocals).unwrap();
        assert_eq!(p.gains(Side::From, 0.49)[bass], 1.0);
        assert_eq!(p.gains(Side::From, 0.5)[bass], 0.0);
        assert_eq!(p.gains(Side::Into, 0.5)[bass], 1.0);
        assert_eq!(p.gains(Side::Into, 0.874)[vocals], 0.0);
        assert_eq!(p.gains(Side::Into, 0.875)[vocals], 1.0);
        // Everything is the next track's by the end.
        assert_eq!(p.gains(Side::From, 1.0), [0.0; 4]);
        assert_eq!(p.gains(Side::Into, 1.0), [1.0; 4]);
        assert_eq!(Preset::VocalsLast.lanes(8)[vocals].gain(0.99), 1.0);
    }

    /// A track at 119 BPM counted in whole bars for a stretch (its beats
    /// two seconds apart there, as an analysis may count a ballad's verse)
    /// into one at 124: the beats pair one to one at one speed throughout,
    /// and the playing track's tempo glides only the 4 % between them.
    #[test]
    fn a_stretch_counted_in_bars_pairs_on_the_usual_beat() {
        let period = 60.0 / 119.0;
        let mut beats: Vec<f64> = (0..40).map(|i| f64::from(i) * period).collect();
        let bars_from = beats[39] + 4.0 * period;
        beats.extend((0..12).map(|i| bars_from + f64::from(i) * 4.0 * period));
        let back = beats[beats.len() - 1] + period;
        beats.extend((0..40).map(|i| back + f64::from(i) * period));
        let mut a = Analysis {
            duration: beats[beats.len() - 1] + 1.0,
            downbeats: (0..40)
                .step_by(4)
                .chain(40..52)
                .chain((52..92).step_by(4))
                .collect(),
            beats,
            ..Analysis::default()
        };
        a.derive();
        let b = analysis(124.0, 0.5);
        // The window from the first bar of the stretch, two of its bars.
        let first = 36;
        let near = beat_tempo_near(&a, a.beats[first], TEMPO_SPAN).unwrap();
        assert!((near - 119.0).abs() < 1.0, "{near}");
        let per = pairing_for(Some(near), beat_tempo(&b));
        assert_eq!(per, Pairing::One);
        let window = window(&a, first, 3).unwrap();
        let plan = plan(
            0,
            &a,
            &b,
            window,
            0,
            Preset::Fade.lanes(3),
            TempoMode::Varispeed,
            per,
        );
        for w in plan.beats.windows(2) {
            let speed = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            assert!((speed - 119.0 / 124.0).abs() < 0.01, "{:?}", plan.beats);
        }
        let glide = plan.glide_of(&plan.time_map());
        assert!((glide.end_speed() - 124.0 / 119.0).abs() < 0.01);
    }

    #[test]
    fn a_plan_pairs_the_beats_through_the_window() {
        let a = analysis(120.0, 0.0);
        let b = analysis(125.0, 1.0);
        // Bars 3 to 5 of A (beats 8..16): 4.0 s to 8.0 s.
        let w = window(&a, 9, 2).unwrap();
        assert_eq!(w, (4.0, 8.0, 2));
        let plan = plan(
            3,
            &a,
            &b,
            w,
            4,
            Preset::Fade.lanes(2),
            TempoMode::Stretch,
            Pairing::One,
        );
        assert_eq!((plan.from, plan.into), (3, 4));
        assert_eq!(plan.start, 4.0);
        assert_eq!(plan.length, 4.0);
        assert!((plan.entry - b.beats[4]).abs() < 1e-12);
        assert_eq!(plan.beats.len(), 9, "8 beats and the one that ends it");
        assert!((plan.beats[8].0 - 8.0).abs() < 1e-9);
        assert!((plan.beats[8].1 - b.beats[12]).abs() < 1e-9);
        // Past the last bar: to the track's end.
        assert_eq!(window(&a, 62, 8).unwrap().2, 1);
    }

    #[test]
    fn lanes_take_points_by_bar() {
        let mut lanes = Preset::Fade.lanes(4);
        set_point(&mut lanes, 5, 2, 4, 0.25, true);
        assert_eq!(point_at(&lanes, 5, 2, 4).unwrap().gain, 0.25);
        assert!(point_at(&lanes, 5, 2, 4).unwrap().cut);
        assert_eq!(lanes[5].gain(0.5), 0.25);
        set_point(&mut lanes, 5, 2, 4, 2.0, false);
        assert_eq!(point_at(&lanes, 5, 2, 4).unwrap().gain, 1.0);
        assert_eq!(lane_name(5), "next drums");
    }
}
