//! Automation: gain curves an input is heard through on top of everything
//! else (its faders, its lanes' faders, its strips), timed on the stream
//! frames of the output open on it, as a controller plans a transition.
//!
//! - **A curve** ([`GainCurve`]) is a list of breakpoints `(stream frame,
//!   gain)`, immutable once made. Between two points the gain moves
//!   power-linearly ([`super::equal_power`]); before the first it is the
//!   first's, after the last the last's. Two points on one frame are a
//!   cut, smoothed linearly over [`CUT_FADE`] from the gain heard before it
//!   to the curve after it, so a jump never clicks. (Linearly: an
//!   equal-power move starts infinitely steep.)
//! - **A set** ([`Automation`]) per input: the input's own curve (over
//!   every lane), main's (heard on main only: an input with parts heard on
//!   main, because none are there or valid), and one per part. It holds on
//!   the frames its output does not supply (a pad, a pause): a curve moves
//!   only with its stream.
//! - **Replaced whole** ([`super::Fader::automate`], or cleared: every
//!   curve 1): over the first [`CUT_FADE`] of frames supplied after, each
//!   gain goes linearly from where it was heard to the new set's, so a
//!   replace never clicks either. A set given to an input that has heard
//!   nothing of its output yet is in force at once. The set ends with the
//!   output it was given for (a set given with none open is the next
//!   one's).
//! - **Touched:** a part's curve away from exactly 1 has the input heard
//!   on its parts (where they are there); main's curve never does. Once
//!   the parts' curves and main's are past their last points (and their
//!   smoothing) at exactly 1, the parts' switch may go back to main; the
//!   input's own curve, over main and the parts alike, does not hold it.
//!   With every curve so, the set is settled: the input is heard as
//!   without it, bit for bit.
//! - **The mixer thread** owns a set once given: it works out its cuts'
//!   smoothing for the device's rate in place, and frees a set it is done
//!   with, as it frees the messages it takes.

use std::time::Duration;

use super::envelope::Envelope;
use super::equal_power;
use super::input::Given;
use super::parts::Touch;
use crate::error::Error;

/// How long a cut in a [`GainCurve`], or a set replaced, is smoothed over.
pub const CUT_FADE: Duration = Duration::from_millis(10);

/// Where each curve sits in a set, and in what the mixer keeps of one: the
/// input's own, main's, then the parts'.
pub(crate) const INPUT: usize = 0;
pub(crate) const MAIN: usize = 1;

/// Where part `k`'s curve sits (from 0).
pub(crate) const fn part_curve(k: usize) -> usize {
    2 + k
}

/// The curves of an input of `parts` parts.
pub(crate) const fn curves_of(parts: usize) -> usize {
    part_curve(parts)
}

/// `a` to `b`, a share `t` of the way, in a straight line.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// A cut in a curve: its frame, and the gain heard just before it.
#[derive(Clone, Copy, Debug)]
struct Cut {
    frame: u64,
    before: f64,
}

/// A gain curve on stream frames: breakpoints, power-linear between them
/// (see the module's docs).
#[derive(Clone, Debug)]
pub struct GainCurve {
    /// The breakpoints, their frames in order.
    points: Box<[(u64, f64)]>,
    /// The cuts (two points or more on one frame), in order.
    cuts: Box<[Cut]>,
    /// The frames the cuts are smoothed over that their `before` gains
    /// were worked out for (0: not yet).
    fade: u64,
}

impl PartialEq for GainCurve {
    fn eq(&self, other: &GainCurve) -> bool {
        self.points == other.points
    }
}

impl GainCurve {
    /// The curve through `points` (`(stream frame, gain)`, the frames in
    /// order; two on one frame make a cut there). [`Error::InvalidArgument`]
    /// when there are none, a frame comes before the one before it, or a
    /// gain is not finite or below 0.
    pub fn new(points: &[(u64, f64)]) -> Result<GainCurve, Error> {
        let invalid = |why: &str| Err(Error::InvalidArgument(why.into()));
        if points.is_empty() {
            return invalid("a gain curve needs a point");
        }
        if points.windows(2).any(|w| w[1].0 < w[0].0) {
            return invalid("a gain curve's frames go in order");
        }
        if points.iter().any(|&(_, g)| !g.is_finite() || g < 0.0) {
            return invalid("a gain curve's gains are finite and at least 0");
        }
        let mut cuts: Vec<Cut> = points
            .windows(2)
            .filter(|w| w[0].0 == w[1].0)
            .map(|w| Cut {
                frame: w[0].0,
                before: 0.0,
            })
            .collect();
        cuts.dedup_by_key(|c| c.frame);
        Ok(GainCurve {
            points: points.into(),
            cuts: cuts.into(),
            fade: 0,
        })
    }

    /// Its breakpoints.
    pub fn points(&self) -> &[(u64, f64)] {
        &self.points
    }

    /// The gain at stream frame `frame` of an output at `rate` (a cut
    /// smoothed over [`CUT_FADE`] at that rate).
    pub fn gain(&self, frame: u64, rate: u32) -> f64 {
        let fade = cut_frames(rate);
        if self.fade == fade {
            return self.at(frame);
        }
        let mut prepared = self.clone();
        prepared.prepare(fade);
        prepared.at(frame)
    }

    /// Works out the gain heard just before each cut, its cuts smoothed
    /// over `fade` frames (nothing when it has been for that already).
    fn prepare(&mut self, fade: u64) {
        if self.fade == fade {
            return;
        }
        self.fade = fade;
        let mut prev: Option<Cut> = None;
        for k in 0..self.cuts.len() {
            let frame = self.cuts[k].frame;
            let left = self.left(frame);
            // Within the smoothing of the cut before: as smoothed there.
            let before = match prev {
                Some(p) if frame - p.frame < fade => {
                    lerp(p.before, left, (frame - p.frame) as f64 / fade as f64)
                }
                _ => left,
            };
            self.cuts[k].before = before;
            prev = Some(self.cuts[k]);
        }
    }

    /// The gain at `frame`, as prepared: within a cut's smoothing, from the
    /// gain heard just before it to the curve's.
    fn at(&self, frame: u64) -> f64 {
        let raw = self.raw(frame);
        let k = self.cuts.partition_point(|c| c.frame <= frame);
        let Some(cut) = k.checked_sub(1).map(|i| self.cuts[i]) else {
            return raw;
        };
        let into = frame - cut.frame;
        if into >= self.fade {
            return raw;
        }
        lerp(cut.before, raw, into as f64 / self.fade as f64)
    }

    /// The curve's gain at `frame`, unsmoothed: on a cut, after it.
    fn raw(&self, frame: u64) -> f64 {
        let p = &self.points;
        let i = p.partition_point(|q| q.0 <= frame);
        match (i.checked_sub(1).map(|j| p[j]), p.get(i)) {
            (None, _) => p[0].1,
            (Some((_, g)), None) => g,
            (Some((f0, g0)), Some(&(f1, g1))) => {
                equal_power(g0, g1, (frame - f0) as f64 / (f1 - f0) as f64)
            }
        }
    }

    /// The curve's gain coming up to the cut on frame `cut`: its first
    /// point there.
    fn left(&self, cut: u64) -> f64 {
        let i = self.points.partition_point(|q| q.0 < cut);
        self.points[i].1
    }

    /// The first frame from which it is its last gain for good, as
    /// prepared.
    fn end(&self) -> u64 {
        let last = self.points[self.points.len() - 1].0;
        self.cuts
            .last()
            .map_or(last, |c| last.max(c.frame + self.fade))
    }

    /// Its last gain.
    fn last(&self) -> f64 {
        self.points[self.points.len() - 1].1
    }
}

/// [`CUT_FADE`] in frames at `rate`.
fn cut_frames(rate: u32) -> u64 {
    (CUT_FADE.as_secs_f64() * f64::from(rate)).round().max(1.0) as u64
}

/// An input's automation: the curves it is heard through, on top of its
/// faders and strips (see the module's docs). A curve not given is 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Automation {
    /// By where they sit ([`INPUT`], [`MAIN`], [`part_curve`]).
    curves: Vec<Option<GainCurve>>,
}

impl Default for Automation {
    fn default() -> Automation {
        Automation {
            curves: vec![None; curves_of(0)],
        }
    }
}

impl Automation {
    /// No curves: every gain 1.
    pub fn new() -> Automation {
        Automation::default()
    }

    /// The input's own curve, over every lane.
    pub fn input(self, curve: GainCurve) -> Automation {
        self.with(INPUT, curve)
    }

    /// Main's curve: heard on main only (a plain input's one lane, or a
    /// lane input's main while it is heard in its parts' place).
    pub fn main(self, curve: GainCurve) -> Automation {
        self.with(MAIN, curve)
    }

    /// Part `index`'s curve (from 0, the parts in the order the input was
    /// made with, [`super::Mixer::input_lanes`]).
    pub fn part(self, index: usize, curve: GainCurve) -> Automation {
        self.with(part_curve(index), curve)
    }

    fn with(mut self, at: usize, curve: GainCurve) -> Automation {
        if self.curves.len() <= at {
            self.curves.resize(at + 1, None);
        }
        self.curves[at] = Some(curve);
        self
    }

    /// The parts it has a place for, at most.
    pub(crate) fn parts(&self) -> usize {
        self.curves.len() - curves_of(0)
    }

    /// The curve at `k` ([`INPUT`], [`MAIN`], [`part_curve`]).
    fn curve(&self, k: usize) -> Option<&GainCurve> {
        self.curves.get(k).and_then(Option::as_ref)
    }

    /// Its cuts' smoothing worked out for `fade` frames.
    fn prepare(&mut self, fade: u64) {
        for curve in self.curves.iter_mut().flatten() {
            curve.prepare(fade);
        }
    }
}

/// The automation's gains at the last frame mixed, as the desk shows
/// them ([`super::InputView::automation`]).
#[derive(Clone, Debug, PartialEq)]
pub struct AutomationView {
    /// The input's own curve's.
    pub input: f64,
    /// Main's curve's.
    pub main: f64,
    /// Each part's curve's.
    pub parts: Vec<f64>,
    /// Every curve is past its last point, at exactly 1, and any replace
    /// smoothed: the input is heard as without it.
    pub settled: bool,
}

/// An input's automation as the mixer thread runs it: the set in force,
/// and this chunk's gains of each curve.
pub(super) struct AutomationRun {
    set: Option<Box<Automation>>,
    /// This chunk's gains, by where the curves sit.
    gains: Vec<Envelope>,
    /// Each curve's gain at the last frame filled (held where no frame is
    /// supplied).
    held: Vec<f64>,
    /// Each gain where it was heard when the set in force was given, and
    /// the frames supplied since (smoothed from there for the first
    /// [`CUT_FADE`] of them; `u64::MAX`: no smoothing).
    from: Vec<f64>,
    since: u64,
    /// Nothing of its output has been heard since it was opened.
    fresh: bool,
    /// The last stream frame a chunk supplied, since the set was given.
    frame: Option<u64>,
    /// Some part's gain was away from 1 at the last frame filled.
    touched: bool,
    /// How the parts stood this chunk, for the switch, and whether every
    /// curve had settled.
    touch: Touch,
    settled: bool,
}

impl AutomationRun {
    /// No set, for an input of `parts` parts.
    pub(super) fn new(parts: usize) -> AutomationRun {
        let n = curves_of(parts);
        AutomationRun {
            set: None,
            gains: (0..n).map(|_| Envelope::new(1.0)).collect(),
            held: vec![1.0; n],
            from: vec![1.0; n],
            since: u64::MAX,
            fresh: true,
            frame: None,
            touched: false,
            touch: Touch::UNTOUCHED,
            settled: true,
        }
    }

    /// `set` is in force from the next chunk (`None`: every curve 1),
    /// smoothed from the gains heard unless none have been. Returns the
    /// set it replaces.
    pub(super) fn set(&mut self, set: Option<Box<Automation>>) -> Option<Box<Automation>> {
        if self.fresh {
            self.since = u64::MAX;
        } else {
            self.from.copy_from_slice(&self.held);
            self.since = 0;
        }
        self.frame = None;
        std::mem::replace(&mut self.set, set)
    }

    /// Its output is gone: so is the set given for it, at once.
    pub(super) fn reset(&mut self) {
        self.held.fill(1.0);
        self.fresh = true;
        self.since = u64::MAX;
        self.frame = None;
        self.touched = false;
        self.touch = Touch::UNTOUCHED;
        self.settled = true;
        self.set = None;
    }

    /// This chunk's gains of the curve over lane `k` of the input: the
    /// input's own over main (lane 0), part `k - 1`'s over part `k`.
    pub(super) fn lane(&self, k: usize) -> &Envelope {
        let at = k.checked_sub(1).map_or(INPUT, part_curve);
        &self.gains[at]
    }

    /// Of main's curve.
    pub(super) fn main(&self) -> &Envelope {
        &self.gains[MAIN]
    }

    /// How the parts stand as the curves have them this chunk.
    pub(super) fn touch(&self) -> Touch {
        self.touch
    }

    /// Each curve's gains for the `n` frames of the chunk from device frame
    /// `first`, from what its output gave (`None`: nothing), at `rate`: on
    /// the frames given, the curve's at their stream frames (smoothed from
    /// the last set's after a replace); on the others, held. A curve past
    /// its end is one gain for the chunk. Allocates nothing once its gains
    /// have grown to the largest chunk.
    pub(super) fn fill(&mut self, first: u64, n: usize, rate: u32, given: Option<Given>) {
        let fade = cut_frames(rate);
        if let Some(set) = &mut self.set {
            set.prepare(fade);
        }
        let range = given.map_or(0..0, |g| g.at..g.at + g.got as usize);
        let stream = |i: usize| given.map_or(0, |g| g.frame + (i - g.at) as u64);
        let since = self.since;
        for k in 0..self.gains.len() {
            let curve = self.set.as_deref().and_then(|s| s.curve(k));
            let (from, mut g) = (self.from[k], self.held[k]);
            let ended = range.is_empty() || curve.is_none_or(|c| stream(range.start) >= c.end());
            if since >= fade && ended {
                if !range.is_empty() {
                    g = curve.map_or(1.0, GainCurve::last);
                }
                self.gains[k].set_flat(g);
                self.held[k] = g;
                continue;
            }
            let mut s = since;
            self.gains[k].fill_with(n, |i| {
                if range.contains(&i) {
                    let to = curve.map_or(1.0, |c| c.at(stream(i)));
                    g = if s < fade {
                        lerp(from, to, s as f64 / fade as f64)
                    } else {
                        to
                    };
                    s = s.saturating_add(1);
                }
                g
            });
            self.held[k] = g;
        }
        if !range.is_empty() {
            self.frame = Some(stream(range.end - 1));
            self.since = since.saturating_add(range.len() as u64);
            self.fresh = false;
        }
        // Where the parts' gains leave exactly 1, or come back to it.
        let parts = part_curve(0)..self.gains.len();
        let away = |gains: &[Envelope], i: usize| parts.clone().any(|k| gains[k].at(i) != 1.0);
        let changed = (0..n).find(|&i| away(&self.gains, i) != self.touched);
        self.touched = n > 0 && away(&self.gains, n - 1);
        let smoothed = self.since >= fade;
        let at_rest = |k: usize| {
            let curve = self.set.as_deref().and_then(|s| s.curve(k));
            smoothed
                && curve.is_none_or(|c| {
                    c.last() == 1.0 && self.frame.is_some_and(|frame| frame >= c.end())
                })
        };
        let (switch, every) = (
            (MAIN..self.gains.len()).all(&at_rest),
            (INPUT..self.gains.len()).all(&at_rest),
        );
        self.touch = Touch {
            touched: self.touched,
            at: changed.map(|i| first + i as u64),
            settled: switch,
        };
        self.settled = every;
    }

    /// Each curve's gain at the last frame filled, and whether every curve
    /// has settled (`None`: no set, and none being smoothed away).
    pub(super) fn view(&self) -> Option<(&[f64], bool)> {
        let shown = self.set.is_some() || !self.settled;
        shown.then_some((&self.held[..], self.settled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const FADE: u64 = 480; // 10 ms at 48 kHz

    /// A curve, its cuts prepared for [`RATE`].
    fn curve(points: &[(u64, f64)]) -> GainCurve {
        let mut c = GainCurve::new(points).unwrap();
        c.prepare(FADE);
        c
    }

    #[test]
    fn a_curve_moves_power_linearly_between_its_points() {
        let c = curve(&[(100, 1.0), (300, 0.0), (500, 0.5)]);
        assert_eq!(c.at(0), 1.0);
        assert_eq!(c.at(100), 1.0);
        for f in 100..300 {
            let t = (f - 100) as f64 / 200.0;
            assert_eq!(c.at(f), (1.0 - t).sqrt(), "{f}");
        }
        assert_eq!(c.at(300), 0.0);
        assert_eq!(c.at(400), equal_power(0.0, 0.5, 0.5));
        assert_eq!(c.at(500), 0.5);
        assert_eq!(c.at(u64::MAX), 0.5);
        assert_eq!(c.end(), 500);
        assert_eq!(c.gain(400, RATE), c.at(400));
    }

    #[test]
    fn a_cut_is_smoothed_linearly_from_the_gain_before_it() {
        // Held at 1, cut to 0.25 at 1 000, then held.
        let c = curve(&[(0, 1.0), (1_000, 1.0), (1_000, 0.25)]);
        assert_eq!(c.at(999), 1.0);
        assert_eq!(c.at(1_000), 1.0);
        for f in 1_000..1_000 + FADE {
            let t = (f - 1_000) as f64 / FADE as f64;
            assert_eq!(c.at(f), lerp(1.0, 0.25, t), "{f}");
        }
        assert_eq!(c.at(1_000 + FADE), 0.25);
        assert_eq!(c.end(), 1_000 + FADE);
        // A cut that starts a ramp: from the gain before it onto the ramp.
        let r = curve(&[(0, 0.0), (0, 1.0), (2_000, 0.0)]);
        assert_eq!(r.at(0), 0.0);
        let f = FADE / 2;
        assert_eq!(r.at(f), lerp(0.0, r.raw(f), 0.5));
        assert_eq!(r.at(FADE), r.raw(FADE));
        // Unprepared, worked out for the rate asked.
        let raw = GainCurve::new(&[(0, 1.0), (1_000, 1.0), (1_000, 0.25)]).unwrap();
        assert_eq!(raw.gain(1_100, RATE), c.at(1_100));
    }

    #[test]
    fn a_cut_within_anothers_smoothing_goes_on_from_where_that_one_is() {
        let points = [
            (0, 1.0),
            (1_000, 1.0),
            (1_000, 0.0),
            (1_100, 0.0),
            (1_100, 1.0),
        ];
        let c = curve(&points);
        let before = c.at(1_099);
        assert!(before > 0.0 && before < 1.0);
        assert_eq!(c.at(1_100), lerp(1.0, 0.0, 100.0 / FADE as f64));
        // No step bigger than the smoothing's own.
        let steps = (999..1_700).map(|f| (c.at(f + 1) - c.at(f)).abs());
        assert!(steps.fold(0.0, f64::max) <= 1.0 / FADE as f64 + 1e-12);
        assert_eq!(c.at(1_100 + FADE), 1.0);
    }

    #[test]
    fn a_curve_ends_after_its_last_cuts_smoothing() {
        // The last point inside the last cut's smoothing: not at its last
        // gain until the smoothing is over.
        let c = curve(&[(0, 0.2), (1_000, 0.2), (1_000, 1.0), (1_100, 0.6)]);
        assert_eq!(c.end(), 1_000 + FADE);
        assert_ne!(c.at(1_000 + FADE - 1), 0.6);
        assert_eq!(c.at(1_000 + FADE), 0.6);
        // The last cut long before the last point.
        let d = curve(&[(0, 0.2), (0, 1.0), (5_000, 0.6)]);
        assert_eq!(d.end(), 5_000);
    }

    #[test]
    fn a_curve_refuses_points_out_of_order_and_bad_gains() {
        assert!(GainCurve::new(&[]).is_err());
        assert!(GainCurve::new(&[(5, 1.0), (4, 1.0)]).is_err());
        assert!(GainCurve::new(&[(5, -0.1)]).is_err());
        assert!(GainCurve::new(&[(5, f64::NAN)]).is_err());
        assert_eq!(curve(&[(5, 0.5)]).at(0), 0.5);
    }

    fn all(e: &Envelope, n: usize) -> Vec<f64> {
        (0..n).map(|i| e.at(i)).collect()
    }

    fn given(at: usize, frame: u64, got: u64) -> Option<Given> {
        Some(Given { at, frame, got })
    }

    #[test]
    fn the_run_holds_where_nothing_is_supplied_and_settles_at_1() {
        let mut run = AutomationRun::new(1);
        let c = curve(&[(0, 1.0), (100, 0.0), (300, 1.0)]);
        run.set(Some(Box::new(Automation::new().part(0, c.clone()))));
        // Frames 40..100 of the stream from chunk frame 20; held around.
        run.fill(1_000, 100, RATE, given(20, 40, 60));
        let got = all(run.lane(1), 100);
        assert_eq!(got[..20], [1.0; 20]);
        for (i, &g) in got.iter().enumerate().take(80).skip(20) {
            assert_eq!(g, c.at(40 + i as u64 - 20), "{i}");
        }
        assert_eq!(got[80..], [got[79]; 20]);
        assert_eq!(run.lane(0).flat(), Some(1.0));
        // Away from 1 from the first frame supplied.
        let t = run.touch();
        assert!(t.touched && !t.settled);
        assert_eq!(t.at, Some(1_020));
        // Nothing supplied: held, nothing changes.
        run.fill(1_100, 100, RATE, None);
        assert_eq!(run.lane(1).flat(), Some(got[79]));
        assert_eq!(run.touch().at, None);
        // Past its last point, at 1: untouched and settled.
        run.fill(1_200, 100, RATE, given(0, 250, 100));
        let t = run.touch();
        assert!(!t.touched && t.settled, "{t:?}");
        assert_eq!(t.at, Some(1_250));
        assert_eq!(run.view(), Some((&[1.0, 1.0, 1.0][..], true)));
        // From then on, one gain a chunk.
        run.fill(1_300, 100, RATE, given(0, 350, 100));
        assert_eq!(run.lane(1).flat(), Some(1.0));
    }

    #[test]
    fn a_replace_is_smoothed_linearly_from_the_gains_heard() {
        let mut run = AutomationRun::new(0);
        let input = |g: f64| Some(Box::new(Automation::new().input(curve(&[(0, g)]))));
        run.set(input(0.5));
        // Nothing heard yet: in force at once.
        run.fill(0, 100, RATE, given(0, 0, 100));
        assert_eq!(run.lane(0).flat(), Some(0.5));
        // Replaced: from 0.5 to 0.1 over the first 480 frames supplied
        // after, a pause between.
        assert!(run.set(input(0.1)).is_some());
        run.fill(100, 300, RATE, given(0, 100, 300));
        run.fill(400, 100, RATE, None);
        assert!(!run.touch().settled && run.view().is_some());
        run.fill(500, 300, RATE, given(0, 400, 300));
        let got = all(run.lane(0), 300);
        for (i, &g) in got.iter().enumerate() {
            let s = 300 + i as u64;
            let want = if s < FADE {
                lerp(0.5, 0.1, s as f64 / FADE as f64)
            } else {
                0.1
            };
            assert!((g - want).abs() < 1e-12, "{i}: {g} for {want}");
        }
        // Cleared: back to 1 over the smoothing, then settled, no set.
        run.set(None);
        run.fill(800, 300, RATE, given(0, 700, 300));
        assert_eq!(run.lane(0).at(0), 0.1);
        assert!((run.lane(0).at(240) - lerp(0.1, 1.0, 0.5)).abs() < 1e-12);
        assert_eq!(run.view().map(|v| v.1), Some(false));
        run.fill(1_100, 480, RATE, given(0, 1_000, 480));
        assert_eq!(run.lane(0).at(180), 1.0);
        assert!(run.touch().settled);
        assert_eq!(run.view(), None);
        run.fill(1_580, 480, RATE, given(0, 1_480, 480));
        assert_eq!(run.lane(0).flat(), Some(1.0));
    }

    #[test]
    fn mains_curve_never_touches_the_parts_and_the_inputs_never_holds_them() {
        let mut run = AutomationRun::new(2);
        let auto = Automation::new()
            .main(curve(&[(0, 0.5)]))
            .input(curve(&[(0, 1.0), (1_000, 0.0)]));
        run.set(Some(Box::new(auto)));
        run.fill(0, 10, RATE, given(0, 0, 10));
        assert_eq!(run.main().flat(), Some(0.5));
        let t = run.touch();
        assert!(!t.touched && !t.settled);
        // Main's settled, the input's still moving: the switch may go back.
        let auto = Automation::new().input(curve(&[(0, 1.0), (1_000, 0.0)]));
        run.set(Some(Box::new(auto)));
        run.fill(10, 480, RATE, given(0, 10, 480));
        run.fill(490, 10, RATE, given(0, 490, 10));
        assert!(run.touch().settled);
        assert_eq!(run.view().map(|v| v.1), Some(false));
    }
}
