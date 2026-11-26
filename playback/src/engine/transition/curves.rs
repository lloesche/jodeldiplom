//! A transition's plan compiled into each head's automation ([`compile`]):
//! each stem's envelope its part's gain curve on the head's own stream
//! frames, their mean main's.

use jodel_output::mixer::{Automation, GainCurve};

use crate::stems::{Stem, LANES};
use crate::transition::{Envelope, Side, TransitionPlan};
use crate::warp::Glide;

/// Seconds of output between the points of a curve that the plan's
/// envelopes do not give exactly (through a glide, or a mean of lanes).
const STEP: f64 = 0.01;

/// A share of the window just before a point of an envelope.
const EPSILON: f64 = 1e-12;

/// A head's automation: main's curve and each part's, in [`LANES`]' order.
pub(super) struct Curves {
    pub(super) main: GainCurve,
    pub(super) parts: Vec<GainCurve>,
}

impl Curves {
    /// The automation of these curves.
    pub(super) fn automation(&self) -> Automation {
        self.with(Automation::new(), |c| c.clone())
    }

    /// Each curve made by `curve`, as an automation on top of `base`.
    pub(super) fn with(
        &self,
        base: Automation,
        curve: impl Fn(&GainCurve) -> GainCurve,
    ) -> Automation {
        let main = base.main(curve(&self.main));
        self.parts
            .iter()
            .enumerate()
            .fold(main, |a, (k, c)| a.part(k, curve(c)))
    }
}

/// A gain held from stream frame `frame` on.
pub(super) fn held(frame: u64, gain: f64) -> GainCurve {
    GainCurve::new(&[(frame, gain)]).expect("one finite point")
}

/// The plan's lanes as each head's automation (A's, then B's), at `rate`:
/// with the window from A's stream frame `w0`, and B's frame 0 on A's
/// `origin`. Each part's curve is its stem's envelope at A's time heard on
/// each frame (through the glide's inverse, A's time along the output's),
/// main's their mean; before the window A is whole and B silent.
pub(super) fn compile(
    plan: &TransitionPlan,
    glide: &Glide,
    w0: u64,
    origin: u64,
    rate: u32,
) -> [Curves; 2] {
    let rate = f64::from(rate);
    // A's stream frame (fractional) where it is at its time `a`.
    let frame_of = |a: f64| w0 as f64 + (glide.inverse(a) - plan.start) * rate;
    let head = |side: Side, before: f64, shift: u64| {
        let lanes = LANES.map(|s: Stem| plan.lane(side, s));
        let lane_curve = |envelopes: &[&Envelope]| {
            let differ = envelopes.windows(2).any(|w| w[0] != w[1]);
            curve(
                plan,
                envelopes,
                before,
                !glide.is_identity() || differ,
                |a| (frame_of(a) - shift as f64).round().max(0.0) as u64,
            )
        };
        Curves {
            main: lane_curve(&lanes),
            parts: lanes.iter().map(|&e| lane_curve(&[e])).collect(),
        }
    };
    [head(Side::From, 1.0, 0), head(Side::Into, 0.0, origin)]
}

/// The curve of the mean of `envelopes` over the window and after it (at
/// `before` before it), on the frames `frame_of` puts A's times on: a point
/// on each of theirs (two on a cut), and every [`STEP`] between when
/// `dense`.
fn curve(
    plan: &TransitionPlan,
    envelopes: &[&Envelope],
    before: f64,
    dense: bool,
    frame_of: impl Fn(f64) -> u64,
) -> GainCurve {
    let mut shares: Vec<f64> = envelopes
        .iter()
        .flat_map(|e| e.points.iter().map(|p| p.at))
        .filter(|&s| s > 0.0)
        .collect();
    let last = shares.iter().copied().fold(0.0, f64::max);
    if dense && plan.length > 0.0 {
        let step = STEP / plan.length;
        shares.extend((1..).map(|k| k as f64 * step).take_while(|&s| s < last));
    }
    shares.sort_by(f64::total_cmp);
    shares.dedup();
    // The mean just before `share` and just after it.
    let mean = |share: f64| {
        let n = envelopes.len().max(1) as f64;
        let (left, right) = envelopes
            .iter()
            .map(|e| around(e, share))
            .fold((0.0, 0.0), |(l, r), (a, b)| (l + a, r + b));
        (left / n, right / n)
    };
    let at = |share: f64| frame_of(plan.start + share * plan.length);
    // Two points on a frame are a cut (smoothed): only where the gain cuts.
    let near = |x: f64, y: f64| (x - y).abs() <= 1e-9;
    let mut points = vec![(at(0.0), before)];
    let (_, first) = mean(0.0);
    if !near(first, before) {
        points.push((at(0.0), first));
    }
    for share in shares {
        let (left, right) = mean(share);
        if !near(left, right) {
            points.push((at(share), left));
        }
        points.push((at(share), right));
    }
    GainCurve::new(&points).expect("frames in order, gains finite")
}

/// `envelope`'s gain just before `share` and just after it: two on a cut
/// there, else the same.
fn around(envelope: &Envelope, share: f64) -> (f64, f64) {
    let at = envelope.gain(share);
    let points = &envelope.points;
    if !points.iter().any(|p| p.at == share && p.cut) {
        return (at, at);
    }
    let after = points
        .iter()
        .rev()
        .find(|p| p.at == share)
        .map_or(at, |p| p.gain);
    (envelope.gain(share - EPSILON), after)
}
