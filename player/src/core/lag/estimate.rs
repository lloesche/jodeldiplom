//! The singer's delay estimated from their singing: while Karaoke mode
//! sings, the last [`WINDOW`] of the singer's pitch frames are kept (when
//! each was heard, before any delay is taken off, and its pitch), and
//! every [`EVERY`] of the song they are held against its notes at each
//! delay from 0 to [`MOST`] in steps of [`STEP`] ([`measure`]): how much
//! of the singing scores at each, as the scoring at the song's level
//! scores it (at medium for easy: [`judged_at`]; [`fit`]). On real singing that is a broad hill, and one
//! window of it a noisy one, so the measures are averaged into a curve
//! (the first ones evenly, then each followed in over the last half minute
//! or so: [`FOLLOW`]), each delay weighed with its neighbours ([`SPREAD`]);
//! the delay at its top, found between the delays tried, is the estimate
//! ([`pick`]): once enough frames land on notes, after [`FIRST_MEASURES`],
//! and only when it is clearly better than the one before (at first the
//! delay measured or the echo's it would correct, [`Estimator::anchor`], else
//! the default). A measure takes about 1.3 ms on a real song's 20 s (the
//! delays tried 20 ms apart for that).
//!
//! What counts is what the scoring counts: each word's share of its note
//! time hit, up to the share its level asks. Counting every frame hit
//! instead favours a later delay: a singer holds a vowel past a word's
//! end, and a later delay pulls those frames back into it, though the
//! word scores no more for them (on a real headset's singing, frames
//! favoured 160 to 200 ms, the words 140 to 160).

use std::collections::VecDeque;
use std::path::PathBuf;

use super::DEFAULT_LATE;
use crate::core::singing::{Hit, Level, Singing, SungKey};

/// How much of the singing is held against the notes, in seconds.
pub const WINDOW: f64 = 20.0;
/// How often, in seconds of the song.
pub const EVERY: f64 = 2.0;
/// The latest delay tried, in seconds.
pub const MOST: f64 = 0.4;
/// The step between the delays tried, in seconds (the estimate is found
/// between them, to [`RESOLUTION`]: [`pick`]).
pub const STEP: f64 = 0.02;
/// What the estimate is rounded to, in seconds.
pub const RESOLUTION: f64 = 0.01;
/// The measures to take in before a first estimate: no one window, nor a
/// few, decides it.
pub const FIRST_MEASURES: usize = 8;
/// The frames with a pitch that must land on notes for an estimate (3 s
/// of them).
pub const LEAST_VOICED: usize = 300;
/// How much more a new estimate must score than the one before: this
/// share more, and at least [`MARGIN_LEAST`] of a word.
pub const MARGIN: f64 = 0.03;
/// The least more, in words.
pub const MARGIN_LEAST: f64 = 0.2;
/// What the share of the frames that hit weighs, in words: enough to tell
/// apart delays at which as many words score, too little to outweigh one.
pub const TIE: f64 = 0.5;

/// The level the singing is held against the notes at for the estimate,
/// sung at `level`: medium for easy (easy lets so much count that every
/// delay near the right one scores alike, a hill too flat to read; how
/// late the voice comes is the equipment's, not the level's), else the
/// level itself.
pub fn judged_at(level: Level) -> Level {
    match level {
        Level::Easy => Level::Medium,
        other => other,
    }
}

/// How well `frames` (when each was heard, before any delay, and its
/// pitch: none where nothing was sung) fit the notes of `singing` heard
/// `lag` seconds late, as its scoring would have it: each word's credit
/// ([`Level::credit`]: its frames that hit, the singer's own key followed
/// where the level allows, against what the level asks), made smooth
/// ([`Credit::smooth`]: a word that scores counts one, whatever more of it
/// is hit), summed; and, to tell apart the delays at which as many words
/// score, the share of the frames with a pitch on the notes that hit,
/// [`TIE`] of a word at most. And how many frames with a pitch land on
/// notes.
///
/// [`Level::credit`]: crate::core::singing::Level::credit
/// [`Credit::smooth`]: crate::core::singing::Credit::smooth
pub fn fit(singing: &Singing, frames: &[(f64, Option<f32>)], lag: f64) -> (f64, usize) {
    let level = judged_at(singing.level);
    let mut key = SungKey::default();
    // The word being heard (frames come in time, so a word's come
    // together): which, its frames on its notes, those that hit and half
    // hit.
    let mut word: Option<((usize, usize), f64, f64, f64)> = None;
    let mut scored = 0.0;
    let (mut hits, mut on) = (0.0, 0);
    for &(t, cents) in frames {
        let Some((at, note)) = singing.word_at(t - lag) else {
            continue;
        };
        if word.is_none_or(|w| w.0 != at) {
            if let Some((_, noted, hit, half)) = word {
                scored += level.credit(noted, hit, half).smooth();
            }
            word = Some((at, 0.0, 0.0, 0.0));
        }
        let Some(w) = word.as_mut() else {
            continue;
        };
        w.1 += 1.0;
        let Some(cents) = cents else {
            continue;
        };
        let cents = f64::from(cents);
        key.hear(cents, note);
        on += 1;
        match level.judge(key.off(level, cents, note)) {
            Some(Hit::Full) => {
                w.2 += 1.0;
                hits += 1.0;
            }
            Some(Hit::Half) => {
                w.3 += 1.0;
                hits += 0.5;
            }
            None => {}
        }
    }
    if let Some((_, noted, hit, half)) = word {
        scored += level.credit(noted, hit, half).smooth();
    }
    (scored + TIE * hits / on.max(1) as f64, on)
}

/// The delays tried: from 0 to [`MOST`] in steps of [`STEP`].
fn lags() -> impl Iterator<Item = f64> {
    (0..=(MOST / STEP).round() as usize).map(|k| k as f64 * STEP)
}

/// How well `frames` fit at each delay tried ([`fit`]); none while fewer
/// than [`LEAST_VOICED`] with a pitch land on notes.
pub fn measure(singing: &Singing, frames: &[(f64, Option<f32>)]) -> Option<Vec<f64>> {
    // Too few with a pitch at all (silence): no fit tried.
    if frames.iter().filter(|f| f.1.is_some()).count() < LEAST_VOICED {
        return None;
    }
    let mut on = 0;
    let curve = lags()
        .map(|lag| {
            let (score, n) = fit(singing, frames, lag);
            on = on.max(n);
            score
        })
        .collect();
    (on >= LEAST_VOICED).then_some(curve)
}

/// The delays tried either side that each is weighed with, in steps.
pub const SPREAD: usize = 2;

/// `curve` (a value a delay tried) weighed with its neighbours, nearer
/// ones more: a single delay that fits by chance does not stand out.
fn spread(curve: &[f64]) -> Vec<f64> {
    (0..curve.len())
        .map(|i| {
            let (mut sum, mut weights) = (0.0, 0.0);
            for (j, v) in curve
                .iter()
                .enumerate()
                .take(i + SPREAD + 1)
                .skip(i.saturating_sub(SPREAD))
            {
                let w = (SPREAD + 1 - i.abs_diff(j)) as f64;
                sum += w * v;
                weights += w;
            }
            sum / weights
        })
        .collect()
}

/// `smooth` at `lag` seconds, between the delays tried.
fn at(smooth: &[f64], lag: f64) -> f64 {
    let x = (lag / STEP).clamp(0.0, (smooth.len() - 1) as f64);
    let i = (x.floor() as usize).min(smooth.len().saturating_sub(2));
    let f = x - i as f64;
    smooth[i] + f * (smooth.get(i + 1).copied().unwrap_or(smooth[i]) - smooth[i])
}

/// The delay `curve` (how well the singing fits at each delay tried) is
/// best at, each weighed with its neighbours ([`SPREAD`]), found between
/// them (the top of a parabola through the best and its two neighbours)
/// to [`RESOLUTION`]; with an estimate before (`current`), only when it
/// fits clearly better than that one ([`MARGIN`]).
pub fn pick(curve: &[f64], current: Option<f64>) -> Option<f64> {
    let smooth = spread(curve);
    let (best, top) =
        smooth
            .iter()
            .enumerate()
            .fold(None, |b: Option<(usize, f64)>, (i, &v)| match b {
                Some((_, t)) if t >= v => b,
                _ => Some((i, v)),
            })?;
    let between = match (best.checked_sub(1), smooth.get(best + 1)) {
        (Some(before), Some(&after)) => {
            let (a, c) = (smooth[before], after);
            let bend = a - 2.0 * top + c;
            if bend < 0.0 {
                (0.5 * (a - c) / bend).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    let lag = ((best as f64 + between) * STEP / RESOLUTION).round() * RESOLUTION;
    let Some(now) = current else {
        return Some(lag);
    };
    let was = at(&smooth, now);
    (top >= was + (was * MARGIN).max(MARGIN_LEAST)).then_some(lag)
}

/// How near the best a delay must fit for [`pick_near`] to take it: this
/// share of the best.
pub const NEAR: f64 = 0.005;
/// How much better than a delay measured the best must fit for the
/// singing to correct it ([`pick_near`]): this share more, and at least
/// [`MARGIN_LEAST`] of a word. Less than [`MARGIN`]: the correction goes
/// only so far ([`super::CORRECT_WITHIN`]), and its curve is the whole
/// singing's, not the last half minute's.
pub const MARGIN_NEAR: f64 = 0.01;

/// The delay nearest `anchor` (a delay measured, which the estimate would
/// correct) that `curve` fits nearly as well as its best at ([`NEAR`]),
/// found between the delays tried to [`RESOLUTION`]: the singing moves a
/// delay measured only as far as it must (a curve's top is broad, longer
/// delays catching the vowels held past a word). Only when the best fits
/// better than `anchor` ([`MARGIN_NEAR`]); with an estimate before
/// (`current`), that one while it still fits nearly as well. None (an
/// estimate before let go) where no delay within
/// [`super::CORRECT_WITHIN`] of `anchor` fits nearly as well as the best:
/// the singing points elsewhere, and the one measured stands. (A broad
/// top whose very best lies further, the delays near `anchor` on it too,
/// is not elsewhere: its best wanders over it, window by window.) The
/// estimate to hold.
pub fn pick_near(curve: &[f64], anchor: f64, current: Option<f64>) -> Option<f64> {
    let smooth = spread(curve);
    let top = smooth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let near = top - (top * NEAR).max(0.0);
    match current {
        // Kept while it fits within twice as much: no back and forth.
        Some(c) if at(&smooth, c) >= top - 2.0 * (top * NEAR).max(0.0) => return Some(c),
        None => {
            let was = at(&smooth, anchor);
            if top < was + (was * MARGIN_NEAR).max(MARGIN_LEAST) {
                return None;
            }
        }
        Some(_) => {}
    }
    let steps = (MOST / RESOLUTION).round() as usize;
    (0..=steps)
        .map(|k| k as f64 * RESOLUTION)
        .filter(|&lag| at(&smooth, lag) >= near)
        .min_by(|a, b| (a - anchor).abs().total_cmp(&(b - anchor).abs()))
        .filter(|lag| (lag - anchor).abs() <= super::CORRECT_WITHIN + 1e-9)
}

/// The delay from 0 to [`MOST`] that `frames` fit best, once enough land
/// on notes ([`measure`], [`pick`]).
pub fn best(singing: &Singing, frames: &[(f64, Option<f32>)], current: Option<f64>) -> Option<f64> {
    pick(&measure(singing, frames)?, current)
}

/// How much of each new measure goes into the curve the estimate is
/// picked from (the rest is the curve before): about the last half
/// minute of singing counts.
pub const FOLLOW: f64 = 0.07;

/// The estimate as the singing goes.
#[derive(Clone, Debug, Default)]
pub struct Estimator {
    /// The frames of the last [`WINDOW`]: when, the pitch.
    frames: VecDeque<(f64, Option<f32>)>,
    /// The song they are of.
    song: Option<PathBuf>,
    /// When the frames were last held against the notes, and when the
    /// last frame was heard (seconds into the song).
    last_run: Option<f64>,
    last: Option<f64>,
    /// How well the singing fits at each delay tried, followed measure by
    /// measure ([`FOLLOW`]).
    curve: Option<Vec<f64>>,
    /// The measures followed into it.
    measures: usize,
    /// The estimate, in seconds. It and the curve are kept from song to
    /// song (they are the singer's and their equipment's, not the song's),
    /// not for another output or microphone (`super::set_pair`).
    pub lag: Option<f64>,
    /// The delay measured or the echo's which the estimate corrects
    /// (`super::correction`; [`Estimator::anchor`]): the last there was.
    anchored: Option<f64>,
}

impl Estimator {
    /// The delay measured or the echo's now (none: neither), which the
    /// estimate would correct: until there is an estimate, one must fit
    /// clearly better than it ([`pick_near`]; than [`DEFAULT_LATE`]
    /// without one ever). None after one (the echo unsteady a while) keeps
    /// the last. Another than the last (a sync again; the echo settled
    /// elsewhere, further than it moves while steady,
    /// [`super::STEADY_WITHIN`]) lets the estimate go: it must win against
    /// the new one afresh (the curve is kept, it is the singing's).
    pub fn anchor(&mut self, at: Option<f64>) {
        if let (Some(a), Some(was)) = (at, self.anchored) {
            if (a - was).abs() > super::STEADY_WITHIN + 1e-9 {
                self.lag = None;
            }
        }
        self.anchored = at.or(self.anchored);
    }

    /// A frame heard `t` seconds into the song (before any delay is taken
    /// off) at `cents` (none: nothing sung), against `singing`'s notes.
    /// Another song, or the song from further back (sought back, sung
    /// again), starts the frames over.
    pub fn hear(&mut self, singing: &Singing, t: f64, cents: Option<f32>) {
        let song = singing.song();
        if self.song.as_deref() != song || self.last.is_some_and(|last| t < last - 1.0) {
            self.frames.clear();
            self.last_run = None;
            self.song = song.map(PathBuf::from);
        }
        self.last = Some(t);
        self.frames.push_back((t, cents));
        while self.frames.front().is_some_and(|f| f.0 < t - WINDOW) {
            self.frames.pop_front();
        }
        if self.last_run.is_some_and(|run| t - run < EVERY) {
            return;
        }
        self.last_run = Some(t);
        let Some(now) = measure(singing, self.frames.make_contiguous()) else {
            return;
        };
        // The first measures are averaged (no one window decides), then
        // each is followed in by [`FOLLOW`]; correcting a delay measured,
        // by a quarter of that (it is the equipment's, steady: a few
        // windows sung oddly never move it).
        self.measures += 1;
        let least = if self.anchored.is_some() {
            FOLLOW / 4.0
        } else {
            FOLLOW
        };
        let weight = (1.0 / self.measures as f64).max(least);
        let curve = match self.curve.take() {
            Some(was) => was
                .iter()
                .zip(&now)
                .map(|(w, n)| w + weight * (n - w))
                .collect(),
            None => now,
        };
        // Against the delay measured (or the default) until there is an
        // estimate: one must fit clearly better than that, after enough
        // measures.
        if self.measures >= FIRST_MEASURES || self.lag.is_some() {
            match self.anchored {
                // Correcting a delay measured: as little as fits, none
                // where the singing points elsewhere.
                Some(from) => self.lag = pick_near(&curve, from, self.lag),
                None => {
                    if let Some(lag) = pick(&curve, Some(self.lag.unwrap_or(DEFAULT_LATE))) {
                        self.lag = Some(lag);
                    }
                }
            }
        }
        self.curve = Some(curve);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_rhythm::waveform::Melody;
    use std::path::Path;
    use std::sync::Arc;

    /// A song of 40 words, half a second each, on a melody that moves
    /// (C4, E4, G4, D4, A4 over and over), and its notes made at `level`.
    fn song(level: Level) -> (Singing, impl Fn(f64) -> Option<f32>) {
        song_of(level, 40)
    }

    /// [`song`] of `n` words.
    fn song_of(level: Level, n: usize) -> (Singing, impl Fn(f64) -> Option<f32>) {
        const PITCH: [u16; 5] = [6000, 6400, 6700, 6200, 6900];
        let mut m = Melody {
            cents: vec![0; n * 50 + 100],
            sure: vec![0; n * 50 + 100],
        };
        for k in 0..n {
            for i in k * 50..(k + 1) * 50 {
                m.cents[i] = PITCH[k % 5];
                m.sure[i] = 255;
            }
        }
        let words = (0..n)
            .map(|k| jodel_lyrics::Word {
                start: Some(k as f64 * 0.5),
                end: Some((k + 1) as f64 * 0.5),
                confidence: 0.9,
                ..jodel_lyrics::Word::new("la")
            })
            .collect();
        let lyrics = Arc::new(jodel_lyrics::Lyrics {
            lines: vec![jodel_lyrics::Line {
                words,
                ..jodel_lyrics::Line::default()
            }],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        });
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        let pitch = move |t: f64| {
            (t >= 0.0 && t < n as f64 * 0.5).then(|| f32::from(PITCH[(t / 0.5) as usize % 5]))
        };
        (s, pitch)
    }

    /// 20 s of the singer heard `lag` late, a frame every 10 ms: the
    /// song's pitch sung a little flat on a share `voiced` of the frames
    /// (none on the consonants and breaths), now and then an octave out.
    fn heard(
        pitch: &impl Fn(f64) -> Option<f32>,
        lag: f64,
        voiced: f64,
    ) -> Vec<(f64, Option<f32>)> {
        heard_for(pitch, lag, voiced, 2000)
    }

    /// [`heard`] for `frames` frames.
    fn heard_for(
        pitch: &impl Fn(f64) -> Option<f32>,
        lag: f64,
        voiced: f64,
        frames: u32,
    ) -> Vec<(f64, Option<f32>)> {
        (0..frames)
            .map(|i| {
                let t = f64::from(i) / 100.0;
                let sung = (f64::from(i) * 0.618_034).fract() < voiced;
                let octave = if i % 37 == 0 { 1200.0 } else { 0.0 };
                let c = pitch(t - lag).filter(|_| sung).map(|c| c - 80.0 + octave);
                (t, c)
            })
            .collect()
    }

    #[test]
    fn singing_heard_120_ms_late_is_found_within_10_ms() {
        for level in Level::ALL {
            let (s, pitch) = song(level);
            let frames = heard(&pitch, 0.12, 0.5);
            let lag = best(&s, &frames, None).expect("an estimate");
            assert!((lag - 0.12).abs() <= 0.0101, "{level:?}: {lag}");
            // From itself: nothing better.
            assert_eq!(best(&s, &frames, Some(lag)), None, "{level:?}");
        }
        // From an estimate far off: taken where the words score clearly
        // better (on hard; easier levels score as many words 230 ms off).
        let (s, pitch) = song(Level::Hard);
        let frames = heard(&pitch, 0.12, 0.5);
        assert!(best(&s, &frames, Some(0.35)).is_some_and(|l| (l - 0.12).abs() <= 0.0101));
    }

    #[test]
    fn a_new_estimate_must_fit_clearly_better() {
        // A hill topping at 200 ms, 100 words there.
        let curve: Vec<f64> = lags().map(|l| 100.0 - 50.0 * (l - 0.2).abs()).collect();
        assert!(pick(&curve, None).is_some_and(|l| (l - 0.2).abs() < 1e-9));
        // From 100 ms (5 words less): taken; from 160 ms (2 less, under
        // 3 %): kept.
        assert!(pick(&curve, Some(0.1)).is_some_and(|l| (l - 0.2).abs() < 1e-9));
        assert_eq!(pick(&curve, Some(0.16)), None);
    }

    #[test]
    fn too_few_frames_with_a_pitch_give_none() {
        let (s, pitch) = song(Level::Medium);
        // A tenth of 20 s: 200 frames, under the 300 needed.
        let frames = heard(&pitch, 0.12, 0.1);
        assert!(frames.iter().filter(|f| f.1.is_some()).count() < LEAST_VOICED);
        assert_eq!(best(&s, &frames, None), None);
    }

    #[test]
    fn the_estimator_settles_as_the_song_is_sung() {
        let (s, pitch) = song(Level::Hard);
        let mut e = Estimator::default();
        let frames = heard(&pitch, 0.25, 0.5);
        // The first time through: too few measures for a first estimate
        // (no few windows decide it).
        for &(t, cents) in &frames {
            e.hear(&s, t, cents);
        }
        assert_eq!(e.lag, None);
        // Sung again: once enough are in, it is found.
        for &(t, cents) in &frames {
            e.hear(&s, t, cents);
        }
        assert!(
            e.lag.is_some_and(|l| (l - 0.25).abs() <= 0.0101),
            "{:?}",
            e.lag
        );
        // Sung as the default has it: nothing clearly better, none.
        let mut d = Estimator::default();
        let frames = heard(&pitch, DEFAULT_LATE, 0.5);
        for _ in 0..2 {
            for &(t, cents) in &frames {
                d.hear(&s, t, cents);
            }
        }
        assert_eq!(d.lag, None);
        // Sung again from the start: the frames start over, the estimate
        // stays.
        e.hear(&s, 0.0, None);
        assert!(e.frames.len() == 1 && e.lag.is_some());
    }

    /// `frames` heard against `s`'s notes, the delay in force followed
    /// frame by frame as the game does ([`super::super::Lag::settle`]; the
    /// clock: the song's) from the one `measured`; when the estimate first
    /// came.
    fn follow(
        lag: &mut super::super::Lag,
        s: &Singing,
        frames: &[(f64, Option<f32>)],
        measured: f64,
    ) -> Option<f64> {
        let known = super::super::Known {
            measured: Some(measured),
            ..Default::default()
        };
        let mut found = None;
        for &(t, cents) in frames {
            lag.estimate.hear(s, t, cents);
            lag.settle(known, t, false);
            if found.is_none() && lag.estimate.lag.is_some() {
                found = Some(t);
            }
        }
        found
    }

    /// A delay measured 36 ms short of how late the singer comes: the
    /// estimate, held against it, finds the singer's, and the delay in
    /// force moves there within a few seconds of it; silence after moves
    /// nothing. Sung as measured, nothing is estimated.
    #[test]
    fn the_singing_corrects_a_delay_measured_a_little_short() {
        use super::super::{Lag, Source};
        // A song of a minute and a half, sung with a pitch on a third of
        // the frames: a word's credit short of what hard asks, so each
        // frame on its note counts.
        let (s, pitch) = song_of(Level::Hard, 180);
        let measured = 0.264;
        let mut lag = Lag::default();
        lag.step(measured, Source::Measured, 0.0, true);
        let frames = heard_for(&pitch, 0.30, 0.35, 6000);
        let found = follow(&mut lag, &s, &frames, measured).expect("an estimate");
        assert!(found <= 30.0, "found after {found} s");
        let sung = lag.estimate.lag;
        assert!(sung.is_some_and(|l| (l - 0.30).abs() <= 0.0101), "{sung:?}");
        assert!((lag.late - 0.30).abs() <= 0.0101, "{}", lag.late);
        // Then silence: nothing moves.
        let was = (lag.estimate.lag, lag.late);
        let silence: Vec<(f64, Option<f32>)> =
            (6000..9000).map(|i| (f64::from(i) / 100.0, None)).collect();
        follow(&mut lag, &s, &silence, measured);
        assert_eq!((lag.estimate.lag, lag.late), was);
        // Sung as measured: nothing fits clearly better, no estimate.
        let frames = heard_for(&pitch, measured, 0.35, 6000);
        assert_eq!(follow(&mut Lag::default(), &s, &frames, measured), None);
        // Sung 80 ms later than measured: the singing points elsewhere, so
        // an estimate made near is let go, and nothing is corrected.
        let frames = heard_for(&pitch, measured + 0.08, 0.35, 6000);
        let mut far = Lag::default();
        far.estimate.lag = Some(0.29);
        follow(&mut far, &s, &frames, measured);
        assert_eq!((far.estimate.lag, far.corrected), (None, 0.0));
    }

    /// A sync again (another delay measured): the estimate made against
    /// the one before goes, and corrects nothing until it wins against the
    /// new one; the same again (after the echo unsteady a while) keeps it.
    #[test]
    fn a_sync_again_lets_the_estimate_go() {
        let (s, pitch) = song_of(Level::Hard, 180);
        let mut lag = super::super::Lag::default();
        follow(&mut lag, &s, &heard_for(&pitch, 0.30, 0.35, 6000), 0.264);
        let e = &mut lag.estimate;
        assert!(e.lag.is_some());
        e.anchor(None);
        assert_eq!(e.anchored, Some(0.264), "kept while none");
        e.anchor(Some(0.264));
        assert!(e.lag.is_some(), "the same one again");
        e.anchor(Some(0.29));
        assert_eq!(e.lag, None);
        let to = super::super::target(Some(0.29), None, e.lag, None);
        assert_eq!((to.late, to.corrected), (0.29, 0.0));
    }

    /// Noise (a pitch anywhere, frame by frame, as loud as singing) never
    /// moves a delay measured, at any level.
    #[test]
    fn noise_never_corrects_a_delay_measured() {
        for level in Level::ALL {
            let (s, _) = song_of(level, 180);
            let noise: Vec<(f64, Option<f32>)> = (0..9000)
                .map(|i| {
                    let r = (f64::from(i) * 0.618_034).fract();
                    let q = (f64::from(i) * 0.414_214).fract();
                    let c = (q < 0.7).then_some(4800.0 + 2400.0 * r as f32);
                    (f64::from(i) / 100.0, c)
                })
                .collect();
            let mut lag = super::super::Lag::default();
            follow(&mut lag, &s, &noise, 0.264);
            assert_eq!(lag.estimate.lag, None, "{level:?}");
        }
    }

    /// Easy's singing is held against the notes as medium's: the same
    /// estimate.
    #[test]
    fn easy_is_estimated_as_medium() {
        let frames = heard(&song(Level::Medium).1, 0.12, 0.5);
        let (easy, medium) = (song(Level::Easy).0, song(Level::Medium).0);
        assert_eq!(judged_at(Level::Easy), Level::Medium);
        assert_eq!(best(&easy, &frames, None), best(&medium, &frames, None));
        assert!(best(&easy, &frames, None).is_some());
    }
}
