//! The singer's delay estimated from their singing: while Karaoke mode
//! sings, the last [`WINDOW`] of the singer's pitch frames are kept (when
//! each was heard, before any delay is taken off, and its pitch), and
//! every [`EVERY`] of the song they are held against its notes at each
//! delay from 0 to [`MOST`] in steps of [`STEP`] ([`measure`]): how much
//! of the singing scores at each, as the scoring at the song's level
//! scores it ([`fit`]). On real singing that is a broad hill, and one
//! window of it a noisy one, so the measures are averaged into a curve
//! (the first ones evenly, then each followed in over the last half minute
//! or so: [`FOLLOW`]), each delay weighed with its neighbours ([`SPREAD`]);
//! the delay at its top, found between the delays tried, is the estimate
//! ([`pick`]): once enough frames land on notes, after [`FIRST_MEASURES`],
//! and only when it is clearly better than the one before (the default,
//! at first). A measure takes about 1.3 ms on a real song's 20 s (the
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
use crate::core::singing::{Hit, Singing, SungKey};

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
    let level = singing.level;
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
}

impl Estimator {
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
        // each is followed in by [`FOLLOW`].
        self.measures += 1;
        let weight = (1.0 / self.measures as f64).max(FOLLOW);
        let curve = match self.curve.take() {
            Some(was) => was
                .iter()
                .zip(&now)
                .map(|(w, n)| w + weight * (n - w))
                .collect(),
            None => now,
        };
        // Against the default until there is an estimate: one must fit
        // clearly better than that, after enough measures.
        if self.measures >= FIRST_MEASURES || self.lag.is_some() {
            if let Some(lag) = pick(&curve, Some(self.lag.unwrap_or(DEFAULT_LATE))) {
                self.lag = Some(lag);
            }
        }
        self.curve = Some(curve);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::Level;
    use jodel_rhythm::waveform::Melody;
    use std::path::Path;
    use std::sync::Arc;

    /// A song of 40 words, half a second each, on a melody that moves
    /// (C4, E4, G4, D4, A4 over and over), and its notes made at `level`.
    fn song(level: Level) -> (Singing, impl Fn(f64) -> Option<f32>) {
        const PITCH: [u16; 5] = [6000, 6400, 6700, 6200, 6900];
        let n = 40usize;
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
        (0..2000)
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
}
