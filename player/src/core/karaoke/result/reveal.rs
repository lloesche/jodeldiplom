//! A song's result revealed ([`Reveal`], the results screen's animation):
//! a timeline of beats, each landing
//! at its time after the result is first shown. The sheet unrolling and
//! the two stars by its heading popping on; the title; the points
//! counting up, tick by tick, faster and faster; the words hit; the best
//! streak; each phrase's gem in song order; the verdict (the medal stamped
//! down, or a cheer where there is none, or a word that medals need the
//! whole song); then a new best, or the best before; and the end, after
//! which START goes on (the examiner signing and dating it then). It is a
//! pure function of the result: the stage draws it and the sounds play it
//! from the same beats, at the time since it was shown ([`Shown::time`]),
//! as the singing's moments are one stream. Each beat's time is when it
//! lands: what leads up to it (the medal falling, [`DROP`]) comes before
//! it. What takes a while from its beat (the sheet unrolling, an entry
//! written in by hand, the signature and the date) takes it here too
//! ([`UNROLL`], [`WORDS_FILL`], [`STREAK_WRITE`], [`GRADE_WRITE`],
//! [`SIGNING`], [`DATING`]), for the picture and the sounds alike.
//!
//! Once settled, the screen twinkles a while ([`TWINKLE`]), then goes
//! still: its time stops there, so nothing is drawn again until something
//! else changes.

use super::verdict::{SongResult, Verdict};
use crate::core::shine::Tier;

/// How long the sheet takes to unroll from its beat (the first), in
/// seconds.
pub const UNROLL: f64 = 0.55;
/// When the two stars by the heading pop on, one after the other, as the
/// sheet's front passes them.
const HEADING_STARS: [f64; 2] = [UNROLL * 0.5, UNROLL * 0.5 + 0.08];
/// When the title lands.
const TITLE: f64 = 0.1;
/// When the points start counting up.
const COUNT_FROM: f64 = 0.55;
/// The most ticks the count-up makes (one a point at most).
const TICKS: u32 = 18;
/// How the count speeds up: the share counted is the time's share to this
/// power, so the ticks (at even steps of it) come faster and faster.
const SPEED_UP: f64 = 1.8;
/// How long the words hit take to be written in, in seconds.
pub const WORDS_FILL: f64 = 0.6;
/// How long the best streak takes to be written in.
pub const STREAK_WRITE: f64 = 0.45;
/// How long the verdict's grade takes to be written in.
pub const GRADE_WRITE: f64 = 0.6;
/// How long all the gems take to pop in, at most (from the first to the
/// last, however many), and how far apart two come at most.
const GEMS: f64 = 1.3;
const GEM_MOST: f64 = 0.12;
/// How long a medal falls before it is stamped down on its beat.
pub const DROP: f64 = 0.22;
/// How long a rubber stamp (a new best's, a silent song's) takes from its
/// beat to come down onto the sheet.
pub const STAMP_DOWN: f64 = 0.1;
/// How long the verdict takes to settle after its beat.
const STAMP: f64 = 0.95;
/// How long after the end everything has settled (the last beat's motion
/// over): where a skip puts the reveal.
const SETTLE: f64 = 1.5;
/// How long the settled screen twinkles before it goes still.
const TWINKLE: f64 = 10.0;
/// How long the signature takes to be written from the end.
pub const SIGNING: f64 = 0.9;
/// When after the end the date starts to be written.
pub const DATE_FROM: f64 = SIGNING * 0.4;
/// How long the date takes to be written.
pub const DATING: f64 = 0.6;

/// Eased out, slowing to its end (`x` 0 to 1).
pub fn ease_out(x: f64) -> f64 {
    1.0 - (1.0 - x.clamp(0.0, 1.0)).powi(3)
}

/// How much of the sheet has unrolled `age` seconds after its beat
/// ([`Beat::Paper`]), 0 to 1: quickly, then slowing ([`UNROLL`]).
pub fn unrolled(age: f64) -> f64 {
    ease_out(age / UNROLL)
}

/// How much of the signature is written `age` seconds after the end, 0 to
/// 1, along its stroke: quickly, then slowing ([`SIGNING`]).
pub fn signed(age: f64) -> f64 {
    ease_out(age / SIGNING)
}

/// Which of the heading's two stars.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The left one, first.
    Left,
    /// The right one.
    Right,
}

/// What lands at a beat of the reveal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Beat {
    /// The sheet starts to unroll ([`UNROLL`]).
    Paper,
    /// A star by the heading pops on: the left one, then the right.
    Heading(Side),
    /// The song's title.
    Title,
    /// The count-up a step on: tick `k` (from 0) of [`Reveal::ticks`].
    Tick(u32),
    /// The points all counted.
    Points,
    /// The words hit: they start to be written in ([`WORDS_FILL`]).
    Words,
    /// The best streak: it starts to be written in ([`STREAK_WRITE`]).
    Streak,
    /// Phrase `i`'s gem, in its tier's colour (none: a plain one).
    Gem(usize, Option<Tier>),
    /// The verdict: a medal stamped down (having fallen [`DROP`] before),
    /// a cheer, or the word that medals need the whole song.
    Verdict(Verdict),
    /// A new best ([`SongResult::new_best`]).
    NewBest,
    /// The best before, shown calmly (where the demo key sang, with the
    /// word that this is no best: 0 where there was none).
    Best(u32),
    /// The reveal is over: the examiner signs it ([`SIGNING`]) and dates
    /// it ([`DATE_FROM`], [`DATING`]).
    End,
}

/// A result's reveal: its beats in order, each at its time (seconds since
/// the result was first shown), the count-up's span and ticks, and how far
/// apart the gems come.
#[derive(Clone, Debug, PartialEq)]
pub struct Reveal {
    beats: Vec<(f64, Beat)>,
    points: u32,
    count: (f64, f64),
    ticks: u32,
    gem_every: f64,
}

impl Reveal {
    /// The reveal of `r`.
    pub fn of(r: &SongResult) -> Reveal {
        let mut beats = vec![(0.0, Beat::Paper), (TITLE, Beat::Title)];
        let stars = [Side::Left, Side::Right].into_iter().zip(HEADING_STARS);
        beats.extend(stars.map(|(side, at)| (at, Beat::Heading(side))));
        // The points: longer for more of them, the ticks at even steps of
        // what is counted (so faster and faster), then all of them.
        let span = (0.7 + 0.18 * f64::from(r.points + 1).log10()).min(1.5);
        let n = r.points.min(TICKS);
        for k in 1..n {
            let at = COUNT_FROM + span * (f64::from(k) / f64::from(n)).powf(1.0 / SPEED_UP);
            beats.push((at, Beat::Tick(k - 1)));
        }
        let mut t = COUNT_FROM + span;
        beats.push((t, Beat::Points));
        t += 0.25;
        beats.push((t, Beat::Words));
        t += WORDS_FILL + 0.15;
        beats.push((t, Beat::Streak));
        t += 0.5;
        // The gems, one after another, closer the more there are: all of
        // them within GEMS.
        let phrases = r.phrases.len();
        let gem_every = (GEMS / phrases.saturating_sub(1).max(1) as f64).min(GEM_MOST);
        for (i, p) in r.phrases.iter().enumerate() {
            beats.push((t + gem_every * i as f64, Beat::Gem(i, p.tier)));
        }
        t += match phrases {
            0 => 0.1,
            n => gem_every * (n - 1) as f64 + 0.45,
        };
        // A medal falls first, and lands on its beat.
        let verdict = r.verdict();
        if matches!(verdict, Verdict::Medal(_)) {
            t += DROP;
        }
        beats.push((t, Beat::Verdict(verdict)));
        t += STAMP;
        // The demo's song has its best line too: that it is none.
        let best = if r.new_best() {
            Some(Beat::NewBest)
        } else {
            r.best.or(r.demo.then_some(0)).map(Beat::Best)
        };
        if let Some(best) = best {
            beats.push((t, best));
            t += 0.6;
        }
        beats.push((t, Beat::End));
        Reveal {
            beats,
            points: r.points,
            count: (COUNT_FROM, COUNT_FROM + span),
            ticks: n.saturating_sub(1),
            gem_every,
        }
    }

    /// Its beats in order, each at its time.
    pub fn beats(&self) -> &[(f64, Beat)] {
        &self.beats
    }

    /// When `beat` lands, if it does.
    pub fn when(&self, beat: Beat) -> Option<f64> {
        self.beats
            .iter()
            .find(|(_, b)| *b == beat)
            .map(|(at, _)| *at)
    }

    /// How long ago `beat` landed at `t`: none before it (or without it).
    pub fn age(&self, beat: Beat, t: f64) -> Option<f64> {
        self.when(beat).map(|at| t - at).filter(|&a| a >= 0.0)
    }

    /// The first beat that matches `which`, and when.
    pub fn find(&self, which: impl Fn(&Beat) -> bool) -> Option<(f64, Beat)> {
        self.beats.iter().copied().find(|(_, b)| which(b))
    }

    /// The beats landing after `from` and up to `to`.
    pub fn between(&self, from: f64, to: f64) -> impl Iterator<Item = (f64, Beat)> + '_ {
        self.beats
            .iter()
            .copied()
            .filter(move |&(at, _)| at > from && at <= to)
    }

    /// When it is over.
    pub fn end(&self) -> f64 {
        self.beats.last().map_or(0.0, |(at, _)| *at)
    }

    /// When everything has settled: where a skip puts it.
    pub fn settled(&self) -> f64 {
        self.end() + SETTLE
    }

    /// When it goes still: its time stops there ([`TWINKLE`]).
    pub fn still(&self) -> f64 {
        self.settled() + TWINKLE
    }

    /// When the points count up: from, to.
    pub fn count(&self) -> (f64, f64) {
        self.count
    }

    /// How many ticks the count-up makes.
    pub fn ticks(&self) -> u32 {
        self.ticks
    }

    /// How far apart the gems come, in seconds.
    pub fn gem_every(&self) -> f64 {
        self.gem_every
    }

    /// The points shown at `t`: counting up from nothing, faster and
    /// faster, to all of them.
    pub fn points_at(&self, t: f64) -> u32 {
        let (from, to) = self.count;
        let u = ((t - from) / (to - from)).clamp(0.0, 1.0);
        (f64::from(self.points) * u.powf(SPEED_UP)).floor() as u32
    }
}

/// How a result shown moves at a moment: its reveal under way (every
/// refresh), settled and twinkling (calmly), or still (not at all).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    /// Its reveal under way, or its last beats settling.
    Revealing,
    /// Settled, its medal and gems twinkling.
    Twinkling,
    /// Still: nothing of it moves.
    Still,
}

/// A result shown: which one (`id`, one a result), how the song was sung,
/// its reveal, when it was first shown (ms on the core's clock) and how
/// far a skip put it ahead.
#[derive(Clone, Debug, PartialEq)]
pub struct Shown {
    /// Which result it is: another for each song sung.
    pub id: u64,
    /// How the song was sung.
    pub result: SongResult,
    /// Its reveal.
    pub reveal: Reveal,
    /// When it was first shown, ms.
    pub since_ms: u64,
    /// How far ahead of its clock a skip put it, seconds.
    pub ahead: f64,
}

impl Shown {
    /// `result`, the `id`th, shown from `now_ms`.
    pub fn new(result: SongResult, id: u64, now_ms: u64) -> Shown {
        Shown {
            id,
            reveal: Reveal::of(&result),
            result,
            since_ms: now_ms,
            ahead: 0.0,
        }
    }

    /// Its reveal's time at `now_ms`: seconds since it was shown, and what
    /// a skip put it ahead; held once it is still ([`Reveal::still`]).
    pub fn time(&self, now_ms: u64) -> f64 {
        let t = now_ms.saturating_sub(self.since_ms) as f64 / 1000.0 + self.ahead;
        t.min(self.reveal.still())
    }

    /// Whether its reveal is under way at `now_ms` (before its end).
    pub fn revealing(&self, now_ms: u64) -> bool {
        self.time(now_ms) < self.reveal.end()
    }

    /// How it moves at `now_ms`.
    pub fn motion(&self, now_ms: u64) -> Motion {
        let t = self.time(now_ms);
        if t < self.reveal.settled() {
            Motion::Revealing
        } else if t < self.reveal.still() {
            Motion::Twinkling
        } else {
            Motion::Still
        }
    }

    /// Its reveal completed at once at `now_ms`: everything shown and
    /// settled.
    pub fn skip(&mut self, now_ms: u64) {
        self.ahead += (self.reveal.settled() - self.time(now_ms)).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::super::Medal;
    use super::*;
    use crate::core::shine::Shone;

    fn phrase(tier: Option<Tier>) -> Shone {
        Shone {
            line: 0,
            end: 1.0,
            value: 0.8,
            top: 0.9,
            tier,
        }
    }

    fn result(points: u32, tiers: &[Option<Tier>], best: Option<u32>, whole: bool) -> SongResult {
        let (hit, words) = (40, 45);
        SongResult {
            title: "Song".into(),
            points,
            hit,
            words,
            streak: 17,
            phrases: tiers.iter().map(|&t| phrase(t)).collect(),
            medal: whole.then(|| Medal::of(hit, words)).flatten(),
            best,
            whole,
            demo: false,
        }
    }

    fn kinds(r: &Reveal) -> Vec<Beat> {
        r.beats().iter().map(|(_, b)| *b).collect()
    }

    const GOLD: Beat = Beat::Verdict(Verdict::Medal(Medal::Gold));

    /// A Gold run with a new best: every beat once, in order, in time, the
    /// gems in song order; the count rising to the points at its end, its
    /// ticks coming faster and faster; the medal a fall after the gems.
    #[test]
    fn the_beats_come_in_order() {
        let tiers = [
            Some(Tier::TOP),
            None,
            Some(Tier::named("Good")),
            Some(Tier::TOP),
        ];
        let r = Reveal::of(&result(12_400, &tiers, Some(9_000), true));
        let times: Vec<f64> = r.beats().iter().map(|(at, _)| *at).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "{times:?}");
        let b = kinds(&r);
        let at = |beat: Beat| b.iter().position(|x| *x == beat).unwrap();
        assert_eq!(
            b[..4],
            [
                Beat::Paper,
                Beat::Title,
                Beat::Heading(Side::Left),
                Beat::Heading(Side::Right)
            ]
        );
        assert_eq!(r.when(Beat::Heading(Side::Right)), Some(HEADING_STARS[1]));
        assert_eq!(r.ticks(), 17);
        assert!(at(Beat::Title) < at(Beat::Tick(0)));
        assert!(at(Beat::Tick(16)) < at(Beat::Points));
        assert!(at(Beat::Points) < at(Beat::Words) && at(Beat::Words) < at(Beat::Streak));
        let gems: Vec<(usize, Option<Tier>)> = b
            .iter()
            .filter_map(|x| match x {
                Beat::Gem(i, t) => Some((*i, *t)),
                _ => None,
            })
            .collect();
        assert_eq!(gems, tiers.iter().copied().enumerate().collect::<Vec<_>>());
        assert!(at(Beat::Streak) < at(Beat::Gem(0, Some(Tier::TOP))));
        let last_gem = r.when(Beat::Gem(3, Some(Tier::TOP))).unwrap();
        assert!(r.when(GOLD).unwrap() - last_gem >= DROP + 0.4);
        assert!(at(GOLD) < at(Beat::NewBest));
        assert_eq!(*b.last().unwrap(), Beat::End);
        assert!(r.end() > 3.0 && r.end() < 8.0, "a few seconds: {}", r.end());
        // The ticks closer and closer together.
        let ticks: Vec<f64> = r
            .beats()
            .iter()
            .filter(|(_, b)| matches!(b, Beat::Tick(..)))
            .map(|(at, _)| *at)
            .collect();
        let gaps: Vec<f64> = ticks.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.windows(2).all(|g| g[1] < g[0]), "{gaps:?}");
        // The count: none before, all of it from its end, rising.
        let (from, to) = r.count();
        assert_eq!(r.points_at(from - 0.1), 0);
        assert_eq!(r.points_at(to), 12_400);
        assert!(r.points_at((from + to) / 2.0) < 12_400 / 2);
        assert!(r.points_at(to - 0.05) > r.points_at(to - 0.3));
    }

    /// However many phrases, their gems take at most GEMS.
    #[test]
    fn many_gems_come_in_a_moment() {
        for n in [1, 10, 200] {
            let r = Reveal::of(&result(500, &vec![None; n], None, true));
            let first = r.when(Beat::Gem(0, None)).unwrap();
            let last = r.when(Beat::Gem(n - 1, None)).unwrap();
            assert!(last - first <= GEMS + 1e-9, "{n}: {}", last - first);
            assert!(r.gem_every() <= GEM_MOST);
        }
    }

    /// No phrases: no gems, the medal right after the streak. No medal:
    /// a cheer, then the best before, calmly. Not sung whole: the word
    /// that medals need the whole song, no new best (however high).
    #[test]
    fn a_result_without_phrases_medal_or_its_whole() {
        let r = Reveal::of(&result(300, &[], None, true));
        let b = kinds(&r);
        assert!(!b.iter().any(|x| matches!(x, Beat::Gem(..))));
        assert!(b.contains(&GOLD) && b.contains(&Beat::NewBest));

        let mut none = result(500, &[None], Some(2_000), true);
        none.medal = None;
        let b = kinds(&Reveal::of(&none));
        assert!(b.contains(&Beat::Verdict(Verdict::Cheer)) && b.contains(&Beat::Best(2_000)));
        assert!(!b.contains(&Beat::NewBest));

        let part = result(9_999, &[Some(Tier::named("Great"))], Some(100), false);
        let b = kinds(&Reveal::of(&part));
        assert!(b.contains(&Beat::Verdict(Verdict::Whole)) && b.contains(&Beat::Best(100)));
        assert!(!b.contains(&Beat::NewBest));
        // Not whole and never sung before: nothing of a best.
        let b = kinds(&Reveal::of(&result(10, &[], None, false)));
        assert!(!b.iter().any(|x| matches!(x, Beat::Best(_) | Beat::NewBest)));
        // No points: no ticks, all of nothing at once.
        let zero = Reveal::of(&result(0, &[], None, true));
        assert!(!kinds(&zero).iter().any(|x| matches!(x, Beat::Tick(..))));
        assert_eq!(zero.points_at(100.0), 0);
    }

    /// A skip completes the reveal at once: past its end and settled, and
    /// nothing lands after it; it twinkles a while, then its time stops.
    #[test]
    fn a_skip_completes_the_reveal_and_it_goes_still() {
        let mut s = Shown::new(
            result(1_000, &[Some(Tier::named("Good"))], None, true),
            1,
            5_000,
        );
        assert_eq!(s.time(5_000), 0.0);
        assert!(s.revealing(6_000));
        assert_eq!(s.motion(6_000), Motion::Revealing);
        s.skip(6_000);
        let t = s.time(6_000);
        assert!((t - s.reveal.settled()).abs() < 1e-9);
        assert!(!s.revealing(6_000));
        assert_eq!(s.motion(6_000), Motion::Twinkling);
        assert_eq!(s.reveal.between(t, s.time(60_000)).count(), 0);
        assert_eq!(s.reveal.points_at(t), 1_000);
        // Skipped again (or once over): nothing moves back.
        s.skip(7_000);
        assert!((s.time(7_000) - (t + 1.0)).abs() < 1e-9);
        // Still after its twinkle: its time held.
        let late = 6_000 + (TWINKLE * 1000.0) as u64 + 500;
        assert_eq!(s.motion(late), Motion::Still);
        assert_eq!(s.time(late), s.time(late + 60_000));
    }
}
