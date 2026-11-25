//! How a song was sung, shown when it ends ([`SongResult`]), its medal and
//! what it comes to ([`Verdict`]), and the words for that: kind ones,
//! never shaming (no medal is cheered on; a song not sung from its start
//! is told what medals take), as the result is shown and written in on its
//! diploma ([`verdict_words`], [`Verdict::grade`]).

use super::Tally;
use crate::core::shine::{Shone, Tier};

/// A medal for a song sung: by the words hit (not by the points, which
/// the streaks multiply).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Medal {
    /// Half of them.
    Bronze,
    /// 70 %.
    Silver,
    /// 85 %.
    Gold,
}

impl Medal {
    /// The medal for `hit` of `words` words, if any.
    pub fn of(hit: u32, words: u32) -> Option<Medal> {
        if words == 0 {
            return None;
        }
        let share = f64::from(hit) / f64::from(words);
        if share >= 0.85 {
            Some(Medal::Gold)
        } else if share >= 0.7 {
            Some(Medal::Silver)
        } else if share >= 0.5 {
            Some(Medal::Bronze)
        } else {
            None
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Medal::Bronze => "Bronze",
            Medal::Silver => "Silver",
            Medal::Gold => "Gold",
        }
    }
}

/// What a song sung comes to: a medal, a cheer without one, (not sung
/// from its start) a word that medals and bests need the whole song, or
/// (nothing scored) a wink at the microphone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// The medal won.
    Medal(Medal),
    /// Sung whole, no medal this time.
    Cheer,
    /// Not sung from its start.
    Whole,
    /// Nothing scored: not a word hit, or none heard at all.
    Silent,
}

/// The name of a song not sung from its start, and of one with nothing
/// scored.
const WHOLE: &str = "Nice singing!";
const SILENT: &str = "Silent night?";

impl Verdict {
    /// The grade written in on the diploma: a medal as its Latin, a cheer
    /// as a line of its own, else the verdict's name.
    pub fn grade(self) -> &'static str {
        match self {
            Verdict::Medal(Medal::Gold) => "summa cum laude",
            Verdict::Medal(Medal::Silver) => "magna cum laude",
            Verdict::Medal(Medal::Bronze) => "cum laude",
            Verdict::Cheer => "participated with great enthusiasm",
            Verdict::Whole => WHOLE,
            Verdict::Silent => SILENT,
        }
    }
}

/// The words for how `r` was sung under the medal (or what stands for
/// it): its name, and a line under it.
pub fn verdict_words(r: &SongResult) -> (&'static str, &'static str) {
    match r.verdict() {
        Verdict::Whole => (WHOLE, "Sing it from the start for a medal and a best"),
        Verdict::Medal(Medal::Gold) => ("GOLD", "Superstar!"),
        Verdict::Medal(Medal::Silver) => ("SILVER", "Brilliant!"),
        Verdict::Medal(Medal::Bronze) => ("BRONZE", "Well sung!"),
        Verdict::Cheer if f64::from(r.hit) >= 0.4 * f64::from(r.words) => {
            ("So close!", "Bronze is just a few more words away")
        }
        Verdict::Cheer => ("Good warm-up!", "Sing it again and watch it grow"),
        Verdict::Silent => (SILENT, "Did the mic fall asleep? Wake it up and sing along"),
    }
}

/// How a song was sung, shown when it ends.
#[derive(Clone, Debug, PartialEq)]
pub struct SongResult {
    /// The song's title.
    pub title: String,
    /// The points.
    pub points: u32,
    /// The words hit.
    pub hit: u32,
    /// The words scored.
    pub words: u32,
    /// The most words hit in a row.
    pub streak: u32,
    /// Each phrase that shone, in order: its shine and tier.
    pub phrases: Vec<Shone>,
    /// The medal won, if any (none unless sung whole).
    pub medal: Option<Medal>,
    /// The best score of the song before (none: the first).
    pub best: Option<u32>,
    /// Whether the song was sung whole (from its start): only then a medal
    /// and a best.
    pub whole: bool,
    /// Whether the demo key sang some of it ([`super::Demo`]): shown all
    /// the same, never a best.
    pub demo: bool,
}

impl SongResult {
    /// How `title` was sung, by its `tally` and the `phrases` that shone,
    /// the song having `notes` words on notes and `best` the best before:
    /// sung whole where most of its words were heard (from its start), and
    /// only then a medal by the words hit.
    pub fn new(
        title: String,
        tally: &Tally,
        phrases: Vec<Shone>,
        notes: usize,
        best: Option<u32>,
    ) -> SongResult {
        let (hit, words) = (tally.hit_words, tally.scored_words);
        let whole = u64::from(words) * 10 >= notes as u64 * 9;
        SongResult {
            title,
            points: tally.score,
            hit,
            words,
            streak: tally.best_streak,
            phrases,
            medal: whole.then(|| Medal::of(hit, words)).flatten(),
            best,
            whole,
            demo: tally.demo,
        }
    }

    /// Whether it is a new best: a medal or a cheer (sung whole, something
    /// scored, none of it by the demo key), beating the best before (or the
    /// song's first score).
    pub fn new_best(&self) -> bool {
        !self.demo
            && matches!(self.verdict(), Verdict::Medal(_) | Verdict::Cheer)
            && self.best.is_none_or(|b| self.points > b)
    }

    /// What it comes to.
    pub fn verdict(&self) -> Verdict {
        match (self.points, self.whole, self.medal) {
            (0, _, _) => Verdict::Silent,
            (_, false, _) => Verdict::Whole,
            (_, true, Some(m)) => Verdict::Medal(m),
            (_, true, None) => Verdict::Cheer,
        }
    }

    /// The best tier a phrase reached, if any.
    pub fn best_tier(&self) -> Option<Tier> {
        self.phrases.iter().filter_map(|p| p.tier).max()
    }

    /// How many phrases reached each tier (as [`Tier::ALL`]: Good first).
    pub fn tier_counts(&self) -> [u32; Tier::COUNT] {
        let mut out = [0; Tier::COUNT];
        for t in self.phrases.iter().filter_map(|p| p.tier) {
            out[t.index()] += 1;
        }
        out
    }
}
