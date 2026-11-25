//! A song's result in Karaoke mode's picture ([`ResultShown`]): what was
//! shown and its reveal's time, for the stage to draw; and its words, kind
//! ones and never shaming (no medal is cheered on; a song not sung from
//! its start is told what medals take), as the stage writes them and as
//! the picture's lines where no stage draws it.

use std::sync::Arc;

use super::mode::line;
use super::{points_text, LineKind, LyricLine};
use crate::core::karaoke;
use crate::core::singing::{Medal, Shown, SongResult, Verdict};
use crate::core::State;

/// A song's result as the mode shows it: what was shown, its reveal's time
/// now ([`karaoke::result_now`]), and the song START plays.
#[derive(Clone, Debug, PartialEq)]
pub struct ResultShown {
    /// The result and its reveal.
    pub shown: Arc<Shown>,
    /// Seconds into its reveal.
    pub t: f64,
    /// The title of the song START plays (none queued: empty).
    pub next: String,
    /// Whether that is the song just sung, again (the queue played out).
    pub again: bool,
}

impl ResultShown {
    /// The words for the song START plays: the next one, the one just sung
    /// again, or none queued.
    pub fn next_words(&self) -> String {
        match (self.next.is_empty(), self.again) {
            (true, _) => "Queue some songs, then START".to_owned(),
            (false, true) => format!("Again: {}", self.next),
            (false, false) => format!("Next: {}", self.next),
        }
    }
}

/// The result shown now, at its reveal's time, with the song START plays:
/// the one held, or (none held, the queue played out) the queue's current
/// one, the song just sung.
pub fn result_now(state: &State) -> Option<ResultShown> {
    let (shown, t) = karaoke::result_now(state)?;
    let next = karaoke::song_index(state).map(|i| {
        let path = &state.queue.paths[i];
        state
            .details
            .get(path)
            .and_then(|d| d.title.clone())
            .unwrap_or_else(|| crate::core::shown_title(state, path))
    });
    Some(ResultShown {
        shown: shown.clone(),
        t,
        next: next.unwrap_or_default(),
        again: state.karaoke.held.is_none(),
    })
}

/// The words for how the song was sung under the medal (or what stands
/// for it): its name, and a line under it.
pub fn verdict_words(r: &SongResult) -> (&'static str, &'static str) {
    match r.verdict() {
        Verdict::Whole => (
            "Nice singing!",
            "Sing it from the start for a medal and a best",
        ),
        Verdict::Medal(Medal::Gold) => ("GOLD", "Superstar!"),
        Verdict::Medal(Medal::Silver) => ("SILVER", "Brilliant!"),
        Verdict::Medal(Medal::Bronze) => ("BRONZE", "Well sung!"),
        Verdict::Cheer if f64::from(r.hit) >= 0.4 * f64::from(r.words) => {
            ("So close!", "Bronze is just a few more words away")
        }
        Verdict::Cheer => ("Good warm-up!", "Sing it again and watch it grow"),
        Verdict::Silent => (
            "Silent night?",
            "Did the mic fall asleep? Wake it up and sing along",
        ),
    }
}

/// The Yodel Diploma's printed words (the stage hands the result out as
/// one): its heading, the lines round the song's title (yodelled duly, or
/// with nothing scored very quietly), its entries' names, what the seal's
/// place, the signature and the date are marked with, and the rubber stamp
/// a silent song gets. Nobody's name: it certifies the song sung. (The
/// grades are the stage's, with each verdict's look.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiplomaWords {
    /// Its heading.
    pub heading: &'static str,
    /// The line over the song's title.
    pub certify: &'static str,
    /// The line under it: yodelled duly, and very quietly.
    pub yodelled: (&'static str, &'static str),
    /// Its entries: the points, the words hit, the best streak.
    pub entries: [&'static str; 3],
    /// Where the seal goes, while it waits.
    pub seal: &'static str,
    /// Under the signature, and under the date.
    pub signed: (&'static str, &'static str),
    /// The rubber stamp of a song where nothing was scored.
    pub asleep: &'static str,
}

/// The Yodel Diploma's printed words.
pub const DIPLOMA: DiplomaWords = DiplomaWords {
    heading: "Yodel Diploma",
    certify: "This is to certify that the song",
    yodelled: (
        "has been duly yodelled, with these results",
        "has been yodelled (very, very quietly), with these results",
    ),
    entries: ["Points", "Words hit", "Best streak"],
    seal: "L.S.",
    signed: ("Chief Yodel Examiner", "Date"),
    asleep: "Zzz…",
};

/// A day as the diploma writes it: "24 November 2026".
pub fn date_words(year: i32, month: u32, day: u32) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month = MONTHS[(month.clamp(1, 12) - 1) as usize];
    format!("{day} {month} {year}")
}

/// The words under the best: a new one's (its first score, or how much it
/// went up from), or the one before; where the demo key sang, that it is
/// none.
pub fn best_words(r: &SongResult) -> String {
    if r.demo {
        return "Sung by the demo key: not a best".to_owned();
    }
    match (r.new_best(), r.best) {
        (true, None) => "Your first score!".to_owned(),
        (true, Some(b)) => format!("Up from {}", points_text(b)),
        (false, Some(b)) => format!("Your best: {}", points_text(b)),
        (false, None) => String::new(),
    }
}

/// The result as the picture's lines, all of it at once, `next` the words
/// for the song START plays ([`ResultShown::next_words`]): what a window
/// that draws no stage shows of it (the stage draws it instead where it
/// draws).
pub fn plain_lines(r: &SongResult, next: &str) -> Vec<LyricLine> {
    let (name, verdict) = verdict_words(r);
    let counts = format!(
        "{} points · {} of {} words · {} in a row",
        points_text(r.points),
        r.hit,
        r.words,
        r.streak
    );
    [
        (r.title.clone(), LineKind::Info),
        (format!("{name} {verdict}"), LineKind::Current),
        (counts, LineKind::Next),
        (best_words(r), LineKind::Info),
        (format!("{next} · START"), LineKind::Info),
    ]
    .into_iter()
    .filter(|(text, _)| !text.is_empty())
    .map(|(text, kind)| line(text, 0.0, kind))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(medal: Option<Medal>, whole: bool, best: Option<u32>) -> SongResult {
        SongResult {
            title: "A Song".into(),
            points: 12_400,
            hit: 40,
            words: 45,
            streak: 17,
            phrases: Vec::new(),
            medal,
            best,
            whole,
            demo: false,
        }
    }

    /// Without a stage, the result is its words: the title, the verdict,
    /// the numbers, the best, the next song and START.
    #[test]
    fn plain_lines_say_it_all() {
        let lines = plain_lines(&result(Some(Medal::Silver), true, None), "Next Song");
        let text: Vec<String> = lines.into_iter().map(|l| l.text).collect();
        assert_eq!(text[0], "A Song");
        assert!(text[1].starts_with("SILVER"), "{text:?}");
        assert!(text.iter().any(|t| t.contains("40 of 45 words")));
        let last = text.last().unwrap();
        assert!(last.contains("Next Song") && last.ends_with("START"));
    }

    /// The song START plays: the next, the one just sung again after the
    /// last, or none queued.
    #[test]
    fn the_next_words_name_the_song_start_plays() {
        let shown = |next: &str, again| ResultShown {
            shown: Arc::new(Shown::new(result(None, true, None), 1, 0)),
            t: 0.0,
            next: next.into(),
            again,
        };
        assert_eq!(shown("B", false).next_words(), "Next: B");
        assert_eq!(shown("A", true).next_words(), "Again: A");
        assert_eq!(shown("", true).next_words(), "Queue some songs, then START");
    }

    #[test]
    fn the_words_are_kind() {
        let mut r = result(None, true, Some(100));
        r.hit = 5;
        let (name, line) = verdict_words(&r);
        assert!(!name.to_lowercase().contains("no ") && !line.to_lowercase().contains("no "));
        let (_, line) = verdict_words(&result(None, false, None));
        assert!(line.contains("from the start"));
        assert_eq!(best_words(&result(None, true, None)), "Your first score!");
        assert_eq!(best_words(&result(None, false, None)), "");
        assert!(best_words(&result(None, true, Some(20_000))).starts_with("Your best"));
        // Nothing scored: a wink at the microphone, cheeky, never shaming.
        let mut silent = result(None, true, Some(100));
        (silent.points, silent.hit) = (0, 0);
        let (name, line) = verdict_words(&silent);
        assert_eq!(name, "Silent night?");
        assert!(line.contains("mic") && !line.to_lowercase().contains("no "));
        assert_eq!(best_words(&silent), "Your best: 100");
    }

    /// The diploma writes the day out, and names nobody.
    #[test]
    fn the_diploma_s_words() {
        assert_eq!(date_words(2026, 11, 24), "24 November 2026");
        let words = format!("{DIPLOMA:?}").to_lowercase();
        assert!(!words.contains("name") && !words.contains("bearer"));
    }
}
