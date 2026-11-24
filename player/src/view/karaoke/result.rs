//! A song's result in Karaoke mode's picture ([`ResultShown`]): what was
//! shown and its reveal's time, for the stage to draw; and its words, kind
//! ones and never shaming (no medal is cheered on; a song not sung from
//! its start is told what medals take), as the stage writes them and as
//! the picture's lines where no stage draws it.

use std::sync::Arc;

use super::{line, points_text, LineKind, LyricLine};
use crate::core::karaoke;
use crate::core::singing::{Medal, Shown, SongResult, Verdict};
use crate::core::State;

/// A song's result as the mode shows it: what was shown, its reveal's time
/// now ([`karaoke::result_now`]), and the song coming next.
#[derive(Clone, Debug, PartialEq)]
pub struct ResultShown {
    /// The result and its reveal.
    pub shown: Arc<Shown>,
    /// Seconds into its reveal.
    pub t: f64,
    /// The next song's title.
    pub next: String,
}

/// The result shown now, at its reveal's time, with the song coming next.
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
    }
}

/// The words under the best: a new one's (its first score, or how much it
/// went up from), or the one before.
pub fn best_words(r: &SongResult) -> String {
    match (r.new_best(), r.best) {
        (true, None) => "Your first score!".to_owned(),
        (true, Some(b)) => format!("Up from {}", points_text(b)),
        (false, Some(b)) => format!("Your best: {}", points_text(b)),
        (false, None) => String::new(),
    }
}

/// The words for the song coming next.
pub fn next_words(next: &str) -> String {
    if next.is_empty() {
        "Queue some songs, then START".to_owned()
    } else {
        format!("Next: {next}")
    }
}

/// The result as the picture's lines, all of it at once, `next` the song
/// coming: what a window that draws no stage shows of it (the stage draws
/// it instead where it draws).
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
        (format!("{} · START", next_words(next)), LineKind::Info),
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
    }
}
