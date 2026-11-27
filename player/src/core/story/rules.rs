//! Every rule a plan is held to, in one place: the counts the prompt asks
//! for and those the answer is taken with ([`Count`]: the prompt says
//! [`Count::said`], the check says [`Count::check`]), how long a scene
//! lasts, and the answer's length, which the scenes asked for are held
//! within.

use super::Song;

/// A count of things a plan has: what they are, how many the prompt asks
/// for and how many an answer may have (a little more leeway than asked).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    /// What they are, as the prompt and the errors name them.
    pub what: &'static str,
    /// How many are asked for, fewest and most (fewest 0: "at most").
    pub asked: (usize, usize),
    /// How many are taken.
    pub taken: (usize, usize),
}

impl Count {
    /// How many are asked for, in words: `5`, `3 to 5`, `at most 6`.
    pub fn asked(&self) -> String {
        match self.asked {
            (0, most) => format!("at most {most}"),
            (a, b) if a == b => a.to_string(),
            (a, b) => format!("{a} to {b}"),
        }
    }

    /// The count and what they are, as the prompt says it: `3 to 5 motifs`.
    pub fn said(&self) -> String {
        format!("{} {}", self.asked(), self.what)
    }

    /// `n` of them taken, or why not: `2 motifs: 3 to 5 were asked for`.
    pub fn check(&self, n: usize) -> Result<(), String> {
        if n < self.taken.0 || n > self.taken.1 {
            return Err(format!(
                "{n} {}: {} were asked for",
                self.what,
                self.asked()
            ));
        }
        Ok(())
    }
}

/// What a character's name is, as the prompt and the errors say it.
pub const NAME: &str = "one or two lowercase words";

/// Whether `name` is a character's name: [`NAME`] (letters, figures,
/// apostrophes and hyphens; one space at most, none at either end).
pub fn is_name(name: &str) -> bool {
    let fine = |ch: char| ch.is_lowercase() || ch.is_ascii_digit() || " '-".contains(ch);
    !name.is_empty()
        && name.split(' ').count() <= 2
        && name.chars().all(fine)
        && name.trim() == name
}

/// The characters.
pub const CHARACTERS: Count = Count {
    what: "characters",
    asked: (0, 6),
    taken: (0, 6),
};
/// The words of a picture's prompt.
pub const PROMPT_WORDS: Count = Count {
    what: "words",
    asked: (15, 35),
    taken: (5, 70),
};
/// The style's keywords for the image model.
pub const KEYWORDS: Count = Count {
    what: "comma-separated keywords",
    asked: (6, 10),
    taken: (3, 14),
};
/// The palette's colours.
pub const PALETTE: Count = Count {
    what: "colours",
    asked: (5, 5),
    taken: (3, 6),
};
/// The motifs.
pub const MOTIFS: Count = Count {
    what: "recurring objects or symbols from the lyrics",
    asked: (3, 5),
    taken: (1, 6),
};
/// The mood words.
pub const MOOD: Count = Count {
    what: "mood words",
    asked: (2, 4),
    taken: (1, 5),
};

/// Every count but the scenes' (which depend on the song: [`scenes`]).
pub const COUNTS: [Count; 6] = [CHARACTERS, PROMPT_WORDS, KEYWORDS, PALETTE, MOTIFS, MOOD];

/// How long a scene lasts, about, in seconds.
pub const SCENE_SECONDS: (f64, f64) = (10.0, 20.0);
/// The fewest scenes a song is asked for.
pub const FEWEST_SCENES: usize = 3;
/// The most tokens an answer may take.
pub const MOST_ANSWER: usize = 5_500;
/// What a scene takes of the answer, about (measured: 4 425 tokens for
/// 30 scenes with the rest), and what the rest takes.
pub const TOKENS_A_SCENE: usize = 150;
/// What the story, the characters, the look and the shaders take.
pub const ANSWER_REST: usize = 800;
/// How long a line with no end is taken to last, seconds.
pub const UNTIMED_LINE_SECONDS: f64 = 5.0;

/// The scenes `song` is asked for: one every [`SCENE_SECONDS`], at least
/// [`FEWEST_SCENES`], never more than it has lines nor than fit in
/// [`MOST_ANSWER`]; taken from half the fewest to half as many again as
/// the most (within the lines and the answer's room). A song with no
/// lines is never asked (`crate::work::story::plan`).
pub fn scenes(song: &Song) -> Count {
    let lines = song.lines.len();
    let fit = (MOST_ANSWER - ANSWER_REST) / TOKENS_A_SCENE;
    let most = lines.min(fit);
    let lo = ((song.duration / SCENE_SECONDS.1).round() as usize)
        .max(FEWEST_SCENES)
        .min(most);
    let hi = ((song.duration / SCENE_SECONDS.0).round() as usize)
        .max(lo + 2)
        .min(most);
    Count {
        what: "scenes",
        asked: (lo, hi),
        taken: (lo.div_ceil(2), (hi + hi / 2).min(most)),
    }
}
