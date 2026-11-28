//! Every rule a plan is held to, in one place: the counts the prompt asks
//! for and the most an answer is taken with ([`Count`]: the prompt says
//! [`Count::said`], the reading keeps at most [`Count::most`]), the most
//! scenes a plan has (never said: a number in the prompt anchors the
//! model), the answers' and the follow-up's rooms, and what a character's
//! name is.

/// A count of things a plan has: what they are, how many the prompt asks
/// for, and the most an answer is taken with (those past it left out).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    /// What they are, as the prompt and the notes name them.
    pub what: &'static str,
    /// How many are asked for, fewest and most (fewest 0: "at most").
    pub asked: (usize, usize),
    /// The most taken.
    pub most: usize,
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

    /// `all` held to its most, those past it left out: what was left out,
    /// said, where any was.
    pub fn hold<T>(&self, all: &mut Vec<T>) -> Option<String> {
        let n = all.len();
        (n > self.most).then(|| {
            all.truncate(self.most);
            format!("{n} {}: the first {} kept", self.what, self.most)
        })
    }
}

/// What a character's name is, as the prompt says it.
pub const NAME: &str = "one or two lowercase words";

/// `written` as a character's name ([`NAME`]): trimmed, in lower case, an
/// underscore a space, only letters, figures, apostrophes, hyphens and
/// single spaces, its first two words; empty where nothing is left.
pub fn name(written: &str) -> String {
    let kept: String = (written.trim().to_lowercase().chars())
        .map(|c| if c == '_' { ' ' } else { c })
        .filter(|&c| c.is_alphanumeric() || " '-".contains(c))
        .collect();
    let words: Vec<&str> = kept.split_whitespace().take(2).collect();
    words.join(" ")
}

/// Names that say a group (whole words, case ignored): a crowd is never a
/// character (it would be drawn as one portrait); it goes in the scenes'
/// prompts.
pub const GROUPS: &[&str] = &[
    "crowd",
    "crowds",
    "people",
    "group",
    "audience",
    "choir",
    "army",
    "gang",
    "crew",
    "band",
    "family",
    "friends",
    "strangers",
    "dancers",
    "children",
    "kids",
    "soldiers",
    "villagers",
    "everyone",
    "everybody",
    "others",
    "lovers",
    "couple",
    "we",
    "us",
    "they",
    "them",
];

/// Whether the name `name` says a group ([`GROUPS`]).
pub fn is_group(name: &str) -> bool {
    (name.split(|c: char| !c.is_alphanumeric()))
        .any(|w| GROUPS.iter().any(|g| g.eq_ignore_ascii_case(w)))
}

/// A name as names are compared: lower case, without spaces, hyphens and
/// underscores.
fn plain(s: &str) -> String {
    (s.chars())
        .filter(|c| !" -_".contains(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// How near `candidate` is to `written` as [`nearest`] orders them
/// (smaller is nearer): the same but for case and marks first, then one
/// holding the other (of three characters at least), then the fewest
/// characters changed.
fn nearness(written: &str, candidate: &str) -> (bool, bool, usize) {
    let (w, p) = (plain(written), plain(candidate));
    let held = |a: &str, b: &str| b.chars().count() >= 3 && a.contains(b);
    let holds = held(&p, &w) || held(&w, &p);
    (p != w, !holds, changes(&w, &p))
}

/// The nearest of `among` to `written` (none where `among` is empty), as
/// [`nearness`] orders them, the first of equals: what a name no one
/// knows is taken as, where the model was not asked again or answered no
/// better.
pub fn nearest<'a>(written: &str, among: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    (among.into_iter()).min_by_key(|c| nearness(written, c))
}

/// The thing of `all` named `written` (by `name`, case and marks
/// ignored), else the nearest by name ([`nearest`]); none where `all` is
/// empty.
pub fn nearest_by<'a, T>(written: &str, all: &'a [T], name: impl Fn(&T) -> &str) -> Option<&'a T> {
    all.iter().min_by_key(|t| nearness(written, name(t)))
}

/// Whether `candidate` is near enough to `written` to be taken for it: the
/// same but for case and marks, one holding the other, or a third of its
/// characters changed at most. Where none is, a name no one knows is left
/// out rather than taken as something else.
pub fn near(written: &str, candidate: &str) -> bool {
    let (other, apart, changed) = nearness(written, candidate);
    !other || !apart || changed * 3 <= plain(written).chars().count()
}

/// The fewest characters added, taken away or changed to make `a` into
/// `b` (Levenshtein's distance).
fn changes(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut last = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let here = row[j + 1];
            row[j + 1] = (last + usize::from(ca != cb)).min(row[j] + 1).min(here + 1);
            last = here;
        }
    }
    row[b.len()]
}

/// The characters.
pub const CHARACTERS: Count = Count {
    what: "characters",
    asked: (0, 6),
    most: 6,
};
/// The words of a picture's prompt.
pub const PROMPT_WORDS: Count = Count {
    what: "words",
    asked: (15, 35),
    most: 70,
};
/// The style's keywords for the image model.
pub const KEYWORDS: Count = Count {
    what: "comma-separated keywords",
    asked: (6, 10),
    most: 14,
};
/// The palette's colours.
pub const PALETTE: Count = Count {
    what: "colours",
    asked: (5, 5),
    most: 6,
};
/// The motifs.
pub const MOTIFS: Count = Count {
    what: "recurring objects or symbols from the lyrics",
    asked: (3, 5),
    most: 6,
};
/// The mood words.
pub const MOOD: Count = Count {
    what: "mood words",
    asked: (2, 4),
    most: 5,
};

/// Every count the prompt says.
pub const COUNTS: [Count; 6] = [CHARACTERS, PROMPT_WORDS, KEYWORDS, PALETTE, MOTIFS, MOOD];

/// The most scenes a plan has: those past it are merged into the last
/// kept (it shows on to the song's end), so every line keeps a picture.
/// Never said in the prompt: the model decides how many scenes the song
/// needs (most take 5 to 10).
pub const MOST_SCENES: usize = 20;

/// The most tokens an answer may take (20 scenes take about 3 000, the
/// rest about 800).
pub const MOST_ANSWER: usize = 5_500;
/// The most tokens a song's prompt is given room for (the measured ones
/// take 9 268 to 9 480: the looks' catalogue in brief and the lyrics); a
/// longer one may be short of room for its thought on any card.
pub const MOST_PROMPT: usize = 16_384;
/// The most tokens the model's thought may take, where the context leaves
/// it more: a guard against a thought that never ends (the longest
/// measured was 1 743 tokens; one that never ends is cut off after about
/// 3 minutes on an RTX 5080).
pub const MOST_THOUGHT: usize = 16_384;
/// The most tokens the follow-up's questions take (each a sentence or
/// two, the nearest few choices listed; the styles all).
pub const MOST_ASK: usize = 2_048;
/// The most tokens the model's thought over the follow-up may take, where
/// the context leaves it more.
pub const FOLLOW_UP_THOUGHT: usize = 4_096;
/// The most tokens the follow-up's answer (a small patch) may take.
pub const MOST_PATCH: usize = 1_024;
/// How long a line with no end is taken to last, seconds.
pub const UNTIMED_LINE_SECONDS: f64 = 5.0;
