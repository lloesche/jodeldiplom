//! Who is in a plan's pictures, and how they look there: a character's
//! looks are its appearance alone ([`appearance`]: an action or a habit in
//! them would be drawn into its portrait and so into every scene it is
//! in), and a scene's cast is who its prompt names ([`cast`]: a
//! character's portrait is a scene's reference only where the scene shows
//! it).

use super::Character;

/// The words that make a clause of a character's looks an action where
/// one is its head ("smoking a cigarette", "holding a guitar").
const ACTIONS: &[&str] = &[
    "smoking", "smokes", "holding", "holds", "carrying", "carries", "playing", "plays", "singing",
    "sings", "dancing", "dances", "drinking", "drinks", "eating", "eats", "reading", "reads",
    "writing", "writes", "walking", "walks", "running", "runs", "sitting", "sits", "leaning",
    "leans", "staring", "stares", "gazing", "gazes", "glancing", "smiling", "smiles", "smirking",
    "laughing", "laughs", "crying", "cries", "frowning", "waving", "pointing", "posing", "poses",
];

/// A stance: an action at a clause's head only where no height follows
/// it ("standing by the door", not "stands at six feet").
const STANCES: &[&str] = &["standing", "stands"];

/// The words of a height (a figure too).
const HEIGHTS: &[&str] = &[
    "tall", "short", "feet", "foot", "ft", "cm", "m", "metres", "meters", "inches", "inch",
];

/// What may follow an action at the head of a clause, where the clause is
/// that action (an object, a place, nothing at all, a manner); where
/// anything else follows, the word describes ("dancing eyes", "running
/// shoes").
const AFTER_ACTION: &[&str] = &[
    "a",
    "an",
    "the",
    "his",
    "her",
    "their",
    "its",
    "my",
    "your",
    "some",
    "on",
    "at",
    "in",
    "into",
    "with",
    "to",
    "by",
    "around",
    "against",
    "up",
    "down",
    "out",
    "off",
    "over",
    "through",
    "cigarette",
    "cigarettes",
];

/// The words of a habit, left out of a clause ("always dressed in black":
/// "dressed in black").
const HABITS: &[&str] = &[
    "always",
    "often",
    "usually",
    "constantly",
    "perpetually",
    "forever",
];

/// The words passed over before a clause's head ("is smoking", "while
/// smoking").
const FILLERS: &[&str] = &[
    "is",
    "are",
    "with",
    "while",
    "always",
    "often",
    "usually",
    "constantly",
    "perpetually",
    "forever",
];

/// The one prop a clause of looks is left out for wherever it is (the
/// reported case: "perpetually smoking" drawn into every picture).
const PROPS: &[&str] = &["cigarette", "cigarettes"];

/// The words that join two clauses of looks.
const JOINS: &[&str] = &["and", "while", "who"];

/// `text`'s words as names and prompts are compared: in lower case, a
/// curly apostrophe a straight one, a hyphen a space (so "Jean-Paul" is
/// "jean paul"), letters, figures and apostrophes inside a word kept, a
/// trailing `'s` left out.
fn words(text: &str) -> Vec<String> {
    (text.replace('’', "'").replace('-', " ").to_lowercase())
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\''))
        .map(|w| w.strip_suffix("'s").unwrap_or(w).to_owned())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Whether the clause `words` (in lower case) is an action or a prop: its
/// head (after the fillers) an action that something of an action
/// follows (or nothing; a stance only where no height follows), or a prop
/// anywhere in it.
fn doing(words: &[&str]) -> bool {
    let bare = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let plain: Vec<String> = words.iter().map(|w| bare(w)).collect();
    if plain.iter().any(|w| PROPS.contains(&w.as_str())) {
        return true;
    }
    let mut rest = plain.iter().skip_while(|w| FILLERS.contains(&w.as_str()));
    let Some(head) = rest.next() else {
        return false;
    };
    let after: Vec<&String> = rest.collect();
    let height =
        |w: &&String| HEIGHTS.contains(&w.as_str()) || w.chars().any(|c| c.is_ascii_digit());
    if STANCES.contains(&head.as_str()) && after.iter().any(height) {
        return false;
    }
    (ACTIONS.contains(&head.as_str()) || STANCES.contains(&head.as_str()))
        && after
            .first()
            .is_none_or(|n| AFTER_ACTION.contains(&n.as_str()) || n.ends_with("ly"))
}

/// The words of `text` from its byte `from` with their byte spans.
fn spans(text: &str, from: usize) -> Vec<(usize, usize)> {
    let mut all = Vec::new();
    let mut start = None;
    for (k, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(k),
            (true, Some(s)) => {
                all.push((from + s, from + k));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        all.push((from + s, from + text.len()));
    }
    all
}

/// `looks` as appearance alone, the text as written but for what is left
/// out of it: each part (between a comma, a semicolon or a full stop
/// before a space) that is an action or a prop ([`doing`]) with the mark
/// before it; a clause after "and", "while" or "who" that is one, with
/// that word; the words of a habit. Never all of it (where nothing would
/// be left, the looks stay as written). And what was left out (none: the
/// looks as they were).
pub fn appearance(looks: &str) -> (String, Vec<String>) {
    // The parts: each from the mark before it (or the start) to its end.
    let bytes = looks.as_bytes();
    let mut marks: Vec<usize> = Vec::new();
    for (k, c) in looks.char_indices() {
        let full_stop = c == '.' && bytes.get(k + 1).is_none_or(|b| b.is_ascii_whitespace());
        if c == ',' || c == ';' || full_stop {
            marks.push(k);
        }
    }
    let mut parts: Vec<(usize, usize, usize)> = Vec::new();
    let mut from = 0;
    for &m in marks.iter().chain([&looks.len()]) {
        let cut = parts.last().map_or(0, |&(_, _, end)| end);
        parts.push((cut, from, m));
        from = m + 1;
    }
    let lower = |w: (usize, usize)| looks[w.0..w.1].to_lowercase();
    let mut gone: Vec<(usize, usize)> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for &(cut, start, end) in &parts {
        let words = spans(&looks[start..end], start);
        // The clauses: the part's head, then one after each join.
        let joins: Vec<usize> = (0..words.len())
            .filter(|&k| JOINS.contains(&lower(words[k]).as_str()))
            .collect();
        let head_end = joins.first().copied().unwrap_or(words.len());
        let said =
            |ws: &[(usize, usize)]| -> Vec<String> { ws.iter().map(|&w| lower(w)).collect() };
        let text_of = |ws: &[(usize, usize)]| -> String {
            ws.first()
                .map_or(String::new(), |f| looks[f.0..ws[ws.len() - 1].1].to_owned())
        };
        let is_doing = |ws: &[(usize, usize)]| {
            let w = said(ws);
            !ws.is_empty() && doing(&w.iter().map(String::as_str).collect::<Vec<_>>())
        };
        let head = &words[..head_end];
        let mut kept_words: Vec<(usize, usize)> = Vec::new();
        if is_doing(head) {
            // The whole part, or its head with the mark before it.
            let stop = if joins.is_empty() {
                end
            } else {
                head[head.len() - 1].1
            };
            gone.push((cut, stop));
            out.push(text_of(head));
        } else {
            kept_words.extend_from_slice(head);
        }
        for (j, &at) in joins.iter().enumerate() {
            let until = joins.get(j + 1).copied().unwrap_or(words.len());
            let clause = &words[at + 1..until];
            if is_doing(clause) {
                let stop = clause.last().map_or(words[at].1, |w| w.1);
                gone.push((words[at].0, stop));
                out.push(text_of(clause));
            } else {
                kept_words.extend_from_slice(&words[at..until]);
            }
        }
        for &w in &kept_words {
            if HABITS.contains(&lower(w).as_str()) {
                gone.push(w);
                out.push(looks[w.0..w.1].to_owned());
            }
        }
    }
    if gone.is_empty() {
        return (looks.to_owned(), Vec::new());
    }
    gone.sort_unstable();
    let mut left = String::new();
    let mut at = 0;
    for (a, b) in gone {
        if a >= at {
            left.push_str(&looks[at..a]);
            at = b;
        } else {
            at = at.max(b);
        }
    }
    left.push_str(&looks[at..]);
    let tidy = tidied(&left);
    if !tidy.chars().any(char::is_alphanumeric) {
        return (looks.to_owned(), Vec::new());
    }
    (tidy, out)
}

/// `text` tidied after parts were cut out of it: its spaces single, none
/// before a mark, no mark or join word at its start, no mark twice.
fn tidied(text: &str) -> String {
    let mut s = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [
        (" ,", ","),
        (" ;", ";"),
        (" .", "."),
        (",,", ","),
        (", ,", ","),
    ] {
        while s.contains(from) {
            s = s.replace(from, to);
        }
    }
    loop {
        let trimmed = s.trim_start_matches([',', ';', '.', ' ']);
        let first = trimmed.split(' ').next().unwrap_or_default();
        let after_join = JOINS
            .contains(&first.to_lowercase().as_str())
            .then(|| trimmed[first.len()..].trim_start());
        let next = after_join.unwrap_or(trimmed).to_owned();
        if next == s {
            return s;
        }
        s = next;
    }
}

/// Whether `prompt` names the character `name`: its words ([`words`]),
/// one after another, in the prompt's.
pub fn names(prompt: &str, name: &str) -> bool {
    let (prompt, name) = (words(prompt), words(name));
    !name.is_empty() && (prompt.windows(name.len())).any(|w| w == name.as_slice())
}

/// The cast of a scene whose prompt is `prompt` and that lists `listed`:
/// the characters described (`characters`) its prompt names, in their
/// order; and, in words, who was listed but is not in the prompt, and who
/// is in it but was not listed.
pub fn cast(
    prompt: &str,
    listed: &[String],
    characters: &[Character],
) -> (Vec<String>, Vec<String>) {
    let seen: Vec<String> = (characters.iter())
        .filter(|c| names(prompt, &c.name))
        .map(|c| c.name.clone())
        .collect();
    let mut said = Vec::new();
    for name in listed.iter().filter(|n| !seen.contains(n)) {
        said.push(format!(
            "{name:?} is listed but not in its prompt: no reference"
        ));
    }
    for name in seen.iter().filter(|n| !listed.contains(n)) {
        said.push(format!(
            "{name:?} is in its prompt but was not listed: seen"
        ));
    }
    (seen, said)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A character's looks keep its appearance alone: an action or a prop
    /// at the head of a clause left out (the Echt plan's "perpetually
    /// smoking"), a habit's word left out of its clause, what only looks
    /// like an action kept, and never all of it.
    #[test]
    fn looks_are_appearance_and_the_cast_follows_the_prompt() {
        let cases: [(&str, &str); 19] = [
            (
                "Female, late 20s, slender, dark wavy hair, wearing a simple black turtleneck, perpetually smoking.",
                "Female, late 20s, slender, dark wavy hair, wearing a simple black turtleneck.",
            ),
            // What is kept is kept as written.
            (
                "a woman with salt and pepper hair, always smoking",
                "a woman with salt and pepper hair",
            ),
            (
                "dark hair, often barefoot, a black and white striped shirt",
                "dark hair, barefoot, a black and white striped shirt",
            ),
            ("1.80 m tall, always smoking", "1.80 m tall"),
            (
                "a man who is tall and thin, holding a pipe",
                "a man who is tall and thin",
            ),
            ("stands at six feet, wool coat", "stands at six feet, wool coat"),
            ("looking elderly, grey coat", "looking elderly, grey coat"),
            ("looks in her 30s", "looks in her 30s"),
            (
                "wearing a black turtleneck and always smoking",
                "wearing a black turtleneck",
            ),
            (
                "A tall woman with dark hair who is always smiling",
                "A tall woman with dark hair",
            ),
            ("Tall woman. Dark hair. Holding a cigarette.", "Tall woman. Dark hair."),
            ("Man, 60s, grey beard, holding a guitar, wool sweater", "Man, 60s, grey beard, wool sweater"),
            ("always dressed in black, often barefoot", "dressed in black, barefoot"),
            ("Man, reading glasses, running shoes", "Man, reading glasses, running shoes"),
            ("good-looking, tired-looking eyes", "good-looking, tired-looking eyes"),
            ("a crooked smile, a gap-toothed smile", "a crooked smile, a gap-toothed smile"),
            ("dancing eyes, a playing-card tattoo", "dancing eyes, a playing-card tattoo"),
            ("Woman, smoky eyes, a dancer's build", "Woman, smoky eyes, a dancer's build"),
            // Nothing left: the looks stay as written.
            ("Smoking a cigarette", "Smoking a cigarette"),
        ];
        for (looks, want) in cases {
            assert_eq!(appearance(looks).0, want, "{looks}");
        }
        assert_eq!(
            appearance("wearing a black turtleneck and always smoking").1,
            ["always smoking"]
        );
        let people = [
            Character {
                name: "elara".into(),
                looks: "Woman".into(),
            },
            Character {
                name: "old man".into(),
                looks: "Man".into(),
            },
            Character {
                name: "mary-jane".into(),
                looks: "Girl".into(),
            },
            Character {
                name: "o'brien".into(),
                looks: "Man".into(),
            },
        ];
        assert!(names("A street at dusk, Elara's coat in the wind", "elara"));
        assert!(!names(
            "A desolate landscape, a single small figure",
            "elara"
        ));
        assert!(names("the old man walks away", "old man"));
        assert!(names("Mary-Jane runs for the bus", "mary-jane"));
        assert!(names("O’Brien's boat at the quay", "o'brien"));
        assert!(names("Jean-Paul at the bar", "jean paul"));
        assert!(names("Jean Paul at the bar", "jean-paul"));
        // The Echt plan's scene 6: listed with her, its prompt without her.
        let (seen, said) = cast(
            "A close-up of a cracked, dry earth field under a heavy, overcast sky, the old man far off",
            &["elara".into()],
            &people,
        );
        assert_eq!(seen, ["old man"]);
        assert_eq!(said.len(), 2);
    }
}
