//! What the model is asked for a song's plan ([`prompt`]): the story from
//! the lyrics alone (the numbered lines with their times, the rules of a
//! scene), then the art from the file's tags, what the model knows of the
//! artist and the song, and the storyboard it has just written, then the
//! shaders from the looks' catalogue in brief
//! ([`jodel_looks::catalogue::brief`]), and the answer's JSON laid out
//! field by field. Every count it asks for is [`super::rules`]'s.
//!
//! The prompt is made the same way every time for the same song, so the
//! same song always gives the same plan.

use std::fmt::Write as _;

use jodel_looks::catalogue;

use super::rules::{self, CHARACTERS, KEYWORDS, MOOD, MOTIFS, PALETTE, PROMPT_WORDS};
use super::{Song, Style};

/// What the model is asked for `song`'s plan.
pub fn prompt(song: &Song) -> String {
    let scenes = rules::scenes(song);
    let last = song.lines.len().saturating_sub(1);
    let mut numbered = String::new();
    for (k, line) in song.lines.iter().enumerate() {
        let _ = writeln!(numbered, "{k}. [{:.1}] {}", line.start, line.text);
    }
    let mut about = vec![format!("title: {}", song.title)];
    if !song.artist.is_empty() {
        about.push(format!("artist: {}", song.artist));
    }
    if !song.album.is_empty() {
        about.push(format!("album: {}", song.album));
    }
    for (name, value) in &song.tags {
        about.push(format!("{name}: {value}"));
    }
    let styles: Vec<String> = (Style::ALL.iter())
        .map(|s| format!("\"{}\"", s.name()))
        .collect();
    let by = if song.artist.is_empty() {
        String::new()
    } else {
        format!(" by {}", song.artist)
    };
    let (short, long) = rules::SCENE_SECONDS;
    format!(
        r##"You direct an illustrated karaoke video for the song "{title}"{by}: still pictures, one after another, that tell the story of the lyrics while they are sung, with a look of their own over them. Write one JSON answer: first the story, then the art, then its shaders.

THE STORY comes from the lyrics alone. The song's tags and what you know of the artist play no part in it.

The lyrics, numbered lines with their start times in seconds (the song lasts {seconds:.0} seconds):
{numbered}
Rules for the story:
- What happens comes from the lyrics only: no events, people or places that are not in them.
- Plan {scenes}. Every numbered line belongs to exactly one scene. The scenes take consecutive lines in order, from line 0 to line {last}: none left out, none taken twice.
- Scene changes follow the song's sections (verse, chorus, bridge); a scene lasts about {short:.0} to {long:.0} seconds.
- The characters come from the lyrics (the "I", the "you", others named in the song), not from the singer: {characters}. Describe each once with fixed looks (age, gender, build, hair, clothes), so every picture can repeat them. A name is {name}.
- A scene's fields, each written once, in this order:
  - "first_line", "last_line": its lines. "section": the song's section (verse 1, chorus, ...).
  - "shot": "wide", "medium" or "close-up"; vary them.
  - "characters": the names of the characters seen (may be empty).
  - "shows": what the scene shows, one sentence.
  - "prompt": a concrete visual description for an image generator, {prompt_words}: place, time of day, action, composition, foreground and background. Name characters by their name only. No style words. Nothing written in the picture: never the words text, letters, sign, poster, logo, label, banner, billboard or caption (say "the border at night", not "a border sign").

THE ART comes once every scene is written: from the file's tags, what you yourself know of the artist, the album and the song (their visual identity, the era, the genre, the mood), and the story above.
The file's tags:
{about}
First say what you know of them and how sure you are. If you are sure, lean on it; if not, say so and go by the tags and the lyrics. Do not invent facts.
- "style": exactly one of {styles}. Pick the one that fits the artist, the genre, the era and the mood best, and be bold: every song should look strongly different.
- "style_keywords": {keywords} for the image generator (medium, lighting, texture, era); never text, lettering or posters.
- "palette": {palette} as "#rrggbb". "motifs": {motifs}.

THE SHADER, the very last: the shaders drawn over the pictures, from the catalogue below.
- "preset": the one id of the preset that fits the genre, the era and the mood best, exactly as the catalogue quotes it.
- "add": passes the preset has not, to add after its passes, in order; may be empty.
- "knobs": a few knobs moved, of the preset's passes or those added, pass by pass, each to a number within its range, such as {{"grain": {{"amount": 0.5}}}}; may be empty.
- "drive": what in the music moves a knob (or a pass's "mix") of those passes: each a pass, one of its knobs, a driver and an amount from -1 to 1, such as [{{"pass": "grain", "knob": "amount", "by": "bass", "amount": 0.6}}]; may be empty.
- "mood": {mood}.

{brief}
Answer only JSON, in this order:
{{"story": "two sentences on what the lyrics tell",
 "characters": [{{"name": "narrator", "looks": "..."}}],
 "scenes": [{{"first_line": 0, "last_line": 2, "section": "verse 1", "shot": "wide", "characters": ["narrator"], "shows": "...", "prompt": "..."}}],
 "art": {{"knowledge": "what you know, 1 to 3 sentences", "confidence": "high, medium or low", "genre": "...", "era": "...", "style": "...", "style_keywords": "...", "palette": ["#rrggbb"], "motifs": ["..."], "why": "one sentence on why this art"}},
 "shader": {{"preset": "an id", "add": [], "knobs": {{}}, "drive": [], "mood": ["..."]}}}}"##,
        title = song.title,
        seconds = song.duration,
        scenes = scenes.said(),
        characters = CHARACTERS.said(),
        name = rules::NAME,
        prompt_words = PROMPT_WORDS.said(),
        keywords = KEYWORDS.said(),
        palette = PALETTE.said(),
        motifs = MOTIFS.said(),
        mood = MOOD.said(),
        about = about.join("\n"),
        styles = styles.join(", "),
        brief = catalogue::brief(),
    )
}

/// The prompt asked again after an answer was refused for `why`: said
/// before it and again after it, where the answer starts.
pub fn again(prompt: &str, why: &str) -> String {
    format!(
        "Your last answer to what follows was refused: {why}\nMend exactly that.\n\n{prompt}\n\nRemember, your last answer was refused: {why}\nWrite the whole JSON again, keeping to every rule."
    )
}
