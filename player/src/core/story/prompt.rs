//! What the model is asked for a song's plan ([`prompt`]): the story from
//! the lyrics alone (the numbered lines with their times, the rules of a
//! scene: what the song is about, not its lines word for word; a new scene
//! as the lyrics move, as many as the song needs; a section that comes
//! back showing its earlier scene again; a plot of events, a beginning, a
//! turn and an end; a storyboard's shots, not a portrait session: things
//! happening in places, characters doing something and seldom near, many
//! scenes with none, the camera's distance and angle varied; characters
//! only where beings are to be seen; an example of such scenes), then the
//! art from the file's tags, what the model knows of
//! the artist and the song, and the storyboard it has just written, then
//! the shaders from the looks' catalogue in brief
//! ([`jodel_looks::catalogue::brief`]), and the answer's JSON laid out
//! field by field. Every count it asks for is [`super::rules`]'s; the
//! most scenes a plan has is never said (a number would anchor the
//! model).
//!
//! The prompt is made the same way every time for the same song, so the
//! same song always gives the same plan.

use std::fmt::Write as _;

use jodel_looks::catalogue;

use super::rules::{self, CHARACTERS, KEYWORDS, MOOD, MOTIFS, PALETTE, PROMPT_WORDS};
use super::{Angle, Song, Style};

/// What the model is asked for `song`'s plan.
pub fn prompt(song: &Song) -> String {
    let mut numbered = String::new();
    for (k, line) in song.lines.iter().enumerate() {
        let _ = writeln!(numbered, "{k}. [{}] {}", clock(line.start), line.text);
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
    let angles: Vec<String> = (Angle::ALL.iter())
        .map(|a| format!("\"{}\"", a.name()))
        .collect();
    let by = if song.artist.is_empty() {
        String::new()
    } else {
        format!(" by {}", song.artist)
    };
    format!(
        r##"You direct an illustrated karaoke video for the song "{title}"{by}: still pictures, one after another, that show what the song is about while it is sung, with a look of their own over them. Write one JSON answer: first the story, then the art, then its shaders.

THE STORY comes from the lyrics alone. The song's tags and what you know of the artist play no part in it.

The lyrics, numbered lines with the time each starts at, minutes:seconds (the song lasts {length}):
{numbered}
Rules for the story:
- Show what the song tells, not its lines word for word: what happens to its people, in its world, with its images and metaphors as places and things the story passes through. A line such as "my heart is a frozen lake" is a frozen lake at dawn that someone crosses alone, not someone singing about a heart.
- What happens comes from the lyrics: their people, their events, their images. Give them a place and a time to happen in, and people around (passers-by, a crowd, a driver), but no other story.
- A new scene starts where the lyrics move: a new place, time, event or feeling. Plan as many scenes as the song needs, no more: a song that stays in one place may need only one or two, a song that travels many. A scene shows from its first line until the next scene starts; the first starts at line 0.
- A section that comes back (a chorus sung again, a refrain) shows its earlier scene again: give it "same_as", the first line of that earlier scene, and nothing new is drawn for it.
- A plot across the scenes, so a viewer follows what happens from picture to picture: a beginning (where we are, who is there), a turn (something happens, changes or goes wrong) and an end (what is left after it). The "story" says it in three sentences, and the scenes follow it in order. Each scene's "shows" names an event, something that happens or changes (someone leaves, a door slams, a glass falls, the train pulls out, the rain starts, the room is empty now), never only a mood or a symbol.
- A storyboard, not a portrait session. Each scene is a still from a film of something happening in a place, often with other people around (passers-by, a crowd, a waiter, a driver: describe them in the prompt):
  - characters do something with their hands or their feet (walk, carry, open, pack, throw, dance, run, drive, turn away, fall asleep), often small in the frame or seen from behind; never just standing or looking, never posed. A feeling is shown by what they do and where they are, not by a face;
  - mix wide shots of the place, action at medium distance, and a detail or an object in close-up (hands, a cup, keys, a window, a phone, a shoe in a puddle), the object being used, found, left or broken; a face only now and then, at most one scene in five;
  - many scenes have no character at all: the place, an object, the weather, something left behind;
  - no subject (an object, a place) in more than two scenes drawn;
  - vary the camera: its angle, and the subject's size and place in the frame (tiny in a wide landscape, off-centre, partly in frame, a detail filling it). Two scenes in a row never share both their shot and their angle.
- A protagonist to follow: where the song tells of people (most songs do), its story has a lead, the "I" or the "you" of the song or someone standing in for them, seen in several scenes (not all), so a viewer follows them from picture to picture; others it tells of may be characters too. No characters only where the song is truly about places, objects or abstractions. Characters are beings to be seen (a person, an animal, a creature, a machine with a face), never the singer as a singer: {characters}, each one single being. A crowd or a group is never a character: describe it in the scenes' prompts. Describe each character once with fixed looks, appearance only (for a person: age, gender, build, face, hair, clothes), so every picture can repeat them: never an action, a habit, a prop, an expression or a pose in the looks (not "smoking", "holding a guitar", "smiling"); those belong in a scene's prompt. A name is {name}.
- A scene's fields, each written once, in this order:
  - "first_line": the number of its first line. "section": the song's section (verse 1, chorus, ...).
  - "same_as": only in a section that comes back: the first_line of the earlier scene it shows again (it needs no prompt then).
  - "shot": "wide", "medium" or "close-up".
  - "angle": where the camera looks from: {angles}.
  - "characters": the names of the characters its prompt shows (often empty).
  - "shows": the event the scene shows, one sentence.
  - "prompt": a concrete visual description for an image generator, {prompt_words}: what happens, the place, the time of day, the people around, where the subject is in the frame, foreground and background. For each character in it, say what they do, their pose and how big they are in the frame (small in the distance, from behind, at the edge, only their hands). Name characters by their name only. No style words. Nothing written in the picture: never the words text, letters, sign, poster, logo, label, banner, billboard or caption (say "the border at night", not "a border sign").

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
Answer only JSON, in this order (the scenes below show the form only, what each kind of scene holds; yours come from this song):
{{"story": "...",
 "characters": [{{"name": "mara", "looks": "..."}}],
 "scenes": [{{"first_line": 0, "section": "verse 1", "shot": "wide", "angle": "high", "characters": [], "shows": "the day starts in the place", "prompt": "the whole place seen from above at a time of day, its people busy and small in the frame, what is in front and what is behind"}},
            {{"first_line": 4, "section": "verse 1", "shot": "medium", "angle": "from behind", "characters": ["mara"], "shows": "mara arrives and does something", "prompt": "mara seen from behind at the left of the frame, carrying or opening or pushing something, people passing her on both sides, the place ahead of her, the light"}},
            {{"first_line": 6, "section": "chorus", "shot": "close-up", "angle": "eye level", "characters": [], "shows": "a thing of the story is found, used, left or broken", "prompt": "hands doing something with that thing, the surface it lies on, the light falling on it"}},
            {{"first_line": 11, "section": "verse 2", "shot": "wide", "angle": "low", "characters": [], "shows": "something happens to the place or its people", "prompt": "the place from low down as it changes, people reacting small in the frame, the weather, something left behind in front"}},
            {{"first_line": 17, "section": "chorus", "same_as": 6}}],
 "art": {{"knowledge": "what you know, 1 to 3 sentences", "confidence": "high, medium or low", "genre": "...", "era": "...", "style": "...", "style_keywords": "...", "palette": ["#rrggbb"], "motifs": ["..."], "why": "one sentence on why this art"}},
 "shader": {{"preset": "an id", "add": [], "knobs": {{}}, "drive": [], "mood": ["..."]}}}}"##,
        title = song.title,
        length = clock(song.duration),
        characters = CHARACTERS.said(),
        name = rules::NAME,
        prompt_words = PROMPT_WORDS.said(),
        angles = angles.join(", "),
        keywords = KEYWORDS.said(),
        palette = PALETTE.said(),
        motifs = MOTIFS.said(),
        mood = MOOD.said(),
        about = about.join("\n"),
        styles = styles.join(", "),
        brief = catalogue::brief(),
    )
}

/// `seconds` as the prompt tells a time, minutes:seconds (`1:05`): a line's
/// number and its time never look alike, so a scene's first line is not
/// written as a time.
fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}
