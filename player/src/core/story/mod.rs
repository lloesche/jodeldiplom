//! The karaoke story's plan: what a language model (Gemma 4 E4B, greedy,
//! so the same song always gives the same plan) writes for a song from
//! its timed lyrics and its tags, in one JSON answer, read into a
//! [`Plan`]:
//!
//! - first the story, from the lyrics alone: what they tell, the
//!   characters (each with a short visual description, so every picture
//!   repeats it), and the scenes by the song's sections, each taking
//!   consecutive lines (its time span is theirs), with what it shows, a
//!   picture prompt (no text in the picture, ever) and who appears;
//! - then, so that the whole storyboard is already written when it is
//!   chosen, the song's art: what the model knows of the artist, the
//!   album and the song (and how sure it is), the genre, the era, an art
//!   style from a wide list ([`Style`]), a palette, motifs ([`Art`]);
//! - and the very last, the shaders over the pictures, picked from the
//!   looks' catalogue (a preset, passes added after it, knobs moved
//!   within their ranges, what in the music moves them, mood tags), kept
//!   as the [`jodel_looks::Look`] they make ([`shader`]).
//!
//! What the model is asked is [`prompt::prompt`], every count in it from
//! [`rules`]; its answer is read and held to every rule strictly
//! ([`check::read`]): a preset there is none of, a knob out of its range,
//! a line in no scene or in two, a character no one described, each is a
//! precise error, and the plan is not taken. While it writes, the prose of
//! it (the story, the characters, what each scene shows) is told as it
//! comes ([`prose::Prose`]). The job that asks it, keeps its plan and
//! tells the preparation view is `crate::work::story`.
//!
//! What the image model is asked for the plan's pictures (a portrait of
//! each character, then each scene with those portraits as its reference
//! pictures) is [`pictures::pictures`]; the job that draws them is
//! `crate::work::story::draw`.

pub mod check;
pub mod pictures;
pub mod prompt;
pub mod prose;
pub mod rules;
pub mod shader;

use std::fmt;

/// Art styles, each its name as the model writes it and the words that
/// carry it into the image model's prompts.
macro_rules! styles {
    ($($s:ident $name:literal $words:literal,)*) => {
        /// The art styles a song's pictures can be drawn in. None of them
        /// asks for lettering: no poster, no typography.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        pub enum Style {
            $(#[doc = $words] #[serde(rename = $name)] $s,)*
        }

        impl Style {
            /// Every style.
            pub const ALL: &[Style] = &[$(Style::$s,)*];

            /// Its name, as the model writes it.
            pub fn name(self) -> &'static str {
                match self { $(Style::$s => $name,)* }
            }

            /// The words that carry it into the image model's prompts.
            pub fn words(self) -> &'static str {
                match self { $(Style::$s => $words,)* }
            }
        }
    };
}

styles! {
    Watercolor "watercolor" "watercolor painting, soft washes, paper texture, loose brush strokes",
    Gouache "gouache painting" "gouache painting, matte opaque colours, visible brush strokes",
    Oil "oil painting" "oil painting, thick impasto brush strokes, canvas texture",
    Impressionist "impressionist painting" "impressionist painting, broken colour, dabs of light, plein air",
    Surrealist "surrealist painting" "surrealist painting, dreamlike, smooth glazes, uncanny light",
    InkWash "ink wash" "ink wash painting, sumi brush, grey washes, rice paper",
    Charcoal "charcoal drawing" "charcoal drawing, smudged tones, rough paper, strong contrast",
    ColouredPencil "coloured pencil" "coloured pencil drawing, hatched strokes, soft paper grain",
    ComicInk "comic ink" "comic book ink drawing, heavy black ink lines, cross-hatching, halftone shading",
    PopArt "pop-art halftone" "pop art, Ben-Day dots halftone, bold outlines, primary colours",
    Psychedelic "psychedelic 60s cartoon" "psychedelic 1960s art, cartoon illustration, bold black outlines, flat saturated colours, swirling shapes",
    ArtNouveau "art nouveau" "art nouveau illustration, flowing organic lines, ornamental curves, muted gold",
    UkiyoE "ukiyo-e woodblock" "ukiyo-e woodblock print, flat colours, bold outlines, washi paper texture",
    Linocut "linocut print" "linocut print, carved lines, two-colour ink",
    Risograph "risograph print" "risograph print, grainy spot colours, slight misregistration",
    StainedGlass "stained glass" "stained glass window, black lead lines, glowing coloured glass",
    PaperCutOut "paper cut-out" "layered paper cut-out collage, craft paper textures, soft cast shadows, handmade",
    Claymation "claymation" "claymation stop-motion scene, plasticine figures, clay texture, miniature set",
    LowPoly "low-poly 3D" "low-poly 3D render, faceted geometry, flat shaded polygons",
    AnimeCel "anime cel" "anime cel animation still, clean line art, flat cel shading, painted background",
    Storybook "storybook illustration" "children's storybook illustration, soft gouache and pencil, warm and gentle",
    Synthwave "neon synthwave" "neon synthwave illustration, glowing magenta and cyan neon, retro 80s, night",
    PixelArt "pixel art" "16-bit pixel art, limited palette, crisp pixels",
    FilmPhoto "35mm film photo" "35mm film photograph, film grain, natural light, candid",
    FilmNoir "black-and-white film noir" "black and white film noir still, hard shadows, venetian blind light, high contrast",
    Polaroid "polaroid snapshot" "instant film snapshot, faded colours, soft focus, flash light",
}

/// A colour of a palette, written `#rrggbb`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub [u8; 3]);

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [r, g, b] = self.0;
        write!(f, "#{r:02x}{g:02x}{b:02x}")
    }
}

impl std::str::FromStr for Rgb {
    type Err = String;

    /// `#rrggbb` (either case).
    fn from_str(s: &str) -> Result<Rgb, String> {
        let hex = s
            .strip_prefix('#')
            .filter(|h| h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| format!("the colour {s:?} is not \"#rrggbb\""))?;
        let byte = |k: usize| u8::from_str_radix(&hex[k..k + 2], 16).expect("hex, checked");
        Ok(Rgb([byte(0), byte(2), byte(4)]))
    }
}

impl serde::Serialize for Rgb {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for Rgb {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Rgb, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A song as the plan is made from it: its title, artist and album, the
/// tags of its file that say something of it (no technical ones), how
/// long it is, and its sung lines with their times. It reads and writes
/// as JSON only so that a song can be given to the end-to-end test.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Song {
    /// Its title.
    pub title: String,
    /// Its artist.
    pub artist: String,
    /// Its album.
    #[serde(default)]
    pub album: String,
    /// Its file's tags that say something of it, as (name, value).
    #[serde(default)]
    pub tags: Vec<(String, String)>,
    /// How long it is, seconds.
    pub duration: f64,
    /// Its sung lines, in order.
    pub lines: Vec<Line>,
}

/// A sung line: when it starts and ends (seconds into the song) and its
/// words.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Line {
    /// When it starts.
    pub start: f64,
    /// When it ends.
    pub end: f64,
    /// Its words.
    pub text: String,
}

/// The tags of a file a plan is told of (ASCII case ignored): what says
/// something of the song, its era or its mood; none of the technical ones
/// (encoders, rates, numbers on the disc).
const TELLING_TAGS: &[&str] = &[
    "albumartist",
    "album artist",
    "date",
    "year",
    "originaldate",
    "originalyear",
    "genre",
    "style",
    "mood",
    "composer",
    "lyricist",
    "label",
    "organization",
    "publisher",
    "copyright",
    "comment",
    "description",
];

/// The longest tag value told, in characters.
const TAG_CHARS: usize = 200;

impl Song {
    /// The song of `info` (its title, artist and album, else the file's
    /// name; its telling tags, each once; its length, else the end of its
    /// last line) with its `lyrics`' sung lines ([`lines_of`]).
    pub fn of(info: &jodel_playback::TrackInfo, lyrics: &jodel_lyrics::Lyrics) -> Song {
        let tags = &info.tags;
        let title = tags.title.clone().unwrap_or_else(|| {
            (info.path.file_stem())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        let mut told: Vec<(String, String)> = Vec::new();
        for (name, value) in &tags.all {
            let value = value.trim();
            let telling = TELLING_TAGS.iter().any(|t| t.eq_ignore_ascii_case(name));
            let new = !told.iter().any(|(n, _)| n.eq_ignore_ascii_case(name));
            if telling && new && !value.is_empty() {
                told.push((name.to_lowercase(), value.chars().take(TAG_CHARS).collect()));
            }
        }
        let lines = lines_of(lyrics);
        let last = lines.last().map_or(0.0, |l| l.end);
        Song {
            title,
            artist: tags.artist.clone().unwrap_or_default(),
            album: tags.album.clone().unwrap_or_default(),
            tags: told,
            duration: info.duration().map_or(last, |d| d.as_secs_f64().max(last)),
            lines,
        }
    }
}

/// The sung lines of `lyrics` with their times: neither breaks nor lines
/// wholly in parentheses (backing vocals echo or answer the lead's line,
/// and the story is the lead's: numbered for the model they would take
/// scenes of their own), nor lines with no time at all. Each starts when
/// it starts ([`jodel_lyrics::Line::start`], else when its source synced
/// it) and ends with its last word, else when its source has it over
/// ([`jodel_lyrics::scaffold::line_end`]), else at the next line's start,
/// else [`rules::UNTIMED_LINE_SECONDS`] after it starts.
pub fn lines_of(lyrics: &jodel_lyrics::Lyrics) -> Vec<Line> {
    let all = &lyrics.lines;
    let start = |l: &jodel_lyrics::Line| l.start().or(l.synced.map(|s| s.start));
    let sung = |l: &jodel_lyrics::Line| !l.is_break && !l.backing && !l.words.is_empty();
    let mut lines = Vec::new();
    for (k, line) in all.iter().enumerate() {
        let Some(at) = start(line).filter(|_| sung(line)) else {
            continue;
        };
        let next = all[k + 1..].iter().filter(|l| sung(l)).find_map(start);
        let end = (line.words.last().and_then(|w| w.end))
            .or_else(|| jodel_lyrics::scaffold::line_end(all, k))
            .or(next)
            .unwrap_or(at + rules::UNTIMED_LINE_SECONDS)
            .max(at);
        lines.push(Line {
            start: at,
            end,
            text: line.text(),
        });
    }
    lines
}

/// A song's plan, read from the model's answer and checked: the story in
/// two sentences, its characters, its scenes in order (together taking
/// every line once, each with its time span), the song's art, and the
/// look the shaders make (the preset, the passes added, the knobs moved,
/// what the music moves, the mood among its tags). It is kept as it
/// reads and writes ([`Plan::to_json`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Plan {
    /// What the lyrics tell.
    pub story: String,
    /// Who is in it.
    pub characters: Vec<Character>,
    /// Its scenes.
    pub scenes: Vec<Scene>,
    /// The song's art.
    pub art: Art,
    /// The look over its pictures.
    pub look: jodel_looks::Look,
}

/// A character: its name (how the scenes and the prompts name it) and how
/// it looks, the same in every picture.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Character {
    /// Its name.
    pub name: String,
    /// How it looks.
    pub looks: String,
}

/// A scene: the lines it takes (first and last, from 0), its time span
/// (from its first line's start, the first scene's from the song's start,
/// to the next scene's start, the last one's to the song's end), the
/// song's section it is in, its shot, the characters seen, what it shows
/// and its picture's prompt.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scene {
    /// Its first line.
    pub first_line: usize,
    /// Its last line.
    pub last_line: usize,
    /// When it starts, seconds into the song.
    pub from: f64,
    /// When it ends.
    pub to: f64,
    /// The song's section (verse 1, chorus, ...).
    pub section: String,
    /// Its shot.
    pub shot: Shot,
    /// The characters seen, by name.
    pub characters: Vec<String>,
    /// What it shows, a sentence.
    pub shows: String,
    /// Its picture's prompt for the image model.
    pub prompt: String,
}

/// How near a scene's camera is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Shot {
    /// The whole place.
    Wide,
    /// People from the waist up.
    Medium,
    /// A face, a hand, a thing.
    CloseUp,
}

impl Shot {
    /// Its name, as the model writes it and the image model is told it.
    pub fn name(self) -> &'static str {
        match self {
            Shot::Wide => "wide",
            Shot::Medium => "medium",
            Shot::CloseUp => "close-up",
        }
    }
}

/// How sure the model is of what it knows of the song.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Sure.
    High,
    /// Fairly.
    Medium,
    /// Hardly.
    Low,
}

/// The song's art, as the model writes it: what it knows of the artist,
/// the album and the song, and how sure it is; its genre and era; the
/// art style with its keywords, a palette, the motifs, and why.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Art {
    /// What it knows.
    pub knowledge: String,
    /// How sure it is.
    pub confidence: Confidence,
    /// The genre.
    pub genre: String,
    /// The era.
    pub era: String,
    /// The art style.
    pub style: Style,
    /// Keywords for the image model, comma-separated.
    pub style_keywords: String,
    /// The palette.
    pub palette: Vec<Rgb>,
    /// Recurring objects or symbols from the lyrics.
    pub motifs: Vec<String>,
    /// Why.
    pub why: String,
}

impl Plan {
    /// The character named `name`.
    pub fn character(&self, name: &str) -> Option<&Character> {
        self.characters.iter().find(|c| c.name == name)
    }

    /// The plan as the cache keeps it: JSON, laid out for people.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a plan writes as JSON")
    }

    /// A plan kept as `json` ([`Plan::to_json`]).
    pub fn from_json(json: &str) -> Result<Plan, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
pub(crate) mod tests;
