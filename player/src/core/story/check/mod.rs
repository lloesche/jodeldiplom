//! A model's answer read as a plan, forgivingly ([`read`]): a small model
//! slips, and its work is never thrown away for a slip. Everything usable
//! is kept, every mending is said ([`Read::notes`], for the trace, to see
//! what to fix in the prompt), and the answer is refused only when nothing
//! is left to draw: no JSON object at all, or not one scene ([`Refused`],
//! with what was mended before).
//!
//! - The JSON is mended first ([`super::repair`]: a trailing comma, a
//!   bracket missing, a quote unescaped, an answer cut off with every
//!   whole scene kept).
//! - A field missing takes its default, one unknown is left out; a count
//!   past its most keeps its first ([`super::rules`]); a number out of its
//!   range is held to it ([`fields`]).
//! - The scenes' lines come from their first lines alone (each runs to the
//!   next one's start), so no line is in no scene or in two; first lines
//!   written by the clock are read as the lines sung then
//!   ([`place::by_the_clock`]): those written as minutes:seconds always,
//!   numbers where two or more are past the last line (within the song's
//!   length) and more scenes go up in order read so. The first
//!   scene starts at line 0; the scenes whose first lines go up in order
//!   (the most of them) stand, and each other one is placed between those
//!   around it, asked of the model where it fits (else left out: its lines
//!   shown by the scene before). The scenes past [`MOST_SCENES`] are
//!   merged into the last kept ([`place`]).
//! - A scene that shows an earlier one again (`same_as`: a line of that
//!   scene, or its place among the scenes where only that is of its
//!   section) draws nothing new; one that names no scene before it is
//!   asked; each one's reading is said.
//! - A character's looks are its appearance alone (an action, a habit, a
//!   prop or an expression in them left out: [`cast::appearance`]), and a
//!   scene's cast is who its prompt names ([`cast::cast`]).
//! - A character is named as [`super::rules::name`] has it; one without
//!   looks, one named twice, a group (a crowd is no portrait: it stays in
//!   the prompts), and one seen in no scene drawn are left out, and a
//!   scene naming a character no one described keeps its picture without
//!   the name.
//! - A picture's prompt asking for writing is asked again (else its
//!   writing left out); one missing is its scene's description, or where
//!   that is missing too asked for (else its lines' words); a style, a
//!   preset or a pass no one knows is asked again, then taken as the
//!   nearest there is ([`super::follow`]); a scene of the very words of an
//!   earlier one drawn anew is asked whether it shows that one again.
//! - A scene's angle is read by its name or what it says, else eye level.
//!   Scenes that read as a portrait session (more than half of those
//!   drawn showing one character near, or close-ups, or of one shot and
//!   angle: [`super::rules::BOARD`]) are asked to be rewritten, some of
//!   them, as a storyboard's shots ([`board`]); never refused for it.
//!
//! What needs the model's judgement is [`Read::questions`]; their answers
//! (or none: the nearest valid choice) are taken with [`Read::settle`]
//! ([`settle`]), and the plan made with [`Read::plan`].

mod board;
mod fields;
mod place;
mod settle;

use serde_json::Value;

pub use self::fields::text_word;
use self::fields::{without_text, words_of};
use super::cast;
use super::repair::{self, list, text};
use super::rules::{MOST_SCENES, PROMPT_WORDS};
use super::shader::Shader;
use super::{Angle, Art, Character, Confidence, Plan, Rgb, Scene, Shot, Song, Style};

/// Why an answer with nothing to draw is refused.
pub const NOTHING: &str = "nothing to draw";

/// What starts a note that says how something was read (not mended).
pub const READ: &str = "read:";

/// An answer refused (it has nothing to draw): why, and what was mended
/// of it before.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    /// Why.
    pub why: String,
    /// What was mended.
    pub notes: Vec<String>,
}

/// The earlier scene a scene shows again: as the model wrote it (a
/// number: the first line of a scene, a line of it, or its place among
/// the scenes), or settled (its place).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Again {
    /// As written.
    Written(i64),
    /// Settled.
    Scene(usize),
}

impl Again {
    /// The number written (a scene settled: its place).
    fn written(self) -> i64 {
        match self {
            Again::Written(n) => n,
            Again::Scene(k) => k as i64,
        }
    }
}

/// A scene as read, its first line placed.
#[derive(Clone, Debug)]
struct Drafted {
    /// Its place among the scenes as the model wrote them (what a
    /// `same_as` read as a scene's place names).
    at: usize,
    /// Its first line as written (none: none, or no number).
    written: Option<f64>,
    /// Its first line and the scene it shows again where either was
    /// written as a time (minutes:seconds), that time in seconds: for the
    /// lines sung then ([`place::by_the_clock`]).
    times: (Option<f64>, Option<f64>),
    /// Its first line: as written where it stands among the others, else
    /// where it is placed for now.
    first: usize,
    /// Whether its first line stands as written (among the others).
    standing: bool,
    /// Where its first line may go, where it was not the model's (it is
    /// asked).
    bounds: Option<(usize, usize)>,
    /// The earlier scene it shows again.
    same_as: Option<Again>,
    section: String,
    shot: Shot,
    angle: Angle,
    characters: Vec<String>,
    shows: String,
    prompt: String,
}

/// The art as read: the style as written where it is none of them.
#[derive(Clone, Debug)]
struct ArtRead {
    knowledge: String,
    confidence: Confidence,
    genre: String,
    era: String,
    style: Result<Style, String>,
    style_keywords: String,
    palette: Vec<Rgb>,
    motifs: Vec<String>,
    why: String,
}

/// An answer read: everything usable of it, what was mended, and the
/// song's lines it is read for.
#[derive(Clone, Debug)]
pub struct Read {
    story: String,
    characters: Vec<Character>,
    scenes: Vec<Drafted>,
    art: ArtRead,
    shader: Shader,
    /// The song's lines' words.
    lines: Vec<String>,
    /// The song's length, seconds, and its lines' starts.
    end: f64,
    starts: Vec<f64>,
    /// What was mended, each said.
    pub notes: Vec<String>,
}

/// The plan in a model's `answer` for `song`, read forgivingly (see the
/// module's doc): what it holds, its questions still to settle; refused
/// ([`NOTHING`], why, and what was mended) only where it holds no JSON
/// object or no scene.
pub fn read(answer: &str, song: &Song) -> Result<Read, Refused> {
    let mut notes = Vec::new();
    let Some(value) = repair::json(answer, &mut notes) else {
        let why = format!("{NOTHING}: the answer holds no JSON object");
        return Err(Refused { why, notes });
    };
    let field = |key: &str| value.get(key);
    if let Value::Object(top) = &value {
        fields::unknown(top, &fields::FIELDS, "the answer", &mut notes);
    }
    let characters = fields::characters(field("characters"), &mut notes);
    let n = song.lines.len();
    let mut scenes = Vec::new();
    for (k, v) in list(field("scenes")).iter().enumerate() {
        match v {
            Value::Object(f) => scenes.push(fields::scene(f, k, &characters, &mut notes)),
            other => notes.push(format!("scene {k} is no object ({other}): left out")),
        }
    }
    if scenes.is_empty() || n == 0 {
        let why = format!("{NOTHING}: the answer has no scene");
        return Err(Refused { why, notes });
    }
    let starts: Vec<f64> = song.lines.iter().map(|l| l.start).collect();
    let end = (song.lines.iter().map(|l| l.end)).fold(song.duration, f64::max);
    place::by_the_clock(&mut scenes, &starts, end, &mut notes);
    place::place(&mut scenes, n, &mut notes);
    if scenes.len() > MOST_SCENES {
        notes.push(format!(
            "{} scenes: those past the {MOST_SCENES}th merged into it",
            scenes.len()
        ));
        scenes.truncate(MOST_SCENES);
    }
    let art = fields::art(field("art"), &mut notes);
    let shader = Shader::of(field("shader"), &mut notes);
    Ok(Read {
        story: text(field("story")).unwrap_or_default(),
        characters,
        scenes,
        art,
        shader,
        lines: song.lines.iter().map(|l| l.text.clone()).collect(),
        end,
        starts,
        notes,
    })
}

impl Read {
    /// The first and last line of each scene, as placed now.
    fn spans(&self) -> Vec<(usize, usize)> {
        let n = self.lines.len();
        (self.scenes.iter().enumerate())
            .map(|(k, s)| {
                let next = (self.scenes.get(k + 1)).map_or(n, |o| o.first.max(s.first + 1));
                (s.first, next - 1)
            })
            .collect()
    }

    /// The scene a line `line` is in, among the first `before` scenes; none
    /// where it is in none of them (or no line).
    fn scene_of(&self, line: i64, before: usize) -> Option<usize> {
        let line = usize::try_from(line)
            .ok()
            .filter(|&l| l < self.lines.len())?;
        let j = (0..self.scenes.len())
            .rev()
            .find(|&j| self.scenes[j].first <= line)?;
        (j < before).then_some(j)
    }

    /// The earlier scene the scene `k` names to show again, and how it was
    /// read: the scene settled; else the number written read as a line,
    /// where that scene is of the same section; else as a scene's place,
    /// where that one is; else as a line, else as a place, where either is
    /// an earlier scene.
    fn named_again(&self, k: usize) -> Option<(usize, &'static str)> {
        let n = match self.scenes[k].same_as? {
            Again::Scene(j) => return (j < k).then_some((j, "as settled")),
            Again::Written(n) => n,
        };
        let by_line = self.scene_of(n, k);
        let by_place = (0..k).find(|&j| self.scenes[j].at as i64 == n);
        let section = &self.scenes[k].section;
        let same = |j: usize| !section.is_empty() && self.scenes[j].section == *section;
        match (by_line, by_place) {
            (Some(l), _) if same(l) => Some((l, "a line of it")),
            (_, Some(p)) if same(p) => Some((p, "its place among the scenes, of the same section")),
            (Some(l), _) => Some((l, "a line of it")),
            (None, Some(p)) => Some((p, "its place among the scenes")),
            (None, None) => None,
        }
    }

    /// The scene `k` shows again, as it is now: the earlier scene it names
    /// ([`Read::named_again`]; that one's own, where it shows another
    /// again).
    fn shown_again(&self, k: usize) -> Option<usize> {
        let (mut j, _) = self.named_again(k)?;
        while let Some((i, _)) = self.named_again(j) {
            j = i;
        }
        Some(j)
    }

    /// The plan: the scenes placed for good ([`place::for_good`]) and timed
    /// by their first lines; each shown again resolved (said; one that
    /// names no scene before it drawn anew where it has its own prompt,
    /// else showing the nearest earlier scene of its section, else the one
    /// before); each drawn with a prompt (its description where it has
    /// none, else its lines' words) asking for no writing, of
    /// [`PROMPT_WORDS`] at most, its cast those its prompt names; the
    /// characters seen in no scene drawn left out; the style, the preset
    /// and the passes still unknown taken as the nearest; each mending said
    /// in the notes.
    pub fn plan(mut self) -> (Plan, Vec<String>) {
        let scenes = std::mem::take(&mut self.scenes);
        self.scenes = place::for_good(scenes, self.lines.len(), &mut self.notes);
        let again: Vec<Option<usize>> = (0..self.scenes.len()).map(|k| self.resolve(k)).collect();
        let spans = self.spans();
        for (k, s) in self.scenes.iter_mut().enumerate() {
            if again[k].is_some() {
                continue;
            }
            if s.prompt.is_empty() {
                let (a, b) = spans[k];
                s.prompt = match s.shows.is_empty() {
                    true => words_of(&self.lines[a..=b]),
                    false => s.shows.clone(),
                };
                self.notes.push(format!(
                    "scene {k} has no prompt: drawn from {:?}",
                    s.prompt
                ));
            }
            if text_word(&s.prompt).is_some() {
                s.prompt = without_text(&s.prompt);
                self.notes.push(format!(
                    "scene {k}'s prompt without its writing: {:?}",
                    s.prompt
                ));
            }
            let words: Vec<String> = s.prompt.split_whitespace().map(str::to_owned).collect();
            if words.len() > PROMPT_WORDS.most {
                s.prompt = words[..PROMPT_WORDS.most].join(" ");
                self.notes.push(format!(
                    "scene {k}'s prompt of {} words: its first {} kept",
                    words.len(),
                    PROMPT_WORDS.most
                ));
            }
            let (seen, said) = cast::cast(&s.prompt, &s.characters, &self.characters);
            for said in said {
                self.notes.push(format!("scene {k}: {said}"));
            }
            s.characters = seen;
        }
        let seen: Vec<&String> = (self.scenes.iter().zip(&again))
            .filter(|(_, a)| a.is_none())
            .flat_map(|(s, _)| &s.characters)
            .collect();
        let (seen, unseen): (Vec<Character>, Vec<Character>) =
            (std::mem::take(&mut self.characters))
                .into_iter()
                .partition(|c| seen.contains(&&c.name));
        for c in unseen {
            self.notes.push(format!(
                "the character {:?} is seen in no scene drawn: left out",
                c.name
            ));
        }
        self.characters = seen;
        let style = match self.art.style.clone() {
            Ok(style) => style,
            Err(written) => {
                let style = self.nearest_style("");
                self.notes
                    .push(format!("the style {written:?} taken as {:?}", style.name()));
                style
            }
        };
        let look = (self.shader).to_look(&self.art.why, &self.art.genre, &mut self.notes);
        let end = self.end;
        let starts: Vec<f64> = (self.scenes.iter().enumerate())
            .map(|(k, s)| if k == 0 { 0.0 } else { self.starts[s.first] })
            .collect();
        let scenes = (self.scenes.into_iter().enumerate())
            .map(|(k, s)| Scene {
                first_line: spans[k].0,
                last_line: spans[k].1,
                from: starts[k],
                to: starts.get(k + 1).copied().unwrap_or(end),
                same_as: again[k],
                section: s.section,
                shot: s.shot,
                angle: s.angle,
                characters: s.characters,
                shows: s.shows,
                prompt: s.prompt,
            })
            .collect();
        let a = self.art;
        let plan = Plan {
            story: self.story,
            characters: self.characters,
            scenes,
            art: Art {
                knowledge: a.knowledge,
                confidence: a.confidence,
                genre: a.genre,
                era: a.era,
                style,
                style_keywords: a.style_keywords,
                palette: a.palette,
                motifs: a.motifs,
                why: a.why,
            },
            look,
        };
        (plan, self.notes)
    }

    /// The scene `k` shows again, for good (see [`Read::plan`]), its
    /// reading said in the notes: the earlier scene it names; one naming
    /// none before it with a prompt of its own drawn anew, else the nearest
    /// earlier scene of its section, else the one before.
    fn resolve(&mut self, k: usize) -> Option<usize> {
        let again = self.scenes[k].same_as?;
        if let (Some(j), Some((_, how))) = (self.shown_again(k), self.named_again(k)) {
            self.notes.push(format!(
                "{READ} scene {k} shows scene {j} again (same_as {} read as {how})",
                again.written()
            ));
            return Some(j);
        }
        let s = &self.scenes[k];
        if k == 0 || !s.prompt.is_empty() || !s.shows.is_empty() {
            self.scenes[k].same_as = None;
            self.notes.push(format!(
                "scene {k} shows no scene before it again: drawn anew"
            ));
            return None;
        }
        let drawn = |j: &usize| self.scenes[*j].same_as.is_none();
        let section = (0..k)
            .rev()
            .filter(drawn)
            .find(|&j| self.scenes[j].section == s.section);
        let j = section.or_else(|| (0..k).rev().find(drawn)).unwrap_or(0);
        self.scenes[k].same_as = Some(Again::Scene(j));
        self.notes.push(format!(
            "scene {k} shows no scene before it again, and has no prompt: it shows scene {j} again"
        ));
        Some(j)
    }
}
