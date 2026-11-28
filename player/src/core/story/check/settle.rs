//! What of a read answer needs the model's judgement ([`Read::questions`],
//! asked in the words of [`follow`]), and the follow-up's answers taken
//! ([`Read::settle`]): each valid choice as answered, any other as the
//! nearest valid one.

use serde_json::Value;

use super::super::follow::{self, Question, What};
use super::super::repair::{number, text};
use super::super::rules::nearest_by;
use super::super::Style;
use super::fields::{plain, text_word, without_text, words_of};
use super::{Again, Read};

impl Read {
    /// What needs the model's judgement, as the follow-up asks it
    /// ([`follow`]): a scene's first line that fits nowhere, a scene shown
    /// again that names no scene before it, a picture's prompt missing or
    /// asking for writing, the style, the preset or a pass no one knows,
    /// each scene of the same words as an earlier one drawn anew, and a
    /// storyboard that reads as a portrait session ([`Read::board`]).
    pub fn questions(&self) -> Vec<Question> {
        let mut asked = Vec::new();
        let spans = self.spans();
        for (k, s) in self.scenes.iter().enumerate() {
            if let Some(bounds) = s.bounds {
                asked.push(follow::start(k, &s.shows, s.written, bounds));
            }
            // A scene shown again needs no prompt (one naming no scene
            // before it shows the nearest, where it has none).
            if let Some(again) = s.same_as {
                if self.shown_again(k).is_none() {
                    let earlier: Vec<usize> = (0..k)
                        .filter(|&j| self.scenes[j].same_as.is_none())
                        .map(|j| self.scenes[j].first)
                        .collect();
                    asked.push(follow::again(k, &s.shows, again.written(), &earlier));
                }
                continue;
            }
            let (a, b) = spans[k];
            let word = text_word(&s.prompt);
            if word.is_some() || (s.prompt.is_empty() && s.shows.is_empty()) {
                let lines = words_of(&self.lines[a..=b]);
                asked.push(follow::prompt(k, &s.shows, &lines, word));
            }
        }
        if let Err(written) = &self.art.style {
            asked.push(follow::style(written));
        }
        let ids: Vec<&str> = (jodel_looks::presets::all().iter())
            .map(|p| p.id.as_str())
            .collect();
        if jodel_looks::presets::named(&self.shader.preset).is_none() {
            asked.push(follow::preset(&self.shader.preset, &self.art.genre, &ids));
        }
        let names: Vec<&str> = jodel_looks::passes::PASSES.iter().map(|p| p.name).collect();
        for (index, pass) in self.shader.add.iter().enumerate() {
            if jodel_looks::passes::named(pass).is_none() {
                asked.push(follow::pass(index, pass, &names));
            }
        }
        asked.extend(self.repeats(&spans));
        asked.extend(self.board());
        asked
    }

    /// The question of each scene drawn anew whose lines have the very
    /// words of an earlier scene drawn (`spans`, each scene's lines).
    fn repeats(&self, spans: &[(usize, usize)]) -> Vec<Question> {
        let said = |(a, b): (usize, usize)| -> String {
            let words: Vec<String> = self.lines[a..=b].iter().map(|l| plain(l)).collect();
            words.join(" / ")
        };
        let drawn = self.drawn();
        let mut asked = Vec::new();
        for (i, &k) in drawn.iter().enumerate() {
            let words = said(spans[k]);
            let earlier =
                (drawn[..i].iter()).find(|&&j| !words.is_empty() && said(spans[j]) == words);
            if let Some(&j) = earlier {
                let s = &self.scenes[k];
                asked.push(follow::reuse(
                    k,
                    &s.shows,
                    spans[k],
                    (j, self.scenes[j].first),
                ));
            }
        }
        asked
    }

    /// `questions` settled with the follow-up's `answers` (in their order;
    /// none: not answered, or not asked): each answer that is a valid
    /// choice taken, any other taken as the nearest valid one, each said in
    /// the notes.
    pub fn settle(&mut self, questions: &[Question], answers: &[Option<Value>]) {
        for (k, q) in questions.iter().enumerate() {
            let answer = answers.get(k).cloned().flatten();
            let took = self.answer(&q.what, answer.as_ref());
            self.notes.push(format!("question {}: {took}", k + 1));
        }
    }

    /// `what` settled with `answer` (none: not answered): what was taken,
    /// in words.
    pub(super) fn answer(&mut self, what: &What, answer: Option<&Value>) -> String {
        let answered = answer.filter(|a| !a.is_null());
        let said = answer.map_or("nothing".to_owned(), Value::to_string);
        match *what {
            What::Start { scene, lo, hi } => {
                let s = &mut self.scenes[scene];
                let within = |l: &f64| l.fract() == 0.0 && *l >= lo as f64 && *l <= hi as f64;
                s.bounds = None;
                match number(answered).filter(within) {
                    Some(line) => {
                        s.first = line as usize;
                        format!("scene {scene} starts at line {}, as answered", s.first)
                    }
                    None => format!(
                        "scene {scene} starts at line {} (the nearest; answered {said})",
                        s.first
                    ),
                }
            }
            // Not answered, it stays for the plan to show the nearest.
            What::Again { scene } if answer.is_none() => {
                format!("scene {scene}: not answered (the nearest taken)")
            }
            What::Again { scene } => {
                self.scenes[scene].same_as =
                    number(answered).map(|l| Again::Written(l.round() as i64));
                match self.shown_again(scene) {
                    Some(j) => format!("scene {scene} shows scene {j} again, as answered"),
                    None => {
                        self.scenes[scene].same_as = None;
                        format!("scene {scene} drawn anew (answered {said})")
                    }
                }
            }
            What::Prompt { scene } => {
                let written = text(answered).filter(|p| !p.is_empty() && text_word(p).is_none());
                let s = &mut self.scenes[scene];
                match written {
                    Some(prompt) => {
                        s.prompt = prompt;
                        format!("scene {scene}'s prompt written again: {:?}", s.prompt)
                    }
                    None => {
                        s.prompt = without_text(&s.prompt);
                        format!("scene {scene}'s prompt kept, without writing (answered {said})")
                    }
                }
            }
            What::Style => {
                let written = text(answered).unwrap_or_default();
                let style = Style::named(&written).unwrap_or_else(|| self.nearest_style(&written));
                self.art.style = Ok(style);
                let how = match Style::named(&written) {
                    Some(_) => "as answered",
                    None => "the nearest",
                };
                format!("the style {:?}, {how} (answered {said})", style.name())
            }
            What::Preset => {
                if let Some(id) = text(answered).filter(|id| !id.is_empty()) {
                    self.shader.preset = id;
                }
                format!("the preset {:?} (answered {said})", self.shader.preset)
            }
            What::Pass { index } => {
                if let Some(name) = text(answered).filter(|n| !n.is_empty()) {
                    self.shader.add[index] = name;
                }
                format!(
                    "the pass to add {:?} (answered {said})",
                    self.shader.add[index]
                )
            }
            What::Reuse { scene, earlier } => {
                let line = self.scenes[earlier].first as f64;
                let yes = match answered {
                    Some(Value::Bool(b)) => *b,
                    Some(Value::String(s)) => {
                        ["true", "yes", "y", "1"].contains(&s.trim().to_lowercase().as_str())
                    }
                    other => number(other).is_some_and(|n| n == 1.0 || n == line),
                };
                if yes {
                    self.scenes[scene].same_as = Some(Again::Scene(earlier));
                    format!("scene {scene} shows scene {earlier} again, as answered")
                } else {
                    format!("scene {scene} drawn anew (answered {said})")
                }
            }
            What::Redraw { ref scenes } => self.redraw(scenes, answered),
        }
    }

    /// The style nearest to `written` (where nothing was, to what the
    /// answer wrote for it, else its genre, else its keywords).
    pub(super) fn nearest_style(&self, written: &str) -> Style {
        let first = self.art.style.as_ref().err().map(String::as_str);
        let to = [
            Some(written),
            first,
            Some(&self.art.genre),
            Some(&self.art.style_keywords),
        ]
        .into_iter()
        .flatten()
        .find(|w| !w.is_empty())
        .unwrap_or_default();
        *nearest_by(to, Style::ALL, |s| s.name()).expect("there are styles")
    }
}
