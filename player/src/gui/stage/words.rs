//! The words on the stage's lane: each a tube through its notes
//! ([`super::super::snake::Snake`], its bubble with the game on:
//! [`super::super::pop`]), laid out in time one after another (no two
//! meet: each starts when it is sung, or just after the one before ends
//! where that is later, and lasts as long as it is sung, or as its text,
//! made smaller to fit before the next word down to [`SMALLEST_TEXT`] of
//! its size, or a circle, whichever is longest), its text inside it on
//! its flattest stretch, letter by letter at the tube's height there.

use std::collections::HashMap;

use super::super::big_text::Face;
use super::super::pop::{self, Laid};
use super::super::snake::Snake;
use super::super::stage_gpu::{Instance, Skin, Tint};
use super::{edge_fade, rgba, Lane, Song, Stage, ACTIVE, GOLD, GOLD_DEEP, GOLD_TOP, SMOKE, WHITE};
use crate::core::singing::Heard;
use crate::view::bubbles::Liquid;

/// The least a word's text is made, a share of its size, to fit before
/// the next word.
const SMALLEST_TEXT: f32 = 0.75;
/// How much of what is behind a word's bubble its dark body hides.
const BODY: f32 = 0.88;

/// A word laid on the lane: its line and word, when it is sung (from, to),
/// where it is across (its left and right), its tube, its text's size and
/// width, where its text sits along it (from its left) and whether on one
/// pitch there ([`text_on`]).
pub(super) struct Word {
    pub line: usize,
    pub word: usize,
    pub time: (f64, f64),
    pub x: (f32, f32),
    pub snake: Snake,
    px: f32,
    width: f32,
    text_at: f32,
    level: bool,
}

impl Word {
    /// Where across `t` (seconds into the song) is along it: its sung time
    /// over all of it.
    pub fn across(&self, t: f64) -> f32 {
        self.x.0 + self.share(t) * (self.x.1 - self.x.0)
    }

    /// How much of its sung time is gone at `t`.
    fn share(&self, t: f64) -> f32 {
        let (start, end) = self.time;
        ((t - start) / (end - start).max(1e-3)) as f32
    }

    /// Its middle line's middle.
    pub fn centre(&self) -> (f32, f32) {
        self.snake.centre()
    }
}

/// The words laid out, by line and word.
pub(super) struct Words {
    pub laid: Vec<Word>,
    pub at: HashMap<(usize, usize), usize>,
}

/// How far the singer's arrow is drawn towards the note's tube when the
/// voice is right on it.
const PULL: f32 = 0.5;
/// How far off the note (hundredths of a semitone) the pull is gone.
const PULL_REACH: f64 = 200.0;

impl Words {
    /// Line `l`'s word `w`, where laid.
    pub fn get(&self, l: usize, w: usize) -> Option<&Word> {
        self.at.get(&(l, w)).map(|&k| &self.laid[k])
    }

    /// Where the singer's frame `h` is drawn down the lane: at the voice's
    /// own pitch, drawn towards its word's tube the nearer it is to the
    /// note ([`PULL`] of the way when right on it, none [`PULL_REACH`] off
    /// or more); none off the notes or the words laid.
    pub fn y_of(&self, song: &Song, lane: &Lane, h: &Heard) -> Option<f32> {
        let l = h.line?;
        let notes = song.singing.notes.lines.get(l)?;
        let n = notes.notes.iter().find(|n| h.t >= n.start && h.t < n.end)?;
        let word = self.get(l, n.word)?;
        // A hit's pitch is drawn on its note: the voice is the note and how
        // far off it was.
        let off = h.off.unwrap_or(0.0);
        let voice = if h.hit { h.cents + off } else { h.cents };
        let own = lane.y(notes.height(voice));
        let tube = word.snake.y_at(word.across(h.t));
        let near = (1.0 - off.abs() / PULL_REACH).clamp(0.0, 1.0) as f32;
        let near = if h.off.is_some() { near } else { 0.0 };
        Some(own + (tube - own) * PULL * near)
    }
}

/// A word's text to draw over the trail: it, its word, and when it ends,
/// whether it was hit, how much shows, how far across its liquid reaches
/// unbroken from its left ([`pop::front`]).
pub(super) struct Label {
    text: String,
    word: usize,
    end: f64,
    hit: bool,
    fade: f32,
    poured: f32,
}

impl Stage {
    /// The words of `song`'s lines about now (`from` to `to`, seconds into
    /// the song) laid out on `lane`, their tubes `p` thick.
    pub(super) fn lay_words(&mut self, song: &Song, lane: &Lane, (from, to): (f64, f64)) -> Words {
        let p = lane.pill;
        let text_px = (p * 0.56).round().max(9.0);
        let pad = p * 0.34;
        let singing = song.singing;
        // Each word with notes about now: its line, word and sung time.
        let mut words: Vec<(usize, usize, f64, f64)> = Vec::new();
        for (l, notes) in singing.notes.lines.iter().enumerate() {
            let (Some(first), Some(last)) = (notes.notes.first(), notes.notes.last()) else {
                continue;
            };
            if last.end < from || first.start > to {
                continue;
            }
            for n in &notes.notes {
                match words.last_mut() {
                    Some(wd) if wd.0 == l && wd.1 == n.word => wd.3 = wd.3.max(n.end),
                    _ => words.push((l, n.word, n.start, n.end)),
                }
            }
        }
        let gap = f64::from(p * 0.15 / lane.pps);
        let mut out = Words {
            laid: Vec::with_capacity(words.len()),
            at: HashMap::new(),
        };
        let mut last_end = f64::NEG_INFINITY;
        for (k, &(l, w, start, end)) in words.iter().enumerate() {
            let text = word_text(song, l, w);
            let t0 = start.max(last_end + gap);
            let room = words.get(k + 1).map_or(f32::INFINITY, |next| {
                ((next.2 - t0) as f32) * lane.pps - p * 0.15
            });
            let mut px = text_px;
            let mut width = self.measure(Face::BUBBLE, text, px);
            if width + 2.0 * pad > room {
                let fit = ((room - 2.0 * pad) / width.max(1.0)).clamp(SMALLEST_TEXT, 1.0);
                // In steps of two pixels, so that the atlas keeps few sizes.
                px = ((text_px * fit) / 2.0).floor().max(4.0) * 2.0;
                width = self.measure(Face::BUBBLE, text, px);
            }
            let least = (width + 2.0 * pad).max(p);
            let t1 = end.max(t0 + f64::from(least / lane.pps));
            last_end = t1;
            // As long as its times say, whatever its place in the window.
            let long = ((t1 - t0) as f32) * lane.pps;
            let x = (lane.x(t0), lane.x(t0) + long);
            let mut word = Word {
                line: l,
                word: w,
                time: (start, end),
                x,
                snake: Snake {
                    points: Vec::new(),
                    radius: p / 2.0,
                },
                px,
                width,
                text_at: 0.0,
                level: true,
            };
            // Through its notes, each at its own height: made from its own
            // left, its text placed on it there, then moved to where it
            // is. So both come out the same every frame as it scrolls: on
            // the window's pixels, rounding alone would choose between two
            // stretches as flat (a glide's two ends) and its text jump
            // from one to the other and back.
            let notes = &singing.notes.lines[l].notes;
            let mut through: Vec<(f32, f32, f32)> = (notes.iter().filter(|n| n.word == w))
                .map(|n| {
                    let (a, b) = (word.share(n.start), word.share(n.end));
                    (a * long, b * long, lane.y(n.height))
                })
                .collect();
            if let Some(first) = through.first_mut() {
                first.0 = 0.0;
            }
            if let Some(last) = through.last_mut() {
                last.1 = long;
            }
            let shape = Snake::through(&through, (0.0, long), p / 2.0);
            (word.text_at, word.level) = text_on(&shape, (long, width));
            word.snake = shape.moved(x.0);
            out.at.insert((l, w), out.laid.len());
            out.laid.push(word);
        }
        out
    }

    /// Draws the `words` showing in the window `(x, w)` across at `now`
    /// (the song's time; `clock` the singing's) into `out` (their points
    /// into `paths`): each tube, its bubble (the game on) with its liquid,
    /// straining, bursting as it fills and gone after, its bursts into
    /// `sprays`. Their texts, to draw over the trail.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_words(
        &mut self,
        song: &Song,
        (lane, (x, w)): (&Lane, (f32, f32)),
        words: &Words,
        (out, paths): (&mut Vec<Instance>, &mut Vec<[f32; 4]>),
        sprays: &mut Vec<super::super::particles::Spray>,
        (now, clock): (f64, f64),
    ) -> Vec<Label> {
        let p = lane.pill;
        let mut tubes = pop::Tubes::new(song.singing, clock);
        let mut labels = Vec::new();
        for (k, word) in words.laid.iter().enumerate() {
            let (x0, x1) = word.x;
            if x1 < x - p || x0 > x + w + p {
                continue;
            }
            let fade = edge_fade(x0, (x, w)).min(edge_fade(x1, (x, w)));
            if fade <= 0.0 {
                continue;
            }
            let (start, end) = word.time;
            let across = |t: f64| word.across(t);
            let at = Laid {
                snake: &word.snake,
                ends: word.x,
                across: &across,
            };
            let liquid = tubes.liquid(song.game, word.line, word.word);
            if let Some(b) = &liquid {
                sprays.extend(pop::sprays(b, &at, p, (clock, lane.pps)));
            }
            // Dark inside, so their words read over any cover; outlined in
            // gold with a soft halo behind, brightest while sung.
            let (body, edge, glow) = if end <= now {
                (rgba(SMOKE, 0.8), rgba(GOLD, 0.4), p * 0.2)
            } else if start <= now {
                (rgba(SMOKE, BODY), rgba(ACTIVE, 1.0), p * 0.5)
            } else {
                (rgba(SMOKE, BODY), rgba(GOLD, 0.9), p * 0.3)
            };
            let faded = |c: [f32; 4]| [c[0], c[1], c[2], c[3] * fade];
            let skin = Skin {
                body: faded(body),
                edge: faded(edge),
                glow,
                outline: (p * 0.05).max(1.2),
            };
            // Their liquid, as the time went by under the playhead.
            let tint = Tint {
                top: rgba(GOLD_TOP, fade),
                bottom: rgba(GOLD_DEEP, 0.8 * fade),
                glow: p * 0.35,
            };
            pop::draw(liquid.as_ref(), (out, paths), &at, (&skin, &tint), p);
            let filled = liquid.as_ref().map_or(0.0, Liquid::filled);
            let text = word_text(song, word.line, word.word);
            // Burst, its words go with it.
            if !text.is_empty() && !liquid.as_ref().is_some_and(Liquid::torn) {
                labels.push(Label {
                    text: text.to_owned(),
                    word: k,
                    end,
                    hit: filled > 0.5 * (end - start),
                    fade,
                    poured: liquid
                        .as_ref()
                        .map_or(f32::NEG_INFINITY, |b| pop::front(b, &at)),
                });
            }
        }
        labels
    }

    /// Draws the words' `labels` (of `words`) into `out`, each on the
    /// flattest stretch of its tube, its letters riding the tube's middle
    /// there at `now`: dark on the gold as far as it is poured unbroken;
    /// faint once over and missed.
    pub(super) fn draw_labels(
        &mut self,
        out: &mut Vec<Instance>,
        words: &Words,
        labels: Vec<Label>,
        now: f64,
    ) {
        for label in labels {
            let word = &words.laid[label.word];
            let fade = label.fade;
            let alpha = if label.end <= now && !label.hit {
                0.45
            } else {
                0.95
            } * fade;
            let (split, done) = (label.poured, [0.27, 0.15, 0.03, 0.95 * fade]);
            let (px, colors) = (word.px, (rgba(WHITE, alpha), done));
            // Lit, its letters get a soft dark rim: the gold of a note further
            // on, poured after one missed, is bright.
            let rim = label.hit.then_some((super::Shade::RIM, alpha));
            let cap = self.cap_middle(Face::BUBBLE, px) - super::COVER_PAD as f32;
            // Where its layout put it ([`text_on`]): a run on its pitch
            // where the tube is flat there, else each letter at the tube's
            // height under it.
            let mut pen = word.x.0 + word.text_at;
            if word.level {
                let at = (pen, word.snake.y_at(pen + word.width / 2.0) - cap);
                if let Some(rim) = rim {
                    let whole = (at, f32::INFINITY);
                    self.shade(out, (Face::BUBBLE, &label.text), (px, 1.0), whole, rim);
                }
                self.text(out, Face::BUBBLE, &label.text, (px, 1.0), at, split, colors);
                continue;
            }
            let mut buf = [0u8; 4];
            for c in label.text.chars() {
                let letter = c.encode_utf8(&mut buf);
                let wide = self.measure(Face::BUBBLE, letter, px);
                let y = word.snake.y_at(pen + wide / 2.0) - cap;
                if let Some(rim) = rim {
                    let whole = ((pen, y), f32::INFINITY);
                    self.shade(out, (Face::BUBBLE, letter), (px, 1.0), whole, rim);
                }
                self.text(
                    out,
                    Face::BUBBLE,
                    letter,
                    (px, 1.0),
                    (pen, y),
                    split,
                    colors,
                );
                pen += wide;
            }
        }
    }
}

/// Where text `width` across sits in a word's tube `shape` (made from its
/// left, at 0, `long` across): its left, on the flattest stretch
/// ([`Snake::flattest`], half a radius in from the ends), and whether the
/// tube is on one pitch all under it.
fn text_on(shape: &Snake, (long, width): (f32, f32)) -> (f32, bool) {
    let inset = shape.radius * 0.5;
    let pen = shape.flattest(width, (inset, long - inset));
    let ys = [pen, pen + width, pen + width / 2.0].map(|x| shape.y_at(x));
    (pen, ys[0] == ys[1] && ys[0] == ys[2])
}

/// Line `l`'s word `w`'s text in `song` (none where it has none).
fn word_text<'a>(song: &Song<'a>, l: usize, w: usize) -> &'a str {
    song.sung
        .get(l)
        .and_then(|line| line.words.get(w))
        .map_or("", |w| w.text.as_str())
}

#[cfg(test)]
mod tests {
    use crate::core::singing::Singing;
    use crate::gui::stage::{Song, Stage};
    use crate::gui::stage_gpu::{kind, StageFrame};
    use crate::view::karaoke::KaraokePicture;
    use ratatui::layout::Rect;

    /// A song of one short word, "my", gliding up from one note to
    /// another as long, its singing ready.
    fn glide() -> (Singing, std::sync::Arc<jodel_lyrics::Lyrics>) {
        let line = jodel_lyrics::Line {
            words: vec![jodel_lyrics::Word {
                start: Some(1.0),
                end: Some(1.4),
                confidence: 0.9,
                ..jodel_lyrics::Word::new("my")
            }],
            ..jodel_lyrics::Line::default()
        };
        let lyrics = std::sync::Arc::new(jodel_lyrics::Lyrics {
            lines: vec![line],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        });
        let mut melody = jodel_rhythm::waveform::Melody {
            cents: vec![0; 300],
            sure: vec![0; 300],
        };
        for i in 100..140 {
            melody.cents[i] = if i < 120 { 6000 } else { 6700 };
            melody.sure[i] = 255;
        }
        let mut singing = Singing::default();
        let notes = crate::core::singing::melodic(&melody);
        singing.prepare(std::path::Path::new("a"), &lyrics, &notes);
        (singing, lyrics)
    }

    /// The word's text in `frame`, each piece where it is from the tube's
    /// left (across) and down.
    fn text_in_tube(frame: &StageFrame) -> Vec<(f32, f32)> {
        let tubes: Vec<_> = (frame.instances.iter())
            .filter(|i| i.params[0] == kind::TUBE)
            .collect();
        let [tube] = tubes[..] else {
            panic!("one tube: {}", tubes.len());
        };
        let left = frame.paths[tube.shape[3] as usize][0];
        let q = tube.quad;
        (frame.instances.iter())
            .filter(|i| i.params[0] == kind::TEXT)
            .filter(|i| {
                let (x, y) = ((i.quad[0] + i.quad[2]) / 2.0, (i.quad[1] + i.quad[3]) / 2.0);
                x > q[0] && x < q[2] && y > q[1] && y < q[3]
            })
            .map(|i| (i.quad[0] - left, i.quad[1]))
            .collect()
    }

    /// A word's text sits in the same place in its tube every frame as the
    /// tube scrolls across the window, at any frame rate: a glide (the
    /// word's two notes as long) is as flat at its two ends, and where its
    /// text goes is not left to how the window's pixels round, frame by
    /// frame (it jumped from one end to the other and back).
    #[test]
    fn a_word_s_text_stays_put_in_its_tube_as_it_scrolls() {
        let (singing, lyrics) = glide();
        let song = Song::new(&singing, lyrics.lines.iter().collect(), Some(3.0));
        let area = Rect::new(0, 0, 100, 24);
        let picture = KaraokePicture {
            area,
            bar: crate::core::karaoke::controls::bar(area),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        };
        for (w, h, fps) in [(1600.0, 900.0, 60.0), (1512.0, 982.0, 120.0)] {
            let mut stage = Stage::default();
            let mut first: Option<Vec<(f32, f32)>> = None;
            // From coming in on the right to the playhead.
            let frames = (3.0 * fps) as usize;
            for f in 0..frames {
                let now = -2.0 + f as f64 / fps;
                let frame = stage.frame(&song, &picture, (0.0, 0.0, w, h), now, 1.0 / fps as f32);
                let text = text_in_tube(&frame);
                assert!(!text.is_empty(), "{w}x{h} at {now:.3}: its text");
                let first = first.get_or_insert_with(|| text.clone());
                let moved = (first.iter().zip(&text))
                    .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
                    .fold(0.0, f32::max);
                assert!(
                    text.len() == first.len() && moved < 0.01,
                    "{w}x{h} at {now:.3}: its text moved {moved} px in its tube, \
                     {first:?} then {text:?}"
                );
            }
        }
    }
}
