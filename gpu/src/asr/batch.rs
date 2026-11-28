//! Pieces of the same length heard together: the encoder and the prompts
//! over all of them at once, then the decoder a step for all, a row each,
//! until each has ended (its tokens past its end not read). Each row's
//! sums are what they would be alone, so each piece gets the tokens it
//! gets on its own; one that has not ended when the room the pieces
//! together have is full is heard again on its own ([`super::NEW_AT_ONCE`]).
//!
//! The steps go a few at a time ([`STEPS_A_READ`]): the prompts' first
//! tokens are read while the first steps run (the next token and its
//! embedding are chosen on the GPU), and a piece heard alone (a part of a
//! transcript, its tokens many) has each read of its tokens made while the
//! next steps run, so the GPU does not wait for the reads, the steps
//! running one read past its end. Pieces heard together (glimpses, their
//! tokens few) wait for each read instead: a read more of steps for all of
//! them would cost more than the waits. A step is never split between
//! submissions (`Recorded::whole`): however the GPU is shared with the
//! window, a step costs one submission and one wait at most.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use jodel_lyrics::asr::{loop_start, text, DIM};

use super::ops::{each, Op};
use super::{Buf, Inner, Kernel, Shape, MAX_NEW, STEPS_A_READ};
use crate::pace::Phase;

/// What the recogniser did so far ([`super::GpuAsr::hearing`]), for
/// traces: the difference of two ([`Hearing::since`]) is what it did
/// between them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hearing {
    /// The runs (pieces heard together, or one alone), and their pieces.
    pub runs: usize,
    pub pieces: usize,
    /// The encoder and the prompts, and their GPU work.
    pub prompted: Duration,
    pub prompt_gpu: crate::pace::Tally,
    /// The tokens written, the decoder's steps (each a token for every
    /// piece of its run), their time and their GPU work.
    pub tokens: usize,
    pub steps: usize,
    pub decoded: Duration,
    pub step_gpu: crate::pace::Tally,
}

impl Hearing {
    /// What was added since `earlier` (an earlier one of the same
    /// recogniser).
    pub fn since(&self, earlier: &Hearing) -> Hearing {
        Hearing {
            runs: self.runs.saturating_sub(earlier.runs),
            pieces: self.pieces.saturating_sub(earlier.pieces),
            prompted: self.prompted.saturating_sub(earlier.prompted),
            prompt_gpu: self.prompt_gpu.since(&earlier.prompt_gpu),
            tokens: self.tokens.saturating_sub(earlier.tokens),
            steps: self.steps.saturating_sub(earlier.steps),
            decoded: self.decoded.saturating_sub(earlier.decoded),
            step_gpu: self.step_gpu.since(&earlier.step_gpu),
        }
    }

    fn add(&mut self, run: &Hearing) {
        self.runs += run.runs;
        self.pieces += run.pieces;
        self.prompted += run.prompted;
        self.prompt_gpu = self.prompt_gpu.and(&run.prompt_gpu);
        self.tokens += run.tokens;
        self.steps += run.steps;
        self.decoded += run.decoded;
        self.step_gpu = self.step_gpu.and(&run.step_gpu);
    }
}

impl std::fmt::Display for Hearing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let steps = self.steps.max(1) as f64;
        write!(
            f,
            "{} pieces in {} runs: the encoder and prompts {:.2} s ({} submissions, {} waits), \
             {} tokens in {} steps in {:.2} s ({:.0} tokens a second; {:.2} ms, {:.2} submissions \
             and {:.2} waits for the GPU a step)",
            self.pieces,
            self.runs,
            self.prompted.as_secs_f64(),
            self.prompt_gpu.submissions,
            self.prompt_gpu.waits,
            self.tokens,
            self.steps,
            self.decoded.as_secs_f64(),
            self.tokens as f64 / self.decoded.as_secs_f64().max(1e-9),
            self.decoded.as_secs_f64() * 1000.0 / steps,
            self.step_gpu.submissions as f64 / steps,
            self.step_gpu.waits as f64 / steps,
        )
    }
}

/// A piece's tokens while it is decoded.
struct Written {
    ids: Vec<u32>,
    /// It has ended (its end written, or a loop cut short), or has written
    /// [`MAX_NEW`].
    done: bool,
}

impl Written {
    fn new(first: f32) -> Written {
        let mut w = Written {
            ids: Vec::new(),
            done: false,
        };
        w.take(&[first]);
        w
    }

    /// Takes the next tokens read, up to an end or a loop.
    fn take(&mut self, got: &[f32]) {
        for &v in got {
            let t = v as u32;
            self.ids.push(t);
            if text::is_end(t) {
                break;
            }
            if let Some(keep) = loop_start(&self.ids) {
                self.ids.truncate(keep);
                self.ids.push(text::IM_END);
                break;
            }
        }
        self.done = self.ids.last().is_some_and(|&t| text::is_end(t)) || self.ids.len() >= MAX_NEW;
    }

    /// The tokens, without the end.
    fn finish(mut self) -> Vec<u32> {
        while self.ids.last().is_some_and(|&t| text::is_end(t)) {
            self.ids.pop();
        }
        self.ids.truncate(MAX_NEW);
        self.ids
    }
}

impl Inner {
    /// The tokens written for each of `features` (each `[MELS][frames]`),
    /// heard together; `None` for one not ended in the room they have
    /// together (never for one heard on its own).
    pub(super) fn hear(
        &mut self,
        features: &[&[f32]],
        frames: usize,
    ) -> Result<Vec<Option<Vec<u32>>>, String> {
        let shape = Shape::with_room(features.len(), frames, self.new_at_once);
        let (b, cap, r) = (shape.pieces, shape.positions, shape.rows);
        let (t0, prompts) = (Instant::now(), self.paces.prompt.tally());
        let steps_before = self.paces.steps.tally();
        let ops = self.prompt_run(features, &shape);
        let read_all = Some((Buf::Tokens, 0, b * cap));
        // The runs made and not read yet (each its first token's place and
        // its tokens a piece), in their order, each into its own staging
        // buffer: two at most (the prompts and the first steps; a piece
        // alone's steps), else one.
        let encoded = self.submit(&ops, 1, read_all, Phase::Prompt, 0)?;
        let mut flying = VecDeque::from([(encoded, r, 1)]);
        let mut t1 = None;
        // None until the prompts' first tokens are read.
        let mut written: Option<Vec<Written>> = None;
        // Every piece has written as many as the others still writing.
        let a_read = STEPS_A_READ;
        let (mut made, mut steps, mut runs) = (1, 0, 1);
        loop {
            let done = written.as_ref().is_some_and(|w| w.iter().all(|w| w.done));
            let ahead = if b == 1 || written.is_none() { 2 } else { 1 };
            if flying.len() < ahead && !done && r + made + a_read < cap {
                let mut ops = Vec::new();
                let pos = r + made - 1;
                self.step_ops(&mut ops, &shape, pos);
                // A step's dispatches, kept whole in a submission.
                let whole = ops.len();
                for k in 1..a_read {
                    self.step_ops(&mut ops, &shape, pos + k);
                }
                let running = self.submit(&ops, whole, read_all, Phase::Steps, runs % 2)?;
                flying.push_back((running, pos + 1, a_read));
                (made, steps, runs) = (made + a_read, steps + a_read, runs + 1);
                continue;
            }
            let Some((running, at, count)) = flying.pop_front() else {
                break;
            };
            let first = written.is_none();
            let got = self.read(running, if first { Phase::Prompt } else { Phase::Steps })?;
            let w = written.get_or_insert_with(|| {
                t1 = Some(Instant::now());
                (0..b).map(|k| Written::new(got[k * cap + r])).collect()
            });
            if !first {
                for (k, w) in w.iter_mut().enumerate() {
                    if !w.done {
                        w.take(&got[k * cap + at..][..count]);
                    }
                }
            }
        }
        let written = written.unwrap_or_default();
        let t1 = t1.unwrap_or(t0);
        self.hearing.add(&Hearing {
            runs: 1,
            pieces: b,
            prompted: t1 - t0,
            prompt_gpu: self.paces.prompt.tally().since(&prompts),
            tokens: written.iter().map(|w| w.ids.len()).sum(),
            steps,
            decoded: t1.elapsed(),
            step_gpu: self.paces.steps.tally().since(&steps_before),
        });
        Ok(written
            .into_iter()
            .map(|w| {
                // Alone, a piece always has room for MAX_NEW.
                (w.done || b == 1).then(|| w.finish())
            })
            .collect())
    }

    /// The encoder and the prompts of `shape`'s pieces (`features`, as
    /// [`Inner::hear`]'s): each piece's tokens so far (`shape.positions`
    /// a piece), its first written after its prompt.
    #[cfg(test)]
    pub(super) fn prompted(
        &mut self,
        features: &[&[f32]],
        shape: &Shape,
    ) -> Result<Vec<f32>, String> {
        let ops = self.prompt_run(features, shape);
        let read = Some((Buf::Tokens, 0, shape.pieces * shape.positions));
        self.run(&ops, read, Phase::Prompt)
    }

    /// The dispatches of [`Inner::prompted`], `features` uploaded and the
    /// prompts' tokens written first.
    fn prompt_run(&mut self, features: &[&[f32]], shape: &Shape) -> Vec<Op> {
        let (b, cap, r) = (shape.pieces, shape.positions, shape.rows);
        self.upload(features, shape);
        let prompt = text::prompt(shape.steps);
        let mut ids = vec![0.0f32; b * cap];
        for k in 0..b {
            for (slot, &id) in ids[k * cap..].iter_mut().zip(&prompt) {
                *slot = id as f32;
            }
        }
        self.program.queue.write_buffer(
            &self.buffers[&Buf::Tokens],
            0,
            &crate::engine::bytes(&ids),
        );
        let mut ops = self.encoder_ops(shape);
        let rows = b * r;
        each(
            &mut ops,
            Kernel::Embed,
            [Buf::Tokens, Buf::Small, Buf::X],
            rows * DIM,
            &[
                rows,
                0,
                self.tensors.embed.at,
                0,
                self.tensors.embed.format.bits(),
                r,
                cap,
            ],
        );
        let audio_at = prompt
            .iter()
            .position(|&i| i == text::AUDIO_PAD)
            .unwrap_or(0);
        each(
            &mut ops,
            Kernel::Copy,
            [Buf::Audio, Buf::Small, Buf::X],
            b * shape.steps * DIM,
            &[
                b * shape.steps,
                DIM,
                0,
                DIM,
                audio_at * DIM,
                DIM,
                shape.steps,
                r * DIM,
            ],
        );
        self.prompt_ops(&mut ops, shape);
        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_piece_stops_at_its_end_or_a_loop() {
        let mut w = Written::new(5.0);
        assert!(!w.done);
        w.take(&[6.0, 7.0, text::IM_END as f32, 9.0]);
        assert!(w.done);
        assert_eq!(w.finish(), vec![5, 6, 7]);
        let mut w = Written::new(1.0);
        for _ in 0..8 {
            w.take(&[2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
            if w.done {
                break;
            }
        }
        assert!(w.done, "a loop ends it");
        assert!(w.finish().len() < 40);
    }
}
