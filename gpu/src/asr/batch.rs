//! Pieces of the same length heard together: the encoder and the prompts
//! over all of them at once, then the decoder a step for all, a row each,
//! until each has ended (its tokens past its end not read). Each row's
//! sums are what they would be alone, so each piece gets the tokens it
//! gets on its own; one that has not ended when the room the pieces
//! together have is full is heard again on its own ([`super::NEW_AT_ONCE`]).

use std::time::Instant;

use jodel_lyrics::asr::{loop_start, text, DIM};

use super::ops::each;
use super::shape::read;
use super::{Buf, Inner, Kernel, Runs, Shape, MAX_NEW};

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
        let t0 = Instant::now();
        let got = self.prompted(features, &shape)?;
        let prompted = t0.elapsed();
        let t1 = Instant::now();
        let mut written: Vec<Written> = (0..b).map(|k| Written::new(got[k * cap + r])).collect();
        // Every piece has written as many as the others still writing.
        let a_read = read(b);
        let mut decoded = 1;
        let (mut steps, mut dispatches) = (0, 0);
        let submitted = crate::pace::submissions();
        loop {
            if written.iter().all(|w| w.done) || r + decoded + a_read >= cap {
                break;
            }
            let mut ops = Vec::new();
            let pos = r + decoded - 1;
            for k in 0..a_read {
                self.step_ops(&mut ops, &shape, pos + k);
            }
            let got = self.run(&ops, Some((Buf::Tokens, 0, b * cap)), Runs::Steps)?;
            steps += a_read;
            dispatches += ops.len();
            for (k, w) in written.iter_mut().enumerate() {
                if !w.done {
                    w.take(&got[k * cap + pos + 1..][..a_read]);
                }
            }
            decoded += a_read;
        }
        if std::env::var_os("JODEL_ASR_TIMES").is_some() {
            let tokens: usize = written.iter().map(|w| w.ids.len()).sum();
            let decoding = t1.elapsed();
            let a_step = |n: f64| n / steps.max(1) as f64;
            eprintln!(
                "asr: {b} pieces of {} steps, {r} rows: encoder+prefill {prompted:.2?}, {tokens} tokens in {steps} steps in {decoding:.2?} ({:.2?}, {} dispatches, {:.2} submissions a step)",
                shape.steps,
                decoding.div_f64(steps.max(1) as f64),
                a_step(dispatches as f64),
                a_step((crate::pace::submissions() - submitted) as f64),
            );
        }
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
    pub(super) fn prompted(
        &mut self,
        features: &[&[f32]],
        shape: &Shape,
    ) -> Result<Vec<f32>, String> {
        let (b, cap, r) = (shape.pieces, shape.positions, shape.rows);
        self.upload(features, shape);
        let prompt = text::prompt(shape.steps);
        let mut ids = vec![0.0f32; b * cap];
        for k in 0..b {
            for (slot, &id) in ids[k * cap..].iter_mut().zip(&prompt) {
                *slot = id as f32;
            }
        }
        self.queue
            .write_buffer(&self.buffers[&Buf::Tokens], 0, &super::super::bytes(&ids));
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
        self.run(&ops, Some((Buf::Tokens, 0, b * cap)), Runs::Prompt)
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
