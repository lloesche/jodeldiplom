//! What the tests share: an engine rendered on its own, a note, the synth
//! rendered, and the measures of a sound ([`measure`]).

#![allow(dead_code)]

pub mod measure;

use jodel_synth::dsp::Rate;
use jodel_synth::{Engine, NoteOn, Synth, VoiceMod, BLOCK};

/// The rate the tests run at.
pub const RATE: u32 = 48_000;

/// The modulation of a note held on `note`, its brightness moved
/// `bright` octaves.
pub fn held(note: f32, bright: f32) -> VoiceMod {
    VoiceMod {
        phase_inc: Rate::new(RATE).inc(note),
        pitch: note,
        bright,
        shape: 0.0,
    }
}

/// A note at full velocity, in the middle.
pub fn on(note: f32, seed: u32) -> NoteOn {
    NoteOn {
        note,
        velocity: 1.0,
        seed,
        pan: 0.0,
        shape: 0.0,
        glide: None,
        line: 0,
    }
}

/// `seconds` of engine `e` playing `p` on `note` (at unity level, at
/// `rate`), its brightness moved `bright` octaves.
pub fn solo<E: Engine>(
    e: &E,
    p: &E::Params,
    rate: u32,
    note: f32,
    seconds: f32,
    seed: u32,
    bright: f32,
) -> Vec<f32> {
    let r = Rate::new(rate);
    let mut v = E::Voice::default();
    e.note_on(p, &mut v, &on(note, seed));
    let m = VoiceMod {
        phase_inc: r.inc(note),
        ..held(note, bright)
    };
    let blocks = (seconds * rate as f32) as usize / BLOCK;
    let mut out = vec![0.0; blocks * BLOCK];
    for block in out.chunks_exact_mut(BLOCK) {
        e.render(p, &mut v, &m, block);
    }
    out
}

/// `frames` of `s`, interleaved stereo.
pub fn render(s: &mut Synth, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0; 2 * frames];
    s.render_to(&mut out);
    out
}

/// The left side of interleaved stereo.
pub fn left(stereo: &[f32]) -> Vec<f32> {
    stereo.iter().step_by(2).copied().collect()
}
