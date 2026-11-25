//! Our own inference of the Beat This! network, small version (Foscarin,
//! Schlüter and Widmer, "Beat this! Accurate beat tracking without DBN
//! postprocessing", ISMIR 2024; code and weights under the MIT licence,
//! see `models/LICENSE-beat-this`).
//!
//! It reads its own spectrogram ([`NetFrontend`]) and gives, per frame at
//! 50 frames a second, a beat logit and a downbeat logit. The weights are
//! built into the crate; `tools/convert_beat_this.py` made them from the
//! authors' `small0` checkpoint.
//!
//! A track is run in pieces of 30 s (1500 frames) that overlap by 12
//! frames; the 6 frames at each edge of a piece are dropped, and where
//! pieces overlap the earlier one counts, as in the authors' inference.
//!
//! A [`NetRunner`] runs a piece's layers somewhere else (the GPU, see
//! `jodel-gpu`); [`Network::predict_with`] takes one, and a piece it fails
//! on runs here instead. Its layers and weights file are the crate's
//! shared ones ([`crate::nn`]).

pub mod mel;
mod model;

use std::sync::OnceLock;

use crate::nn::{threads, weights::Weights};
pub use mel::{NetFrontend, NetMel};
use model::{Model, MAX_FRAMES};

/// The weights, built in.
static WEIGHTS: &[u8] = include_bytes!("../../models/beat_this_small.bin");

/// Frames dropped at each edge of a piece.
const BORDER: usize = 6;

/// The most frames of a piece (30 s).
pub const PIECE_FRAMES: usize = MAX_FRAMES;

/// The built-in weights, parsed afresh (for a [`NetRunner`] to load).
pub fn built_in_weights() -> Result<Weights, String> {
    Weights::parse(WEIGHTS)
}

/// Runs the network's layers on a piece somewhere else than here (a
/// GPU), with the built-in weights.
pub trait NetRunner: Send + Sync {
    /// What it runs on, for reports.
    fn name(&self) -> String;

    /// The beat and downbeat logits of `frames` spectrogram frames
    /// (frame-major, [`mel::BANDS`] each; at most [`PIECE_FRAMES`]), as
    /// the network here gives them.
    fn forward(&self, spect: &[f32], frames: usize) -> Result<(Vec<f32>, Vec<f32>), String>;
}

/// The network's output for a track.
#[derive(Clone, Debug, Default)]
pub struct Logits {
    /// Per frame: whether a beat is there (beats include downbeats).
    pub beat: Vec<f32>,
    /// Per frame: whether a downbeat is there.
    pub downbeat: Vec<f32>,
}

/// The network, loaded once.
pub struct Network {
    model: Model,
}

impl Network {
    /// The built-in network, loaded on first use.
    pub fn shared() -> &'static Network {
        static NETWORK: OnceLock<Network> = OnceLock::new();
        NETWORK.get_or_init(|| {
            let weights = Weights::parse(WEIGHTS).expect("the built-in weights");
            Network {
                model: Model::load(weights).expect("the built-in weights"),
            }
        })
    }

    /// The logits for a whole track's spectrogram.
    pub fn predict(&self, mel: &NetMel) -> Logits {
        self.predict_with(mel, None)
    }

    /// The logits for a whole track's spectrogram, each piece through
    /// `runner` where there is one (a piece it fails on runs here).
    pub fn predict_with(&self, mel: &NetMel, runner: Option<&dyn NetRunner>) -> Logits {
        let n = mel.frames;
        let mut out = Logits {
            beat: vec![-1000.0; n],
            downbeat: vec![-1000.0; n],
        };
        if n == 0 {
            return out;
        }
        let threads = threads();
        let step = MAX_FRAMES - 2 * BORDER;
        let mut starts: Vec<isize> = (0..)
            .map(|i| i as isize * step as isize - BORDER as isize)
            .take_while(|&s| s < n as isize - BORDER as isize)
            .collect();
        if n > step {
            *starts.last_mut().unwrap() = n as isize - (MAX_FRAMES - BORDER) as isize;
        }
        // The earlier piece wins where they overlap: write the last first.
        for &start in starts.iter().rev() {
            let from = start.max(0) as usize;
            let to = (start + MAX_FRAMES as isize).min(n as isize) as usize;
            let left = (-start).max(0) as usize;
            let right =
                (start + MAX_FRAMES as isize - n as isize).clamp(0, BORDER as isize) as usize;
            let frames = left + (to - from) + right;
            let mut spect = vec![0.0f32; frames * mel::BANDS];
            spect[left * mel::BANDS..][..(to - from) * mel::BANDS]
                .copy_from_slice(&mel.data[from * mel::BANDS..to * mel::BANDS]);
            let (beat, downbeat) = runner
                .and_then(|r| r.forward(&spect, frames).ok())
                .filter(|(b, d)| b.len() == frames && d.len() == frames)
                .unwrap_or_else(|| self.model.forward(&spect, frames, threads));
            for i in BORDER..frames.saturating_sub(BORDER) {
                let at = start + i as isize;
                if at >= 0 && (at as usize) < n {
                    out.beat[at as usize] = beat[i];
                    out.downbeat[at as usize] = downbeat[i];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_weights_load() {
        let _ = Network::shared();
    }
}
