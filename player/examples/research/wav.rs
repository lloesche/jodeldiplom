//! A separated voice as the research tools keep it.

use std::path::Path;

/// A 16 kHz mono 32-bit float WAV file's samples.
pub fn read_wav(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    jodel_dsp::wav::parse(&bytes)?.float32()
}
