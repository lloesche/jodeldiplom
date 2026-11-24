//! A separated voice as the research tools keep it.

use std::path::Path;

/// A 16 kHz mono 32-bit float WAV file's samples (after a 44-byte header).
pub fn read_wav(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .get(44..)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    )
}
