//! The banks of sounds kept on disk ([`crate::cache::SFX_BANK`]): one
//! file a rate in the data folder's `sounds` ([`crate::dirs::Data::sounds`]),
//! so a rate the player has had before costs nothing, even after a
//! restart. A file is a bank as made by this build's sounds only: its key
//! is [`SOUNDS_KEY`], a hash of the code that makes them (taken by
//! `build.rs`), so a change to the sounds makes them again on its own.
//! Each sound is kept under its name ([`Sound::name`]) and found by it, so
//! the order of the sounds in a bank plays no part.
//!
//! The layout, little-endian:
//!
//! ```text
//! "JDLS"  version: u8  key: u64  rate: u32  channels: u16  sounds: u32
//! each sound: its name's length in bytes: u16, its name (UTF-8), its
//!             length in samples: u32
//! the samples, each sound's in turn: f32 ...
//! ```
//!
//! A file is read with every length checked against what it holds; one
//! that does not match in any way (another key, rate or layout, short,
//! long, a sound of this build's missing) is not read, and the bank is
//! made again and written over it.

use std::path::{Path, PathBuf};

use jodel_output::mixer::SoundBank;

use super::Sound;

include!(concat!(env!("OUT_DIR"), "/sfx_key.rs"));

/// The first bytes of a bank's file.
const MAGIC: &[u8; 4] = b"JDLS";
/// The channels of every sound.
const CHANNELS: u16 = 2;
/// The bytes before the sounds' names and lengths.
const HEAD: usize = 4 + 1 + 8 + 4 + 2 + 4;

/// Sounds at `rate` as kept, each under its name, made by the sounds of
/// `key`.
pub fn to_bytes(rate: u32, sounds: &[(String, &[f32])], key: u64) -> Vec<u8> {
    let samples: usize = sounds.iter().map(|s| s.1.len()).sum();
    let mut out = Vec::with_capacity(HEAD + 32 * sounds.len() + 4 * samples);
    out.extend_from_slice(MAGIC);
    out.push(crate::cache::SFX_BANK.version as u8);
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&(sounds.len() as u32).to_le_bytes());
    for (name, s) in sounds {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for v in sounds.iter().flat_map(|s| s.1.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The sounds `bytes` keep, each under its name, when they are a whole
/// file at `rate` made by the sounds of `key` in this layout.
pub fn from_bytes(bytes: &[u8], key: u64, rate: u32) -> Option<Vec<(String, Vec<f32>)>> {
    let take = |at: &mut usize, n: usize| {
        let b = bytes.get(*at..at.checked_add(n)?)?;
        *at += n;
        Some(b)
    };
    let mut at = 0;
    if take(&mut at, 4)? != MAGIC || take(&mut at, 1)?[0] != crate::cache::SFX_BANK.version as u8 {
        return None;
    }
    let kept_key = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
    let kept_rate = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?);
    let channels = u16::from_le_bytes(take(&mut at, 2)?.try_into().ok()?);
    if kept_key != key || kept_rate != rate || channels != CHANNELS {
        return None;
    }
    let count = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?) as usize;
    let mut heads = Vec::new();
    for _ in 0..count {
        let len = u16::from_le_bytes(take(&mut at, 2)?.try_into().ok()?);
        let name = std::str::from_utf8(take(&mut at, usize::from(len))?).ok()?;
        let n = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?) as usize;
        if !n.is_multiple_of(usize::from(CHANNELS)) {
            return None;
        }
        heads.push((name.to_owned(), n));
    }
    let samples: usize = heads.iter().map(|h| h.1).sum();
    if bytes.len() != at.checked_add(samples.checked_mul(4)?)? {
        return None;
    }
    let sounds = heads
        .into_iter()
        .map(|(name, n)| {
            let b = &bytes[at..at + 4 * n];
            at += 4 * n;
            let sound = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect();
            (name, sound)
        })
        .collect();
    Some(sounds)
}

/// The banks kept in a folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankStore {
    dir: PathBuf,
}

impl BankStore {
    /// The banks kept in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> BankStore {
        BankStore { dir: dir.into() }
    }

    /// The file of the bank at `rate`.
    pub fn file(&self, rate: u32) -> PathBuf {
        self.dir.join(format!("{rate}.bank"))
    }

    /// The bank at `rate` as kept, if one made by this build's sounds is:
    /// each of [`Sound::all`] found by its name, in that order (a sound
    /// missing, or two of one name: none).
    pub fn load(&self, rate: u32) -> Option<SoundBank> {
        let bytes = std::fs::read(self.file(rate)).ok()?;
        let mut kept = from_bytes(&bytes, SOUNDS_KEY, rate)?;
        let mut bank = SoundBank::new(rate);
        for sound in Sound::all() {
            let name = sound.name();
            let mut found = kept.iter().filter(|k| k.0 == name);
            if found.next().is_none() || found.next().is_some() {
                return None;
            }
            let k = kept.iter().position(|k| k.0 == name)?;
            bank.add(kept.swap_remove(k).1);
        }
        Some(bank)
    }

    /// Keeps `bank` (each sound under the name of [`Sound::all`]'s at its
    /// place), in place of the one at its rate (whole or not at all).
    pub fn save(&self, bank: &SoundBank) -> Result<(), String> {
        let fail = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
        std::fs::create_dir_all(&self.dir).map_err(|e| fail(&self.dir, e))?;
        let file = self.file(bank.rate());
        let sounds: Vec<(String, &[f32])> = Sound::all()
            .into_iter()
            .filter_map(|s| Some((s.name(), bank.sound(s.index())?)))
            .collect();
        let bytes = to_bytes(bank.rate(), &sounds, SOUNDS_KEY);
        durable::write_atomic(&file, &bytes).map_err(|e| fail(&file, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Vec<(String, Vec<f32>)> {
        vec![
            ("one".into(), vec![0.25, -0.5, 1.0, f32::MIN_POSITIVE]),
            ("none".into(), Vec::new()),
            ("two".into(), vec![-1.0, 0.125]),
        ]
    }

    fn bytes_of(sounds: &[(String, Vec<f32>)], rate: u32, key: u64) -> Vec<u8> {
        let refs: Vec<(String, &[f32])> = sounds.iter().map(|s| (s.0.clone(), &s.1[..])).collect();
        to_bytes(rate, &refs, key)
    }

    #[test]
    fn sounds_come_back_bit_for_bit_and_only_as_they_were_made() {
        let bytes = bytes_of(&tiny(), 48_000, 7);
        assert_eq!(from_bytes(&bytes, 7, 48_000), Some(tiny()));
        // Another key (the sounds changed), another rate: not read.
        assert_eq!(from_bytes(&bytes, 8, 48_000), None);
        assert_eq!(from_bytes(&bytes, 7, 44_100), None);
        // Short, long, or its header wrong anywhere: not read.
        assert_eq!(from_bytes(&bytes[..bytes.len() - 1], 7, 48_000), None);
        assert_eq!(from_bytes(&[&bytes[..], &[0]].concat(), 7, 48_000), None);
        for at in [0, 4, 17, 19, HEAD, HEAD + 5 + 3] {
            let mut wrong = bytes.clone();
            wrong[at] ^= 0x40;
            assert_eq!(from_bytes(&wrong, 7, 48_000), None, "byte {at}");
        }
        assert_eq!(from_bytes(&[], 7, 48_000), None);
    }

    /// A bank of every sound, sound `k` four samples of `k`.
    fn full(rate: u32) -> SoundBank {
        let mut b = SoundBank::new(rate);
        for k in 0..Sound::COUNT {
            b.add(vec![k as f32; 4]);
        }
        b
    }

    #[test]
    fn the_store_finds_each_sound_by_its_name() {
        let dir = std::env::temp_dir().join(format!("jodel-sfx-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = BankStore::new(&dir);
        assert_eq!(store.load(48_000), None, "none yet");
        // A bank of every sound: kept and read back.
        store.save(&full(48_000)).unwrap();
        assert_eq!(store.load(48_000), Some(full(48_000)));
        // Kept in another order (the sounds listed otherwise when it was
        // made): each still found by its name.
        let all = Sound::all();
        let reversed: Vec<(String, Vec<f32>)> = all
            .iter()
            .rev()
            .map(|s| (s.name(), vec![s.index() as f32; 4]))
            .collect();
        std::fs::write(store.file(48_000), bytes_of(&reversed, 48_000, SOUNDS_KEY)).unwrap();
        assert_eq!(store.load(48_000), Some(full(48_000)));
        // A sound missing, or one there twice: made again.
        let short = &reversed[1..];
        std::fs::write(store.file(48_000), bytes_of(short, 48_000, SOUNDS_KEY)).unwrap();
        assert_eq!(store.load(48_000), None);
        let twice = [&reversed[..], &reversed[..1]].concat();
        std::fs::write(store.file(48_000), bytes_of(&twice, 48_000, SOUNDS_KEY)).unwrap();
        assert_eq!(store.load(48_000), None);
        // One of other sounds (a key not this build's): made again.
        std::fs::write(store.file(48_000), bytes_of(&reversed, 48_000, !SOUNDS_KEY)).unwrap();
        assert_eq!(store.load(48_000), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
