//! The banks of sounds made, by rate ([`Banks`]): the device's rate
//! changes with the songs picked, and a bank made once is kept for the
//! next time, the last [`KEPT`] rates of them, in memory (a bank is
//! shared with the desk, so taking one kept costs nothing), and on disk
//! ([`BankStore`], when there is a data folder) for the rates had before,
//! across runs. One not kept in memory is read from disk or made (and
//! written there) on a thread of its own; until it is there, the desk has
//! none at its rate and the effects are silent.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use jodel_output::mixer::SoundBank;

use super::store::BankStore;

/// The rates whose banks are kept.
pub const KEPT: usize = 3;

/// The banks made, and one being made.
pub struct Banks {
    /// The newest used last.
    kept: VecDeque<(u32, Arc<SoundBank>)>,
    making: Option<(u32, Receiver<SoundBank>)>,
    /// How a bank is made.
    make: fn(u32) -> SoundBank,
    /// Where banks are kept on disk.
    store: Option<BankStore>,
}

impl Default for Banks {
    fn default() -> Self {
        Banks::new(super::bank)
    }
}

impl Banks {
    /// Banks made by `make`.
    pub fn new(make: fn(u32) -> SoundBank) -> Banks {
        Banks {
            kept: VecDeque::new(),
            making: None,
            make,
            store: None,
        }
    }

    /// Banks kept on disk in `store` from now on (`None`: none).
    pub fn set_store(&mut self, store: Option<BankStore>) {
        self.store = store;
    }

    /// The bank at `rate`, if it is made: kept, or just made (then kept,
    /// the oldest rate's let go of beyond [`KEPT`]). Otherwise it is set
    /// making (a bank being made at another rate is left to finish
    /// unheeded).
    pub fn at(&mut self, rate: u32) -> Option<Arc<SoundBank>> {
        if let Some((made, rx)) = &self.making {
            match rx.try_recv() {
                Ok(bank) => {
                    let made = *made;
                    self.making = None;
                    self.keep(made, Arc::new(bank));
                }
                Err(TryRecvError::Disconnected) => self.making = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some(k) = self.kept.iter().position(|(r, _)| *r == rate) {
            let entry = self.kept.remove(k).expect("found");
            let bank = entry.1.clone();
            self.kept.push_back(entry);
            return Some(bank);
        }
        if self.making.as_ref().is_none_or(|(r, _)| *r != rate) {
            let (tx, rx) = mpsc::channel();
            let (make, store) = (self.make, self.store.clone());
            let started = std::thread::Builder::new()
                .name("jodel-sfx-bank".into())
                .spawn(move || {
                    let _ = tx.send(load_or_make(rate, make, store.as_ref()));
                });
            if started.is_ok() {
                self.making = Some((rate, rx));
            }
        }
        None
    }

    /// The rates kept, the oldest used first.
    pub fn rates(&self) -> Vec<u32> {
        self.kept.iter().map(|(r, _)| *r).collect()
    }

    fn keep(&mut self, rate: u32, bank: Arc<SoundBank>) {
        self.kept.retain(|(r, _)| *r != rate);
        self.kept.push_back((rate, bank));
        while self.kept.len() > KEPT {
            self.kept.pop_front();
        }
    }
}

/// The bank at `rate`: as kept in `store`, else made by `make` (and kept
/// there; one that cannot be written is used all the same).
pub fn load_or_make(rate: u32, make: fn(u32) -> SoundBank, store: Option<&BankStore>) -> SoundBank {
    if let Some(bank) = store.and_then(|s| s.load(rate)) {
        return bank;
    }
    let bank = make(rate);
    if let Some(store) = store {
        let _ = store.save(&bank);
    }
    bank
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    static MADE: AtomicUsize = AtomicUsize::new(0);

    fn tiny(rate: u32) -> SoundBank {
        MADE.fetch_add(1, Ordering::SeqCst);
        let mut b = SoundBank::new(rate);
        b.add(vec![0.0; 2]);
        b
    }

    /// Asks for `rate` until it is there.
    fn wait(banks: &mut Banks, rate: u32) -> Arc<SoundBank> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(b) = banks.at(rate) {
                return b;
            }
            assert!(Instant::now() < deadline, "no bank at {rate}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_bank_is_made_once_and_the_last_three_rates_are_kept() {
        let mut banks = Banks::new(tiny);
        let first = wait(&mut banks, 44_100);
        assert_eq!(first.rate(), 44_100);
        let made = MADE.load(Ordering::SeqCst);
        // Kept: at once, the same bank, nothing made.
        let again = banks.at(44_100).expect("kept");
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(MADE.load(Ordering::SeqCst), made);
        for rate in [48_000, 96_000] {
            wait(&mut banks, rate);
        }
        assert_eq!(banks.rates(), [44_100, 48_000, 96_000]);
        // 44.1 kHz used again: the newest; a fourth rate lets the oldest
        // used (48 kHz) go.
        assert!(banks.at(44_100).is_some());
        wait(&mut banks, 192_000);
        assert_eq!(banks.rates(), [96_000, 44_100, 192_000]);
        assert_eq!(MADE.load(Ordering::SeqCst), made + 3);
    }

    static MADE_FULL: AtomicUsize = AtomicUsize::new(0);

    /// A bank of every sound, each a frame.
    fn full(rate: u32) -> SoundBank {
        MADE_FULL.fetch_add(1, Ordering::SeqCst);
        let mut b = SoundBank::new(rate);
        for k in 0..crate::sfx::Sound::COUNT {
            b.add(vec![k as f32 / 64.0, -(k as f32) / 64.0]);
        }
        b
    }

    #[test]
    fn a_bank_made_once_is_read_from_disk_after() {
        let dir = std::env::temp_dir().join(format!("jodel-sfx-banks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = BankStore::new(&dir);
        let made = load_or_make(48_000, full, Some(&store));
        assert_eq!(MADE_FULL.load(Ordering::SeqCst), 1);
        assert!(store.file(48_000).is_file());
        // Another run: read, not made.
        let mut banks = Banks::new(full);
        banks.set_store(Some(store.clone()));
        let read = wait(&mut banks, 48_000);
        assert_eq!(*read, made);
        assert_eq!(MADE_FULL.load(Ordering::SeqCst), 1);
        // A file that is not a bank: made again, and written over.
        std::fs::write(store.file(48_000), b"JDLS").unwrap();
        assert_eq!(load_or_make(48_000, full, Some(&store)), made);
        assert_eq!(MADE_FULL.load(Ordering::SeqCst), 2);
        assert_eq!(store.load(48_000), Some(made));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
