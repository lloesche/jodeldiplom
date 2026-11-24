//! The karaoke game's sound effects: our own sounds, made in code
//! ([`sounds`], from [`synth`]), played through the desk's effects input
//! on its Effects strip ([`jodel_output::mixer::Effects`]), each on the
//! frame of the song where its moment is seen ([`moments`]).
//!
//! A moment's time is the singing's own (the song as heard less how late
//! the voice comes: [`crate::core::lag`]), so it is seen when the song as
//! heard is that much later: the sound is cued on that frame of the
//! playing head's output ([`crate::core::Playing::frame_at`]), and the desk
//! starts it on that very frame. One with no frame to time it by yet (no
//! position told since a seek or a jump) waits here until there is, or it
//! is too late. A moment for now (the streak, a phrase's tier) starts on
//! the next period mixed. The stream going back (a seek, the song sung
//! again) drops what is cued and not heard yet; so does the song's anchor
//! forgotten (a jump takes back frames and gives their numbers to other
//! music). The moments are those of the song the singing is of, even as
//! the next track comes (its last phrase's tier is decided then).
//!
//! The bank is made for the device's rate on a thread of its own, and
//! kept for when the device comes back to that rate ([`banks`]); until it
//! is there, nothing sounds. With the Effects strip down (−∞ or muted) no
//! moment is cued.

pub mod banks;
pub mod moments;
pub mod sounds;
pub mod store;
pub mod synth;

use jodel_output::mixer::{EffectCue, Effects, Fader, SoundBank};

pub use moments::{Cued, Frame, Moments};
pub use sounds::Sound;

use crate::core::desk::StripKind;
use crate::core::State;

/// Every sound, made at `rate`, in [`Sound::index`]'s order.
pub fn bank(rate: u32) -> SoundBank {
    let mut bank = SoundBank::new(rate);
    for sound in Sound::all() {
        bank.add(sound.render(rate).interleaved());
    }
    bank
}

/// The desk as the effects play on it.
pub struct Desk<'a> {
    /// Its effects input.
    pub effects: &'a Effects,
    /// The engine's heads' input faders, by head id: the frames a moment
    /// is cued on are its head's.
    pub heads: [&'a Fader; 2],
    /// The device's rate (0: closed).
    pub rate: u32,
    /// The data folder, where the banks made are kept (`None`: in memory
    /// only).
    pub data: Option<&'a std::path::Path>,
}

/// The most cues with a time kept waiting for a position of the song to
/// time them by.
const PENDING: usize = 64;

/// The game's sounds followed frame by frame.
#[derive(Default)]
pub struct Sfx {
    moments: Moments,
    /// The banks made, by rate.
    banks: banks::Banks,
    /// The rate of the bank sent to the desk.
    loaded: Option<u32>,
    /// The song's anchors forgotten as last seen
    /// ([`crate::core::Playing::anchors_forgotten`]).
    anchors: u64,
    /// Cues with a time (the singing's) and no frame of the song's to time
    /// them by yet (no position told since a seek or a jump).
    pending: Vec<(EffectCue, f64)>,
}

impl Sfx {
    /// A frame drawn: the bank for the device's rate, then the game's new
    /// moments cued on `desk`.
    pub fn frame(&mut self, state: &State, desk: Desk<'_>) {
        self.load(&desk);
        let karaoke = &state.karaoke;
        // The song the singing is of, whatever plays now: its last
        // phrase's tier is decided as the next track comes.
        let sung = karaoke.singing.song().is_some() && !karaoke.singing.notes.lines.is_empty();
        if !(sung && crate::core::karaoke::in_mode(state) && state.config.karaoke_game) {
            self.moments.reset();
            self.pending.clear();
            return;
        }
        // A seek or a jump: the frames cued on are of another place now.
        let anchors = state.playing.anchors_forgotten;
        if anchors != self.anchors {
            self.anchors = anchors;
            desk.effects.cancel();
        }
        let clock = karaoke.lag.sung_clock(state.heard());
        let frame = self.moments.frame(&karaoke.singing, clock);
        if frame.sought {
            desk.effects.cancel();
            self.pending.clear();
        }
        if state.desk.setting(StripKind::Effects).gain() == 0.0 {
            self.pending.clear();
            return;
        }
        let cues = frame.cues.into_iter().map(|c| {
            let cue = EffectCue {
                sound: c.sound.index(),
                gain: c.gain,
                pan: c.pan,
            };
            (cue, c.at)
        });
        let waited = std::mem::take(&mut self.pending);
        let waited = waited.into_iter().map(|(cue, at)| (cue, Some(at)));
        for (cue, at) in waited.chain(cues) {
            let Some(at) = at else {
                desk.effects.play(cue);
                continue;
            };
            if at < clock - moments::STALE {
                continue;
            }
            // Heard that much later than the singing's time.
            let frame = state.playing.frame_at(at + karaoke.lag.late);
            match frame.and_then(|f| Some((*desk.heads.get(f.head)?, f.frame))) {
                Some((fader, frame)) => {
                    desk.effects.play_at(cue, fader, frame);
                }
                None if self.pending.len() < PENDING => self.pending.push((cue, at)),
                None => {}
            }
        }
    }

    /// The bank at the device's rate, sent to the desk once it is made
    /// (kept from before, or made now on a thread of its own: until then
    /// the desk has none at its rate, and the effects are silent).
    fn load(&mut self, desk: &Desk<'_>) {
        let rate = desk.rate;
        if rate == 0 || self.loaded == Some(rate) {
            return;
        }
        let data = desk.data.map(|d| crate::dirs::Data(d.to_path_buf()));
        let store = data.map(|d| store::BankStore::new(d.sounds()));
        self.banks.set_store(store);
        if let Some(bank) = self.banks.at(rate) {
            desk.effects.load(bank);
            self.loaded = Some(rate);
        }
    }
}

#[cfg(test)]
mod tests;
