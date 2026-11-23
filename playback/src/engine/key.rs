//! The key shift's policy ([`crate::Command::Key`]: what it does is told
//! there): which key is asked for, when the playing head takes it, and
//! what a pick does to it. The moves are [`super::along`]'s: a segment
//! nothing of which is written yet starts on its map in the key
//! ([`Engine::start_along`]); one being heard gives back what the output
//! has not played of it ([`Engine::take_back`]) and joins its map there
//! ([`Engine::join_along`]). The pass itself is [`super::shift`]'s.

use std::sync::Arc;

use crate::error::Error;
use crate::transition::TempoMode;
use crate::warp::TimeMap;

use super::api::Event;
use super::shift::KEY_RANGE;
use super::track::{AlongMap, HeardFrom, HeardWarp, SegStart, Track};
use super::Engine;

/// Said when new settings wait for the key.
pub(super) const SETTINGS_WAIT: &str =
    "The new settings apply once the song is back in its own key or at the next song";

impl Engine {
    /// [`crate::Command::Key`]: the key asked for, on the head on air now
    /// (unless it cannot take it: a notice says why, and the key stays).
    /// With nothing playing it is for the track picked next.
    pub(super) fn set_key(&mut self, semitones: i8) {
        let key = semitones.clamp(-KEY_RANGE, KEY_RANGE);
        // The same key again is said again (and asked again of a head that
        // plays another: refused before, it may take it now).
        let shifted = self.head.playing.as_ref().map(Track::key);
        if key == self.key && shifted.is_none_or(|k| k == key) {
            return self.emit(Event::Key { semitones: key });
        }
        let index = self.head.playing.as_ref().map(|t| t.index);
        let why = if self.transition.is_some() {
            Some("a transition is set up".to_owned())
        } else {
            self.head.playing.as_ref().and_then(|t| self.map_refusal(t))
        };
        if let Some(why) = why {
            self.notice(index, format!("The key stays as it is: {why}"));
            return self.emit(Event::Key {
                semitones: self.key,
            });
        }
        self.key = key;
        self.key_song = self.head.playing.as_ref().map(|t| t.info.path.clone());
        self.emit(Event::Key { semitones: key });
        if let Err(error) = self.key_now() {
            self.fail(index, error);
        }
    }

    /// Track `index` is picked: the key goes back to 0 unless it is the
    /// track the key is on (or the key is for whatever is picked next).
    pub(super) fn key_picked(&mut self, index: usize) {
        let other = self
            .key_song
            .as_ref()
            .is_some_and(|song| self.queue.get(index) != Some(song));
        if self.key != 0 && other {
            self.key = 0;
            self.emit(Event::Key { semitones: 0 });
        }
    }

    /// A track was installed or run into: it is the one the key is on, and
    /// takes it from its first frame (unless it cannot: a notice says why).
    pub(super) fn track_installed(&mut self) {
        let Some(track) = self.head.playing.as_ref() else {
            return;
        };
        self.key_song = Some(track.info.path.clone());
        if self.key == 0 {
            return;
        }
        if let Some(why) = self.map_refusal(track) {
            let message = format!("This track plays in its own key: {why}");
            return self.notice(Some(track.index), message);
        }
        if let Err(error) = self.key_now() {
            self.fail(None, error);
        }
    }

    /// The key asked for, on the playing head as soon as the output can:
    /// a segment nothing of which is written takes it from its first frame;
    /// else the output gives back what it has not played of the segment and
    /// the head joins its map in the key there.
    pub(super) fn key_now(&mut self) -> Result<(), Error> {
        let key = self.key;
        if self.set_pending_key(key) {
            return Ok(());
        }
        let (Some(track), Some(out)) = (self.head.playing.as_ref(), self.head.out.as_ref()) else {
            return Ok(());
        };
        if track.key() == key {
            return Ok(());
        }
        let (written, seg_out, index) = (out.output.written(), track.seg_out, track.index);
        let (plain, begun) = (track.plain_route(), track.segment.begun);
        let clock = out.clock.clone();
        // What else would move the head goes: a move to new settings under
        // way waits for the key's way back, a jump is dropped.
        if self.head.switch.is_some() && plain {
            self.notice(Some(index), SETTINGS_WAIT.into());
        }
        self.cancel_switch();
        self.drop_jump_fill();
        if self.head.pending_jump.take().is_some() {
            let message = "The jump waiting for its time is dropped: the key changed";
            self.notice(Some(index), message.into());
        }
        // Nothing of the segment written yet (a count-in before it, or the
        // track just installed): in the key from its first frame.
        if written < seg_out || (written == seg_out && begun == SegStart::Cold) {
            let heard_from = HeardFrom {
                clock,
                origin: None,
                at: seg_out,
            };
            let Some(mut track) = self.head.playing.take() else {
                return Ok(());
            };
            let map = self.key_map(&track, seg_out);
            self.start_along(&mut track, map, seg_out, heard_from);
            self.requeue_heard(&track, seg_out);
            self.head.playing = Some(track);
            return Ok(());
        }
        if !self.offline {
            self.take_back(written.saturating_sub(seg_out))?;
        }
        let (Some(mut track), Some(out)) = (self.head.playing.take(), self.head.out.as_ref())
        else {
            return Ok(());
        };
        let at = out.output.written();
        let map = self.key_map(&track, at);
        self.head.lanes.prepare_along(&track, &map.heard, at);
        let result = self.join_along(&mut track, map);
        self.head.playing = Some(track);
        result
    }

    /// Whether the head on air plays in a key not its own.
    pub(super) fn head_shifted(&self) -> bool {
        self.head.playing.as_ref().is_some_and(|t| t.key() != 0)
    }

    /// The map `track` takes the key on: its own, or (on its own route)
    /// the identity from output frame `at`, where it is then.
    fn key_map(&self, track: &Track, at: u64) -> AlongMap {
        if let Some(along) = &track.along {
            return AlongMap {
                key: self.key,
                ..along.map.clone()
            };
        }
        let heard = HeardWarp {
            map: Arc::new(TimeMap::new(&[], 0.0)),
            origin: at,
            a0: track.heard().seconds_at(at),
            out_rate: track.route.output_rate,
        };
        AlongMap {
            heard: Arc::new(heard),
            tempo: TempoMode::Varispeed,
            key: self.key,
        }
    }
}
