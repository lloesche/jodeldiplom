//! What the window asks of the [`App`] for each frame: where the song is
//! heard, as a frame draws it (and what the microphone heard meanwhile),
//! the spectrum, and whether the picture moves by itself and how smoothly.

use jodel_playback::PlayState;

use super::App;
use crate::core;
use crate::spectrum::SpectrumFrame;

/// How far the position reported may be from the frames' own clock and
/// still be eased towards (further: a seek, a jump; it is taken at once).
const HELD_BACK: f64 = 0.25;

/// The share of the way the frames' clock goes to the position reported
/// each frame.
const CATCH_UP: f64 = 0.1;

impl App {
    /// [`App::heard`] for a frame about to be drawn, kept in the state so
    /// that a click on the Transition tab lands where it was seen.
    pub fn heard_for_frame(&mut self) -> f64 {
        let reported = self.heard();
        let mut heard = reported;
        let now = self.clock.now();
        // The position comes every 100 ms, a little early or late: drawn as it
        // is, what scrolls by (the Transition lanes, the Karaoke tab's words)
        // steps unevenly. While the track plays on, the frames keep a clock of
        // their own, running with time from the last frame's position and
        // going a share of the way to the one reported, never back; a seek or
        // a jump (further off) is taken at once.
        let track = self.state.playing.track.as_ref().map(|t| t.index);
        if self.state.playing.state == PlayState::Playing {
            if let Some((was, last, at)) = self.last_drawn {
                let ahead = last + now.duration_since(at).as_secs_f64();
                let off = reported - ahead;
                if was == track && off.abs() < HELD_BACK {
                    heard = (ahead + CATCH_UP * off).max(last);
                }
            }
        }
        self.last_drawn = Some((track, heard, now));
        self.note_pair();
        // A sync under way hears the voice; the singer's delay follows.
        let sync = crate::sync::frame(&mut self.clicks, self.mic.listen(&mut self.state));
        for action in sync {
            self.dispatch(action);
        }
        let echo = self
            .state
            .mic
            .and_then(|m| m.echo())
            .map(|(_, learnt, delay)| (learnt, delay.map(f64::from)));
        let now = self.now_ms() as f64 / 1000.0;
        core::lag::follow(&mut self.state, echo, now);
        self.mic.sing(&mut self.state);
        if let Some(said) = self.mic.save_kept(&self.state, false) {
            self.dispatch(said);
        }
        if let Some(said) = self.mic.echo_recorded() {
            self.dispatch(said);
        }
        if core::desk::shown(&self.state) {
            self.look_at_desk();
        }
        self.state.drawn_heard = Some(heard);
        self.state.drawn_ms = Some(self.now_ms());
        if let Some(engine) = &mut self.engine {
            engine.sound_effects(&self.state);
        }
        heard
    }

    /// Seconds into the playing track as heard now: the last position
    /// reported (every 100 ms) and the time since, while playing.
    pub fn heard(&self) -> f64 {
        let playing = &self.state.playing;
        let since = if playing.state == PlayState::Playing {
            self.clock.since(self.position_at).as_secs_f64().min(0.25)
        } else {
            0.0
        };
        playing.position + since
    }

    /// Whether the picture changes by itself (something is playing, the
    /// spectrum still falling, the microphone's meter shown), so the
    /// frontend should keep redrawing. A spectrum standing still (frozen
    /// while paused) is no reason: the window then stops drawing until
    /// something changes.
    pub fn animating(&self) -> bool {
        self.loading()
            // A preview's players come and go as their fades end.
            || self.engine.as_ref().is_some_and(|e| e.previews.is_some())
            || self.state.playing.state == PlayState::Playing
            // The Karaoke tab's meter moves with the voice, music or not.
            || (self.mic.input().is_some() && core::karaoke::on_tab(&self.state))
            // The Mixer tab's meters move (and fall) by themselves.
            || core::desk::shown(&self.state)
            || self.state.seek_drag.is_some()
            || self.state.transition.slide.is_some()
            || self.state.karaoke.resume_at.is_some()
            // The sync's dots light up with its clicks.
            || self.clicks.is_some()
            || (self.state.playing.state != PlayState::Paused && self.spectrum.moving())
    }

    /// Whether something shown is still loading, with a spinner turning:
    /// the folder listed, the microphone opening, lyrics looked up, a
    /// track in the queue shown being got ready.
    fn loading(&self) -> bool {
        let state = &self.state;
        let karaoke = core::karaoke::mic_wanted(state);
        let lyrics = core::karaoke::shown(state);
        state.browser.loading
            || (state.screen == core::Screen::Main
                && state.side == core::Side::Queue
                && !state.working.is_empty())
            || (karaoke && self.mic.opening())
            || (lyrics
                && state
                    .karaoke
                    .lyrics
                    .values()
                    .any(|l| matches!(l, core::karaoke::SongLyrics::Pending)))
    }

    /// Whether what moves needs 60 frames a second to look smooth: what
    /// follows the pointer (a seek drag, a view being dragged), a block
    /// sliding, the lyrics editor's playhead over the words it times, and
    /// words to be read as they go by (Karaoke mode; the
    /// Karaoke tab's voice lane is [`App::layer_moves`]). The spectrum, the
    /// clock and the Transition tab's lanes going by as a track plays (a few
    /// pixels a frame at 30) do with 30.
    pub fn smooth(&self) -> bool {
        let state = &self.state;
        state.seek_drag.is_some()
            || state.transition.slide.is_some()
            || state.transition.pan.is_some()
            || state.karaoke.dragging.is_some()
            || state.desk.dragging.is_some()
            // Karaoke mode: the lyrics fill and the count-in's numbers
            // shrink as they are sung.
            || (core::karaoke::in_mode(state) && state.playing.state == PlayState::Playing)
            // A paused song's count-in.
            || state.karaoke.resume_at.is_some()
            // Karaoke mode's controls fading in or out.
            || core::karaoke::controls::fading(state)
            // The lyrics editor's playhead, or a word or its view dragged.
            || core::lyric_edit::screen::moving(state)
    }

    /// Whether a lane that follows the playhead goes by (a song playing:
    /// the Karaoke tab's voice lane, the Transition tab's playing track): it
    /// needs a frame every refresh too, but only the lanes move in most of
    /// them ([`crate::gui::lanes::Lanes::frame`]).
    pub fn layer_moves(&self) -> bool {
        let state = &self.state;
        let playing = state.playing.state == PlayState::Playing;
        // The Transition tab's playing track, its view following the playhead.
        let transition = state.screen == core::Screen::Main
            && state.side == core::Side::Transition
            && state.palette.is_none()
            && core::transition::tab::block(state, core::lanes::Block::Now)
                .start
                .is_none()
            && state.transition.slide.is_none();
        playing && (core::karaoke::on_tab(state) || transition)
    }

    /// The spectrum of what is heard now ([`crate::spectrum::SpectrumGate`]): where the
    /// screen shows one.
    pub fn spectrum(&mut self) -> SpectrumFrame {
        let playing = self.state.playing.state != PlayState::Stopped;
        // A song, or a preview (with no song too): the mixer's tap has
        // what the device plays.
        let heard = (self.engine.as_ref())
            .filter(|e| e.previews.is_some() || playing)
            .map(|e| &e.tap);
        let now = self.clock.now();
        self.spectrum.frame(now, self.state.fit.spectrum, heard)
    }

    /// The Mixer tab's look for a frame: the desk and the meters since the
    /// last look (the one place they are taken), each fader strip found by
    /// its id.
    fn look_at_desk(&mut self) {
        let Some(engine) = &self.engine else {
            self.state.desk.master = None;
            return;
        };
        let look = engine.look();
        let now = self.now_ms();
        core::desk::feed(&mut self.state, look, now);
    }
}
