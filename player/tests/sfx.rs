//! The song's frames a time of it is heard on ([`Playing::frame_at`]): what
//! the karaoke game's sounds are cued on (a moment at a time of the
//! singing, plus how late the voice comes) and a stem's change on a bar,
//! through the last position told and the output's speed through the
//! track.

mod common;

use jodeldiplom::core::{Anchor, HeadFrame, Playing};

#[test]
fn a_time_lands_on_the_frame_it_is_heard_on() {
    let mut state = common::playing_state(120, 36);
    // None before a position is told.
    assert_eq!(state.playing.frame_at(84.0), None);
    let index = state.playing.track.as_ref().unwrap().index;
    state.playing.anchor = Some(Anchor {
        index,
        at: HeadFrame {
            head: 1,
            frame: 1_000_000,
        },
        seconds: 83.0,
    });
    // Nor with one: the output's speed through the track is not known
    // before two.
    assert_eq!(state.playing.frame_at(84.0), None);
    // At its speed through the track once it is (a varispeed).
    state.playing.speed = Some(44_100.0 * 1.25);
    let at = |frame| Some(HeadFrame { head: 1, frame });
    assert_eq!(state.playing.frame_at(84.0), at(1_055_125));
    // Before the output's first frame: none.
    assert_eq!(state.playing.frame_at(0.0), None);
    // A position of another track: none.
    state.playing.anchor.as_mut().unwrap().index = index + 1;
    assert_eq!(state.playing.frame_at(84.0), None);
    // Forgotten (a seek, a jump): none, and counted, so what was worked
    // out from it is known to be off.
    let before = state.playing.anchors_forgotten;
    state.playing.forget_anchor();
    assert_eq!(state.playing.frame_at(84.0), None);
    assert_eq!(state.playing.anchors_forgotten, before + 1);
    assert_eq!(Playing::default().anchors_forgotten, 0);
}
