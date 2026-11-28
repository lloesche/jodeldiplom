//! The microphones' list: opened from the Karaoke tab's microphone or the
//! Mixer tab's MIC strip while it listens, the input devices listed afresh
//! each time it opens (in the background, [`Effect::ListMics`]); one
//! chosen from it is taken as `i` takes the next one, on the same path
//! ([`Effect::ChooseMic`]): kept in the settings, opened in the
//! background, one that does not open said as it is.

use crate::core::{Effect, Side, State};

/// A microphone (input device) to choose from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicChoice {
    /// Its id, as the settings keep it.
    pub id: String,
    /// Its name.
    pub name: String,
}

/// Which microphone [`Effect::ChooseMic`] takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MicPick {
    /// The one after the microphone in use, going round (`i`).
    Next,
    /// This one, chosen from the list (as listed: one gone since fails to
    /// open, and says why).
    Device(MicChoice),
}

/// The microphones' list, while it is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicList {
    /// The tab it opened on (the Karaoke tab or the Mixer tab): it closes
    /// once that tab no longer shows.
    pub on: Side,
    /// The microphones, once listed (`None`: being listed).
    pub mics: Option<Vec<MicChoice>>,
}

/// Whether the microphones' list can open (or stay open): the microphone
/// shows (on the Karaoke tab, or on the Mixer tab with the MIC strip
/// listening) and is open or did not open, not still opening.
pub fn choosable(state: &State) -> bool {
    let shown = &state.karaoke.mic;
    (state.showing(Side::Karaoke) || super::desk::listens(state))
        && (shown.name.is_some() || shown.error.is_some())
}

/// [`crate::core::Action::MicList`]: the list opens on the tab showing,
/// its microphones listed afresh, or closes where it is open there.
pub(crate) fn toggle(state: &mut State, fx: &mut Vec<Effect>) {
    let on = state.side;
    if state.karaoke.mic_list.as_ref().is_some_and(|l| l.on == on) {
        state.karaoke.mic_list = None;
    } else if choosable(state) {
        state.karaoke.mic_list = Some(MicList { on, mics: None });
        fx.push(Effect::ListMics);
    }
}

/// [`crate::core::Action::Mics`]: the microphones listed, into the list
/// open (one closed meanwhile takes nothing); why they could not be, said,
/// the list empty.
pub(crate) fn listed(state: &mut State, result: Result<Vec<MicChoice>, String>) {
    let Some(list) = &mut state.karaoke.mic_list else {
        return;
    };
    match result {
        Ok(mics) => list.mics = Some(mics),
        Err(why) => {
            list.mics = Some(Vec::new());
            state.error(format!("Listing microphones failed: {why}"));
        }
    }
}

/// [`crate::core::Action::ChooseMic`]: microphone `index` of the list
/// chosen, taken as `i` takes the next one; the list closes.
pub(crate) fn choose(state: &mut State, index: usize, fx: &mut Vec<Effect>) {
    let Some(list) = state.karaoke.mic_list.take() else {
        return;
    };
    if let Some(mic) = list.mics.and_then(|m| m.into_iter().nth(index)) {
        fx.push(Effect::ChooseMic(MicPick::Device(mic)));
    }
}

/// The list closes once the microphone no longer shows where it opened,
/// or is opening again.
pub(crate) fn settle(state: &mut State) {
    let gone = state
        .karaoke
        .mic_list
        .as_ref()
        .is_some_and(|l| !state.showing(l.on) || !choosable(state));
    if gone {
        state.karaoke.mic_list = None;
    }
}

/// The microphone the settings name (it may not have opened), else the
/// one open (the system's default input): its id.
pub fn current(state: &State) -> Option<&str> {
    state
        .config
        .karaoke_mic_device
        .as_deref()
        .or(state.karaoke.mic.id.as_deref())
}
