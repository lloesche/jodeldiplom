//! A player on a mixer's desk as the app has it: a group strip for the
//! player, a strip inside it for each of its stems (in [`LANES`]' order),
//! and a lane input of them ([`Mixer::input_lanes`]) for the engine to play
//! on: its main on the group, each stem's part on its strip
//! ([`crate::MakeStems`]). With every strip as it starts the mixer plays
//! main, bit for bit.

use jodel_output::mixer::{Fader, InputTarget, Mixer, Role, Strip};

use crate::stems::{Stem, LANES};

/// A player's strips and the faders of its lane input ([`StemDesk::new`]).
pub struct StemDesk {
    /// The player's strip: a group, its main heard on it.
    pub group: Strip,
    /// A strip inside the group for each stem, in [`LANES`]' order.
    pub stems: Vec<Strip>,
    /// The input's fader over every lane (a controller's duck).
    pub fader: Fader,
    /// Each stem's lane's own fader, in [`LANES`]' order.
    pub lanes: Vec<Fader>,
}

impl StemDesk {
    /// The strips on `mixer`, the group named `name`, each stem's strip by
    /// the stem's name; and the engine's target, a lane input of main and
    /// the four stems in `role` at `gain`.
    pub fn new(mixer: &Mixer, name: &str, role: Role, gain: f64) -> (StemDesk, InputTarget) {
        let group = mixer.strip(name);
        let stems: Vec<Strip> = LANES
            .iter()
            .map(|s| mixer.strip_in(&group, s.id()))
            .collect();
        let refs: Vec<&Strip> = stems.iter().collect();
        let (input, fader, lanes) = mixer.input_lanes(&group, &refs, role, gain);
        let desk = StemDesk {
            group,
            stems,
            fader,
            lanes,
        };
        (desk, input)
    }

    /// The strip of `stem`.
    pub fn stem(&self, stem: Stem) -> &Strip {
        let k = LANES.iter().position(|&s| s == stem).unwrap_or(0);
        &self.stems[k]
    }
}
