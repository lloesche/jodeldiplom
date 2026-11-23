//! A player on a mixer's desk as the app has it: a group strip for the
//! player, a strip inside it for each of its stems (in [`LANES`]' order)
//! ([`StemDesk`]), and a head on them for the engine to play on
//! ([`StemDesk::head`]), a lane input ([`Mixer::input_lanes`]): its main
//! on the group, each stem's part on its strip ([`crate::MakeStems`]).
//! With every strip as it starts the mixer plays main, bit for bit. A
//! player with two heads ([`StemDesk::two_heads`]) has a second such input
//! on the same strips, for a transition's next track: the user's stem
//! levels apply to both.

use jodel_output::mixer::{Mixer, Role, Strip};

use crate::engine::{Head, HeadFaders};
use crate::stems::{Stem, LANES};

/// A player's strips: its group, and a strip in it for each stem.
pub struct StemDesk {
    /// The player's strip: a group, its main heard on it.
    pub group: Strip,
    /// A strip inside the group for each stem, in [`LANES`]' order.
    pub stems: Vec<Strip>,
}

impl StemDesk {
    /// The strips on `mixer`, the group named `name`, each stem's strip by
    /// the stem's name; and a head on them in `role` at `gain`, for a
    /// player with one ([`crate::Player::new`] on its target).
    pub fn new(mixer: &Mixer, name: &str, role: Role, gain: f64) -> (StemDesk, Head) {
        let desk = StemDesk::strips(mixer, name);
        let head = desk.head(mixer, role, gain);
        (desk, head)
    }

    /// The same strips, and two heads on them for one player
    /// ([`crate::Player::on_desk`]), each a lane input at `gain`: the first
    /// the desk's Lead, the second a guest (a transition's next track; the
    /// two swap roles at each hand-over, their faders stay each input's).
    pub fn two_heads(mixer: &Mixer, name: &str, gain: f64) -> (StemDesk, Head, Head) {
        let desk = StemDesk::strips(mixer, name);
        let lead = desk.head(mixer, Role::Lead, gain);
        let guest = desk.head(mixer, Role::Guest, gain);
        (desk, lead, guest)
    }

    /// A head on these strips: a lane input of main and the four stems in
    /// `role` at `gain`.
    pub fn head(&self, mixer: &Mixer, role: Role, gain: f64) -> Head {
        let refs: Vec<&Strip> = self.stems.iter().collect();
        let (target, input, lanes) = mixer.input_lanes(&self.group, &refs, role, gain);
        Head {
            target: Box::new(target),
            faders: HeadFaders { input, lanes },
        }
    }

    /// The group named `name` on `mixer` and a strip in it for each stem.
    fn strips(mixer: &Mixer, name: &str) -> StemDesk {
        let group = mixer.strip(name);
        let stems = LANES
            .iter()
            .map(|s| mixer.strip_in(&group, s.id()))
            .collect();
        StemDesk { group, stems }
    }

    /// The strip of `stem`.
    pub fn stem(&self, stem: Stem) -> &Strip {
        let k = LANES.iter().position(|&s| s == stem).unwrap_or(0);
        &self.stems[k]
    }
}
