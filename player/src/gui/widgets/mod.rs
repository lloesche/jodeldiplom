//! The shared widgets every native screen is built from, in the look of
//! [`super::look`]: buttons, surfaces, the progress bar, the track (a
//! seek bar's or a slider's), the spectrum, the icons, the covers, a
//! list's rows and the tabs; a card's items that take a click and
//! segmented controls, switches and checks, key hints, links, a text
//! field; keeping what the keys are on in view; an area that takes the
//! pointer over what something else draws; an envelope's curve, a
//! piano keyboard and a waveform with its beats. A screen that needs one
//! that is not here adds it here, made generically, and never makes its
//! own.

pub mod area;
pub mod button;
pub mod cover;
pub mod envelope;
pub mod field;
pub mod hide;
pub mod hints;
pub mod icons;
pub mod item;
pub mod link;
pub mod panel;
pub mod piano;
pub mod progress;
pub mod reveal;
pub mod rows;
pub mod spectrum;
pub mod switch;
pub mod tabs;
pub mod track;
pub mod wave;

pub use area::area;
pub use button::{chip, ghost, primary, round, secondary};
pub use envelope::envelope;
pub use field::field;
pub use hide::hide_pointer;
pub use hints::hints;
pub use icons::{icon, Icon};
pub use item::{item, segment, segmented};
pub use link::link;
pub use panel::{badge, card, key, lit_tag, named_tag, note, panel, tag};
pub use piano::piano;
pub use progress::progress;
pub use reveal::Reveal;
pub use rows::list_row;
pub use spectrum::spectrum;
pub use switch::{check, switch};
pub use tabs::{tabs, Tab};
pub use track::track;
pub use wave::wave;
