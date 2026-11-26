//! The shared widgets every native screen is built from, in the look of
//! [`super::look`]: buttons, surfaces, the progress bar, the track (a
//! seek bar's or a slider's), the spectrum, the icons, the covers, a
//! list's rows and the tabs; a card's items that take a click and
//! segmented controls, switches and checks, key hints; and keeping what
//! the keys are on in view. A screen that needs one that is not here adds
//! it here, made generically, and never makes its own.

pub mod button;
pub mod cover;
pub mod hints;
pub mod icons;
pub mod item;
pub mod panel;
pub mod progress;
pub mod reveal;
pub mod rows;
pub mod spectrum;
pub mod switch;
pub mod tabs;
pub mod track;

pub use button::{chip, ghost, primary, round, secondary};
pub use hints::hints;
pub use icons::{icon, Icon};
pub use item::{item, segment, segmented};
pub use panel::{badge, card, key, note, panel, tag};
pub use progress::progress;
pub use reveal::Reveal;
pub use rows::list_row;
pub use spectrum::spectrum;
pub use switch::{check, switch};
pub use tabs::{tabs, Tab};
pub use track::track;
