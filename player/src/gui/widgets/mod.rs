//! The shared widgets every native screen is built from, in the look of
//! [`super::look`]: buttons, surfaces, the progress bar, the track (a
//! seek bar's or a slider's), the spectrum, the icons, the covers, a
//! list's rows and the tabs. A screen that needs one that is not here
//! adds it here, made generically, and never makes its own.

pub mod button;
pub mod cover;
pub mod icons;
pub mod panel;
pub mod progress;
pub mod rows;
pub mod spectrum;
pub mod tabs;
pub mod track;

pub use button::{chip, ghost, primary, round, secondary};
pub use icons::{icon, Icon};
pub use panel::{badge, card, key, note, panel, tag};
pub use progress::progress;
pub use rows::list_row;
pub use spectrum::spectrum;
pub use tabs::{tabs, Tab};
pub use track::track;
