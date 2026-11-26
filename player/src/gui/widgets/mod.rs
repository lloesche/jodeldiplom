//! The shared widgets every native screen is built from, in the look of
//! [`super::look`]: buttons, surfaces, the progress bar, the icons and the
//! covers. A screen that needs one that is not here adds it here, made
//! generically, and never makes its own.

pub mod button;
pub mod cover;
pub mod icons;
pub mod panel;
pub mod progress;

pub use button::{ghost, primary, secondary};
pub use icons::{icon, Icon};
pub use panel::{badge, card, note, panel};
pub use progress::progress;
