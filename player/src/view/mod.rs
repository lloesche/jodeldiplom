//! What the screens say, as models the window draws in iced's widgets:
//! the list pane with its tabs ([`list`]), the Transition tab
//! ([`transition`]), the Karaoke tab and Karaoke mode ([`karaoke`], the
//! sync's panel [`sync`]), the right column's text under the cover or the
//! summary under the list ([`info`]), the command palette over everything
//! ([`command_palette`]) and the bottom bar ([`bottom`], the spectrum's
//! [`plot`]); and whole screens: Now playing ([`now::NowPlaying`]), the
//! settings ([`settings`]), the keys ([`help`]) and the lyrics editor
//! ([`lyrics`]), in the player's colours ([`palette`]).

pub mod bottom;
pub mod bubbles;
pub mod command_palette;
pub mod help;
pub mod info;
pub mod karaoke;
pub mod lanes;
pub mod list;
pub mod lyrics;
pub mod now;
pub mod palette;
pub mod plot;
pub mod settings;
pub mod setup;
pub mod sync;
pub mod transition;
