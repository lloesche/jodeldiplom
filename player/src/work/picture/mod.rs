//! The picture tools a story's scenes are taken apart with
//! ([`super::scenes`]): pictures as float planes (blurred, resized, a
//! plate made behind a subject, its hole filled: [`fill`]), masks on them
//! ([`mask`]), depth as nearness ([`NearMap`], [`nearness`]), a picture's
//! main colours ([`palette`]), and how they are kept ([`codec`]).

pub mod codec;
pub mod fill;
pub mod mask;
mod near;
mod palette;

pub use fill::{box_blur, plate, resize, rgba, Planes};
pub use near::{nearness, NearMap};
pub use palette::{palette, Swatch};
