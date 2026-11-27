//! The picture tools the taking apart of a cover ([`super::cover`]) and of
//! a story's scenes ([`super::scenes`]) share: pictures as float planes
//! (blurred, grown, resized, a hole filled, a plate made: [`fill`]), masks
//! on them ([`mask`]), depth as nearness ([`NearMap`], [`nearness`]), a
//! picture's main colours ([`palette`]), and how they are kept ([`codec`]).

pub mod codec;
pub mod fill;
pub mod mask;
mod near;
mod palette;

pub use fill::{box_blur, dilate, fill, hole, plate, push_pull, resize, rgba, Planes};
pub use near::{nearness, NearMap};
pub use palette::{palette, Swatch};
