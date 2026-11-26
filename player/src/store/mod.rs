//! What the player keeps on disk between runs, every file through the
//! `durable` crate.
//!
//! - Where: the two folders, the settings and the data ([`dirs`]), and a
//!   move of the data folder that no interruption loses ([`mover`]).
//! - What each kind of file holds, and the version it is read in
//!   ([`cache`]: the schemas).
//! - The user's own, beside the settings file or in the data folder: the
//!   session and the queues saved ([`session`]), the beat grids fixed by
//!   hand ([`grids`]), the singer's delays ([`delays`]), the best scores
//!   ([`scores`]), the singing kept ([`keep`]) and the game's sounds'
//!   patches ([`patches`]).
//! - What could be made again: the stems kept ([`stems`]). The workers
//!   keep the rest of it ([`crate::work`]).

pub mod cache;
pub mod delays;
pub mod dirs;
pub mod grids;
pub mod keep;
pub mod mover;
pub mod patches;
pub mod scores;
pub mod session;
pub mod stems;
