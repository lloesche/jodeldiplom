//! The shapes the stage's shader draws beside its pills and text, each with
//! its twin on the processor that the shader is held to
//! ([`crate::gui::shader_check`]): the word bubbles ([`pop`]: a tube
//! through a word's notes, [`snake`], filled with its liquid, [`liquid`]),
//! the score meter and the shine's other looks ([`glory`]), and the
//! result diploma's paper ([`paper`]) with what is stuck on and pressed
//! into it ([`seal`]).

pub mod glory;
pub mod liquid;
pub mod paper;
pub mod pop;
pub mod seal;
pub mod snake;
