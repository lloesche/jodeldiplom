//! How strong each effect is, frame by frame: one clock for every held
//! effect, the singing's glow as it eases ([`crate::gui::rainbow::Glow`]):
//! an effect a row holds while at its tier is held at that row's strength
//! at the row's glow, the straight line between rows as the glow eases
//! ([`crate::tiers::held_through`]), none below the first row that has it.
//! A row's phrase fires its `on_phrase` effects once (at their strength);
//! reaching a rung of the multiplier fires that rung's row's `on_rung` and
//! holds its `at_rung` (eased in and out as the glow is). The music moves
//! what a row holds by its [`super::Mod`]s ([`Music::get`]): its strength
//! and its look ([`Looks`], which the shaders read every frame).

use super::music::Music;
use super::{strength, At, Effects, Fx, Looks, Out, Param, Slot, Use};
use crate::gui::particles::Spray;
use crate::gui::rainbow::Glow;
use crate::gui::stage_gpu::Instance;
use crate::gui::streak::Step;
use crate::tiers::{held_through, TIERS};

/// Every effect's strength in a frame, and what goes with it: the
/// singing's clock and glow, each effect's strength now (eased, or as it
/// fired), the strength it is going to (its aim), the strength its rows
/// list where the glow comes from or goes to, the more (what it is eased
/// within), what fired this frame (and how many tiers the phrase rose),
/// the looks, and the music.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// The singing's clock.
    pub clock: f64,
    /// The singing's glow.
    pub glow: Glow,
    now: [f32; Fx::COUNT],
    aim: [f32; Fx::COUNT],
    listed: [f32; Fx::COUNT],
    fired: [bool; Fx::COUNT],
    rose: u32,
    looks: Looks,
    /// The music.
    pub music: Music,
}

/// An effect in a frame, as it draws: the singing's clock, its strength
/// now (as it fired, if it did), the strength it is going to, the strength
/// its rows list for the glow's change, whether it fired this frame, the
/// tiers the phrase rose, the singing's glow and the music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Live {
    /// The singing's clock.
    pub clock: f64,
    /// Its strength now.
    pub strength: f32,
    /// The strength it is going to.
    pub aim: f32,
    /// The strength listed where the glow comes from or goes to, the more:
    /// how strong an effect that eases by the glow itself (the rainbow)
    /// shows.
    pub listed: f32,
    /// Whether it fired.
    pub fired: bool,
    /// The tiers the phrase rose.
    pub rose: u32,
    /// The singing's glow.
    pub glow: Glow,
    /// The music.
    pub music: Music,
}

impl Frame {
    /// `fx` in it.
    pub fn live(&self, fx: Fx) -> Live {
        let i = fx.index();
        Live {
            clock: self.clock,
            strength: self.now[i],
            aim: self.aim[i],
            listed: self.listed[i],
            fired: self.fired[i],
            rose: self.rose,
            glow: self.glow,
            music: self.music,
        }
    }

    /// The effects' looks, as the music moved them.
    pub fn looks(&self) -> Looks {
        self.looks
    }
}

/// What the effects keep between frames: each its own [`Slot`], the
/// strength each is held at by the rung (eased), and the rung.
#[derive(Clone, Debug, Default)]
pub struct Board {
    slots: [Slot; Fx::COUNT],
    rungs: [Glow; Fx::COUNT],
    rung: u32,
}

impl Board {
    /// The frame of the singing's `step` by the tiers' rows, the `music`
    /// at `clock` (the singing's time).
    pub fn frame(&mut self, step: &Step, music: &Music, clock: f64) -> Frame {
        self.frame_of(TIERS.iter().map(|t| (t.glow, &t.fx)), step, music, clock)
    }

    /// The frame of `step` by `rows` (each a glow and its effects, Good
    /// first), the `music` at `clock`. Sought back before what an effect
    /// keeps, it is as long ago as can be; the rung changed, its effects
    /// ease to the new rung's (a rung reached or lost) or snap there (no
    /// moment with it: a seek).
    pub fn frame_of<'a>(
        &mut self,
        rows: impl Iterator<Item = (f32, &'a Effects)> + Clone,
        step: &Step,
        music: &Music,
        clock: f64,
    ) -> Frame {
        let t = clock as f32;
        for s in &mut self.slots {
            if t < s.last {
                s.last = f32::NEG_INFINITY;
            }
        }
        let row = |i: usize| rows.clone().nth(i).map(|(_, e)| e);
        let rung_row = |rung: u32| (rung as usize).checked_sub(1).and_then(row);
        if step.at_rung != self.rung {
            let to = rung_row(step.at_rung);
            for fx in Fx::ALL {
                let g = &mut self.rungs[fx.index()];
                let level = to.map_or(0.0, |r| strength(r.at_rung, fx));
                *g = match step.rung {
                    Some(_) => Glow {
                        level,
                        from: g.at(t).0,
                        since: t,
                    },
                    None => Glow {
                        level,
                        from: level,
                        since: t - 10.0,
                    },
                };
            }
            self.rung = step.at_rung;
        }
        let k = step.glow.at(t).0;
        let mut frame = Frame {
            clock,
            glow: step.glow,
            now: [0.0; Fx::COUNT],
            aim: [0.0; Fx::COUNT],
            listed: [0.0; Fx::COUNT],
            fired: [false; Fx::COUNT],
            rose: 0,
            looks: Looks::SPECS,
            music: *music,
        };
        let peak = step.glow.level.max(step.glow.from);
        for fx in Fx::ALL {
            let i = fx.index();
            let held = |glow: f32| {
                held_through(glow, rows.clone().map(|(g, e)| (g, strength(e.held, fx))))
            };
            let rung = &self.rungs[i];
            frame.now[i] = held(k).max(rung.at(t).0);
            frame.aim[i] = held(step.glow.level).max(rung.level);
            frame.listed[i] = held(peak).max(rung.level.max(rung.from));
        }
        let mut fired: Vec<&Use> = Vec::new();
        if let Some(r) = step
            .phrase
            .and_then(|p| p.tier)
            .and_then(|t| row(t.index()))
        {
            fired.extend(r.on_phrase);
            frame.rose = step.ups;
        }
        let up = step.rung.filter(|(a, b)| b > a).map(|(_, b)| b);
        if let Some(r) = up.and_then(rung_row) {
            fired.extend(r.on_rung);
        }
        for u in &fired {
            let i = u.fx.index();
            frame.fired[i] = true;
            frame.now[i] = frame.now[i].max(u.strength);
        }
        let tier_row = step.tier.and_then(|t| row(t.index()));
        let held = tier_row.map(|r| r.held).into_iter();
        for u in held.chain(rung_row(self.rung).map(|r| r.at_rung)).flatten() {
            modulate(&mut frame, u, false);
        }
        for u in fired {
            modulate(&mut frame, u, true);
        }
        frame
    }

    /// The effects' shapes at `at` (a slot of shapes) in `frame`, into
    /// `shapes`.
    pub fn shapes(&mut self, at: &At, frame: &Frame, shapes: &mut Vec<Instance>) {
        let mut sprays = Vec::new();
        self.draw(
            at,
            frame,
            &mut Out {
                shapes,
                sprays: &mut sprays,
            },
        );
        debug_assert!(sprays.is_empty(), "{:?} takes no bursts", at.layer());
    }

    /// The effects' bursts at `at` (a slot of bursts) in `frame`, into
    /// `sprays`.
    pub fn sprays(&mut self, at: &At, frame: &Frame, sprays: &mut Vec<Spray>) {
        let mut shapes = Vec::new();
        self.draw(
            at,
            frame,
            &mut Out {
                shapes: &mut shapes,
                sprays,
            },
        );
        debug_assert!(shapes.is_empty(), "{:?} takes no shapes", at.layer());
    }

    /// Asks the effects drawing at `at`'s slot to draw there (in their
    /// order: [`Fx::ALL`]), those with any strength or aim or that fired
    /// in `frame`, into `out`.
    fn draw(&mut self, at: &At, frame: &Frame, out: &mut Out) {
        let layer = at.layer();
        for fx in Fx::ALL {
            let spec = fx.spec();
            if !spec.at.contains(&layer) {
                continue;
            }
            let live = frame.live(fx);
            if live.strength == 0.0 && live.aim == 0.0 && !live.fired {
                continue;
            }
            (spec.draw)(at, &live, &mut self.slots[fx.index()], out);
        }
    }
}

/// `u`'s mods applied to `frame`: a held effect's strength by as much
/// again as each mod says (never below none), its look's numbers moved by
/// as much. A fired effect fires as its row lists it: its strength is not
/// moved.
fn modulate(frame: &mut Frame, u: &Use, fired: bool) {
    let i = u.fx.index();
    for m in u.mods {
        let by = m.amount * frame.music.get(m.by);
        match m.param {
            Param::Strength => {
                debug_assert!(!fired, "{:?} fired with its strength moved", u.fx);
                if !fired {
                    frame.now[i] = (frame.now[i] * (1.0 + by)).max(0.0);
                }
            }
            Param::Look(n) => frame.looks.0[i][usize::from(n.min(3))] += by,
        }
    }
}

#[cfg(test)]
mod tests;
