//! The Synth tab ([`Side::Synth`]): a workbench for the game's sounds.
//! Every sound is listed by its moment ([`crate::sfx::bench::SOUNDS`]);
//! a click plays it on the effects input, as the game plays it but by a
//! player of the bench's own (the game's is never touched), in the song's
//! key and tempo, or one chosen in the tab's header, in the theme the tab
//! compares (Classic or Spooky: [`State::bench_theme`]; the game's own
//! sounds stay in the theme in force). The atoms a sound plays are its patches: the one
//! selected is shown and edited (its engine, its envelope, its parameters,
//! by their tables: [`jodel_synth::params`]) in the theme compared, and an
//! edit is heard on the next play, the game's in that theme too: the
//! patches live in the state's book ([`crate::core::patch_book`]), which
//! the effects play every theme's from. A patch kept is written to the
//! data folder ([`crate::store::patches`]) and loaded over the built-in
//! one at the start; reverted, the built-in one comes back. The keyboard
//! plays the selected atom's patch: two octaves on screen, and the
//! computer's keys (`a` to `'` the white keys, `w e t y u o p` the black
//! ones, `z` and `x` an octave down and up), each note held as long as
//! its key (its own: the others held hold on).

use jodel_rhythm::key::Key;
use jodel_synth::engines::AnyParams;
use jodel_synth::Patch;

use super::keymap::Op;
use super::{Effect, Side, State};
use crate::sfx::bench::{self, Held, Sound, What, SOUNDS};
use crate::sfx::conductor::Atom;
use crate::theme::{Theme, THEMES};

/// The keyboard's lowest note on screen (C4).
pub const LOW: f32 = 60.0;
/// The keys it has on screen: two octaves.
pub const KEYS: u8 = 24;

/// The farthest the computer's keys shift by octaves, either way.
const OCTAVES: i32 = 2;

/// The tempo the sounds play at with none chosen and no song, and the
/// tempo's step and range in the header.
pub const TEMPO: f64 = 120.0;
const TEMPO_STEP: f64 = 5.0;
const TEMPOS: (f64, f64) = (40.0, 240.0);

/// The Synth tab.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynthTab {
    /// The sound selected ([`SOUNDS`]).
    pub sound: usize,
    /// Which of its atoms is shown.
    pub atom: usize,
    /// The theme compared (none: the settings').
    pub theme: Option<&'static Theme>,
    /// The key chosen (none: the song's).
    pub key: Option<Key>,
    /// The tempo chosen, beats a minute (none: the song's).
    pub tempo: Option<f64>,
    /// The computer's keys held: each one's place on the keyboard and its
    /// note.
    pub held: Vec<(u8, f32)>,
    /// The note the pointer holds on the keyboard.
    pub mouse: Option<f32>,
    /// The atom the notes held play on (another shown lets them go).
    pub on: Option<Atom>,
    /// The computer's keys' octave shift.
    pub octave: i32,
    /// The parameter whose slider the pointer holds.
    pub dragging: Option<usize>,
}

/// What the tab asks the effects to play ([`Effect::Bench`]), in the
/// theme it compares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bench {
    /// A sound, in `key` (none: unpitched) at `bpm`.
    Play {
        /// The sound.
        what: What,
        /// The key.
        key: Option<Key>,
        /// The tempo.
        bpm: f64,
        /// The theme.
        theme: &'static Theme,
    },
    /// A key of the keyboard pressed: `note` on `atom`'s patch, held by
    /// `held`.
    Down {
        /// The atom.
        atom: Atom,
        /// The note.
        note: f32,
        /// What holds it.
        held: Held,
        /// The theme.
        theme: &'static Theme,
    },
    /// Its key let go of: its note alone, whatever patch and theme it
    /// was pressed on.
    Up {
        /// What held it.
        held: Held,
    },
}

/// The pointer and the window on the tab.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SynthAction {
    /// A sound clicked: selected, and played.
    Pick(usize),
    /// One of its atoms shown.
    Atom(usize),
    /// The atom's patch on another engine, at its default parameters.
    Engine(&'static str),
    /// A parameter's slider pressed or dragged to `share` of its way.
    Slide {
        /// The parameter (its place among the patch's knobs:
        /// [`Patch::knobs`]).
        knob: usize,
        /// Where, 0 to 1.
        share: f32,
    },
    /// The slider let go of.
    SlideEnd,
    /// A parameter set to `value` (a choice's option, a switch).
    Set {
        /// The parameter.
        knob: usize,
        /// Its value.
        value: f32,
    },
    /// The theme compared.
    Theme(&'static Theme),
    /// The key a step round (0: the song's again).
    KeyStep(i32),
    /// The tempo a step up or down (0: the song's again).
    TempoStep(i32),
    /// The atom's patch kept.
    Keep,
    /// The atom's patch back to its built-in one.
    Revert,
    /// The pointer holds this note on the keyboard (none: no note).
    Press(Option<f32>),
    /// Every key let go of (the window left).
    LetGo,
}

impl State {
    /// The theme the Synth tab plays and edits its sounds in: the one it
    /// compares, else the one in force ([`State::theme`]). A sounds-only
    /// override, on purpose: comparing a theme there never changes how
    /// the game looks or sounds.
    pub fn bench_theme(&self) -> &'static Theme {
        self.synth.theme.unwrap_or_else(|| self.theme())
    }
}

/// `atom`'s patch in the tab's theme as the book has it: edited, kept, or
/// built in.
pub fn patch(state: &State, atom: Atom) -> Patch {
    state.patches.patch(state.bench_theme(), atom)
}

/// The sound selected.
pub fn sound(state: &State) -> &'static Sound {
    let sounds = &*SOUNDS;
    &sounds[state.synth.sound.min(sounds.len() - 1)]
}

/// The atom shown: the selected sound's.
pub fn atom(state: &State) -> Atom {
    let atoms = &sound(state).atoms;
    atoms
        .get(state.synth.atom)
        .or(atoms.first())
        .copied()
        .unwrap_or(Atom::Pop)
}

/// The key the sounds play in: the one chosen, else the song's (shifted
/// as it is heard), else none.
pub fn key(state: &State) -> Option<Key> {
    state.synth.key.or_else(|| {
        let key = super::karaoke::rhythm_of(state)?.analysis.key?;
        let tonic = (i32::from(key.tonic()) + i32::from(state.playing.key)).rem_euclid(12);
        Some(Key::on(tonic as u8, key.minor()))
    })
}

/// The tempo they play at: the one chosen, else the song's, else
/// [`TEMPO`].
pub fn tempo(state: &State) -> f64 {
    state.synth.tempo.unwrap_or_else(|| {
        super::karaoke::rhythm_of(state)
            .and_then(|r| r.analysis.summary)
            .map(|s| s.bpm)
            .filter(|b| *b > 0.0)
            .unwrap_or(TEMPO)
    })
}

/// The note on the keyboard at the computer key's place `k`, the keys
/// shifted `octave` octaves.
pub fn key_note(octave: i32, k: u8) -> f32 {
    LOW + 12.0 * octave as f32 + f32::from(k)
}

/// The keys chosen round: no key (the song's), then C major to B major,
/// then C minor to B minor.
fn key_step(key: Option<Key>, by: i32) -> Option<Key> {
    let at = key.map_or(0, |k| 1 + i32::from(k.minor()) * 12 + i32::from(k.tonic()));
    let k = (at + by).rem_euclid(25);
    (k > 0).then(|| Key::on(((k - 1) % 12) as u8, k > 12))
}

/// Plays the sound selected.
fn play(state: &State, fx: &mut Vec<Effect>) {
    fx.push(Effect::Bench(Bench::Play {
        what: sound(state).what,
        key: key(state),
        bpm: tempo(state),
        theme: state.bench_theme(),
    }));
}

/// The atom's patch in the tab's theme changed by `change`, if that
/// changes it.
fn edit(state: &mut State, change: impl FnOnce(&mut Patch)) {
    let (theme, atom) = (state.bench_theme(), atom(state));
    state.patches.edit(theme, atom, change);
}

/// The sound at `k` selected, its first atom shown.
fn select(state: &mut State, k: usize) {
    let k = k.min(SOUNDS.len() - 1);
    if k != state.synth.sound {
        state.synth.sound = k;
        state.synth.atom = 0;
    }
}

/// The first sound of the moment `by` moments on from the selected one's.
fn moment_step(state: &mut State, by: i32) {
    let moments: Vec<_> = bench::moments().collect();
    let now = state.synth.sound;
    let at = (moments.iter())
        .position(|(_, span)| span.contains(&now))
        .unwrap_or(0) as i32;
    let (_, to) = &moments[(at + by).clamp(0, moments.len() as i32 - 1) as usize];
    select(state, to.start);
}

/// A note of the keyboard pressed, on the atom shown, held by `held`.
fn down(state: &mut State, note: f32, held: Held, fx: &mut Vec<Effect>) {
    let atom = atom(state);
    state.synth.on = Some(atom);
    let theme = state.bench_theme();
    fx.push(Effect::Bench(Bench::Down {
        atom,
        note,
        held,
        theme,
    }));
}

/// A note held by `held` let go of; none held any more, no atom played
/// on.
fn up(state: &mut State, held: Held, fx: &mut Vec<Effect>) {
    fx.push(Effect::Bench(Bench::Up { held }));
    if state.synth.held.is_empty() && state.synth.mouse.is_none() {
        state.synth.on = None;
    }
}

/// Every note held let go of, and a slider.
fn let_go(state: &mut State, fx: &mut Vec<Effect>) {
    let mut held: Vec<Held> = state.synth.held.drain(..).map(|h| Held::Key(h.0)).collect();
    held.extend(state.synth.mouse.take().map(|_| Held::Pointer));
    for h in held {
        up(state, h, fx);
    }
    state.synth.on = None;
    state.synth.dragging = None;
}

/// The computer key at place `k` of the keyboard let go of
/// ([`crate::core::Action::Released`] of [`Op::SynthNote`]): its note,
/// if it held one.
pub(crate) fn key_up(state: &mut State, k: i8, fx: &mut Vec<Effect>) {
    let Ok(k) = u8::try_from(k) else {
        return;
    };
    if let Some(i) = state.synth.held.iter().position(|h| h.0 == k) {
        state.synth.held.remove(i);
        up(state, Held::Key(k), fx);
    }
}

/// What the pointer and the window do on the tab.
pub fn act(state: &mut State, what: SynthAction, fx: &mut Vec<Effect>) {
    match what {
        SynthAction::Pick(k) => {
            select(state, k);
            play(state, fx);
        }
        SynthAction::Atom(k) => {
            let n = sound(state).atoms.len();
            state.synth.atom = k.min(n.saturating_sub(1));
        }
        SynthAction::Engine(name) => {
            if let Some(engine) = AnyParams::of_engine(name) {
                edit(state, |p| {
                    if p.engine.engine() != name {
                        p.engine = engine;
                    }
                });
            }
        }
        SynthAction::Slide { knob, share } => {
            state.synth.dragging = Some(knob);
            edit(state, |p| {
                if let Some(n) = p.knob(knob) {
                    p.set(knob, n.spec.kind.at(share));
                }
            });
        }
        SynthAction::SlideEnd => state.synth.dragging = None,
        SynthAction::Set { knob, value } => edit(state, |p| {
            p.set(knob, value);
        }),
        SynthAction::Theme(t) => state.synth.theme = Some(t),
        SynthAction::KeyStep(by) => {
            state.synth.key = if by == 0 {
                None
            } else {
                key_step(state.synth.key, by)
            };
        }
        SynthAction::TempoStep(by) => {
            state.synth.tempo = (by != 0).then(|| {
                let now = (tempo(state) / TEMPO_STEP).round() * TEMPO_STEP;
                (now + f64::from(by) * TEMPO_STEP).clamp(TEMPOS.0, TEMPOS.1)
            });
        }
        SynthAction::Keep => keep(state, fx),
        SynthAction::Revert => revert(state, fx),
        SynthAction::Press(note) => {
            if note != state.synth.mouse {
                if std::mem::replace(&mut state.synth.mouse, note).is_some() {
                    up(state, Held::Pointer, fx);
                }
                if let Some(new) = note {
                    down(state, new, Held::Pointer, fx);
                }
            }
        }
        SynthAction::LetGo => let_go(state, fx),
    }
}

/// The atom's patch kept as it is now, written to the data folder.
fn keep(state: &mut State, fx: &mut Vec<Effect>) {
    let (theme, atom) = (state.bench_theme(), atom(state));
    match state.patches.keep(theme, atom) {
        Ok(patch) => {
            fx.push(Effect::KeepPatch {
                theme,
                atom,
                patch: Some(patch),
            });
            state.info(format!("{} kept", atom.name()));
        }
        Err(super::patch_book::PatchStatus::AsBuilt) => {
            state.info("Nothing to keep: it is as built");
        }
        Err(_) => state.info("Kept already"),
    }
}

/// The atom's patch back to its built-in one, what was kept of it
/// forgotten.
fn revert(state: &mut State, fx: &mut Vec<Effect>) {
    let (theme, atom) = (state.bench_theme(), atom(state));
    let Some(kept) = state.patches.revert(theme, atom) else {
        return;
    };
    if kept {
        fx.push(Effect::KeepPatch {
            theme,
            atom,
            patch: None,
        });
    }
    state.info(format!("{} as built", atom.name()));
}

/// What a key does on the tab; whether it was the tab's.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    let n = SOUNDS.len();
    let at = state.synth.sound;
    match op {
        Op::Up => select(state, at.saturating_sub(1)),
        Op::Down => select(state, (at + 1).min(n - 1)),
        Op::PageUp => moment_step(state, -1),
        Op::PageDown => moment_step(state, 1),
        Op::Top => select(state, 0),
        Op::Bottom => select(state, n - 1),
        Op::Open => play(state, fx),
        Op::SynthAtom(by) => {
            let atoms = sound(state).atoms.len().max(1) as i32;
            state.synth.atom = (state.synth.atom as i32 + i32::from(by)).rem_euclid(atoms) as usize;
        }
        Op::SynthNote(k) => {
            let Ok(k) = u8::try_from(k) else {
                return true;
            };
            if !state.synth.held.iter().any(|h| h.0 == k) {
                let note = key_note(state.synth.octave, k);
                state.synth.held.push((k, note));
                down(state, note, Held::Key(k), fx);
            }
        }
        Op::SynthOctave(by) => {
            state.synth.octave = (state.synth.octave + i32::from(by)).clamp(-OCTAVES, OCTAVES);
        }
        Op::SynthTheme => {
            let at = crate::sfx::conductor::place(state.bench_theme());
            act(
                state,
                SynthAction::Theme(THEMES[(at + 1) % THEMES.len()]),
                fx,
            );
        }
        Op::SynthKey(by) => act(state, SynthAction::KeyStep(i32::from(by)), fx),
        Op::SynthTempo(by) => act(state, SynthAction::TempoStep(i32::from(by)), fx),
        Op::SynthKeep => act(state, SynthAction::Keep, fx),
        Op::SynthRevert => act(state, SynthAction::Revert, fx),
        _ => return false,
    }
    true
}

/// After any action: the notes held are let go of, and a slider, once the
/// tab is left; the notes, once another atom shows.
pub(crate) fn settle(state: &mut State, fx: &mut Vec<Effect>) {
    let t = &state.synth;
    let left = !state.showing(Side::Synth)
        && (!t.held.is_empty() || t.mouse.is_some() || t.dragging.is_some());
    if left || t.on.is_some_and(|a| a != atom(state)) {
        let_go(state, fx);
    }
}
