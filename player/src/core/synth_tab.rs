//! The Synth tab ([`Side::Synth`]): a workbench for the game's sounds.
//! Every sound is listed by its moment ([`crate::sfx::bench::SOUNDS`]);
//! a click plays it through the conductor on the effects input, in the
//! song's key and tempo, or one chosen in the tab's header. The atoms a
//! sound plays are its patches: the one selected is shown and edited
//! (its engine, its envelope, its parameters, by their tables:
//! [`jodel_synth::params`]), in the theme compared (Classic or Spooky), and
//! an edit is heard on the next play: the effects play each atom's patch
//! as this tab has it ([`patch`]), a change told the synth as a command.
//! A patch kept is written to the data folder ([`crate::patches`]) and
//! loaded over the built-in one at the start; reverted, the built-in one
//! comes back. The keyboard plays the selected atom's patch: two octaves
//! on screen, and the computer's keys (`a` to `'` the white keys, `w e t y
//! u o p` the black ones, `z` and `x` an octave down and up), each note
//! held as long as its key.

use jodel_rhythm::key::Key;
use jodel_synth::engines::AnyParams;
use jodel_synth::params::Knobs;
use jodel_synth::Patch;

use super::keymap::Op;
use super::{Effect, Screen, Side, State};
use crate::sfx::bench::{Moment, Sound, What, SOUNDS};
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

/// A patch edited on the tab: whose (a theme's atom's), what it is now,
/// and what is kept of it (none: nothing, the built-in one at the start).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edited {
    /// The theme's name.
    pub theme: &'static str,
    /// The atom.
    pub atom: Atom,
    /// What it is now.
    pub patch: Patch,
    /// What is kept in the data folder.
    pub kept: Option<Patch>,
}

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
    /// The patches edited or kept.
    pub patches: Vec<Edited>,
    /// The computer's keys held: each one's place on the keyboard and its
    /// note.
    pub held: Vec<(i8, f32)>,
    /// The note the pointer holds on the keyboard.
    pub mouse: Option<f32>,
    /// The atom the notes held play on (another shown lets them go).
    pub on: Option<Atom>,
    /// The computer's keys' octave shift.
    pub octave: i32,
    /// The parameter whose slider the pointer holds.
    pub dragging: Option<usize>,
}

/// What the tab asks the effects to play ([`Effect::Bench`]).
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
    },
    /// A key of the keyboard pressed: `note` on `atom`'s patch.
    Down {
        /// The atom.
        atom: Atom,
        /// The note.
        note: f32,
    },
    /// Its key let go of.
    Up {
        /// The atom.
        atom: Atom,
        /// The note.
        note: f32,
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
        /// The parameter (its place in the patch's [`Knobs`]).
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
    /// A computer key of the keyboard let go of: its place.
    KeyUp(i8),
    /// Every key let go of (the window left).
    LetGo,
}

/// Whether the tab shows.
pub fn shown(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Synth
}

/// Whether it is what the keys act on.
pub fn on_tab(state: &State) -> bool {
    shown(state) && state.palette.is_none()
}

/// The theme the effects play in: the tab's while it shows, else the
/// settings'.
pub fn theme(state: &State) -> &'static Theme {
    match state.synth.theme {
        Some(t) if shown(state) => t,
        _ => state.config.karaoke_theme,
    }
}

/// `atom`'s patch in `theme` as the tab has it: edited, kept, or built in.
pub fn patch(state: &State, theme: &Theme, atom: Atom) -> Patch {
    edited(state, theme, atom).map_or_else(|| theme.patch(atom), |e| e.patch)
}

/// Its edit, if it has one.
pub fn edited<'a>(state: &'a State, theme: &Theme, atom: Atom) -> Option<&'a Edited> {
    (state.synth.patches.iter()).find(|e| e.theme == theme.name && e.atom == atom)
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
    }));
}

/// The atom's patch changed by `change`, if that changes it.
fn edit(state: &mut State, change: impl FnOnce(&mut Patch)) {
    let theme = theme(state);
    let atom = atom(state);
    let mut p = patch(state, theme, atom);
    change(&mut p);
    let tab = &mut state.synth;
    match (tab.patches.iter_mut()).find(|e| e.theme == theme.name && e.atom == atom) {
        Some(e) => e.patch = p,
        None if p != theme.patch(atom) => tab.patches.push(Edited {
            theme: theme.name,
            atom,
            patch: p,
            kept: None,
        }),
        None => {}
    }
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
    let sounds = &*SOUNDS;
    let mut moments: Vec<Moment> = sounds.iter().map(|s| s.moment).collect();
    moments.dedup();
    let now = sound(state).moment;
    let at = moments.iter().position(|&m| m == now).unwrap_or(0) as i32;
    let to = moments[(at + by).clamp(0, moments.len() as i32 - 1) as usize];
    if let Some(k) = sounds.iter().position(|s| s.moment == to) {
        select(state, k);
    }
}

/// The note on the keyboard at the computer key's place `k`.
fn key_note(state: &State, k: i8) -> f32 {
    LOW + 12.0 * state.synth.octave as f32 + f32::from(k)
}

/// A note of the keyboard pressed, on the atom shown.
fn down(state: &mut State, note: f32, fx: &mut Vec<Effect>) {
    let atom = atom(state);
    state.synth.on = Some(atom);
    fx.push(Effect::Bench(Bench::Down { atom, note }));
}

/// A note held let go of, on the atom the notes held play on; none held
/// any more, none.
fn up(state: &mut State, note: f32, fx: &mut Vec<Effect>) {
    let atom = state.synth.on.unwrap_or_else(|| self::atom(state));
    fx.push(Effect::Bench(Bench::Up { atom, note }));
    if state.synth.held.is_empty() && state.synth.mouse.is_none() {
        state.synth.on = None;
    }
}

/// Every note held let go of, and a slider.
fn let_go(state: &mut State, fx: &mut Vec<Effect>) {
    let mut notes: Vec<f32> = state.synth.held.drain(..).map(|h| h.1).collect();
    notes.extend(state.synth.mouse.take());
    for note in notes {
        up(state, note, fx);
    }
    state.synth.on = None;
    state.synth.dragging = None;
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
                if let Some(spec) = p.spec(knob) {
                    p.set(knob, spec.kind.at(share));
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
                if let Some(old) = std::mem::replace(&mut state.synth.mouse, note) {
                    up(state, old, fx);
                }
                if let Some(new) = note {
                    down(state, new, fx);
                }
            }
        }
        SynthAction::KeyUp(k) => {
            if let Some(i) = state.synth.held.iter().position(|h| h.0 == k) {
                let (_, note) = state.synth.held.remove(i);
                up(state, note, fx);
            }
        }
        SynthAction::LetGo => let_go(state, fx),
    }
}

/// The atom's patch kept as it is now, written to the data folder.
fn keep(state: &mut State, fx: &mut Vec<Effect>) {
    let (theme, atom) = (theme(state), atom(state));
    let Some(e) =
        (state.synth.patches.iter_mut()).find(|e| e.theme == theme.name && e.atom == atom)
    else {
        return super::update::say_info(state, "Nothing to keep: it is as built");
    };
    if e.kept == Some(e.patch) {
        return super::update::say_info(state, "Kept already");
    }
    e.kept = Some(e.patch);
    fx.push(Effect::KeepPatch {
        theme: theme.name,
        atom,
        patch: Some(e.patch),
    });
    super::update::say_info(state, format!("{} kept", atom.name()));
}

/// The atom's patch back to its built-in one, what was kept of it
/// forgotten.
fn revert(state: &mut State, fx: &mut Vec<Effect>) {
    let (theme, atom) = (theme(state), atom(state));
    let tab = &mut state.synth;
    let Some(i) = (tab.patches.iter()).position(|e| e.theme == theme.name && e.atom == atom) else {
        return;
    };
    let gone = tab.patches.remove(i);
    if gone.kept.is_some() {
        fx.push(Effect::KeepPatch {
            theme: theme.name,
            atom,
            patch: None,
        });
    }
    super::update::say_info(state, format!("{} as built", atom.name()));
}

/// What a key does on the tab; whether it was the tab's.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if !on_tab(state) {
        return false;
    }
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
            if !state.synth.held.iter().any(|h| h.0 == k) {
                let note = key_note(state, k);
                state.synth.held.push((k, note));
                down(state, note, fx);
            }
        }
        Op::SynthOctave(by) => {
            state.synth.octave = (state.synth.octave + i32::from(by)).clamp(-OCTAVES, OCTAVES);
        }
        Op::SynthTheme => {
            let now = theme(state);
            let at = THEMES.iter().position(|t| *t == now).unwrap_or(0);
            state.synth.theme = Some(THEMES[(at + 1) % THEMES.len()]);
        }
        Op::SynthKey(by) => act(state, SynthAction::KeyStep(i32::from(by)), fx),
        Op::SynthTempo(by) => act(state, SynthAction::TempoStep(i32::from(by)), fx),
        Op::SynthKeep => keep(state, fx),
        Op::SynthRevert => revert(state, fx),
        _ => return false,
    }
    true
}

/// After any action: the notes held are let go of, and a slider, once the
/// tab is left; the notes, once another atom shows.
pub(crate) fn settle(state: &mut State, fx: &mut Vec<Effect>) {
    let t = &state.synth;
    let left = !shown(state) && (!t.held.is_empty() || t.mouse.is_some() || t.dragging.is_some());
    if left || t.on.is_some_and(|a| a != atom(state)) {
        let_go(state, fx);
    }
}
