//! The Synth tab as the window shows it ([`crate::gui::synth`]): the
//! header (the key and the tempo the sounds play in, each chosen or the
//! song's, and the theme compared), the sounds by moment (the selected
//! one marked, one with an edited patch marked too), the selected sound's
//! atoms (the one shown lit), the shown atom's patch (where it stands, its
//! engine, its envelope, its parameters that apply, each off the built-in
//! one or not) and the keyboard (the keys held lit, the computer's keys
//! written on theirs). Its outline ([`Synth`]'s `Display`) is what the
//! shots write.

use std::fmt;

use jodel_synth::params::Knob;
use jodel_synth::patch::{Adsr, ENVELOPE};

use crate::core::keymap::{self, Context, KeyCode, Op};
use crate::core::patch_book::PatchStatus;
use crate::core::synth_tab::{self, KEYS, LOW};
use crate::core::State;
use crate::sfx::bench::{self, Moment, Sound, SOUNDS};
use crate::sfx::conductor::Atom;
use crate::theme::Theme;

/// The tab.
pub struct Synth {
    /// The key the sounds play in (`"no key"`).
    pub key: String,
    /// Whether it is chosen (not the song's).
    pub key_chosen: bool,
    /// Their tempo, beats a minute.
    pub tempo: f64,
    /// Whether it is chosen.
    pub tempo_chosen: bool,
    /// The theme compared.
    pub theme: &'static Theme,
    /// The sounds, by moment.
    pub moments: Vec<(Moment, Vec<SoundRow>)>,
    /// The sound selected.
    pub sound: &'static Sound,
    /// Its place.
    pub selected: usize,
    /// Its atoms.
    pub atoms: Vec<AtomChip>,
    /// The shown atom's patch.
    pub patch: PatchView,
    /// The keyboard.
    pub keyboard: Keyboard,
}

/// A sound in the list.
pub struct SoundRow {
    /// Its place among the sounds.
    pub k: usize,
    /// Its name.
    pub name: &'static str,
    /// Whether it is the one selected.
    pub selected: bool,
    /// Whether a patch of it is off the built-in one in the theme
    /// compared.
    pub edited: bool,
}

/// An atom of the selected sound.
pub struct AtomChip {
    /// The atom.
    pub atom: Atom,
    /// Its place among the sound's.
    pub place: usize,
    /// Whether it is the one shown.
    pub shown: bool,
    /// Whether its patch is off the built-in one.
    pub edited: bool,
}

/// A parameter shown: what it is now, and whether that is off the
/// built-in patch's.
pub struct KnobRow {
    /// It.
    pub knob: Knob,
    /// Off the built-in one (in the accent).
    pub off: bool,
}

impl KnobRow {
    /// Its value as it is shown.
    pub fn text(&self) -> String {
        self.knob.spec.kind.text(self.knob.value)
    }
}

/// The shown atom's patch.
pub struct PatchView {
    /// Whose.
    pub atom: Atom,
    /// Where it stands.
    pub status: PatchStatus,
    /// Its engine.
    pub engine: &'static str,
    /// Its envelope.
    pub adsr: Adsr,
    /// Its parameters that apply, its own first, then its engine's.
    pub knobs: Vec<KnobRow>,
    /// The parameter whose slider the pointer holds.
    pub held: Option<usize>,
}

impl PatchView {
    /// The envelope's parameters.
    pub fn envelope(&self) -> impl Iterator<Item = &KnobRow> {
        (self.knobs.iter()).filter(|r| r.knob.spec.group == ENVELOPE)
    }

    /// The other parameters in their groups, the engine's first, each
    /// group in the order its first parameter comes.
    pub fn groups(&self) -> Vec<(&'static str, Vec<&KnobRow>)> {
        let engine_first = (self.knobs.iter().filter(|r| r.knob.of_engine))
            .chain(self.knobs.iter().filter(|r| !r.knob.of_engine));
        let mut groups: Vec<(&'static str, Vec<&KnobRow>)> = Vec::new();
        for row in engine_first.filter(|r| r.knob.spec.group != ENVELOPE) {
            let group = row.knob.spec.group;
            match groups.iter_mut().find(|(g, _)| *g == group) {
                Some((_, rows)) => rows.push(row),
                None => groups.push((group, vec![row])),
            }
        }
        groups
    }
}

/// The keyboard: the keys lit (held), the labels on keys (the computer's
/// keys', the Cs), the computer keys' octave and the atom it plays, and
/// the notes held.
pub struct Keyboard {
    /// The keys lit, from the lowest C.
    pub lit: Vec<u8>,
    /// The labels on keys.
    pub labels: Vec<(u8, String)>,
    /// The computer keys' octave shift.
    pub octave: i32,
    /// The atom it plays.
    pub atom: Atom,
    /// The notes held.
    pub held: Vec<f32>,
}

impl Synth {
    /// The tab as `state` has it.
    pub fn of(state: &State) -> Synth {
        let tab = &state.synth;
        let theme = synth_tab::theme(state);
        let book = &state.patches;
        let edited = |a: Atom| book.status(theme, a) != PatchStatus::AsBuilt;
        let moments = (bench::moments())
            .map(|(moment, span)| {
                let rows = (span.map(|k| SoundRow {
                    k,
                    name: &SOUNDS[k].name,
                    selected: k == tab.sound,
                    edited: SOUNDS[k].atoms.iter().any(|&a| edited(a)),
                }))
                .collect();
                (moment, rows)
            })
            .collect();
        let sound = synth_tab::sound(state);
        let shown = synth_tab::atom(state);
        let atoms = (sound.atoms.iter().enumerate())
            .map(|(place, &atom)| AtomChip {
                atom,
                place,
                shown: atom == shown,
                edited: edited(atom),
            })
            .collect();
        Synth {
            key: synth_tab::key(state).map_or_else(|| "no key".to_owned(), |k| k.name()),
            key_chosen: tab.key.is_some(),
            tempo: synth_tab::tempo(state),
            tempo_chosen: tab.tempo.is_some(),
            theme,
            moments,
            sound,
            selected: tab.sound,
            atoms,
            patch: patch(state, theme, shown),
            keyboard: keyboard(state, shown),
        }
    }
}

/// `atom`'s patch in `theme` as the tab shows it.
fn patch(state: &State, theme: &Theme, atom: Atom) -> PatchView {
    let p = state.patches.patch(theme, atom);
    let built = theme.patch(atom);
    let same_engine = built.engine.engine() == p.engine.engine();
    let knobs = (p.knobs().filter(|n| n.applies))
        .map(|knob| {
            let was = (same_engine || !knob.of_engine)
                .then(|| built.knob(knob.k))
                .flatten()
                .map(|n| n.value);
            KnobRow {
                knob,
                off: was != Some(knob.value),
            }
        })
        .collect();
    PatchView {
        atom,
        status: state.patches.status(theme, atom),
        engine: p.engine.engine(),
        adsr: p.adsr,
        knobs,
        held: state.synth.dragging,
    }
}

/// The keyboard: lit where held, the computer's keys written on theirs,
/// the Cs on theirs where no computer key is.
fn keyboard(state: &State, atom: Atom) -> Keyboard {
    let tab = &state.synth;
    let on_screen = |note: f32| {
        let k = (note - LOW).round();
        (0.0..f32::from(KEYS)).contains(&k).then_some(k as u8)
    };
    let held: Vec<f32> = (tab.held.iter().map(|h| h.1)).chain(tab.mouse).collect();
    let lit = held.iter().copied().filter_map(on_screen).collect();
    let mut labels: Vec<(u8, String)> = computer_keys()
        .filter_map(|(k, c)| {
            Some((
                on_screen(synth_tab::key_note(tab.octave, k))?,
                c.to_string(),
            ))
        })
        .collect();
    for c in [0u8, 12] {
        if !labels.iter().any(|(k, _)| *k == c) {
            labels.push((c, format!("C{}", 4 + c / 12)));
        }
    }
    Keyboard {
        lit,
        labels,
        octave: tab.octave,
        atom,
        held,
    }
}

/// The computer's keys of the keyboard, each its place over the lowest C
/// and its letter: the keys of the Synth tab's rows ([`Op::SynthNote`]).
fn computer_keys() -> impl Iterator<Item = (u8, char)> {
    (0..KEYS).filter_map(|k| {
        let op = Op::SynthNote(i8::try_from(k).ok()?);
        match keymap::keys_for(Context::Synth, op).first()?.code {
            KeyCode::Char(c) => Some((k, c)),
            _ => None,
        }
    })
}

/// The outline: the key, the tempo and the theme; the sounds by moment,
/// the selected one marked; the patch shown, each parameter that applies
/// with its value; the keys held.
impl fmt::Display for Synth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "synth: {}, {:.0} bpm, {}",
            self.key, self.tempo, self.theme.name
        )?;
        for (moment, rows) in &self.moments {
            writeln!(f, "[{}]", moment.name())?;
            for r in rows {
                let mark = if r.selected { ">" } else { " " };
                writeln!(f, "{mark} {}", r.name)?;
            }
        }
        let p = &self.patch;
        writeln!(
            f,
            "patch {} ({}) on {}",
            p.atom.name(),
            p.status.name(),
            p.engine
        )?;
        for r in &p.knobs {
            writeln!(f, "  {} = {}", r.knob.spec.key(), r.text())?;
        }
        let held: Vec<String> = (self.keyboard.held.iter())
            .map(|n| format!("{n:.0}"))
            .collect();
        writeln!(f, "keys held: {}", held.join(" "))
    }
}
