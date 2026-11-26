//! What the table is shown as: the help screen ([`help_sections`]), the
//! command palette ([`palette_entries`]), and the keys a screen's foot or
//! a tab's border names ([`hint`]).

use super::{Binding, Context, Op, KEYMAP};

/// The keys of `binding`, written for people (`N / shift+→`).
pub fn keys_label(binding: &Binding) -> String {
    binding
        .keys
        .iter()
        .map(|k| k.label())
        .collect::<Vec<_>>()
        .join(" / ")
}

/// The first key bound to `op` in `context`, else anywhere
/// ([`Context::Global`], as [`super::lookup`] falls back to it), written as
/// the help writes it (`←`, `enter`): for the keys a screen's foot or a
/// tab's border names. Empty where it is bound to none.
pub fn hint(context: Context, op: Op) -> String {
    [context, Context::Global]
        .iter()
        .find_map(|&c| KEYMAP.iter().find(|b| b.context == c && b.op == op))
        .and_then(|b| b.keys.first())
        .map_or_else(String::new, |k| k.label())
}

/// The first keys of `ops` in `context` ([`hint`]), one after the other
/// with a space between: `← →` for a pair that steps back and on.
pub fn keys(context: Context, ops: &[Op]) -> String {
    ops.iter()
        .map(|&op| hint(context, op))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One section of the help screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelpSection {
    /// The context title.
    pub title: &'static str,
    /// (keys, what they do).
    pub lines: Vec<(String, &'static str)>,
}

/// The help's sections that are not the table's: typing, the preview and
/// the mouse, after it. Each its title and its lines (what is done, what it
/// does).
#[rustfmt::skip]
const MORE: &[(&str, &[(&str, &str)])] = &[
    (
        "Typing in the lists",
        &[
            ("a-z 0-9 …", "jump to the first name (as shown) starting with what you type"),
            ("the same letter", "again: the next name starting with it"),
            ("capitals", "the commands (Q quit, N next, S stop …); lowercase letters only type"),
        ],
    ),
    (
        "Preview (hold ctrl in the browser)",
        &[
            ("hold ctrl", "the song playing fades out and waits; the file selected plays from a third in"),
            ("ctrl+↑ / ↓ / pgup / pgdn / home / end", "the file selected next plays from its third (a folder: silence)"),
            ("ctrl+click / pointer / wheel", "the same with the mouse: the row under the pointer is selected"),
            ("release ctrl", "back to the song that was playing, where it was (paused stays paused)"),
            ("ctrl+enter", "queue the file (as enter) and go back; nothing previewed is queued"),
        ],
    ),
    (
        "Mouse",
        &[
            ("click", "select a row; a tab (Browser, Queue, Transition, Karaoke, Mixer) shows it"),
            ("double click", "as enter: open folder; queue file or playlist (queue: play the track)"),
            ("right click", "queue file or folder"),
            ("wheel", "scroll the list"),
            ("click / drag", "the progress bar seeks"),
            ("click", "drums / bass / vocals / other switch a stem on or off (with a GPU)"),
            ("click", "the cover opens Now playing; the device name its setting"),
        ],
    ),
];

/// The help screen: the table's rows by context, each context's section
/// where its first row is, then [`MORE`].
pub fn help_sections() -> Vec<HelpSection> {
    let mut sections: Vec<(Context, HelpSection)> = Vec::new();
    for binding in KEYMAP {
        let Some(line) = binding.help_line() else {
            continue;
        };
        match sections.iter_mut().find(|(c, _)| *c == binding.context) {
            Some((_, section)) => section.lines.push(line),
            None => sections.push((
                binding.context,
                HelpSection {
                    title: binding.context.title(),
                    lines: vec![line],
                },
            )),
        }
    }
    let more = MORE.iter().map(|&(title, lines)| HelpSection {
        title,
        lines: lines
            .iter()
            .map(|&(keys, what)| (keys.to_owned(), what))
            .collect(),
    });
    sections.into_iter().map(|(_, s)| s).chain(more).collect()
}

/// The help screen as a search after `/` leaves it: only the lines whose
/// keys or text contain `query` (case ignored, as the browser's filter),
/// each under its section's title. Sections left empty go; an empty query
/// keeps everything.
pub fn help_matching(query: &str) -> Vec<HelpSection> {
    let needle = query.to_lowercase();
    help_sections()
        .into_iter()
        .filter_map(|mut section| {
            section.lines.retain(|(keys, text)| {
                keys.to_lowercase().contains(&needle) || text.to_lowercase().contains(&needle)
            });
            (!section.lines.is_empty()).then_some(section)
        })
        .collect()
}

/// One palette entry: an operation and its key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteEntry {
    /// The operation.
    pub op: Op,
    /// Its name.
    pub name: &'static str,
    /// Its key, written for people.
    pub keys: String,
}

/// Every command the palette offers, from [`KEYMAP`] (each operation
/// once, with its first binding's keys).
pub fn palette_entries() -> Vec<PaletteEntry> {
    let mut out: Vec<PaletteEntry> = Vec::new();
    for binding in KEYMAP {
        if !binding.op.in_palette() || out.iter().any(|e| e.op == binding.op) {
            continue;
        }
        out.push(PaletteEntry {
            op: binding.op,
            name: binding.op.name(),
            keys: keys_label(binding),
        });
    }
    out
}

/// The palette entries matching `text` (every word of it, case ignored).
pub fn palette_matches(text: &str) -> Vec<PaletteEntry> {
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    palette_entries()
        .into_iter()
        .filter(|e| {
            let name = e.name.to_lowercase();
            words.iter().all(|w| name.contains(w.as_str()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The help's seek keys say the step the player seeks by.
    #[test]
    fn the_seek_keys_help_says_the_seek_step() {
        let step = format!(" {} s", crate::core::SEEK_STEP);
        for op in [Op::SeekBack, Op::SeekForward] {
            let row = KEYMAP.iter().find(|b| b.op == op).unwrap();
            assert!(row.help.ends_with(&step), "{row:?}");
        }
    }

    /// A hint is the first key in the context, else anywhere's, else none.
    #[test]
    fn a_hint_is_the_first_key_in_its_context_or_anywhere() {
        assert_eq!(hint(Context::Desk, Op::DeskPrevStrip), "←");
        assert_eq!(hint(Context::List, Op::Bottom), "end");
        assert_eq!(hint(Context::Transition, Op::Help), "?");
        assert_eq!(hint(Context::Fades, Op::Back), "esc");
        assert_eq!(hint(Context::Desk, Op::GridHalf), "");
    }
}
