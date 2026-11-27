//! The one table of key bindings.
//!
//! The window translates its key events into [`Key`] and asks [`lookup`]
//! (or [`action`]) what it means in the current state. The help screen
//! ([`help_sections`]), the command palette ([`palette_entries`]) and the
//! keys the screens' feet and the tabs' borders name ([`hint`]) are
//! written from the same [`KEYMAP`] table, so they always match.
//!
//! - [`table`]: the table, a row a binding;
//! - [`op`]: the operations ([`Op`]), their names in the palette, where
//!   each is done and what it is;
//! - [`help`]: the help, the palette and the hints, from the table.

use crate::core::{Screen, Side, State, TYPE_AHEAD_MS};

mod help;
mod op;
mod table;

pub use help::{
    help_matching, help_sections, hint, keys, keys_label, palette_entries, palette_matches,
    HelpSection, PaletteEntry,
};
pub use op::{Kind, Op};
pub use table::KEYMAP;

/// A key, as the window reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// A character, already shifted (`A`, `?`).
    Char(char),
    /// Enter / return.
    Enter,
    /// Escape.
    Esc,
    /// Backspace.
    Backspace,
    /// Tab.
    Tab,
    /// Arrow up.
    Up,
    /// Arrow down.
    Down,
    /// Arrow left.
    Left,
    /// Arrow right.
    Right,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Home.
    Home,
    /// End.
    End,
    /// Delete.
    Delete,
}

/// A key with its modifiers (shift is in the character).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    /// The key.
    pub code: KeyCode,
    /// Control held.
    pub ctrl: bool,
    /// Alt (option) held.
    pub alt: bool,
    /// Shift held, for keys other than characters (a character already
    /// says it: `Q`, `?`).
    pub shift: bool,
}

impl Key {
    /// A key without modifiers.
    pub const fn plain(code: KeyCode) -> Key {
        Key {
            code,
            ctrl: false,
            alt: false,
            shift: false,
        }
    }

    /// A key other than a character, with shift.
    pub const fn shift(code: KeyCode) -> Key {
        Key {
            code,
            ctrl: false,
            alt: false,
            shift: true,
        }
    }

    /// A character key without modifiers.
    pub const fn ch(c: char) -> Key {
        Key::plain(KeyCode::Char(c))
    }

    /// A character key with alt.
    pub const fn alt(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: false,
            alt: true,
            shift: false,
        }
    }

    /// A character key with shift, as the window reports a digit held with
    /// it (the key without the modifier, for any keyboard layout).
    pub const fn shifted(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: false,
            alt: false,
            shift: true,
        }
    }

    /// A character key with control.
    pub const fn ctrl(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: true,
            alt: false,
            shift: false,
        }
    }

    /// How the help screen writes the key.
    pub fn label(self) -> String {
        let base = match self.code {
            KeyCode::Char(' ') => "space".to_owned(),
            KeyCode::Char(c) => c.to_string(),
            KeyCode::Enter => "enter".into(),
            KeyCode::Esc => "esc".into(),
            KeyCode::Backspace => "backspace".into(),
            KeyCode::Tab => "tab".into(),
            KeyCode::Up => "↑".into(),
            KeyCode::Down => "↓".into(),
            KeyCode::Left => "←".into(),
            KeyCode::Right => "→".into(),
            KeyCode::PageUp => "pgup".into(),
            KeyCode::PageDown => "pgdn".into(),
            KeyCode::Home => "home".into(),
            KeyCode::End => "end".into(),
            KeyCode::Delete => "del".into(),
        };
        match (self.ctrl, self.alt, self.shift) {
            (true, _, _) => format!("ctrl+{base}"),
            (false, true, _) => format!("alt+{base}"),
            (false, false, true) => format!("shift+{base}"),
            _ => base,
        }
    }
}

/// Where a binding applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Context {
    /// Everywhere, unless a more specific context binds the key, and not
    /// while typing (except keys with control).
    Global,
    /// Everywhere as [`Context::Global`] is, and in the lyrics editor too,
    /// which takes no other key from elsewhere: playing, the volume, the
    /// stems, quitting.
    Playback,
    /// The browser and queue list.
    List,
    /// The queue (before [`Context::List`]).
    Queue,
    /// Typing a filter after `/`.
    Filter,
    /// Typing in the command palette.
    Palette,
    /// The Now playing screen.
    NowPlaying,
    /// The settings screen.
    Settings,
    /// The device list in the settings.
    DeviceList,
    /// The folder list the data folder is chosen in.
    FolderList,
    /// The help screen.
    Help,
    /// Typing a search of the help after `/`.
    HelpSearch,
    /// The Transition tab.
    Transition,
    /// Editing a transition's fades on the Transition tab.
    Fades,
    /// The Karaoke tab.
    Karaoke,
    /// Karaoke mode.
    KaraokeMode,
    /// Syncing, on the Karaoke tab.
    Sync,
    /// The Mixer tab.
    Desk,
    /// The lyrics editor.
    LyricsEditor,
    /// Writing a word's or a line's text in the lyrics editor (the text
    /// field takes the rest of the keys).
    LyricsText,
    /// The Synth tab.
    Synth,
    /// The Shader tab.
    Shader,
    /// Typing a search of the Shader tab's presets (after `/`).
    ShaderSearch,
}

impl Context {
    /// The help section title.
    pub fn title(self) -> &'static str {
        match self {
            Context::Global => "Anywhere",
            Context::Playback => Context::Global.title(),
            Context::List => "Browser and queue",
            Context::Queue => "Queue",
            Context::Filter => "Filter (after /)",
            Context::Palette => "Command palette (after :)",
            Context::NowPlaying => "Now playing",
            Context::Settings => "Settings",
            Context::DeviceList => "Device list",
            Context::FolderList => "Data folder list",
            Context::Help => "Help",
            Context::HelpSearch => "Searching the help (after /)",
            Context::Transition => "Transition tab",
            Context::Fades => "Transition tab, editing the fades (after e)",
            Context::Karaoke => "Karaoke tab",
            Context::KaraokeMode => "Karaoke mode (fullscreen)",
            Context::Sync => "Syncing (s on the Karaoke tab)",
            Context::Desk => "Mixer tab",
            Context::LyricsEditor => "Lyrics editor (e on the Karaoke tab)",
            Context::LyricsText => "Lyrics editor, writing (after enter)",
            Context::Synth => "Synth tab",
            Context::Shader => "Shader tab",
            Context::ShaderSearch => "Shader tab, searching the presets (after /)",
        }
    }

    /// The help's section its rows are in: anywhere's for playing's (it is
    /// anywhere's, but for the lyrics editor taking its keys too).
    pub fn help_section(self) -> Context {
        match self {
            Context::Playback => Context::Global,
            context => context,
        }
    }

    /// Keys type text here: unbound characters are typed.
    pub fn types(self) -> bool {
        matches!(
            self,
            Context::Filter | Context::Palette | Context::HelpSearch | Context::ShaderSearch
        )
    }
}

/// One line of the keymap.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    /// Where it applies.
    pub context: Context,
    /// The keys (any of them).
    pub keys: &'static [Key],
    /// What they do.
    pub op: Op,
    /// The help text; empty: not in the help (another row says it).
    pub help: &'static str,
    /// The keys as the help writes them, where not one by one: `1 … 9`
    /// for a row that speaks for nine.
    pub help_keys: Option<&'static str>,
    /// Whether what it does lasts as long as its key is held: let go of,
    /// the key is [`Action::Released`](crate::core::Action::Released) of
    /// it (whatever context is in force then).
    pub held: bool,
}

impl Binding {
    /// The row with its keys written `keys` in the help.
    pub const fn said(self, keys: &'static str) -> Binding {
        Binding {
            help_keys: Some(keys),
            ..self
        }
    }

    /// The row held: what it does lasts as long as its key.
    pub const fn held(self) -> Binding {
        Binding { held: true, ..self }
    }

    /// Its line in the help (the keys, what they do), if it has one.
    pub fn help_line(&self) -> Option<(String, &'static str)> {
        let keys = self
            .help_keys
            .map_or_else(|| keys_label(self), str::to_owned);
        (!self.help.is_empty()).then_some((keys, self.help))
    }
}

/// The contexts searched after a place's own: anywhere's, playing's.
pub const ANYWHERE: [Context; 2] = [Context::Global, Context::Playback];

/// The contexts in force, most specific first.
pub fn contexts(state: &State) -> Vec<Context> {
    if state.palette.is_some() {
        return vec![Context::Palette];
    }
    // A place's own, then anywhere's.
    let anywhere = |own: &[Context]| {
        let mut all = own.to_vec();
        all.extend(ANYWHERE);
        all
    };
    match &state.screen {
        Screen::Main if state.browser.filter.as_ref().is_some_and(|f| f.editing) => {
            vec![Context::Filter]
        }
        Screen::Main => match state.side {
            Side::Transition if state.transition.draft.edit.is_some() => {
                anywhere(&[Context::Fades])
            }
            Side::Transition => anywhere(&[Context::Transition]),
            Side::Karaoke if state.karaoke.sync.is_some() => anywhere(&[Context::Sync]),
            Side::Karaoke => anywhere(&[Context::Karaoke]),
            Side::Mixer => anywhere(&[Context::Desk]),
            Side::Synth => anywhere(&[Context::Synth]),
            Side::Shader if crate::core::shader_tab::searching(state) => {
                vec![Context::ShaderSearch]
            }
            Side::Shader => anywhere(&[Context::Shader]),
            Side::Queue => anywhere(&[Context::Queue, Context::List]),
            Side::Browser => anywhere(&[Context::List]),
        },
        Screen::NowPlaying => anywhere(&[Context::NowPlaying]),
        Screen::Settings if state.settings.folders.is_some() => anywhere(&[Context::FolderList]),
        Screen::Settings if state.settings.picking => anywhere(&[Context::DeviceList]),
        Screen::Settings => anywhere(&[Context::Settings]),
        Screen::Help if state.help_search.as_ref().is_some_and(|s| s.editing) => {
            vec![Context::HelpSearch]
        }
        Screen::Help => anywhere(&[Context::Help]),
        Screen::Karaoke => anywhere(&[Context::KaraokeMode]),
        Screen::Lyrics(editor) if editor.screen.writing() => vec![Context::LyricsText],
        // Its own keys, and playing's: nothing that leaves it.
        Screen::Lyrics(_) => vec![Context::LyricsEditor, Context::Playback],
    }
}

/// What a key means now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meaning {
    /// An operation.
    Op(Op),
    /// A character to type into the filter or the palette.
    Type(char),
}

/// The keys of the first row of `context` doing `op`: what the screens
/// write on what plays it (none: no row).
pub fn keys_for(context: Context, op: Op) -> &'static [Key] {
    (KEYMAP.iter())
        .find(|b| b.context == context && b.op == op)
        .map_or(&[], |b| b.keys)
}

/// What `key` let go of ends: each held row's op it is a key of, in any
/// context.
pub fn released(key: Key) -> impl Iterator<Item = Op> {
    (KEYMAP.iter())
        .filter(move |b| b.held && b.keys.contains(&key))
        .map(|b| b.op)
}

fn find(context: Context, key: Key) -> Option<Op> {
    KEYMAP
        .iter()
        .find(|binding| binding.context == context && binding.keys.contains(&key))
        .map(|binding| binding.op)
}

/// Whether a key of `context` does `op`.
pub fn binds(context: Context, op: Op) -> bool {
    KEYMAP.iter().any(|b| b.context == context && b.op == op)
}

/// Whether an unmodified `c` jumps to a name in the list (as typing does
/// in a file explorer) rather than running its command: on the main
/// screen, in the browser and in the queue, lowercase letters, digits and
/// the other name characters do; capitals, `,`, `/`, `:` and `?` keep
/// their commands. Space is part of the name while one is being typed
/// (within [`TYPE_AHEAD_MS`] of the last letter), and play/pause otherwise.
fn types_ahead(state: &State, c: char) -> bool {
    state.palette.is_none()
        && state.screen == Screen::Main
        && !state.side.wide()
        && !state.browser.filter.as_ref().is_some_and(|f| f.editing)
        && !c.is_uppercase()
        && !c.is_control()
        && (c != ' ' || typing_a_name(state))
        && !matches!(c, ',' | '/' | ':' | '?')
}

/// Whether a name is being typed: something was, less than
/// [`TYPE_AHEAD_MS`] ago.
fn typing_a_name(state: &State) -> bool {
    !state.browser.typed.is_empty()
        && state.now_ms.saturating_sub(state.browser.typed_at) <= TYPE_AHEAD_MS
}

/// What a key does in `state`, as an action.
pub fn action(state: &State, key: Key) -> Option<crate::core::Action> {
    use crate::core::Action;
    match lookup(state, key)? {
        Meaning::Op(op) => Some(Action::Op(op)),
        Meaning::Type(c) => Some(Action::Type(c)),
    }
}

/// What `key` does in `state`, if anything.
pub fn lookup(state: &State, key: Key) -> Option<Meaning> {
    // Ctrl held for a preview: its moves and `enter` are the keys alone
    // (no other binding has ctrl with them).
    let moves = matches!(
        key.code,
        KeyCode::Up
            | KeyCode::Down
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::Enter
    );
    let key = if key.ctrl && moves && crate::core::preview::held(state) {
        Key { ctrl: false, ..key }
    } else {
        key
    };
    if let KeyCode::Char(c) = key.code {
        if !key.ctrl && !key.alt && types_ahead(state, c) {
            return Some(Meaning::Type(c));
        }
    }
    let contexts = contexts(state);
    let typing = contexts.first().is_some_and(|c| c.types());
    for &context in &contexts {
        if let Some(op) = find(context, key) {
            return Some(Meaning::Op(op));
        }
    }
    if typing {
        if key.ctrl {
            if let Some(op) = ANYWHERE.into_iter().find_map(|c| find(c, key)) {
                return Some(Meaning::Op(op));
            }
        }
        if let KeyCode::Char(c) = key.code {
            if !key.ctrl && !key.alt {
                return Some(Meaning::Type(c));
            }
        }
    }
    None
}
