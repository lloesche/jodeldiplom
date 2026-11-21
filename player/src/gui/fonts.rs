//! The window's fonts.
//!
//! The main font is Sarasa Fixed J (Iosevka's Latin with Source Han Sans
//! for Japanese, Chinese and Korean, each of those exactly two cells wide)
//! when the window binary was built with it; else the embedded Cascadia
//! Mono ([`FONT`]). Characters the main font has no glyph for come from
//! Noto Emoji (monochrome, drawn in the text colour) when built in, then
//! from the system's fonts, as a terminal falls back to other fonts.
//!
//! The window binary embeds Sarasa Fixed and Noto Emoji when they are found
//! at build time (see `build.rs`; they are in `fonts`) and
//! hands them over with [`use_fonts`] before it starts; the terminal binary
//! and the tests do not, so they keep Cascadia Mono.
//!
//! Each system has a list of fonts known to cover what the others lack; a
//! font is read only when a character first needs it, and kept for the
//! rest of the run. On Linux, fontconfig (`fc-match`) is asked for any
//! character none of the listed fonts has.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ab_glyph::{Font, FontRef};

use super::FONT;

/// The fonts the window binary was built with.
#[derive(Clone, Copy, Default)]
struct BuiltIn {
    main: Option<&'static [u8]>,
    emoji: Option<&'static [u8]>,
    lyrics: Option<&'static [u8]>,
}

static BUILT_IN: OnceLock<BuiltIn> = OnceLock::new();

/// Hands the fonts the window binary embeds (Sarasa Fixed, Noto Emoji,
/// Atkinson Hyperlegible Next for lyrics) to the window; call it before the
/// window starts. Fonts that do not parse are left out.
pub fn use_fonts(
    main: Option<&'static [u8]>,
    emoji: Option<&'static [u8]>,
    lyrics: Option<&'static [u8]>,
) {
    let parses = |data: &&'static [u8]| FontRef::try_from_slice(data).is_ok();
    let _ = BUILT_IN.set(BuiltIn {
        main: main.filter(parses),
        emoji: emoji.filter(parses),
        lyrics: lyrics.filter(parses),
    });
}

/// The font lyrics are drawn in where the window draws them large (the
/// Karaoke tab's voice lane, Karaoke mode): the built-in Atkinson
/// Hyperlegible Next, a font made to be read at a glance (proportional,
/// each letter unlike the others); `None` without it (the main font then).
pub fn lyrics_font() -> Option<&'static [u8]> {
    built_in().lyrics
}

fn built_in() -> BuiltIn {
    BUILT_IN.get().copied().unwrap_or_default()
}

/// The main font: the built-in Sarasa Fixed, else Cascadia Mono.
pub fn main_font() -> &'static [u8] {
    built_in().main.unwrap_or(FONT)
}

/// Whether the window draws with the built-in Sarasa Fixed (else Cascadia
/// Mono), and whether it has the built-in Noto Emoji.
pub fn built_in_fonts() -> (bool, bool) {
    let fonts = built_in();
    (fonts.main.is_some(), fonts.emoji.is_some())
}

/// The fonts read so far, by file (`None`: missing or unreadable).
fn loaded() -> &'static Mutex<HashMap<PathBuf, Option<FontRef<'static>>>> {
    static LOADED: OnceLock<Mutex<HashMap<PathBuf, Option<FontRef<'static>>>>> = OnceLock::new();
    LOADED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A font other than the main one that has a glyph for `c`: the built-in
/// emoji font, then the system's.
pub fn font_for(c: char) -> Option<FontRef<'static>> {
    if let Some(emoji) = built_in()
        .emoji
        .and_then(|data| FontRef::try_from_slice(data).ok())
        .filter(|f| f.glyph_id(c).0 != 0)
    {
        return Some(emoji);
    }
    for path in candidates() {
        if let Some(font) = load(path).filter(|f| f.glyph_id(c).0 != 0) {
            return Some(font);
        }
    }
    let path = fontconfig_match(c)?;
    load(&path).filter(|f| f.glyph_id(c).0 != 0)
}

/// The font in `path` (the first face of a collection), read once.
fn load(path: &Path) -> Option<FontRef<'static>> {
    let mut loaded = loaded().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(font) = loaded.get(path) {
        return font.clone();
    }
    let font = std::fs::read(path).ok().and_then(|data| {
        // Kept for the rest of the run: glyphs refer to it.
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        FontRef::try_from_slice_and_index(data, 0).ok()
    });
    loaded.insert(path.to_path_buf(), font.clone());
    font
}

/// The fonts to try, in order, that exist on this system.
fn candidates() -> &'static [PathBuf] {
    static CANDIDATES: OnceLock<Vec<PathBuf>> = OnceLock::new();
    CANDIDATES.get_or_init(|| {
        known_fonts()
            .into_iter()
            .filter(|path| path.is_file())
            .collect()
    })
}

/// Fonts covering CJK text and symbols, the fixed-width ones first (they
/// fit the cell grid best).
#[cfg(windows)]
fn known_fonts() -> Vec<PathBuf> {
    let windows = std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into());
    let fonts = PathBuf::from(windows).join("Fonts");
    [
        "msgothic.ttc", // Japanese, fixed width
        "YuGothM.ttc",  // Japanese
        "msyh.ttc",     // Chinese (simplified)
        "msjh.ttc",     // Chinese (traditional)
        "malgun.ttf",   // Korean
        "simsun.ttc",   // Chinese
        "seguisym.ttf", // symbols
        "Nirmala.ttc",  // Indic scripts
        "segoeui.ttf",  // most other scripts
    ]
    .iter()
    .map(|name| fonts.join(name))
    .collect()
}

#[cfg(target_os = "macos")]
fn known_fonts() -> Vec<PathBuf> {
    [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", // Japanese (Hiragino Sans)
        "/System/Library/Fonts/Hiragino Sans GB.ttc",      // Chinese
        "/System/Library/Fonts/PingFang.ttc",              // Chinese
        "/System/Library/Fonts/STHeiti Light.ttc",         // Chinese
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",      // Korean
        "/System/Library/Fonts/Apple Symbols.ttf",         // symbols
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    .iter()
    .map(PathBuf::from)
    .collect()
}

#[cfg(not(any(windows, target_os = "macos")))]
fn known_fonts() -> Vec<PathBuf> {
    [
        // Noto Sans CJK, where Debian/Ubuntu, Fedora and Arch put it.
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    .iter()
    .map(PathBuf::from)
    .collect()
}

/// The font fontconfig picks for `c` (Linux and other Unix systems), asked
/// once per character.
#[cfg(not(any(windows, target_os = "macos")))]
fn fontconfig_match(c: char) -> Option<PathBuf> {
    static ASKED: OnceLock<Mutex<HashMap<char, Option<PathBuf>>>> = OnceLock::new();
    let asked = ASKED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = asked.lock().unwrap_or_else(|e| e.into_inner()).get(&c) {
        return found.clone();
    }
    let found = std::process::Command::new("fc-match")
        .arg("--format=%{file}")
        .arg(format!(":charset={:x}", u32::from(c)))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|file| PathBuf::from(file.trim()))
        .filter(|path| path.is_file());
    asked
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(c, found.clone());
    found
}

#[cfg(any(windows, target_os = "macos"))]
fn fontconfig_match(_c: char) -> Option<PathBuf> {
    None
}
