//! Two things for the player, `jodeldiplom`:
//!
//! - Its fonts: Sarasa Fixed J, Noto Emoji, Atkinson Hyperlegible Next
//!   (for lyrics), and Karaoke mode's Nunito, Fredoka and M PLUS Rounded
//!   1c, from `fonts` (in the repository; or the folder
//!   `JODELDIPLOM_FONTS` names). `$OUT_DIR/fonts.rs` embeds them with
//!   `include_bytes!` as `EMBEDDED_FONTS`; the binary includes that file
//!   and hands the fonts to the library. Without them the window uses the
//!   embedded Cascadia Mono.
//! - The key of its game sounds kept on disk: `$OUT_DIR/sfx_key.rs`, a
//!   hash of the code that makes them (`src/sfx/`'s `synth.rs`,
//!   `sounds.rs`, `cheers.rs`, `diploma.rs` and `pen.rs`, and what the
//!   diploma's are timed to: the reveal's times and the signature's
//!   stroke, `src/core/singing/`'s `reveal.rs` and `signature.rs`; not how
//!   they are played), so a bank kept by another build of them is made
//!   again (`src/sfx/store.rs`).
//! - Its icon on Windows (GNU and GNU-LLVM targets): `windres` compiles
//!   `assets/jodeldiplom.rc` into an object linked into the executable.
//!   `WINDRES` names another resource compiler; without one the
//!   executable builds without its icon (the window still sets it at run
//!   time).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The fonts' files (see `fonts/README.md`), by their field in
/// `jodeldiplom::gui::fonts::BuiltIn`.
const FONTS: [(&str, &str); 6] = [
    ("main", "SarasaFixedJ-Regular.ttf"),
    ("emoji", "NotoEmoji.ttf"),
    ("lyrics", "AtkinsonHyperlegibleNext-SemiBold.ttf"),
    ("words", "Nunito.ttf"),
    ("shout", "Fredoka.ttf"),
    ("rounded", "MPLUSRounded1c-ExtraBold.ttf"),
];

fn main() {
    embed_fonts();
    sounds_key();
    windows_icon();
}

/// The files whose code makes the game's sounds.
const SOUND_SOURCES: [&str; 7] = [
    "src/sfx/synth.rs",
    "src/sfx/sounds.rs",
    "src/sfx/cheers.rs",
    "src/sfx/diploma.rs",
    "src/sfx/pen.rs",
    "src/core/singing/reveal.rs",
    "src/core/singing/signature.rs",
];

fn sounds_key() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // FNV-1a, 64 bits, over the files as text (line endings as LF).
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for file in SOUND_SOURCES {
        let path = manifest.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path).expect("read a sound's source");
        for b in file.bytes().chain(text.replace("\r\n", "\n").bytes()) {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    let code = format!(
        "/// The key of the sounds this build makes (build.rs).\npub const SOUNDS_KEY: u64 = {hash:#018x};\n"
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("sfx_key.rs");
    std::fs::write(out, code).expect("write sfx_key.rs");
}

fn embed_fonts() {
    println!("cargo:rerun-if-env-changed=JODELDIPLOM_FONTS");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = env::var_os("JODELDIPLOM_FONTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("fonts"));
    let field = |(name, file): &(&str, &str)| -> String {
        let path = dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_file() {
            format!("    {name}: Some(include_bytes!({:?})),\n", absolute(&path))
        } else {
            format!("    {name}: None,\n")
        }
    };
    let fields: String = FONTS.iter().map(field).collect();
    let code = format!(
        "// The window player's fonts found at build time (build.rs).\n\
         const EMBEDDED_FONTS: jodeldiplom::gui::fonts::BuiltIn = jodeldiplom::gui::fonts::BuiltIn {{\n{fields}}};\n"
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("fonts.rs");
    std::fs::write(out, code).expect("write fonts.rs");
}

fn absolute(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn windows_icon() {
    println!("cargo:rerun-if-changed=assets/jodeldiplom.rc");
    println!("cargo:rerun-if-changed=assets/jodeldiplom.ico");
    println!("cargo:rerun-if-env-changed=WINDRES");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:warning=no icon in jodeldiplom.exe: MSVC targets are not set up for it");
        return;
    }
    let windres = env::var("WINDRES").unwrap_or_else(|_| {
        let host = env::var("HOST").unwrap_or_default();
        let target = env::var("TARGET").unwrap_or_default();
        if host == target {
            "windres".into()
        } else {
            // Cross compilers name it after the target: x86_64-w64-mingw32-windres
            // (mingw-w64), aarch64-w64-mingw32-windres (llvm-mingw).
            let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
            format!("{arch}-w64-mingw32-windres")
        }
    });
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("jodeldiplom-icon.o");
    let status = Command::new(&windres)
        .arg("--input")
        .arg(manifest.join("assets/jodeldiplom.rc"))
        .arg("--include-dir")
        .arg(manifest.join("assets"))
        .arg("--output-format=coff")
        .arg("--output")
        .arg(&out)
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-arg-bin=jodeldiplom={}", out.display());
        }
        Ok(s) => println!("cargo:warning=no icon in jodeldiplom.exe: {windres} failed ({s})"),
        Err(e) => println!("cargo:warning=no icon in jodeldiplom.exe: {windres}: {e}"),
    }
}
