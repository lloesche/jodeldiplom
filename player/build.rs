//! Two things for the player, `jodeldiplom`:
//!
//! - Its fonts: Sarasa Fixed J, Noto Emoji and Atkinson Hyperlegible Next
//!   (for lyrics), from `fonts` (in the repository; or the folder
//!   `JODELDIPLOM_FONTS` names). `$OUT_DIR/fonts.rs` embeds them with
//!   `include_bytes!`; the binary includes that file and hands the fonts
//!   to the library. Without them the window uses the embedded Cascadia
//!   Mono.
//! - Its icon on Windows (GNU and GNU-LLVM targets): `windres` compiles
//!   `assets/jodeldiplom.rc` into an object linked into the executable.
//!   `WINDRES` names another resource compiler; without one the
//!   executable builds without its icon (the window still sets it at run
//!   time).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The fonts' files (see `fonts/README.md`).
const MAIN_FONT: &str = "SarasaFixedJ-Regular.ttf";
const EMOJI_FONT: &str = "NotoEmoji.ttf";
const LYRICS_FONT: &str = "AtkinsonHyperlegibleNext-SemiBold.ttf";

fn main() {
    embed_fonts();
    windows_icon();
}

fn embed_fonts() {
    println!("cargo:rerun-if-env-changed=JODELDIPLOM_FONTS");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = env::var_os("JODELDIPLOM_FONTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("fonts"));
    let constant = |name: &str, file: &str| -> String {
        let path = dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_file() {
            format!(
                "const {name}: Option<&[u8]> = Some(include_bytes!({:?}));\n",
                absolute(&path)
            )
        } else {
            format!("const {name}: Option<&[u8]> = None;\n")
        }
    };
    let code = format!(
        "// The window player's fonts found at build time (build.rs).\n{}{}{}",
        constant("EMBEDDED_MAIN_FONT", MAIN_FONT),
        constant("EMBEDDED_EMOJI_FONT", EMOJI_FONT),
        constant("EMBEDDED_LYRICS_FONT", LYRICS_FONT),
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
