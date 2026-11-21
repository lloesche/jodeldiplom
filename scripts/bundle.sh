#!/usr/bin/env bash
# Build the player, Jodeldiplom (terminal and window), into one bundle. The
# window player is built with its fonts (Sarasa Fixed J, Atkinson
# Hyperlegible Next, Noto Emoji) inside; on macOS it is the app
# "Jodeldiplom.app" instead of a binary in bin. The fonts come from
# player/fonts, or from the folder JODELDIPLOM_FONTS names.
#
#   bash scripts/bundle.sh BUILD_DIRECTORY [linux|macos|windows-x64|windows-arm64]
#
# The bundle lands in BUILD_DIRECTORY/jodeldiplom-PLATFORM: bin/ with the
# executables (or bin/ and the app on macOS), and the licences.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash %s BUILD_DIRECTORY [linux|macos|windows-x64|windows-arm64]\n' "$0" >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p -- "$1"
work=$(cd -- "$1" && pwd)
platform=${2:-linux}
stage=$work/jodeldiplom-$platform
players=(jodeldiplom jodeldiplom-gui)
case "$platform" in
    linux|macos)
        cargo_target=()
        rust_output=$work/rust/release
        ;;
    windows-x64|windows-arm64)
        players=(jodeldiplom.exe jodeldiplom-gui.exe)
        if [[ $platform == windows-x64 ]]; then
            target=x86_64-pc-windows-gnu
            compiler=x86_64-w64-mingw32-gcc
            export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=$compiler
        else
            target=aarch64-pc-windows-gnullvm
            compiler=aarch64-w64-mingw32-clang
            export CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_LINKER=$compiler
            # Resolve the Rust target's unwind library to the static archive.
            unwind_dir=$work/arm64-unwind
            mkdir -p -- "$unwind_dir"
            cp -- "$("$compiler" -print-file-name=libunwind.a)" "$unwind_dir/"
            export CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_RUSTFLAGS="-L native=$unwind_dir"
        fi
        cargo_target=(--target "$target")
        rust_output=$work/rust/$target/release
        ;;
    *)
        printf 'Unknown build platform: %s\n' "$platform" >&2
        exit 2
        ;;
esac

fonts=${JODELDIPLOM_FONTS:-$repo/player/fonts}
for file in SarasaFixedJ-Regular.ttf NotoEmoji.ttf AtkinsonHyperlegibleNext-SemiBold.ttf \
    LICENSE-SarasaGothic.txt LICENSE-NotoEmoji.txt LICENSE-AtkinsonHyperlegibleNext.txt; do
    if [[ ! -f $fonts/$file ]]; then
        printf 'No %s in %s\n' "$file" "$fonts" >&2
        exit 1
    fi
done
# The window player's build embeds the fonts from here (player/build.rs).
export JODELDIPLOM_FONTS=$fonts

cargo build --locked --release --manifest-path "$repo/Cargo.toml" -p jodeldiplom \
    --target-dir "$work/rust" "${cargo_target[@]}"

rm -rf -- "$stage"
mkdir -p -- "$stage/bin"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$repo/player/Cargo.toml" | head -n 1)
if [[ $platform == macos ]]; then
    # The terminal player is a plain binary; the window player is an app.
    cp -- "$rust_output/${players[0]}" "$stage/bin/"
    bash "$repo/player/macos/make-app.sh" "$rust_output/${players[1]}" "$fonts" "$version" "$stage"
    window="$stage/Jodeldiplom.app"
else
    for player in "${players[@]}"; do
        cp -- "$rust_output/$player" "$stage/bin/"
    done
    window=$stage/bin/${players[1]}
fi
cp -- "$repo/LICENSE" "$stage/LICENSE.txt"
cp -- "$repo/player/assets/CascadiaMono-OFL.txt" "$stage/LICENSE-CascadiaMono.txt"
cp -- "$fonts/LICENSE-SarasaGothic.txt" "$fonts/LICENSE-NotoEmoji.txt" \
    "$fonts/LICENSE-AtkinsonHyperlegibleNext.txt" "$stage/"
for file in "$repo"/rhythm/models/LICENSE-*; do
    cp -- "$file" "$stage/$(basename -- "$file").txt"
done
printf 'Player: %s (window: %s)\n' "$stage/bin/${players[0]}" "$window"
