#!/usr/bin/env bash
# Build the player, Jodeldiplom, into one bundle. It is built with its
# fonts (Sarasa Fixed J, Atkinson Hyperlegible Next, Noto Emoji, and
# Karaoke mode's Nunito, Fredoka and M PLUS Rounded 1c) inside;
# on macOS it is the app "Jodeldiplom.app" instead of a binary in bin. The
# fonts come from player/fonts, or from the folder JODELDIPLOM_FONTS names.
#
#   bash scripts/bundle.sh BUILD_DIRECTORY [linux|macos|windows-x64]
#
# The bundle lands in BUILD_DIRECTORY/jodeldiplom-PLATFORM: bin/ with the
# executable (the app instead on macOS), and the licences.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash %s BUILD_DIRECTORY [linux|macos|windows-x64]\n' "$0" >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p -- "$1"
work=$(cd -- "$1" && pwd)
platform=${2:-linux}
stage=$work/jodeldiplom-$platform
player=jodeldiplom
case "$platform" in
    linux|macos)
        cargo_target=()
        rust_output=$work/rust/release
        ;;
    windows-x64)
        player=jodeldiplom.exe
        target=x86_64-pc-windows-gnu
        export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
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
    Nunito.ttf Fredoka.ttf MPLUSRounded1c-ExtraBold.ttf \
    LICENSE-SarasaGothic.txt LICENSE-NotoEmoji.txt LICENSE-AtkinsonHyperlegibleNext.txt \
    LICENSE-Nunito.txt LICENSE-Fredoka.txt LICENSE-MPLUSRounded1c.txt; do
    if [[ ! -f $fonts/$file ]]; then
        printf 'No %s in %s\n' "$file" "$fonts" >&2
        exit 1
    fi
done
# The player's build embeds the fonts from here (player/build.rs).
export JODELDIPLOM_FONTS=$fonts

# (An empty array expanded as "${a[@]}" is an unbound variable under set -u
# in the bash macOS has, 3.2: it is expanded only when set.)
cargo build --locked --release --manifest-path "$repo/Cargo.toml" -p jodeldiplom \
    --target-dir "$work/rust" ${cargo_target[@]+"${cargo_target[@]}"}

rm -rf -- "$stage"
mkdir -p -- "$stage"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$repo/player/Cargo.toml" | head -n 1)
if [[ $platform == macos ]]; then
    bash "$repo/player/macos/make-app.sh" "$rust_output/$player" "$fonts" "$version" "$stage"
    built="$stage/Jodeldiplom.app"
else
    mkdir -p -- "$stage/bin"
    cp -- "$rust_output/$player" "$stage/bin/"
    built=$stage/bin/$player
fi
cp -- "$repo/LICENSE" "$stage/LICENSE.txt"
cp -- "$repo/player/assets/CascadiaMono-OFL.txt" "$stage/LICENSE-CascadiaMono.txt"
cp -- "$fonts/LICENSE-SarasaGothic.txt" "$fonts/LICENSE-NotoEmoji.txt" \
    "$fonts/LICENSE-AtkinsonHyperlegibleNext.txt" "$fonts/LICENSE-Nunito.txt" \
    "$fonts/LICENSE-Fredoka.txt" "$fonts/LICENSE-MPLUSRounded1c.txt" "$stage/"
for file in "$repo"/rhythm/models/LICENSE-*; do
    cp -- "$file" "$stage/$(basename -- "$file").txt"
done
printf 'Player: %s\n' "$built"
