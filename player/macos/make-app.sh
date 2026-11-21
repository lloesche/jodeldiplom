#!/usr/bin/env bash
# Put the window player into a macOS app bundle:
#   OUT_DIR/Jodeldiplom.app/Contents/
#     Info.plist                    (from Info.plist here, with VERSION)
#     MacOS/jodeldiplom-gui          (BINARY)
#     Resources/jodeldiplom.icns     (the icon)
#     Resources/LICENSE-*.txt       (the built-in fonts' licences, from FONTS_DIR)
#     Resources/models/             (MODELS_DIR's files: the speech model and recogniser)
# and sign it ad hoc where codesign exists (macOS), so the bundle's
# contents are sealed as macOS expects. Runs anywhere (without signing).
set -euo pipefail

if [[ $# -ne 4 && $# -ne 5 ]]; then
    printf 'Usage: bash %s BINARY FONTS_DIR VERSION OUT_DIR [MODELS_DIR]\n' "$0" >&2
    exit 2
fi
models=${5:-}
binary=$1
fonts=$2
version=$3
here=$(cd -- "$(dirname -- "$0")" && pwd)
assets=$here/../assets
mkdir -p -- "$4"
app=$(cd -- "$4" && pwd)/"Jodeldiplom.app"

rm -rf -- "$app"
mkdir -p -- "$app/Contents/MacOS" "$app/Contents/Resources"
sed "s/@VERSION@/$version/g" "$here/Info.plist" > "$app/Contents/Info.plist"
printf 'APPL????' > "$app/Contents/PkgInfo"
install -m 755 -- "$binary" "$app/Contents/MacOS/jodeldiplom-gui"
install -m 644 -- "$assets/jodeldiplom.icns" "$app/Contents/Resources/"
for file in LICENSE-SarasaGothic.txt LICENSE-NotoEmoji.txt LICENSE-AtkinsonHyperlegibleNext.txt; do
    install -m 644 -- "$fonts/$file" "$app/Contents/Resources/"
done
install -m 644 -- "$assets/CascadiaMono-OFL.txt" "$app/Contents/Resources/LICENSE-CascadiaMono.txt"
if [[ -n $models ]]; then
    mkdir -p -- "$app/Contents/Resources/models"
    for file in "$models"/*.safetensors "$models"/*.jdlasr; do
        [[ -f $file ]] || { printf 'No model %s\n' "$file" >&2; exit 1; }
        # A clone on APFS: no second copy, and at once.
        to="$app/Contents/Resources/models/$(basename -- "$file")"
        cp -c -- "$file" "$to" 2>/dev/null || cp -- "$file" "$to"
        chmod 644 -- "$to"
    done
fi
if command -v codesign > /dev/null; then
    codesign --force --sign - --timestamp=none "$app"
    codesign --verify --strict "$app"
fi
printf 'App: %s\n' "$app"
