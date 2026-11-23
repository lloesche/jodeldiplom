# The player's fonts

The player (`jodeldiplom`) builds these three fonts in
(`build.rs` embeds them; `JODELDIPLOM_FONTS` names another folder holding
the same files):

| File | Font | From | SHA-256 |
|---|---|---|---|
| `SarasaFixedJ-Regular.ttf` | Sarasa Fixed J 1.0.43, unhinted | `SarasaFixedJ-TTF-Unhinted-1.0.43.7z` of [be5invis/Sarasa-Gothic v1.0.43](https://github.com/be5invis/Sarasa-Gothic/releases/tag/v1.0.43) | `7e4c0cd92d6e057be2dde0cccee704ec42dfdad1a2c6ecb83842235354067957` |
| `NotoEmoji.ttf` | Noto Emoji (variable weight) | `ofl/notoemoji/NotoEmoji[wght].ttf` of [google/fonts at 8b0a1d0](https://github.com/google/fonts/tree/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/notoemoji) | `de6c18832938afc99caf132b39d6a30a19bac7f2e812e28db2535b4608d27551` |
| `AtkinsonHyperlegibleNext-SemiBold.ttf` | Atkinson Hyperlegible Next SemiBold, for lyrics (the Braille Institute's font for legibility) | `fonts/ttf/` of [googlefonts/atkinson-hyperlegible-next at 7925f50](https://github.com/googlefonts/atkinson-hyperlegible-next/tree/7925f50f649b3813257faf2f4c0b381011f434f1/fonts/ttf) | `8e2fff779f35ac25239656bb0f3b2bd35959a2009e3dd21a12141c3b9030362b` |

All three are under the SIL Open Font License 1.1, which lets them be bundled
with software as long as their licences go along:

| File | Licence of | SHA-256 |
|---|---|---|
| `LICENSE-SarasaGothic.txt` | Sarasa Gothic (`LICENSE` at v1.0.43) | `32c932e0dbae4f6e6386964bbc2d04178707665a05ca65cf636241af13d50a53` |
| `LICENSE-NotoEmoji.txt` | Noto Emoji (`OFL.txt` at the commit above) | `500bb1ccf43df7bbb522112f9133a52b16e1c35e809632f5d8609b179152de5b` |
| `LICENSE-AtkinsonHyperlegibleNext.txt` | Atkinson Hyperlegible Next (`OFL.txt` at the commit above) | `aca6a428580965d2297d1b718042dd427c2a9443ece3b0d02d758e161e0c4030` |

The fonts are used unmodified. The macOS app carries their licences
(`macos/make-app.sh`), and so do the bundles `ffmpeg/build.sh` makes.
