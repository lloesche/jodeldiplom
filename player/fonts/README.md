# The player's fonts

The player (`jodeldiplom`) builds these fonts in
(`build.rs` embeds them; `JODELDIPLOM_FONTS` names another folder holding
the same files):

| File | Font | From | SHA-256 |
|---|---|---|---|
| `SarasaFixedJ-Regular.ttf` | Sarasa Fixed J 1.0.43, unhinted | `SarasaFixedJ-TTF-Unhinted-1.0.43.7z` of [be5invis/Sarasa-Gothic v1.0.43](https://github.com/be5invis/Sarasa-Gothic/releases/tag/v1.0.43) | `7e4c0cd92d6e057be2dde0cccee704ec42dfdad1a2c6ecb83842235354067957` |
| `NotoEmoji.ttf` | Noto Emoji (variable weight) | `ofl/notoemoji/NotoEmoji[wght].ttf` of [google/fonts at 8b0a1d0](https://github.com/google/fonts/tree/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/notoemoji) | `de6c18832938afc99caf132b39d6a30a19bac7f2e812e28db2535b4608d27551` |
| `AtkinsonHyperlegibleNext-SemiBold.ttf` | Atkinson Hyperlegible Next SemiBold, for lyrics (the Braille Institute's font for legibility) | `fonts/ttf/` of [googlefonts/atkinson-hyperlegible-next at 7925f50](https://github.com/googlefonts/atkinson-hyperlegible-next/tree/7925f50f649b3813257faf2f4c0b381011f434f1/fonts/ttf) | `8e2fff779f35ac25239656bb0f3b2bd35959a2009e3dd21a12141c3b9030362b` |
| `Nunito.ttf` | Nunito 3.602 (variable weight, 200 to 1000), Karaoke mode's words at 800 and 900 | `ofl/nunito/Nunito[wght].ttf` of [google/fonts at 06bfe0d](https://github.com/google/fonts/tree/06bfe0d62a98375cb09c942fa18e68bd32cff057/ofl/nunito) | `bb55a5ca5c2042335b3991af27c4d0705d0ef41cac6164ac737fd8f2a1e85207` |
| `Fredoka.ttf` | Fredoka 2.001 (variable width and weight, 300 to 700), Karaoke mode's shouts and numbers at 600 and 700, its figures laid out tabular | `ofl/fredoka/Fredoka[wdth,wght].ttf` of [google/fonts at 06bfe0d](https://github.com/google/fonts/tree/06bfe0d62a98375cb09c942fa18e68bd32cff057/ofl/fredoka) | `2ba02e68b152868aef9ba28e24b3648c7d457fe6f25c761f2c2c53fb61a73fc8` |
| `MPLUSRounded1c-ExtraBold.ttf` | M PLUS Rounded 1c ExtraBold 1.060, Karaoke mode's Japanese (kana and the kanji it has; Korean falls on to Sarasa) | `ofl/mplusrounded1c/` of [google/fonts at 06bfe0d](https://github.com/google/fonts/tree/06bfe0d62a98375cb09c942fa18e68bd32cff057/ofl/mplusrounded1c) | `990f04475a93ef0c7a3fadbaa3307d449172c75ca7c0f9bd754c614b93190e15` |

All of them are under the SIL Open Font License 1.1, which lets them be
bundled with software as long as their licences go along:

| File | Licence of | SHA-256 |
|---|---|---|
| `LICENSE-SarasaGothic.txt` | Sarasa Gothic (`LICENSE` at v1.0.43) | `32c932e0dbae4f6e6386964bbc2d04178707665a05ca65cf636241af13d50a53` |
| `LICENSE-NotoEmoji.txt` | Noto Emoji (`OFL.txt` at the commit above) | `500bb1ccf43df7bbb522112f9133a52b16e1c35e809632f5d8609b179152de5b` |
| `LICENSE-AtkinsonHyperlegibleNext.txt` | Atkinson Hyperlegible Next (`OFL.txt` at the commit above) | `aca6a428580965d2297d1b718042dd427c2a9443ece3b0d02d758e161e0c4030` |
| `LICENSE-Nunito.txt` | Nunito (`ofl/nunito/OFL.txt` at the commit above) | `580df76c95a1ec5ab878ceb25bb3d85c6a076804e9c970c8c6972aea775fdf65` |
| `LICENSE-Fredoka.txt` | Fredoka (`ofl/fredoka/OFL.txt` at the commit above) | `5c9e7eee5c6b25f4b05b8d53b2e470ea4962f9ced742d044a98f7d95d1375bab` |
| `LICENSE-MPLUSRounded1c.txt` | M PLUS Rounded 1c: google/fonts carries no `OFL.txt` for it (its `METADATA.pb` says OFL); the font's own copyright notice (its name table) over the OFL 1.1 text as google/fonts carries it | `d269c6410fb8afa3a91fce98d1f741255324548cc987a262d5bc2162ab7d325d` |

The fonts are used unmodified. The macOS app carries their licences
(`macos/make-app.sh`), and so do the bundles `ffmpeg/build.sh` makes.
