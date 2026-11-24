# Jodeldiplom

A music player with a plain file browser instead of a library database, in
a window: `jodeldiplom` (iced on wgpu, its CPU renderer without a GPU;
the screens not ported to iced yet are ratatui widgets rasterised in
pixels, shown inside it); on macOS the app
`Jodeldiplom.app`. Its command-line jobs (`--help`, `--diag`,
`--list-devices`, `--clean-cache`, `--clean-diag`) print to the terminal
it is started from.

What it does:

- **Formats**: FLAC, WAV (RF64), AIFF/AIFF-C, CAF, ALAC (in M4A/MP4 and
  CAF), AAC-LC (in M4A/MP4), MP3, MP2, MP1 and Ogg Vorbis, read with
  symphonia, lossy ones gapless. HE-AAC and Opus are refused with a message
  saying so.
- **Bit-perfect, exclusive output**: the device is opened exclusively
  (WASAPI exclusive mode on Windows, the Core Audio HAL in hog mode on
  macOS, ALSA `hw` on Linux) at the file's own rate, in an integer format
  that holds every bit, so the samples reach the DAC unchanged: no system
  mixer, resampler or software volume in between. The signal path on
  screen says whether a track plays bit-perfect.
- **Gapless**: tracks at one output format play into one stream; lossy
  files have their encoder delay and padding trimmed. A format change
  reopens the device cleanly.
- **Resampling** only where the device takes no rate that carries the file
  bit-perfect: our own rational polyphase FIR in `f64` (≥ 160 dB stopband),
  requantized once to 24 bits with TPDF dither, with optional headroom.
- **A mixer in front of the device**: several players on one device; the
  browser's preview of a song is a second player on it. Alone at gain 1 the
  main player reaches the device bit for bit.
- **Stems**: with a GPU, drums, bass, vocals and other are separated as the
  track plays (our own inference of Mini-BS-RoFormer through `wgpu`), and
  each can be switched off; the stems of a track are kept on disk once
  separated.
- **Beat-matched transitions**: crossfades from one track into the next
  with the beats paired, each stem on its own envelope, by varispeed or a
  time-stretch, on the Transition tab or on the next-track key.
- **Rhythm analysis**: beats, downbeats, tempo and key of every track
  (spectral flux and our own inference of the Beat This! network voting on
  one grid, a meter that listens for the backbeat), and beat grids that can
  be fixed by hand.
- **Lyrics and karaoke**: lyrics looked up on LRCLIB, every line and word
  timed by our own forced alignment (wav2vec2), checked against what is
  sung by a speech recogniser (Qwen3-ASR, packed at 4 bits); a fullscreen
  karaoke mode with a count-in, the voice turned down, and the microphone
  echo-cancelled and scored, never played.

The player's own documentation, keys and settings are in
[player/README.md](player/README.md).

## Building

```sh
cargo build --release -p jodeldiplom
```

builds the player into `target/release`: `jodeldiplom` (`.exe` on
Windows, where a release build has no console window of its own and
attaches to the one it is started from). `make` (or `make jodeldiplom`)
does the same and, on macOS, puts the player into the app
`Jodeldiplom.app` (signed ad hoc, through `player/macos/make-app.sh`);
`make diag` runs the player's `--diag`, `make clean-cache` forgets its
analyses, stems and lyrics.

- **Rust**: the player needs Rust 1.90 or newer (`jodel-gpu` 1.87;
  `durable`, `jodel-output`, `jodel-playback`, `jodel-rhythm` and
  `jodel-lyrics` build with 1.85). The CI builds with Rust 1.100.0.
- **Windows**: the GNU toolchain (`x86_64-pc-windows-gnu`) with MSYS2's
  UCRT64 environment on `PATH` (`C:\msys64\ucrt64\bin`: the linker, and
  `windres`, which builds the player's icon into its executable;
  without it the executable builds without its icon, and the window still
  sets it at run time).
- **Linux**: the ALSA headers (`libasound2-dev` or `alsa-lib-devel`) and
  `pkg-config` to build, the ALSA library (`libasound.so.2`) to run.

The player's fonts (Sarasa Fixed J, Noto Emoji, Atkinson Hyperlegible
Next) are built in from [player/fonts](player/fonts), or from
the folder `JODELDIPLOM_FONTS` names. The speech models are not part of a
build: the player gets them on its first start (its setup screen) and
checks them against the checksums built into it.

`bash scripts/bundle.sh BUILD_DIRECTORY [linux|macos|windows-x64|windows-arm64]`
builds the bundle for one platform in `BUILD_DIRECTORY/jodeldiplom-PLATFORM`:
`bin/jodeldiplom` (on macOS the app `Jodeldiplom.app` instead) and every
licence. From Linux it
cross-builds for Windows: `windows-x64` with MinGW-w64 GCC and the Rust
target `x86_64-pc-windows-gnu`, `windows-arm64` with LLVM-MinGW and
`aarch64-pc-windows-gnullvm`. The CI workflow
(`.github/workflows/build.yml`) checks formatting and lints, runs the tests
(on Linux, macOS and Windows) and builds the bundles for Linux, macOS and
Windows on x64 and ARM64; a tag starting with `v` publishes them as a
release.

## Testing

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
```

No test needs an audio device. The shared fixtures are in
[tests/fixtures](tests/fixtures): small format files (`formats`, made by
its `make.py` with FFmpeg) and two PCM FLAC files (`pcm`, made by its
`make.py`).

## Crates

| Crate | Folder | What it is |
|---|---|---|
| `durable` | [durable](durable) | every file the workspace writes: whole or not at all, made durable (temporary file, fsync, rename, directory fsync), and journals for changes of several files |
| `jodel-output` | [output](output) | bit-perfect exclusive PCM output (WASAPI, Core Audio HAL, ALSA hw), the mixer, microphone input with echo cancellation and singer isolation |
| `jodel-playback` | [playback](playback) | the playback engine: files to the device, the rate policy, the resampler, gapless track changes, seeks, jumps, stems, transitions, events timed to what is heard |
| `jodel-rhythm` | [rhythm](rhythm) | beat, downbeat, tempo, key and pitch analysis, and the stem separator on the CPU, in plain Rust |
| `jodel-tensors` | [tensors](tensors) | a `.safetensors` checkpoint's tensors as f32, for the speech, depth and cut-out models |
| `jodel-gpu` | [gpu](gpu) | the stem separator, the beat network, the speech recogniser and the covers' depth and cut-out models on the GPU, through `wgpu` |
| `jodel-lyrics` | [lyrics](lyrics) | lyrics for karaoke: LRC in and out, tokens for a speech model, forced alignment of words, the speech recogniser and its 4-bit packing |
| `jodeldiplom` | [player](player) | the player: the window on its UI core and widgets, the browser, queue, Transition tab, karaoke, setup |

## Licences

The code is under the MIT licence ([LICENSE](LICENSE)).

Built-in models and fonts keep their own licences, kept beside them:

- the beat, stem, key and pitch networks' weights in
  [rhythm/models](rhythm/models): `LICENSE-beat-this`,
  `LICENSE-mini-bs-roformer`, `LICENSE-musical-key-cnn`, `LICENSE-fcpe`;
- the player's fonts in [player/fonts](player/fonts):
  `LICENSE-SarasaGothic.txt`, `LICENSE-NotoEmoji.txt`,
  `LICENSE-AtkinsonHyperlegibleNext.txt`, and the fallback font Cascadia
  Mono's, [player/assets/CascadiaMono-OFL.txt](player/assets/CascadiaMono-OFL.txt).

The singer-isolation network (`output/models/voice-v6.bin`) and the
recogniser's calibration statistics (`lyrics/models/asr-calibration.stats`)
are our own.

Two speech models are not in the repository; the player's setup downloads
them from Hugging Face and checks their SHA-256s: `facebook/wav2vec2-base-960h`
(Apache 2.0) for the word timing, and `Qwen/Qwen3-ASR-0.6B` (at a fixed
revision) for the speech recogniser, which setup packs at 4 bits into
`qwen3-asr-0.6b-q4.jdlasr` in the data folder. Nor are the two models that
take covers apart for karaoke's background, downloaded the same way at fixed
revisions: `depth-anything/DA3-SMALL` (Depth Anything 3 Small, Apache 2.0)
as `da3-small.safetensors` and `PramaLLC/BEN2` (BEN2 Base, MIT) as
`ben2-base.safetensors`.
