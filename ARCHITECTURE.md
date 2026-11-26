# Architecture

Jodeldiplom is a music player for folders of files that turns into a
karaoke game. Every song queued is got ready in the background: its beats
and bars found, its stems separated, its lyrics fetched and timed word by
word, its cover taken apart into layers. It then plays as it is, mixes
into the next song with the beats matched, or is sung: the microphone is
listened to (never played), the singer's pitch held against the original
singer's, and every reward drawn on the GPU and sounded by a synthesizer
of our own on the exact frame of its moment.

This file is the map: which crate holds what, how data moves, which
threads run and what they may not do, where state lives, how to add the
common things, and what the project's own words mean. The details are in
the module docs, which explain each part and its decisions; the rest of
the documentation:

- [README.md](README.md): what the player does, building, testing, the
  crates in one table, the licences.
- [player/README.md](player/README.md): the player as its user sees it:
  keys, screens, settings, folders.
- [player/INTERACTIONS.md](player/INTERACTIONS.md): every rule of how the
  controls interact, each named in the tests that hold it.
- [PERFORMANCE.md](PERFORMANCE.md): what is computed when, the GPU's rules
  and the frame budgets.
- [docs/window.md](docs/window.md): how the window is built.
- [docs/look.md](docs/look.md): the native look.
- [docs/tiers-and-effects.md](docs/tiers-and-effects.md): the tiers, the
  rungs, the effects library, the music clock, the game's sounds and the
  themes.
- [docs/macos-window.md](docs/macos-window.md): what the window does on
  macOS, learnt by experiment.
- [docs/mixer-plan.md](docs/mixer-plan.md): the mixer, step by step as
  built.
- [docs/todo.md](docs/todo.md): the work agreed on for later.

## The crates

```text
jodeldiplom ─────► jodel-playback, jodel-gpu, jodel-rhythm, jodel-lyrics,
                   jodel-output, jodel-synth, jodel-tensors, jodel-dsp,
                   durable
jodel-gpu ───────► jodel-rhythm, jodel-lyrics, jodel-tensors
jodel-playback ──► jodel-output, jodel-rhythm, jodel-dsp, durable
jodel-output ────► jodel-synth, jodel-tensors, jodel-dsp, durable
jodel-lyrics ────► jodel-tensors, jodel-dsp, durable
jodel-rhythm ────► jodel-tensors, jodel-dsp
jodel-synth ─────► jodel-dsp
jodel-rhythm-tools ► jodel-rhythm, jodel-tensors, jodel-dsp, durable
jodel-tensors, jodel-dsp, durable: leaves
```

Each line is a crate and what it depends on. The synth uses `durable` only
in its tests and examples, and `jodel-testkit` (what the tests share) is
a dev-dependency only, of the output, playback, synth and player; there
are no cycles.

| Crate | Folder | Holds | Depends on |
|---|---|---|---|
| `durable` | `durable/` | every file write: whole or not at all, made durable; journals and atomic folders | nothing |
| `jodel-tensors` | `tensors/` | the networks' numeric base: a `.safetensors` checkpoint's tensors as `f32`, half floats, the JSON reader, what the CPU kernels share (`par_chunks`, `threads`, `fast!`) | nothing |
| `jodel-dsp` | `dsp/` | the signal tools: the FFTs (complex and real, `f32` or `f64`), the Hann window and the windowed sinc's parts, the Slaney mel scale, xorshift, FNV-1a, WAV files as bytes, the 24-bit `FULL_SCALE` and `PEAK` | nothing |
| `jodel-testkit` | `testkit/` | what the tests share: `real_time`/`heavy`, `fixture`, `TempDir`, the counting allocator (a dev-dependency only) | nothing |
| `jodel-synth` | `synth/` | the real-time synthesizer: ten engines, a voice host, chorus, delay, hall and dynamics; nothing allocates after construction | `jodel-dsp` |
| `jodel-rhythm` | `rhythm/` | the music analysis on the CPU: the beat grid voted on by several trackers, the meter and bars, the key, a voice's pitch (YIN live, a network for whole songs), the stem separator, waveforms, loudness | `jodel-tensors`, `jodel-dsp` |
| `jodel-rhythm-tools` | `rhythm/tools/` | the tools that pack and measure the pitch network's built-in weights | `jodel-rhythm`, `jodel-tensors`, `jodel-dsp`, `durable` |
| `jodel-lyrics` | `lyrics/` | lyrics as data, LRC in and out, the speech model and the recogniser on the CPU, forced alignment, whether a text fits a recording | `durable`, `jodel-tensors`, `jodel-dsp` |
| `jodel-gpu` | `gpu/` | every network on the GPU through `wgpu`: stems, beats, speech, the recogniser, the covers' depth and cut-out; the GPU shared with the window in short submissions | `jodel-rhythm`, `jodel-lyrics`, `jodel-tensors` |
| `jodel-output` | `output/` | the device, shared through the system's mixer (WASAPI, Core Audio HAL, ALSA); the mixer (the desk); microphone capture, the echo canceller and the singer's isolation | `durable`, `jodel-synth`, `jodel-tensors`, `jodel-dsp` |
| `jodel-playback` | `playback/` | the engine: files decoded, routed, resampled, gapless; seeks, bar jumps, stems as parts of the output, transitions on two heads, the key shift | `durable`, `jodel-output`, `jodel-rhythm`, `jodel-dsp` |
| `jodeldiplom` | `player/` | the application: the pure core, the window, the workers, the stores, the game | all of the above |

Two choices hold the graph in shape. `jodel-gpu` sits above the domain
crates: it runs their models, so it knows them. And `jodel-output` holds
everything that touches an audio device or runs on its clock (capture, the
echo canceller, the desk); decoding and anything that looks ahead stay out
of it, in `jodel-playback`.

## The player's layers

The player crate is one crate in layers. The core is pure; everything with
side effects is outside it and driven by what it returns.

```text
 keys, pointer, widgets ──► gui::window ──► Action ──► core::update(State, Action) ──► Vec<Effect>
                                ▲                         (pure: no IO, no clock,          │
                                │ draws                    no threads, no drawing)         ▼
 gui::screens ◄── view (models) ◄── State                                    app::App carries them out:
 gui::stage, fx, backdrop, lanes                                              engine commands, worker jobs,
                                ▲                                             writes, device switches
                                └──── Action ◄── workers, engine events ◄─────────┘
```

| Layer | Modules | What it is |
|---|---|---|
| Core | `core` | `State` (what the player shows and knows), `Action` (every input: keys, widgets, the engine, the workers), `update` (pure), `Effect` (what to do), the keymap table, `Fit` (the window's size as facts). Sub-cores per feature: `core::desk`, `core::transition`, `core::karaoke`, `core::singing`, `core::shine`, `core::lag`, `core::sync`, `core::preview`, `core::stems`, `core::key`. |
| The game's data | `tiers`, `theme` | the tier table: everything a tier takes, looks like and sounds like, a row each; the themes that override its colours, the effects' looks and the sounds' patches by key. |
| Presentation models | `view` | what each screen says, as plain data (rows, labels, tones, the Karaoke picture, the bubbles' fill). |
| The running player | `app`, `desk_wiring`, `preview`, `sync`, `clock` | `App` holds the `State`, carries out the effects (`app::effects`), pumps the channels, answers the window's frames (`app::frame`); its subsystems own their parts: the microphone and the singing (`app::mic`), the cover heard (`work::art`), the spectrum (`spectrum`), the session kept (`store::session`), a data folder's move (`store::mover`); `desk_wiring` builds the desk and the engine on it; `preview` and `sync` play on it beside the engine. |
| Workers | `work` (`work::rhythm`, `work::lyrics`, `work::cover`, `work::art`, `work::shine`, `work::loudness`, `work::download`), `browser` (details, search), `setup`, `store::mover`, `store::grids` | background threads, each with its queue (`work`: a `Queue` its threads wait on, the `Lanes` of what is wanted now and ahead, a `Job` that tells how it stands and waits for a retry); results come back as `Action`s through an `Outbox` (`work::outbox`). |
| Stores | `store` (`store::cache`, `store::dirs`, `store::mover`, `store::session`, `store::grids`, `store::delays`, `store::scores`, `store::keep`, `store::patches`, `store::stems`), `config`; the workers keep their own (`work::rhythm`, `work::lyrics`, `work::cover`, `work::shine`, `work::loudness`) | what is kept on disk, where, and in which schema. |
| Sounds | `sfx` | the game's moments as sounds: the conductor plans notes on the song's grid and key for the synth on the desk's effects input. |
| The window | `gui` | the iced application (`gui::window`), its screens (`gui::screens`) built from the look's tokens and shared widgets (`gui::look`, `gui::widgets`), the Mixer tab (`gui::mixer`), the lanes (`gui::lanes`), and Karaoke mode on the GPU: the stage (`gui::stage`, its GPU side `gui::stage::gpu` and its shader's shapes `gui::stage::shapes`), the effects (`gui::fx`), the backdrop (`gui::backdrop`), the light (`gui::bloom`), the bursts (`gui::particles`). |
| Platform | `cli`, `diag`, `single`, `icon`, `calendar`, `trace` | the command line, `--diag`, one player at a time, the icon, the date, the timing trace. |

The core never reads a file, the clock or a thread. Time comes in as
`Action::Tick`; the window's size as `Action::Fit`; everything the engine
and the workers learn as their own actions. That is what lets
`INTERACTIONS.md` be tested rule by rule without a window or a device.

## How data moves

### A file played

```text
 file ─► source (symphonia: decode, tags, cover; lossy, float and 32-bit
         samples requantized once to 24 bits)
      ─► route (rate::choose_route: the file's rate against the desk's one format)
      ─► resampler (rational polyphase FIR in f64, only where the rates differ)
      ─► the head on air: as it is, or read along a time map (a transition's
         tempo glide, the key shift)
      ─► an output of lanes: main (the file) and, while a stem is touched,
         the four stems as parts beside it (read from the stems kept, or
         separated live on the GPU by the "jodel-stems" thread)
      ─► the output's ring (half a second)                      [jodel-playback]
 ──────────────────────────────────────────────────────────────────────────────
      ─► the mixer ("jodel-mixer"): each input's lanes on their strips,
         faders and strip ramps on one clock, groups, solo in place,
         automation curves, links between inputs, the effects input's synth
      ─► the device's short ring (40 ms) ─► WASAPI / Core Audio / ALSA
      ─► taps on each buffer the device takes: the spectrum's ring, the
         microphone's echo reference                           [jodel-output]
```

The engine plays on the desk's Lead input; `desk_wiring` sets it up: the
player's strip as a group with a strip per stem inside it, the preview's
strip, the Effects strip for the game's sounds, the microphone's strip
(listened to only) and the master. With every strip as it starts, the main
player alone at gain 1 reaches the device as written.

A transition plays the next track on the player's second head, a second
lane input linked to the first (`Fader::follow`: its frame 0 on the
playing head's stream frame `origin`), both heard through automation
curves the plan sets. At the hand-over the two heads swap roles. The
preview's players are guests on their own strip; their crossfades are
fader ramps on the mixer's clock.

### A song got ready

`Effect::Prepare` sends the queue's songs to three workers at once, the
song playing first:

```text
 rhythm worker ─ decode once ─┬─ mono at 22.05 kHz ─► jodel_rhythm::Analyzer:
                              │    flux and the Beat This! network, each through
                              │    trackers, voted on one grid; meter, bars ─► rhythm/*.txt
                              ├─ mono at 44.1 kHz ─► the key
                              └─ stereo at 44.1 kHz ─► the separator on the GPU
                                   ─► stems kept (stems/), the waveforms and the
                                      melody (rhythm/*.wave)
 lyrics worker ─ the file's tag, a .lrc beside it, the cache, LRCLIB
              ─► the text that fits the recording (the recogniser's hearing)
              ─► the words aligned to the vocal stem (wav2vec2, CTC), the source's
                 line times as the scaffold ─► lyrics/*.timed.lrc
 cover worker ─ depth and cut-out on the GPU, text and faces found on the CPU
              ─► plate, subject, depth, rim, palette ─► covers/*.layers
```

Each answers with `Action::Rhythm`, `Action::Lyrics` or a job's state, and
wakes the window. The GPU work of all three is background work: it gives
way while the window draws something moving (`jodel_gpu::pace`).

In Karaoke mode two more are asked for as a song is sung: the original
singer's phrases measured from the vocal stem (`work::shine`, the shine's
yardstick) and the song's loudness as played (`work::loudness`, which the game's
sounds follow).

### The game loop

```text
 microphone ─► capture thread ("jodel-input-*"), mono, a lock-free ring
            ─► the tap in the device thread: the voice brought over to the
               device's clock, lined up with what the device plays
            ─► "jodel-mic-clean": echo canceller, suppressor, singer isolated
            ─► voice blocks, each with the stream frame it was sung at
 ───────────────────────────────────────────── UI thread, each frame drawn (MicSession::sing)
            ─► VoiceClock: where in the song each block was sung, less the
               singer's delay (core::lag)
            ─► YIN pitch frames (jodel_rhythm::yin)   [or the demo key's perfect singer]
            ─► Singing::hear: each frame against the note (in the singer's key),
               the bubbles filling, words scored, the streak and its rung,
               each phrase's shine and tier
            ─► one stream of moments (core::singing::moments)
                 ├─► the stage: the glow and the effects' board, the bubbles
                 │   bursting, the meter, the badge, the score (gui::stage, gui::fx)
                 └─► the sounds: each moment an event for the conductor
                     (sfx::moments), planned as notes on the song's stream
                     frames, on its beat grid and in its key (sfx::conductor)
                     ─► a fixed-size queue ─► the effects input's synth on the
                        mixer thread ─► the device, on the frame of the moment
```

When a song ends its result is revealed (`core::karaoke::result::Reveal`): one
timeline of beats that the stage draws (the diploma) and the conductor
sounds, by the same clock.

## Threads, and what each may do

| Thread | Crate | What it does | Rules |
|---|---|---|---|
| The UI thread | player | iced's event loop: `Gui::update`, `App::pump`, `core::update`, the pitch tracker, the conductor's planning, the stage's instances, the spectrum's FFT (at most 30 a second) | never waits on a device, the disk or the network; slow updates and frames go into the trace (`JODELDIPLOM_TRACE`) |
| `jodel-mixer` | output | the desk: drains every input a device period at a time, sums, writes the device; renders the effects synth | real time: no allocation, no lock, no wait, no I/O after it is set up; it never waits for a late input |
| `jodel-mixer-aside` | output | what the mixer thread must not do: sets the microphone's tap on the device, drops what the mixer lets go of | ordered, one at a time; joined at quit |
| the device threads (`jodel-output-wasapi`, `jodel-output-alsa`, Core Audio's IOProc) | output | hand each buffer to the device; run the taps | real time: no allocation, no lock; the timeline is theirs, read by others through a sequence lock |
| capture threads (`jodel-input-wasapi`, `jodel-input-alsa`, Core Audio's input) | output | the microphone into a lock-free ring | real time |
| `jodel-mic-clean` | output | the echo canceller, the suppressor, the isolation network | off the device thread; may fall behind, never holds it up |
| `jodel-playback` | playback | the engine: commands in, decode, route, write the output's ring, events out | may block on files; writes ahead of the device by its ring |
| `jodel-stems` | playback | a playing track's stems: read from the file kept, or separated live on the GPU | the playing track's separator never rests for the window |
| `jodel-drop` | playback | drops what the engine is done with (closing a file on a network share can take a while) | |
| `jodel-sync` | player | the sync's clicks, on an output of their own | |
| workers | player | details, rhythm, lyrics, covers, shine, loudness, grids, the stems store's pruning, the mover, setup, the sweep of crash leftovers, device and microphone opening, the art's decode | each owns its queue; answers as `Action`s plus a wake of the window; GPU work marks itself background and gives way to the window |

Real-time rules, and what holds them:

- The mixer and device threads never allocate, lock or wait once they
  run. Counting allocators check it on the named threads:
  `output/tests/device_alloc.rs`, `automation_alloc.rs`,
  `effects_alloc.rs`, and the synth's own `synth/tests/no_alloc.rs`;
  `player/tests/conductor.rs` checks that the conductor and its synth
  allocate nothing once made, through every kind of event.
- What crosses into a real-time thread goes through fixed-size queues
  (the effects input's commands, the strips' changes) or a published
  snapshot (the timeline's position, `output/src/seqlock.rs`). What the
  mixer thread lets go of is dropped on `jodel-mixer-aside`.
- Everything that may be bursty or look ahead (decoding, stems,
  time-stretch, resampling a file) runs in the player, upstream of its
  ring. The desk holds only real-time processing: a player is a tape
  machine plugged into a channel (`docs/mixer-plan.md`).
- The mixer thread runs at real-time priority while it does audio work:
  MMCSS "Pro Audio" on Windows, Mach's time-constraint policy on macOS
  (`output/src/backend/priority.rs`).
- The GPU has one queue for everything: the networks run in submissions
  of a few milliseconds, background work rests while the window animates,
  and the playing track's stems go first (`gpu/src/pace.rs`). The window's
  passes follow `PERFORMANCE.md`: one pipeline per group of shapes, screen
  passes only as strips, the frame budgets measured by `frame_cost`.

## Where state lives

| State | Where | Who changes it |
|---|---|---|
| Everything the screens show and every rule's memory | `core::State`, held by `app::App` | `core::update` only |
| How the window looks frame to frame (pacing, the pointer held, the covers' images, the lanes' tiles, the faders' glide, the stage and its effects' board, the backdrop's visualiser) | `gui::window::Gui` and what it holds (`gui::paint::Paint`, Karaoke mode's `gui::stage_view::StageView`) | the window; no rule of the core depends on it |
| The engine's tracks, heads and routes | the `jodel-playback` thread | its commands (`jodel_playback::Command`); it reports as `Event`s, turned into `Action`s |
| The desk's strips, inputs, gains and rings | the `jodel-mixer` thread | handles (`Strip`, `Fader`, `Effects`); read back as `DeskView` and meters |
| The engine, the mixer and the game's sounds | `desk_wiring::Engine` (inside `App`), with `sfx::Sfx` and its conductor | `App`, as effects come |
| The settings | `config.toml` in the settings folder, read into `Config` (in `State`) | the Settings screen and the keys, saved by `Effect::SaveConfig` |
| The user's own records | the settings folder: the session and saved queues, grids fixed by hand, the singer's delays, kept singing (`recordings/`), the data move's journal | their stores; every older version of them is still read |
| What can be made again | the data folder: the models, analyses and waveforms, stems, lyrics, shine, loudness, covers, and the best scores | the workers, through `durable`; read only in the schema version written (`store/cache.rs`) |

Two folders, decided once at the start (`store::dirs::Folders`) and handed to
everything that reads or writes there:

- **Settings**: small, the user's, where the system keeps an app's
  settings, never moved.
- **Data**: large, could be made again, its parent chosen in the
  settings. Moving it is a journal of folders to move and a file at a
  time (`store::mover`); while it moves, every store writes where things go and
  reads from wherever a file still is.

Every file is written through `durable` (a temporary beside it, fsync,
rename, the folder fsynced); clippy refuses any other write
(`clippy.toml`). The cache contract is in `player/src/store/cache.rs`: a schema
version per kind of file, a sample of each version pinned byte for byte
in `player/tests/schemas/`, content versions (the analysis's, the
aligner's, the covers') apart from the schemas.

## How to add things

Most of what grows is data: a row in a table, read by everything that
needs it. Where that is so, it is said; the tests named check the table.

**A key.** One row in `core::keymap::table::KEYMAP`: where it applies, its
keys, its `Op`, its words in the help. The help screen, the command
palette and every hint a screen shows are written from the table. What the
`Op` does goes in `core::update`'s `apply_op`, or in the module of its
concern that it calls (`core::browse`, `queue`, `settings`, `help`, a
tab's own).

**A setting.** A field of `config::Config` (its doc says what it means) and
one row in the `settings!` table in `config.rs`: its key and the kind of
value. The file is read, written and merged from the table. To show it on
the Settings screen: a `core::SettingsRow` and its row in the screen's
model (`view::settings`).

**A tier.** One row in `tiers::TIERS`: its name, bars, streak (its rung of
the multiplier), colour, meter fill, glow, the effects it lists, the
backdrop's light, its sticker, its sound layers. The meter's marks, the
WGSL arrays, the rungs, the diploma and the sounds all read the table.
`every_row_is_in_order` holds the rows in order and each unlike the one
before. Calibrate its bar on real songs with `examples/shine.rs`. See
`docs/tiers-and-effects.md`.

**An effect.** A module in `gui/fx/` with a `const SPEC: Spec` (its slots
in the stage's draw order, its draw function, its kinds of shape, its WGSL,
its lyric treatment if it is one, its look's four numbers, its shader's
reference in Rust), and one line in the `registry!` of `gui/fx/mod.rs`.
Then a tier's row lists it. `the_registry_is_whole_and_its_names_unique`
checks the registry, `shader_check` holds the WGSL to the Rust reference
on a GPU, `tests/fx_digest.rs` pins the stage's draw list (a change on
purpose takes the new digest from the test's output), and `frame_cost`
measures it against the budget.

**A sound.** A sound is notes of atoms. An atom is one variant in
`atoms!` of `sfx/conductor/palette.rs` and its patch for the synth there.
A sound made of atoms is a function in `sfx/conductor/sounds.rs`. It plays
either as a layer a tier's row lists (`Streak` or `Chime` in
`sfx/conductor/layers.rs`: one variant, its arm, and the rows that want
it) or for a moment of the game (`sfx::moments` makes the moment an
`Event`; the conductor plans it at once, on the next beat or on a
downbeat). The conductor's tests in `sfx/conductor/tests.rs` hold the
levels, the grid and the key.

**A theme.** A `static` `Theme` in a module under `player/src/theme/`
holding only what it changes (tiers' colours by name, the count-in's
accent, effects' looks, atoms' patches, effects held throughout, the
backdrop's grade), and its place in `theme::THEMES`; the setting
`karaoke_theme` finds it by name. `every_theme_overrides_what_there_is_once`
checks its keys. See `docs/tiers-and-effects.md`.

**A screen.** Not one row: a `core::Screen` (or a `core::Side` for the list
pane's tabs), its keys' rows in the keymap with a `Context` of its own,
its model in `view/` (what it says, testable as text), its view in
`gui/screens/` (built from `gui::look` and `gui::widgets`, never its own
numbers), and an arm in `gui::screens::view`. What its size changes for
the keys goes into `core::Fit`. Its interactions are rules in
`INTERACTIONS.md` with simulator tests in `player/tests/screens.rs`. See
`docs/window.md`.

**A kind of file kept.** A `store::cache::Schema` in `store/cache.rs` (in `ALL`), its
sample pinned by `tests/schemas.rs` (`UPDATE_SCHEMAS=1` adds a new
version's), written through `durable`, read and written through the mover
(`store::mover::locate`, `store::mover::current`) if it is in the data folder.

**A network.** Its CPU version in the crate of its domain (`jodel-rhythm`,
`jodel-lyrics`), checked against the published model's outputs; its GPU
version in `jodel-gpu` on the shared engine (`gpu/src/engine`), checked
against the CPU one, its runs paced (`gpu/src/pace.rs`).

## Glossary

The project names things as a person would say them. This is what each
name means in the standard words, and where it is.

| Word | Standard term | Where |
|---|---|---|
| along | playback along a time map (time-warped reading) | `playback/src/engine/along.rs` |
| atom (palette atom) | an instrument preset: one synth patch a sound is made of | `player/src/sfx/conductor/palette.rs` |
| backdrop | the karaoke background visualiser: the cover in depth layers, lit by the stems | `player/src/gui/backdrop` |
| badge | the combo multiplier's display ("Amazing 82 in a row ×16") | `player/src/gui/stage/badge.rs` |
| board | the effects' strength envelopes, eased frame by frame | `player/src/gui/fx/board.rs` |
| bubble | a word's note target: the capsule under a sung word that fills as it is hit | `player/src/core/bubbles.rs`, `player/src/gui/stage/shapes/pop.rs` |
| burst | a particle effect drawn on the GPU from one record | `player/src/gui/particles.rs` |
| conductor | the adaptive sound-effect scheduler: plans the game's sounds on the song's beat grid and key | `player/src/sfx/conductor` |
| cues (rhythm) | onset detection curves: kick, snare, hats, bass and chord changes | `rhythm/src/cues.rs` |
| delay (the singer's) | input latency: from a frame leaving for the device to the microphone hearing it sung | `player/src/core/lag` |
| demo key | autoplay: a perfect singer while `p` is held | `player/src/core/singing/demo.rs` |
| desk | the mixing console: the mixer as the user sees it | `output/src/mixer`, `player/src/core/desk` |
| diploma | the results screen, drawn as a certificate titled "Jodeldiplom" | `player/src/gui/stage/result.rs`, `player/src/core/karaoke/result/verdict.rs` |
| Effects strip | the sound-effects bus | `player/src/desk_wiring.rs` |
| epoch | one opening of the device; a time of another never comes | `output/src/mixer/mod.rs` |
| fit (lyrics) | whether a lyrics text matches a recording, by speech recognition | `lyrics/src/fit.rs` |
| `Fit` (core) | layout metrics the window reports to the core: page sizes, which strips show | `player/src/core/fit.rs` |
| glimpses | sliding-window speech recognition over the voice | `lyrics/src/glimpses.rs` |
| glory | the score meter's shader, with the countdown ring and the progress marks | `player/src/gui/stage/shapes/glory.rs` |
| glow | the combo glow's level, 0 to 1, from the tier the singing is at | `player/src/gui/shine.rs`, `player/src/tiers.rs` |
| grab | pointer capture: a press that holds the pointer until release | `player/src/gui/pointer.rs` |
| guest | a secondary mixer input (a preview), against the Lead | `output/src/mixer/mod.rs` |
| head | a deck: one of the engine's two outputs on the desk; the head on air and the parked one | `playback/src/engine/heads.rs` |
| heard | the playback position as heard, latency-compensated | `player/src/core/mod.rs` (`State::heard`) |
| HUD | the heads-up display over the lane: badge, meter, score | `player/src/gui/stage` |
| lane (output) | one channel group of a multi-lane stream: main or one stem part, sharing one ring | `output/src/lanes.rs` |
| lane (stage) | the note highway the bubbles scroll along to the playhead | `player/src/gui/stage.rs` |
| lanes (tabs) | the Transition and Karaoke tabs' waveform tracks | `player/src/core/lanes.rs`, `player/src/gui/lanes` |
| Lead | the primary mixer input, whose frames the microphone is timed in | `output/src/mixer/mod.rs` |
| link (follow) | sample-accurate sync of one input to another's stream | `output/src/mixer/follow.rs` |
| liquid | the fill shader of a bubble | `player/src/gui/stage/shapes/liquid.rs` |
| marquee | the chasing light bulbs round the lane | `player/src/gui/fx/bulbs.rs` |
| moment | a game event (a word scored, a bubble bursting, a rung reached, a phrase rated) in one ordered stream | `player/src/core/singing/moments.rs` |
| music clock | audio-reactive modulation signals: beat and bar phase, pulses, levels, pitch | `player/src/gui/fx/music.rs` |
| odometer | the rolling score counter | `player/src/gui/stage/score.rs` |
| pacer | frame pacing: one tick per display refresh, sent from inside the redraw | `player/src/gui/pacer.rs` |
| par | the score the original singer's everyday hearings reach on a phrase, the bars' scale | `player/src/core/shine/reference.rs` |
| part | a stem carried beside main in a lane stream | `output/src/lanes.rs`, `playback/src/engine/lanes` |
| pop | a bubble bursting, and its sound; `gui::stage::shapes::pop` draws the bubbles on the stage | `player/src/gui/stage/shapes/pop.rs`, `player/src/sfx/conductor/sounds.rs` |
| reveal | the results screen's animation timeline | `player/src/core/karaoke/result/reveal.rs` |
| rescue shot | forgiveness for one missed word: the next word hit shoots it and keeps the combo | `player/src/core/singing/bursts.rs` |
| rung | a combo multiplier level (×2 to ×32), reached at so many words in a row, named after a tier | `player/src/core/singing/mod.rs` (`rung`), `player/src/tiers.rs` |
| scaffold | the line times of synced lyrics used as soft anchors for the word alignment | `lyrics/src/scaffold.rs` |
| seal | the results screen's sticker and wax-seal shaders | `player/src/gui/stage/shapes/seal.rs` |
| shine | a phrase's vocal similarity score against the original singer's | `player/src/core/shine` |
| slam | a big title word hitting the screen ("DIVA!", "×32") | `player/src/gui/fx/slam.rs` |
| snake | a bubble's tube geometry: a capsule following a word's notes | `player/src/gui/stage/shapes/snake.rs` |
| spread | timing words without a model: line times split by syllables | `lyrics/src/spread.rs` |
| stage | Karaoke mode's GPU renderer: SDF shapes and text from an atlas, a pipeline per group | `player/src/gui/stage.rs`, `player/src/gui/stage/gpu.rs` |
| stinger | the short musical hit when a rung is reached | `player/src/sfx/conductor/sounds.rs` |
| streak | the combo: words hit in a row | `player/src/core/singing/tally.rs`, `player/src/gui/streak.rs` |
| strip | a mixer channel strip (fader, mute, solo, meters); a group is a strip with strips inside it | `output/src/mixer/strip.rs`, `player/src/core/desk` |
| sync | latency calibration: clicks played, the singer's answers timed | `player/src/core/sync`, `player/src/sync.rs` |
| take | a recording of a song sung, kept for tuning the game | `player/src/store/keep.rs` |
| tap | a read-only hook on each buffer the device takes | `output/src/tap.rs` |
| theme | a skin: tier colours, effect looks and sound patches overridden by key (Classic, Spooky) | `player/src/theme.rs` |
| tier | a phrase's rating grade (Good, Great, Awesome, Amazing, Diva) | `player/src/tiers.rs` |
| verdict | the song's medal or the words in its place | `player/src/core/karaoke/result/verdict.rs` |
| vote | beat-tracker ensemble: the grid the candidates agree on, and the agreement as confidence | `rhythm/src/vote.rs` |
| wake | an event-loop wake-up from another thread | `player/src/gui/wake.rs` |
| yardstick | the original singer's phrase measured, the reference a take is held to | `player/src/core/shine/measure.rs` |
