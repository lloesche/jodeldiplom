# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **Fullscreen does not follow a display change** (the owner on the Mac,
  2026-11-28): the external screen went off mid-song and the game came
  back on the laptop's screen at the external screen's 4K size, not
  refitted; leaving and entering Karaoke mode fixed it. Refit on a scale
  factor or monitor change.
- **Deiche's tempo needs a tap** (the story test on the Mac, 2026-11-28):
  Kettcar's "Deiche" reads 95 in some sections and 190 in others (7 of
  8 candidates read 190, the snare on every other beat, the meter's two
  levels nearly tied). Tap it, add it to the truth list, refit if needed.
- **The look over the lyrics** (the owner, 2026-11-28): the story's look
  shaders were a bit much and the pills with the words hard to read;
  tune the strength over the lyric band.
- **The song's objects in the game, not in the story:** a few objects
  per song (from the look's motifs) replacing the game's generic
  graphics where it already puts things: the coin out of a word hit,
  the rungs' fountain and Diva's rain, the marquee's bulbs, the
  diploma's border and seal, a burst at a scene's cut. Never drifting
  over the picture (tried: it looks pasted on). Never a square or boxy
  object that looks cut out of something: drawn as one isolated object
  (no background, card, frame or label), glowing ones on black and added
  as light, and refused when its outline is box-like (straight runs
  along its edge, its box mostly filled).
- **The Synth tab's layers:** the layers a tier or rung switches on are
  shown over the sound, not edited there (they are the tier table's
  rows); if they should be tried out on the tab too, a row's layers
  become user data the way the patches are.

- **A theme as one file** (the owner, 2026-11-27): rework theming so a
  theme is a single file (YAML or similar) holding everything, loaded
  from disk instead of compiled in: its assets, its sounds' patches (in
  the Synth tab's patch text), its colours, fire, bulbs, coins, grade
  and effects. Also instructions for the story's language model, so a
  theme can have stories of its own, and shader preferences that change
  the look shaders' inputs for every story (Spooky's foggier, say).
  Today the theme's sounds are already swapped by atom
  (`Theme::sounds`) and the Synth tab's patch files are read over them,
  but the built-in themes are Rust constants and the conductor's
  patterns (when a sound lands, the arpeggios) are code.

## Faster inference on Apple's GPUs (jodel-gpu)

Measured on the owner's M4 Max (Metal, the f32 path), 2026-11-28:

- **Gemma's step, past its one-row matrix multiplies**: those reached the
  70 to 101 tok/s others get on the same chip (74.2 where it was 50.4,
  shaped as llama.cpp's Metal kernels). What is left: the residual steps
  (1.39 s of a 512-token answer, three single-workgroup dispatches a
  layer) and the cost of each of the 564 dispatches a step.
- **Gemma's prompt** on Apple's 8x8 matrices, as klein's multiply and
  attention already are (339 tokens a second at 12K today).

klein is within what others measure on the same chip (about 24 s a
picture; 18 to 27 s at about 1 MP), so nothing is open there.

- **klein's first picture still bursts on the 5080** (2026-11-28, the
  frames probe): five or so frames of 24 ms in a row, once a first
  picture, in 6 of 8 runs with submissions sized by their costs and 1 of
  5 before. Traced in one run to single dispatches on their first use
  running 7 to 17 times their estimate (one slice estimated at 10 ms took
  120 ms, at the decoder's start): no pace sizes one dispatch. Find what
  the first use waits for (the decoder's buffers made resident, a
  pipeline compiled late) and whether its first submissions can touch it
  first. The Mac's frames were clean (klein's first picture 13.2 ms at
  most).
- **Gemma's prompt cut by the pace's correction** (2026-11-28): its
  multiplies and attention are cut at their path's measured rate
  (`llm::plan::cost`), so on a GPU slower than the one measured each
  slice is as much longer, and no pace can split it. A prompt is planned
  for each prompt: cut at the rate over what the prompt's pace has
  learned (its estimates of the arithmetic over their time, one at most),
  with slack in the room (`ContextPlan::peak_bytes`) for the dispatches
  that adds.
- **Checkpoints mapped into memory, not read** (2026-11-28): a story
  load's fresh memory on the processor is the checkpoints' own
  (`SafeTensors::open` reads a whole `.safetensors` file, some 9.4 GB for
  klein's parts; `Gguf::read` gives each of Gemma's tensors a buffer of
  its own, some 5 GB in all), every page zeroed by the system first, while
  the same bytes sit in the file cache too. Mapped, the pages are the
  file's: nothing zeroed, and the system can drop them under pressure.
  `jodel-tensors` depends on nothing, so the mapping is its own
  (`mmap`/`MapViewOfFile`), and `Tensors::raw`'s borrowed bytes come from
  it.

## Checks on the Mac (the owner)

- The sounds in a real song with the conductor (are they "tinny"?).
- The mic: unplug it, change its rate, switch mics.

## From the whole-codebase review

Refactors, roughly in the order they unblock each other.

- **Settings as one table** (parse, print and merge from it), embedded
  covers keyed by all their bytes, folder listing out of `app.rs`.
- **One atomic writer** for downloads, the mover and setup; the mover's
  "still to move" figure.
- **Output structure:** the mixer's run split, inputs publishing their
  own clock, one backend bootstrap.
- **Targets offer a format, not a set**, then float audio end to end
  (dither and the pack modes gone).
- **The player crate split**, starting with a pure `jodel-karaoke`
  (singing, shine, bubbles, the tier data); pitch tracking off the UI
  thread.
- **Layout, view and the iced port's last items (W-L):** `Pane`, the
  view's module split, module grouping; after the Karaoke clean-up.
- **A nightly GPU run** (`shader_check`, `frame_cost` budgets) on the
  owner's machine.
- **The rest of the tests on the wall clock** onto the test's time
  (`SimClock::Manual`, `AppClock::stepped`): the sync's measured delay
  (`player/tests/sync.rs`, a microphone in real time too), the transition
  and stem mutes through the whole player (`app.rs`, `desk.rs`), the
  mixer's own tests in real time (`output/tests/mixer.rs`, `lanes.rs`,
  `link.rs`). A seek's flush first: it waits for the device to take what
  is mixed, which a device on the test's time does only as the test moves.
