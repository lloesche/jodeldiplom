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

- **Gemma's one-row matrix multiplies** (`gemv_f32_q4k` 6.4 s of 12.5 s
  GPU in a 512-token answer, 53 tok/s): the target is the 70 to 101
  tok/s others get on the same chip. Shape them as MLX and llama.cpp's
  Metal kernels do (gpu/README.md, "Rules for kernels").
- **Gemma's prompt** on Apple's 8x8 matrices, as klein's multiply and
  attention already are (339 tokens a second at 12K today).

klein is within what others measure on the same chip (about 24 s a
picture; 18 to 27 s at about 1 MP), so nothing is open there.

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
