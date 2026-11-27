# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **Next, in this order:** the look shaders finished (the rest of the
  passes, hundreds of presets, every one dancing to the song, the
  guards), then the story generator's first version in the player, as
  the prototype's v1: one picture per scene, taken apart by its depth
  map into layers (cut-outs) and moved in parallax, faces held flat.
- **Story scenes instead of the cover** (prototyped, kept outside the
  repository; worth porting as an optional story mode). The language
  model (Gemma 4 E4B at 4 bits, greedy so the same song gives the same
  plan) reads the lyrics alone for the story (characters, scenes by the
  song's sections); then, with that storyboard in its context, the look
  from the file's tags, what it knows of the artist and album, genre,
  era and mood: the art style from a wide list, palette, motifs, and the
  song's look shaders (a preset from the catalogue, knob tweaks within
  range, which music signal drives what, mood tags), its output held to
  the catalogue's schema as it is written. One answer: the look's fields
  come last in the plan's JSON, so every scene is already written when
  the model chooses it. FLUX.2 klein 4B (Apache-2.0, BFL's FP8
  weights, 4 steps; about 3 s a picture, 7.3 GB at its peak, so it fits
  8 GB) draws one whole picture per scene; it won a bake-off against
  SDXL-Lightning and Z-Image Turbo hands down (faces, following the
  scene). Its text encoder is Qwen3-4B (Qwen's FP8), run for all of a
  song's prompts and unloaded before the image model loads. Characters
  from the lyrics get a portrait each, given to the scenes as klein's
  own reference images (no IP-Adapter); the cover as a style reference
  only when it is striking, its lettering painted out first. No face
  redrawing afterwards (tried: it changed good faces and planted
  features on profiles). Distilled, it takes no negative prompt: the
  prompt itself asks for no text, writing or lettering; no text check
  (the owner: what slips through is unreadable anyway). Each scene
  moves by a slow camera push and either a depth
  warp from its depth map (no edge to tear, so stronger than the
  prototype's 1.5 %; the depth held flat over each whole head, found
  from the face and the depth around it so a profile's nose and the
  hair come with it, so a head moves as one piece and never bends)
  or, where its depth splits cleanly (a subject well
  in front of its background), two or three depth layers in parallax
  with their edges feathered and lit into the scene (v1 and v2 of "Into
  the Great Wide Open" show it working, painted styles most of all);
  cut on the downbeats, the look's shaders over it. No second language-model pass to judge the pictures.
  Dropped after trying: every layer drawn apart (a collage, bad faces),
  an upscaling model, Gemma's audio input (useless for music).
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

- **Preparing by model, not by song.** The big models don't fit beside
  each other and take about a minute to load, so the work for the songs
  in the queue goes by stage: one queue per model (the language model's
  questions and plans, then the image model's scenes and sprites, then
  depth and the cut), and one GPU scheduler that, where the models do
  not fit together, keeps one big model at a time and drains its queue
  for every song before it swaps: the language model writes every
  waiting song's storyboard, then the image model draws them all; a song
  queued meanwhile waits until the image model's queue is empty, then
  the language model is loaded again for it. Once every queue is empty
  and has stayed so for a minute, every model is let go, on any card:
  the memory goes back to the game and the other programs. Each queue
  is first in, first out, no exceptions (not even for the song playing
  next). Singing takes the GPU first. How many models stay loaded
  follows the card's memory, read per backend through wgpu's escape
  hatch (wgpu itself does not say): DXGI's budget and usage on DirectX
  12, `VK_EXT_memory_budget` on Vulkan, the device's recommended working
  set on Metal. A model stays resident while what is loaded fits the
  budget less a margin for the window and other programs, and is let go
  only when the next one would not fit (the oldest unused first): on
  16 GB the language model, the encoder and the image model stay loaded
  together, on 8 GB they take turns, slower but the same results.
  Going over must never be found out by failing: Windows pages video
  memory out to system memory silently and everything crawls. Each
  model says what it holds and its peak while it works, so the plan is
  made before loading. The existing workers (stems, the speech model,
  word timing, the scenes' depth and cut-out) under the same scheduler;
  each stage its icon in the queue, lit only when done.
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

## The player

- **Start-up (being done):** the window took about 20 s to show. It
  opens at once on the setup screen, which shows what is loading, and
  goes to the player when it is ready.
- **Ctrl to preview (being done):** right after start nothing played
  until a song had been started and stopped in the queue.
- **AAC/M4A:** refused only because the decoder's `aac` feature is off;
  turn it on once checked (gapless, seeks, HE-AAC refused).

## Faster inference (jodel-gpu)

All our networks run on our own WGSL kernels, so one engine's speed-ups
help the speech model, the stems, the beats, the word timing and the
scenes' depth and cut-out alike. Measure first (per-kernel GPU time,
submissions and waits per network, bandwidth- or compute-bound), then in
order of what it buys:

- **Next, once the story in pre-processing lands: Gemma's context, as
  large as the card allows** (the owner, 2026-11-27: "optimize the heck
  out of this"). Thinking needs room, and today 18,432 positions is the
  most: the key/value cache is one buffer of about 112 KB a position,
  and the largest storage binding is about 2 GB. The sliding-window
  layers keep a ring of their window, so only the global layers grow
  with the context; their cache split across bindings or kept in f16 if
  it still must. The context is sized by the card: 8 GB is the least
  that must work, and a 16 GB card uses its memory for a longer one. The
  plans token for token as before, a test past 32k, and the footprints
  measured again; the thought's room grows to match.
- **Drop DirectX 12 entirely** (in the same piece of work): Vulkan on
  Windows and Linux, Metal on the Mac. On a card with Vulkan the models
  run on it already and the memory budget comes from
  `VK_EXT_memory_budget`; the DirectX 12 adapter and its DXGI figures
  only feed the diagnostic report. First print each backend's real
  limits, to know whether the 2 GB binding is the driver's or `wgpu`'s.
- **Q4 matmuls on int8 dot products:** two nibbles a word into int8
  lanes (`w & 0x0F0F0F0F`, `(w >> 4) & 0x0F0F0F0F`), the inputs rounded
  to int8 per block of 32 in the same even/odd order with a per-block
  sum for the zero point (as llama.cpp's q8_1), four multiply-adds an
  instruction with `dot4I8Packed` (DP4a on NVIDIA, AMD, Intel; Metal
  emulates it, so the path is picked per adapter).
- **f16** for the activations where accuracy allows (twice the rate on
  Apple and AMD).
- **Matmul tiling:** workgroup-memory tiles, several outputs a thread,
  vec4 loads, subgroup reductions; a GEMV kernel for one token at a time
  and a GEMM one for batches.
- **Fusion:** bias, activation, residual and norm in the matmul's tail;
  flash attention (online softmax) for the long sequences.
- **Fewer round trips:** the decoder's loop on the GPU (argmax there,
  the tokens read back in batches), more dispatches a submission, no
  per-run buffers or staging, bind groups kept.
- **Workgroup sizes** per vendor, tuned once and kept; background work
  flat out while the window rests.

Being measured and done; the Mac's numbers to follow.

## Checks on the Mac (the owner)

- The sounds in a real song with the conductor (are they "tinny"?).
- `frame_cost` with every effect wired, against the budget.
- `shine --take` on real takes, Diva earned in a row.
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
