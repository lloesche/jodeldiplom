# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **The cover's subject: a different model.** Tuning around BEN v2
  never satisfied (half a face cut off on Amy Macdonald's "This Is the
  Life", the title torn out of the background as blurred blocks). A
  "most eye-catching thing" model guesses; the plan is to say what the
  subject is: faces and a person detector give the boxes, SAM 2.1
  (Apache-2.0) cuts what they point at, a "stands out" model only for
  covers without a person; text left in the background or filled by a
  real inpainting model (LaMa) instead of the cheap fill. A bake-off on
  the owner's covers decides (kept outside the repository).
- **Story scenes instead of the cover.** The lyrics read for what they
  tell (Gemma 4 E4B, Apache-2.0: text, image and audio in), the song cut
  into scenes by its sections, each scene generated (SDXL-Lightning,
  4 steps, on an 8 GB card, RGBA for sprites; characters from the
  lyrics, not the artist; the cover only when it is striking art for
  the song), taken apart into depth layers and played as parallax (or
  3D) cut on the downbeats. Heavy to prepare, cheap to play. The song's
  look (palette, style, motifs, the effects' flavours) chosen the same
  way. The language model is Gemma 4 E4B at 4 bits (3.4 GB resident;
  E2B wrote half the scenes and made things up), its audio input of no
  use for music. The story comes from the lyrics alone; the look is a
  separate question, from the file's tags, what the model knows of the
  artist and album (and how sure it is), the genre, the era and the
  mood. A Python prototype first (kept outside the repository), then our own
  inference if it is worth it. Also to measure there: the image model at
  fp16, Q8 and Q4; Gemma 4 E2B against E4B.
- **The theme in the window**: Spooky is chosen in the settings file
  only (`karaoke_theme`); a switch for it on the Karaoke tab (after its
  native controls), and the rest of the window's accent following it.
- **The rays on the ladder?** The backdrop's spotlight rays follow how
  well the singing shines (the phrase's tier), everything else that
  builds up follows the multiplier. If they should come in order with
  the rest for every singer, they move to the rungs too.
- **The Synth tab's layers:** the layers a tier or rung switches on are
  shown over the sound, not edited there (they are the tier table's
  rows); if they should be tried out on the tab too, a row's layers
  become user data the way the patches are.

- **Preparing by model, not by song.** The big models don't fit beside
  each other and take about a minute to load, so the work for the songs
  in the queue goes by stage: one queue per model (the language model's
  questions and plans, then the image model's scenes and sprites, then
  depth and the cut), and one GPU scheduler keeping one big model at a
  time, draining its queue for every song before it swaps, with no
  exception for the song playing next. Singing takes the GPU first. The existing workers (stems, the speech model,
  word timing, the cover's vision) under the same scheduler; each stage
  its icon in the queue, lit only when done.

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
cover's vision alike. Measure first (per-kernel GPU time, submissions
and waits per network, bandwidth- or compute-bound), then in order of
what it buys:

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
