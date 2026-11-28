# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **A story round starts once the queue has settled** (the owner,
  2026-11-28: the goal is the fewest model swaps, not a story soon). At
  start the queued songs' lyrics come back from the cache one after
  another and the story worker began its round with the first alone,
  so the second song's plan waited behind the first one's drawing (two
  turns of each model where one would do). The worker waits until no
  song has been asked for in a couple of seconds, then plans every song,
  encodes every prompt, draws every picture. A song ready mid-round
  waits for the next round, as agreed.
- **The scenes show what the song is about,** not its lines word for word
  (on the Echt song the prompts described the lyrics almost literally):
  images and metaphors of its story, with the story-shape change below.
- **The story's shape follows the lyrics** (the owner, 2026-11-28; next
  once Gemma's context change lands, as it touches the same planner).
  The story is an alternative to the album cover, the karaoke the game:
  - **Scenes as the story moves, not by the clock.** A new scene when
    the lyrics move (a new place, time, event or feeling), not every
    10–20 s (`SCENE_SECONDS` today forces a count from the length alone:
    18 for a 4-minute song). The model decides how many; the only limits
    are at least one (a story needs a picture) and a ceiling of 20 the
    model is never told (a number in the prompt anchors it): enforced
    when the plan is read, the scenes past the 20th merged into it (it
    shows on to the song's end), so every line keeps a picture.
  - **A repeated section reuses its scene.** A chorus that comes back
    names the earlier scene it shows again, drawing nothing new.
  - **Characters only when the song is about people.** The "I" and the
    "you" need not be people on screen. A story may be places, objects,
    a journey, nature or abstract imagery (a spaceship, a storm), with no
    characters at all; a character need not be human. The prompt's
    example is not anchored on a "narrator".
  - **Crowds and groups** go in a scene's description, never as a
    character (no portrait of a crowd).
  - Most songs then take about 5–10 pictures, some one or two, some 20.
  - **Forgiving reading** (the owner: be very error tolerant, a tiny
    model makes mistakes; never discard its work for a slip). Repair
    what can be, keep everything usable, refuse only when nothing is
    left to draw:
    - line ranges from the scenes' starts alone (each scene runs to the
      next one's start), so gaps, overlaps and a wrong `last_line`
      cannot happen;
    - slightly broken JSON repaired before reading (a trailing comma, a
      missing closing bracket, an unescaped quote, an answer cut off:
      every whole scene kept);
    - an unknown name (preset, style, pass): the correction pass, then
      the nearest valid one;
    - a value out of range clamped; a missing field its default;
    - a scene naming a character no one described: the name dropped,
      the scene kept.
    What needs judgement goes back to the model in the same conversation
    (its prompt and answer still in the cache, so a follow-up costs only
    the new words): only what is wrong, its choices listed, and a small
    patch for answer, merged into the plan (the owner: keep the context
    and fix only what is wrong). Only if the patch is wrong too, the
    nearest valid choice is taken. The same turn can ask a question
    where it clearly helps (a repeated chorus drawn twice: reuse the
    first?). Every repair and every follow-up in the trace, to see what
    to fix in the prompt.
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
