# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **The arrow trails in auto mode** (the owner on the Mac, 2026-11-28,
  holding `p`): the pitch arrow seemed to lag behind, at a steady 60 fps
  on the external screen and 120 on the laptop's. Not the frame rate:
  look at the arrow's smoothing, the pitch tracker's window and the
  delay the game allows for the microphone (auto mode sings exactly on
  the notes).
- **Fullscreen does not follow a display change** (the owner on the Mac,
  2026-11-28): the external screen went off mid-song and the game came
  back on the laptop's screen at the external screen's 4K size, not
  refitted; leaving and entering Karaoke mode fixed it. Refit on a scale
  factor or monitor change.
- **The look over the lyrics** (the owner, 2026-11-28): the story's look
  shaders were a bit much and the pills with the words hard to read;
  tune the strength over the lyric band.
- **A storyboard, not a portrait session** (the owner, 2026-11-28, on the
  Echt and "Suddenly I See" pictures: "portrait photography in 9 out of
  10", "same woman, same pose, different background", "all from the same
  angle, same subject at the same size, no plot or action visible"; next,
  once the forgiving reading lands):
  - **The reference carries identity only.** klein copies the reference
    picture's pose, framing and expression. The portrait is a neutral
    reference (neutral expression, plain background, no props, no
    action), the scene takes its head crop (512², as now), and the scene
    prompt asks for this person's face and clothes in a new pose and
    action (find how FLUX.2's guide phrases a reference's role: "subject
    from image 1"), each character's pose, action and framing always
    stated in the scene's prompt.
  - **A plot across the scenes:** a beginning, a turn and an end, so a
    viewer follows what happens from picture to picture; each scene's
    `shows` names an event, something that changes, not a mood.
  - **Storyboard shots:** each scene shows something happening (an event,
    an action, a meeting, in a place, often with others around: passers-by
    and crowds in the prompt); characters doing something, often small in
    the frame or from behind, never posed or looking at the viewer; wide
    shots of the place, action at medium, a detail or an object in
    close-up, a face only now and then (at most one scene in five); many
    scenes with no character at all (the place, an object, a symbol, the
    weather). The prompt's example does this.
  - **Camera variety:** angles (eye level, high, low, bird's-eye, from
    behind, over the shoulder, through a window) as a scene's field or in
    its prompt, the subject's size and place varied (tiny in a wide
    landscape, off-centre, partly in frame, a detail filling it); two
    scenes in a row never share both angle and distance.
  - **Checks on read, never a refusal:** where more than about half the
    scenes put the same character in the prompt, close-ups dominate, or
    one angle and distance does, the follow-up asks once to redraw some
    of them ("these N scenes all show elara posed; rewrite K of them to
    show the place, an object or an action"), and the patch is merged.
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
