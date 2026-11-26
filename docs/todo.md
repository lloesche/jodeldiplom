# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **Cover separation picks up text.** The subject cut out of the cover
  (the depth of field, the beat's push, the rays fanning out from behind
  it) works well on faces, but the matte often takes some of the cover's
  lettering as part of the subject: a person and half a word come out
  sharp and pushed by the beat, the rest of the word blurred. The cut
  should keep to people and objects, never text (and fall back to no
  subject rather than a wrong one).
- **A lyrics editor.** The words still often miss the voice. An editor
  over the vocal stem's waveform where a word can be moved and made
  longer or shorter by hand; the pitch of the new span is read again by
  the aligner, so the bubbles follow it. The text can be edited too.
- **Words not to sing.** In the same editor, mark words (backing
  vocals, ad-libs) as not sung: no bubbles for them, and the vocal stem
  turned up over them so they are heard. More generally, the vocal
  stem's level can be set across the words of the song.
- **The badge in narrow and upright windows.** The rung's pill and the
  multiplier over the playhead crowd each other below a certain width.
- **The HUD a little smaller** (about 7 %), and maybe the lane lower,
  for more room between the marquee and the top of the window.
- **Empty takes are not saved.** A take with no voice in it (the mic
  muted, nothing sung) should not be kept.
- **Themes** (stage 6 of the tiers plan): the theme hook, and a
  "Spooky" demo theme to prove it (green fire, candy coins, a cold
  grade, fog at the edges).
- **A sung announcer**: the slam's words ("DIVA!", the rung's
  multiplier) also called out, sung in the song's key.
- **The rays on the ladder?** The backdrop's spotlight rays follow how
  well the singing shines (the phrase's tier), everything else that
  builds up follows the multiplier. If they should come in order with
  the rest for every singer, they move to the rungs too.

- **The conductor's follow-ups:** one clock per sound, its cue queue on
  the shared queue type, the Ta-Daa chime as data.
- **A Synth tab**, a workbench like the Transition tab:
  - every sound the game makes, listed by moment (word pops, rung
    stingers, chimes, the payout, the diploma's sounds), each played
    on a click, in the current song's key and tempo or a chosen one;
  - its patch shown and editable: the engine, the envelope, the
    parameters, the layers a tier or rung switches on;
  - the synth's engines played freely (a keyboard on screen and the
    computer's keys), to try sounds out;
  - a patch kept once it sounds right. The patches stay plain data so
    the tab can show them.
- **AAC/M4A:** refused only because the decoder's `aac` feature is off;
  turn it on once checked (gapless, seeks, HE-AAC refused).

## Checks on the Mac (the owner)

- The sounds in a real song with the conductor (are they "tinny"?).
- `frame_cost` with every effect wired, against the budget.
- `shine --take` on real takes, Diva earned in a row.
- The mic: unplug it, change its rate, switch mics.

## From the whole-codebase review

Refactors, roughly in the order they unblock each other.

- **One synthesizer:** the game plays jodel-synth through the conductor;
  delete the old `sfx/synth.rs`, the sound banks and `build.rs`'s sound
  key.
- **`update.rs` split by concern** (browse, queue, settings, work).
- **The keymap as a table**, and "which song Karaoke means" computed
  once (today four ways; Held, esc, Enter can re-hold the song just
  sung).
- **`canvas.rs` in pieces** and the frame plumbing (a layer-only frame
  must keep the stage; lane tiles capped at 16 slots).
- **Settings as one table** (parse, print and merge from it), embedded
  covers keyed by all their bytes, folder listing out of `app.rs`.
- **One atomic writer** for downloads, the mover and setup; the mover's
  "still to move" figure.
- **INTERACTIONS.md honest again:** every test it names exists (a test
  that fails when one is missing), K12 for the listen-only mic.
- **The lyrics crate:** one word model, one cost model; the lyrics
  policy (fitting, preparing the text) moved into the crate.
- **One GPU engine** for every network; a lost GPU retried, the CPU
  fallback said.
- **Output structure:** the mixer's run split, inputs publishing their
  own clock, one backend bootstrap.
- **Targets offer a format, not a set**, then float audio end to end
  (dither and the pack modes gone).
- **The player crate split**, starting with a pure `jodel-karaoke`
  (singing, shine, bubbles, the tier data); pitch tracking off the UI
  thread.
- **A shared numeric base:** the tensors crate plus one DSP leaf (the
  FFTs, windows, WAV) instead of copies.
- **Deterministic tests** on a simulated clock, real-device tests opt in,
  a nightly GPU run (`shader_check`, `frame_cost` budgets) on the owner's
  machine.
- **Docs:** an ARCHITECTURE.md with a glossary, the tier, effects and
  sound plans moved into `docs/`, macOS window behaviour written down.

## The player

- **Get rid of the text UI:** the window still draws every screen but
  the Mixer tab and the Karaoke stage as terminal-style text cells
  rasterised on the CPU, and the core still measures space in cells.
  Still cells: the tabs header, the bottom bar, the command palette, the
  browser and its filter, the queue, the right column (info, links, the
  mini summary), Now Playing, Settings, Help, the Transition and Karaoke
  tabs, the setup screen, and the text and controls over the Karaoke
  stage. Port it to iced screen by screen (the core off cells first,
  then the bottom bar, then browser and queue), ending with the
  terminal library (ratatui), the cell layout, `canvas.rs`, `raster.rs`
  and `view/` deleted (planned as 17 steps).
