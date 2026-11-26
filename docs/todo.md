# Later

Work agreed on but not started, roughly in the order it was raised.

## Karaoke

- **The cover's subject, what is left.** A text and a face detector now
  clean the cut (on 302 covers from the library, judged by eye: text in
  the subject 72 to 46, subjects missed 37 to 25, no straight edges
  bitten out of people). Left: lettering touching a face stays with it
  (Evanescence's title); logos the text detector misses (Coldplay,
  Bowie); a person turned away with lettering round her has no subject
  (Lily Allen); busy collages pass as one subject (LSD, Les Babacools);
  illustrations and text printed on objects are mostly missed.
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
- **AAC/M4A:** refused only because the decoder's `aac` feature is off;
  turn it on once checked (gapless, seeks, HE-AAC refused).

## Checks on the Mac (the owner)

- The sounds in a real song with the conductor (are they "tinny"?).
- `frame_cost` with every effect wired, against the budget.
- `shine --take` on real takes, Diva earned in a row.
- The mic: unplug it, change its rate, switch mics.

## From the whole-codebase review

Refactors, roughly in the order they unblock each other.

- **The keymap as a table**, and "which song Karaoke means" computed
  once (today four ways; Held, esc, Enter can re-hold the song just
  sung).
- **Settings as one table** (parse, print and merge from it), embedded
  covers keyed by all their bytes, folder listing out of `app.rs`.
- **One atomic writer** for downloads, the mover and setup; the mover's
  "still to move" figure.
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
- **A nightly GPU run** (`shader_check`, `frame_cost` budgets) on the
  owner's machine.
- **The rest of the tests on the wall clock** onto the test's time
  (`SimClock::Manual`, `AppClock::stepped`): the sync's measured delay
  (`player/tests/sync.rs`, a microphone in real time too), the transition
  and stem mutes through the whole player (`app.rs`, `desk.rs`), the
  mixer's own tests in real time (`output/tests/mixer.rs`, `lanes.rs`,
  `link.rs`). A seek's flush first: it waits for the device to take what
  is mixed, which a device on the test's time does only as the test moves.
