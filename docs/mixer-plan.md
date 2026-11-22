# The mixer: where it goes from here

A plan; its first step is built ("Step 1, as built", below). It starts from the mixer as it is (one device, any
number of players on it, gain ramps on one shared clock, bit-perfect
passthrough when one player plays alone at full gain) and grows it into a
desk: devices plugged in as ports, channel strips with filters, buses, a
routing matrix to one or more outputs, and taps.

## The one rule: what runs in real time, and what runs ahead

Two kinds of processing, kept apart:

- **Real time**: on a device's clock, a block every period (10 ms), must
  finish in time, never looks ahead. Gain and fades, filters, the limiter
  and dither, the echo canceller, the voice isolation network (10 ms
  frames), pitch detection, the spectrum.
- **Ahead of time**: may be bursty, may need look-ahead or the whole file,
  may run on the GPU or faster than real time. Decoding, stem separation,
  time-stretch and warp, resampling a file; further out the whole-file
  analysis (beats, lyrics timing, melody, loudness).

Between them is a ring: the player fills it ahead, the desk drains it on the
clock. So:

**The desk holds only real-time processing; a player is everything upstream
of its ring.** A player is a tape machine plugged into a channel.

This decides how fast a control reacts:

- **On the desk** (within the device buffer, about 40 ms): mute, a fader, a
  filter, where the preview goes.
- **In a player** (once its unplayed audio is taken back and made again,
  about 70 ms now): seek, loop, tempo.

Each control is put on the side that gives it the speed it needs.

## The desk

| Real desk | Ours |
|---|---|
| Inputs and outputs on the back | **Ports**: the devices plugged in (speakers, later headphones, the microphone). The desk opens and closes them; nothing else touches audio hardware. |
| Channel strip | **Strip**, one per source (a player's output, a microphone): a chain of filters, a fader whose ramps run on the clock, sends. |
| Groups, aux, cue | **Buses**: main, cue (the preview), later aux. |
| Matrix | **Routes** from buses to outputs. Now: main to the speakers. |
| Insert with a key input | **Sidechain**: a filter with a second input. The echo canceller is one: the microphone is its input, what went to the speakers its sidechain. |
| Direct out to a recorder | **Taps**: listen anywhere without changing anything. The spectrum, the singing recorder, pitch detection. |

How it runs:

- Changes to the routing (a strip added, a route changed) are built off the
  audio thread into a new plan and swapped in whole.
- Parameter changes are messages stamped with a desk frame, as the ramps are
  now.
- The audio thread never allocates, locks or waits on I/O.
- Few, concrete node types (gain, filter, echo canceller, isolation, limiter,
  dither, taps), not a plug-in system. Decoding never goes into the desk.

## What a desk picture hides

### Clocks

A studio runs on one word clock; our devices each have their own crystal and
drift apart by tens of parts per million.

- The desk is split into **clock domains**, one per device, with one master
  (the speakers).
- Audio that crosses domains goes through a resampler that follows the drift
  (a bridge).
- The microphone already lives in a domain of its own; the echo canceller's
  delay finder copes with that today.
- The preview on headphones is the cue bus crossing into the headphones'
  domain through a bridge. That is fine: the cue need not be bit-perfect.

### Bit-perfect

A real desk always processes. Ours, by default, does nothing: the song
reaches the DAC bit for bit, and the device's rate follows the track.

- Per output and per block, the desk checks whether the path from one source
  is **transparent** (unity faders, no filter acting, nothing else heard),
  and then copies that source's samples as they came. Today's "one input
  heard at unity" rule, widened from an input to a path.
- One source per output owns its **format** (now: the main player). Its
  tracks set the device's rate; every other source is resampled inside its
  own player.

## Karaoke as routing

```
mic port ─► strip: echo canceller (sidechain: speaker output) ─► voice isolation
                └─ taps: pitch detector (the game), recorder (Keep)
```

- The microphone is never played. In the desk that is a rule checked when
  the routing is built (no route from a microphone to an output), not a
  property that holds only because no such path happens to exist.
- Hearing a song's singing again is a player playing the recording, on a
  strip of its own, synced to the song (below).
- Live monitoring would be one more route; the real problems there are
  latency and feedback, not routing.

## Stems: one transport, several outputs

Separation gives four signals that are sample-aligned by construction. Four
players would throw that away (four play heads to keep together).

- A player has **one transport** (play head, queue, seek) and **several
  outputs**: `main` (the file as decoded, the bit-perfect path) and, once
  separated, `vocals`, `drums`, `bass`, `other`. Each output can feed a
  strip of its own; they stay together because they share the transport.
- Stem levels move out of the engine onto **strip faders**: muting the
  vocals is instant (now it waits for the player's buffer to play out), and
  the Transition tab's per-stem lanes become fader ramps on the clock.
- **Main or stems**: the stems are made at the separator's rate in floating
  point, so their sum is close to the song but never bit for bit. With every
  stem at full the strip takes `main`; touching any stem crossfades `main`
  into the stems over a few milliseconds (the engine makes this switch
  inside transitions now; it moves to the desk).
- **Separation stays ahead of time**: in the player, ahead of the play head,
  or read from the stem cache.

### Independent stems: a second transport

"Loop the vocals while the rest plays on" is a **second deck** on the same
song: it plays only its `vocals` output, with a loop region, its time tied
to the first deck's beat grid. The stems are cached as files, so that deck
reads the stem file; nothing is separated twice.

## Sync: one clock, a time map per deck

Every deck has a **time map** from its media time to desk frames: straight,
or warped onto a beat grid. Starts, loops and tempo changes are scheduled in
desk frames. One mechanism gives:

- loops of one stem against the others,
- beat-matched transitions,
- several decks,
- the karaoke replay (the kept voice in time with the song).

## Three tiers

```
 analysis (whole file, offline, cached)       beats · lyrics timing · melody · loudness · stems
        │  media + metadata, versioned
        ▼
 players (ahead of time, one transport each)  decode · stems · warp/stretch · resample
        │  outputs: main, vocals, drums, bass, other (through rings)
        ▼
 desk (real time, one per clock domain)
   ports in ─► strips [filters · fader · sends] ─► buses ─► routes ─► ports out
                       ▲ sidechain                  │
                       └──── taps: spectrum · recorder · pitch · echo reference
```

Controllers sit beside these and drive them: the preview (as now), Karaoke,
an auto-DJ that owns two decks and makes the transitions, the UI.

## What changes in the code we have

- **Transitions**: `playback/src/engine/transition.rs` mixes two tracks
  inside one engine with its own adder. It becomes two decks, the second
  started at a desk frame with a warped time map, and fader ramps per stem
  output. The engine shrinks to transport, queue and gapless. Gapless within
  an album stays inside one player: it is a transport's business and stays
  bit-perfect.
- **Stem levels**: the engine's stem summing moves onto strip faders.
- **The microphone**: the microphone, the echo canceller and voice isolation
  (already in the output crate) move under the desk; the app stops opening
  the microphone itself.
- **The preview** behaves as now. The ducking becomes a policy: when the cue
  bus has no output of its own, main is ducked and the cue mixed in.
- **Crates** keep their jobs: output = desk, devices, real-time filters;
  playback = players; rhythm, lyrics, gpu = analysis; player = controllers
  and the routing set up.

## Order of work (each step shippable)

1. **The desk's shape**: strips, a bus, an output, filter chains and taps.
   The microphone becomes a port in its own domain, the echo canceller a
   sidechain filter on its strip, the recorder and the pitch detector taps.
   Nothing heard changes.
2. **Player outputs**: stems as outputs of their own, stem levels on strip
   faders, the main-to-stems crossfade on the desk.
3. **Decks**: time maps and scheduled starts across decks; transitions
   rebuilt on two decks; the engine's own transition mixing deleted. Stem
   loops and the karaoke replay come out of this.
4. **Headphones**: the cue bus to a second output through a clock bridge.
5. **Maybe**: the routing kept as data, shown and edited.

Every step keeps bit-perfect the default and proves it with the same
bit-exact tests, run through the desk.

### Step 1, as built

- **Strips** (`Mixer::strip(name)`): the user's channels, each with a
  level, mute and solo (in place), changed over a 20 ms linear ramp. An
  input plugs into one (`Mixer::input(&strip, role, gain)`) and is heard at
  its controllers' fader times its strip's gain; passthrough is decided on
  that product. The app makes "Player" (the engine) and "Preview". Nothing
  is kept between runs: a strip starts at 0 dB.
- **Meters** (`Mixer::meters()`): the mixer thread takes each strip's
  peak per channel after its gain (its inputs summed), the peak of what
  goes to the device and whether the limiter acted, in atomics; a look
  takes them since the last one (the Mix tab's).
- **`Mixer::desk()`**: the whole desk as the caller sees it, looking
  changes nothing: the device, the clock, passthrough, clips, the strips
  with their inputs (fader gain, heard, format, ring fill), the
  microphone's port.
- **The microphone's port** (`Mixer::set_mic`): the desk taps its device
  for the microphone on each Lead output. Its chain: the clock bridge, the
  echo canceller with the speaker output as its sidechain, then the singer
  isolated; its taps are the voice queue the pitch detector and Keep read.
  It has no strip and no route, so it cannot reach an output. The engine no
  longer knows of it.
- **Bus and route types are deferred to step 4**: there is one bus and one
  output now, so they would have nothing to choose.

### Step 2, as built

- **Groups and changes on a frame** (`Mixer::strip_in(parent, name)`): a
  strip inside another; along a path the gain is every level times
  not-muted, times the solo gate (soloing a strip keeps its path and its
  children, and silences the rest). `Strip::set_level_at` and
  `set_mute_at` land on a frame of a player's output (`Fader::at_stream`).
  A group meters its subtree.
- **Lane inputs** (`Mixer::input_lanes(main, parts, role, gain)`): one
  stream of `main` and four parts (S24-in-32 whatever the device) and a
  control word a frame (`None`, `Present`, `Ending`). Main goes to the
  group, each part to its strip. While nothing is touched the mixer plays
  main, bit for bit; with a part's strip touched it switches to the
  parts' sum over a linear 10 ms crossfade on the control word's frames,
  and back once untouched and the ramps are over. Flush, retract and
  pause act on every lane at once: one transport.
- **The engine writes the parts** (`playback/src/engine/lanes/`,
  `Command::MakeStems(Never | Kept | Separate)`): the stems of the track
  played, from the stem cache or separated, resampled to the output's
  rate and laid on the frames main is heard on, adding nothing of their
  own: stems that add up to the track add up to main within a few 24-bit
  steps at any rate, through seeks, jumps, gapless changes and route
  switches (a real separator's stems add up to the song only roughly,
  about −30 dB off it, and are played as separated); on a device they are
  never waited for.
- **The app plays on the desk** (`player/src/desk_wiring.rs`,
  `core/stems.rs`): the PLAYER group with DRUMS, BASS, VOCALS and OTHER
  inside it. The stem keys mute a stem's strip on the next bar's frame
  (0.1 s ahead at the soonest); `alt+0` puts the strips back, and the file
  is heard bit-perfect again. The stems are made while a strip is touched
  or the Mix, Karaoke or Transition tab shows. The Karaoke voice is the
  vocals lane's fader.
- **The engine's own stem mix is gone** for levels: no stems in the
  settings, no stems route for a track's levels, no stems kept beside the
  file for jumps (the parts' run keeps them), no stem changes timed in the
  engine. A level is heard within the device's buffer, not after the
  engine's half second.
- **Transitions still mix in the engine** until step 3: both tracks on
  their stems, faded by the plan's lanes, with each stem at its gain as the
  desk plays it (`Command::StemMask`, changes landing on their bars); the
  desk hears that as main. The playing track keeps its parts until its move
  onto its stems has met it, and the next track's move back onto its file
  waits for its parts, so a mixer playing the stems hears no jump at
  either end.
- **The first touch heard fast** (`playback/src/engine/lanes/catch_up.rs`):
  stems asked for while none are made start on the frame heard, not the
  one written next; once they are made a little past the write head, the
  output gives back what it has not played (as a jump does) and the same
  main is written again from there with its parts, the control word's
  changes given back with the frames (only frames whose words all said
  the parts are absent; parts a separator made again after falling
  behind are caught up so too). The playing track keeps what the output
  may give back behind what it wrote for that, also without a jump window
  (a jump back within it is met from it). What is guaranteed: every frame
  the output still gives back once the parts are made is written again
  with them. Kept stems are there in 11–17 ms, so from the cache a stem
  muted 0.1 s ahead is heard on its frame on a quick machine (the app's
  lead is 0.1 s then, a second while a transition is armed or heard);
  separated ones once the first chunk is (0.2–0.5 s on an RTX 5080), from
  the frame given back then, so while the stems are not there and none are
  kept the app's lead is 0.75 s (`SEPARATE_SOON`), a bar the stems meet. Nothing is written again over a jump's
  crossfade still held, nor while a route switch, a transition or a
  waiting jump counts on the frames written: a touch then lands as before,
  up to about a second late.
  The parts' resampling (12 % of a core at 384 kHz) and the memory of the
  stems kept are in `PERFORMANCE.md`.
- **Still to come**: checks on real files at a USB DAC's rates.

## Risks

- **Clock bridges** are the one hard piece (drift found without audible
  wobble). Not needed before a second output, but the domain boundary is
  there from step 1, since the microphone needs it already.
- **A plug-in system** is the temptation to resist: few node types, no
  decoding in the desk.
- **Decoding and separation cost**: four decks with stems make GPU
  separation of uncached files the bottleneck, not the desk.
