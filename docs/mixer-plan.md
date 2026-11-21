# The mixer: where it goes from here

A plan, not yet built. It starts from the mixer as it is (one device, any
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
  the Mix tab's per-stem lanes become fader ramps on the clock.
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

## Risks

- **Clock bridges** are the one hard piece (drift found without audible
  wobble). Not needed before a second output, but the domain boundary is
  there from step 1, since the microphone needs it already.
- **A plug-in system** is the temptation to resist: few node types, no
  decoding in the desk.
- **Decoding and separation cost**: four decks with stems make GPU
  separation of uncached files the bottleneck, not the desk.
