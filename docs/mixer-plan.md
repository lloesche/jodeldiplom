# The mixer: where it goes from here

A plan; its first three steps are built ("Step 1, as built" to "Step 3,
as built", below), and so are "One device, one rate" and "The key shift".
It grows the mixer (one device, any number of players on it, gain ramps
on one shared clock) into a desk: devices plugged in as ports, channel
strips with filters, buses, a routing matrix to one or more outputs, and
taps. As built, the desk opens its one device once, shared, through the
system's mixer at the system's rate, and every input is resampled to that
rate: a song at it reaches the system's mixer bit for bit, one at another
is resampled, and the player calls nothing bit-perfect, since the
system's mixer comes after it ("One device, one rate, as built").

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

A real desk always processes. Ours, by default, does nothing: a song at
the desk's rate reaches the DAC bit for bit. The device runs at one rate
from start to end ("One device, one rate, as built").

- Per output and per block, the desk checks whether the path from one source
  is **transparent** (unity faders, no filter acting, nothing else heard),
  and then copies that source's samples as they came. Today's "one input
  heard at unity" rule, widened from an input to a path.
- The desk owns the **format**: every source (songs, stems, the
  preview, the game's sounds) is resampled to it inside its own player.

## Karaoke as routing

```
mic port ─► strip: echo canceller (sidechain: speaker output) ─► residual suppressor ─► voice isolation
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
  into the stems over a few milliseconds (the engine made this switch
  inside transitions; it is the desk's since step 2).
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

- **Transitions**: `playback/src/engine/transition.rs` mixed two tracks
  inside one engine with its own adder. It becomes two decks, the second
  started at a desk frame with a warped time map, and fader ramps per stem
  output (done in step 3: two heads of one engine, linked on the desk, and
  automation). The engine shrinks to transport, queue and gapless. Gapless within
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
  and the routing set up (amended in step 3: a controller the playback
  tests need lives in playback).

## Order of work (each step shippable)

1. **The desk's shape** (built): strips, a bus, an output, filter chains
   and taps. The microphone becomes a port in its own domain, the echo
   canceller a sidechain filter on its strip, the recorder and the pitch
   detector taps. Nothing heard changes.
2. **Player outputs** (built): stems as outputs of their own, stem levels
   on strip faders, the main-to-stems crossfade on the desk.
3. **Decks** (built): time maps and scheduled starts across decks;
   transitions rebuilt on two decks; the engine's own transition mixing
   deleted. Stem loops and the karaoke replay are to be made of the same
   pieces (a link and automation), later.
4. **Headphones**: the cue bus to a second output through a clock bridge.
5. **Maybe**: the routing kept as data, shown and edited.

Every step keeps the bit-exact tests passing, run through the desk: a
song at the desk's rate, alone at unity, still comes out bit for bit.

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
  takes them since the last one (the Mixer tab's).
- **`Mixer::desk()`**: the whole desk as the caller sees it, looking
  changes nothing: the device, the clock, passthrough, clips, the strips
  with their inputs (fader gain, heard, format, ring fill), the
  microphone's port.
- **The microphone's port** (`Mixer::set_mic`): the desk taps its device
  for the microphone on each Lead output. Its chain: the clock bridge, the
  echo canceller with the speaker output (left and right) as its
  sidechain, the residual suppressor, then the singer isolated; its taps
  are the voice queue the pitch detector and Keep read.
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
  or the Mixer, Karaoke or Transition tab shows. The Karaoke voice is the
  vocals lane's fader.
- **The engine's own stem mix is gone** for levels: no stems in the
  settings, no stems route for a track's levels, no stems kept beside the
  file for jumps (the parts' run keeps them), no stem changes timed in the
  engine. A level is heard within the device's buffer, not after the
  engine's half second.
- **Transitions still mixed in the engine** in this step: both tracks on
  their stems, faded by the plan's lanes, with each stem at its gain as
  the desk played it (a stem mask sent by the app, changes landing on
  their bars); the desk heard that as main. Step 3 moved them onto the
  desk, and the mask, the engine's stems route and every path kept for
  them are gone ("Step 3, as built").
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
  lead is 0.1 s then; a second while a transition was armed or heard,
  until step 3);
  separated ones once the first chunk is (0.2–0.5 s on an RTX 5080), from
  the frame given back then, so while the stems are not there and none are
  kept the app's lead is 0.75 s (`SEPARATE_SOON`), a bar the stems meet. Nothing is written again over a jump's
  crossfade still held, nor while a route switch, a transition or a
  waiting jump counts on the frames written: a touch then lands as before,
  up to about a second late.
  The parts' resampling (12 % of a core at 384 kHz) and the memory of the
  stems kept are in `PERFORMANCE.md`.
- **Still to come**: checks on real files at a USB DAC's rates.

### Step 3, as built

- **One engine, two heads** (`Player::on_desk`, `StemDesk::two_heads`):
  one queue and one cue clock, two lane inputs on the same PLAYER group
  and stem strips, each with its own input and lane faders. The engine
  plays on the head on air; a transition's next track plays on the other
  one, and at the hand-over the two swap (the faders stay each input's).
  Two players behind a wrapper were the other way: they would need queue
  edits translated between them, settings and capabilities refreshed on
  both, a cue command across players and output frames routed per deck,
  several hundred lines more, and the main risk. With one engine the
  queue, the cues, a queue change during a mix and `HoldAtEnd` stay as
  they were. Both heads on the same strips means a stem key, a solo and
  the Karaoke voice keep one target and act on "the music". `TrackStart`,
  `Position` and `TransitionState::Started` say which head their frames
  are of.
- **The desk link** (`Fader::follow(&leader, origin)`,
  `output/src/mixer/follow.rs`): the follower's stream frame 0 lands on
  the leader's stream frame `origin`, from inside a chunk, and from there
  it takes exactly the frames the leader gave, at the same offsets: it
  moves with the leader's stream, never with the device's frames. A
  leader short of frames or paused holds the follower too, and nothing of
  it is lost (a follower of a leader that does not run does not keep the
  device running). A follower that is late drops its frames up to the
  frame due, an absolute stream frame, counted (`Fader::dropped`,
  `InputView::dropped`) and counted as played in its position; the
  leader's are never dropped, so a hiccup or a pause of the playing song
  costs the incoming one nothing it would not have lost alone. The
  follower's `at_stream` is the leader's from `origin` on, exact before it
  has been mixed. Offline no period takes the leader past `origin` before
  the follower has begun, so renders are the same bytes however the
  writers run. A leader at its end lets its followers play on by
  themselves; a flush, a close or a new output on either side unlinks.
- **Automation** (`Fader::automate(Automation)`, `GainCurve`): breakpoint
  curves on the input's own stream frames, one over the whole input, one
  for main alone and one per part, multiplied on top of every fader and
  strip. They move only as the input's stream does, so a pad or a pause
  holds them and a follower's curves keep to its leader. A set is replaced
  or cleared whole, smoothed from the gain heard over 10 ms; a cut inside
  a curve is smoothed so too. A part's curve away from 1 keeps the parts
  heard; once every curve but the input's has ended at exactly 1 the
  switch goes back to main bit for bit and passthrough returns (`settled`
  in `InputView::automation`). Nothing is allocated on the mixer thread
  for them. Automation is a layer of its own rather than ramps on the
  existing faders: the input fader is the preview's duck and the vocals
  lane's is the Karaoke voice, and plan ramps on them would fight.
- **One equal-power curve**: the square of the gain moves in a straight
  line (`sqrt((1 − t)a² + t b²)`), for the desk's `Curve::EqualPower` and
  the plan's `Envelope::gain` alike. It keeps the power for any
  complementary pair, not only 0 and 1, and the Transition tab already
  drew it.
- **Roles** (`Fader::set_role`): a guest made the desk's Lead, or back,
  without the device opening again: its capabilities are asked again and
  the microphone's tap follows the newest Lead. At a hand-over the next
  track's head is made Lead before the old one is a guest, so Karaoke's
  microphone follows the lead song (the only Lead with an output open is
  never demoted).
- **Warped stem lanes** (`Command::Along`, internal: `engine/along.rs`,
  the lanes' `Line` with a map, `Warping` on any number of channels): a
  head read along a time map has its stems read along the map at the
  separator's 44.1 kHz (varispeed, or a stretcher a channel steered onto
  the map) before the parts' resampler, and those are its parts; its main
  is their sum, so main and the parts stay on the same frames for the
  desk's 10 ms switch and nothing is separated twice. Without a
  separator, or once the stems failed, the whole mix goes straight into
  "other" (no transform; the other parts exactly silent). On a device,
  stems not separated in time give way to the whole mix (with a
  `PartsStopped` saying so) rather than an underrun; a next track still
  silent before its window waits for them instead. The move onto the map
  is a segment joined at its frame, main crossfaded once from the route's.
  Once the map runs straight (and a stretch is steered back onto it) the
  head goes back onto its own route at the same format, a switch met
  where the map puts it, and plays the file bit for bit from there; one
  that does not come about is asked again a second later. While a head
  waits for its map or is read along it nothing reroutes it (new settings
  wait), jumps are refused and no transition is set up. Mono files are doubled;
  more than two channels are refused.
- **A transition compiled onto the desk** (`playback/src/engine/
  transition.rs`): armed, it waits until the playing track (A) is 4 s
  before the window. Then the next one (B) opens on the second head at the
  open output's format (its route chosen for that format alone), linked
  to A at `origin`, the window's start less a second (`NEXT_LEAD`), read
  along the plan's map from its first frame, and written ahead on turns of
  its own. With a glide A reads along the glide, and B's map is laid
  through it, so B's beats fall on A's as they are heard. The plan's eight
  lanes become each head's automation on its own stream frames: each
  stem's envelope its part's curve, at A's time heard on each frame
  (through the glide's inverse; a point every 10 ms where the curve is not
  one of the plan's own), and main's curve the lanes' mean, so a head
  heard without parts is never heard at full by mistake; before the
  window A is whole and B silent. A is written up to the window's end and
  no further. There B is made Lead, A's output plays out what it holds and
  closes, and the heads swap; A's end and B's start are cued on one frame,
  on B's clock from then on (A's frames less `origin`). B settles off its
  map onto its own route, its automation goes, and the transition is
  finished. `Next` in the window fades A out over 30 ms from where the
  desk mixes it next (`Fader::mixed_to`) and puts B's curves at their end
  values, and hands over once A is written to its fade's end. A mix given
  up in its window (the queue changed, B failed) fades B out over 30 ms
  and brings A back whole. Pause pauses A, and the link holds B; a seek,
  Play, Previous, Stop or Load drop B; new settings wait for the end. A
  transition on a plain target is `Cancelled("needs a desk")`: the engine
  no longer sums. Deleted with it: the adder and its tanh soft limit, the
  stem gain closures and their limiter, the stems route (`Stage`), the
  stem mask, and the transition-only stem paths. The desk's look-ahead
  limiter and dither take the overlap, so a loud overlap measures slightly
  differently than before.
- **The rate after a mix**: the next track plays at the desk's one
  format, as every track does ("One device, one rate, as built"); the
  format the engine once held after a mix is gone.
- **The app on two heads** (`player/src/desk_wiring.rs`,
  `core/stems.rs`): a stem key lands on its bar on the head on air, 0.1 s
  ahead (0.75 s while the stems are still to be separated), during a mix
  too; the Karaoke voice is on both heads' vocals lanes; the preview ducks
  both; the singing kept around a hand-over is timed on its own head. The
  Mixer tab shows the head on air (LEAD) and, while a mix runs or the old
  head plays out, the other (NEXT), each with its curve's gain while it is
  automated, and any frames dropped on the PLAYER strip.
- **The crate rule, amended**: a controller of players that tests need
  (the transition compiler) lives in playback; the app's policy (the
  presets, the next key's mix, the cues) stays in the player.
- **Still to come**: checks on real files at a USB DAC's rates, through
  the device in real time.

### One device, one rate, as built

- **The desk opens the output device once**, as the mixer starts
  (`Mixer::open`), shared, through the system's mixer (WASAPI shared mode,
  the Core Audio HAL without hog mode, ALSA `default`), at the desk's
  format (`desk_format`): the device's one format (`mix_format`): stereo,
  at the system's mix rate (set in the system's sound settings; the player
  has no rate setting), as a rule in 32-bit float; a test's stand-in
  offers one rate too. Other applications play beside it, and a call's
  screen share hears the music and the karaoke; exclusive access is gone
  with bit-perfect playback at a song's own rate, which bought it nothing
  any more, and the player no longer calls anything bit-perfect: after
  the device comes the system's mixer, which may resample and sets the
  volume. The player's own volume is the gain of a group every strip is
  in (`alt+↑` `alt+↓`, the master's fader keys), at most 0 dB. It holds it open until
  the mixer is dropped: in real time it plays all along, silence where
  nothing sounds, never paused and never closed when an output closes. A
  device not there as the mixer starts has no format yet (`Mixer::format`
  is `None`); a device that fails closes, its outputs failing with its
  error. Either way the next output opened asks the device again what it
  takes, decides the format and opens it (after a failure, another
  epoch: a ramp under way ends on its target, a change for a time of the
  last epoch never comes).
- **Everything is resampled to the desk's rate**: every input is offered
  that one format (`InputTarget::capabilities`), so each player routes to
  it (songs, stems, the preview), the game's sounds are made at it, and
  the microphone's reference is at it. A song at that rate plays bit for
  bit, as before; one at another is resampled, never a reason to open the
  device again. A file of another channel count is made stereo in its
  player as it is read (`playback/src/remix.rs`): mono doubled exactly,
  more channels folded down with the ITU-R BS.775 coefficients (the centre
  and the surrounds at −3 dB, the LFE left out, each side scaled to stay
  within full scale), not lossless then. This ends bit-perfect playback at a song's own rate, by
  the user's choice: no gap, no click and no re-learning between songs.
- **Deleted with it**: the device closing with the last output and opening
  again at the next one's format; the Lead's and the guest's format
  negotiation (the roles now only say whose frames the microphone's tap
  tells); the format the engine held after a mix (`pinned`, the D7
  policy); the effects' keeping the device open, the reopening for them
  and its linger; the rate cap (`max_rate`, its setting and
  `--max-rate`), in the player and in the playback crate, and the setting
  that skipped files at another rate (`--no-resample`).
- **The only reopening** is the user's: another device starts a new mixer
  with the queue and the place carried over. The per-device rate setting
  (`rate."<id>"`, its settings row, `--rate`) is gone: the system sets the
  rate. A change of the device's format in the system's settings fails the
  stream as the device lost (Windows invalidates it; on macOS a listener
  on the device's nominal rate, its being alive and the stream's IO
  format fails it), and the next output opens it at the new one: the
  engine asks its input what it takes for each route (it keeps nothing it
  was told), so the next song is routed to the new rate.
- **The echo canceller** hears one rate all along, so it keeps what it
  has learnt of the room from song to song (it starts again only at
  another rate). Shared, the system adds its own buffering after the
  stream (on Windows about 25 ms, measured); the device clock the
  position comes from runs after it, so the canceller's and the
  karaoke's delays are measured with it in.

### The key shift, as built

- **In the map's pass** (`Command::Key { semitones }`, −6 to +6;
  `playback/src/engine/key.rs`, `engine/shift.rs`, `engine/warping.rs`):
  the head on air reads its stems along a map, the identity on its own
  route, n semitones apart. Every stem but the drums is stretched by
  r = 2^(n/12) (a phase vocoder a channel, at 1/r of the map's speed when
  the map is stretched to) and read r times faster through the map's
  reader, at the stretched frame whose input time the map gives each
  output frame: as long as before, every frequency ×r, on the drums'
  instants. Main is their sum, as along any map; without stems (none, or
  late on a device) the whole mix is shifted as "other". Back at 0 on the
  identity map the head returns to its own route, bit for bit. 6 shifted
  channels and 2 read along cost 6–9 % of a core at 44.1 kHz
  (`PERFORMANCE.md`), before the parts' resampling.
- **Heard at once**: the output gives back what it has not played (as a
  jump; the head keeps the output's ring on a device's output of lanes),
  and the head starts again there in the new key from its stems kept,
  crossfaded as a move onto a map is: 50–60 ms after the command on a
  paced device. A track nothing of which is written takes it from its
  first frame (a count-in's silence first).
- **The rules**: the key stays for the tracks the queue runs into and for
  the track it is on picked again (Karaoke's START replays the song held
  at its start), and one asked while nothing plays is for the track
  picked next; another track picked puts it back at 0. While shifted, new
  settings wait for the key's way back or the next track and jumps are
  refused, each with a notice. No transition
  while the key is shifted, and no key change while one is set up: the
  next track would need the key on its own head, and the playing one its
  glide stacked on it (a key carried to B at the hand-over only if the
  app asks again, later). `Event::Key` says the key asked for,
  `Position::key` the key heard, on its frame.
- **The app**: `-`, `+` and `0` on the Karaoke tab (its Key slider) and
  the Mixer tab (the PLAYER strip shows it); the song's notes and the
  scoring move with the key heard, so singing in the new key scores.

### Later

- The karaoke replay and stem loops, as players of their own on the link
  and automation.
- The key on both heads of a transition (harmonic mixing): B's map in its
  own key, A's glide on top of its key.

## Risks

- **Clock bridges** are the one hard piece (drift found without audible
  wobble). Not needed before a second output, but the domain boundary is
  there from step 1, since the microphone needs it already.
- **A plug-in system** is the temptation to resist: few node types, no
  decoding in the desk.
- **Decoding and separation cost**: four decks with stems make GPU
  separation of uncached files the bottleneck, not the desk.
