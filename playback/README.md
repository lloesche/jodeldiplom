# jodel-playback

The playback engine between files and [`jodel-output`](../output): FLAC,
WAV, AIFF, CAF and ALAC files to bit-perfect, exclusive device output, and AAC, MP3 (MP2, MP1) and Ogg Vorbis decoded to
24 bits, gapless. It has no UI; a player puts its own on top.

## Modules

- `source`: opens files with symphonia 0.6 and streams interleaved
  24-bit samples (right-aligned in `i32`, as the output takes them). `Source::seek` lands on an exact frame. `TrackInfo` holds the
  container and codec, rate, channels, bits, the bit rate of lossy files,
  length, tags (title, artist, album, track number, `ORIGINALSAMPLERATE`,
  and every tag as read) and the cover (the embedded front cover, else
  `cover.*`, `folder.*` or `front.*` in the folder, as raw bytes and a MIME
  type). `probe_header` reads the same from the headers alone (no audio is
  decoded, no folder cover).

  | Container | Codecs | Tags, cover |
  |---|---|---|
  | FLAC | FLAC | Vorbis comments, picture blocks |
  | WAV (RF64), AIFF/AIFF-C, CAF | PCM (integer, float); CAF also ALAC | RIFF INFO, ID3v2 |
  | MP4/M4A | ALAC, AAC-LC (HE-AAC is reported as not supported) | MP4 atoms, `covr` |
  | MPEG audio | MP3, MP2, MP1 | ID3v2 (`APIC`), ID3v1, APE |
  | Ogg | Vorbis | Vorbis comments, `METADATA_BLOCK_PICTURE` |

  Symphonia features: `flac`, `wav`, `pcm`, `aiff`, `caf`, `alac`, `isomp4`,
  `aac`, `mp1`, `mp2`, `mp3`, `ogg`, `vorbis`, `id3v1`, `id3v2`. Symphonia
  0.6 has no Opus decoder; Opus files are refused with a message saying so.

  AAC (`aac`): symphonia decodes AAC-LC only, mono or stereo, in
  1024-frame packets. `aac::config` reads the stream's
  AudioSpecificConfig itself and refuses, naming the reason, HE-AAC and
  HE-AAC v2 (SBR and PS, signalled hierarchically or by the sync
  extensions after an LC header, which symphonia would take for LC), HE-AAC
  whose SBR is not signalled at all but whose track runs at twice the LC
  core's rate, the other profiles (Main, LTP, LD, ELD, USAC, ...),
  more than two channels, program config elements and 960-frame packets:
  such a file is never played at its core rate without its upper band. An
  HE-AAC file that signals nothing and whose container states the core
  rate cannot be told from AAC-LC without decoding its SBR data; it plays as
  its LC core. The rate is the AudioSpecificConfig's (what the decoder
  gives; every decoded packet, of any codec, is checked against the rate
  the track states), the bit rate the
  average the `esds` descriptor states. AAC plays from MP4 only (an ADTS
  `.aac` stream or AAC in CAF is refused).

  Samples: integer sources of at most 24 bits pass exactly (16 bits shifted
  up). Lossy decoders, floating-point PCM and integer PCM above 24 bits are
  requantized to 24 bits once, with the resampler's TPDF requantizer
  (`Requantizer`), and the samples that clip are counted
  (`Source::clipped`, reported as `Event::Clipped`). A packet whose samples
  all lie on the 24-bit grid already (32-bit files holding 24-bit audio,
  digital silence) passes exactly. Each packet's dither is seeded from its
  first frame index, so a seek gives the same samples as reading from the
  start. When the route resamples, these samples go into the resampler
  unquantized (`Source::read_unquantized`, `Resampler::process_float`) and
  are quantized once, after it. Files above 24 bits and lossy files are
  never routed as lossless.

  Gapless: the decoders drop the encoder delay and padding the container
  states (an MP3 file's LAME/Xing header, an Ogg stream's granule
  positions), so a lossy track's length is its original sample count and
  its first sample is the first one encoded. AAC in MP4 is trimmed by the
  source itself (the demuxer applies no edit list): the delay and length
  come from the iTunes `iTunSMPB` tag when it fits in the track, else from
  the track's edit list (`elst`, one edit after any empty ones; within a
  tick of a coarse movie timescale the track's own length is taken), else
  every decoded frame plays. Lossy seeks decode from 8,192 frames before
  the target (the MP3 bit reservoir, the Vorbis window, the AAC overlap;
  for AAC into the encoder delay) and drop what comes before it; a
  container that cannot seek or read on from where it landed is decoded
  from the start instead. Every seek starts a new decoder, so a seek to a
  frame always gives the same samples (an AAC stream's noise-substituted
  (PNS) bands the same noise, though not the noise a decode from the start
  has there). Seeks are
  sample-accurate on every format.
- `rate`: `choose_route(file, capabilities, settings)` is a pure function
  from the file, the device's exclusive capability matrix and the settings
  (rate cap, resampling, headroom) to a `Route` (output
  rate, channels, format preference, resample step) or a `NoRoute` with a
  sentence saying why nothing fits.

  | File | Route |
  |---|---|
  | lossless PCM (FLAC, WAV, AIFF, CAF, ALAC) | its own rate and channels, untouched |
  | lossy (MP3, AAC, Vorbis, ...) | decoded to 24 bits, its own rate and channels; never lossless |

  The route fits when its rate is within the cap and the device takes it
  in a format holding every bit of the file (24-bit formats for lossy and
  floating-point files). Such a bit-perfect route always wins.

  When none fits and `Settings::resample` is on (the default), the route
  gains a `Resample` step to a rate the device takes in a 24-bit format
  within the cap, and is no longer lossless: the lowest such rate above
  the file's own, else the highest one below.

  A route that is not bit-perfect carries the headroom setting
  (`Settings::headroom`, none, 0.5, 1, 2 or 3 dB; `Route::headroom`) at
  its one requantization to 24 bits: a resampled route's resampler scales
  its input by `10^(−dB/20)` in its `f64` computation, and a lossy or
  floating-point file played at its own rate is scaled by the source
  (`Source::set_gain`) before it is turned into 24 bits, so overs of a
  near-full-scale signal need not clip, with no extra rounding. A lossy
  file's route always has at least `LOSSY_HEADROOM` (1.5 dB): an MP3 of a
  loud master decodes up to about 1 dB over full scale. The path says so:
  `44.1 kHz decoded from MP3 (-1.5 dB)`. Bit-perfect routes never carry
  any.

  | Device | File | Route |
  |---|---|---|
  | GC7: 48/96/192 kHz, S16/S24_3 | PCM 44.1 / 88.2 / 176.4 kHz | → 48 / 96 / 192 kHz (160/147) |
  | GC7 | PCM 176.4 kHz, capped at 96 kHz | → 96 kHz (80/147) |
  | GC7 | PCM 352.8 / 384 kHz | → 192 kHz (80/147, 1/2) |
  | GC7 | PCM 8 channels, 44.1 / 192 kHz | → 48 / 96 kHz (8 channels exist at 48 and 96 kHz only) |
  | HDMI: 44.1/48 kHz | PCM 88.2 / 96 kHz | → 48 kHz (80/147, 1/2) |
  | 16-bit only | anything 24-bit | no route (nothing to resample to in 24 bits) |

- `resample`: `Resampler`, a synchronous rational polyphase FIR (our own
  code), for any pair of rates, up or down. The ratio `up/down` is the
  exact rate ratio reduced by the gcd. The filter (`FilterDesign`, cached
  per rate pair) is a Kaiser-windowed sinc designed in `f64` at open:
  passband flat to `min(20 kHz, 0.45 · min(from, to))`, stopband from half
  the lower rate at ≥ 160 dB (designed for 166 dB), linear phase. Its group
  delay is compensated: the first output frame is the instant of the first
  input frame, and `n` input frames give exactly `round(n · up / down)`
  output frames. All arithmetic is `f64`; the output is requantized once to
  24 bits with TPDF dither (±1 LSB, a seeded generator, so the same input
  gives the same output); samples that clip (inter-sample overs) saturate
  and are counted. `set_gain` scales the input (the headroom).
  `process_f64` gives the unquantized output for analysis.
- `stems`: a track's stems (drums, bass, vocals, other; `stems::LANES` is
  the order a player hands them on in). A stem run reads the file as it is
  stored, makes it stereo at 44.1 kHz (`StereoFeed`) and separates it by
  `jodel_rhythm::stems::Splitter` in a thread of its own, up to about 20 s
  ahead. The engine hands the stems on beside the file as parts of an
  output of lanes (**Stem lanes**, below: a mixer plays them on its strips,
  so a stem's level is a fader); a head read along a time map (a
  transition's tracks) has them read along it as its parts, its main
  their sum. `PlayerOptions::separator` gives the separator (called in
  that thread, so a GPU set up there never holds up playback); without
  one only kept stems are parts, and a head read along a map has its
  whole mix as its "other" part, so a transition fades its tracks whole.
  A dropped run stops before its next chunk. With
  `PlayerOptions::stem_store`, a run first asks the store for the track's
  kept stems and reads them from its frame on (a second at a time, about
  150 times faster than they play): a track separated once (by the
  player's analysis) is never separated again, from any frame, in any run
  (a seek, a jump, a transition's tracks).
- `stem_file`: a track's four stereo stems at 44.1 kHz in a file: blocks
  of 4096 frames, each channel as 16-bit values times a power of two of
  its own (2^-16 to 2^4, so quiet stretches keep their detail and loud
  ones never clip), coded losslessly (silent, constant, or a fixed
  predictor of order 0 to 3 and a Rice code), an index for reading from
  any frame. `StemWriter` writes it as the stems come (under a temporary
  name, renamed once whole); `StemFile` reads it; `StemStore` is what the
  engine asks. About 26–28 MB a minute of music, within 90 dB of the
  separation.
- `transition`: `TransitionPlan`, a beat-matched crossfade from the
  playing track into the next: the window on the playing track (`start`,
  `length`, whole bars), where the next track is at its start (`entry`),
  the paired beats (playing-track time, next-track time), eight gain
  `Envelope`s (the playing track's four stems, then the next one's, in
  `Stem::ALL` order), the `TempoMode`, the `settle` time and `glide` (the
  tempo glides across the window rather than after it). An envelope's
  points are a gain at a time with a cut flag: between two points the
  gain moves on an equal-power curve (the desk's own,
  `jodel_output::mixer::equal_power`: the square of the gain moves in a
  straight line), and a cut steps at the point.
  `TransitionPlan::time_map` is the map the engine reads the next track
  along (a frontend draws the next track's playhead by it).
- `warp`: `TimeMap`, the next track's time at each moment of the playing
  track's, through the paired beats: each beat's speed is a least-squares
  slope over the four beats on either side, the map is a monotone cubic
  (Fritsch–Carlson) through the beats at those slopes, and after the last
  pair the speed glides back to 1 over `settle`. `VarResampler` reads a
  stream at any changing position along it: a Kaiser-windowed sinc (32
  taps a side, a 512-step phase table, interpolated between steps) whose
  cutoff falls with the speed, so a faster read does not alias. `Glide`
  is the playing track's time along the output's through a window whose
  tempo glides: speed 1 up to it, then a smoothstep to its end speed (at
  most 30 % off 1) over as long as the playing track needs to cover its
  window, and that speed after; `TimeMap::through_glide` moves a map's
  knots to the output times their playing-track times are heard at.
- `stretch`: `Stretcher`, the time-stretch: a phase vocoder (2,048-point
  frames, an output hop of 512) with identity phase locking around each
  spectral peak and the phases reset at onsets, so the tempo changes and
  the pitch and the drum hits stay.
- `engine`: `Player` with commands and an event stream (below).
- `tap`: `SpectrumTap`, a mixer's monitor: the most recent frames the
  device plays, for spectra.
- `target`: `DeviceTarget` (a real device), `NullTarget` (nothing, on a
  simulated clock), `MemoryTarget` (records every stream, for tests), and the
  `OutputTarget` trait (`jodel_output`'s) for others. A mixer's input target
  (`jodel_output::mixer::InputTarget`) is one: several players on one
  device, each on an input of a strip (`let strip = mixer.strip("Player");
  let (target, fader) = mixer.input(&strip, Role::Lead, 1.0);
  Player::new(Box::new(target), ..)`); alone at gain 1 with its strip at
  0 dB the engine reaches the device bit for bit through it. The
  microphone is the mixer's (`Mixer::set_mic`), not the engine's.
- `desk`: `StemDesk`, a player on a mixer's desk as the app has it: a
  group strip, a strip inside it for each stem, and a lane input on them
  (`StemDesk::new`, for `Player::new`), or two (`StemDesk::two_heads`, for
  `Player::on_desk`: the engine's two heads, the user's stem levels acting
  on both).

## The player

```rust
let (player, events) = Player::new(Box::new(DeviceTarget::find("default")?), PlayerOptions::default());
player.load(paths)?;
player.play(0)?;
```

Commands (`Command`, or the methods of the same names): `Load`, `Play(index)`,
`Enqueue`, `Pause`, `Resume`, `Seek(seconds)`, `Next`, `Previous`, `Stop`,
`SetSettings`, `Transition(plan)`, `CancelTransition`, `Jump { index, to,
at: Now | At(seconds) }` and `JumpWindow { back, ahead }` (jumps within the
playing track: below), `PlayIn { index, lead }` (the track from its start
after `lead` seconds of silence: its `TrackStarted` comes as the silence
begins to be heard and its positions run negative through it, so a
count-in drawn from them falls on what is heard) and `HoldAtEnd(bool)`
(each track's end stops playback instead of running into the next:
`Held { index }` names the one that would have come), and `PlayAt {
index, share }` (the track from `share` of its length, its start when the
length is not known: a preview from a third in), and `MakeStems(Never |
Kept | Separate)` (the stems as parts of an output of lanes: below).
`Player::on_desk(lead, guest, options)` starts the engine on two heads
of a desk (`StemDesk::two_heads`), which transitions need: a player on
one target (`Player::new`) plays everything else as before.

Events (`Event`), track events in the order they are heard:

| Event | When |
|---|---|
| `QueueChanged { queue, current }` | after `Load` and `Enqueue` |
| `State(Playing \| Paused \| Stopped)` | play state changes |
| `TrackStarted(TrackStart)` | a track starts to be heard: `info` (format, tags, cover), `route`, `output` (`StreamInfo`), `device`, `bit_perfect`, and `head`, the head it plays on (0 the first, 1 a desk's second: a transition's next track) |
| `Position { index, head, seconds, frame, output_frame, generation }` | every `position_interval` while the output clock moves; `output_frame` counts that head's frames |
| `Seeked { index, seconds }` | a seek took effect |
| `Clipped { index, samples, output_frame }` | the resampler or the source's requantization clipped (the count since the track or the seek started; at most four a second of output) |
| `TrackEnded { index, finished }` | played to its end, or left |
| `Error { index, error, skipped }` | `skipped`: only that track is lost (unreadable, `NoRoute`) |
| `Notice { index, message }` | new settings the track being heard keeps playing through (no route with them, or their output did not open) |
| `PartsStopped { index, reason }` | the track's stems as parts of the output failed: it has none until they are asked for again |
| `Transition(Armed \| Started { head, output_frame, origin } \| Finished \| Cancelled { reason })` | a transition was taken, its window began to be heard (at that output frame of that head; the next track's head's frame 0 is that head's `origin`), the next track is back at its own tempo, or it was called off and why |
| `Jumped { index, seconds }` | a `Jump` is heard: the track plays on from there |
| `QueueEnded` | the last track played out |

Timing: track, clip, skipped-track and queue-end events are queued
against the output frame they belong to and sent when
`jodel_output::Output::position` reaches it. Positions come from the same
clock, never from the decode position.

The decode thread reads the source a block at a time and resamples what
it reads when the route has a resampler (whole-track streams, so the
lengths are exact); otherwise the samples go to the output as read.

- **Seek**: the source moves to the exact frame and the output is
  flushed.
- **Gapless**: at the end of a track the next one is opened at once. If its
  route plays at the open output's rate, channels and a format of its route,
  it is written into the same output with no gap; otherwise the output plays
  out, closes and reopens at the new format.
- **Pause** pauses the output and stops writing; the bytes are unchanged.
- **Settings** (`SetSettings`): when the track being played takes another
  route with them, it moves to that route without a break. The new route
  opens a little before the frame being written (0.1 s for a resampler)
  and decodes ahead while the old one plays on, in
  the turns the output leaves: while it holds half its buffer or more, is
  paused, or took nothing of the last write. Once the new route has caught
  up, the two meet at the first frame after what is written where both
  are exact: a track frame that both routes' rate ratios turn into whole
  output frames, so the old route's output ends at frame `S` and the new
  one's begins there with track frame `F`, nothing lost or repeated. At
  the same output format they share the output, crossfaded over
  `PlayerOptions::switch_fade` (20 ms; zero meets at one frame), and
  nothing is flushed. At another format the old route fades out over the
  same time up to `S`, the output plays out and opens at the new format,
  and the new route plays from `F`, faded in, with 200 ms of it decoded
  already, so the only break is the device's own reopen. `TrackStarted`
  says where (`output_frame`, `from_frame`), and the position goes on from
  there. Settings never stop playback or skip the track: if it has no
  route with them, or the new route's output does not open, it keeps the
  route it has (back on it at the same frame, in the second case) and a
  `Notice` says so; a new route that fails before they meet decodes once
  more from the start of the track, then the track keeps its route. One
  that never catches up (a machine too slow for both) leaves the track on
  its route to its end. The new settings apply to the tracks that start later;
  a later track that has no route is skipped as any file without one.
- **Transitions** (`Transition(plan)`, `engine/transition.rs`): the
  playing track (A) crossfades into the next (B) over the plan's window,
  each stem on its own envelope, with B's beats on A's. The engine does
  not mix them: the desk does, each on a head of the player's own
  (`Player::on_desk`); on a plain target a plan is `Cancelled("needs a
  desk")`. The plan is taken only for the track playing and the one after
  it, with a window ahead of what is heard; `Armed` says so, else
  `Cancelled` says why. About 4 s before the window B opens on the second
  head at the open output's format (its route is chosen for that format
  alone, so the device never opens again; more than two channels are
  refused), linked on the desk to A's head (`Fader::follow`) with its
  frame 0 on A's stream frame `origin`, a second before the window
  (`NEXT_LEAD`), and read along the plan's time map from its first frame
  (**Read along a map**, below): its stems along the map, by
  `VarResampler` (varispeed: the pitch moves with the speed) or by a
  `Stretcher` per channel steered onto the map (time-stretch: the pitch
  stays), are its parts, its main their sum. It writes ahead on turns of
  its own; the link moves it with A's stream frame for frame, so a pause
  or an underrun of A holds it, and frames of it that are late are
  dropped on the desk (counted), never A's. With `glide`, the tempo moves
  from A's to B's across the window: A is read along a `Glide`
  (`TransitionPlan::glide_of`: its speed ends at the inverse of B's
  locked speed over the window's last quarter), B's map is laid onto the
  output's time through it, so its beats still fall on A's as they are
  heard, and B is at its own tempo at the window's end (A's position
  follows the glide); a playing track that cannot glide (off its own
  route, say) has the transition called off. The plan's eight lanes
  become each head's automation on its own stream frames
  (`Fader::automate`): each stem's envelope its part's curve, at A's time
  heard on each frame (through the glide's inverse), and main's curve the
  lanes' mean, for a head heard without parts; before the window A is
  whole and B silent. Two loud masters overlapping are taken by the
  desk's look-ahead limiter. A is written up to the window's end and no
  further; there B's head is made the desk's Lead, A's output plays out
  what it holds and closes, and the heads swap: `TrackEnded` of A and
  `TrackStarted` of B come on one frame, and from there the positions
  count B's head's frames (A's less `origin`). B's position follows the
  map while its speed glides back to 1 (with `glide`, what is left of
  it). A little after that it goes back onto its own route at the same
  format, bit-perfect again where it was before, its automation goes, and
  the transition is `Finished`. Without a separator (or once its stems
  failed) each head's whole mix is its "other" part, so every plan is a
  beat-matched fade of the whole tracks.

  The device keeps the open format after a mix: the tracks after it are
  routed at it (resampled where they must be), gaplessly, until playback
  stops, the queue ends, a track is picked (`Play`, `PlayAt`, `Previous`,
  `Next` outside a mix, `Load`) or the settings change; a track with no
  route there (resampling off) opens at its own rate.

  On a device, the engine never lets the output run dry for a
  transition: B's stems not separated in time give way to its whole mix
  (`PartsStopped` says so), and B's frames late are dropped on the desk.
  With `PlayerOptions::offline` (an output not played in real time: a
  file being rendered, a test's), it waits instead: the desk holds A at
  `origin` until B has begun, and B's writing waits for its stems, so a
  render is the same bytes every time.

  A mixer's stem levels act on both heads, on their frames (a bar, say:
  the app mutes a stem's strip 0.1 s ahead, during a mix too); they never
  move the transition or call it off. `Pause` pauses A, and the link
  holds B. `Next` in the window fades A out over 30 ms from where the desk
  mixes it next and puts B at its lanes' end values (`Cancelled`: cut to
  the next track), and hands over once A is written to its fade's end;
  `Next` while B settles plays the track after it. Other settings wait for
  the end (a notice says so), and so does a new plan or
  `CancelTransition` once it is heard (one armed is replaced or called
  off). Only moving playback calls it off: a seek, another track, a new
  queue, its next track taken out of the queue. Called off in the window
  (B gone from the queue, or failed), B fades out over 30 ms and A comes
  back whole; a seek, `Play`, `Previous`, `Stop` or `Load` drop B at once.
  A playing track called off while it glides goes on from where the
  glide has it, at its own speed (it meets its own route where its map
  puts it). A seek while B settles is a plain seek of B.
- **Read along a map** (`engine/along.rs`, the engine's own
  `Command::Along`): on an output of lanes, a head's stems are read along
  a time map at the separator's 44.1 kHz (8 channels, before the parts'
  resampler) as its parts, and its main is their sum, so main and the
  parts stay on the same frames for a mixer's switch and the control
  word stays `Present`; without a separator (or once the stems failed)
  the whole mix is read into "other" alone, the other parts exactly
  silent. The move onto the map is a segment joined at its frame (the
  head writes no further until then), main crossfaded once from its
  route's. Once the map runs straight (and a stretch is steered back onto
  it) the head goes back onto its own route at the same format, a switch
  met where the map puts it, and plays the file bit for bit from there;
  one that does not come about is asked again a second later. Meanwhile
  nothing reroutes the head (new settings wait), jumps are refused and no
  transition is set up. A mono file's stems are doubled; a track of more
  than two channels is refused.
- **Jumps** (`Jump { index, to, at }`): the playing track goes to `to`
  seconds (a bar's start, say), at once or when it reaches `at`, never
  with a gap. With a window (`JumpWindow { back, ahead }`, seconds) the
  playing track keeps what it wrote for `back` seconds and the output's
  ring, and decodes `ahead` seconds ahead while the output is full: its
  buffer is a stretch of the track in order. A jump inside it moves where
  the buffer is written from. At once, the output first gives back what
  it has not played (`Output::retract`; the device keeps a few of its
  buffers, so the jump is heard some 30–60 ms after the command); at a
  time, the track is written up to that frame. The target's first
  `switch_fade` frames are crossfaded from what would have played next
  (that is the next bar's downbeat on a bar jump, so the attack stays),
  and the segment starts at the target, as a seek's would, with the buffer
  going on as it is; `Jumped` comes as it is heard. A
  time already written is taken back first; one too late for that lands
  as far after the target as it is late, so the beat goes on. The stem
  parts follow a jump on their own (their run keeps the separated stems
  for the window: **Stem lanes**). A target outside the window is read
  again from a little before it (as a switch's new route is), while the
  track plays on, and takes over once it has a seam's worth: later, never
  broken. No window while a transition is set up (it cuts the playing
  track's output at the window's end), nor for a track read along a map
  (until it is back on its own route, a jump there reads its target
  again); arming one reads the track again from what is written. `player/examples/jumps.rs` plays jumps on a real file in real
  time and reports when each was heard and the longest digital silence.
- **Stem lanes** (`MakeStems`, `engine/lanes/`): on a target that takes
  lanes (`OutputTarget::lanes() >= 5`: a mixer's lane input) and with a
  separator or a stem store, every output opens with five lanes
  (`jodel_output::lanes`): main, then drums, bass, vocals and other
  (`stems::LANES`), and a control word a frame. Main is what the track
  plays, as it would be alone; the stems are written beside it on the same
  frames, so a seek, a flush or a jump taken back acts on all of them.
  `MakeStems::Never` (the default) writes none (`Parts::None`); `Kept`
  takes them from the stem store only (asked in the run's thread; a track
  not kept has none, quietly, and nothing is separated); `Separate` from
  the store, else separated as the track plays. A stem run
  (`StemRun::take`, the four stems unmixed, no limiter; their clips are not
  reported yet) is resampled to the output's rate with the
  track's own filter and gain, on 4 or 8 channels, requantized to 24 bits,
  and laid out in the track's channels (a mono file's stems averaged, more
  than two channels the stems on the front pair and nothing on the rest).
  It starts on an output frame a run frame falls on exactly (the output
  frame ↔ track time line of the segment; a run starts a whole number of
  the separator's frames from the segment's track frame), so the stems add
  up to main within a few 24-bit steps at any rate (a file at another rate
  than 44.1 kHz goes through two filters, which ring apart for a couple of
  milliseconds where the file cuts in and out). Each segment of the track
  (`Track::begin_segment`, `continue_segment`) has an id and says how it
  began (`SegStart`): a new one (a track, a seek) has its stems start from
  silence on its first frame, as main's resampler starts there (from the
  stems kept when their grid has that frame, else a new run from it); a
  joined one (a jump, a route met at another gain) from the stems kept,
  0.1 s before it, their first frames crossfaded from what would have
  played next over the frames main's are (the length main used; a
  transition's next track, read along its map from its first frame, over
  none). A route met at the same
  format and gain goes on in the same segment. The stems are cut or padded
  to the track's end. The control word keeps a promise (`lanes/control.rs`,
  its changes logged, so frames given back take theirs back): after
  `Present` the stems never stop at once, they say `Ending` for a mixer's
  switch's length (`ending_frames`) while still valid, then `None`; the
  buffer keeps those frames whatever starts again (but on a new track's
  first frame, where main cuts: `jodel_output::lanes`). On a device the
  stems are never waited for: before a write only what it needs is made
  (a seam's length past it, nothing while absent), the rest after, a few
  pieces at most; while the ring is full they are made on, and once none
  are left to make the write waits for room as on a plain output (the
  thread does not spin on the full ring); through a count-in
  (`PlayIn`) they are made from the track's first frame, so they are
  there as it starts. Stems asked for while the output holds frames
  written without them (`MakeStems` from `Never`, the first touch of a
  stem's strip) start on the frame heard; once they are made a little past
  what is written, the output gives back what it has not played and the
  same main is written again with them, the control word's changes given
  back with the frames (`engine/lanes/catch_up.rs`; the track keeps what
  the output may give back behind what it wrote for it, also without a
  jump window, and a jump back within it is met from it; not while a
  route switch, a transition or a jump is under way, nor over a jump's
  crossfade). Kept stems are there in milliseconds, separated ones once
  the first chunk is. Parts that ran out and are made again (a separator
  that fell behind) are caught up the same way. They say `Present` once
  0.5 s of them is made ahead (and no sooner again after they ran out),
  and at a track's end they run to its last frame (main decoded to its
  end first, to know where that is) when the next track's (separated from
  its start once the playing file is read, or its parts made to its end)
  are ready, or nothing follows; else they end before it. A route switch
  to another format ends them before main fades out, and they come back
  only after the new route's rise. Offline the writing waits for them
  (and for the next track's). A failing separator (`PartsStopped`, for
  that track) leaves none. A head read along a map (**Read along a
  map**) has parts throughout, its main their sum, and goes back onto its
  file as any route switch does, so a mixer playing the parts at other
  levels across a transition hears no jump at either end. Without a lane
  target the output is as it always was.
- **A busy device**: opening an output that the device reports busy
  (`DeviceBusy`: in use by another application, hog mode held, `EBUSY`, as
  it can be right after one of our outputs closed) is tried again every
  100 ms for up to `OPEN_BUSY_WAIT` (1 s) before it fails.
- **Resampling** runs after the source.
  Each track, and each seek, is its own resampler stream, starting from
  silence and flushed at its end, so a seek gives exactly a fresh resample
  from the target and gapless tracks on one resampled route add their exact
  lengths. Positions stay in the file's frames and seconds.

## The spectrum tap

`SpectrumTap::new(frames)`, set as a mixer's monitor
(`mixer.set_monitor(Some(Box::new(tap.clone())))`), is a lock-free ring of
the last `frames` frames the device plays (stereo; mono is doubled, wider
outputs give their first two channels), tagged with their device frame
index; each opening of the device is a generation. `read_heard(&mut buf)`
fills `buf` with the frames that end at the frame being heard now;
`read(generation, end, &mut buf)` reads any recent window. A read that
raced the writer returns `TapMiss::Overwritten` and can be retried.

## Example

```sh
cargo run --release -p jodel-playback --example play -- FILE... --null --seconds 5
cargo run --release -p jodel-playback --example play -- FILE... --device "USB DAC"
```

`--null` plays into a null output in real time (every rate and format
accepted), so no device is touched. Other options: `--max-rate HZ`,
`--no-resample`.

## Testing

`cargo test -p jodel-playback` needs no device. The engine tests play the
PCM fixtures (`tests/fixtures/pcm`, 44.1 kHz 16-bit and 48 kHz 24-bit
FLAC, made by its `make.py`) into memory outputs and compare the bytes
with the files' own samples: gapless concatenation, a reopen at a format
change, seeks, pause and resume, `NoRoute` on a packed-24-only
48 kHz-family device, skipped files, settings changes, the spectrum
tap on a mixer, and event order. A gated test output lets each test place
its command at an exact output frame. `tests/mixer.rs` plays the same
routes through a mixer's Lead input, on its one mix path (FLAC, packed
24-bit, gapless, a reopen at a format change, plain and
resampled PCM, pause and resume, seeks and a jump), and gets the bytes it
gets without one, and `PlayAt` starts at the frame a third in.
`tests/lanes.rs` and `tests/lanes_control.rs` (the control word; helpers in
`tests/common/lanes.rs`) run those cases (and `PlayAt`) through a mixer's lane
input with the stems made (offline, so they are there on every frame,
untouched): the device gets the same bytes. On a memory target that keeps
the lane stream (`MemoryTarget::with_lanes`) the test separator's stems
(a quarter of the mix each) add up to main within 64 steps (4–6 measured) at 44.1, 48, 96 and 192 kHz, from a seek's
very frame (from the stems kept) and across a jump's crossfade, also for a
96 kHz file at 48 kHz, and a jump at every 397th frame over a piece of
stems (so one lands where the parts made ahead end); every control word
sequence keeps the promise (and a property test of the buffer and the
control timeline together keeps what is promised there); a jump taking
back frames that said the parts end (an output holding what is written,
`HoldingTarget`) has them go on; stems kept longer than the track do not
hold the next one up offline; it
starts `None` and turns `Present` once the stems are separated, and is
`Present` from the first frame from a store (offline); stems separated
slower than played run out, end, and are there again, also after a jump
back; the next track's stems follow gaplessly, and when they fail this
track's end before its last frame; a failing separator leaves `None` and
says so, a track not kept says nothing; a file of no stated length plays
to its end offline; mono and six-channel files have the stems in their
channels; the stems end on the track's last frame; and on the mixer a
muted drums strip gives three quarters of the file, unmuted the file again
bit for bit. `tests/lanes_transition.rs` plays a transition on a desk's
two heads with the drums muted on their strip: three quarters of what
the transition plays on every frame, on both tracks; with a stem soloed a
quarter; with a stem muted on a bar inside the window, half from that
bar; and nothing is heard of the next track's move back onto its file.
`tests/along.rs` reads a head along a map: a click track's clicks land on
the frames the map puts them on (varispeed, stretched, resampled to
48 kHz), the head goes back onto its file bit for bit once the map runs
straight (stretched too; resampled, to within the dither), the move onto
the map crossfades once, mono files, no separator (the whole mix as
"other"), new settings and a transition waiting or refused, a switch back
that fails asked again, and on a real-time device stems that fall behind
giving way to the whole mix. The transition tests (`tests/transition.rs`,
`transition_rules.rs`, `transition_desk.rs`, `transition_stems.rs`;
helpers in `tests/common/transition.rs`) run on a memory desk: the next
track's beats on the playing one's (varispeed, stretched, gliding, a glide
planned late or called off), the hand-over, a fade of whole tracks without
a separator, `Next` in the window without a click (also in real time),
`Next` while prepared or settling, `Stop`, `Play` and `Previous` dropping
the next track, the next track moved away fading out, a pause in the
window held by the link, the rate held after a mix (and given up where a
track has no route at it), two hand-overs swapping the heads' roles with
the microphone hearing throughout, a next track whose stems are late
playing whole, stems muted on the desk at any point of a mix, the rules
the player's `INTERACTIONS.md` names, a failing separator, and a
transition at 192 kHz ending on the playing track's last frame; with no
second head a transition is called off. `tests/switch.rs` switches routes mid-track: at one
rate, a chain of four routes (one made while paused) is one stream equal
to each route's own output from the frame it took over at; a crossfade is
exact outside its window; at another format the old route fades out at
the frame the new one, in the next stream, fades in from; a seek during a
switch plays the new route from there; and on a device that plays in real
time nothing is padded.

`tests/resample.rs` checks the resampler over a matrix of rate pairs (44.1
to 48 and back, 88.2 to 96 and back, 176.4 to 192, 44.1 to 192, 192 to 96,
384 to 192, 352.8 to 192, 96 to 44.1, 384 to 48, 44.1 to 32, 48 to 22.05,
44.1 to 88.2, 48 to 192 kHz): the designed taps' response on a dense FFT
(passband ripple ≤ 0.001 dB, stopband ≥ 160 dB), tones end to end before
requantization (level within 0.001 dB, every image and alias ≥ 160 dB
down, stopband tones ≥ 160 dB down when going down), a symmetric impulse
response at the expected instant, exact lengths for any chunking,
independent channels, the dither, clip counting, and the speed (release
builds only; print the numbers with
`cargo test --release -p jodel-playback --test resample -- --nocapture`).
`tests/formats.rs` covers the other formats with the small fixtures in
`tests/fixtures/formats` (made by its `make.py` with FFmpeg, see its
README): AIFF (written by the test, 16 and 24 bits), ALAC in MP4 and CAF
decode to exactly the PCM of the FLAC fixture and seek exactly; MP3,
Vorbis and AAC-LC in MP4 (its edit list) have their original sample
count, start on the first sample (the best-matching lag against the
source signal is 0) and seek sample-accurately (equal to the full decode
from the target, on and around packet edges, through the engine too);
tags and covers from MP4 atoms, ID3v2 and Vorbis comments;
user-defined ID3v2 tags; HE-AAC, signalled after an LC header or not at
all, is refused with a clear message (`src/source/aac.rs` tests the header
variants, the iTunSMPB tag and the edit-list rules); 32-bit integer and float WAV are requantized (exact
where they fit, dithered otherwise, clips counted); lossy tracks play
gaplessly into one stream, never bit-perfect, and a resampled lossy track
equals a fresh resample of the unquantized decode.
`tests/resampled.rs` plays through the engine on the GC7's matrix:
44.1 kHz PCM and FLAC at 48 kHz (bytes equal a fresh resample), a seek,
gapless lengths, clip events, and turning resampling off.
