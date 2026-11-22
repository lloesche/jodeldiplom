# jodel-playback

The playback engine between files and [`jodel-output`](../output): FLAC,
WAV, AIFF, CAF and ALAC files to bit-perfect, exclusive device output, and MP3 (MP2, MP1) and Ogg Vorbis decoded to
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
  | MP4/M4A | ALAC (AAC is reported as not supported) | MP4 atoms, `covr` |
  | MPEG audio | MP3, MP2, MP1 | ID3v2 (`APIC`), ID3v1, APE |
  | Ogg | Vorbis | Vorbis comments, `METADATA_BLOCK_PICTURE` |

  Symphonia features: `flac`, `wav`, `pcm`, `aiff`, `caf`, `alac`, `isomp4`,
  `mp1`, `mp2`, `mp3`, `ogg`, `vorbis`, `id3v1`, `id3v2`. Symphonia 0.6 has
  no Opus decoder; Opus and AAC files are refused with a message saying so.

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
  its first sample is the first one encoded. Lossy seeks decode from 8,192
  frames before the target (the MP3 bit reservoir, the Vorbis window) and
  drop what comes before it; a container that cannot seek or read on from
  where it landed is decoded from the start instead. Seeks are
  sample-accurate on every format.
- `rate`: `choose_route(file, capabilities, settings)` is a pure function
  from the file, the device's exclusive capability matrix and the settings
  (rate cap, resampling, headroom) to a `Route` (stage, output
  rate, channels, format preference, resample step) or a `NoRoute` with a
  sentence saying why nothing fits.

  | File | Route |
  |---|---|
  | lossless PCM (FLAC, WAV, AIFF, CAF, ALAC) | its own rate and channels, untouched |
  | lossy (MP3, Vorbis, ...) | decoded to 24 bits, its own rate and channels; never lossless |

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
  so a stem's level is a fader), and a transition sums them faded by its
  lanes on a `Stage::Stems` route at the open output's format (never
  bit-perfect), kept under −0.2 dBFS by a limiter that looks 2 ms ahead in
  the stems the run has (no delay; exactly no gain where nothing goes over,
  so the file and its four stems still sound alike: stems summed again go
  over full scale, the band without its singer up to 2.15 dB on 24 songs,
  which the requantization clipped), then resampled to the device or
  requantized at 44.1 kHz. `PlayerOptions::separator` gives the separator
  (called in that thread, so a GPU set up there never holds up playback);
  without one only kept stems are parts, and a transition fades its tracks
  whole. If a transition's separator fails, `Event::StemsStopped` says so
  and the track goes on on its own route from the frame after what was
  written, nothing flushed. A dropped run stops before its next chunk. With
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
  gain moves on an equal-power curve, and a cut steps at the point.
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
length is not known: a preview from a third in), `MakeStems(Never |
Kept | Separate)` (the stems as parts of an output of lanes: below), and
`StemMask { gains, at }` (each stem's gain as a mixer plays the parts,
drums, bass, vocals, other, from `at` seconds into the playing track or
at once: what a transition mixes its tracks' stems with, below).

Events (`Event`), track events in the order they are heard:

| Event | When |
|---|---|
| `QueueChanged { queue, current }` | after `Load` and `Enqueue` |
| `State(Playing \| Paused \| Stopped)` | play state changes |
| `TrackStarted(TrackStart)` | a track starts to be heard: `info` (format, tags, cover), `route`, `output` (`StreamInfo`), `device`, `bit_perfect` |
| `Position { index, seconds, frame, output_frame, generation }` | every `position_interval` while the output clock moves |
| `Seeked { index, seconds }` | a seek took effect |
| `Clipped { index, samples, output_frame }` | the resampler or the source's requantization clipped (the count since the track or the seek started; at most four a second of output) |
| `TrackEnded { index, finished }` | played to its end, or left |
| `Error { index, error, skipped }` | `skipped`: only that track is lost (unreadable, `NoRoute`) |
| `Notice { index, message }` | new settings the track being heard keeps playing through (no route with them, or their output did not open) |
| `Transition(Armed \| Started { output_frame } \| Finished \| Cancelled { reason })` | a transition was taken, its window began to be heard (at that output frame), the next track is back at its own tempo, or it was called off and why |
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
  opens a little before the frame being written (1 s for a transition's
  stems, 0.1 s for a resampler) and decodes ahead while the old one plays on, in
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
- **Transitions** (`Transition(plan)`): the playing track crossfades into
  the next over the plan's window, each stem on its own envelope, with
  the next track's beats on the playing one's. The plan is taken only for
  the track playing and the one after it, with a window ahead of what is
  heard; `Armed` says so, else `Cancelled` says why. About 4 s before the
  window the playing track moves onto stems at the open output format
  (the seamless switch above; with every gain at 1 the stems add up to
  the track), and the next track opens on stems at the same format,
  decoding ahead in the switch's turns. Through the window both are
  written: the next track's 44.1 kHz stem sum is read along the time map,
  by `VarResampler` (varispeed: the pitch moves with the speed) or by a
  `Stretcher` per channel steered onto the map (time-stretch: the pitch
  stays), its stems weighted by its envelopes, resampled to the output
  rate and added to the playing track's. With `glide`, the tempo moves
  from the playing track's to the next one's across the window: the
  playing track moves onto a fresh stems track read along a `Glide`
  (`TransitionPlan::glide_of`: its speed ends at the inverse of the next
  track's locked speed over the window's last quarter), the next track's
  map is laid onto the output's time through it, so its beats still fall
  on the playing track's as they are heard, and the next track is at its
  own tempo at the window's end (the playing track's position follows
  the glide). Two loud masters added go over
  full scale: the sum is kept as it is up to −1 dBFS and its peaks above
  that bend smoothly toward full scale (a tanh knee), instead of being
  clipped flat. At the window's end the next track takes the output on from
  that frame, gaplessly, and `TrackStarted` comes at that frame; its
  position follows the map while its speed glides back to 1 (with
  `glide`, what is left of it). A little
  after that (`Finished`) it is straightened onto a plain segment and
  moves back onto its own route, bit-perfect again where it was before.
  Without a separator (or once it failed, until stems may work again:
  `StemsStopped`) the stems are the whole track (`stems::whole_mix`), so
  every plan is a beat-matched fade of the whole tracks; whether a track
  is whole is its own, and a whole track takes no `StemMask`.

  On a device, the engine never lets the output run dry for a
  transition:

  - a piece of the window whose next-track frames are more than 150 ms
    late goes without them;
  - with `glide`, a playing track not on its stems by the window goes on
    alone (`Cancelled`).

  With `PlayerOptions::offline` (an output not played in real time: a
  file being rendered, a test's), the engine waits instead:

  - the playing track holds within a second of the window until it is on
    its stems;
  - each piece of the mix waits for the next track's frames.

  A mixer's
  stem gains (`StemMask`) changed at any time go into its fades, on both
  tracks, at the time they are heard at (a bar, say: the next track makes its output only a
  quarter of a second ahead, so a change asked for a second ahead falls
  on it there too); they never move it or call it off. Other settings
  wait for its end (a notice says so), and so does a new plan or
  `CancelTransition` once it is heard (one armed is replaced or called
  off). Only moving playback calls it off: a seek, another track, a new
  queue, its next track taken out of the queue. Then up to the
  window's end the playing track goes on alone, at its full level (the
  next track is dropped), and after it the next track carries on from
  where it is, on a plain timeline; a playing track called off while it
  glides goes on from where the glide has it, at its own speed (a track
  read along a map meets its new route where its map puts it).
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
  broken. No window while a transition is set up (it mixes into the
  buffer as it is made), nor for a track it left on its stems (a jump
  there reads its target again); arming one reads the track again from
  what is written. `player/examples/jumps.rs` plays jumps on a real file in real
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
  played next over the frames main's are (the length main used); a
  continued one (a transition's timeline made straight) the same with no
  crossfade. A route met at the same
  format and gain goes on in the same segment. The stems are cut or padded
  to the track's end. The control word keeps a promise (`lanes/control.rs`,
  its changes logged, so frames given back take theirs back): after
  `Present` the stems never stop at once, they say `Ending` for a mixer's
  switch's length (`ending_frames`) while still valid, then `None`; the
  buffer keeps those frames whatever starts again (but on a new track's
  first frame, where main cuts: `jodel_output::lanes`). On a device the
  stems are never waited for: before a write only what it needs is made
  (a seam's length past it, nothing while absent), the rest after, a few
  pieces at most, and a full ring is left alone; through a count-in
  (`PlayIn`) they are made from the track's first frame, so they are
  there as it starts. They say `Present` once
  0.5 s of them is made ahead (and no sooner again after they ran out),
  and at a track's end they run to its last frame (main decoded to its
  end first, to know where that is) when the next track's (separated from
  its start once the playing file is read, or its parts made to its end)
  are ready, or nothing follows; else they end before it. A route switch
  to another format ends them before main fades out, and they come back
  only after the new route's rise. Offline the writing waits for them
  (and for the next track's). A transition's stems route and mix and a
  failing separator (`PartsStopped`, for that track) have none, but for a
  transition's two ends, so a mixer playing the parts at other levels hears
  no jump: the playing track keeps its parts while the transition is armed
  and until its move onto its stems has met it, and they end only once
  that move's crossfade is over (`Track::parts_hold`); the next track, left
  on its stems when the transition is over, has parts again there, and
  while `StemMask` has a gain other than one (now or in a change to come)
  its move back onto its file waits until they have been there for a
  mixer's switch and its end, so the mixer plays them across it (2 s of
  output at most, `PARTS_PATIENCE`, then it moves anyway). Inside the
  transition the two tracks' stems are mixed at `StemMask`'s gains (`engine/stem_mask.rs`): its changes at times of a
  track are kept in order, each landing at its own (one at an earlier time
  is never moved by a later one), and one whose time has been written, or
  whose track is no longer played, is in force. So what is heard as main
  there is what the mixer would play. Without a lane
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
the lane stream (`MemoryTarget::with_lanes`) the stems add up to main
within 64 steps (4–6 measured) at 44.1, 48, 96 and 192 kHz, from a seek's
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
bit for bit. `tests/lanes_transition.rs` plays a transition through the
mixer with the drums muted there and masked in the engine: three quarters
of what the transition plays on every frame, across the playing track's
move onto its stems and the next one's back onto its file; with a stem
soloed a quarter; with a stem muted on a bar inside the window, half from
that bar; and on a memory target keeping the lane stream, the next track
goes back onto its file only once its parts have been there for a
mixer's switch and its end. `tests/switch.rs` switches routes mid-track: at one
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
decode to exactly the PCM of the FLAC fixture and seek exactly; MP3 and
Vorbis have their original sample count, start on the first sample (the
best-matching lag against the source signal is 0) and seek
sample-accurately (equal to the full decode from the target, through the
engine too); tags and covers from MP4 atoms, ID3v2 and Vorbis comments;
user-defined ID3v2 tags; AAC is refused
with a clear message; 32-bit integer and float WAV are requantized (exact
where they fit, dithered otherwise, clips counted); lossy tracks play
gaplessly into one stream, never bit-perfect, and a resampled lossy track
equals a fresh resample of the unquantized decode.
`tests/resampled.rs` plays through the engine on the GC7's matrix:
44.1 kHz PCM and FLAC at 48 kHz (bytes equal a fresh resample), a seek,
gapless lengths, clip events, and turning resampling off.
