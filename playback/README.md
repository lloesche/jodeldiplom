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
  (rate cap, resampling, headroom, stems) to a `Route` (stage, output
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
- `stems`: stem playback. `Settings::stems` is a `StemMix`: each stem's
  (drums, bass, vocals, other) level in percent, 0 off; with all four at
  100 the file plays on its usual route, with any lower (or off) on a
  `Stage::Stems` route: the file read as it is stored, made stereo at 44.1 kHz (`StereoFeed`), separated
  by `jodel_rhythm::stems::Splitter` in a thread of its own, up to about 20 s
  ahead, and the stems that are on summed (a 10 ms fade when one is
  switched), kept under −0.2 dBFS by a limiter that looks 2 ms ahead in the
  stems the run has (no delay; exactly no gain where nothing goes over, so
  the file and its four stems still sound alike: stems summed again go
  over full scale, the band without its singer up to 2.15 dB on 24 songs,
  which the requantization clipped), then resampled to the device or
  requantized at 44.1 kHz; with
  an output open, at its rate when the device takes it there
  (`choose_route_near`), so stem mode keeps the output open. It
  is never bit-perfect. `PlayerOptions::separator` gives the separator
  (called in that thread, so a GPU set up there never holds up playback);
  without one the mix is ignored. A new mix on the stem route is summed
  from then on, with no restart; the next track's separation starts once
  the playing file has been read, so track changes stay gapless; if the
  separator fails, `Event::StemsStopped` says so and the track goes on on
  its usual route from the frame after what was written, nothing flushed.
  A dropped run stops before its next chunk. With
  `PlayerOptions::stem_store`, a run first asks the store for the track's
  kept stems and reads them from its frame on (a second at a time, about
  150 times faster than they play): a track separated once (by the
  player's analysis) is never separated again, from any frame, in any run
  (a seek, a jump, the Transition tab's stems kept ready, a transition's
  tracks).
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
  device, each on an input (`let (target, fader) = mixer.input(Role::Lead,
  1.0); Player::new(Box::new(target), ..)`); alone at gain 1 the engine
  reaches the device bit for bit through it.

## The player

```rust
let (player, events) = Player::new(Box::new(DeviceTarget::find("default")?), PlayerOptions::default());
player.load(paths)?;
player.play(0)?;
```

Commands (`Command`, or the methods of the same names): `Load`, `Play(index)`,
`Enqueue`, `Pause`, `Resume`, `Seek(seconds)`, `Next`, `Previous`, `Stop`,
`SetSettings`, `Transition(plan)`, `CancelTransition`, `SetStemsAt { stems, at }`
(the stems heard become `stems` from `at` seconds into the playing track,
to the frame: below), `Jump { index, to, at: Now | At(seconds) }`,
`JumpWindow { back, ahead }` and `KeepStems(bool)` (jumps within the
playing track: below), `PlayIn { index, lead }` (the track from its start
after `lead` seconds of silence: its `TrackStarted` comes as the silence
begins to be heard and its positions run negative through it, so a
count-in drawn from them falls on what is heard) and `HoldAtEnd(bool)`
(each track's end stops playback instead of running into the next:
`Held { index }` names the one that would have come), `PlayAt { index,
share }` (the track from `share` of its length, its start when the length
is not known: a preview from a third in), and `SetMic(Option<Mic>)` (a
microphone listened to, never played: lined up with what the output plays
by a tap that only reads its buffers, `jodel_output::Output::set_tap`, set
again on every output opened).

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
- **Stems at a time** (`SetStemsAt`): the stems that are on change on the
  frame of the playing track at `at` (with the usual 10 ms fade), not as
  soon as the output has played what it holds. Up to there the track
  plays the stems it had. The first stem off moves the track onto its
  stems at once (every stem on, so it sounds as the file) and takes the
  stem off at `at`; all on again plays the stems to `at` and then moves
  back to the file. The time is held to the output frame it is heard at
  as the track plays when the change is asked for: a `Jump` in between
  (a bar looped) does not move it, so it comes then, wherever the track
  is. A time already made, or one the route cannot be met by, is taken
  as soon as it can be. `SetSettings` with other stems, a seek or another
  track drop a change still to come.
- **Settings** (`SetSettings`): when the track being played takes another
  route with them, it moves to that route without a break. The new route
  opens a little before the frame being written (1 s for stems, 0.1 s for
  a resampler) and decodes ahead while the old one plays on, in
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
  moves back onto its own route, bit-perfect again where it was before
  (with stems off, onto a fresh stems route: its stems follow the
  settings again, no longer the transition's fades).
  Without a separator the stems are the whole track (`stems::whole_mix`),
  so every plan is a beat-matched fade of the whole tracks.

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

  Stems
  switched at any time go into its fades, on both tracks, at the time
  they are heard at (a bar, say: the next track makes its output only a
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
  as far after the target as it is late, so the beat goes on. A stems
  route mixes the stems that are on as its buffer is made (so a stem
  switched follows at once), and its buffer keeps only what the output
  may give back: its stem run keeps the separated stems instead, `back`
  seconds behind what it mixes (and the output's ring) and `ahead`
  seconds taken from the separation ahead of it (some 1.4 MB a second).
  A jump there mixes the target from them with the stems on now, the
  resampler to the device's rate started again 0.1 s before it on its
  exact grid (what it makes for that is dropped), so the output is what
  the track plays as there, heard as soon as on the file. With
  `KeepStems(true)` the playing track is separated while it plays its
  file too (a separation running alongside, the stems following what is
  written), and the stems route a stem switched off moves to takes those
  stems over, so the bars before the switch are there for a jump back
  (and a switch back to the file keeps them for the next time). A target
  outside all that is read again from a little before it (as a switch's
  new route is), while the track plays on, and takes over once it has a
  seam's worth: later (about half a second on stems with a fast GPU),
  never broken. No window while a transition is set up (it mixes into the
  buffer as it is made); arming one reads the track again from what is
  written. `player/examples/jumps.rs` plays jumps on a real file in real
  time and reports when each was heard and the longest digital silence.
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
gets without one, and `PlayAt` starts at the frame a third in. `tests/switch.rs` switches routes mid-track: at one
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
