# jodel-output

Bit-perfect PCM output to audio devices. The crate opens a device
**exclusively**, at exactly the stream's rate, in an integer format the device
takes natively, so the samples reach the DAC unchanged: no OS mixer,
resampler or software volume in between. It works for any integer PCM.

| OS | Backend |
|---|---|
| Windows | WASAPI exclusive mode, event-driven (`windows` 0.62) |
| macOS | Core Audio HAL: hog mode, mixing off, physical format at the nominal rate (integer; a float-only device, such as the built-in speakers, as 24-bit samples converted exactly to float), our own IOProc (`objc2-core-audio` 0.3) |
| Linux | ALSA `hw:` devices, resampling off, exact rate (`alsa` 0.12) |

There is no shared-mode fallback: a device that cannot be opened exclusively
in a requested format is an error.

## API

- `devices()`, `default_device()`, `find_device(query)`: the output devices
  (`Device { id, name, is_default }`).
- `volume(id)`, `set_volume(id, level)`: the device's own volume control
  (`Volume { level, db, muted }`), the one the system's slider moves and
  the device applies after it has the samples. macOS only: its volume keys
  do not reach a device held in hog mode, so the player sets it; on
  Windows and Linux the system's keys reach the device, and `volume` says
  `None`.
- `capabilities(id)`: the rate × channels × format combinations the device
  accepts exclusively, probed without playing anything.
- `open(id, &OpenRequest::new(rate, channels))`: opens exclusively in the
  first of `OpenRequest::formats` the device accepts (default order: S24in32,
  S32, S24in32lo, S24_3, S16). `choose_format` is the rule, as a pure
  function.
- The `Output` trait: `write` (blocking with a timeout, into a ring the
  backend drains from its own thread), `drain`, `position` (stream frames
  played, from the device clock), `latency`, `pause`/`resume`, `flush`,
  `retract` (takes back frames written but not yet taken by the device,
  keeping three device buffers queued, without stopping it: a jump heard at
  once with no gap; `written` goes back by as much), `stats` (underruns,
  clipped samples), `clock()` (a `Send + Sync` view for other threads),
  `close`.
- `FileOutput` (the exact device bytes to a file, memory or any writer) and
  `NullOutput`, with a simulated clock (`SimClock::Instant`, `Manual`,
  `Realtime`), for tests.
- `Output::set_tap(Option<Box<dyn Tap>>)`: code the device thread shows
  every buffer it hands the device, after the ring's frames are copied out:
  the buffer unpacked to 24-bit values in a scratch of its own, with the
  stream frame it starts at. A tap only reads; the buffer is never packed
  again, so the device gets exactly the bytes written, tap or not. The
  device thread takes a new tap at its next buffer without a lock and hands
  the old one back, so it is dropped on the caller's thread. An output keeps
  its tap until it closes; set it again after a reopen.
- `OutputTarget`: something outputs are opened on (`name`, `capabilities`,
  `open`, and `lanes`, 1 unless it takes lane streams): a device, a
  mixer's input, a test's memory.
- **Lanes** (`jodel_output::lanes`): one stream carrying a player's outputs
  side by side, so they share one ring, one write head and one timeline (a
  seek, a flush, a retract act on all of them). `OpenRequest::lanes(n)`
  asks for `n` (1, a plain stream, by default); `Output::lanes()` says what
  an output took. A frame of `n > 1` lanes at `channels` channels is
  `lane_width(channels, n)` = `channels × n + 1` samples:
  `[main ch…][part 1 ch…]…[part n-1 ch…][control]`. Main is the player's
  output as it would be alone (the file, bit for bit); the parts together
  make it (stems; the crate knows them only by number) and are never passed
  through; the control word is a `Parts` (`None` 0, `Present` 1, `Ending`
  2): whether the parts are valid on that frame. The ring holds lane
  streams in `LANE_SAMPLE` (S24-in-32) whatever the device's format, so main
  reaches the device exactly as from a stream of its own; `Output::format`
  stays the device's (what main is written for), and `Core` strides and
  packs by the ring's format (`PackMode::Exact` checks main against the
  device's format too: it may reach the device as written). Every target
  checks the lanes it is asked for (`OpenRequest::check_lanes(offered)`):
  a device takes one, and more is `InvalidArgument`, as from
  `FileOutput::open`; `FileOutput::open_lanes` takes any and plays the
  wide stream as written, so a test sees every lane. `SWITCH_FADE` (10 ms)
  is how long a mixer switches between main and the parts, and
  `ending_frames(rate)` how long after a frame that says `Ending` the
  writer keeps the parts valid (rounded as the mixer rounds its ramps).
- `Requantizer`: requantization to 24 bits (or 16, `quantize_to`) with
  TPDF dither (±1 LSB), saturating and counting clips; the mixer's, and
  the playback engine's resampler and lossy sources.

## Mixer (`jodel_output::mixer`)

Several players on one device. `Mixer::open(target, MixerOptions)` takes
an `OutputTarget` and runs one thread, `jodel-mixer` (MMCSS "Pro Audio" on
Windows, real-time on macOS), which opens the device at once at the desk's
format and holds it open until the mixer is dropped: `desk_format(caps,
rate)` picks stereo at `MixerOptions::rate` (`DESK_RATE`, 48 kHz, unless
set), else 44.1 kHz where the device lacks it, else the lowest rate the
device takes (one that takes only 96 kHz runs at 96 kHz), in the format of
`SampleFormat::PREFERENCE` with the most bits it takes there;
`Mixer::format()` tells it. Every output on an input plays at that
format (each player resamples to it), so the device is never opened again
for a song, and a song at the desk's rate reaches it bit for bit. Another
device or another rate is another mixer. A device that fails is opened
again by the next output opened. `mixer.strip(name)` makes a
`Strip`, a channel of the desk, and `mixer.strip_in(&group, name)` one in
a group, another strip (clones are the same strip; it lives while a clone,
a child or an input on it does). `mixer.input(&strip, role, gain)` plugs
an input into it: an `InputTarget`, where one player opens its outputs as
on a device (opening another closes the one before), and a `Fader`, its
gain, which any number of hands may hold. Each output opened is a full
`Output` in the desk's format (its own ring, `pause`, `flush`,
`retract`, `drain`, clock), so the engine does not know the difference;
its `StreamInfo` carries the device's backend, exclusive flag and latency.

- **One mix path**, a device period (10 ms) at a time, as fast as the
  device takes frames. A device that plays in real time
  (`StreamInfo::realtime`, set by each backend, and by `FileOutput` for
  `SimClock::Realtime`) takes the next period as soon as it has room, all
  along: silence where nothing plays (it is never paused), and an output
  short of frames is silence for the rest of it (its underrun).
  The mixer does not wait for a late output: its own ring (the player's
  half second) is what rides out hiccups, so one short of a period is far
  behind, while the device's short ring must stay full or every input
  glitches when the mixer thread wakes late (waiting while the device had
  frames to spare was tried, and underran on macOS).
  One that plays only what it is given (a file, a test's memory) gets a
  period once every running output holds one, or is draining; so the
  tests run the same path. On it, a period no output gave a frame to is
  not mixed: the outputs it was sized for stopped since (closed, paused or
  flushed between the mixer's look at their rings and its take), and
  mixing it would play a period of silence as if written. With nothing
  running the thread sleeps until an output starts (or a handle asks
  something). Nothing of an input with no output running is heard, so a
  strip's or a fader's ramp under way or due now ends at once (a mute set
  before an output's first audio is in force from its first frame); a
  fader's ramp a crossfade's other side waits on keeps waiting.
- **Two gains:** an input is heard at its fader's gain (the controllers':
  a preview's crossfade) times its strip's (the user's:
  `Strip::set_level(gain)` linear, `set_mute(bool)`, `set_solo(bool)`,
  solo in place: while any strip is soloed, the others are silent). A
  strip's change is a 20 ms linear ramp (`STRIP_RAMP`), so it never clicks;
  a strip starts at 0 dB and is kept nowhere.
- **Groups:** a group acts on its children as a VCA does, and may have
  inputs of its own (`Strip::parent()` names a child's). An input is heard
  at its fader times its strip's path: the product, over the strip and
  every group above it, of the level (0 while muted), times the solo gate
  (1 while nothing is soloed anywhere, or the strip or a group above it
  is; else 0). Soloing a child silences its siblings, its group's own
  inputs and every other strip, but it is still heard through its group's
  level and mute; soloing a group keeps all its children. Each strip
  runs three ramps: `heard`, to its level × mute × gate, for its own inputs
  (a strip at the top of the desk has only this one, as before groups),
  `vca`, to its level × mute, for its children's, and `lead`, as `heard`
  but with the gate open also while a child's is, for the main of a lane
  input on it (below). So a mute and a solo set together are one ramp,
  never two multiplied into a bump.
- **Scheduled changes:** `Strip::set_level_at(gain, start)` and
  `set_mute_at(on, start)` take a `Start` as a fader's ramp does
  (`Start::Now`; `Start::At { epoch, time }`, never once the device has
  opened again; `Start::FirstAudio`, the first audio of an output opened
  after it on an input of the strip or of its children). One for a time
  starts its 20 ms ramp on that very desk frame, inside the period mixed;
  a frame already gone is the next mixed. Each setting has at most one
  change waiting, and a new one (now or later) takes its place.
  `Strip::level()` and `mute()` are the settings in force: a change now
  at once, a scheduled one once its frame is mixed. `pending_level()` and
  `pending_mute()` are the changes waiting, so a caller toggles against
  what is meant (the pending one, else the one in force). The handle
  writes what it sets at once; the mixer thread writes each setting again
  as it takes a change of it, so they end as the mixer has them (a start
  that never comes takes its change off again).
- **Lane inputs:** `mixer.input_lanes(&main, &[&part, …], role, gain)`
  gives the target, the input's fader (over every lane: a duck) and one
  fader per part (the controllers' gain of that part); `mixer.input` is the
  one-lane case. Every part's strip is a child of main's (D3: main's group
  level and mute act on the parts as on main). The target offers
  `lanes()` = 1 + the parts and opens 1 lane (main alone) or all of them; a
  plain input refuses more than one.
  Lanes are the units heard: main is heard at the input's fader ×
  (1 − s) × its strip's path through `lead` (a part soloed stands for main
  while main is heard in the parts' place: no parts yet, or none at all;
  once they are heard, each part's own gate applies, and a plain input on
  main's strip, such as a sync's clicks, stays gated as ever), part k at
  the input's fader × s × part k's
  fader × its strip's path, where `s` is the switch between main and the
  parts, linear over `SWITCH_FADE` (10 ms: main and the parts' sum are the
  same signal, so equal power would bump). `s` goes to 1 when the parts are
  touched (a part's fader target times its strip's own level and mute is
  not 1, or its strip's solo gate is not main's) and the control word says
  `Present`; it comes back to 0 only once nothing is touched and every
  part's ramps (its fader's, its strip's, main's strip's) have settled by
  the end of a period, which is how main returns bit for bit. From a frame
  that says `Ending` it fades to 0 (the writer keeps the parts valid that
  long); on one that says `None` it is 0 at once. The control word's
  changes, and the parts touched (by a strip's change landing, a fader's
  ramp starting), land on their very frame inside the period mixed. A
  frame the stream does not supply (not running, kept out by a flush, run
  dry) has no word: nothing of the input is heard there and the switch
  keeps what it knew; on the first frame after such a gap (a seek, a
  resume, a new output, an underrun, the very first) `s` is at its aim at
  once, so a stem muted stays muted across a seek. While `s` is 0 and the
  parts are present, each part's strip still meters its part (at what it
  would be heard at switched in), and its group meters main alone.
  `Fader::at_stream(stream_frame)` is the `Start` of that frame of its
  input's output, as it plays now: the input's map (the same from any of
  its faders) of where its last frames mixed went on the device, read
  through a sequence lock; `Start::Now` until frames of the output are
  mixed after it is opened or flushed.
- **Bit for bit:** a lane is heard when it supplied frames at a gain other
  than 0 (a lane at 0 for the whole period is not, whatever ramp runs
  under it). When one exact lane is heard (main, or a plain input's) at
  gain exactly 1 (faders, switch and strip path) with no ramp running, its
  samples go to the device untouched (a starved or silent input changes
  nothing); a part never does, so an S16 device gets parts mixed and
  dithered while main, 16-bit content in its S24-in-32 lane, still passes
  through exactly. `Mixer::passthrough()` says whether the last frames
  did. Otherwise the sum is in `f64`, then a look-ahead
  limiter (2 ms; it holds that much back while mixing) turns the gain down
  smoothly where the sum would go over full scale, only where the gains
  add up to more than 1 (a lone fade is never touched); then one
  requantization to the device's bits (16 or 24) with TPDF dither at its
  LSB; clips are counted over every opening of the device
  (`DeskView::master.clipped`).
- **Links:** `follower.follow(&leader, origin)` (faders of two inputs of
  one mixer, each with an output open) has the follower's stream frame 0
  land on the leader's stream frame `origin`, from inside a period, and
  from there move with the leader's stream, never with the device's
  frames (`mixer/follow.rs`): each period the follower fills exactly the
  frames its leader gave, at the same offsets. A leader short of frames,
  paused or flushed holds its follower, which loses nothing (silence
  where they were, the device playing on). A follower that is late drops
  its frames up to the
  frame due, an absolute stream frame, counted (`Fader::dropped`,
  `Stats::dropped`) and counted as played in its position; a leader's are
  never dropped. The follower's `at_stream` is its leader's from `origin`
  on, exact before it has been mixed, and a fade for its first audio
  starts on `origin`. Offline (a device that plays what it is given at
  once) no period takes the leader past `origin` before the follower has
  begun, so a render is the same however the writers are scheduled. A
  leader at its end (drained) lets its followers play on by themselves.
  A flush or a close of either output, a new output on either input, or
  `Fader::unfollow` ends the link; a leader follows no one and a follower
  leads no one. A player's two heads use it for a transition (the next
  track follows the playing one).
- **Automation:** `Fader::automate(Automation)` gives an input gain
  curves (`GainCurve`: breakpoints `(stream frame, gain)`, equal-power
  between them, a cut smoothed linearly over `CUT_FADE`, 10 ms) on the
  stream frames of its output: its own over every lane, main's (heard on
  main only), and one per part, on top of every fader and strip. They
  move only as the stream does (a pad or a pause holds them). A set is
  replaced whole, or cleared (`Fader::clear_automation`), each gain going
  linearly from where it was heard over `CUT_FADE`; it ends with its
  output (a set given with none open is the next one's). A part's curve
  away from 1 has the input heard on its parts; main's curve never does,
  and the input's own never keeps them. Once every curve is past its end
  at exactly 1, the input is heard bit for bit again. The mixer thread
  evaluates the curves without allocating; `InputView::automation` shows
  their gains and `InputView::dropped` a follower's late frames.
- **The desk:** `Mixer::desk()` gathers on the caller's thread what the
  mixer thread publishes, and looking changes nothing: the master (the
  device's name and `StreamInfo`, or none while closed; the `Timebase`,
  passthrough, clips), the strips in the order made, a group before its
  children (id, name, parent, the level, mute and solo in force, and each
  input's role, fader gain, whether it was heard, its lanes as
  `LaneView { strip, gain, heard }` (the gain each was heard at), the
  switch of one with parts as `PartsView { state, share }`, its output's
  format, ring fill and capacity), and the microphone's port
  (`MicView { port, refused }`). The per-input state, passthrough and clips
  are atomics; the clock sits behind a lock the mixer takes every turn for
  an instant, what the device runs at and the microphone's port behind
  locks of their own, taken only when they change; the strips' list is the
  handles' own.
- **Meters:** `Mixer::meters()` takes the peaks since the last look: per
  channel of what goes to the device, and of each strip's subtree (the
  lanes on it and on its children) summed after their gains (by strip id), with
  whether the limiter acted. Peaks are
  linear (1.0: full scale), up to 8 channels (`METER_CHANNELS`); the mixer
  raises them with `fetch_max` in atomics and a look takes them with
  `swap`, so it is for the one who shows them.
- **The clock:** `Mixer::timebase()` gives `Timebase { epoch, rate,
  frame, heard }`: the device's opening is an epoch with its own frames
  (another one only where the device failed and was opened again). `Fader::set(Ramp { start, to, over, curve })` changes the gain
  without blocking, in place of a ramp still waiting: from `Start::Now`
  (the next frame mixed), `Start::At { epoch, time }` (a time of an epoch;
  never, once the device has opened again), or `Start::FirstAudio` (the
  first frame of an output opened on the input after the ramp is set: a
  fade in from what is loaded next), over `over` (rounded to frames at the
  device's rate), equal-power (`Curve::EqualPower`, `equal_power(a, b,
  t)`: the square of the gain moves in a straight line, so opposite
  ramps keep their summed power between any two gains) or linear;
  `Ramp::set(g)` is at once.
  `Mixer::crossfade(into, start, out, over)` fades `out` down on the very
  frame `into` starts up. A ramp on an input with no output, or a paused
  one, ends at once (nothing of it is heard); one held back for a flush
  or a prefill goes on from where it is. `Fader::gain()` and `settled()`
  say how it stands.
- **A short device buffer:** the device holds `MixerOptions::buffer`
  (40 ms); the outputs keep the long rings the engine rides out decoding
  hiccups with, so a gain change is heard within 40 ms. An output's
  `position` leaves out what the device has not played yet (the mixer
  tells each output what of it is still downstream), so positions, cues
  and karaoke stay on what is heard. A `flush` (a seek) drops what the
  device holds as well unless another input is heard.
- **One format:** no resampling in the mixer. Every input, Lead or
  guest, is offered the desk's one format (`InputTarget::capabilities`),
  and an output asked at another is refused (`Error::FormatNotAccepted`).
  The roles say only which input the microphone's frames are told in:
  `Fader::set_role` changes one as it plays, the microphone's tap going
  to the newest Lead output; the only Lead with an output open cannot be
  made a guest.
- **The microphone's port:** `Mixer::set_mic(Option<Input>)`. The mixer
  taps the device for the microphone while a Lead output is open, again for
  each one, so its echo canceller hears the whole mix as its sidechain, at
  the desk's one rate all along (what it has learnt of the room is kept
  from song to song); a frame map of the Lead's chunks tells it the
  stream frames in the Lead's own count. A device that runs no taps is
  told in `Mixer::mic()` (`MicPort::CannotHear(why)`, each time counted
  in `refused`). A microphone replaced or taken away is let go of on a
  thread of its own, never on the mixer's (its last handle stops capture
  and waits for its cleaner). The microphone has no strip and is no input, so nothing of
  it can reach the device. Outputs on inputs run no taps (`set_tap` says
  `Unsupported`).
- **Monitor:** `Mixer::set_monitor(Box<dyn Monitor>)` sees every frame as
  it goes to the device (`opened(info, clock)`, `mixed(samples, channels,
  first_frame)`, `closed()`), on the mixer's thread: the player's
  spectrum.

## Microphone (`jodel_output::input`)

For karaoke: the singer heard, never played. Nothing of the microphone
reaches the output; it is listened to, to measure the singing, so the
music's echo is taken out of it and the voice kept with where in the stream
it was sung.

- `input_devices()`, `default_input()`, `find_input(query)`: the capture
  devices.
- `open_input(id)` / `open_input_with(id, Mono)`: opens one in shared mode
  at its own rate, a period of about 5 ms where the system allows it, and
  captures on its own thread into a lock-free ring of `f32`, mixed down to
  mono (`Mono::Auto`: one channel as it is, the mean of two, the first of
  more; or `First`, `Mean`, `Channel(n)`).
- `Input` (cheap to clone): `peak` (a meter of the voice as captured),
  `voice_peak` (one of the singer isolated, below), `take_voice` (the voice
  the canceller and the suppressor give, as the game hears it,
  `VoiceBlock`s of 128 samples with the stream frame each was sung at, the
  last 11 s or so kept), `only_silence(after)` (capture has run that long
  and every sample was exactly zero: a muted input, or on macOS a process
  without the microphone permission), `status` (`MicStatus`: fill and
  target, the clock correction, underruns, overruns, resyncs, blocks
  dropped), `failure`, and `tap()`, the `MicTap` to pass to `set_tap` (the
  mixer does, `Mixer::set_mic`).
  Capture stops when the `Input` and its taps are dropped.
- `MicTap`, in the device thread, resamples the voice to the output's rate
  (up to 352.8 kHz and beyond; cubic, varispeed) and lines it up with what
  the buffer plays, left and right (a mono output's channel on both; of
  more than two, every channel after the first two on both halves alike),
  all brought down to about 44.1
  or 48 kHz (the output's rate divided by 1, 2, 4 or 8), and queues them in
  blocks of 128 for the cleaner, a thread of its own. A control loop steers
  the ratio within ±0.2 % of nominal to hold the ring's fill at two capture
  periods. It measures once per device buffer, counting the time since the
  last capture packet, so it sees the fill as if the voice arrived
  continuously. The voice starts after both sides have run 100 ms, drops
  out on a dry ring (counted) until the target is queued again, and a fill
  held more than a period over its target (a capture burst, a device-thread
  stall) is skipped at once. Capture frames the ring has no room for are
  counted as overruns only while the loop reads it: before (no tap yet,
  warming up) the ring keeps the first second that came and drops the
  rest, and the loop starts from the newest of that.
- The echo canceller (`aec`, on as an input opens; `Input::set_echo_cancel`),
  on the cleaner's thread. The rule above the others: the singer is never
  taken out. It has three parts:
  - a delay finder, matching the microphone's bins against the music's at
    every delay up to 1 s, their phases only (matched by how the level
    moves in bands, as before, it was a beat off as often as not: a song's
    loudness repeats every beat), moves the filter's window to where the
    echo starts when it does not reach it (a wireless speaker); once
    learnt, the window follows the echo's loudest part only where the
    finder hears the echo too (a filter that had learnt something else
    dragged it off the echo for good), and the delay it tells is the
    finder's;
  - a partitioned-block frequency-domain adaptive filter (200 ms of echo)
    on two channels of what plays, what left and right play alike and what
    they play apart, normalised per frequency by the inverse of their 2 × 2
    power: a laptop's microphone sits nearer one speaker, and from the mean
    of the two the best linear filter took only 6 to 7 dB of a MacBook's
    echo out, however long, whatever model of its speakers' distortion,
    from left and right apart 13 dB (a song in mono leaves the second
    channel silent, and the canceller works as one on the mean would, but
    for a little regularisation, the gain below and the finder). Each step
    is divided by how many channels a frequency has (one to two: a step
    along both takes out twice what one along one does, and lets twice as
    much in of a voice the detector missed), what the two play apart
    learns only where it holds more than 3 % of a frequency's power (30 %
    under about 400 Hz, where left and right play nearly alike: weaker, it
    learns mostly chance, and the bass kept 6 of its 10 dB), and until the
    filter's window has filled (the canceller started with the music
    already playing) its steps are taken as if it had (by the few blocks
    held alone they overshot the echo by up to 10 dB for a second). The
    estimate of the filter in use takes a gain that
    follows what of it the microphone holds (a speaker's limiter riding the
    music's loudness: as a passage 7 dB louder began, a MacBook played it 2
    dB louder, and the filter learnt before overshot by 8 dB), only ever
    down, its chance wobble shrunk away. Two filters: a shadow that learns,
    and the filter in use, which copies the shadow only once it leaves
    clearly less, and not while someone sings. While someone sings the
    shadow learns at a tenth of its pace (a held note would draw it towards
    taking the note out; held still, with a voice sung all through it never
    learnt the echo between the notes it missed). Singing is told by its
    pitch (`aec::singing`):
    every 6 ms the leftover and the estimate, low-passed and taken down to
    about 11 kHz, are looked at for the lag where each is most like itself;
    where the leftover is clearly pitched (0.8) and the estimate is not at
    that lag (under 0.4; the music's own notes, in the leftover too, are as
    clear there), over the leftover's own floor (a hum is pitched, but
    steady), in three of the last eight looks, someone sings, held for
    0.2 s; or where the leftover is 7 times what the filter usually leaves.
    By its level alone (ten times, as before) a singer 8 dB under a
    MacBook's echo was never told apart, and the filters learnt from them.
    A room that changes while the
    detector takes it for singing (the volume turned up: the leftover jumps
    as a voice's would) is learnt by a trial filter apart, which takes over
    only once it has left a quarter of what the filter in use leaves for a
    second running: a room that changed does that soon, a voice never. The
    filters learn along the music prewhitened (each frequency's steady part
    taken out, so a held tone does not leave the rest of its band
    unlearnt). A filter in use worse than the microphone for 0.2 s running,
    its gain not mending that, gives way to the shadow or the trial if
    either does better, else everything starts over (started over within
    30 ms, it was learning again every few seconds on a MacBook);
  - a residual suppressor (`aec::Suppressor`) after it: per frequency, the
    share of the canceller's estimate that its leftover is, learnt where
    only the echo is heard (and again for a new filter), takes the voice's
    bin down by what that share accounts for (a gain of √(1 − 1.5 ×
    leftover/power), never under −20 dB), in frames of two blocks under a
    √Hann window, one block late. It is for the pitch the game hears: where
    the singer is the louder a bin passes nearly whole, where the leftover
    is, it goes. A singer under the echo loses some of their level to it:
    through `aec_room`'s laptop (five songs, the echo 10 dB over the singer)
    0.1 to 0.9 dB through a linear speaker and 0 to 5.5 dB (median 0.2)
    through a limiting one, for 0.2 to 11 dB more of the echo out where
    nobody sang; on a MacBook's real recordings with a separated voice
    mixed in, 2 to 5 dB (the voice 0 to 10 dB under the microphone's
    echo), for 5 to 7 dB more out, the game's pitch then found on 1.3 to 7
    times as many of the voice's frames. The level matters not to the
    pitch, only how far the voice stands out.
  The canceller's own voice is the microphone less the echo predicted,
  nothing more taken out; `take_voice`, the meter and `record_echo` have the
  suppressor's. `Input::record_echo` writes what the canceller works with
  for some seconds to a WAV file (five float channels at its rate: the
  microphone, what plays left and right, the canceller's voice, the voice
  suppressed), queued by the cleaner and written by a thread of its own,
  for tuning it to a room.
  `Input::echo_status` says how much echo goes, whether it has learnt, and
  how late the echo comes.
- The singer isolated (`isolate`), after the canceller on its thread, for
  the meter: what the canceller leaves of the band (through a laptop's
  speakers most of it) and the room's own sounds (a fire, a fan) taken
  out. A small network trained on singing (0.87 M weights, built into the
  binary as f16; about 1.5 % of a core) hears 16 kHz (`Downsampler`: the
  filter its training material was resampled with) in 20 ms frames every
  10 ms, each against the level of the last two seconds so a singer close
  to the microphone is not taken for the band for being loud, and gives a
  gain per frequency and how likely a voice is there; the gains applied,
  overlap-added back, about 11 ms late plus a hop's wait. On 24 songs by
  artists it never heard: a clean voice kept (median −0.1 dB), the band
  the canceller left down 23 dB, an echo never cancelled 24 dB, a fire 40
  dB; trained to lose the voice less rather than take out more band, a
  voice under that band 5 dB louder than itself loses 1.8 dB (11.5 dB in
  its worst 5 %). It hears the voice suppressed, not what played.
- `Input::feed(rate, period)` gives an input fed by the caller (`MicFeed`)
  on a simulated clock, its cleaner run by the caller (`MicFeed::clean`),
  for tests with `FileOutput`.

| OS | Capture backend |
|---|---|
| Windows | WASAPI shared, event-driven, at the endpoint's mix format; `IAudioClient3` for a period near 5 ms, else the engine's default (10 ms) |
| macOS | Core Audio HAL input IOProc on the device's first input stream, interleaved or not, float or integer. On the device an output of ours holds in hog mode the IO buffer is left as the output set it, and a listener follows its nominal rate when the output reopens. Needs the system's microphone permission (otherwise silence) |
| Linux | ALSA through its plugins: `default` and `plughw:CARD=<id>,DEV=<n>`, rate resampling off where allowed, float where the plugin converts |

Samples are `i32` words holding signed 24-bit values, right-aligned: what the
playback engine hands over. `pack` documents the layouts and what happens to
a value that does not fit (outside 24 bits, or low bits for 16-bit output):
`PackMode::Saturate` (default) clamps and counts it in `Stats`,
`PackMode::Exact` fails the write. Nothing ever wraps.

## Examples

```sh
cargo run -p jodel-output --example devices                       # list devices and the exclusive capability matrix
cargo run -p jodel-output --example silence -- default 48000 2    # 2 s of digital silence, exclusive
cargo run -p jodel-output --example silence -- default 96000 2 --controls   # also pause, resume and flush
cargo run -p jodel-output --example coreaudio_info -- --tone default        # macOS: raw formats and volume controls, then a -20 dBFS tone
cargo run -p jodel-output --example mic                                      # list outputs and inputs
cargo run -p jodel-output --example mic -- default default 48000 30          # 30 s of silence, the default microphone tapped beside it
```

`mic` takes `--mono first|mean|auto|<n>` and `--every MS` (how often it
prints the fill, target, clock correction, the peak as captured and as
cleaned, and the counters). The microphone is never played.

`silence` only ever writes zero samples. `<device>` is `default`, an id, or
part of a name.

## Testing

`cargo test -p jodel-output` covers the packers, format negotiation, the ring,
the position timeline and the shared handle logic (start threshold,
backpressure, underrun, pause, flush, drain, close) through `FileOutput`.
The mixer's units: ramps land on their frames (an equal-power pair keeps
its power, a new ramp starts where the last one is, one on nothing
mixed settles), two inputs sum and are dithered once, a lone fade of a
full-scale master is never limited, a sum over full scale is turned down
smoothly ahead of its peak and never clips, a 16-bit device gets 16-bit
samples dithered without bias, and the tap's frame map. `tests/mixer.rs`
runs it in real time on a simulated device: an output's position is what
the device has played (within a period), a ramp lands on its frame with
everything before it bit for bit, a crossfade's two sides start on the
frame the guest's first audio is, a starved guest at gain 1 leaves the
Lead bit for bit, a flush drops what the device held, the microphone's
port is told the Lead's frames while a guest played first, hears song
after song on the one opening of the device and never reaches it, the
monitor sees two inputs summed (the device open after them), a fade out
on an input with nothing to play settles, a seek keeps a fade under way
and a pause ends it, every input (Lead or guest) is offered the desk's one
format and another is refused, a closed input cannot open again while the
device plays on, a device not there as the mixer starts is opened by the
first output once it is, and a device that fails fails its outputs and is
opened again by the next (another epoch, at the same format, a ramp under
way ending on its target and a change for the last epoch never coming).
Strips: one at 0 dB keeps its input bit for bit, a level change is a
20 ms linear ramp, mute and solo silence strips without a click, and the
meters read each strip and the sum, then nothing once taken.
`tests/desk.rs`:
a group's level and mute act on its child's input and its own, ramped from
the frame set; a child's solo leaves it alone, a group's solo keeps its
children and its own input, and solo off brings all back; a mute and a
level set for desk frames off the periods start their ramps on those very
frames (one set before them taking the place of another), the device
getting the input bit for bit up to the period the change falls in and
again after; a group mute for the first audio waits while an older
child's output plays and starts on the first frame of one opened since on
another child; a lone input on a child
of a group, all at 0 dB, reaches the device bit for bit, and again bit for
bit once a group's ramp to -6 dB and back has played out; and a group
meters its subtree. The units in `mixer/strips.rs` check the path rule
with a group in a group, solos on a strip made late or gone, a solo
released and a mute set together staying silent (and 5 ms apart, one
straight rise and one straight fall), two scheduled changes multiplied
within one period, the pending changes shown, the first audio over a
subtree, and a change to where a strip is keeping its path flat;
`mixer/envelope.rs` multiplies flat and per-frame gains. Where frames must
line up exactly, the tests run on
a device that takes what it is given at once, with outputs started short
of a period, so the mixer waits for all of them; `tests/common` holds the
simulated device (in any sample format), the microphone fed in real time
and the helpers the files share. `tests/lanes.rs` writes a lane stream
(main `x`, each of four parts `x/4`, a control word per test): untouched
with the parts present, the device gets main bit for bit; a part's strip
muted switches to the parts' sum from that frame (the switch and the
part's ramp as computed), and unmuted, the switch holds until the part's
ramp has settled and then main is bit for bit again; a part's strip at
-6 dB is heard at it; the control word's `None` (main however touched),
`Present` off a period (the switch from that very frame) and `Ending`
(back to main over 10 ms, then bit for bit); the input's fader scales
main and the parts; a part's fader below 1 switches, and back at 1 main
returns; a flush, a retract and a pause act on every lane; the
microphone is told the lane stream's frames; on a 16-bit device a soloed
part is requantized and dithered at 16 bits while main passes through
exactly; and `Fader::at_stream` gives the device frame of a stream frame
4 800 frames behind, on which a group mute starts. The units in
`mixer/parts.rs` run the switch over the control word's frames (and
`Ending` then `Present` again), hold it until the parts have settled, and
keep it across a gap in the stream, back at its aim at once; `lanes.rs`
checks the frame layout and `ending_frames`. Also in `tests/lanes.rs`: a
part touched from the first frame, across a pause, a flush and a new output
(real time) and across an underrun in the middle of the switch, never lets
a frame of main through; `Ending` then `Present` again switches back in;
on a 16-bit device main is bit for bit again once a solo is released;
odd lane widths (stereo, 1 to 3 parts; the units in `lanes.rs` lay out and
split lanes of 1 and 3 channels too); the parts'
strips meter the parts while main is heard; a part soloed keeps main heard
while the control word is `None` and is heard alone once it is `Present`;
targets refuse lanes they do
not take; a part's strip must be a child of main's; `at_stream` is `Now`
after a flush until frames are mixed again; and a level and a fade set
before the device opens are in force from its first frame.
`tests/link.rs` checks links sample for sample: a follower's frame 0 on
its leader's `origin` inside a period (a plain input and a lane input,
its parts too), a fade for its first audio starting there, one linked
after it gave frames starting where they are due; a leader short of
frames or paused holding it, nothing lost, the device pausing with the
leader; a leader at its end letting it play on; a late follower dropping
its frames up to where it is due, counted and in its position; offline
the same render however the writers run, and a follower paused after its
origin holding nothing up; a retract that writes the same frames again
keeping the link; a flush of either side or a new output on either input
ending it; links one level deep and only between outputs open; and a
follower's `at_stream` its leader's before it is mixed.
`tests/automation.rs` checks curves frame by frame: on a plain input (and
passthrough after a curve ends at 1), across a pause that parts stream
frames from device frames, on a follower across its leader's underrun
and pause, replaced and cleared mid-curve (smoothed, no step larger than
the smoothing's); on a lane input the parts' curves with the parts there,
main's without, a part curve through exactly 1 that keeps the switch,
an input curve that does not hold it, and main back bit for bit; a set
ending with its output and one given before an output opens; and roles
(a guest made Lead takes the microphone's tap and is asked again, no new
opening, the only Lead playing refused). `tests/automation_alloc.rs`
counts what the mixer thread allocates and frees while every curve
moves: nothing.
`tests/mic.rs` runs the microphone path on it: a tap (a probe, a microphone
tapped and singing, its canceller on or off, or none) never changes a byte,
in every format; a `MicFeed` tone comes out of the cleaner as it came in,
in blocks that follow each other 128 frames on; a running device swaps taps
without a lock; the loop holds a mic clock 0.1 % fast for ten simulated
minutes (44.1 kHz out, 48 kHz in) and 0.1 % slow at 352.8 kHz, within
0.05 ms of its target and 5 ppm of the offset; the voice comes back after a
capture gap, a device stall and a capture burst; exact zeros are told apart
from a quiet input; and the cleaner takes what plays, heard 30 ms later,
out of the voice (43 dB down once learnt), lets a voice sung over it
through (within 0.1 dB), and records five channels at its rate (what
played as the music, left and right alike from a mono output, the
microphone at its echo's level), in a WAV header other readers take
(WAVE_FORMAT_EXTENSIBLE, float, front left, right, centre, back left and
right, with a `fact` chunk). The cleaner's own tests: a click comes out in
the block stamped as the one it was heard in, at its sample, the canceller
on or off; at a gap the block held is dropped and the next run's first
given only with its successor. `aec`'s own
tests: a room's echo (45 ms, a 40 ms tail) down 25 dB, a voice over it
kept within 0.1 dB and 33 dB over what is left; a moved speaker learnt again; the volume turned up 12 dB
learnt again (28 dB down after; 2.5 without the trial filter); a note held
for four seconds kept whole, the filter still learnt after it; no
reference, the voice as it is; and a laptop's speakers (a limiter, soft
clipping, no bass) in rooms with a 0.15 to 0.4 s reverberation, the echo
40, 120 and 300 ms late (the last past the window, found by the delay
finder): with the music alone the meter's peak falls by more than 20 dB
(24 to 33), the delay is found within 15 ms, and a voice as loud as the
echo is kept within 0.5 dB, 20 to 41 dB over the rest. A singer as a singer
sounds (syllables of harmonics, consonants, soft ones) through such a room,
from twice as loud as the echo to 24 dB under it: no syllable loses
anything (each within 1 dB, the soft ones and the onsets too). Two
speakers heard apart (the microphone by the left one): 28 dB out from left
and right, 12.4 dB from their mean; music whose loudness repeats every 250
ms, its echo 300 ms late: found within 15 ms; a loud passage through a
limiter (the gain pulled down 8 dB): never started over, 35 dB out after;
the suppressor: 9.5 dB more of a laptop's leftover out, the singer kept
within 0.1 dB, and without an echo the voice exactly as it came. Singing
(`aec::tests::singing`): a singer 8 dB under a stereo song's echo through
a laptop's two speakers, for 14 s once learnt: the echo under them still
goes as before they sang (17 dB, 19 at the end; told apart by its level
alone, 8 at the end), the singer kept whole; the laptop moved under such a
singer (the echo 25 ms later, another way): 11 dB of the new echo out 5 s
on (before, 7); a steady hum 20 dB under the echo, pitched and nothing of
the music's: learnt as without it; two speakers each its own way to the
microphone: 14.5 dB out in the third second, 24 once learnt (before 17.7
and 27: a step along both channels at once was twice what it is now,
which on an echo without noise or a voice learns faster, and lets a voice
the detector misses in twice as much). `isolate`'s
own tests: the network streamed matches the trained one (PyTorch, the same
f16 weights) on a made-up signal within 1e-5 in its gains and presence, fed
in any pieces; a steady hiss goes (75 dB); the downsampler keeps 200 Hz to
6 kHz within 1 % and folds 12 and 20 kHz back under −46 dB. On two real
takes replayed (`aec_replay`), its output per second lies within 0.02 dB of
PyTorch's (median; the worst 1.2 dB, in near-silent seconds).

On real songs (`cargo run --release -p jodel-playback --example aec_room --
SONG VOICE.wav`: a song through a simulated laptop, a separated voice from
another song as the singer, the echo 10 dB over them; five songs): learnt
within 0.4 to 1.3 s, the singer kept within 0.4 dB in every one. Through a
linear speaker the voice stands 11 to 22 dB over what is left and 18 to
33 dB of the echo goes; through one with a hard limiter (its gain moving
with the music's loudness, which no linear filter follows) 9 to 12 dB go,
and what is left is about as loud as the voice. How hard a real laptop's
limiter is, a recording (`r` on the Karaoke tab) shows. Measured again on
five songs' separated bands (from `voice-train`) with the canceller as it
is now: through a linear speaker 14 to 33 dB of the echo goes (as before,
within 0.6 dB; one the canceller had failed on, 4.6 dB, now 14.2), through
a limiting one 5 to 15 dB (before 4 to 14), the singer kept within 0.2 dB;
the suppressor's cost and gain are above.

Recordings go through the canceller again with `cargo run --release -p
jodel-output --example aec_replay -- RECORDING.wav...` (one canceller over
all of them, as one session; `--fresh` for one each), second by second,
the canceller's voice written beside each as `RECORDING.wav.replayed.wav`,
the suppressed one as `RECORDING.wav.suppressed.wav` and the singer
isolated from that as `RECORDING.wav.isolated.wav` (16 kHz, as the meter
hears it, with the share of a core that took); it takes recordings of five
channels, older ones of three (the mean of left and right for both), and
files of two (microphone, music), a made-up session. Two later ones from
a MacBook (built-in speakers and microphone): the echo 80.7 ms late,
steady within ±13 µs; the best linear filter from the mean took out 5.8
and 6.6 dB, no model of the speakers' distortion (polynomial, soft clip,
limiter, before the filter) more than 0.3 dB besides; the canceller now
never starts over on them (before, four times in 30 s) and the suppressor
takes the echo about 14 dB down where nobody sang (the canceller alone
about 7). Five from a MacBook's own
speakers and microphone (the echo 81 ms late): 5 to 7 dB of the echo goes
(its bass the speakers do not play, and their limiter, which no linear
filter follows). On them the filter that learns grew beyond numbers within
a song, a held bass note's steps whitened many times too large, and a
canceller from one song never learnt the next; a step is now bounded
(`WHITE_GAIN_MOST`), and a filter beyond numbers is started over (tests: a
held band and a singer in its key; both filters spoilt on purpose). On a
made-up session (a real song's band through a made-up laptop, a voice
singing over its quiet start) a filter learnt from the voice over music
near silence met the music at full level, and the voice came out 45 dB
over all the microphone heard; an estimate 20 dB over the microphone's
power in a block (`ASTRAY`) is no echo, takes nothing out and is started
over, and a filter in use worse than nothing before anything is learnt is
dropped, the shadow learning on (test: both filters made 200 times too
loud). At 6 dB a working filter was started over too often (a singer's
sound against the echo's runs up to 9 dB over in a block): a song through a
laptop lost a third of what was taken out.

One of five channels from a MacBook's own speakers (left and right
recorded apart; the user singing 8 dB under the echo for most of its
30 s): the best linear filter from left and right took 13.1 dB of the echo
out (11.7 fitted on one half and tested on the other), from their mean 7.
The canceller replayed (`aec_replay`; the echo removed where nobody sang,
0.3 to 3.1 s and 17 to 20.4 s, then under 300 Hz; a second pass after the
first, as warm): before these changes 3.9 dB (1.5 under 300 Hz) fresh and
6.0 (0.9) warm, the suppressed voice 5.0 and 15.4; now 8.6 (7.9) fresh and
13.7 (10.0) warm, suppressed 9.1 and 16.2. Of the 581 moments where the
best filter's leftover has the user's pitch, the suppressed voice has it
on 2.8 % before (2.9 warm), now on 75 % (85 % warm). The 6.2 s without
singing looped five times: 6.4, 11.2, 11.9, 12.0, 11.4 dB a pass before
(the bass 3 to 6.5), now 8.6, 13.7, 14.0, 14.1, 14.3 (the bass 8 to 10);
10 dB of it went after 5.5 s before, now after about 3 s. With a
separated voice mixed in (16 voices, the pitch the game hears scored where
the user does not sing), 5 and 10 dB under the echo: the pitch found on
5.3 and 1.9 % of the voice's frames fresh before, 5.2 and 2.2 % warm; now
22 and 12 % fresh, 44 and 22 % warm (the best filter's leftover: 51 and
23 %, fitted on the other half 37 and 12 %). On the two older mono
recordings, mixed the same way and scored from 2 s: 12.7 and 5.2 % fresh,
13.9 and 5.6 % warm on one before, now 13.9, 5.0, 19.6 and 6.7 %; 6.4 and
1.4, 6.9 and 1.7 % on the other, now 6.6, 1.4, 7.6 and 1.6 %. The
detector costs about 1.4 % of a core (the canceller replays 30 s in 0.4 s
more).

The device paths need a device:

- **Windows:** `devices` and `silence` have been run on the developer's
  machine (see the phase report). `mic` has run there with the GC7's
  microphone input (WASAPI shared, 48 kHz, a 7 ms period from
  `IAudioClient3`) into the monitor's HDMI output at 44.1 and 48 kHz, voice
  level 0: no underruns on either side, the fill within about 0.5 ms of its
  14 ms target, the correction settling within a few hundred ppm. Not yet
  tried by ear: the input carried no signal (its peak stayed near
  −160 dBFS), and the output was not the DAC.
- **macOS** (not yet run on a Mac): on the MacBook, from the repository's root:
  1. `cargo test -p jodel-output` (includes the physical-format mapping test).
  2. `cargo run -p jodel-output --example devices`: every output device with
     its physical formats. Nothing is changed on the device.
  3. `cargo run -p jodel-output --example silence -- default 48000 2 --controls`,
     then the same at a rate the DAC lists above 48 kHz. It takes hog mode
     (other apps go silent for the run), sets the physical format, plays
     silence, and restores format, mixing and hog mode on close. The opened
     line says `float_io` when the IOProc format is Float32 rather than the
     integer physical format. A device whose physical formats are all
     32-bit float (the built-in speakers, many virtual devices) lists them
     as S24in32 and opens with `float_io`: every 24-bit value is a float
     exactly, so the samples reach it unchanged.
  The IOProc publishes the position under a short mutex; if that ever
  shows up as glitches, it is the first thing to replace with atomics.
  4. The microphone: `mic` lists the inputs; `mic -- default default 48000 30`,
     then the same with
     the input on the device the output plays on, if it has one (the hog-mode
     case), at two rates in turn. The terminal it runs in needs microphone permission
     (System Settings → Privacy & Security → Microphone), or the voice is
     silence and the peak stays at −∞.
- **Linux:** type-checked and unit-tested in WSL, which has no sound card.
  On a box with a DAC: `devices`, then `silence -- hw:CARD=<id>,DEV=0 <rate> 2`.
  A sound server (PipeWire, PulseAudio) holding the card makes the open fail
  with "device in use". The microphone (type-checked only): `mic` lists
  `default` and the `plughw:` inputs; `mic -- hw:CARD=<id>,DEV=0 default`.
