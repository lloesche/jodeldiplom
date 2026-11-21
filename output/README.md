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
  `open`): a device, a mixer's input, a test's memory.
- `Requantizer`: requantization to 24 bits (or 16, `quantize_to`) with
  TPDF dither (±1 LSB), saturating and counting clips; the mixer's, and
  the playback engine's resampler and lossy sources.

## Mixer (`jodel_output::mixer`)

Several players on one device. `Mixer::open(target, MixerOptions)` takes
an `OutputTarget` and runs one thread, `jodel-mixer` (MMCSS "Pro Audio" on
Windows), which holds the device; the device opens with the first output
opened on an input and closes with the last. `mixer.strip(name)` makes a
`Strip`, a channel of the desk (clones are the same strip; it lives while
a clone or an input on it does). `mixer.input(&strip, role, gain)` plugs
an input into it: an `InputTarget`, where one player opens its outputs as
on a device (opening another closes the one before), and a `Fader`, its
gain, which any number of hands may hold. Each output opened is a full
`Output` in the device's format (its own ring, `pause`, `flush`,
`retract`, `drain`, clock), so the engine does not know the difference;
its `StreamInfo` carries the device's backend, exclusive flag and latency.

- **One mix path**, a device period (10 ms) at a time, as fast as the
  device takes frames. A device that plays in real time
  (`StreamInfo::realtime`, set by each backend, and by `FileOutput` for
  `SimClock::Realtime`) takes the next period as soon as it has room: an
  output short of frames is silence for the rest of it (its underrun).
  One that plays only what it is given (a file, a test's memory) gets a
  period once every running output holds one, or is draining; so the
  tests run the same path. With nothing running the thread sleeps until
  an output starts (or a handle asks something).
- **Two gains:** an input is heard at its fader's gain (the controllers':
  a preview's crossfade) times its strip's (the user's:
  `Strip::set_level(gain)` linear, `set_mute(bool)`, `set_solo(bool)`,
  solo in place: while any strip is soloed, the others are silent). A
  strip's change is a 20 ms linear ramp (`STRIP_RAMP`), so it never clicks;
  a strip starts at 0 dB and is kept nowhere.
- **Bit for bit:** an input is heard when it supplied frames at a gain
  other than 0. When one is heard at gain exactly 1 (fader times strip)
  with no ramp running, its samples go to the device untouched (a starved
  or silent input changes nothing); `Mixer::passthrough()` says whether
  the last frames did. Otherwise the sum is in `f64`, then a look-ahead
  limiter (2 ms; it holds that much back while mixing) turns the gain down
  smoothly where the sum would go over full scale, only where the gains
  add up to more than 1 (a lone fade is never touched); then one
  requantization to the device's bits (16 or 24) with TPDF dither at its
  LSB; clips are counted over every opening of the device
  (`DeskView::master.clipped`).
- **The desk:** `Mixer::desk()` gathers on the caller's thread what the
  mixer thread publishes, and looking changes nothing: the master (the
  device's name and `StreamInfo`, or none while closed; the `Timebase`,
  passthrough, clips), the strips in the order made (id, name, level, mute,
  solo, and each input's role, fader gain, whether it was heard, its
  output's format, ring fill and capacity), and the microphone's port
  (`MicView { port, refused }`). The per-input state, passthrough and clips
  are atomics; the clock sits behind a lock the mixer takes every turn for
  an instant, what the device runs at and the microphone's port behind
  locks of their own, taken only when they change; the strips' list is the
  handles' own.
- **Meters:** `Mixer::meters()` takes the peaks since the last look: per
  channel of what goes to the device, and of each strip's inputs summed
  after its gain (by strip id), with whether the limiter acted. Peaks are
  linear (1.0: full scale), up to 8 channels (`METER_CHANNELS`); the mixer
  raises them with `fetch_max` in atomics and a look takes them with
  `swap`, so it is for the one who shows them.
- **The clock:** `Mixer::timebase()` gives `Timebase { epoch, rate,
  frame, heard }`: each opening of the device is an epoch with its own
  frames. `Fader::set(Ramp { start, to, over, curve })` changes the gain
  without blocking, in place of a ramp still waiting: from `Start::Now`
  (the next frame mixed), `Start::At { epoch, time }` (a time of an epoch;
  never, once the device has opened again), or `Start::FirstAudio` (the
  first frame of an output opened on the input after the ramp is set: a
  fade in from what is loaded next), over `over` (rounded to frames at the
  device's rate), equal-power (`Curve::EqualPower`: a rise on a quarter
  sine, a fall on a quarter cosine) or linear; `Ramp::set(g)` is at once.
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
  device holds as well unless another input is heard; with nothing
  running the device pauses, holding what it has.
- **Formats:** no resampling in the mixer. The Lead owns the device's
  format: when it opens at another one, every other output is closed
  (`Error::Closed`) and the device plays out and opens again. A guest
  never displaces one: while the device is open it is offered exactly
  its format (fixed when first asked), so its player resamples; only with
  nothing open does it open the device at its own.
- **The microphone's port:** `Mixer::set_mic(Option<Input>)`. The mixer
  taps the device for the microphone while a Lead output is open, again for
  each one and each opening of the device, so its echo canceller hears the
  whole mix as its sidechain; a frame map of the Lead's chunks tells it the
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
  the canceller gives,
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
  the buffer plays (its channels' mean), both brought down to about 44.1
  or 48 kHz (the output's rate divided by 1, 2, 4 or 8), and queues them in
  blocks of 128 for the cleaner, a thread of its own. A control loop steers
  the ratio within ±0.2 % of nominal to hold the ring's fill at two capture
  periods. It measures once per device buffer, counting the time since the
  last capture packet, so it sees the fill as if the voice arrived
  continuously. The voice starts after both sides have run 100 ms, drops
  out on a dry ring (counted) until the target is queued again, and a fill
  held more than a period over its target (a capture burst, a device-thread
  stall) is skipped at once.
- The echo canceller (`aec`, on as an input opens; `Input::set_echo_cancel`),
  on the cleaner's thread. The rule above the others: the singer is never
  taken out. It has two parts:
  - a delay finder, matching how the music's level moves in 16 bands
    against the microphone's at every delay up to 1 s, moves the filter's
    window to where the echo starts when it does not reach it (a wireless
    speaker); once learnt, the window follows the echo's loudest part;
  - a partitioned-block frequency-domain adaptive filter (200 ms of echo,
    normalised per frequency), two of them: a shadow that learns while
    nobody sings, and the filter in use, which copies the shadow only once
    it leaves clearly less. Neither learns from singing (a held note would
    draw them towards taking the note out). A room that changes while the
    detector takes it for singing (the volume turned up: the leftover jumps
    as a voice's would) is learnt by a trial filter apart, which takes over
    only once it has left a quarter of what the filter in use leaves for a
    second running: a room that changed does that soon, a voice never. The
    filters learn along the music prewhitened (each frequency's steady part
    taken out, so a held tone does not leave the rest of its band
    unlearnt).
  The voice it gives is the microphone less the echo predicted, nothing
  more taken out. A residual suppressor was measured and left out: on real
  songs it took 2 to 3 dB more of the echo out where nobody sang, as much
  of a singer under the echo, and left the voice no further over the rest
  while they sang. `Input::record_echo` writes what the canceller works with
  for some seconds to a WAV file (three float channels at its rate: the
  microphone, what plays, the voice it gives), queued by the cleaner and
  written by a thread of its own, for tuning it to a room.
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
  its worst 5 %). It hears the microphone only, not what played.
  `take_voice` stays the canceller's voice.
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
port is told the Lead's frames while a guest played first, hears each
opening of the device (at two rates) and never reaches it, the monitor
sees two inputs summed, a fade out on an input with nothing to play
settles, a seek keeps a fade under way and a pause ends it, and only the
Lead opens the device again (a guest made while it is open is offered
only its format, and closed when the Lead opens at another). Strips: one
at 0 dB keeps its input bit for bit, a level change is a 20 ms linear
ramp, mute and solo silence strips without a click, a strip's or a fader's
ramp under way ends on its target when the device opens again, and the
meters read each strip and the sum, then nothing once taken. Where frames
must line up exactly, the tests run on a device that takes what it is
given at once, with outputs started short of a period, so the mixer waits
for all of them.
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
through (within 0.1 dB), and records three channels at its rate (what
played as the music, the microphone at its echo's level), in a WAV header
other readers take (WAVE_FORMAT_EXTENSIBLE, float, front left, right and
centre, with a `fact` chunk). `aec`'s own
tests: a room's echo (45 ms, a 40 ms tail) down 27 dB, a voice over it
kept within 0.1 dB and 35 dB over what is left; a moved speaker learnt again; the volume turned up 12 dB
learnt again (29 dB down after; 2.5 without the trial filter); a note held
for four seconds kept whole, the filter still learnt after it; no
reference, the voice as it is; and a laptop's speakers (a limiter, soft
clipping, no bass) in rooms with a 0.15 to 0.4 s reverberation, the echo
40, 120 and 300 ms late (the last past the window, found by the delay
finder): with the music alone the meter's peak falls by more than 20 dB
(24 to 33), the delay is found within 15 ms, and a voice as loud as the
echo is kept within 0.5 dB, 20 to 41 dB over the rest. A singer as a singer
sounds (syllables of harmonics, consonants, soft ones) through such a room,
from twice as loud as the echo to 24 dB under it: no syllable loses
anything (each within 1 dB, the soft ones and the onsets too). `isolate`'s
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
limiter is, a recording (`r` on the Karaoke tab) shows.

Recordings go through the canceller again with `cargo run --release -p
jodel-output --example aec_replay -- RECORDING.wav...` (one canceller over
all of them, as one session; `--fresh` for one each), second by second,
the voice it gives written beside each as `RECORDING.wav.replayed.wav` and
the singer isolated from it as `RECORDING.wav.isolated.wav` (16 kHz, as
the meter hears it, with the share of a core that took); a
file of two channels (microphone, music) is a made-up session. Five from a MacBook's own
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
     case), at two rates in turn. The terminal needs microphone permission
     (System Settings → Privacy & Security → Microphone), or the voice is
     silence and the peak stays at −∞.
- **Linux:** type-checked and unit-tested in WSL, which has no sound card.
  On a box with a DAC: `devices`, then `silence -- hw:CARD=<id>,DEV=0 <rate> 2`.
  A sound server (PipeWire, PulseAudio) holding the card makes the open fail
  with "device in use". The microphone (type-checked only): `mic` lists
  `default` and the `plughw:` inputs; `mic -- hw:CARD=<id>,DEV=0 default`.
