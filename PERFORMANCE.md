# Performance: what the player computes, and when

The aim: everything expensive about a track happens once, when it is got
ready in the queue. After that, playing it, on stems or not, in any tab,
costs about what playing audio costs. The work that cannot be done ahead
is a transition's tempo warp and the mixing of a few streams.

## Where the work was (November 2026)

Measured on an M4 Max (a 6 s chunk of stems: 530 ms of GPU) and read from
the code:

| Work | When | Cost | Done once? |
|---|---|---|---|
| Stem separation | analysis, lyrics timing, and again for the parts made beside the file (from every seek, the next track's from its start), both tracks of a transition | ~9 % of the GPU per track playing, in 0.5 s bursts the window waits behind | no: nothing kept |
| Spectrum FFT | every frame, also where no spectrum is shown (Transition, Now Playing, Karaoke) | 0.5–8 ms a frame on the UI thread (131072 points at 384 kHz) | runs hidden |
| Engine output buffer trim | every 4096 output frames | a ~24 s buffer moved each time: GB/s at 384 kHz | no |
| Stems kept for jumps | once a second | ~70 MB moved a second per stems run | no |
| Frames | every engine event (10 a second) forces an extra, unpaced frame | uneven frame times | no |
| Heard position | extrapolated from when an event was handled | the Transition lanes step back when one is late | no |

## The plan

1. **Stems kept on disk** (`jodel_playback::stem_file`): the analysis
   already separates every track; it now also writes the stems, in a
   compact file (16 bits with a scale per block and channel, lossless
   prediction and Rice coding; silence is nearly free), in the cache. The
   engine's stems runs read the file (`StemStore`) and separate only a
   track that has none. The lyrics timing reads the voice from it. A track
   is ready in the queue once its stems file is there too. The folder is
   kept under a size (oldest first); `make clean-cache` empties it.
2. **The spectrum only where it shows**, its buffers kept, at most 30 a
   second.
3. **Buffers trimmed in steps of a second or more**, not every block.
4. **Frames paced by the frame clock only**: an engine event marks the
   window as needing a frame, it does not force one at once while frames
   are paced anyway.
5. **The heard position never steps back** within a track unless playback
   moved (a seek, a jump, a new track).

Kept as they are, cheap enough or needed live: the transition warp and
its gains, the output resampler and dither, the Transition lanes' shift
(pixels moved in memory; a lane texture as a ring would save its upload,
later), the Transition tab's per-frame transition preview.

## Status

All five steps are in (each with its tests):

1. Stems kept: `jodel_playback::stem_file`, `player/src/stems_cache.rs`,
   the analysis writing them (`rhythm::listen_keeping`), the engine's runs
   reading them, the lyrics timing reading the voice. Two real songs on an
   RTX 5080 (`cargo run --release -p jodeldiplom --example stems_kept --
   FILE`): 26–28 MB a minute, read back about 155 times faster than they
   play, at most −90 dBFS from separating again live (84–92 dB below each
   stem). Separating them took 4.5–7 s of GPU each; playing them now
   takes none.
2. The spectrum is analysed only where it shows, 30 times a second at
   most, its buffers kept.
3. The output buffer and the stems kept for jumps are trimmed in steps of
   a quarter of what they keep (a second at least) instead of every few
   blocks: at 384 kHz the output buffer moved ~74 MB every 43 ms before.
4. Engine events no longer force a frame while frames are paced.
5. The heard position holds rather than stepping back by less than
   0.25 s while the track plays.

Not done yet, smaller: the Transition lanes' shift still uploads the whole lane
rect each frame (a lane texture used as a ring would upload only the new
columns), and the Transition tab's transition preview is rebuilt each frame.

## The stems as parts of the output (November 2026)

Measured on the development machine (Windows, an RTX 5080), release
builds.

**Resampling the parts.** Four stereo stems, 8 channels, from the
separator's 44.1 kHz to the output's rate with the track's own filter
(224 taps a phase, 160 dB stopband), in the engine's pieces of 4096
frames (`cargo test --release -p jodel-playback --test resample --
lane_resampling --nocapture`):

| Output | Frames made a second | Real time | One core |
|---|---|---|---|
| 48 kHz | 3.5 M | 73× | 1.4 % |
| 96 kHz | 4.0 M | 42× | 2.4 % |
| 192 kHz | 3.8 M | 20× | 5.1 % |
| 384 kHz | 3.2 M | 8.4× | 12 % |

Kept as it is: the cost follows the output's rate and the taps, and the
taps follow the narrow band from 20 kHz to 22.05 kHz, not the
attenuation (a filter of 120 dB instead would save under a third). The same
filter as main's is what keeps the resampling from adding anything: stems
that add up to the track (the tests' separator) add up to main within a
few 24-bit steps; an eighth of a core at 384 kHz is not worth losing that.
A real separator's stems add up to the song only roughly (about −30 dB off
it, measured on real songs), and the stems are played as separated.

**Reading the parts along a time map** (a head read along one, mixer
step 3): the same 8 channels, read along the map at the separator's
44.1 kHz before the resampling above, whatever the output's rate
(`cargo test --release -p jodel-playback --lib eight_channels --
--nocapture`): varispeed 1.6 M frames a second, 37× real time (2.7 % of
a core); stretched (a phase vocoder a channel) 1.0 M, 23× (4.3 %). Main
is their sum, so nothing else is read or resampled while a head follows
a map.

**Memory.** A stem run keeps its stems as 8 × f32 at 44.1 kHz,
1.41 MB a second:

- no jump window: 2 s behind what it hands on (what the output may give
  back, the parts made ahead and a second), trimmed a second at a time,
  and what it took ahead: about 4–6 MB, and up to four pieces queued from
  its thread (4 s kept, 5.6 MB; 19 s separated, 27 MB);
- the player's jump window (24 s each way): 25.5 s behind, trimmed a
  quarter at a time (so up to 32 s), and 24 s ahead: 70–79 MB a run
  (the plan's estimate of 58 MB left out the trim's step and the margin
  behind).

The parts made at the output's rate (8 × i32 a frame, what the output may
give back and half a second ahead, about 1.1–1.5 s): 1.7–2.3 MB at
48 kHz, 7–9 MB at 192 kHz, 14–18 MB at 384 kHz. The lane ring of a
mixer's input (half a second of main, the four parts and the control
word, 11 × 4 bytes a stereo frame): 1.1 MB at 48 kHz, 2.1 MB at 96 kHz,
4.2 MB at 192 kHz, 8.4 MB at 384 kHz. Main kept for the frames written
again (below) without a jump window: half a second more of the track's
output, 1.5–1.9 MB at 384 kHz (a jump window keeps it already).

**The first touch.** Stems asked for while none are made start on the
frame heard; once they are made a little past the write head, the output
gives back what it has not played and the engine writes it again with
them (`engine/lanes/catch_up.rs`). Kept stems are there 11–17 ms after
they are asked for (the test build, whose engine is optimized:
`playback/tests/lanes_catch_up.rs`); separated ones once the first 6 s
chunk is: 0.2–0.5 s after the touch on the RTX 5080 at 48, 192 and
384 kHz (a 6 s chunk takes about 0.12 s of GPU, the rest is decoding and
transforming that chunk on the processor). What is guaranteed: every
frame the output still gives back once the parts are made is written
again with them, so a change landing after that is heard on its frame
(from the cache, 0.1 s ahead is enough on this machine; a small loaded
one may need a few tens of milliseconds more), one landing before it from
the first frame given back. Before, the touch was heard only after the
ring (0.5 s) and the parts made ahead (0.5 s) more. While a jump's
crossfade is held, a route switch or a transition is under way or a jump
waits, nothing is written again, and the touch lands as before.
