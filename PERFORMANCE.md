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
| Stem separation | analysis, lyrics timing, and again on every stems start, seek, jump, the Mix tab's shadow run, the next track's prestart, both tracks of a transition | ~9 % of the GPU per track playing, in 0.5 s bursts the window waits behind | no: nothing kept |
| Spectrum FFT | every frame, also where no spectrum is shown (Mix, Now Playing, Karaoke) | 0.5–8 ms a frame on the UI thread (131072 points at 384 kHz) | runs hidden |
| Engine output buffer trim | every 4096 output frames | a ~24 s buffer moved each time: GB/s at 384 kHz | no |
| Stems kept for jumps | once a second | ~70 MB moved a second per stems run | no |
| Frames | every engine event (10 a second) forces an extra, unpaced frame | uneven frame times | no |
| Heard position | extrapolated from when an event was handled | the Mix lanes step back when one is late | no |

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
its gains, the output resampler and dither,
the Mix lanes' shift (pixels moved in memory; a lane texture as a ring
would save its upload, later), the Mix tab's per-frame transition preview.

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

Not done yet, smaller: the Mix lanes' shift still uploads the whole lane
rect each frame (a lane texture used as a ring would upload only the new
columns), and the Mix tab's transition preview is rebuilt each frame.
