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
its gains, the output resampler and dither, the Transition tab's
per-frame transition preview.

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

Not done yet, smaller: the Transition tab's transition preview is rebuilt
each frame. (The lanes no longer go through the pixmap: every lane is
painted once in tiles and scrolled from them, below.)

## The lanes as a widget (the iced port's P2, November 2026)

Every lane (the Transition tab's waveforms, overviews and bar numbers, the
Karaoke tab's voice lane) is painted in tiles of 512 columns, each once,
and shown by the window's own widget over the cells the old canvas leaves
blank (`gui::lanes`): on the GPU one pipeline for all of them, a primitive
a lane, each lane's tiles in a texture array of its own (let go of when
the lanes go), the shader the same for every pixel (two columns read,
mixed by the share of one, the playhead and the card's rounded corners
by arithmetic, no branch on what a lane shows); without a GPU each lane an
image made from its kept tiles, on whole columns, made again only when
it moved or a tile in it was painted, the playhead a quad over it. A
frame where only the lanes move (one follows the playhead) draws nothing
of the canvas. Before, only the playing track's block went into the GPU's
layer; the other lanes (the next track's, the overviews, a view held)
were painted into the pixmap and sent over with it.

Measured on the RTX 5080 machine, shared with other agents' builds
(release; `examples/frames.rs` at 3024 × 1890 at scale 2, ms a frame,
widgets + pictures + lanes + the copy of what changed):

| Screen | before, GPU | after, GPU | before, no GPU | after, no GPU |
|---|---|---|---|---|
| Transition, following | 1.31–1.46 (pictures 0.10, layer 0.02, copy 1.04–1.17, 57 % of frames changing the pixmap) | 0.33–0.44 (lanes 0.13–0.16, 1 %) | 4.0–4.5 (pictures 2.1, copy 1.8–2.2) | 2.7–3.1 (lanes 2.5–2.8) |
| Transition, held | 2.05–2.09 | 0.22–0.35 | 2.06–2.38 | 0.24–0.33 |

`examples/view_cost.rs` (1512 × 982 at scale 2, headless wgpu, median /
worst ms): Transition 0.037 / 0.060 before, 0.038 / 0.106 after (13 lane
primitives); the Karaoke tab 0.038 / 0.186 before, 0.037 / 0.312 after;
Main and the Mixer unchanged within the machine's noise. `frame_cost`
(Karaoke mode, which shows no lanes; its pixmap pass lost the layer's
branch), at 3024 × 1964, the two builds run one after the other: the
pixmap step's median 0.12–0.14 ms before, 0.10–0.15 after; the GPU's
total median 1.12–1.75 ms before, 0.96–1.73 after (the diploma's 1.43 and
1.35, its worst 2.31 and 1.73); the one frame over the budget in both, the
normal scene's first frame timed (3.3 ms in its pixmap step), as before.

## Karaoke mode on one surface (the iced port's K2, November 2026)

Karaoke mode is the GPU's alone: its backdrop is the ground (a triangle
over the window reading it, `ground`, where the pixmap pass was), the
stage draws all the rest over it (the lyrics set in the middle without
the lane, the title and notes with it, the count-in, the score's
odometer, as text from its atlas), and of the pixmap only the controls'
bar shows, a strip of rows (a scissor), while it is laid out. The
processor no longer paints the lyrics, the count-in or the score into
the pixmap, nor sends them over: in Karaoke mode the pixmap changes only
where the controls' bar does. Two pipelines in the first step (the
ground, the pixmap's strip), each the same for every pixel; outside
Karaoke mode the pixmap pass is as it was.

`frame_cost` on the RTX 5080 (two runs after, one before, the machine
shared with other agents' builds and tests), median ms of the first step
(`pixmap` before, `ground` after) at 3024 × 1964: 0.10–0.22 before,
0.15–0.28 and 0.14–0.18 after (the other steps 1.2–2.0× slower in the
first run after: the machine's); at 3840 × 2160: 0.16–0.22 before,
0.16–0.28 and 0.15–0.19 after. The new `lyrics` scene (no lane, the
lyrics set in the middle): its stage step 0.12–0.25 ms, the processor's
stage 0.03–0.05 ms a frame; the processor's pixmap step (0.004–0.013 ms
a frame before, the picture painted only where it changed) is gone.
Every run, before and after, had single frames over the budgets in a
few scenes, each in another scene and step (in the run before, the Diva
peak at 11.8 ms, every step slower at once): the machine's, as the
diploma's spike below.

## The Transition tab native (the iced port's S8, November 2026)

The tab's headers, lane labels, footer and the places that take the
pointer are iced's widgets, laid out in points on the window's pixels;
the lanes go where that layout leaves room, and the old canvas draws
nothing of the tab. Measured as P2's, on the same machine, more loaded
this time (Main's view cost went from 0.25 to 0.39 ms between the runs,
nothing of it changed):

| `examples/frames.rs` (ms a frame) | before, GPU | after, GPU | before, no GPU | after, no GPU |
|---|---|---|---|---|
| Transition, following | 0.62 (widgets 0.36, lanes 0.22) | 0.52 (widgets 0.20, lanes 0.31) | 4.96 | 4.01 |
| Transition, held | 0.40 | 0.26 | 0.47 | 0.29 |

The lanes' share now holds the plots built for them (the canvas's
widgets held them before); no frame of the tab changes the pixmap any
more (1 % before). `examples/view_cost.rs`, Transition: 0.043 / 0.091 ms
(median / worst) before, 0.167 / 0.330 after (the tab's widgets, built
every frame), within the 1.0 / 2.0 budget.


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
(a measurement: `cargo test --release -p jodel-playback --lib
eight_channels -- --ignored --nocapture`): varispeed 1.6 M frames a
second, 37× real time (2.7 % of a core); stretched (a phase vocoder a
channel) 1.0 M, 23× (4.3 %). Main is their sum, so nothing else is read
or resampled while a head follows a map. Without a separator only the
other instruments' pair is read along it (the whole mix), a quarter of
that.

**The key shift** (`Command::Key`, in the same pass): the drums' pair
read along the map as above, the other 6 channels stretched by the
pitch ratio (a phase vocoder a channel at speed 1/r) and read r times
faster through the map's reader, at 44.1 kHz (a measurement: `cargo test
--release -p jodel-playback --lib six_channels -- --ignored
--nocapture`): −6 semitones 18× real time (5.6 % of a core), +2 14×
(7.0 %), +6 12× (8.6 %: the stretchers make r times as many frames as
they take). The parts' resampling to the output's rate comes on top,
unchanged, so a shifted head at 192 kHz is at most 8.6 % + 5.1 %, some
14 % of a core (about 21 % at 384 kHz). Without a separator only the
whole mix's pair is shifted, a third of the stretch.

**A transition's two heads.** Through a mix two stem runs go at once,
one a head. The next track's (B's) starts about 4 s before the window,
from its first frame, and is read along its map as above; the playing
track's (A's) goes on as its parts are made anyway, read along its glide
only when the plan glides (else resampled as any parts, no warp). So a
mix costs at most two reads along a map and two resamplings of 8
channels: at 192 kHz about 2 × (2.7 % + 5.1 %), some 16 % of a core,
and about 32 % at 384 kHz, for the window and B's settling after it.
From kept stems neither head touches the GPU; otherwise both are
separated at once, B's first chunk before its window (a 6 s chunk is
about 0.12 s of the RTX 5080). On a device, B's stems not separated in
time give way to its whole mix (a quarter of the read), and the desk
drops B's frames that come late rather than wait; A is never held up.
The desk's own share (two lane inputs, their automation curves
evaluated per frame, the sum and the look-ahead limiter) is small beside
the resampling, and its mixer thread allocates nothing for the curves.

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
output, 1.5–1.9 MB at 384 kHz (a jump window keeps it already). It is
kept on every device's output of lanes, with stems or not and on a time
map too, so a key change can start again from what the output gives back
(`engine/jump.rs` `window_frames`): the same half second of main, at most
1.9 MB at 384 kHz stereo (4 bytes a sample), 0.2 MB at 48 kHz.

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

**Karaoke mode's stage on the GPU.** The stage draws its shapes with a
pipeline a group of them (`gui::stage_gpu::Group`), each pipeline's
shader carrying only its own group's code, as Apple's GPUs run a shader
at the pace its dearest code sets, for every pixel: split so, the stage
pass on an M-series Mac went from 4.0 / 5.7 / 10.9 ms a frame (normal,
Diva, the diploma) to 0.28 / 0.34 / 1.85 ms. The effects library
(`gui::fx`) adds a group for each effect's shapes and each lyric
treatment, each with only that effect's code (and the helpers the
effects share, `fx_heat` and the rest, once in it). The window makes the
pipelines of the groups the tiers' rows use as it opens (`Group::used`:
14 of the 15, the stage's own 6, the baked paper's among them, the
shapes of the rainbow, the lane's glow, its flames, the edges on fire,
the bulbs and the word slam, and the treatments of the rainbow and the
lyrics on fire; and the one baking the paper, below), another effect's
the first time it draws (only the lyrics frosted, which no row lists);
the light (what blooms) only the groups light can be (the plain shapes,
the lane's glow, its flames and the slammed word): 4. Made the first
time one draws, the slam's pipelines cost a frame of 112 ms (60 ms in
its text's step, 49 in the light's) the first time a word was slammed,
before the rows listed it; made as the window opens, none. The window
opens that much later: `frame_cost` made the device and the window's
pipelines in 1.42 s with the rows wired (two runs) and 1.13 s with only
the first four effects listed (one run, the same session, the same
machine, neither near the 105–115 ms below). On the RTX 5080 (Vulkan,
the driver's cache warm)
the stage's pipelines take about 50 ms (35 ms for the 6 before the
library, 70 ms for all 13), the light's 16 ms, the bursts' 4 ms, the
window's device and all its pipelines about 105–115 ms (110 ms before
the library); the first time the driver compiles the shaders, about
0.6 s for the stage's. `frame_cost` prints the time the device and the
window's pipelines take. Each frame the effects cost the processor a few
slices walked (well under a millisecond); the effects' screen passes are
a step of the window's pass of their own (`fx`), nothing where none
shows: drawn at 3024 × 1964 the edges on fire add 0.04–0.08 ms, the
marquee bulbs 0.05 ms, the lyric treatments and the coins next to
nothing (`examples/fx_frames.rs`). Moving with the beat costs them next
to nothing more: the music is a uniform the shaders already read, and
the fire's strips grow with its flare (the processor works the swell out
from the same music). The word slam is one quad over the lane (9 taps of
the atlas a pixel for its letters and outline, 1 inside a letter), in
the `top` step, and a copy of it in the light, which blooms into its
glow; its dust is a burst; it draws nothing while it shows under 2 %.
The screen's shake is two numbers in the window's uniform the stage and
the bursts share (the backdrop and the controls' bar under the stage never
shake, nor the screen passes that keep still). Every registered effect at full
strength at once over the Diva peak (`frame_cost`'s `all` scene) costs
the RTX 5080 1.8 ms a frame (worst 2.3 ms; the Diva peak alone 1.5 ms,
worst 2.8 ms; the `fx` step 0.21 ms, worst 0.43 ms). With the rows
listing the effects (each tier's scene reaching its rung in its timed
second, so what the rung fires is in it), the medians and worsts at
3024 × 1964 (one run, the machine otherwise idle) were: Good 0.95 /
1.28 ms, Great 1.40 / 2.44, Awesome 1.51 / 2.59, Amazing 1.66 / 2.64,
Diva 1.59 / 3.01, `all` 1.50 / 2.01; no step's worst over 1.3 ms (the
`fx` step's median 0.13 ms at Awesome, 0.18 at Amazing, 0.19 at Diva).
The bursts are one
pipeline for every look (`gui::particles::LOOKS`): a burst's quads are
small and few, their step 0.07–0.13 ms a frame at the Diva peak and with
every effect, and a quad's look is the same over its pixels (one branch
taken by the whole quad).

**The backdrop's depth of field and beat.** Round a subject cut out,
the card reads the cover's levels (made once, as the cover is sent over)
blurred by its depth: two smooth reads of two levels (8 taps) where it
read the cover and the plate sharp before (9), the plate's only where
the cover's blur would carry the subject's colours. The beat's push,
the rim and the spotlight rays are a few numbers in the cover's uniform
and one tap of the subject's rim. `frame_cost` on the RTX 5080 (two runs
of each, the unchanged backdrop and this one in turn, the machine shared
with other agents' builds and tests): the backdrop step's medians over
the scenes 0.18–0.25 ms before, 0.19–0.44 after at 3024 × 1964, and
0.22–0.79 before, 0.25–0.49 after at 3840 × 2160: within the machine's
noise; the worst frames (to 9–20 ms in either build) fell where every
step spiked at once.

**One fire.** The edges on fire and the lane's flames (which set the
marquee's bulbs on fire where both show) draw the same fire, a helper
the effects share (`fx_fire`, `fx_fire_light`, with `fx_heat` and the
rest, once in each effect's pipeline): its noise taken at most four
times a pixel, three octaves and a sway, from up to four edges at the
screen's corners (the edges on fire), once along the lane. On the
RTX 5080 (`fx_frames`, 3024 × 1964, the machine shared with another
agent's GPU work, so the medians moved between runs) the edges on fire
add 0.05–0.09 ms to the `fx` step, 0.11–0.17 ms surging on the beat
(their strips grow with the surge), the marquee on fire (the lane's glow
and flames, the bulbs and the edges on fire at once) 0.14–0.23 ms; with
the rows listing them (`frame_cost`, two runs) the `fx` step's median
was 0.18–0.23 ms at Awesome, 0.20–0.28 at Amazing, 0.22–0.34 at Diva
and 0.29–0.30 with every effect, at 3024 × 1964 and 3840 × 2160, its
worst under 1 ms; at 3024 × 1964 no frame went over 4 ms (the build
before, in the same session, had four scenes' worst over it). The
frames over 4 ms at 3840 × 2160 (one to three a scene, Great's among
them, which draws no fire) are single frames in one run and not the
other, in steps that draw no fire as often as in those that do (Great's
bloom 5.6 ms in one run, 0.8 in the other; the light 6.7 and 0.9 with
every effect): the machine's, as the diploma's spike above.

**The diploma's paper, baked.** `frame_cost` on the M4 Max (Metal,
3024 × 1964, two runs) had every scene within its budget but the
diploma: 1016 frames, the GPU's total median 4.0 ms, mean 4.0, worst
11.4–13.4; the stage step median 1.9, worst 5.8–7.8; the bursts' worst
2.1–3.3 and the composite's and the `fx` step's 1.0–1.4 (all at their
worst in the same frame). Kind by kind the sheet's paper was the dearest:
2.62 ms alone (one a frame, over 0.53 screens), where every other kind
took 0.52–0.8 ms alone, the floor of a pass timed apart. Yet the paper
does not change once it has unrolled: its shader has no clock, and what
moves on it (the stamps, the signature, the stars, the words) are shapes
of their own. So the stage pass bakes it (`gui::stage_gpu::bake`): once a
frame's paper has been the same three frames in a row (a window being
resized is not baked frame by frame), it is drawn into a texture of its
own (in four strips, below; its quad's pixels, half floats, premultiplied, linear where the
frame takes linear light), and from then on read from there, a texel a
pixel, by a shape of its own (`kind::BAKED`, its own pipeline,
`Group::Baked`) while it stays that paper; while it unrolls (0.55 s) it
is drawn as before. The sheet shaken as the seal is stamped down
(0.45 s) shakes by whole pixels (`seal::shake`), everything on it with
it, so the bake is read moved by whole pixels, each pixel as it was
drawn; a paper moved by part of a pixel is another paper. Baked again
where the sheet's size, its inks or anything else of it changes; a
paper larger than the window, never. The texture is the window's size
(47 MB at 3024 × 1964), made as a stage first shows (entering Karaoke
mode) and cleared at once, kept while a stage shows, let go when none
does, made again once a new window size has held three frames: making
it costs about 3 ms on the RTX 5080 (0.6 ms of it on the processor, the
rest the system making the memory the GPU's), which on the diploma's
first frame or its bake frame went over the 4 ms budget, so the reveal
pays only for drawing the paper into it. The pipelines are made as the
window opens. What to do with the texture and a frame's paper (make,
keep, let go; draw, bake, read) is decided apart from the GPU
(`bake::Planner`), tested over frame sequences. It looks the same:
against the same build drawing the paper each frame, every frame of
`result_frames` (six diplomas, 23 moments each at 1600 × 900, the shakes
among them) and `frame_cost`'s pictures at 3024 × 1964 then 1600 × 1000
(a resize) are at most 1 in 255 off, on at most 5 % of the pixels (the
half floats' rounding: a texture of full floats leaves some 70 pixels
so, at twice the memory). On the RTX 5080 (two runs each of the same
build drawing the paper each frame and baking it, the machine shared
with other GPU work, more in some runs than in others) the paper alone
went from 0.35–0.41 ms to 0.28–0.45 (the floor), the stage step from a
median of 0.26–0.27 ms to 0.20–0.24 (p99 0.30–0.36 to 0.26–0.36), and the
diploma's frames (each the lesser of its two runs) were 1.5–1.6 ms at
the median and 1.7–2.3 at the 99th percentile without it, 1.4–1.7 and
1.7–2.8 with it; the worst 1.9–4.1 ms without, 1.9–3.1 with, none at a
moment of the reveal of its own.

**The bake in strips.** Baked so, the M4 Max (`frame_cost --budget 4
--step-budget 2.5` at 3024 × 1964) drew the diploma's paper alone in
0.55 ms, its stage step at a median of 0.58 ms, its frames at a median of
2.54 ms and 3.43 at the 99th percentile, and no singing scene had a frame
over 4 ms. The one frame over the budget was the bake's own, in both runs
of the two: frame 35, 0.583 s into the reveal, its `uploads` step (where
the bake is drawn, in a submission of its own) 2.251 ms, 4.452 ms in all:
the whole sheet drawn into the texture at once. So it is baked in four
strips across it, top to bottom, one a frame (a scissor rect a strip, the
first clearing the texture): while it is baked, the strips in are read
from the bake and the rows below them drawn as they are (the paper's
shape, its quad starting under them), so each of those four frames draws
the paper about once, a quarter of it into the texture; it is read whole
once the last strip is in. Changed, moved (by whole pixels too) or the
window resized before then, it starts over, once it has held still again
(`bake::Planner`, its strips tested over frame sequences). On the RTX 5080
(the diploma's frames each the lesser of two runs; three runs baking at
once, two in strips, the machine shared) the bake frame's `uploads` step
was 0.21–0.26 ms against 0.07–0.08 on the frames round it; in strips it
is 0.15–0.16 ms on each of frames 35–38: a strip about 0.08 ms, a quarter
of the paper's drawing and about 0.05 ms any pass in a submission of its
own costs here, so on this GPU a strip costs half the bake, not a quarter.
The stage step on those frames goes 0.23, 0.21, 0.19, 0.18 ms as less of
the paper is drawn as it is (0.23–0.24 with all of it, 0.18 read), their
totals 1.43–1.48 ms against 1.55 for the bake frame before. It looks the
same: against the build baking at once, `result_frames` (six diplomas, 34
moments each at 1600 × 900: every frame from 0.55 to 0.70 s, the seal
stamped and the sheet shaken in each, and more) are the same but in the
three frames where part of the paper is still drawn as it is (0.58, 0.60,
0.62 s), those at most 1 in 255 off on 1–4 % of the pixels (the bake's
half floats rounded where the drawn rows are not); `frame_cost`'s pictures
at 3024 × 1964 then 1600 × 1000 (a resize) are the same.

**The diploma's spike.** On the RTX 5080 the worst frame is another in
every run (frames 26, 53, 119, 190, 482, 509, 546, 761, 861, 921 and
924 of 1016 in eleven runs), most in the still end of the reveal where
nothing lands, every step of it slower at once (a frame or a few in a
row); nothing is made there (no pipeline, texture or buffer; the bake's
four strips, 0.58–0.63 s in). Each step's time is the GPU's between two timestamps
with a wait and a submission between them, so whatever holds up the GPU
or the process for a frame (another program drawing, the system) shows
in every step of that frame: with another process's tests on the GPU at
the same time, 229 frames went over 4 ms. The spike is the machine's,
not the diploma's. So `frame_cost` times the diploma's reveal twice and
takes each step of each frame at the lesser of the two runs'
(`Judge::MinOfTwo`): a hitch at the same moment both times (the
diploma's) stays, the machine's own, in one run, goes; that series is
judged by its worst frame, as the other scenes are. It says which frame
was the worst and what was on screen then (its time, the beats that had
just landed, how much it drew).

**What keeps the stage fast (the rules for its GPU code).** Held by
tests that need no GPU (`gui::stage_gpu::groups`, `gui::particles`):

- One pipeline a group: the stage's shader picks what it draws by its
  group's constant alone (`GROUP ==`), never by what a shape carries
  (`each_pipeline_carries_only_its_own_effect_and_picks_by_its_group`).
- An effect's code only in its own pipeline (its shapes', its lyric
  treatment's), the shared helpers only where an effect's code is, and an
  effect's look only read (`looks[FX_<IDENT>]`), never chosen among.
- A screen pass draws only strips (round the edges, round the lane), none
  over another, never the whole screen
  (`the_screen_passes_draw_only_strips`).
- The bursts' fragment shader picks a drawing by the quad's look alone,
  one branch a line of `LOOKS`
  (`the_particles_pick_a_drawing_by_its_look_alone`); a look that costs
  much more than the others gets a pipeline of its own.
- Before a new effect lands, `frame_cost --budget` on the Mac: every
  scene's worst frame under 4 ms of GPU in all at 3024 × 1964 at density 2
  (the plan's target, the Diva peak at × 32 included), no step's worst
  over 2.5 ms; the diploma's worst frame of the two runs' lesser times
  (above). It exits with 1 and names what went over, and prints each
  scene's worst frame, what was on screen then, and how many frames went
  over:

  ```text
  L=$(ls "$HOME/Library/Application Support/Jodeldiplom Data/covers/"*.layers | head -1)
  cargo +1.100.0 run --release -p jodeldiplom --example frame_cost -- "$L" --budget 4 --step-budget 2.5
  ```

  (any cover's `.layers` file from the player's data folder; its scenes
  are the normal one, the lyrics set in the middle (no lane), a tier's
  each at its rung, every effect at once and
  the diploma: the RTX 5080 takes 1.3–3.4 ms at worst over the singing
  ones, the diploma 1.6–3.1 ms, each frame the lesser of its two runs;
  before the ground (the iced port's K2) the normal scene's first frame timed went
  over, 4.0–4.7 ms, its pixmap step 3.1–3.3 ms, the whole pixmap sent
  over; the ground sends nothing).
