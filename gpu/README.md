# jodel-gpu

The player's networks on the GPU, through `wgpu` (Vulkan, DirectX 12 or
Metal; no WebGPU or GLES), each giving what its CPU version gives to within
rounding: the stem separator and the beat network of
[`jodel-rhythm`](../rhythm) (`GpuSeparator`, `GpuNetwork`), the speech
model and the speech recogniser of [`jodel-lyrics`](../lyrics)
(`GpuSpeech`, `GpuAsr`), and the cover's depth and cut-out (`GpuDepth`,
`GpuMatte`). Each is a program of kernels on one bind group layout, its
dispatches planned on the processor and run in short submissions
(`src/engine/`: the program, its buffers, parameters and bind groups;
the separator, the beat network and the speech model keep a plan per input
size there).

## The stem separator

```rust
use jodel_gpu::GpuSeparator;
use jodel_rhythm::stems;

if let Some(gpu) = GpuSeparator::shared() {
    let parts = stems::parts(gpu, &left, &right)?; // stereo at 44.1 kHz
}
```

`GpuSeparator::shared()` sets the separator up on the best adapter once
(a high-performance one preferred, a software renderer refused) and is
`None` without one, or when setting it up fails. `report()` says why, as
lines for a diagnostic (the player's `--diag`): every adapter `wgpu` sees
on every backend with the limits the separator needs, the one it would
use, whether it sets up there (or the error), how long a 6 s chunk
takes, and how that splits (its profile: five whole chunks
in a row, the upload and the readback, and each kernel on its own with
the matrix multiplies by layer, the best of three). It runs the same network on the same weights as the
built-in CPU separator: the weights go up once (72 MB, the layers
transposed to input × output), and a chunk runs entirely on the GPU, as
one list of dispatches of six kernels (`src/kernels.wgsl`):

- `matmul`: tiles of 64 × 64 through 16 inputs at a time, each
  invocation keeping 4 × 4 outputs in four vectors, with a bias and
  adding to its output where asked; the tiles hold vectors, so a step
  reads one of each for 16 multiply-adds;
- `rmsnorm`, `rope` (rotary positions over split halves, from a table),
  `geglu` (the gated GELU) and `transpose` (between the frame-major and
  band-major grids);
- `attention`: 8 query heads on 4 key/value heads of 32, an online
  softmax over tiles of 64 keys, so the scores are never stored.

The loops over a head's vectors and a tile's steps are written out, and
no array is indexed in a loop: the loops wgpu writes for Metal are not
always unrolled by Apple's compiler, and such an array may then be kept
in memory rather than registers (attention took 14 times as long on an
M4 Max as on an RTX 5080 that way, where three times is the difference
in compute).

The chunk's STFT goes up and its masks come back; the transforms stay
on the CPU. On an RTX 5080 a 6-second chunk takes 0.12 s, a four-minute
track about 5 s of GPU time.

## Sharing the GPU with the window

A run as one submission keeps the GPU until it is done, and a window
drawing on the same GPU waits for it: a stutter every chunk. So every
network here (the separator, the beat network, the speech models) goes in
submissions of about 5 ms of GPU time each, their size learned as they
run (`pace`), and other work on the GPU gets in between. The next
submission is made while the last one runs (that one waited for then),
so the GPU never waits for the processor between them.

Background work gives way while the window draws at the display's
refresh (something on it moving smoothly): `GpuSeparator::background()` (a
separator of its own, so the analyses never wait for the stems of the
track playing nor hold them up; the shared one where a second does not
fit), `GpuNetwork`, `GpuSpeech` and `GpuAsr`. A window calls
`pace::frame(late)` for each such frame (not for its calm frames, a few
dozen a second at most, which leave the GPU time enough); while it does,
background work waits for each submission and rests after it (before its
next submission, its processor work done meanwhile), keeping to a share of the time that starts at 50 %, is cut by 40 % on
each frame that came late while it ran (to 10 % at least) and climbs 5
points a second while frames are on time (to 75 % at most); otherwise it
runs flat out. `GpuSeparator::shared()` never rests, and while it has
work on the GPU background work waits before each submission (a second
at most at a time), so one network is on the GPU at once.
`pace::report()` says how it went (frames late, rests, time waited);
`GpuSeparator::tally()`, `GpuNetwork::tally()` and `GpuAsr::hearing()`
what each network's GPU work cost (submissions, waits for the GPU, rests,
time held back), for the trace to tell each job's.
`pace::background()` marks a thread as one doing background work, and
`GpuSeparator::for_this_thread()` picks the background separator there
(the shared one elsewhere).

## The beat network

`GpuNetwork` runs the Beat This! network of `jodel_rhythm::net` with the
same built-in weights; it is a `jodel_rhythm::net::NetRunner`, so
`Analyzer::run_network_on(GpuNetwork::shared()?)` has the analysis run
each 30 s piece's layers here, and only the two logits per frame come
back (a piece it fails on runs on the processor). A piece is one list of
dispatches: the spectrogram in, the stem's convolution as patches
through `matmul`, the three front-end blocks (attention and feed-forward
across the bands of each frame, then across the frames of each band, a
convolution halving the bands), the projection, six transformer layers,
the final RMSNorm and the head. It uses `matmul`, `rmsnorm` and `rope`
from `src/kernels.wgsl` in the network's modes (RMSNorm as `x·√d/|x|`,
neighbouring values rotated together) and its own kernels in
`src/net.wgsl`: the spectrogram in, the patches, a per-channel scale and
shift with GELU, GELU, a transpose and the flattening, and attention with
a sigmoid gate per head (heads of 32, written out as above). On an RTX
5080 a 30 s piece takes 11 ms, 40 s of a generated track 0.02 s against
0.8 s on twelve processor threads; the logits agree with the processor's
to 3e-5 and with the authors' PyTorch output to 5e-4.

## The speech recogniser

`GpuAsr` runs `jodel_lyrics::asr`'s Qwen3-ASR-0.6B with its large weights at
4 bits (`jodel_lyrics::asr::quant`: groups of 32 along each output's inputs,
a half-precision scale and offset a group), about 0.5 GB on the GPU
against 1.9 GB as the checkpoint holds them. The log-mels come from the
processor.
- The encoder runs over the whole voice in one list of dispatches: the
  three convolutions as patches through a 4-bit `qa_matmul` (tiles of
  128 × 64; four pieces of a second at a time), the steps placed with their
  positions, and 18 layers with attention inside windows of 8 s.
- The decoder runs the prompt in one pass, filling a cache of keys and
  values.
- Then it writes a token a step, in five dispatches a layer
  (`fused.wgsl`): the row normed through its queries, key and value
  (`qa_rows`, a few rows through a 4-bit matrix); those normed, turned,
  into the cache and attending in chunks of 128 positions
  (`qa_dec_attend`); the chunks put together through the output matrix and
  added to the row; the row normed through gate and up to their product;
  and that through the down matrix, added to the row. Then the last norm,
  the head (each workgroup's largest logit with it), and the next token
  with its embedding (`qa_argmax_embed`): 143 dispatches a step, where a
  kernel for each step of the arithmetic took 340. Each kernel sums in the
  order those did, so the tokens are theirs. Steps go eight at a time
  between reads (four for pieces heard together).
- Pieces of the same length are heard together (`GpuAsr::tokens_of`): a
  voice's 3 s glimpses 16 at a time, the encoder and prompt over all of
  them, then the decoder a row each until every one has ended. A row's
  sums do not depend on the rows with it, so each piece gets the tokens it
  gets alone; one that has not ended in the room the pieces together have
  (49 tokens) is heard again on its own. The parts of a transcript differ
  in length and are heard one at a time.
- The kernels take 24 KB of workgroup memory (`qa_matmul`'s tiles): a GPU
  with only 16 KB has no recogniser.

The voice is heard in parts of at most 30 s, each cut at the quietest moment from 20 s on (jodel_lyrics::asr::parts_of), each on its own. In one pass over minutes the decoder loops (oh oh oh… up to its limit) or stops early. A part's tokens also stop when the last 32 repeat a cycle of four or fewer (loop_start). Each part keeps its start and end.

Each group's scale and offset are fitted by least squares over a spread of scales around its range, as llama.cpp's k-quants do. That halves the words changed against full precision on test clips (52 % to 24 %, most of it the loops of non-English clips). JODEL_ASR_8BIT=enc,conv,proj,embed,head,dec (any of them) holds those matrices at 8 bits instead (9 u32s a group). All at 8 bits changes 2.5 % of the words, as the CPU's 8-bit rounding does.

On an RTX 5080:
- **Speed:** a 30 s part takes 0.12 s for the encoder and prompt, then
  3.5 ms a token (7.5 ms with a kernel for each step of the arithmetic). A
  glimpse takes 20 ms, heard 16 at a time (150 ms one at a time; 25 ms 8
  at a time): 1.9 to 4.8 s of glimpses for a song of 2.5 to 9 minutes,
  where they took 15 to 36 s. With the separator running beside it, 39 ms
  a glimpse (270 ms one at a time). The keys and values of 16 glimpses
  take 390 MB. The CPU's version takes about 13 s for 20 s of voice.
- **Start:** from the packed file it starts in 1.5 s
  (`jodel_lyrics::asr::packed`, written by `asr_pack` in 17 s; 491 MB, the tied
  head kept once). From the checkpoint it takes 13 s.
- **Against CrispASR:** on the 100-song bake-off, the share of LRCLIB's words
  matched (the fit check) has a median of 75.7 %, against CrispASR's 0.6B Q4_K
  at 71.9 % and its 1.7B at 73.6 %. The word error rate is 30.2 %, against 34.2 %
  and 32.4 %. It agrees with the 0.6B on fit or no fit for 97 of 100 songs.
  With the range's rounding instead of the fitted one: 72.1 % and 33.6 %.
- **Accuracy:** the encoder's output matches the CPU's, with the weights
  rounded the same way, to 1e-5. A 20 s clip writes the official model's
  tokens exactly.
- **Timing output:** `GpuAsr::hearing()` says what it did so far (the
  difference of two, `Hearing::since`, what it did between): its runs and
  pieces, the encoder's and prompts' time, the tokens and steps and their
  time, and the GPU's submissions and waits; `gpu_asr_songs` prints it for
  each file on stderr (`--glimpses`: the glimpses' too).

## The cover's depth and cut-out

`GpuDepth` runs Depth Anything 3 Small (Apache-2.0, `depth-anything/DA3-SMALL`)
and `GpuMatte` BEN2 Base (MIT, `PramaLLC/BEN2`) on their checkpoints as
published (read through `jodel-tensors`), in full precision, for the player
to take covers apart (`jodeldiplom::cover`). Both run on one small set of
kernels (`src/vision/*.wgsl` after the shared `src/common.wgsl`): a
convolution as a matrix multiply that reads its patches where they lie (a
linear layer is a 1 × 1 one; a transposed one with its stride as its size
writes its pixels spread out; a band of rows at a time where asked),
attention as two batched matrix multiplies around a softmax (Swin's
relative positions and shifted windows' mask added there), LayerNorm, an
instance norm merged from chunks' means and spreads, resizing as PyTorch's
interpolate, adaptive pooling, strided copies, Swin's windows, 2-D rotary
positions, tables and activations.

The two share one device (`vision::net::Gpu`, set up once when first
needed). Each network keeps its weights on it, with a tail for its size's
constants, and makes its plan's buffers when it runs: another input size
rewrites the tail and makes the plan again, nothing more; `release` lets the
plan's buffers go (the player does after each cover, so one network's are
held at a time).

- **Depth Anything 3 Small** at 504 × 504 (the longest side 504, as its
  processor makes it, by area where it shrinks and bicubic where it grows):
  the ViT-S/14 with its later layers' per-head norms, 2-D rotary positions
  and camera token, then the DPT head's depth branch.
- **BEN2 Base** at 1024 × 1024: the quarters and the half-size whole through
  Swin-B together, then its decoder. Its shallow features are only used
  halved by bilinear interpolation, so a 3 × 3 convolution and that halving
  run as one 4 × 4 convolution; its last stage (1024 × 1024 × 128) is made
  128 rows at a time, twice (first for its instance norm's statistics), so
  neither of those 512 MB maps is ever held.

On an RTX 5080: a cover's depth takes 21 ms (PyTorch's own: 36 ms) and its
matte 270 ms (PyTorch in full precision: 149 ms, in its default half
precision 113 ms). On the GPU the depth holds 129 MB of weights and 247 MB
while it runs, the matte 359 MB and 808 MB. On 42 covers, against the
authors' code in full precision on the same inputs, the depth correlates to
0.999998 or better and is at most 0.27 % of its range off; the matte at
most 0.011 off (the official half-precision run is up to 0.99 off its own
full-precision one). `cargo test -p jodel-gpu --test vision` holds both to
the reference on a generated scene (`tests/fixtures`);
`examples/vision_check.rs` runs the comparison on reference dumps.

## Stem playback

The player also uses it for stem playback (`jodel_playback::stems`): eight
seconds of a generated track separate and play through the engine in
about 0.4 s on the RTX 5080, so the stems stay well ahead of what is
heard.

`cargo test -p jodel-gpu` compares the GPU's masks with the CPU's on a
generated clip (they agree to within 5e-6), times a full chunk, plays
kicks over a chord through the engine with the drums only and the other
stem only (the kicks stay with the drums, the chord with the other), and
compares the beat network's logits with the authors' reference and the
processor's (within 0.02, the same beat and downbeat peaks); it says so
and passes where there is no usable GPU, and fails where there is one
the network does not set up on.
