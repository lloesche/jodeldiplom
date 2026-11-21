# jodel-gpu

The stem separator and the beat network of [`jodel-rhythm`](../rhythm) on
the GPU, through `wgpu` (Vulkan, DirectX 12 or Metal; no WebGPU or GLES).

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
takes, and how that splits (`GpuSeparator::profile`: five whole chunks
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
submissions of about 5 ms of GPU time each, each waited for, their size
learned as they run (`pace`), and other work on the GPU gets in between.

Background work rests between its submissions while the window moves:
`GpuSeparator::background()` (a separator of its own, so the analyses
never wait for the stems of the track playing nor hold them up; the
shared one where a second does not fit), `GpuNetwork`, `GpuSpeech` and
`GpuAsr`. A
window calls `pace::frame(late)` for each frame of something moving; while
it does, background work keeps to a share of the time that starts at
50 %, is cut by 40 % on each frame that came late while it ran (to 10 %
at least) and climbs 5 points a second while frames are on time (to 75 %
at most); while nothing moves it runs flat out. `GpuSeparator::shared()`
never rests, and while it has work on the GPU background work waits
before each submission (a second at most at a time), so one network is on
the GPU at once. `pace::report()` says how it went (frames late, rests,
time waited). `pace::background()` marks a thread as one doing background
work, for its caller to pick the background separator there.

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
  three convolutions as patches through a 4-bit `qa_matmul` (four pieces
  of a second at a time), the steps placed with their positions, and 18
  layers with attention inside windows of 8 s.
- The decoder runs the prompt in one pass, filling a cache of keys and
  values.
- Then it writes a token a step: a 4-bit `qa_matvec` for each matrix, the
  attention over the cache split into chunks of 128 positions and put
  together, and the next token chosen on the GPU (`qa_argmax`). Steps go
  eight at a time between reads.

The voice is heard in parts of at most 30 s, each cut at the quietest moment from 20 s on (jodel_lyrics::asr::parts_of), each on its own. In one pass over minutes the decoder loops (oh oh oh… up to its limit) or stops early. A part's tokens also stop when the last 32 repeat a cycle of four or fewer (loop_start). Each part keeps its start and end.

Each group's scale and offset are fitted by least squares over a spread of scales around its range, as llama.cpp's k-quants do. That halves the words changed against full precision on test clips (52 % to 24 %, most of it the loops of non-English clips). JODEL_ASR_8BIT=enc,conv,proj,embed,head,dec (any of them) holds those matrices at 8 bits instead (9 u32s a group). All at 8 bits changes 2.5 % of the words, as the CPU's 8-bit rounding does.

On an RTX 5080:
- **Speed:** a four-minute voice takes about 4 s (a 30 s part: 0.15 s for the
  encoder and prompt, then about 7 ms a token). The CPU's version takes about
  13 s for 20 s of voice.
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
- **Timing output:** `JODEL_ASR_TIMES=1` prints the time of each part.

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
