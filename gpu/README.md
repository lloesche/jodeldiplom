# jodel-gpu

The player's networks on the GPU, through `wgpu` (Vulkan, DirectX 12 or
Metal; no WebGPU or GLES), each giving what its CPU version gives to within
rounding: the stem separator and the beat network of
[`jodel-rhythm`](../rhythm) (`GpuSeparator`, `GpuNetwork`), the speech
model and the speech recogniser of [`jodel-lyrics`](../lyrics)
(`GpuSpeech`, `GpuAsr`), the cover's depth and cut-out (`GpuDepth`,
`GpuMatte`), karaoke story mode's language model (`Gemma`, Gemma 4
E4B's text decoder) and the story's pictures (FLUX.2 [klein] 4B:
`image::TextEncoder`, `image::Painter`). Each is a program of kernels on one bind group layout, its
dispatches planned on the processor and run in short submissions
(`src/engine/`: the program, its buffers, parameters and bind groups;
every matrix multiply as a `Mul`, what it reads and writes and what becomes
of the products; the separator, the beat network and the speech model keep
a plan per input size there).

A device can be lost: a driver reset, a GPU unplugged or switched. The
networks the process keeps (`GpuSeparator::shared()` and `background()`,
`GpuNetwork::shared()`, the vision networks' device) are set up again on
their next use, and a chunk or piece under way when it happened runs again
on the new one; a setup that failed is tried again 30 s later. A network
the caller holds (`GpuSpeech`, `GpuAsr`, `GpuDepth`, `GpuMatte`, `Gemma`,
the picture model's parts) says so (`is_lost()`), and the player makes it
again.

## The stem separator

```rust
use jodel_gpu::GpuSeparator;
use jodel_rhythm::stems;

if let Some(gpu) = GpuSeparator::shared() {
    let parts = stems::parts(gpu, &left, &right)?; // stereo at 44.1 kHz
}
```

`GpuSeparator::shared()` sets the separator up on the best adapter on
first use (a high-performance one preferred, a software renderer refused)
and is `None` without one, or while setting it up fails. `report()` says why, as
lines for a diagnostic (the player's `--diag`): every adapter `wgpu` sees
on every backend with the limits the separator needs, the one it would
use, whether it sets up there (or the error), how long a 6 s chunk
takes, and how that splits (its profile: five whole chunks
in a row, the upload and the readback, and each kernel on its own with
the matrix multiplies by layer, the best of three). It runs the same network on the same weights as the
built-in CPU separator, packed from it as it holds them
(`CpuSeparator::visit`: the layers input × output) with its sizes
(`jodel_rhythm::stems::{DIM, HIDDEN, HEADS, …}`): the weights go up once
(72 MB), and a chunk runs entirely on the GPU, as
one list of dispatches of five kernels (`src/kernels.wgsl`):

- `matmul`: tiles of 128 rows × 64 columns through 16 inputs at a time,
  the next 16 loaded while these are multiplied, each invocation keeping
  8 × 4 outputs in eight vectors, with a bias and adding to its output
  where asked; the tiles hold vectors, so a step reads three for 32
  multiply-adds. Each output's products are added in the order of its
  inputs, as the 64 × 64 tiles before it did, so it gives the same
  numbers. The feed-forward's gate and up are one matrix, their columns
  in pairs, and the multiply writes each pair's gated GELU (the beat
  network and the speech model use it too);
- `rmsnorm` (a row 32 invocations, neighbours reading neighbouring
  values), `rope` (rotary positions over split halves, from a table) and
  `transpose` (between the frame-major and band-major grids);
- `attention`: 8 query heads on 4 key/value heads of 32, an online
  softmax over tiles of 64 keys, so the scores are never stored; four
  keys' scores at a time, then taken in in their order.

The loops over a head's vectors and a tile's steps are written out, and
no array is indexed in a loop: the loops wgpu writes for Metal are not
always unrolled by Apple's compiler, and such an array may then be kept
in memory rather than registers (attention took 14 times as long on an
M4 Max as on an RTX 5080 that way, where three times is the difference
in compute).

The chunk's STFT goes up and its masks come back; the transforms stay
on the CPU. On an RTX 5080 a four-minute track takes about 3.7 s of GPU
time (4.4 s from its samples to its stems; 6 s before the tiles of 128
× 64, the gated multiply and the norms of 32 invocations a row,
November 2026).

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
what each network's GPU work cost (submissions and the processor's time
recording them, waits for the GPU and their time, rests, time held back),
for the trace to tell each job's.

`jodel_gpu::profile` times the GPU's work for profiles (timestamp queries,
where the adapter has them): each kernel's dispatches, or each submission
as a whole; `examples/gpu_profile.rs` runs every network on a song so and
prints the wall time, the tally and each kernel's GPU time. A dispatch is
timed from where its pass starts to where the next one starts (some
drivers stamp a pass's end as soon as its work has started), and the
typical time leaves out dispatches that waited for other work on the
GPU (the median of a kernel's dispatches of the same size).
`pace::background()` marks a thread as one doing background work, and
`GpuSeparator::for_this_thread()` picks the background separator there
(the shared one elsewhere).

## The beat network

`GpuNetwork` runs the Beat This! network of `jodel_rhythm::net` with the
same built-in weights, packed from the CPU network as it holds them
(`Network::visit`); it is a `jodel_rhythm::net::NetRunner`, so
`Analyzer::run_network_on(GpuNetwork::shared()?)` has the analysis run
each 30 s piece's layers here, and only the two logits per frame come
back (a piece it fails on runs on the processor, and the analysis says
how many did and why: `Evidence::network_fell_back`). A piece is one list of
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
  values; its attention takes a query in two invocations, each with half
  of the head in registers.
- A matrix of many inputs (the convolutions' output, the feed-forwards'
  down, the decoder's output matrix) is cut in parts of at most 1280
  inputs that workgroups take side by side, their sums added up in their
  order after (`qa_split_sum`): a matrix of few rows has few tiles, and a
  tile through all of its inputs kept a workgroup long. The parts follow
  from the inputs alone, so a row's sums do not depend on the rows with
  it.
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
  order those did, so the tokens are theirs. Steps go four at a time
  between reads. The prompt's first tokens are read while the first steps
  run, and a part of a transcript (a piece alone, its tokens many) has each
  read made while the next four steps run: the GPU does not wait for the
  processor to read the tokens and send more, and the steps run one read
  past the end. Pieces heard together (glimpses, a few tokens each) wait
  for each read: four more steps for all of them would cost more.
- The cache keeps each key's values the positions' side by side, so that
  a step's attention reads neighbouring positions in neighbouring
  invocations; the values a position's side by side.
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
- **Speed:** a part of about 24 s takes 82 ms for the encoder and prompt
  (111 ms before the long matrices were cut in parts and the prompt's
  attention held its head in registers), then 1.75 ms a step (2.3 ms
  before the cache's keys were laid out by position, the reads made while
  the next steps run and two rounds of columns taken together; 7.5 ms
  with a kernel for each step of the arithmetic): a four-minute song's
  transcript in 1.8 s (2.4 s before). A
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

The two share one device (`vision::net::Gpu`, set up when first needed,
and again once it is lost). Each network keeps its weights on it, with a tail for its size's
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

## The language model

`Gemma` runs the text decoder of Gemma 4 E4B-it (Apache-2.0,
`google/gemma-4-E4B-it`) for karaoke's story mode: a song's storyboard and
look written from its lyrics, greedily, so the same song always gets the
same plan (`src/llm/`).

- **Packing:** `llm::pack` reads the checkpoint once (all 16 GB, and the
  per-layer embeddings as 11 GB of f32s; 2.5 minutes) and writes our own file
  (`.jdlgemma`, 4.4 GB): every matrix at 4 bits in groups of 32 with a fitted
  half scale and offset (the recogniser's rounding,
  `jodel_lyrics::asr::quant`), the norms as f32s, and the tokeniser.
- **Memory:** loading the weights (`Gemma::open`, read a buffer at a time:
  the embeddings with the per-layer projection, then a buffer each layer
  with its slice of the per-layer embeddings) and the room for a context
  (`Gemma::reserve`: the cache of keys and values, the working buffers of a
  prompt's chunk and a step, the rotary tables; `release` lets it go) are
  separate. `Memory::of(context)` says what each takes before anything is
  loaded: 4450 MB of weights (in 15 buffers, the layers three to one, each
  above the allocator's 256 MB blocks: a layer a buffer, two to a block,
  held 1.3 GB more), and 536 MB for 4041 positions (the story prompt and
  2500 tokens).
- **The architecture** is the reference's (Hugging Face's
  `Gemma4TextModel`): 42 layers of 2560, five sliding-window layers (a window
  of 512, heads of 256) to a global one (heads of 512, a quarter turned);
  8 query heads on 2 key/value heads, queries and keys normed per head, no
  scaling or cap of the scores; the last 18 layers on the keys and values of
  the last earlier layer of their kind; norms before and after attention and
  the GELU-gated feed-forward; each token's per-layer embeddings (and its
  embedding's projection, normed) gating a residual of their own in each
  layer; each layer's output scaled; the head tied to the embeddings, its
  logits capped at 30.
- **Tokeniser:** Gemma's byte-pair encoding, our own (`llm::Tokenizer`), as
  Hugging Face's `tokenizers` does it (specials first, the lowest-ranked
  merge first, byte pieces for what has no piece); it gives the reference's
  ids for the story prompt exactly.
- **A prompt** goes in chunks of up to 512 rows: the 24 layers with keys and
  values of their own over all of them (a tiled matrix multiply, 64 × 64
  tiles; flash attention over tiles of 8 keys, a query's 16 values an
  invocation), the other 18 for its last row alone.
- **A step** is recorded once: every kernel reads the position it decodes at
  from the GPU, and the head's largest logit (capped, the first of equals)
  is put after it as the next token, the position moved on. Steps go eight
  at a time between reads of the tokens, the next eight made while those
  run. A matrix multiply of one row (`gemv_*`) is 16 rows a workgroup, 32
  invocations a row each taking every 32nd half-group of inputs; attention
  is split (8 splits on a sliding layer, 16 on a global one, each taking
  every so many chunks of 64 positions with an online softmax) and joined.
- **Precision:** the rows a matrix multiplies are rounded to 8 bits a block
  of 32 (by the block's largest, its scaled sum kept for the weights'
  offsets, as llama.cpp's q8_1) and multiplied four at a time with
  `dot4I8Packed`, the weights' two nibbles a word unpacked into byte lanes;
  where that instruction is emulated (Metal) they stay f32s
  (`llm::Precision`; `JODEL_GEMMA_PRECISION=plain` or `q8` says otherwise).

On an RTX 5080 (November 2026), the Fast Car storyboard prompt of the story
prototype (1541 tokens): loading takes 5 s, the prompt 0.58 s (2650 tokens a
second), the answer 118 tokens a second (787 tokens in 6.7 s, at 8.5 ms a
step); the prototype's Hugging Face run at 4 bits (bitsandbytes' nf4) wrote
10.7 a second. The GPU holds 5.0 GB (5.2 GB in use, by the driver's count).
As f32s (`plain`) the prompt takes 1.03 s and a step 12 ms (83 a second).
Profiled, a step's GPU time goes mostly to the one-row matrix multiplies
(4.7 ms: the weights' 2.9 GB a token at 610 GB/s), then the residual
steps, the attention, its joins and the rows written for the matrices
(each of the last a few microseconds, 640 dispatches a step).

Against the reference (Hugging Face's own `Gemma4ForCausalLM` in f32 with
our file's weights dequantised, and every matrix's input rounded as ours
where we round it):
- **Plain:** the logits after the prompt agree to 2.0e-4 (of 28; 3.4e-5 on
  average), and the greedy answer is the reference's token for token, all
  1020 of them.
- **At 8 bits:** the rounding of every matrix's rows makes any difference
  of rounding grow: the reference itself, its embeddings' scale nudged by
  one part in a million, moves its logits by 0.53 on average (the best
  logit leads by 5.9). Ours are as far from it (0.48), and 0.31 from the
  plain path's, and the answers part after 15 tokens: as different as two
  roundings of the same model, each as good (both plans take the lyrics'
  18 lines in order, in 7 and 9 scenes).

`cargo test -p jodel-gpu` holds every kernel (both paths) to a reference on
the processor on small shapes, and the tokeniser to its rules.

`examples/gemma.rs` packs the checkpoint, writes after a prompt file (one
user turn) with the timings and the memory held, or writes the logits after
it.

## The story's pictures

`jodel_gpu::image` runs FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0)
from the checkpoints as published, read through `jodel-tensors`: the text
encoder Qwen3-4B from `Qwen/Qwen3-4B-FP8`, the transformer from
`black-forest-labs/FLUX.2-klein-4b-fp8`, the autoencoder from
`black-forest-labs/FLUX.2-klein-4B` (`vae/`). Its two parts load and free
on their own, so a card of 8 GB takes them in turns and one of 16 GB keeps
both:

- `TextEncoder` (2.54 GB): the prompt in the model's chat template (Qwen's
  byte-level BPE, `src/image/text.rs`), padded to 512 tokens; the first 27
  of Qwen3-4B's 36 layers (the embeddings are the hidden states after
  layers 9, 18 and 27, side by side), the token table kept on the
  processor. 0.2 s a prompt.
- `Painter` (3.95 GB, and 2.19 GB while it draws at 1344 × 768 with a
  reference of 512 × 512): the reference pictures through the autoencoder's
  encoder, then four steps of the transformer (5 double-stream and 20
  single-stream blocks; the references' tokens after the picture's, as the
  pipeline takes them) and the flow sampler (its shifted schedule, and the
  starting noise drawn as PyTorch draws it on the processor:
  `image::noise`, bit for bit), then the decoder. One plan of dispatches a
  picture size, kept until `release()`; the autoencoder's buffers are the
  transformer's, taken in turns.

`footprint()` says what each takes before it is loaded (its weights, and
its buffers for a picture size); `bytes()` what it takes now.

The FP8 matrices stay FP8 on the GPU, four values a word, turned to
numbers as the matrix multiply reads them, times the checkpoint's scales
(one a matrix for BFL's, one per 128 × 128 block for Qwen's); the bf16 ones
(the embedders, the modulations, the autoencoder) stay bf16. Activations
are f32. Where the GPU has cooperative matrices (tensor cores) of 16 × 16 ×
16 in f16 with f32 sums, subgroups of 32 and 44 KB of workgroup memory
(Vulkan on NVIDIA), the matrix multiplies and the transformer's attention
run on them (`Precision::Fast`; wgpu's experimental cooperative matrices),
their inputs rounded to f16; elsewhere, and with `Precision::Full`, on f32
multiply-adds. The kernels (`src/image/*.wgsl`):

- the matrix multiply (`gemm.wgsl`, `gemm_coop.wgsl`, the shared tail in
  `tail.wgsl`): tiles of 128 × 128, the next step of k read while this one
  is multiplied, each multiply planned as the engine's `Mul`; its tail
  fused: a bias, added to what is there, added times a gate per column
  (the modulation's gates), or SwiGLU (the gate and up matrices packed in
  pairs of 64 rows, so a tile holds both). A convolution is the same
  multiply reading its patches where they lie, the group norm and SiLU
  before it applied as it reads (doubled first for the decoder's
  upsampling; zero-padded after for the encoder's strided downsampling);
- flash attention (`attention.wgsl`, `attention_coop.wgsl`): an online
  softmax over tiles of keys, the scores never stored. On tensor cores 128
  queries a workgroup share each tile of 32 keys; the largest score is
  moved (and the sums rescaled) only when a tile's is 8 above it, so the
  exps fit f16;
- RMSNorm, LayerNorm with the modulation's shift and scale, the heads'
  norms and rotary positions (into a heads-first layout), the group norm's
  statistics (Welford per chunk, merged), the latents' 2 × 2 patches and
  the batch norm, a softmax and a transpose (the autoencoder's one-head
  attention: two matrix multiplies around a softmax, 2048 queries at a
  time).

Held against the PyTorch reference (diffusers' pipeline with the same FP8
weights) run in f32, on the bake-off's two scenes: the tokens and the noise
are the same; the text encoder's hidden states are within 6e-6 of their
spread (rms), the references' latents 4e-6, the transformer's first step
3e-5, the decoder 1e-6, and the pictures 74 and 68 dB PSNR (f32 here;
mean differences of 0.00 and 0.01 of 255); on tensor cores 52 and 45 dB
(0.14 and 0.36 of 255). The bake-off's own pictures (PyTorch in bf16) are
26.7 and 21.2 dB from the f32 reference: its rounding moves the picture
far more than ours. `examples/klein.rs` makes a picture, renders the
bake-off's scenes (`--bakeoff`: load times, seconds a picture, the GPU's
memory at its peak, the pixel errors), or holds each part against the
reference's dumps (`--check`); `cargo test -p jodel-gpu image` holds every
kernel to the processor on small shapes, on both paths.

On an RTX 5080 (Vulkan, tensor cores): a picture of 1344 × 768 with one
reference in 4.7 s (PyTorch in bf16: 2.9 s; here in f32 without tensor
cores: 11.8 s). Of its GPU time the matrix multiplies take 70 % (about 44
TFLOPS: the transformer's and the decoder's 146 TFLOPs) and the attention
28 % (about 29 TFLOPS); the rest is under 2 %. Loading takes 4.0 s for the
encoder and 3.3 s for the painter (PyTorch: 4.8 and 3.4 s). The GPU's
allocator holds 6.5 GB at its peak with the encoder freed first (PyTorch:
7.3 GB), 9.0 GB with both kept.

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
