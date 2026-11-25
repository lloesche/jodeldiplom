# jodel-rhythm

The player's ([`jodeldiplom`](../player)) analysis of music, in plain Rust
with no dependencies. It began as beat, downbeat and tempo analysis and is
still mostly that: two methods analyse the track, each through several
trackers, and a vote among them picks one beat grid and says how sure it
is. A meter then listens for what a listener hears (the backbeat, accents,
chord changes) to choose which level of that grid's pulse is the beat.

It holds more than the beat:

- **stems** (`stems`): a song separated into bass, drums, other
  instruments and vocals, which the beat analysis listens to and the
  player plays (its weights, 71.6 MB, are built in);
- **the key** (`key`, 1.9 MB of weights) and **the pitch of a voice**
  (`pitch`, 8.1 MB), for the player's display and karaoke;
- **the pitch of a voice as it is sung** (`yin`), live from a microphone;
- **waveforms** (`waveform`): each stem's for display, and the sung
  melody's notes;
- **the networks' shared parts** (`nn`): their layers, their weights file
  and its blocks of fewer bits, and the threads they take (half the
  processors, so playback and the interface keep theirs);
- **generated music** (`synth`) for its own tests, the GPU crate's and
  the player's.

So a crate that wants only a beat grid compiles all of it and builds in
about 90 MB of weights. The player uses every part; it shows the tempo,
the key and a beat pulse, and sings along.

The tools that make and measure the built-in pitch weights are a crate
of their own, kept out of the library: `jodel-rhythm-tools` in
[tools/](tools), beside the Python converters.

Timings below were measured on one machine, a 12th Gen Intel Core
i9-12900K (24 threads; the analysis takes 12) under Windows 11, in a
release build, while other builds were running on it.

```rust
use jodel_rhythm::{Analyzer, Method, SAMPLE_RATE};

let mut analyzer = Analyzer::with_methods(&[Method::Flux, Method::Network]);
analyzer.push(&mono_samples_at_22050_hz); // as often as they come
let analysis = analyzer.finish();
if let Some(summary) = analysis.summary {
    println!("{:.1} BPM, steady: {}", summary.bpm, summary.steady);
}
```

`finish` is two stages: `listen` gathers the `Evidence` (the cues and
every method's candidates), and `Evidence::decide` turns it into the
`Analysis`. Tools that try other meter weights call `decide` again
without listening again. `listen_with` takes the parts from elsewhere,
the stems (below), given by a function that is called while the network
and the flux candidates are worked out, so the stems can still be
finishing on their own thread.

## The parts and the cues

1. **Parts** (`parts`): the mel spectrogram split into drums and tonal
   sound. A median over 0.2 s along time keeps sustained sound, a median
   over 11 bands across frequency keeps hits, and soft masks (each
   median squared over the sum of both squared) share each value between
   the two. This is the fast split; with stems (below), the drum stem is
   `drums` and bass plus other instruments are `tonal`, each through the
   same mel bands.
2. **Cues** (`cues`), each a curve in `0..=1`:
   - `kick`: onsets in the drums below 150 Hz;
   - `snare`: onsets in the drums from 200 Hz to 5 kHz;
   - `hats`: onsets in the drums above 6 kHz;
   - `bass`: onsets in the tonal part below 250 Hz;
   - `change`: how much the tonal spectrum differs between the 0.15 s
     before and after each frame (chord changes).

   The onset cues use the flux of the activations, on a gentler
   compression (`ln(1 + 10·x)`): they must tell instruments apart by
   level, since a snare's noise reaches the kick's bands too, only
   weaker.

## Stems (`stems`)

Our own inference of Mini-BS-RoFormer-18M (HiDolen, MIT; adapted from
lucidrains' BS-RoFormer; trained on MUSDB18-HQ, whose research terms may
reach the weights: see [models/LICENSE-mini-bs-roformer](models/LICENSE-mini-bs-roformer)),
which separates stereo at 44.1 kHz into bass, drums, other instruments
and vocals.

1. **STFT** (`stems::stft`): 2,048-sample periodic Hann frames every 512
   samples, centred, the signal mirrored at the ends; the inverse adds
   the windowed frames back and divides by the summed squared window.
2. **The network** (`stems::model`, 17.9 million weights): each frame's
   1,025 bins (both channels, real and imaginary) cut into 62 bands,
   each through an RMSNorm and a layer to 256; a grid of frames × bands
   with 4 more of each (learned register tokens in the corner); 8 layers
   of a transformer along the frames and one along the bands (RMSNorm,
   attention with 8 query heads on 4 key/value heads and rotary
   positions over split halves, a gated GELU feed-forward step to 768);
   an RMSNorm; per stem and band a layer to a complex mask per bin and
   channel. Each stem is its mask times the mixture, transformed back.
3. **Chunks** (`stems::Splitter`): 6 s, every 4.8 s, faded in and out
   over 0.6 s, added and divided by the summed fades. The splitter takes
   the samples as they come and hands on each finished stretch, so a
   track is never held whole; `stems::PartsBuilder` makes the `Parts`
   from them. `Splitter::overlapped` (the player's analysis) has a thread
   of its own wait for each chunk's masks (on the GPU) while the next
   chunk is transformed and the one before turned back: the same stems,
   bit for bit, about a fifth sooner on an RTX 5080, each stretch handed
   on once the next chunk is in. Stem playback keeps `Splitter::new`, a
   chunk at a time, so its stems come no later.

A `stems::Separator` runs the network on a chunk's STFT. The built-in
`CpuSeparator` is the reference: within 0.005 % of the authors' PyTorch
code on each stem's magnitude per frame and per band
(`tests/fixtures/stems.txt`), and slow (about 3 minutes for a track).
The GPU crate [`jodel-gpu`](../gpu) gives another, a few seconds per track.
The weights are built in from [models/mini_bs_roformer.bin](models)
(`model.safetensors` of the model's page, SHA-256
`61b54904c702c3a4ff52f761bab8ad364554179f493df468439b4fc3a3930d6e`,
converted by [tools/convert_mini_bs_roformer.py](tools/convert_mini_bs_roformer.py)),
in single precision: half precision moved the vocal stem by 0.1 %.

The parts builder also makes each stem's waveform for display
(`waveform::Lane`, from `waveform::LaneBuilder`): the peak and the RMS of
each 10 ms of the stem in mono, each a byte on a scale from −60 dBFS
(0) to full scale (255). A `waveform::Waveform` holds the mix's lane and
the four stems' and is kept as bytes (`to_bytes`, `from_bytes`), about
50 KB a lane for four minutes.

## The methods

### `Method::Flux`: spectral flux

1. **Front end** (`frontend`): 1,024-sample Hann frames every 220
   samples at 22.05 kHz (about 100 frames per second), centred on their
   time; magnitudes through 128 triangular mel bands from 30 Hz to 10 kHz;
   `ln(1 + 1000·x)`. The FFT is our own (`fft`, a real FFT through a
   half-length complex one).
2. **Activations** (`activation`): curves in `0..=1` that rise where a beat
   is likely. `Flux::FULL` (all bands) and `Flux::LOW` (below 250 Hz: kick
   drums and bass) measure how much each band rose against the loudest of
   itself and its neighbours 20 ms earlier, so vibrato does not count. The
   sum loses its one-second moving mean and is scaled by the loudest value
   within three seconds.
3. Each activation goes through the `Hmm` and `Plp` trackers (below):
   four candidates.
4. With stems, the drum stem's full-band flux goes through the `Hmm`:
   one more candidate, voting only where the drums play (`presence`):
   - **present**: the stem's share of the music's energy over ±2 s, in
     decibels, through a logistic curve (half at −25 dB, 3 dB wide). On
     the tracks measured, drums and bass that play lie within 16 dB of
     the music; what leaks into an absent stem lies 37 to 58 dB below
     it;
   - **pulsing**: the highest peak of the stem's onset autocorrelation
     at a beat's lag (0.25 to 1.5 s) over ±4 s, through a logistic curve
     (half at 0.4). A drum groove reaches 0.55 to 0.88; fills, breaks and
     leakage stay lower.

   The flux is tracked where the drums are present (it is zero
   elsewhere), and the candidate keeps its beats where present ×
   pulsing, its say, is at least one half. On a list of 59 tracks it
   agreed with the other candidates 0.83 on average and raised the
   vote's mean confidence (0.823 to 0.827). The drum stem through `Plp`
   (0.75) and the bass stem through either tracker (0.37: a bass line's
   onsets are rarely periodic) lowered it, so they do not vote.

### `Method::Network`: the Beat This! network

Our own inference (`net`) of the small Beat This! network (Foscarin,
Schlüter and Widmer, "Beat this! Accurate beat tracking without DBN
postprocessing", ISMIR 2024), whose code and weights are under the MIT
licence ([models/LICENSE-beat-this](models/LICENSE-beat-this)).

1. **Its spectrogram** (`net::mel`): 1,024-sample periodic Hann frames
   every 441 samples (50 per second), the signal mirrored at both ends;
   magnitudes divided by √1024 through 128 mel bands from 30 Hz to 11 kHz
   on the Slaney scale; `ln(1 + 1000·x)`.
2. **The network** (`net::model`, about 2.1 million weights):
   - a convolution to 32 channels;
   - three blocks that run attention with rotary positions and a
     feed-forward step across the bands, then across the frames, then
     halve the bands and double the channels;
   - six transformer layers of 128;
   - per frame, a beat and a downbeat logit.

   A track runs in 30-second pieces that overlap by a quarter second.
   The layers (`nn::layers`, shared with the other networks) are plain
   loops written so the compiler vectorises them, with an AVX2 build of the
   hot ones picked at run time. The work is spread over half the
   processors (`nn::threads`). A four-minute track takes about 6.5 s.
   A `net::NetRunner` runs the pieces elsewhere (`jodel-gpu`'s `GpuNetwork`,
   the player's analysis wherever there is a GPU):
   `Analyzer::run_network_on` hands it each piece's spectrogram and takes
   its logits, and a piece it fails on runs here instead.
3. The beat probability goes through `Hmm`, `Plp` and `Peaks` (its own
   peaks above one half): three candidates, each with a larger say in the
   vote than the flux ones, since the network knows tempo levels the flux
   cannot tell apart. The downbeat probability finds the bars.

The weights are built in from [models/beat_this_small.bin](models): the
authors' `small0` checkpoint (SHA-256
`6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b`),
converted by [tools/convert_beat_this.py](tools/convert_beat_this.py). The
converter reads the checkpoint without PyTorch, and without running anything
in it. It folds each batch normalisation into a scale and shift, and keeps
single precision: half precision moved the logits by up to 0.012.
`tests/network.rs` checks our output against the authors' PyTorch code on
a generated track (`tests/fixtures/net_logits.txt`). Ours is within
0.00004 of it.

## Trackers (`track`)

- `Hmm`: a hidden Markov model over beat interval × position in the beat
  (35–260 BPM, 100 tempi on a log scale), decoded with Viterbi. Tempo may
  change at each beat with probability ∝ `exp(−100·|ratio − 1|)`, so drift
  is cheap and jumps cost what they should. The beat region (the first
  sixteenth of an interval) takes the activation as its likelihood, the
  rest `(1 − a) / 15`.
- `Plp`: predominant local pulse. A six-second tempogram every tenth of a
  second, a tempo path that pays for changes in octaves, and the windowed
  cosines of the chosen tempi added up; the peaks of their positive part
  are the beats.
- `Peaks`: frames that are the largest within ±70 ms and above 0.5; for
  activations that are already beat probabilities.

## The vote, the meter and what follows

1. **Vote** (`vote`): candidates are compared by their agreement: the
   F-measure (±70 ms) at the tempo level where they agree best (as they
   are, halved or doubled), with the agreement two unrelated grids reach
   by chance taken out. The candidate that agrees most with the others,
   each counted by its weight, wins, and that agreement is the confidence.
   The drum stem's candidate is compared with the others only over the
   stretches where it votes, and counts by its weight times its say
   summed over the track (drums that play half the track count half). It
   judges the others but does not win, since its beats cover only part
   of the track. A track without drums gets no such vote, and its vote
   is the other candidates' alone.
2. **Meter** (`meter`): the winner's grid with a pulse between each two
   beats is the pulse. A beat is every 1, 2, 4, 6 or 8 pulses (×2, ×1,
   ×½, ×⅓ or ×¼ the grid), at each phase. Around each beat of the grid
   (24 beats each side), each grouping is scored by:
   - **backbeat**: how much more snare one set of alternate beats has
     than the other. Counted at the listener's beat the snare is on every
     second beat; counted twice as fast it comes every fourth, so the
     difference halves; counted half as fast it is on every beat or on
     none. The kick is left out, since in hip-hop it often falls between
     the beats;
   - **accent**: kick, snare and bass at the beats against the pulses
     between, which rules out beats too slow to hold them all;
   - **change**: chord changes at the beats against between them;
   - **candidates**: the share (by weight, and a stem's candidate by its
     say there, over the beats where it votes) of the candidates that
     track these beats (F-measure from 0.6 up);
   - **preference**: a mild one for tempi near the centre tempo.

   With stems, the kick, snare and hats are scaled by how present the
   drums are, and the bass by how present the bass is (`Cues::new`), so
   what leaks into an absent stem is no cue, here or for the bars.

   A Viterbi path through the groupings picks one per grid beat. The
   grouping may change only on a beat both groupings share, so the beats
   keep their spacing. Each change costs `Weights::switch` (32 with the
   fitted weights): a grouping must score better by its whole price,
   summed over the beats, before the path moves to it, so only a long
   stretch (a half-time bridge) changes it. Where the grid skips, its
   interval holds as many pulses as fit at its usual spacing, so the
   count goes on through it; up to four usual intervals (a tracker that
   skipped beats or followed half the tempo for a while) those pulses
   may be beats, and a longer gap, a stretch without a beat, gets none.

   The weights (`meter::Weights`) were fitted on a list of 28 tracks
   with tempi tapped by a listener or believed, with the beats example
   (`--truth LIST --fit` in [the player](../player)): the accent counts
   3, the backbeat and the candidates 1 each, the preference 0.6
   (centred on 110 BPM). The chord change got no weight there (it still
   marks the bars). With the meter, 7 of 8 tapped tempi are right (5
   before): two songs whose beat is three eighth notes long now read at
   that beat, and a slow half-time groove at its half tempo.

   Cues from stems are cleaner but scaled differently, so they have their
   own set (`Weights::STEMS`, `Weights::for_parts`, used by
   `Evidence::conclude`): the chord change counts 4, the backbeat 1, the
   accent and the candidates 0.5, the preference 1 (centred on 120 BPM).
   On the same list: 7 of 8 tapped and 19 of 20 believed tempi right (the
   mix's set: 18 of 20), the Bach prelude among them.
3. **Alignment**: each beat moves onto the strongest full-band onset
   within 30 ms of it, where there is a clear one. The network places
   beats only to a frame or so (20 ms); the onsets are sharper.
4. **Bars** (`meter::downbeats`): each beat's accent from the kick and
   the chord change at it, plus the network's downbeat probability when
   it ran (which then counts four times as much). A Viterbi path over
   "position in a bar of 3 or of 4" collects the accents of the beats it
   puts first, and changes the meter only at a price.
5. **Regions** (`vote::regions`): the beats in regions of 16, each with
   the other candidates' agreement there with the chosen grid (a stem's
   candidate counted by its say there, over the beats where it votes),
   so how
   sure the analysis is of where the beats lie, whichever grouping the
   meter took.
6. **Summary** (`summary`): each beat's local tempo (the median of the
   intervals within four beats), and the track's tempo from the beats in
   regions where the candidates agree (`CLEAR`, 0.5). It is steady when
   they stay within 3 %, with the tempo then from straight-line fits
   through the runs of such beats near the median tempo; otherwise the
   median and the 10th–90th percentile range.

With one method, its own candidates vote. A new method is one more
`Method` with its candidates; nothing else changes. `VERSION` changes
whenever results do, so stored analyses are recomputed.

## The key (`key`)

`key::key_of` takes a song at 44.1 kHz mono and names one of the 24 major
and minor keys, with how sure it is. It is our own inference of the KeyNet
of [MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN) (Alexander
Sommer, after Korzeniowski and Widmer, ISMIR 2018; MIT, see
`models/LICENSE-musical-key-cnn`), its weights built in (`models/keynet.bin`,
1.8 MB, from `tools/convert_keynet.py`):

- **The spectrum.** A constant-Q spectrum as `librosa.cqt` makes it for the
  network: 105 bins, 24 an octave from 65 Hz, a frame every 0.2 s, each
  magnitude through `log1p`. With these settings librosa's
  octave-by-octave downsampling scales out exactly, so each bin is one
  Hann-windowed DFT at its frequency over its own length, at the full
  rate, on half the processors (`nn::threads`). The spectrum of a
  four-minute song takes about 0.07 s, the key about 0.17 s.
- **The network.** Nine convolutions with batch norm (folded into a scale
  and a shift) and ELU, three 2×2 max pools, the average over the whole
  song, 24 scores. The largest is the key; its softmax share is the
  confidence.
- **The classes** run round the Camelot wheel: 0–11 minor (0 is G♯ minor,
  1A), 12–23 major (12 is B major, 1B), each next a fifth up. `Key` gives
  the name ("E♭ major"), the Camelot place ("5B") and the scale's pitch
  classes.
- **Tags.** `Key::from_tag` reads a key tag in Camelot ("8A"), Open Key
  ("1d") or a note and mode ("Am", "F♯ minor", "Bbmaj"); the player takes
  a file's `INITIALKEY`/`TKEY` over the network's.

Against the original (PyTorch and librosa, a script kept out of this
repository) on seven songs at 44.1, 48 and 96 kHz: each layer within about
1e-6, the scores within 0.2 %, the same key on all seven.

## The pitch (`pitch`)

`pitch::pitch_of` takes a voice at 16 kHz mono (the vocals stem) and gives
its pitch 100 times a second, with how sure it is. It is our own inference
of [FCPE](https://github.com/CNChTu/FCPE) (CNChTu, MIT, see
`models/LICENSE-fcpe`), the convolution-only model bundled with torchfcpe
0.0.4:

- **The spectrogram.** As the authors' extractor: frames of 1024 samples
  every 160, mirrored 432 samples at each end, 128 Slaney mel bands to
  8 kHz, the natural log.
- **The network.** Two convolutions of 3 frames with a group norm between,
  six layers of a pointwise convolution, a GLU, a depthwise convolution of
  31 frames and another pointwise one (10.8 million weights), then 360
  sigmoids: pitches 20 cents apart from C1 to B6. A frame's pitch is the
  weighted mean of the nine classes round the strongest.
- **In windows.** The group norm takes its statistics over the whole song;
  the rest reaches 92 frames either way. So the statistics come first, from
  the first convolution alone, and the song goes through in windows of 30 s
  with 96 frames more each side: the same values as at once, without the
  song's activations in memory, on half the processors (`nn::threads`): a
  2:21 song in about 3.2 s.
- **The weights** are built in at about 6 bits a weight: Q5_K (`nn::quant`,
  the block layout of GGML's files: blocks of 256 with eight 6-bit scales
  and minimums under a half-precision scale and minimum), fitted by GPTQ on
  the products of the inputs of 120 voice stems, 8.1 MB against 43 MB.
  `tools/convert_fcpe.py` reads the checkpoint (weight norm folded); the
  `pitch_pack` tool (`jodel-rhythm-tools`) packs it into `models/pitch.bin`
  (`BTQ1`, see `nn::weights`), the first convolution and the depthwise ones
  as halves (their rows are not whole blocks), the vectors as they are.
  The fitting and GPTQ are the tools' (`tools/src/pack/`, shared with
  `pitch_quant`, which measures them); the library only reads the blocks.

Against the original (PyTorch, a script kept out of this repository), at
full precision: each layer within about 2e-6, the same pitch on every frame
of a whole song. On 300 voice stems not used for fitting (6.8 million
frames), against the full precision pitch:

| weights | MB | voiced in one only | median | over 50 cents | over 600 | note changed |
|---|---|---|---|---|---|---|
| halves | 21.7 | 0.004 % | 0.01 ¢ | 0.007 % | 0.004 % | 0.012 % |
| Q4_K, plain | 6.8 | 0.76 % | 1.19 ¢ | 1.34 % | 0.77 % | 2.36 % |
| Q4_K, rounding weighted by input energy | 6.8 | 0.50 % | 0.84 ¢ | 0.99 % | 0.58 % | 1.64 % |
| Q4_K, GPTQ | 6.8 | 0.28 % | 0.33 ¢ | 0.48 % | 0.27 % | 0.82 % |
| Q5_K, GPTQ (built in) | 8.1 | 0.14 % | 0.18 ¢ | 0.25 % | 0.14 % | 0.43 % |

The halves are almost exact, so the errors are the rounding's, not frames
the network can't decide; every layer adds some, the output projection and
the last input convolution most for their size.

On a whole mix (not a stem) Q5_K by GPTQ puts 1.6 % of frames over
50 cents, against 4 % for plain Q4_K.

## Limits

The candidates hear onsets, not beats: on music with hi-hats on the
eighth notes the trackers often follow the eighths, and the network
follows the eighth notes of 6/8. The meter fixes most of that from the
cues; without stems the cues come from a fast split of the mix, which a
busy mix blurs.

With stems the chord change counts most, and chords change on bars, so
the meter more often takes a coarser grouping for a stretch (95 % of a
track's time stays in its main grouping, against 98 % with the mix's
cues): one dance track reads its tempo right but is grouped in quarters
for most of it.

Half-time grooves (a kick-kick-clap with the snare on every other beat
of the published tempo) are heard either way by listeners, and the cues
cannot tell which: of two such songs one is heard at the published tempo
and one at half, and the meter takes the same level for both.

Music without a clear beat (ambient, much classical music) gets a low
confidence, and then the player shows no tempo. On a Bach prelude of
running sixteenths without drums the cues barely separate the levels,
and the grid of sixteenths often stays.

## Testing

`cargo test -p jodel-rhythm` runs:
- the unit tests;
- `tests/network.rs`: the network against the authors' output;
- `tests/synthetic.rs`: music generated in code (`synth`: drum kits,
  chords, clicks, plucked notes) with known beats, analysed with both
  methods and with each alone;
- the stems' unit tests: the transforms and their inverses, the
  separator against the authors' output on a generated clip, chunks
  faded back into the track, and kicks going to the drums and a chord
  to the tonal part.

`cargo test -p jodel-rhythm-tools` runs the packing's tests: its
fitting, GPTQ and file writer, read back by the library.

The synthetic cases:
- steady tempi of 90, 128 and 174 BPM;
- a ramp from 100 to 120 BPM, and a step from 90 to 140 BPM;
- swing;
- bars of 4 and of 3;
- a slow piece at 50 BPM;
- the beat a listener counts where the steadiest pulse is faster: a
  backbeat at 84 BPM under hi-hats on the eighths, 6/8 at 56 BPM in
  dotted quarter notes, arpeggios in sixteenths over quarter notes at
  60 BPM, and a backbeat that must stay at 172 BPM;
- drums in noise, and clicks alone;
- noise (not sure) and silence (no beat).

Real music is checked outside the repository: the beats example's
`--truth LIST` compares a list of tracks with tempi a listener tapped,
and `--fit` searches the meter's weights on it.

The checks are beat F-measure, tempo within 1 %, local tempo within 2 %
after a change, downbeats, and beats within 10 ms of the truth.
