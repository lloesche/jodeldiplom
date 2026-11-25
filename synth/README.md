# jodel-synth

A small real-time synthesizer for the karaoke game's sound effects. Ten engines of our own make the sounds:

- NOISE: noise through a resonant filter, from hiss to a chip's noise register.
- ANALOG: two band-limited oscillators, drive and a state-variable filter.
- LOFI: a chip voice.
- PHASE: phase distortion.
- MODAL: a struck body's modes, for bells, glass, coins, kalimba and marimba.
- STRING: an extended Karplus-Strong string.
- FORMANT: a sung or whispered vowel.
- FM: four sine operators modulating one another's phase, for glass bells, an electric piano, glints, sparkles, chimes and brass.
- SUPERSAW: up to seven detuned band-limited saws, for pads and big chords.
- DRUM: synthesized drums: kicks, a snare, a clap, hats, toms, a rim and a crash.

Each module names the published papers and documentation its technique comes from.

A host plays them on a pool of voices whose size it is given once (128 is fine), in blocks of 32 samples; the engines render at unity level and the host ramps each voice's amplitude. A patch is an engine with its parameters, an ADSR, glides of the pitch and the shape, an LFO (in hertz or in beats), sends to the shared effects, and random spreads drawn from each note's seed. Notes are cued on the frame they start on, to the sample. Nothing allocates once the synth is made, so it can render on an audio thread; rendering offline is the same code.

The bus: the voices go dry into the two sides and, by their sends, into a stereo chorus and ensemble, a tempo-synced delay and a hall (a feedback delay network); last, a gentle compressor and a look-ahead limiter with a ceiling of -1 dBFS keep a pile of sounds from clipping.

`cargo run --release -p jodel-synth --example patches -- OUT_DIR` renders the demo palette to WAVs in OUT_DIR's `synth`, `synth-physical`, `synth-fm` and `synth-drums`; `--example bench` times a full pool.

Inspired by DaisySP (Electrosmith, MIT), Plaits (Emilie Gillet, MIT) and Felucca-Engines (Leo Kuroshita / Hügelton Instruments, MIT); all engine code here is our own, written from the published papers each module cites.
