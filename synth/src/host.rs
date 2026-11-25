//! The host: the voices, the cues and the effects, rendered a block at a
//! time.
//!
//! A note is cued on the frame it is to start on ([`Synth::note`]), its
//! patch's random spreads all drawn from its seed as it is cued. The block
//! that frame falls in starts it there, to the sample: a voice's first
//! block is rendered from that sample on. A voice is taken from the pool:
//! a free one, else the oldest of those let go, else the oldest of all;
//! the one taken is rendered there and then for the rest of the block,
//! fading to silence, before the new note takes its place. A mono patch's
//! note retriggers its own sounding voice on its frame: the voice plays up
//! to that sample as it was, then its engine starts the note on it (keeping
//! what it keeps) and its envelope rises again from where it is.
//!
//! Each block a voice's envelope, glides and LFO are worked out once; the
//! engine renders it at unity level, and the host ramps its amplitude from
//! the last block's to this one's over the samples rendered. A voice ends
//! when its envelope's release is over, or its engine says so, on a last
//! block ramping to silence.
//!
//! The bus: each voice goes dry into the two sides and, by its patch's
//! sends, into a chorus (stereo, its chorused return alone), a tempo-synced
//! delay (stereo) and a hall (mono in, stereo out), whose returns are
//! added. Last, a gentle compressor and a look-ahead limiter (ceiling
//! -1 dBFS) keep stacked sounds from turning harsh or clipping; the
//! limiter holds the output back by [`Synth::latency`] frames.
//!
//! The synth is driven from one thread: cue notes and render on the same
//! one (an audio callback that also runs the conductor). Game events from
//! another thread need a single-producer, single-consumer queue of cues to
//! it.

use crate::chorus::{Chorus, ChorusSettings};
use crate::delay::{Delay, DelaySettings};
use crate::dsp::{self, Rate};
use crate::dynamics::{Compressor, CompressorSettings, Limiter, LimiterSettings};
use crate::engines::{AnyVoice, Engines};
use crate::env::Envelope;
use crate::hall::{Hall, HallSettings};
use crate::patch::{LfoRate, Patch};
use crate::{NoteOn, VoiceMod, BLOCK};

/// The notes that can wait to start at once; more are dropped (and
/// counted: [`Synth::dropped`]).
pub const CUES: usize = 256;

/// The level of a full-scale voice at velocity 1 and 0 dB: half scale, so
/// a few voices together still leave headroom.
const HEADROOM: f32 = 0.5;

/// The bus compressor's settings at the start: gentle, only taking the
/// edge off a pile of sounds (2:1 from -10 dBFS, a wide knee).
pub const GENTLE: CompressorSettings = CompressorSettings {
    threshold: -10.0,
    ratio: 2.0,
    knee: 8.0,
    attack: 0.005,
    release: 0.15,
    makeup: 0.0,
};

/// A patch in a [`Bank`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PatchId(u16);

/// The patches a synth plays, registered before it is made.
#[derive(Clone, Debug, Default)]
pub struct Bank {
    patches: Vec<Patch>,
}

impl Bank {
    /// An empty bank.
    pub fn new() -> Bank {
        Bank::default()
    }

    /// Adds `patch`; its id in the synth made of this bank.
    ///
    /// # Panics
    ///
    /// Past 65536 patches.
    pub fn add(&mut self, patch: Patch) -> PatchId {
        let id = u16::try_from(self.patches.len()).expect("at most 65536 patches");
        self.patches.push(patch);
        PatchId(id)
    }

    /// Its patches.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

/// A sounding voice, as [`Synth::voices`] shows it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceView {
    /// Its patch.
    pub patch: PatchId,
    /// Its note.
    pub note: f32,
    /// Whether it is still held (not yet let go).
    pub held: bool,
}

/// A note waiting for its frame, its spreads drawn.
#[derive(Clone, Copy, Debug, Default)]
struct Cue {
    frame: u64,
    patch: u16,
    /// The note, its seed the engine's own (drawn from the note's).
    note: NoteOn,
    /// How long it is held, seconds.
    gate: f32,
    /// Its detune, semitones; its gain (velocity, level, headroom); the
    /// LFO's starting phase.
    detune: f32,
    gain: f32,
    lfo: u32,
}

/// A voice sounding.
#[derive(Clone, Debug)]
struct Sounding {
    patch: u16,
    voice: AnyVoice,
    env: Envelope,
    note: f32,
    detune: f32,
    gain: f32,
    /// Its gains left and right.
    pan: [f32; 2],
    /// Seconds since it started, and how long it is held.
    t: f32,
    gate: f32,
    /// The LFO's phase.
    lfo: u32,
    /// The last block's amplitude and modulation.
    amp: f32,
    last: VoiceMod,
    /// When it started, among all notes.
    serial: u64,
    /// The sample of the block it is rendered from (0 once it has sounded
    /// a block).
    start: usize,
}

/// What the voices add to in a block: the two sides and the sends.
struct Bus {
    left: [f32; BLOCK],
    right: [f32; BLOCK],
    reverb: [f32; BLOCK],
    delay: [[f32; BLOCK]; 2],
    chorus: [[f32; BLOCK]; 2],
}

impl Bus {
    fn new() -> Bus {
        Bus {
            left: [0.0; BLOCK],
            right: [0.0; BLOCK],
            reverb: [0.0; BLOCK],
            delay: [[0.0; BLOCK]; 2],
            chorus: [[0.0; BLOCK]; 2],
        }
    }

    /// `mono[from..to]` at an amplitude ramping from `a0` at `from` towards
    /// `a1` at `to`, at `pan`, with `patch`'s sends.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        mono: &[f32; BLOCK],
        from: usize,
        to: usize,
        a0: f32,
        a1: f32,
        pan: [f32; 2],
        patch: &Patch,
    ) {
        let step = (a1 - a0) / (to - from).max(1) as f32;
        for (k, i) in (from..to).enumerate() {
            let x = mono[i] * (a0 + step * k as f32);
            let (l, r) = (x * pan[0], x * pan[1]);
            self.left[i] += l;
            self.right[i] += r;
            self.reverb[i] += x * patch.reverb;
            self.delay[0][i] += l * patch.delay;
            self.delay[1][i] += r * patch.delay;
            self.chorus[0][i] += l * patch.chorus;
            self.chorus[1][i] += r * patch.chorus;
        }
    }
}

/// The synthesizer.
pub struct Synth {
    rate: Rate,
    patches: Vec<Patch>,
    engines: Engines,
    /// The pool of voices, made once.
    voices: Box<[Option<Sounding>]>,
    /// The notes waiting, in the order of their frames.
    cues: [Cue; CUES],
    waiting: usize,
    dropped: u64,
    stolen: u64,
    serial: u64,
    tempo: f32,
    hall: Hall,
    delay: Delay,
    chorus: Chorus,
    /// The chorus's return level (its settings' mix).
    chorus_return: f32,
    compressor: Compressor,
    limiter: Limiter,
    /// The last block rendered, and how much of it is still to hand out.
    block: [[f32; BLOCK]; 2],
    pending: usize,
    /// The frame of the next block's first sample.
    next: u64,
}

impl Synth {
    /// A synth at `rate` Hz playing `bank` on a pool of `voices` voices, at
    /// 120 beats a minute, the hall, the delay and the chorus at their
    /// defaults, the compressor gentle ([`GENTLE`]) and the limiter's
    /// ceiling -1 dBFS.
    ///
    /// # Panics
    ///
    /// If `rate` is not one the engines take ([`Rate::MIN`] to
    /// [`Rate::MAX`]), or `voices` is 0.
    pub fn new(rate: u32, bank: Bank, voices: usize) -> Synth {
        assert!(voices > 0, "a synth needs at least one voice");
        let rate = Rate::new(rate);
        Synth {
            rate,
            patches: bank.patches,
            engines: Engines::new(rate),
            voices: vec![None; voices].into_boxed_slice(),
            cues: [Cue::default(); CUES],
            waiting: 0,
            dropped: 0,
            stolen: 0,
            serial: 0,
            tempo: 120.0,
            hall: Hall::new(rate),
            delay: Delay::new(rate),
            chorus: {
                let mut c = Chorus::new(rate);
                c.set(ChorusSettings {
                    mix: 1.0,
                    ..ChorusSettings::default()
                });
                c
            },
            chorus_return: ChorusSettings::default().mix,
            compressor: {
                let mut c = Compressor::new(rate);
                c.set(GENTLE);
                c
            },
            limiter: Limiter::new(rate),
            block: [[0.0; BLOCK]; 2],
            pending: 0,
            next: 0,
        }
    }

    /// Its rate, Hz.
    pub fn rate(&self) -> f32 {
        self.rate.hz()
    }

    /// The frame [`Synth::render_to`] hands out next (0 at the start).
    pub fn frame(&self) -> u64 {
        self.next - self.pending as u64
    }

    /// Cues `note` of `patch` to start on frame `frame` (with the patch's
    /// start jitter), held for `length` frames (none: the patch's gate). A
    /// frame already rendered starts it as soon as it can. The note's seed
    /// draws its spreads and its engine's noise. Whether it was taken: not
    /// when the queue is full (it is counted then) or the patch is not in
    /// the bank.
    pub fn note(&mut self, frame: u64, patch: PatchId, note: NoteOn, length: Option<u64>) -> bool {
        let Some(p) = self.patches.get(usize::from(patch.0)) else {
            return false;
        };
        if self.waiting == CUES {
            self.dropped += 1;
            return false;
        }
        let mut r = dsp::seed(note.seed);
        let jitter = dsp::unit(&mut r) * p.spread.start.max(0.0) * self.rate.hz();
        let detune = dsp::white(&mut r) * p.spread.detune / 100.0;
        let level = p.level + dsp::white(&mut r) * p.spread.level;
        let lfo = (dsp::unit(&mut r) * p.spread.lfo_phase.clamp(0.0, 1.0) * 4_294_967_296.0) as u32;
        let cue = Cue {
            frame: frame + jitter as u64,
            patch: patch.0,
            note: NoteOn {
                seed: dsp::xorshift(&mut r),
                ..note
            },
            gate: length.map_or(p.gate, |n| n as f32 / self.rate.hz()),
            detune,
            gain: note.velocity.clamp(0.0, 1.0) * 10f32.powf(level / 20.0) * HEADROOM,
            lfo,
        };
        // After those cued for the same frame, so they start in the order
        // they came.
        let at = self.cues[..self.waiting].partition_point(|c| c.frame <= cue.frame);
        self.cues.copy_within(at..self.waiting, at + 1);
        self.cues[at] = cue;
        self.waiting += 1;
        true
    }

    /// The notes dropped because the queue was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The sounding voices taken for other notes.
    pub fn stolen(&self) -> u64 {
        self.stolen
    }

    /// The tempo the LFOs and the delay count beats at, beats a minute.
    pub fn set_tempo(&mut self, bpm: f32) {
        self.tempo = bpm.max(1.0);
        self.delay.retime(self.tempo);
    }

    /// The hall's settings (the patches' reverb send) from the next block
    /// on.
    pub fn set_reverb(&mut self, s: HallSettings) {
        self.hall.set(s);
    }

    /// The chorus's (the patches' chorus send). Its mix is the return's
    /// level: the chorused signal alone comes back, 0 none of it, 1 all.
    pub fn set_chorus(&mut self, s: ChorusSettings) {
        self.chorus_return = s.mix.clamp(0.0, 1.0);
        self.chorus.set(ChorusSettings { mix: 1.0, ..s });
    }

    /// The compressor's.
    pub fn set_compressor(&mut self, s: CompressorSettings) {
        self.compressor.set(s);
    }

    /// The limiter's.
    pub fn set_limiter(&mut self, s: LimiterSettings) {
        self.limiter.set(s);
    }

    /// The frames the limiter holds the output back by: a note cued on
    /// frame f is heard from frame f plus this.
    pub fn latency(&self) -> usize {
        self.limiter.latency()
    }

    /// The delay's (for setting up: its time jumps; a tempo change glides).
    pub fn set_delay(&mut self, s: DelaySettings) {
        self.delay.set(s, self.tempo);
    }

    /// The voices sounding.
    pub fn voices(&self) -> impl Iterator<Item = VoiceView> + '_ {
        self.voices.iter().flatten().map(|v| VoiceView {
            patch: PatchId(v.patch),
            note: v.note,
            held: v.env.held(),
        })
    }

    /// Renders `out.len() / 2` frames into `out`, interleaved stereo (an
    /// odd last sample is left as it is).
    pub fn render_to(&mut self, out: &mut [f32]) {
        for frame in out.chunks_exact_mut(2) {
            if self.pending == 0 {
                self.render_block();
                self.pending = BLOCK;
            }
            let i = BLOCK - self.pending;
            frame[0] = self.block[0][i];
            frame[1] = self.block[1][i];
            self.pending -= 1;
        }
    }

    fn render_block(&mut self) {
        let mut bus = Bus::new();
        let end = self.next + BLOCK as u64;
        let due = self.cues[..self.waiting].partition_point(|c| c.frame < end);
        for k in 0..due {
            let cue = self.cues[k];
            self.start(&cue, cue.frame.saturating_sub(self.next) as usize, &mut bus);
        }
        self.cues.copy_within(due..self.waiting, 0);
        self.waiting -= due;
        let play = Play {
            rate: self.rate,
            tempo: self.tempo,
            patches: &self.patches,
            engines: &self.engines,
        };
        for slot in self.voices.iter_mut() {
            if let Some(v) = slot {
                if !play.block(v, &mut bus) {
                    play.engines.free(&mut v.voice);
                    *slot = None;
                }
            }
        }
        let [mut chorus_l, mut chorus_r] = bus.chorus;
        self.chorus.process(&mut chorus_l, &mut chorus_r);
        for i in 0..BLOCK {
            bus.left[i] += chorus_l[i] * self.chorus_return;
            bus.right[i] += chorus_r[i] * self.chorus_return;
        }
        let [send_l, send_r] = &bus.delay;
        self.delay
            .process([send_l, send_r], &mut bus.left, &mut bus.right);
        self.hall
            .process(&bus.reverb, &mut bus.left, &mut bus.right);
        self.compressor.process(&mut bus.left, &mut bus.right);
        self.limiter.process(&mut bus.left, &mut bus.right);
        self.block = [bus.left, bus.right];
        self.next = end;
    }

    /// Starts `cue` at sample `offset` of the block.
    fn start(&mut self, cue: &Cue, offset: usize, bus: &mut Bus) {
        let offset = offset.min(BLOCK - 1);
        let play = Play {
            rate: self.rate,
            tempo: self.tempo,
            patches: &self.patches,
            engines: &self.engines,
        };
        let patch = &play.patches[usize::from(cue.patch)];
        let angle = (cue.note.pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
        let pan = [angle.cos(), angle.sin()];
        self.serial += 1;
        let own = if patch.mono {
            self.voices
                .iter_mut()
                .flatten()
                .find(|v| v.patch == cue.patch)
        } else {
            None
        };
        if let Some(v) = own {
            // A retrigger: as it was up to the note's sample, then the note
            // on it, its envelope rising again from where it is.
            play.until(v, offset, bus);
            play.engines.note_on(&patch.engine, &mut v.voice, &cue.note);
            v.env = Envelope::start(v.env.level());
            v.note = cue.note.note;
            v.detune = cue.detune;
            v.gain = cue.gain;
            v.pan = pan;
            v.t = 0.0;
            v.gate = cue.gate;
            v.serial = self.serial;
            return;
        }
        let k = free_or_oldest(&self.voices);
        if let Some(mut old) = self.voices[k].take() {
            play.fade(&mut old, bus);
            play.engines.free(&mut old.voice);
            self.stolen += 1;
        }
        let mut voice = play.engines.voice(&patch.engine);
        play.engines.note_on(&patch.engine, &mut voice, &cue.note);
        self.voices[k] = Some(Sounding {
            patch: cue.patch,
            voice,
            env: Envelope::start(0.0),
            note: cue.note.note,
            detune: cue.detune,
            gain: cue.gain,
            pan,
            t: 0.0,
            gate: cue.gate,
            lfo: cue.lfo,
            amp: 0.0,
            last: VoiceMod::default(),
            serial: self.serial,
            start: offset,
        });
    }
}

/// A free voice, else the oldest let go, else the oldest.
fn free_or_oldest(voices: &[Option<Sounding>]) -> usize {
    if let Some(k) = voices.iter().position(Option::is_none) {
        return k;
    }
    let oldest = |held: bool| {
        voices
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.as_ref().map(|v| (k, v)))
            .filter(|(_, v)| held || !v.env.held())
            .min_by_key(|(_, v)| v.serial)
            .map(|(k, _)| k)
    };
    oldest(false).or_else(|| oldest(true)).unwrap_or(0)
}

/// What a block of voices plays with.
struct Play<'a> {
    rate: Rate,
    tempo: f32,
    patches: &'a [Patch],
    engines: &'a Engines,
}

impl Play<'_> {
    /// One block of `v` into `bus`, from its start on; whether it sounds on.
    fn block(&self, v: &mut Sounding, bus: &mut Bus) -> bool {
        let patch = &self.patches[usize::from(v.patch)];
        let n = BLOCK - v.start;
        let dt = n as f32 / self.rate.hz();
        if v.t >= v.gate {
            v.env.release();
        }
        let adsr = v.env.advance(&patch.adsr, dt);
        // The LFO, after its delay and coming in over its fade.
        let lfo = &patch.lfo;
        let depth = if lfo.fade > 0.0 {
            ((v.t - lfo.delay) / lfo.fade).clamp(0.0, 1.0)
        } else if v.t >= lfo.delay {
            1.0
        } else {
            0.0
        };
        let raw = dsp::sine(v.lfo);
        let wave = raw * depth;
        let hz = match lfo.rate {
            LfoRate::Hz(hz) => hz,
            LfoRate::Beats(beats) => self.tempo / 60.0 / beats.max(1e-3),
        };
        v.lfo = v
            .lfo
            .wrapping_add(self.rate.inc_hz(hz).wrapping_mul(n as u32));
        let amp = self.engines.amp(&patch.engine, &mut v.voice, adsr);
        let ending = v.env.done() || self.engines.done(&v.voice);
        let tremolo = 1.0 - lfo.amp * depth * (0.5 - 0.5 * raw);
        let amp1 = if ending { 0.0 } else { amp * v.gain * tremolo };
        let pitch = v.note + v.detune + patch.pitch.at(v.t) + wave * lfo.pitch;
        let m = VoiceMod {
            phase_inc: self.rate.inc(pitch),
            pitch,
            bright: patch.filter_env * adsr + wave * lfo.bright,
            shape: wave * lfo.shape + patch.shape.at(v.t),
        };
        let mut mono = [0.0; BLOCK];
        self.engines
            .render(&patch.engine, &mut v.voice, &m, &mut mono[v.start..]);
        bus.add(&mono, v.start, BLOCK, v.amp, amp1, v.pan, patch);
        v.amp = amp1;
        v.last = m;
        v.t += dt;
        v.start = 0;
        !ending
    }

    /// The rest of the block of `v`, taken for another note, fading to
    /// silence.
    fn fade(&self, v: &mut Sounding, bus: &mut Bus) {
        if v.amp == 0.0 {
            return;
        }
        let patch = &self.patches[usize::from(v.patch)];
        let mut mono = [0.0; BLOCK];
        self.engines
            .render(&patch.engine, &mut v.voice, &v.last, &mut mono[v.start..]);
        bus.add(&mono, v.start, BLOCK, v.amp, 0.0, v.pan, patch);
    }

    /// `v` as it is up to sample `to` (a retrigger's), at its last block's
    /// modulation and amplitude.
    fn until(&self, v: &mut Sounding, to: usize, bus: &mut Bus) {
        if v.t == 0.0 || to <= v.start {
            // Not sounded yet: it starts there anew.
            v.start = v.start.max(to);
            return;
        }
        let patch = &self.patches[usize::from(v.patch)];
        let mut mono = [0.0; BLOCK];
        self.engines
            .render(&patch.engine, &mut v.voice, &v.last, &mut mono[v.start..to]);
        bus.add(&mono, v.start, to, v.amp, v.amp, v.pan, patch);
        v.start = to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_is_small() {
        // A slot of the pool stays under a kilobyte, so 128 voices take
        // about 100 KB (the strings' lines live in their engine's pool).
        let slot = std::mem::size_of::<Option<Sounding>>();
        println!("a voice slot: {slot} bytes");
        assert!(slot <= 1024, "{slot}");
    }
}
