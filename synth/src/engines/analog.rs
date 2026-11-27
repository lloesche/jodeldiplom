//! ANALOG: a subtractive voice for brass, plucks, zaps and leads. Two
//! band-limited oscillators (the second detuned, or hard-synced to the
//! first), a square an octave down, noise, a drive into a soft clip, and a
//! resonant state-variable filter that follows the note.
//!
//! The host's brightness (the filter envelope, the LFO) moves the cutoff
//! ([`BRIGHT`]); its shape the pulse width or, synced, the second
//! oscillator's interval (the sync sweep, [`SHAPE`]). The cutoff follows
//! the voice's pitch as it is, glides and all, so a falling "pew" takes its
//! filter down with it.
//!
//! Sources: the oscillators and the band-limited sync restart by polyBLEP
//! (Välimäki and Huovilainen 2007; Välimäki, Pekonen and Nam 2012: the
//! residual scaled by the size of the jump, on the samples either side of
//! it); the filter by Simper's trapezoidal SVF (Cytomic, 2013); the noise
//! by xorshift32 (Marsaglia 2003); the soft clip a Padé approximant of
//! tanh.

use crate::dsp::{self, Rate, Svf, SvfState};
use crate::params::{field, options, pick, switch, when, Param};
use crate::{Engine, NoteOn, VoiceMod};

/// The oscillators' wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// Saw.
    Saw,
    /// Square.
    Square,
    /// A pulse of the width.
    Pulse,
    /// Triangle.
    Triangle,
    /// Sine.
    Sine,
}

/// The filter's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Low-pass.
    LowPass,
    /// Band-pass.
    BandPass,
    /// High-pass.
    HighPass,
    /// Notch.
    Notch,
}

/// ANALOG's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The oscillators' wave (synced, the second is a saw).
    pub wave: Wave,
    /// [`Wave::Pulse`]'s width, 0.05 to 0.95 of a cycle.
    pub width: f32,
    /// The second oscillator's interval above the first, semitones.
    pub interval: f32,
    /// Its detune, cents.
    pub detune: f32,
    /// Its level against the first, 0 (the first alone) to 1 (it alone).
    pub mix: f32,
    /// The second hard-synced to the first: it restarts with each of the
    /// first's cycles, so its interval sets a timbre, not a pitch.
    pub sync: bool,
    /// The square an octave below the first, 0 to 1.
    pub sub: f32,
    /// White noise, 0 to 1.
    pub noise: f32,
    /// The drive into the soft clip, 0 (none) to 1 (x8).
    pub drive: f32,
    /// The filter's output.
    pub filter: Filter,
    /// Its cutoff at middle C, Hz.
    pub cutoff: f32,
    /// Its resonance, 0 to 1.
    pub res: f32,
    /// How the cutoff follows the pitch around middle C: 1 a semitone a
    /// semitone.
    pub key_track: f32,
}

impl Default for Params {
    fn default() -> Params {
        PLAIN
    }
}

options! {
    /// The waves and the filter's outputs, in their order, and their names.
    WAVES, WAVE_NAMES: Wave = [
        Wave::Saw => "saw",
        Wave::Square => "square",
        Wave::Pulse => "pulse",
        Wave::Triangle => "triangle",
        Wave::Sine => "sine",
    ]
}
options! {
    FILTERS, FILTER_NAMES: Filter = [
        Filter::LowPass => "low-pass",
        Filter::BandPass => "band-pass",
        Filter::HighPass => "high-pass",
        Filter::Notch => "notch",
    ]
}

/// ANALOG's parameters as data, with their ranges.
pub const PARAMS: &[Param<Params>] = &[
    pick!("osc", "wave", wave, WAVES, WAVE_NAMES),
    when(field!("osc", "width", width, 0.05, 0.95, "", Linear), |p| {
        p.wave == Wave::Pulse
    }),
    field!("osc 2", "interval", interval, -24.0, 24.0, "st", Linear),
    field!("osc 2", "detune", detune, -100.0, 100.0, "ct", Linear),
    field!("osc 2", "mix", mix, 0.0, 1.0, "", Linear),
    switch!("osc 2", "sync", sync),
    field!("mix", "sub", sub, 0.0, 1.0, "", Linear),
    field!("mix", "noise", noise, 0.0, 1.0, "", Linear),
    field!("mix", "drive", drive, 0.0, 1.0, "", Linear),
    pick!("filter", "type", filter, FILTERS, FILTER_NAMES),
    field!("filter", "cutoff", cutoff, 20.0, 20_000.0, "Hz", Log),
    field!("filter", "res", res, 0.0, 1.0, "", Linear),
    field!("filter", "key track", key_track, 0.0, 2.0, "", Linear),
];

/// [`Params::default`]: one saw through an open low-pass.
const PLAIN: Params = Params {
    wave: Wave::Saw,
    width: 0.5,
    interval: 0.0,
    detune: 0.0,
    mix: 0.0,
    sync: false,
    sub: 0.0,
    noise: 0.0,
    drive: 0.0,
    filter: Filter::LowPass,
    cutoff: 12_000.0,
    res: 0.0,
    key_track: 0.0,
};

/// Two detuned saws, a little drive, the filter opened by the envelope.
pub const BRASS: Params = Params {
    detune: 9.0,
    mix: 0.5,
    drive: 0.2,
    cutoff: 500.0,
    res: 0.15,
    key_track: 0.5,
    ..PLAIN
};

/// A saw and a square a fifth up, dark, to be opened by a quick envelope.
pub const PLUCK: Params = Params {
    interval: 7.0,
    detune: 4.0,
    mix: 0.3,
    cutoff: 300.0,
    res: 0.35,
    key_track: 0.6,
    ..PLAIN
};

/// A synced saw: swept by the shape or the pitch, a zap.
pub const ZAP: Params = Params {
    interval: 10.0,
    mix: 1.0,
    sync: true,
    cutoff: 9_000.0,
    res: 0.2,
    key_track: 0.3,
    ..PLAIN
};

/// A square and its sub, round.
pub const BASS: Params = Params {
    wave: Wave::Square,
    sub: 0.6,
    drive: 0.3,
    cutoff: 700.0,
    res: 0.3,
    key_track: 0.4,
    ..PLAIN
};

/// A narrow pulse pair, bright.
pub const LEAD: Params = Params {
    wave: Wave::Pulse,
    width: 0.3,
    detune: 12.0,
    mix: 0.5,
    cutoff: 4_000.0,
    res: 0.25,
    key_track: 0.5,
    ..PLAIN
};

/// A pure sine (a sub-free tone or a soft blip).
pub const SINE: Params = Params {
    wave: Wave::Sine,
    ..PLAIN
};

/// The presets, by name.
pub const PRESETS: [(&str, Params); 6] = [
    ("BRASS", BRASS),
    ("PLUCK", PLUCK),
    ("ZAP", ZAP),
    ("BASS", BASS),
    ("LEAD", LEAD),
    ("SINE", SINE),
];

/// What the host's shape moves: the pulse width by up to 0.45 either way,
/// or, synced, the second oscillator's interval by up to an octave.
pub const SHAPE: &str = "pulse width / sync interval";

/// What the host's brightness opens: the cutoff, an octave an octave.
pub const BRIGHT: &str = "cutoff";

/// One ANALOG voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    ph1: u32,
    ph2: u32,
    sub: u32,
    filter: SvfState,
    /// The noise's generator (0 until the voice first sounds).
    noise: u32,
    /// Synced: what a restart left for the next sample.
    carry: f32,
}

/// ANALOG at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Analog {
    rate: Rate,
}

/// A plain saw at phase `ph`, -1 to 1.
#[inline]
fn naive_saw(ph: u32) -> f32 {
    ph as f32 * (1.0 / 2_147_483_648.0) - 1.0
}

/// One sample of `wave` at `ph` stepping by `inc`, `width` for a pulse.
#[inline]
fn osc(wave: Wave, ph: u32, inc: u32, width: u32) -> f32 {
    match wave {
        Wave::Saw => dsp::saw(ph, inc),
        Wave::Square => dsp::pulse(ph, inc, 1 << 31),
        Wave::Pulse => dsp::pulse(ph, inc, width),
        Wave::Triangle => dsp::triangle(ph),
        Wave::Sine => dsp::sine(ph),
    }
}

impl Engine for Analog {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Analog {
        Analog { rate }
    }

    /// A voice from silence: its noise seeded, the second oscillator a
    /// quarter cycle on from the first (no doubled edge at the start). A
    /// retrigger keeps the phases, the noise and the filter ringing.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        if v.noise == 0 {
            v.noise = dsp::seed(n.seed);
            v.ph2 = v.ph1.wrapping_add(0x4000_0000);
        }
        v.carry = 0.0;
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let inc1 = m.phase_inc;
        let sweep = if p.sync { m.shape * 12.0 } else { 0.0 };
        let inc2 = self
            .rate
            .inc(m.pitch + p.interval + p.detune / 100.0 + sweep);
        let inc_sub = inc1 >> 1;
        let width = (p.width + if p.sync { 0.0 } else { m.shape * 0.45 }).clamp(0.05, 0.95);
        let width = (width * 4_294_967_296.0) as u32;
        let (m1, m2) = (1.0 - p.mix, p.mix);
        let noise = p.noise * 0.5;
        let drive = 1.0 + 7.0 * p.drive;
        let hz = p.cutoff * (m.bright + p.key_track * (m.pitch - 60.0) / 12.0).exp2();
        let svf = Svf::new(hz, dsp::damping(p.res), self.rate);
        for o in out.iter_mut() {
            let a = osc(p.wave, v.ph1, inc1, width);
            let b = if p.sync {
                // A saw restarted by the first: the polyBLEP of each jump
                // (its own wrap, or the restart, scaled by its size) on the
                // sample before it now, on the one after carried.
                let mut b = naive_saw(v.ph2) + v.carry;
                v.carry = 0.0;
                let next1 = v.ph1.wrapping_add(inc1);
                if next1 < inc1 {
                    // The first wraps before the next sample, `d` of a
                    // sample before it: the second restarts there.
                    let d = next1 as f32 / inc1.max(1) as f32;
                    let at = v.ph2.wrapping_add((inc2 as f32 * (1.0 - d)) as u32);
                    let jump = (naive_saw(at) + 1.0) * 0.5;
                    b -= jump * d * d;
                    v.carry = jump * (1.0 - d) * (1.0 - d);
                    v.ph2 = (inc2 as f32 * d) as u32;
                } else {
                    let next2 = v.ph2.wrapping_add(inc2);
                    if next2 < v.ph2 {
                        let d = next2 as f32 / inc2.max(1) as f32;
                        b -= d * d;
                        v.carry = (1.0 - d) * (1.0 - d);
                    }
                    v.ph2 = next2;
                }
                b
            } else {
                let b = osc(p.wave, v.ph2, inc2, width);
                v.ph2 = v.ph2.wrapping_add(inc2);
                b
            };
            v.ph1 = v.ph1.wrapping_add(inc1);
            let mut s = a * m1 + b * m2;
            if p.sub > 0.0 {
                s += dsp::pulse(v.sub, inc_sub, 1 << 31) * p.sub;
                v.sub = v.sub.wrapping_add(inc_sub);
            }
            if noise > 0.0 {
                s += dsp::white(&mut v.noise) * noise;
            }
            if p.drive > 0.0 {
                s = dsp::softclip(s * drive);
            }
            let t = svf.run(&mut v.filter, s * 0.5);
            let y = match p.filter {
                Filter::LowPass => t.lp,
                Filter::BandPass => t.bp,
                Filter::HighPass => t.hp,
                Filter::Notch => t.notch,
            };
            // Linear to half scale, then a knee: only resonant peaks bend.
            *o += dsp::soft_knee(y, 0.5) * 2.0;
        }
    }
}
