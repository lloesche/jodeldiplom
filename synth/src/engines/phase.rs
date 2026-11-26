//! PHASE: phase distortion, for brass, bells, organs and spooky drones.
//!
//! A cosine is read through a bent phase: unbent it is a sine, and the
//! deeper the bend (the DCW) the brighter the wave grows towards a saw, a
//! square or a pulse. The resonant waves are a cosine at a multiple of the
//! pitch that rises with the DCW, restarted each cycle under a window (a
//! saw, a triangle or a trapezoid), so it sweeps like a resonant filter. A
//! second wave can take every other cycle. A second line, detuned, can be
//! mixed in or ring-modulated with the first (a bell at an inharmonic
//! interval); a sine an octave down can be added. The bends' slopes are
//! worked out once a block, so a sample costs a multiply.
//!
//! The host's brightness deepens the DCW ([`BRIGHT`]): a patch's filter
//! envelope is the CZ's DCW envelope. Its shape moves the DCW too
//! ([`SHAPE`]).
//!
//! Sources: phase distortion as Casio described it for the CZ series (the
//! CZ-101 owner's manual: the eight waveforms, DCW, line select 1+1', ring
//! modulation; and US patent 4,658,691, "Electronic musical instrument",
//! 1987, on reading a cosine through a modified phase); the sine table and
//! u32 phases as in [`crate::dsp`].

use crate::dsp::{self, Rate};
use crate::params::{field, pick, when, Param};
use crate::{Engine, NoteOn, VoiceMod};

/// A wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// Towards a saw.
    Saw,
    /// Towards a square.
    Square,
    /// Towards a pulse.
    Pulse,
    /// A saw's bend squeezed into the start of the cycle, then still.
    SawPulse,
    /// A resonance under a saw window.
    ResoSaw,
    /// Under a triangle window.
    ResoTriangle,
    /// Under a trapezoid window.
    ResoTrapezoid,
}

/// The second line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// None.
    Single,
    /// The second, detuned, mixed with the first.
    Mix,
    /// The second, detuned, ring-modulating the first.
    Ring,
}

/// PHASE's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The wave.
    pub wave: Wave,
    /// The wave every other cycle (none: the same).
    pub wave2: Option<Wave>,
    /// The bend's depth, 0 (a sine) to 1.
    pub dcw: f32,
    /// The second line.
    pub line: Line,
    /// Its detune from the first, cents.
    pub detune: f32,
    /// The sine an octave down, 0 to 1.
    pub sub: f32,
}

impl Default for Params {
    fn default() -> Params {
        SINE
    }
}

/// The waves and the lines in their order, and their names; the second
/// wave the same with "the same" first.
const WAVES: [Wave; 7] = [
    Wave::Saw,
    Wave::Square,
    Wave::Pulse,
    Wave::SawPulse,
    Wave::ResoSaw,
    Wave::ResoTriangle,
    Wave::ResoTrapezoid,
];
const WAVE_NAMES: [&str; 7] = [
    "saw",
    "square",
    "pulse",
    "saw pulse",
    "reso saw",
    "reso triangle",
    "reso trapezoid",
];
const SECOND: [Option<Wave>; 8] = [
    None,
    Some(Wave::Saw),
    Some(Wave::Square),
    Some(Wave::Pulse),
    Some(Wave::SawPulse),
    Some(Wave::ResoSaw),
    Some(Wave::ResoTriangle),
    Some(Wave::ResoTrapezoid),
];
const SECOND_NAMES: [&str; 8] = [
    "the same",
    "saw",
    "square",
    "pulse",
    "saw pulse",
    "reso saw",
    "reso triangle",
    "reso trapezoid",
];
const LINES: [Line; 3] = [Line::Single, Line::Mix, Line::Ring];
const LINE_NAMES: [&str; 3] = ["single", "mix", "ring"];

/// PHASE's parameters as data, with their ranges.
pub const PARAMS: &[Param<Params>] = &[
    pick!("phase", "wave", wave, WAVES, WAVE_NAMES),
    pick!("phase", "every other", wave2, SECOND, SECOND_NAMES),
    field!("phase", "bend", dcw, 0.0, 1.0, "", Linear),
    field!("phase", "sub", sub, 0.0, 1.0, "", Linear),
    pick!("line 2", "line", line, LINES, LINE_NAMES),
    when(
        field!("line 2", "detune", detune, -1_200.0, 2_400.0, "ct", Linear),
        |p| p.line != Line::Single,
    ),
];

/// [`Params::default`]: an unbent sine.
const SINE: Params = Params {
    wave: Wave::Saw,
    wave2: None,
    dcw: 0.0,
    line: Line::Single,
    detune: 0.0,
    sub: 0.0,
};

/// A saw opening with the envelope (a filter envelope of about 2.5
/// octaves), two lines a little apart.
pub const BRASS: Params = Params {
    dcw: 0.15,
    line: Line::Mix,
    detune: 7.0,
    ..SINE
};

/// A resonant triangle ring-modulated at an inharmonic 2.76 (1758 cents):
/// a bell.
pub const BELL: Params = Params {
    wave: Wave::ResoTriangle,
    dcw: 0.3,
    line: Line::Ring,
    detune: 1_758.0,
    ..SINE
};

/// A hollow square over a sub, slightly chorused: an organ.
pub const ORGAN: Params = Params {
    wave: Wave::Square,
    dcw: 0.35,
    line: Line::Mix,
    detune: 5.0,
    sub: 0.4,
    ..SINE
};

/// A saw-pulse alternating with a resonant trapezoid: wiry.
pub const WIRE: Params = Params {
    wave: Wave::SawPulse,
    wave2: Some(Wave::ResoTrapezoid),
    dcw: 0.55,
    ..SINE
};

/// A pulse ring-modulated an octave and a few cents up: a spooky,
/// beating drone.
pub const DRONE: Params = Params {
    wave: Wave::Pulse,
    dcw: 0.3,
    line: Line::Ring,
    detune: 1_211.0,
    sub: 0.5,
    ..SINE
};

/// What the host's shape moves: the DCW, by up to half its range.
pub const SHAPE: &str = "DCW";

/// What the host's brightness opens: the DCW, a quarter of its range an
/// octave.
pub const BRIGHT: &str = "DCW";

/// The presets, by name.
pub const PRESETS: [(&str, Params); 5] = [
    ("BRASS", BRASS),
    ("BELL", BELL),
    ("ORGAN", ORGAN),
    ("WIRE", WIRE),
    ("DRONE", DRONE),
];

/// One PHASE voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    /// The lines' phases and the sub's.
    ph: [u32; 3],
    /// Each line on its second wave this cycle.
    second: [bool; 2],
}

/// PHASE at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Phase {
    rate: Rate,
}

/// The deepest bend (a slope never infinite).
const DEEPEST: f32 = 0.98;

/// The highest resonance, times the pitch, at full DCW.
const RESONANCE: f32 = 8.0;

/// One wave's bend for a block.
#[derive(Clone, Copy, Debug)]
struct Bend {
    wave: Wave,
    /// Where the bend breaks (a saw-pulse: where it stops, and where its
    /// saw breaks), and the slopes before and after.
    knee: f32,
    inner: f32,
    rise: f32,
    fall: f32,
    /// The resonance, times the pitch.
    ratio: f32,
}

impl Bend {
    /// `wave` bent `d` deep (0 to [`DEEPEST`]).
    fn new(wave: Wave, d: f32) -> Bend {
        let knee = match wave {
            Wave::Saw | Wave::Square => 0.5 * (1.0 - d),
            Wave::Pulse => 1.0 - d,
            Wave::SawPulse => 1.0 - 0.5 * d,
            _ => 1.0,
        };
        // A saw-pulse bends as a saw inside its squeezed part.
        let (inner, half) = match wave {
            Wave::Square => (knee, 0.5),
            Wave::SawPulse => (0.5 * (1.0 - d), 1.0),
            _ => (knee, 1.0),
        };
        Bend {
            wave,
            knee,
            inner,
            rise: 0.5 / inner,
            fall: 0.5 / (half - inner).max(1e-3),
            ratio: 1.0 + (RESONANCE - 1.0) * d,
        }
    }

    /// The wave at phase `ph` (0 to 1), -1 to 1.
    #[inline]
    fn at(&self, ph: f32) -> f32 {
        let pd = match self.wave {
            // Half a cosine cycle quickly, the other half slowly.
            Wave::Saw => {
                if ph < self.knee {
                    ph * self.rise
                } else {
                    0.5 + (ph - self.knee) * self.fall
                }
            }
            // Each half of the cycle: half a cosine cycle, then still.
            Wave::Square => {
                let (base, local) = if ph < 0.5 { (0.0, ph) } else { (0.5, ph - 0.5) };
                base + (local * self.rise).min(0.5)
            }
            Wave::Pulse => (ph * self.rise * 2.0).min(1.0),
            Wave::SawPulse => {
                if ph >= self.knee {
                    1.0
                } else {
                    let u = ph / self.knee;
                    if u < self.inner {
                        u * self.rise
                    } else {
                        0.5 + (u - self.inner) * self.fall
                    }
                }
            }
            Wave::ResoSaw | Wave::ResoTriangle | Wave::ResoTrapezoid => {
                let window = match self.wave {
                    Wave::ResoSaw => 1.0 - ph,
                    Wave::ResoTriangle => 2.0 * ph.min(1.0 - ph),
                    _ => (4.0 * ph.min(1.0 - ph)).min(1.0),
                };
                let core = cosine(ph * self.ratio);
                return core * window;
            }
        };
        cosine(pd)
    }
}

/// -cos(2 pi `ph`).
#[inline]
fn cosine(ph: f32) -> f32 {
    -dsp::sine(dsp::turns(ph).wrapping_add(0x4000_0000))
}

impl Engine for Phase {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Phase {
        Phase { rate }
    }

    /// Both lines start on their first wave. A retrigger keeps the phases.
    fn note_on(&self, _p: &Params, v: &mut Voice, _n: &NoteOn) {
        v.second = [false; 2];
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let depth = p.dcw + m.bright * 0.25 + m.shape * 0.5;
        let d = depth.clamp(0.0, 1.0) * DEEPEST;
        let b1 = Bend::new(p.wave, d);
        let b2 = Bend::new(p.wave2.unwrap_or(p.wave), d);
        let inc = m.phase_inc;
        let inc2 = self.rate.inc(m.pitch + p.detune / 100.0);
        let sub = p.sub * 0.5;
        let [mut ph0, mut ph1, mut ph2] = v.ph;
        let [mut second0, mut second1] = v.second;
        for o in out.iter_mut() {
            let mut s = (if second0 { &b2 } else { &b1 }).at((ph0 >> 8) as f32 / 16_777_216.0);
            let old = ph0;
            ph0 = ph0.wrapping_add(inc);
            second0 ^= p.wave2.is_some() && ph0 < old;
            if p.line != Line::Single {
                let s2 = (if second1 { &b2 } else { &b1 }).at((ph1 >> 8) as f32 / 16_777_216.0);
                let old = ph1;
                ph1 = ph1.wrapping_add(inc2);
                second1 ^= p.wave2.is_some() && ph1 < old;
                s = if p.line == Line::Ring {
                    s * s2
                } else {
                    (s + s2) * 0.5
                };
            }
            if sub > 0.0 {
                s = s * (1.0 - sub) + dsp::sine(ph2) * sub;
                ph2 = ph2.wrapping_add(inc >> 1);
            }
            *o += s;
        }
        v.ph = [ph0, ph1, ph2];
        v.second = [second0, second1];
    }
}
