//! LOFI: a chip voice, for coins, blips, chip arpeggios and chip vowels.
//!
//! A band-limited pulse of the old chips' four duties, their 32-step
//! four-bit triangle, their 15-bit noise register, or a wave table of 32
//! four-bit samples (our own, made at compile time from harmonic sums:
//! three of them vowels). The pitch can step through a chip arpeggio
//! (sweeps and vibratos are the patch's pitch glide and LFO). [`StepEnv`]
//! is the chips' own volume envelope, sixteen
//! steps on a 240 Hz frame clock, in place of the ADSR's curve (the ADSR
//! then only gates the voice). After it the output can be held at a lower
//! rate, rounded to fewer bits, and dulled by a one-pole tone control.
//!
//! The host's brightness opens the tone control ([`BRIGHT`]); its shape
//! widens or narrows the pulse, or morphs through the wave tables
//! ([`SHAPE`]).
//!
//! Sources: the duties, the triangle's steps, the noise register, the
//! envelope's divider and the 240 Hz frame clock after the NES APU as the
//! nesdev wiki documents it ("APU Pulse", "APU Triangle", "APU Noise",
//! "APU Envelope", "APU Frame Counter"); the wave table's form after the
//! Game Boy's wave channel (Pan Docs, "Audio Registers": 32 four-bit
//! samples); the arpeggio after the tracker effect 0xy (cycling the note,
//! +x and +y semitones each tick; ProTracker's MOD format); the vowels'
//! formants from Peterson and Barney, "Control Methods Used in a Study of
//! the Vowels" (JASA, 1952); the pulse by polyBLEP (Välimäki et al.).

use crate::dsp::{self, Rate};
use crate::{Engine, NoteOn, VoiceMod, BLOCK};

/// The wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// A pulse of the duty.
    Pulse,
    /// The 32-step four-bit triangle.
    Triangle,
    /// The noise register: its long loop clocked 64 times a period of the
    /// note, or its short one 93 times (a tone on the note).
    Noise {
        /// The short loop.
        short: bool,
    },
    /// A wave table.
    Table(Table),
}

/// A pulse's duty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duty {
    /// 12.5 %.
    Eighth,
    /// 25 %.
    Quarter,
    /// 50 %.
    Half,
    /// 75 %.
    ThreeQuarters,
}

/// A wave table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// A sine.
    Sine,
    /// A saw (harmonics at 1/h).
    Saw,
    /// Odd harmonics, hollow.
    Hollow,
    /// Harmonics 1, 2, 3 and 4, an organ.
    Organ,
    /// All harmonics alike, a buzz.
    Buzz,
    /// The vowel "ah".
    VoxA,
    /// The vowel "aw".
    VoxO,
    /// The vowel "ee".
    VoxE,
}

/// The chips' volume envelope: a step down every period + 1 frames (of
/// 240 a second) from 15.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepEnv {
    /// None: the ADSR's curve.
    Off,
    /// Down to 0 once (period 0 to 15), the voice ending there.
    Decay(u8),
    /// Down to 0 and from 15 again.
    Loop(u8),
}

/// The chip arpeggio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arp {
    /// None.
    Off,
    /// The note and its octave.
    Octave,
    /// A major triad.
    Major,
    /// A minor triad.
    Minor,
}

/// LOFI's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The wave.
    pub wave: Wave,
    /// The pulse's duty.
    pub duty: Duty,
    /// The volume envelope.
    pub step_env: StepEnv,
    /// The arpeggio.
    pub arp: Arp,
    /// Its steps a second.
    pub arp_rate: f32,
    /// The rate the output is held at, Hz (0: the full rate).
    pub crush: f32,
    /// The bits it is rounded to (0: not rounded).
    pub bits: u8,
    /// The tone control's corner, Hz (0: open).
    pub tone: f32,
}

impl Default for Params {
    fn default() -> Params {
        CHIP
    }
}

/// [`Params::default`]: a plain square.
const CHIP: Params = Params {
    wave: Wave::Pulse,
    duty: Duty::Half,
    step_env: StepEnv::Off,
    arp: Arp::Off,
    arp_rate: 50.0,
    crush: 0.0,
    bits: 0,
    tone: 0.0,
};

/// A quarter pulse, bright: the coin's two notes.
pub const COIN: Params = Params {
    duty: Duty::Quarter,
    ..CHIP
};

/// A thin pulse stepping down quickly: a menu blip.
pub const BLIP: Params = Params {
    duty: Duty::Eighth,
    step_env: StepEnv::Decay(1),
    ..CHIP
};

/// A square through a major triad at 50 steps a second.
pub const ARP: Params = Params {
    arp: Arp::Major,
    ..CHIP
};

/// The vowel "ah", a little dulled (a patch's LFO gives it a vibrato).
pub const VOX_A: Params = Params {
    wave: Wave::Table(Table::VoxA),
    tone: 3_500.0,
    ..CHIP
};

/// The vowel "aw" the same.
pub const VOX_O: Params = Params {
    wave: Wave::Table(Table::VoxO),
    tone: 3_000.0,
    ..CHIP
};

/// The triangle: a soft chip bass.
pub const BASS: Params = Params {
    wave: Wave::Triangle,
    ..CHIP
};

/// The long noise loop, stepping down: a chip snare or hit.
pub const HIT: Params = Params {
    wave: Wave::Noise { short: false },
    step_env: StepEnv::Decay(0),
    ..CHIP
};

/// A thin pulse, crushed: a chip laser (with a falling pitch glide).
pub const LASER: Params = Params {
    duty: Duty::Eighth,
    crush: 11_025.0,
    bits: 4,
    ..CHIP
};

/// The presets, by name.
pub const PRESETS: [(&str, Params); 8] = [
    ("COIN", COIN),
    ("BLIP", BLIP),
    ("ARP", ARP),
    ("VOX A", VOX_A),
    ("VOX O", VOX_O),
    ("BASS", BASS),
    ("HIT", HIT),
    ("LASER", LASER),
];

/// What the host's shape moves: the pulse's width by up to 0.4 either way,
/// or the wave table, by up to four tables either way, crossfading between
/// neighbours (the triangle and the noise take no shape).
pub const SHAPE: &str = "pulse width / wave table";

/// What the host's brightness opens: the tone control's corner, an octave
/// an octave.
pub const BRIGHT: &str = "tone";

/// The frames of the volume envelope a second.
const FRAMES: f32 = 240.0;

/// A table's 32 samples: the harmonics 1 to 15 at `amps` (sine phase),
/// scaled to the full swing, rounded to four bits (0 to 15), as -1 to 1
/// less their mean (no DC).
const fn table(amps: [f64; 15]) -> [f32; 32] {
    let mut x = [0.0f64; 32];
    let mut peak = 0.0f64;
    let mut n = 0;
    while n < 32 {
        let mut h = 0;
        while h < 15 {
            x[n] += amps[h] * dsp::sin_turn((h + 1) * n, 32);
            h += 1;
        }
        let a = if x[n] < 0.0 { -x[n] } else { x[n] };
        if a > peak {
            peak = a;
        }
        n += 1;
    }
    let mut q = [0.0f64; 32];
    let mut mean = 0.0;
    let mut n = 0;
    while n < 32 {
        let level = ((x[n] / peak + 1.0) * 7.5 + 0.5) as i32;
        q[n] = (level as f64 - 7.5) / 7.5;
        mean += q[n] / 32.0;
        n += 1;
    }
    let mut t = [0.0f32; 32];
    let mut n = 0;
    while n < 32 {
        t[n] = (q[n] - mean) as f32;
        n += 1;
    }
    t
}

/// Harmonics at 1 / h: every one (`step` 1) or every other (2).
const fn falling(step: usize) -> [f64; 15] {
    let mut a = [0.0; 15];
    let mut h = 0;
    while h < 15 {
        if h % step == 0 {
            a[h] = 1.0 / (h + 1) as f64;
        }
        h += 1;
    }
    a
}

/// The harmonics of a voice on 200 Hz under three formants (centre and
/// bandwidth, Hz), falling 6 dB an octave as a glottal source does.
const fn vowel(f: [(f64, f64); 3]) -> [f64; 15] {
    let gains = [1.0, 0.6, 0.3];
    let mut a = [0.0; 15];
    let mut h = 0;
    while h < 15 {
        let freq = 200.0 * (h + 1) as f64;
        let mut k = 0;
        while k < 3 {
            let d = (freq - f[k].0) / (0.5 * f[k].1);
            a[h] += gains[k] / (1.0 + d * d);
            k += 1;
        }
        a[h] /= (h + 1) as f64;
        h += 1;
    }
    a
}

/// The tables, in [`Table`]'s order.
static TABLES: [[f32; 32]; 8] = [
    table([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]),
    table(falling(1)),
    table(falling(2)),
    table([
        1.0, 0.8, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]),
    table([1.0; 15]),
    // Peterson and Barney's men: "hod", "hawed", "heed".
    table(vowel([(730.0, 200.0), (1_090.0, 250.0), (2_440.0, 300.0)])),
    table(vowel([(570.0, 200.0), (840.0, 250.0), (2_410.0, 300.0)])),
    table(vowel([(270.0, 200.0), (2_290.0, 250.0), (3_010.0, 300.0)])),
];

/// One LOFI voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    ph: u32,
    /// The noise register and its clock's phase.
    register: u32,
    clock: u32,
    crush: dsp::Crush,
    /// The tone control's state.
    lp: f32,
    /// The arpeggio's place, in steps.
    arp: f32,
    /// The envelope's volume (15 to 0), its divider, the frame clock's
    /// phase, and whether it ends the voice at 0.
    volume: u8,
    divider: u8,
    frame: f32,
    once: bool,
}

/// LOFI at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Lofi {
    rate: Rate,
    /// A block, seconds.
    block: f32,
}

fn period(env: StepEnv) -> u8 {
    match env {
        StepEnv::Off => 0,
        StepEnv::Decay(p) | StepEnv::Loop(p) => p.min(15),
    }
}

/// Table `at` (fractional, between neighbours), sample `k`.
#[inline]
fn table_at(at: f32, k: usize) -> f32 {
    let i = (at as usize).min(TABLES.len() - 1);
    let j = (i + 1).min(TABLES.len() - 1);
    let t = at - i as f32;
    TABLES[i][k] + (TABLES[j][k] - TABLES[i][k]) * t
}

impl Engine for Lofi {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Lofi {
        Lofi {
            rate,
            block: BLOCK as f32 / rate.hz(),
        }
    }

    /// Each note starts the envelope, the arpeggio and the register afresh
    /// (the register as at power-on). A retrigger keeps the phase, the
    /// crush's held sample and the tone control.
    fn note_on(&self, p: &Params, v: &mut Voice, _n: &NoteOn) {
        v.register = 1;
        v.arp = 0.0;
        v.volume = 15;
        v.divider = period(p.step_env);
        v.frame = 0.0;
        v.once = matches!(p.step_env, StepEnv::Decay(_));
    }

    /// Each frame the divider counts down; at 0 it starts again and the
    /// volume steps down (a looping one from 0 back to 15). The ADSR only
    /// gates it: let go, the steps run on until its release is over.
    fn amp(&self, p: &Params, v: &mut Voice, adsr: f32) -> f32 {
        if p.step_env == StepEnv::Off {
            return adsr;
        }
        v.frame += FRAMES * self.block;
        while v.frame >= 1.0 {
            v.frame -= 1.0;
            if v.divider > 0 {
                v.divider -= 1;
            } else {
                v.divider = period(p.step_env);
                if v.volume > 0 {
                    v.volume -= 1;
                } else if matches!(p.step_env, StepEnv::Loop(_)) {
                    v.volume = 15;
                }
            }
        }
        f32::from(v.volume) / 15.0
    }

    fn done(&self, v: &Voice) -> bool {
        v.once && v.volume == 0
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let steps = match p.arp {
            Arp::Off => [0.0, 0.0, 0.0],
            Arp::Octave => [0.0, 12.0, 0.0],
            Arp::Major => [0.0, 4.0, 7.0],
            Arp::Minor => [0.0, 3.0, 7.0],
        };
        let arp = steps[(v.arp as usize) % 3];
        v.arp = (v.arp + p.arp_rate.max(0.0) * self.block) % 3.0;
        let pitch = m.pitch + arp;
        let inc = self.rate.inc(pitch);
        let duty = match p.duty {
            Duty::Eighth => 0.125,
            Duty::Quarter => 0.25,
            Duty::Half => 0.5,
            Duty::ThreeQuarters => 0.75,
        };
        let width = ((duty + 0.4 * m.shape).clamp(0.05, 0.95) * 4_294_967_296.0) as u32;
        let table = match p.wave {
            Wave::Table(t) => {
                (t as usize as f32 + 4.0 * m.shape).clamp(0.0, (TABLES.len() - 1) as f32)
            }
            _ => 0.0,
        };
        let clock = match p.wave {
            Wave::Noise { short } => self
                .rate
                .clock_inc(dsp::hz(pitch) * if short { 93.0 } else { 64.0 }),
            _ => 0,
        };
        let crush = dsp::CrushSetting::new(p.crush, p.bits, self.rate);
        let pole = if p.tone > 0.0 {
            self.rate.pole(p.tone * m.bright.exp2())
        } else {
            1.0
        };
        for o in out.iter_mut() {
            let s = match p.wave {
                Wave::Pulse => dsp::pulse(v.ph, inc, width),
                Wave::Triangle => {
                    let k = (v.ph >> 27) as i32;
                    let level = if k < 16 { 15 - k } else { k - 16 };
                    (level as f32 - 7.5) / 7.5
                }
                Wave::Noise { short } => {
                    dsp::clock_noise(&mut v.register, &mut v.clock, clock, short);
                    if v.register & 1 != 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Wave::Table(_) => table_at(table, (v.ph >> 27) as usize),
            };
            v.ph = v.ph.wrapping_add(inc);
            let s = v.crush.run(s, &crush);
            v.lp += (s - v.lp) * pole;
            *o += v.lp;
        }
    }
}
