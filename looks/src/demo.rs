//! A made-up song for the looks to move with where there is no real one:
//! a steady beat at a tempo, bars of four, sections that rise and fall
//! (an intro, verses, choruses, a bridge, an outro), and each stem's
//! level made from its hits (kicks on one and three, snares on two and
//! four, hats on the eighths, a bass note each beat, a voice in lines).
//! The preview tool plays it without a song; the tests hold the guards
//! over its curves. The same tempo and length, the same song.

use jodel_rhythm::stems::Stem;
use jodel_rhythm::waveform::{Lane, Waveform, FPS};
use jodel_rhythm::{Analysis, Region};

use crate::drive::{Curves, Signals};

/// A made-up song: its analysis, waveform, sections and sung lines.
#[derive(Clone, Debug)]
pub struct Song {
    /// Its beats and bars.
    pub analysis: Analysis,
    /// Its lanes.
    pub waveform: Waveform,
    /// Where its sections start, seconds.
    pub sections: Vec<f64>,
    /// Its sung lines, start and end, seconds.
    pub lines: Vec<(f64, f64)>,
}

/// How loud each section of eight bars is, in turn (the last held).
const SECTIONS: [f32; 9] = [0.35, 0.6, 1.0, 0.6, 1.0, 0.4, 1.0, 1.0, 0.3];

/// A level (full scale 1) as the lanes' byte.
fn byte(v: f32) -> u8 {
    if v <= 1e-4 {
        return 0;
    }
    let db = 20.0 * v.log10();
    (255.0 * (db + 60.0) / 60.0).round().clamp(0.0, 255.0) as u8
}

/// A hit `since` seconds ago (none yet: nothing), falling by `decay`.
fn hit(since: f64, decay: f64) -> f32 {
    if since < 0.0 {
        0.0
    } else {
        (-since / decay).exp() as f32
    }
}

impl Song {
    /// `seconds` long at `bpm`.
    pub fn new(seconds: f64, bpm: f64) -> Song {
        let beat = 60.0 / bpm;
        let n_beats = (seconds / beat) as usize + 1;
        let beats: Vec<f64> = (0..n_beats).map(|i| i as f64 * beat).collect();
        let bar = 4.0 * beat;
        let section_len = 8.0 * bar;
        let level_at = |t: f64| SECTIONS[((t / section_len) as usize).min(SECTIONS.len() - 1)];
        let frames = (seconds * FPS) as usize;
        let mut lanes: [Lane; 5] = Default::default();
        for f in 0..frames {
            let t = f as f64 / FPS;
            let lvl = level_at(t);
            let in_beat = t.rem_euclid(beat);
            let b = (t / beat) as usize;
            let eighth = t.rem_euclid(beat / 2.0);
            let busy = if lvl > 0.5 { 1.0 } else { 0.25 };
            let kick = if b % 2 == 0 { hit(in_beat, 0.08) } else { 0.0 };
            let snare = if b % 2 == 1 {
                0.8 * hit(in_beat, 0.1)
            } else {
                0.0
            };
            let hats = 0.3 * hit(eighth, 0.03);
            let drums = busy * lvl * (kick + snare + hats).min(1.0);
            let bass = lvl * (0.35 + 0.65 * hit(in_beat, 0.4));
            let bar_n = (t / bar) as usize;
            let sung = lvl > 0.5 && bar_n % 4 < 2;
            let syllable = 0.5 + 0.5 * (t * std::f64::consts::TAU * 4.0 / beat).sin().abs() as f32;
            let vocals = if sung {
                0.8 * syllable * lvl.min(0.9)
            } else {
                0.0
            };
            let other = lvl * (0.45 + 0.1 * (t * 0.7).sin() as f32);
            let mix = (drums * drums + bass * bass + vocals * vocals + other * other).sqrt() * 0.6;
            let levels = [bass, drums, other, vocals];
            for s in Stem::ALL {
                let v = levels[s as usize] * 0.5;
                let peak = if s == Stem::Drums {
                    (v * 2.5).min(1.0)
                } else {
                    v * 1.4
                };
                lanes[s as usize].rms.push(byte(v));
                lanes[s as usize].peak.push(byte(peak));
            }
            lanes[4].rms.push(byte(mix));
            lanes[4].peak.push(byte(mix * 1.6));
        }
        let [l0, l1, l2, l3, mix] = lanes;
        let mut lines = Vec::new();
        let mut k = 0;
        while k as f64 * bar * 4.0 < seconds {
            let start = k as f64 * 4.0 * bar;
            if level_at(start + 0.01) > 0.5 {
                lines.push((start + 0.5 * beat, start + 2.0 * bar - 0.5 * beat));
            }
            k += 1;
        }
        let sections: Vec<f64> = (1..)
            .map(|i| i as f64 * section_len)
            .take_while(|&s| s < seconds)
            .collect();
        Song {
            analysis: Analysis {
                duration: seconds,
                downbeats: (0..n_beats).step_by(4).collect(),
                local_bpm: vec![bpm; n_beats],
                regions: vec![Region {
                    start: 0.0,
                    end: seconds,
                    confidence: 0.9,
                }],
                beats,
                confidence: 0.9,
                ..Analysis::default()
            },
            waveform: Waveform {
                mix,
                stems: Some([l0, l1, l2, l3]),
                melody: None,
            },
            sections,
            lines,
        }
    }

    /// Its curves.
    pub fn curves(&self) -> Curves {
        Curves::new(
            &self.analysis,
            Some(&self.waveform),
            &self.sections,
            &self.lines,
        )
    }
}

/// The drivers of a made-up song at 120 a minute, `t` seconds in (the
/// song two minutes long, round again after).
pub fn signals_at(t: f64) -> Signals {
    use std::sync::OnceLock;
    static CURVES: OnceLock<Curves> = OnceLock::new();
    let c = CURVES.get_or_init(|| Song::new(120.0, 120.0).curves());
    c.at(t.rem_euclid(120.0))
}
