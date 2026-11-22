//! The next track's stems read along a transition's time map
//! ([`Warping`]): at a changing speed (varispeed), or stretched in time with
//! the pitch kept.

use std::sync::Arc;

use crate::stems::RATE;
use crate::stretch::Stretcher;
use crate::transition::TempoMode;
use crate::warp::{TimeMap, VarResampler};

/// How B's stems follow the time map.
enum Reader {
    /// Read at a changing speed: faster is higher (varispeed).
    Speed(VarResampler),
    /// Stretched in time, the pitch kept: one stretcher a channel, their
    /// speed steered so their position follows the map, what one has made
    /// ahead of the other, and the input frames still to drop before the
    /// map's first position (the stretchers start there).
    Stretch(Box<[Stretcher; 2]>, [Vec<f64>; 2], usize),
}

/// Seconds over which a stretcher's position is steered back onto the map.
const STEER: f64 = 0.25;

/// The most the steering changes the map's speed by.
const STEER_LIMIT: f64 = 0.05;

/// B's stems read along the time map.
pub(super) struct Warping {
    map: Arc<TimeMap>,
    /// A's time at B's first output frame.
    a0: f64,
    /// B's time at its stems' first frame.
    b_origin: f64,
    reader: Reader,
    /// The stems have all been read.
    ended: bool,
}

impl Warping {
    /// B read along `map` from A's time `a0`, its stems starting at B's
    /// time `b_origin`, by `tempo`.
    pub(super) fn new(map: Arc<TimeMap>, a0: f64, b_origin: f64, tempo: TempoMode) -> Warping {
        let reader = match tempo {
            TempoMode::Varispeed => Reader::Speed(VarResampler::new(2)),
            TempoMode::Stretch => {
                let speed = map.speed(a0);
                let mut pair = [Stretcher::new(RATE), Stretcher::new(RATE)];
                for s in &mut pair {
                    s.set_speed(speed);
                }
                let skip = ((map.at(a0) - b_origin) * f64::from(RATE)).round().max(0.0);
                Reader::Stretch(Box::new(pair), [Vec::new(), Vec::new()], skip as usize)
            }
        };
        Warping {
            map,
            a0,
            b_origin,
            reader,
            ended: false,
        }
    }

    /// Reads `stems` (interleaved stereo at the stems' rate) along the map
    /// and replaces them with what was read.
    pub(super) fn read(&mut self, stems: &mut Vec<f64>) {
        let Warping {
            map,
            a0,
            b_origin,
            reader,
            ended,
        } = self;
        let rate = f64::from(RATE);
        let position = |k: u64| (map.at(*a0 + k as f64 / rate) - *b_origin) * rate;
        match reader {
            Reader::Speed(reader) => {
                reader.push(stems);
                stems.clear();
                let frames = if *ended {
                    // Up to the end of the input, then nothing more.
                    let taken = reader.taken() as f64;
                    let mut n = 0usize;
                    while position(reader.made() + n as u64) < taken && n < 1 << 20 {
                        n += 1;
                    }
                    n
                } else {
                    usize::MAX
                };
                reader.pull(stems, frames, position);
            }
            Reader::Stretch(pair, ahead, skip) => {
                // The stretchers start at the map's first position.
                let dropped = (*skip).min(stems.len() / 2);
                stems.drain(..2 * dropped);
                *skip -= dropped;
                let skipped = ((position(0)).round().max(0.0)) as usize;
                for (c, s) in pair.iter_mut().enumerate() {
                    let channel: Vec<f64> = stems.iter().skip(c).step_by(2).copied().collect();
                    s.push(&channel);
                    if *ended {
                        s.finish();
                    }
                }
                stems.clear();
                // Steer: the map's speed here, and the way back onto it.
                let k = pair[0].output_position();
                let off = position(k) - skipped as f64 - pair[0].input_position();
                let speed = map.speed(*a0 + k as f64 / rate)
                    * (1.0 + (off / (STEER * rate)).clamp(-STEER_LIMIT, STEER_LIMIT));
                for (s, carry) in pair.iter_mut().zip(ahead.iter_mut()) {
                    s.set_speed(speed);
                    s.pull(carry, usize::MAX >> 1);
                }
                let n = ahead[0].len().min(ahead[1].len());
                stems.reserve(2 * n);
                for i in 0..n {
                    stems.extend([ahead[0][i], ahead[1][i]]);
                }
                for carry in ahead.iter_mut() {
                    carry.drain(..n);
                }
            }
        }
    }

    /// The map it reads along.
    pub(super) fn map(&self) -> &Arc<TimeMap> {
        &self.map
    }

    /// The stems have ended.
    pub(super) fn end(&mut self) {
        self.ended = true;
        if let Reader::Speed(reader) = &mut self.reader {
            reader.end();
        }
    }
}
