//! Frames read along a time map ([`Warping`]): at a changing speed
//! (varispeed), or stretched in time with the pitch kept. Any number of
//! interleaved channels at the stems' rate: a head's four stereo stems read
//! along its map ([`super::lanes`]).

use std::sync::Arc;

use crate::stems::RATE;
use crate::stretch::Stretcher;
use crate::transition::TempoMode;
use crate::warp::{TimeMap, VarResampler};

/// How the frames follow the time map.
enum Reader {
    /// Read at a changing speed: faster is higher (varispeed).
    Speed(VarResampler),
    /// Stretched in time, the pitch kept: one stretcher a channel, their
    /// speed steered so their position follows the map (they all go at
    /// the same speed, so their frames stay on the same instants).
    Stretch(Box<Stretch>),
}

/// The stretchers of [`Reader::Stretch`].
struct Stretch {
    channels: Vec<Stretcher>,
    /// What one has made ahead of the others.
    ahead: Vec<Vec<f64>>,
    /// The input frames still to drop before the map's first position
    /// (the stretchers start there).
    skip: usize,
    /// One channel's input, taken out of the interleaved frames.
    channel: Vec<f64>,
    /// How far (in input frames) their position was off the map at the
    /// last read.
    off: f64,
}

/// Seconds over which a stretcher's position is steered back onto the map.
const STEER: f64 = 0.25;

/// The most the steering changes the map's speed by.
const STEER_LIMIT: f64 = 0.05;

/// Frames read along a time map.
pub(super) struct Warping {
    map: Arc<TimeMap>,
    /// The map's time at the first frame read.
    a0: f64,
    /// The time the map reads at the input's first frame.
    b_origin: f64,
    channels: usize,
    reader: Reader,
    /// The input has all been pushed.
    ended: bool,
}

impl Warping {
    /// `channels` interleaved channels read along `map` from its time
    /// `a0`, their input starting at the map's read time `b_origin`, by
    /// `tempo`.
    pub(super) fn new(
        map: Arc<TimeMap>,
        a0: f64,
        b_origin: f64,
        tempo: TempoMode,
        channels: usize,
    ) -> Warping {
        let channels = channels.max(1);
        let reader = match tempo {
            TempoMode::Varispeed => Reader::Speed(VarResampler::new(channels)),
            TempoMode::Stretch => {
                let speed = map.speed(a0);
                let stretchers = (0..channels)
                    .map(|_| {
                        let mut s = Stretcher::new(RATE);
                        s.set_speed(speed);
                        s
                    })
                    .collect();
                let skip = ((map.at(a0) - b_origin) * f64::from(RATE)).round().max(0.0);
                Reader::Stretch(Box::new(Stretch {
                    channels: stretchers,
                    ahead: vec![Vec::new(); channels],
                    skip: skip as usize,
                    channel: Vec::new(),
                    off: 0.0,
                }))
            }
        };
        Warping {
            map,
            a0,
            b_origin,
            channels,
            reader,
            ended: false,
        }
    }

    /// Reads `frames` (interleaved, at the stems' rate) along the map and
    /// replaces them with what was read.
    pub(super) fn read(&mut self, frames: &mut Vec<f64>) {
        let Warping {
            map,
            a0,
            b_origin,
            channels,
            reader,
            ended,
        } = self;
        let (rate, ch) = (f64::from(RATE), *channels);
        let position = |k: u64| (map.at(*a0 + k as f64 / rate) - *b_origin) * rate;
        match reader {
            Reader::Speed(reader) => {
                reader.push(frames);
                frames.clear();
                let n = if *ended {
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
                reader.pull(frames, n, position);
            }
            Reader::Stretch(s) => {
                // The stretchers start at the map's first position.
                let dropped = s.skip.min(frames.len() / ch);
                frames.drain(..ch * dropped);
                s.skip -= dropped;
                let skipped = position(0).round().max(0.0) as usize;
                for (c, stretcher) in s.channels.iter_mut().enumerate() {
                    s.channel.clear();
                    s.channel.extend(frames.iter().skip(c).step_by(ch).copied());
                    stretcher.push(&s.channel);
                    if *ended {
                        stretcher.finish();
                    }
                }
                frames.clear();
                // Steer: the map's speed here, and the way back onto it.
                let k = s.channels[0].output_position();
                let off = position(k) - skipped as f64 - s.channels[0].input_position();
                s.off = off;
                let speed = map.speed(*a0 + k as f64 / rate)
                    * (1.0 + (off / (STEER * rate)).clamp(-STEER_LIMIT, STEER_LIMIT));
                for (stretcher, carry) in s.channels.iter_mut().zip(s.ahead.iter_mut()) {
                    stretcher.set_speed(speed);
                    stretcher.pull(carry, usize::MAX >> 1);
                }
                let n = s.ahead.iter().map(Vec::len).min().unwrap_or(0);
                frames.reserve(ch * n);
                for i in 0..n {
                    frames.extend(s.ahead.iter().map(|carry| carry[i]));
                }
                for carry in &mut s.ahead {
                    carry.drain(..n);
                }
            }
        }
    }

    /// How far (in input frames) what it reads is off the map: what the
    /// stretchers' steering has yet to take back (none at a changing speed,
    /// which reads on the map).
    pub(super) fn offset(&self) -> f64 {
        match &self.reader {
            Reader::Speed(_) => 0.0,
            Reader::Stretch(s) => s.off,
        }
    }

    /// The input has ended.
    pub(super) fn end(&mut self) {
        self.ended = true;
        if let Reader::Speed(reader) = &mut self.reader {
            reader.end();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// What reading four stereo stems (8 channels at the stems' rate) along
    /// a map costs, both ways, in the stem run's pieces of 4096 frames.
    /// A measurement, not a check (a loaded machine would fail one): run it
    /// in a release build with `--ignored`; `PERFORMANCE.md` has the numbers.
    #[test]
    #[ignore = "a measurement: run it in a release build"]
    fn eight_channels_along_a_map() {
        if cfg!(debug_assertions) {
            println!("speed is measured in release builds only");
            return;
        }
        let (channels, seconds) = (8usize, 8.0);
        let frames = (f64::from(RATE) * seconds) as usize;
        let input: Vec<f64> = (0..frames * channels)
            .map(|k| (k as f64 * 0.001).sin() * 4e6)
            .collect();
        // Up to 5 % faster and back, as a transition's map goes.
        let map = Arc::new(TimeMap::from_knots(
            vec![(1.0, 1.0), (2.0, 2.05), (4.0, 4.15)],
            vec![1.0, 1.05, 1.05],
            1.0,
        ));
        for tempo in [TempoMode::Varispeed, TempoMode::Stretch] {
            let mut warping = Warping::new(Arc::clone(&map), 0.0, 0.0, tempo, channels);
            let (mut made, mut piece) = (0usize, Vec::new());
            let start = Instant::now();
            for chunk in input.chunks(4096 * channels) {
                piece.clear();
                piece.extend_from_slice(chunk);
                warping.read(&mut piece);
                made += piece.len() / channels;
            }
            warping.end();
            piece.clear();
            warping.read(&mut piece);
            made += piece.len() / channels;
            let took = start.elapsed().as_secs_f64();
            let realtime = made as f64 / f64::from(RATE) / took;
            println!(
                "8 ch at 44100 along a map, {tempo:?}: {:>6.2} M frames/s, {realtime:>6.1}x real time ({:.1} % of a core)",
                made as f64 / took / 1e6,
                100.0 / realtime,
            );
            assert!(made > frames / 2, "{tempo:?}: {made} frames made");
        }
    }
}
