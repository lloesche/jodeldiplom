//! Frames read along a time map ([`Warping`]): at a changing speed
//! (varispeed), or stretched in time with the pitch kept; and a key apart
//! ([`Shift`]): every frequency times a ratio, the time the map's. Any
//! number of interleaved channels at the stems' rate (a head's four stereo
//! stems read along its map, [`super::lanes`]), in groups each read its
//! own way and interleaved back: all of them alike without a key; with
//! one, the drums' read along the map alone and the rest shifted.

use std::sync::Arc;

use crate::stems::RATE;
use crate::stretch::Stretcher;
use crate::transition::TempoMode;
use crate::warp::{TimeMap, VarResampler};

use super::shift::{frames_before, Shift};

/// How the frames follow the time map.
enum Reader {
    /// Read at a changing speed: faster is higher (varispeed).
    Speed(VarResampler),
    /// Stretched in time, the pitch kept: one stretcher a channel, their
    /// speed steered so their position follows the map (they all go at
    /// the same speed, so their frames stay on the same instants).
    Stretch(Box<Stretch>),
    /// A key apart: stretched by the ratio and read faster through the
    /// map's reader ([`Shift`]), the stretchers at the map's speed over the
    /// ratio when it is stretched to.
    Shift(Box<Shift>, TempoMode),
}

/// The stretchers of [`Reader::Stretch`].
struct Stretch {
    channels: Vec<Stretcher>,
    /// What one has made ahead of the others.
    ahead: Vec<Vec<f64>>,
    /// The input frames still to drop before the map's first position
    /// (the stretchers start there).
    skip: usize,
    /// The input frame the stretchers' first is (before the input's
    /// first where the map starts before it: silence pushed first).
    origin: i64,
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

/// Channels read one way, and what they made that the other groups have
/// not yet.
struct Group {
    /// Which of the interleaved channels, in order.
    channels: Vec<usize>,
    reader: Reader,
    /// Scratch: its channels' input; then what was read.
    frames: Vec<f64>,
    made: Vec<f64>,
}

/// What a [`Warping`] reads: `channels` interleaved channels along `map`
/// from its time `a0`, their input starting at the map's read time
/// `b_origin`, by `tempo`; `key` semitones apart (none at 0) but for the
/// channels `kept`, read along the map alone (the drums').
pub(super) struct Reading {
    pub(super) map: Arc<TimeMap>,
    pub(super) a0: f64,
    pub(super) b_origin: f64,
    pub(super) tempo: TempoMode,
    pub(super) channels: usize,
    pub(super) key: i8,
    pub(super) kept: Vec<usize>,
}

/// Frames read along a time map.
pub(super) struct Warping {
    map: Arc<TimeMap>,
    /// The map's time at the first frame read.
    a0: f64,
    /// The time the map reads at the input's first frame.
    b_origin: f64,
    channels: usize,
    /// The channels' groups (one: all of them, read alike).
    groups: Vec<Group>,
    /// The input has all been pushed.
    ended: bool,
}

impl Reader {
    /// A reader of `channels` along `map` from its time `a0`, their input
    /// starting at the map's read time `b_origin`, by `tempo`.
    fn new(map: &TimeMap, a0: f64, b_origin: f64, tempo: TempoMode, channels: usize) -> Reader {
        match tempo {
            TempoMode::Varispeed => Reader::Speed(VarResampler::new(channels)),
            TempoMode::Stretch => {
                let speed = map.speed(a0);
                // The map's first position, in input frames: the stretchers
                // start there, after as much silence as it is before the
                // input's first frame (as the varispeed reader reads it).
                let origin = ((map.at(a0) - b_origin) * f64::from(RATE)).round() as i64;
                let silence = vec![0.0; (-origin).max(0) as usize];
                let stretchers = (0..channels)
                    .map(|_| {
                        let mut s = Stretcher::new(RATE);
                        s.set_speed(speed);
                        s.push(&silence);
                        s
                    })
                    .collect();
                Reader::Stretch(Box::new(Stretch {
                    channels: stretchers,
                    ahead: vec![Vec::new(); channels],
                    skip: origin.max(0) as usize,
                    origin,
                    channel: Vec::new(),
                    off: 0.0,
                }))
            }
        }
    }

    /// Reads `frames` (interleaved, `ch` channels) along the map, frame
    /// `k` at input frame `position(k)` where the map's speed is
    /// `speed_at(k)`, and replaces them with what was read (`ended`: the
    /// input has all been pushed).
    fn read(
        &mut self,
        frames: &mut Vec<f64>,
        ch: usize,
        ended: bool,
        speed_at: impl Fn(u64) -> f64,
        position: impl Fn(u64) -> f64,
    ) {
        let rate = f64::from(RATE);
        match self {
            Reader::Speed(reader) => {
                reader.push(frames);
                frames.clear();
                let n = if ended {
                    // Up to the end of the input, then nothing more.
                    frames_before(reader.made(), reader.taken(), &position)
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
                for (c, stretcher) in s.channels.iter_mut().enumerate() {
                    s.channel.clear();
                    s.channel.extend(frames.iter().skip(c).step_by(ch).copied());
                    stretcher.push(&s.channel);
                    if ended {
                        stretcher.finish();
                    }
                }
                frames.clear();
                // Steer: the map's speed here, and the way back onto it.
                let k = s.channels[0].output_position();
                let off = position(k) - s.origin as f64 - s.channels[0].input_position();
                s.off = off;
                let steer = (off / (STEER * rate)).clamp(-STEER_LIMIT, STEER_LIMIT);
                let speed = speed_at(k) * (1.0 + steer);
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
            Reader::Shift(shift, tempo) => {
                // Stretched to the map's speed, its stretchers go at it;
                // read at a changing speed, the map's speed is the reader's.
                let speed = match tempo {
                    TempoMode::Stretch => speed_at(shift.made()),
                    TempoMode::Varispeed => 1.0,
                };
                let input = std::mem::take(frames);
                shift.read(&input, ended, speed, position, frames);
            }
        }
    }
}

impl Warping {
    pub(super) fn new(reading: Reading) -> Warping {
        let Reading {
            map,
            a0,
            b_origin,
            tempo,
            channels,
            key,
            kept,
        } = reading;
        let channels = channels.max(1);
        let group = |channels: Vec<usize>, reader| Group {
            channels,
            reader,
            frames: Vec::new(),
            made: Vec::new(),
        };
        let (kept, shifted): (Vec<usize>, Vec<usize>) = if key == 0 {
            ((0..channels).collect(), Vec::new())
        } else {
            (0..channels).partition(|c| kept.contains(c))
        };
        let mut groups = Vec::new();
        if !kept.is_empty() {
            let reader = Reader::new(&map, a0, b_origin, tempo, kept.len());
            groups.push(group(kept, reader));
        }
        if !shifted.is_empty() {
            let shift = Box::new(Shift::new(key, shifted.len(), RATE));
            groups.push(group(shifted, Reader::Shift(shift, tempo)));
        }
        Warping {
            map,
            a0,
            b_origin,
            channels,
            groups,
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
            groups,
            ended,
        } = self;
        let (rate, ch, ended) = (f64::from(RATE), *channels, *ended);
        let (map, a0, b_origin) = (&**map, *a0, *b_origin);
        let position = |k: u64| (map.at(a0 + k as f64 / rate) - b_origin) * rate;
        let speed_at = |k: u64| map.speed(a0 + k as f64 / rate);
        // All the channels alike: read as they are.
        if let [group] = groups.as_mut_slice() {
            return group.reader.read(frames, ch, ended, speed_at, position);
        }
        for g in groups.iter_mut() {
            g.frames.clear();
            for frame in frames.chunks_exact(ch) {
                g.frames.extend(g.channels.iter().map(|&c| frame[c]));
            }
            let width = g.channels.len();
            g.reader
                .read(&mut g.frames, width, ended, speed_at, position);
            g.made.extend_from_slice(&g.frames);
        }
        // What every group made, frame by frame; at the end the shorter
        // padded with silence.
        let made = groups.iter().map(|g| g.made.len() / g.channels.len());
        let n = if ended {
            made.max().unwrap_or(0)
        } else {
            made.min().unwrap_or(0)
        };
        frames.clear();
        frames.resize(n * ch, 0.0);
        for g in groups.iter_mut() {
            let width = g.channels.len();
            g.made.resize(g.made.len().max(n * width), 0.0);
            for (i, from) in g.made.chunks_exact(width).take(n).enumerate() {
                for (&c, &v) in g.channels.iter().zip(from) {
                    frames[i * ch + c] = v;
                }
            }
            g.made.drain(..n * width);
        }
    }

    /// How far (in input frames) what it reads is off the map: what the
    /// stretchers' steering has yet to take back (none at a changing speed,
    /// which reads on the map, nor where a key shifts, read on it too).
    pub(super) fn offset(&self) -> f64 {
        self.groups
            .iter()
            .find_map(|g| match &g.reader {
                Reader::Stretch(s) => Some(s.off),
                _ => None,
            })
            .unwrap_or(0.0)
    }

    /// The input has ended.
    pub(super) fn end(&mut self) {
        self.ended = true;
        for g in &mut self.groups {
            if let Reader::Speed(reader) = &mut g.reader {
                reader.end();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// A map whose first position is before the input's first frame (a
    /// transition's next track coming in near its start): what is read
    /// starts that much silence later, a click on the frame the map puts
    /// it on, at a changing speed and stretched alike, a key apart too (the
    /// first two channels kept in their key, the others shifted).
    #[test]
    fn a_map_that_starts_before_the_input_reads_silence_first() {
        // 5 % fast; the input starts 0.3 s into the map's time.
        let map = Arc::new(TimeMap::new(&[(0.0, 0.0), (10.0, 10.5)], 0.0));
        let (rate, channels) = (f64::from(RATE), 4);
        let frames = RATE as usize * 2;
        let click = (0.1 * rate) as usize;
        let mut input = vec![0.0; frames * channels];
        for i in 0..40 {
            let v = (2.0 * std::f64::consts::PI * 1_000.0 * i as f64 / rate).sin();
            input[(click + i) * channels..(click + i + 1) * channels].fill(v);
        }
        // The track's 0.4 s, heard at 0.4 / 1.05 s.
        let want = 0.4 / 1.05;
        for (tempo, key) in [
            (TempoMode::Varispeed, 0),
            (TempoMode::Stretch, 0),
            (TempoMode::Stretch, 3),
        ] {
            let mut warping = Warping::new(Reading {
                map: Arc::clone(&map),
                a0: 0.0,
                b_origin: 0.3,
                tempo,
                channels,
                key,
                kept: vec![0, 1],
            });
            let mut out = Vec::new();
            for chunk in input.chunks(4096 * channels) {
                let mut piece = chunk.to_vec();
                warping.read(&mut piece);
                out.extend(piece);
            }
            warping.end();
            let mut piece = Vec::new();
            warping.read(&mut piece);
            out.extend(piece);
            for c in 0..channels {
                let loudest = (0..out.len() / channels)
                    .max_by(|&a, &b| {
                        out[a * channels + c]
                            .abs()
                            .total_cmp(&out[b * channels + c].abs())
                    })
                    .unwrap();
                let at = loudest as f64 / rate;
                assert!(
                    (at - want).abs() < 0.002,
                    "{tempo:?} {key:+}, channel {c}: the click at {at} s, the map's {want} s"
                );
            }
        }
    }

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
            let mut warping = Warping::new(Reading {
                map: Arc::clone(&map),
                a0: 0.0,
                b_origin: 0.0,
                tempo,
                channels,
                key: 0,
                kept: Vec::new(),
            });
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
