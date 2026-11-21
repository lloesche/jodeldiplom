//! A track's waveform for display: the mix and each stem, frame by frame.
//!
//! Each [`Lane`] holds, per 10 ms frame ([`FPS`]), the peak and the RMS
//! of the mono signal (the mean of the channels), each in decibels below
//! full scale squeezed into a byte: 0 at [`FLOOR_DB`] or below, 255 at
//! 0 dBFS. On that scale a quiet passage still shows, as the eye expects
//! of a waveform, and a four-minute track takes about 50 KB a lane.
//! [`LaneBuilder`] makes a lane from samples at any rate;
//! [`Waveform::to_bytes`] and [`Waveform::from_bytes`] keep one on disk.
//! With the stems comes the voice's [`Melody`], at the same frames.

/// Frames per second.
pub const FPS: f64 = 100.0;
/// The level shown as nothing, in dBFS.
pub const FLOOR_DB: f64 = -60.0;
/// The format's first bytes.
const MAGIC: &[u8; 4] = b"JDLW";
/// The format's version: raised with any change to what [`Waveform::to_bytes`]
/// writes (the player's cache schema "waveform").
pub const FORMAT: u8 = 2;

/// One signal's levels, frame by frame, each `0..=255` on the dB scale
/// (see the module).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lane {
    /// The largest magnitude in each frame.
    pub peak: Vec<u8>,
    /// The RMS of each frame.
    pub rms: Vec<u8>,
}

/// A magnitude (full scale 1) as a byte on the dB scale.
fn level(magnitude: f64) -> u8 {
    if magnitude <= 0.0 {
        return 0;
    }
    let db = 20.0 * magnitude.log10();
    (255.0 * (db - FLOOR_DB) / -FLOOR_DB)
        .round()
        .clamp(0.0, 255.0) as u8
}

impl Lane {
    /// The number of frames.
    pub fn len(&self) -> usize {
        self.peak.len()
    }

    /// Whether it has no frames.
    pub fn is_empty(&self) -> bool {
        self.peak.is_empty()
    }

    /// The frames covering `start..end` seconds (at least one where the
    /// lane has any, clamped to it).
    fn frames(&self, start: f64, end: f64) -> std::ops::Range<usize> {
        let n = self.len();
        let lo = ((start * FPS).floor().max(0.0) as usize).min(n);
        let hi = ((end * FPS).ceil().max(0.0) as usize).clamp(lo, n);
        if hi == lo && lo < n {
            lo..lo + 1
        } else {
            lo..hi
        }
    }

    /// The highest peak over `start..end` seconds, `0..=1` on the dB scale
    /// (0 outside the lane).
    pub fn peak_between(&self, start: f64, end: f64) -> f32 {
        let r = self.frames(start, end);
        f32::from(self.peak[r].iter().copied().max().unwrap_or(0)) / 255.0
    }

    /// The highest RMS over `start..end` seconds, likewise.
    pub fn rms_between(&self, start: f64, end: f64) -> f32 {
        let r = self.frames(start, end);
        f32::from(self.rms[r].iter().copied().max().unwrap_or(0)) / 255.0
    }
}

/// Makes a [`Lane`] from mono samples at `rate`, frame by frame.
#[derive(Clone, Debug)]
pub struct LaneBuilder {
    rate: f64,
    lane: Lane,
    /// Samples taken in, and those in the frame being filled.
    taken: u64,
    filled: usize,
    /// The sample after which the frame being filled ends.
    next_end: u64,
    peak: f32,
    sum: f64,
}

impl LaneBuilder {
    /// An empty lane of samples at `rate` hertz.
    pub fn new(rate: u32) -> LaneBuilder {
        LaneBuilder {
            rate: f64::from(rate),
            lane: Lane::default(),
            taken: 0,
            filled: 0,
            next_end: (f64::from(rate) / FPS).round() as u64,
            peak: 0.0,
            sum: 0.0,
        }
    }

    /// The sample after which frame `k` ends.
    fn end_of(&self, k: usize) -> u64 {
        (((k + 1) as f64) * self.rate / FPS).round() as u64
    }

    /// Takes one sample (full scale ±1).
    pub fn push_one(&mut self, x: f32) {
        self.peak = self.peak.max(x.abs());
        self.sum += f64::from(x) * f64::from(x);
        self.filled += 1;
        self.taken += 1;
        if self.taken >= self.next_end {
            self.close();
            self.next_end = self.end_of(self.lane.len());
        }
    }

    /// Takes samples (full scale ±1).
    pub fn push(&mut self, samples: &[f32]) {
        for &x in samples {
            self.push_one(x);
        }
    }

    /// Ends the frame being filled.
    fn close(&mut self) {
        let n = self.filled.max(1) as f64;
        self.lane.peak.push(level(f64::from(self.peak)));
        self.lane.rms.push(level((self.sum / n).sqrt()));
        self.filled = 0;
        self.peak = 0.0;
        self.sum = 0.0;
    }

    /// The lane, with the last frame if it holds any samples.
    pub fn finish(mut self) -> Lane {
        if self.filled > 0 {
            self.close();
        }
        self.lane
    }
}

/// The voice's pitch, frame by frame ([`FPS`]): what
/// [`crate::pitch`] heard in the vocals stem.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Melody {
    /// The pitch in hundredths of a MIDI note (6900 is A4), 0 where there
    /// is none.
    pub cents: Vec<u16>,
    /// How sure the network was there is a pitch, 0 to 255.
    pub sure: Vec<u8>,
}

impl Melody {
    /// The melody of a pitch track.
    pub fn from_pitch(p: &crate::pitch::Pitch) -> Melody {
        let mut m = Melody::default();
        for i in 0..p.hz.len() {
            let cents = p
                .midi(i)
                .map_or(0, |n| (n * 100.0).round().clamp(1.0, 65535.0) as u16);
            m.cents.push(cents);
            m.sure
                .push((p.confidence[i].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        m
    }

    /// Its frames.
    pub fn len(&self) -> usize {
        self.cents.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.cents.is_empty()
    }

    /// Frame `i`'s pitch as a MIDI note number with its fraction; `None`
    /// where there is none.
    pub fn note(&self, i: usize) -> Option<f32> {
        let c = *self.cents.get(i)?;
        (c > 0).then(|| f32::from(c) / 100.0)
    }

    /// The note sung at frame `i`, to the nearest semitone: the median
    /// pitch of the frames within [`Melody::NEAR`] of it that the network
    /// is sure of ([`Melody::SURE`]), where most of them are. A frame's
    /// note depends on its neighbours only, so any stretch of the melody
    /// gives each frame the same.
    pub fn sung(&self, i: usize) -> Option<u8> {
        let lo = i.saturating_sub(Self::NEAR);
        let hi = (i + Self::NEAR + 1).min(self.len());
        if lo >= hi {
            return None;
        }
        let mut near = [0u16; 2 * Self::NEAR + 1];
        let mut n = 0;
        for j in lo..hi {
            if self.cents[j] > 0 && self.sure[j] >= Self::SURE {
                near[n] = self.cents[j];
                n += 1;
            }
        }
        if n <= Self::NEAR {
            return None;
        }
        let near = &mut near[..n];
        near.sort_unstable();
        let median = near[n / 2];
        Some(((f32::from(median) / 100.0).round()).clamp(0.0, 127.0) as u8)
    }

    /// The notes the voice mostly sings in: from the 2nd to the 98th
    /// percentile of its sure frames' pitches, to the semitone; `None`
    /// where it sings none.
    pub fn range(&self) -> Option<(u8, u8)> {
        let mut notes: Vec<u16> = self
            .cents
            .iter()
            .zip(&self.sure)
            .filter(|&(&c, &s)| c > 0 && s >= Self::SURE)
            .map(|(&c, _)| c)
            .collect();
        if notes.is_empty() {
            return None;
        }
        notes.sort_unstable();
        let at = |q: f64| {
            let c = notes[((notes.len() - 1) as f64 * q).round() as usize];
            ((f32::from(c) / 100.0).round()).clamp(0.0, 127.0) as u8
        };
        Some((at(0.02), at(0.98)))
    }

    /// Whether frame `i` is heard surely as a pitch.
    fn is_sure(&self, i: usize) -> bool {
        self.cents.get(i).is_some_and(|&c| c > 0) && self.sure[i] >= Self::SURE
    }

    /// The melody with what is not sung taken out: frames outside every
    /// span of `spans` (seconds, a song's words, give or take
    /// [`Melody::SLACK`]; `None`: all kept), and runs of sure frames
    /// shorter than [`Melody::LEAST_RUN`] (a breath, an echo, the
    /// separation's leftovers between words), whose pitches become none.
    pub fn cleaned(&self, spans: Option<&[(f64, f64)]>) -> Melody {
        let n = self.len();
        let mut keep = vec![spans.is_none(); n];
        for &(s, e) in spans.unwrap_or(&[]) {
            let a = (((s - Self::SLACK) * FPS).floor().max(0.0) as usize).min(n);
            let b = (((e + Self::SLACK) * FPS).ceil().max(0.0) as usize).min(n);
            keep[a..b.max(a)].iter_mut().for_each(|k| *k = true);
        }
        let mut out = self.clone();
        let mut i = 0;
        while i < n {
            if !(keep[i] && self.is_sure(i)) {
                out.cents[i] = 0;
                i += 1;
                continue;
            }
            let mut j = i;
            while j < n && keep[j] && self.is_sure(j) {
                j += 1;
            }
            if j - i < Self::LEAST_RUN {
                out.cents[i..j].iter_mut().for_each(|c| *c = 0);
            }
            i = j;
        }
        out
    }

    /// The notes sung from `start` to `end` seconds (a word): each stretch
    /// of sure frames there in runs of at least [`Melody::LEAST_NOTE`] (a
    /// run reaching past the stretch counted whole), split where the
    /// pitch moves to another semitone (each frame's the median of its
    /// sure neighbours within [`Melody::NEAR`]); a note shorter than
    /// [`Melody::LEAST_NOTE`] joins the one before (or after). Each as
    /// (from, to) seconds and its semitone in hundredths of a MIDI note.
    pub fn notes_between(&self, start: f64, end: f64) -> Vec<(f64, f64, f64)> {
        let n = self.len();
        let a = ((start * FPS).floor().max(0.0) as usize).min(n);
        let b = ((end * FPS).ceil().max(0.0) as usize).min(n);
        // Each kept frame's semitone.
        let mut frames: Vec<(usize, i32)> = Vec::new();
        let mut i = a;
        while i < b {
            if !self.is_sure(i) {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < b && self.is_sure(j) {
                j += 1;
            }
            let mut from = i;
            while from > 0 && self.is_sure(from - 1) && i - from < Self::LEAST_NOTE {
                from -= 1;
            }
            let mut to = j;
            while to < n && self.is_sure(to) && to - j < Self::LEAST_NOTE {
                to += 1;
            }
            if to - from >= Self::LEAST_NOTE {
                for k in i..j {
                    let lo = k.saturating_sub(Self::NEAR).max(from);
                    let hi = (k + Self::NEAR + 1).min(to);
                    let mut near: Vec<u16> = self.cents[lo..hi].to_vec();
                    near.sort_unstable();
                    let semitone = (f64::from(near[near.len() / 2]) / 100.0).round() as i32;
                    frames.push((k, semitone));
                }
            }
            i = j;
        }
        // Runs of one semitone, frame after frame.
        let mut notes: Vec<(usize, usize, i32)> = Vec::new();
        for (k, s) in frames {
            match notes.last_mut() {
                Some(last) if last.1 == k && last.2 == s => last.1 = k + 1,
                _ => notes.push((k, k + 1, s)),
            }
        }
        // The short ones into a neighbour that touches them.
        let mut k = 0;
        while k < notes.len() {
            let (s, e, _) = notes[k];
            if e - s >= Self::LEAST_NOTE || notes.len() == 1 {
                k += 1;
                continue;
            }
            if k > 0 && notes[k - 1].1 == s {
                notes[k - 1].1 = e;
                notes.remove(k);
            } else if k + 1 < notes.len() && notes[k + 1].0 == e {
                notes[k + 1].0 = s;
                notes.remove(k);
            } else {
                k += 1;
            }
        }
        // Neighbours of one semitone again, as one.
        let mut merged: Vec<(usize, usize, i32)> = Vec::new();
        for note in notes {
            match merged.last_mut() {
                Some(last) if last.1 == note.0 && last.2 == note.2 => last.1 = note.1,
                _ => merged.push(note),
            }
        }
        // A short word sung fast has its pitch only for a moment: with no
        // note, its sure frames (at least LEAST_WORD) as one, at their
        // median.
        if merged.is_empty() {
            let sure: Vec<usize> = (a..b).filter(|&i| self.is_sure(i)).collect();
            if sure.len() >= Self::LEAST_WORD {
                let mut cents: Vec<u16> = sure.iter().map(|&i| self.cents[i]).collect();
                cents.sort_unstable();
                let semitone = (f64::from(cents[cents.len() / 2]) / 100.0).round() as i32;
                merged.push((sure[0], sure[sure.len() - 1] + 1, semitone));
            }
        }
        merged
            .into_iter()
            .map(|(s, e, c)| (s as f64 / FPS, e as f64 / FPS, f64::from(c) * 100.0))
            .collect()
    }

    /// How sure the network must be of a frame for [`Melody::sung`] (out
    /// of 255).
    pub const SURE: u8 = 128;
    /// The shortest note [`Melody::notes_between`] keeps apart, in frames
    /// (a pitch passed through on a slide joins its neighbour).
    pub const LEAST_NOTE: usize = 8;
    /// The fewest sure frames that give a word without a note of its own
    /// one note (a short word sung fast: 0.03 s).
    pub const LEAST_WORD: usize = 3;
    /// Seconds before and after a word its notes may reach (a slide into
    /// it, a note held past it).
    pub const SLACK: f64 = 0.1;
    /// The fewest frames in a row a sung note is: shorter runs are a
    /// breath, an echo or the separation's leftovers (0.15 s).
    pub const LEAST_RUN: usize = 15;
    /// The frames either side of a frame that [`Melody::sung`] hears.
    pub const NEAR: usize = 3;
}

/// A track's waveform: the mix, and the stems when they were separated
/// (in [`crate::stems::Stem::ALL`] order: bass, drums, other, vocals),
/// with the voice's melody where it was heard.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Waveform {
    /// The whole music.
    pub mix: Lane,
    /// Each stem.
    pub stems: Option<[Lane; 4]>,
    /// The voice's pitch.
    pub melody: Option<Melody>,
}

impl Waveform {
    /// The track's length the lanes cover, in seconds.
    pub fn duration(&self) -> f64 {
        self.mix.len() as f64 / FPS
    }

    /// The bytes that keep it: `JDLW`, the format, the lane count and
    /// the frame count of each lane, then each lane's peaks and RMS; then
    /// the melody's frame count (u32, 0 for none), its pitches (u16 each)
    /// and how sure each is (a byte each). Little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let lanes: Vec<&Lane> = std::iter::once(&self.mix)
            .chain(self.stems.iter().flatten())
            .collect();
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(FORMAT);
        out.push(lanes.len() as u8);
        for lane in &lanes {
            out.extend_from_slice(&(lane.len() as u32).to_le_bytes());
        }
        for lane in &lanes {
            out.extend_from_slice(&lane.peak);
            out.extend_from_slice(&lane.rms);
        }
        let melody = self.melody.as_ref();
        out.extend_from_slice(&(melody.map_or(0, Melody::len) as u32).to_le_bytes());
        if let Some(m) = melody {
            for c in &m.cents {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&m.sure);
        }
        out
    }

    /// A waveform from [`Waveform::to_bytes`]; `None` for anything else.
    pub fn from_bytes(bytes: &[u8]) -> Option<Waveform> {
        let rest = bytes.strip_prefix(MAGIC)?;
        let (&[format, count], rest) = rest.split_first_chunk::<2>()?;
        if format != FORMAT || !(count == 1 || count == 5) {
            return None;
        }
        let mut lens = Vec::new();
        let mut rest = rest;
        for _ in 0..count {
            let (len, tail) = rest.split_first_chunk::<4>()?;
            lens.push(u32::from_le_bytes(*len) as usize);
            rest = tail;
        }
        let mut lanes = Vec::new();
        for len in lens {
            if rest.len() < 2 * len {
                return None;
            }
            let (peak, tail) = rest.split_at(len);
            let (rms, tail) = tail.split_at(len);
            lanes.push(Lane {
                peak: peak.to_vec(),
                rms: rms.to_vec(),
            });
            rest = tail;
        }
        let (len, tail) = rest.split_first_chunk::<4>()?;
        let len = u32::from_le_bytes(*len) as usize;
        if tail.len() != 3 * len {
            return None;
        }
        let (cents, sure) = tail.split_at(2 * len);
        let melody = (len > 0).then(|| Melody {
            cents: cents
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
            sure: sure.to_vec(),
        });
        let mix = lanes.remove(0);
        let stems = (lanes.len() == 4).then(|| {
            let mut it = lanes.into_iter();
            std::array::from_fn(|_| it.next().expect("four lanes"))
        });
        Some(Waveform { mix, stems, melody })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A melody of `seconds`, sure at `cents` over the frames `runs` give
    /// (from, to), nothing elsewhere.
    fn melody(seconds: f64, runs: &[(usize, usize, u16)]) -> Melody {
        let n = (seconds * FPS) as usize;
        let mut m = Melody {
            cents: vec![0; n],
            sure: vec![0; n],
        };
        for &(a, b, c) in runs {
            for i in a..b {
                m.cents[i] = c;
                m.sure[i] = 255;
            }
        }
        m
    }

    #[test]
    fn what_is_not_sung_is_taken_out_of_the_melody() {
        // A note in a word (1.0 to 1.5 s), a crumb inside it, a breath
        // between words (3.0 s), a note of the next word (4.0 to 4.6 s).
        let m = melody(
            6.0,
            &[
                (100, 150, 6000),
                (200, 205, 6100),
                (300, 320, 6500),
                (400, 460, 6200),
            ],
        );
        let words = [(1.0, 1.5), (1.9, 2.2), (4.0, 4.6)];
        let c = m.cleaned(Some(&words));
        let kept = |a: usize, b: usize| c.cents[a..b].iter().all(|&x| x > 0);
        let gone = |a: usize, b: usize| c.cents[a..b].iter().all(|&x| x == 0);
        assert!(kept(100, 150) && kept(400, 460));
        assert!(gone(200, 205), "a crumb");
        assert!(gone(300, 320), "between the words");
        // Without words: only the crumb goes.
        let c = m.cleaned(None);
        assert!(c.cents[300] > 0 && c.cents[200] == 0);
        // A word's notes: its own.
        assert_eq!(m.notes_between(1.0, 1.5), [(1.0, 1.5, 6000.0)]);
        // A short word with its pitch only for a moment: that, as one
        // note; a moment shorter than LEAST_WORD frames, none.
        assert_eq!(m.notes_between(1.95, 2.05), [(2.0, 2.05, 6100.0)]);
        let blip = melody(3.0, &[(200, 202, 6100)]);
        assert!(blip.notes_between(1.95, 2.05).is_empty());
        // A note running on past the word still counts whole.
        assert_eq!(m.notes_between(4.5, 4.6), [(4.5, 4.6, 6200.0)]);
    }

    #[test]
    fn a_word_s_notes_split_where_the_pitch_moves() {
        // Up a tone after 0.3 s, a slide through a semitone between (0.05 s,
        // joining the note before), down again for the last 0.2 s.
        let mut runs = vec![(100, 130, 6000), (130, 135, 6100), (135, 160, 6200)];
        runs.push((160, 180, 6000));
        let m = melody(3.0, &runs);
        let notes = m.notes_between(1.0, 1.8);
        let cents: Vec<f64> = notes.iter().map(|n| n.2).collect();
        assert_eq!(cents, [6000.0, 6200.0, 6000.0]);
        // Each where it is sung, end to end.
        assert!((notes[0].0 - 1.0).abs() < 1e-9 && (notes[2].1 - 1.8).abs() < 1e-9);
        assert!(notes.windows(2).all(|w| (w[0].1 - w[1].0).abs() < 1e-9));
    }

    #[test]
    fn levels_are_peaks_and_rms_in_decibels() {
        // One second of a full-scale square wave, then one of silence, then
        // a tone at -20 dB, at 22.05 kHz.
        let rate = 22_050;
        let mut b = LaneBuilder::new(rate);
        let square: Vec<f32> = (0..rate)
            .map(|i| if i % 50 < 25 { 1.0 } else { -1.0 })
            .collect();
        b.push(&square);
        b.push(&vec![0.0; rate as usize]);
        let tone: Vec<f32> = (0..rate).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
        b.push(&tone);
        let lane = b.finish();
        assert_eq!(lane.len(), 300);
        assert_eq!(lane.peak[50], 255);
        assert_eq!(lane.rms[50], 255);
        assert_eq!(lane.peak[150], 0);
        // -20 dB is a third of the way down from 0 to -60.
        assert!(
            (i32::from(lane.peak[250]) - 170).abs() <= 2,
            "{}",
            lane.peak[250]
        );
        assert!(lane.rms[250] < lane.peak[250]);
        assert!((lane.peak_between(0.5, 1.5) - 1.0).abs() < 1e-6);
        assert_eq!(lane.peak_between(1.2, 1.8), 0.0);
        assert_eq!(lane.peak_between(10.0, 11.0), 0.0);
    }

    #[test]
    fn a_waveform_survives_its_bytes() {
        let lane = |seed: u8| Lane {
            peak: (0..123).map(|i| (i as u8).wrapping_mul(seed)).collect(),
            rms: (0..123).map(|i| (i as u8).wrapping_add(seed)).collect(),
        };
        let alone = Waveform {
            mix: lane(3),
            stems: None,
            melody: None,
        };
        assert_eq!(Waveform::from_bytes(&alone.to_bytes()), Some(alone.clone()));
        let all = Waveform {
            mix: lane(3),
            stems: Some([lane(5), lane(7), lane(11), lane(13)]),
            melody: Some(Melody {
                cents: vec![0, 6900, 6912, 0],
                sure: vec![0, 200, 255, 1],
            }),
        };
        let bytes = all.to_bytes();
        assert_eq!(Waveform::from_bytes(&bytes), Some(all));
        assert_eq!(Waveform::from_bytes(&bytes[..bytes.len() - 1]), None);
        assert_eq!(Waveform::from_bytes(b"nope"), None);
    }

    /// A note is the median of the sure frames around it: a frame unsure
    /// or a slide at its edge does not move it, a stretch of too few sure
    /// frames has none; the range leaves out the odd stray frame.
    #[test]
    fn notes_are_the_sure_frames_around() {
        let mut m = Melody::default();
        // 20 frames at A4 a little sharp, one wild frame in them, 20
        // unsure, 20 at C5, then two sure frames alone.
        for i in 0..20 {
            m.cents.push(if i == 10 { 8000 } else { 6912 });
            m.sure.push(230);
        }
        for _ in 0..20 {
            m.cents.push(6500);
            m.sure.push(40);
        }
        for _ in 0..20 {
            m.cents.push(7200);
            m.sure.push(250);
        }
        for i in 0..20 {
            m.cents.push(if i % 10 == 0 { 7000 } else { 0 });
            m.sure.push(200);
        }
        assert_eq!(m.sung(10), Some(69));
        assert_eq!(m.sung(0), Some(69));
        assert_eq!(m.sung(30), None);
        assert_eq!(m.sung(41), Some(72));
        assert_eq!(m.sung(70), None);
        assert_eq!(m.sung(1000), None);
        assert_eq!(m.range(), Some((69, 72)));
        assert_eq!(Melody::default().range(), None);
    }
}
