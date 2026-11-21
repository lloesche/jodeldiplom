//! What a device accepts exclusively, and choosing a format from it.

use crate::format::{SampleFormat, StreamFormat};
use std::collections::BTreeSet;

/// The rates [`crate::capabilities`] probes: the 44.1 and 48 kHz families
/// from 1× to 16×.
pub const PROBE_RATES: [u32; 10] = [
    44_100, 48_000, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000, 705_600, 768_000,
];

/// The (rate, channels, format) combinations a device accepts in exclusive
/// mode, as probed by [`crate::capabilities`] or built by hand for tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    formats: BTreeSet<StreamFormat>,
}

impl Capabilities {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one accepted combination.
    pub fn insert(&mut self, format: StreamFormat) {
        self.formats.insert(format);
    }

    /// Whether the device accepts `format`.
    pub fn supports(&self, format: &StreamFormat) -> bool {
        self.formats.contains(format)
    }

    /// Whether nothing was accepted.
    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }

    /// Every accepted combination, ordered by rate, then channels, then format.
    pub fn iter(&self) -> impl Iterator<Item = &StreamFormat> + '_ {
        self.formats.iter()
    }

    /// The channel counts that appear, ascending.
    pub fn channel_counts(&self) -> Vec<u16> {
        let set: BTreeSet<u16> = self.formats.iter().map(|f| f.channels).collect();
        set.into_iter().collect()
    }

    /// The rates accepted with `channels` channels in any format, ascending.
    pub fn rates(&self, channels: u16) -> Vec<u32> {
        let set: BTreeSet<u32> = self
            .formats
            .iter()
            .filter(|f| f.channels == channels)
            .map(|f| f.rate)
            .collect();
        set.into_iter().collect()
    }

    /// The formats accepted at `rate` with `channels` channels.
    pub fn formats(&self, rate: u32, channels: u16) -> Vec<SampleFormat> {
        self.formats
            .iter()
            .filter(|f| f.rate == rate && f.channels == channels)
            .map(|f| f.sample)
            .collect()
    }

    /// The first format in `preference` that the device accepts at `rate`
    /// with `channels` channels (see [`choose_format`]).
    pub fn choose(
        &self,
        rate: u32,
        channels: u16,
        preference: &[SampleFormat],
    ) -> Option<StreamFormat> {
        choose_format(rate, channels, preference, |f| self.supports(&f))
    }
}

impl FromIterator<StreamFormat> for Capabilities {
    fn from_iter<I: IntoIterator<Item = StreamFormat>>(iter: I) -> Self {
        Capabilities {
            formats: iter.into_iter().collect(),
        }
    }
}

/// `preference` in order with repeats removed: the order in which an open
/// tries formats.
pub fn candidates(preference: &[SampleFormat]) -> Vec<SampleFormat> {
    let mut out = Vec::with_capacity(preference.len());
    for &f in preference {
        if !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

/// Format negotiation: the first format in `preference` for which `accepts`
/// says yes at `rate` and `channels`. Nothing else is tried: no other rate,
/// no other channel count, no format outside the list. The backends use the
/// same rule with the device's own answer as `accepts`.
pub fn choose_format(
    rate: u32,
    channels: u16,
    preference: &[SampleFormat],
    mut accepts: impl FnMut(StreamFormat) -> bool,
) -> Option<StreamFormat> {
    candidates(preference)
        .into_iter()
        .map(|sample| StreamFormat::new(rate, channels, sample))
        .find(|f| accepts(*f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use SampleFormat::*;

    /// A device like the one the probes found: packed 24 and 16 bit only, at
    /// 48/96/192 kHz in stereo, and 48/96 kHz in 8 channels.
    fn packed_only_device() -> Capabilities {
        let mut caps = Capabilities::new();
        for rate in [48_000, 96_000, 192_000] {
            for f in [S24Packed, S16] {
                caps.insert(StreamFormat::new(rate, 2, f));
            }
        }
        for rate in [48_000, 96_000] {
            for f in [S24Packed, S16] {
                caps.insert(StreamFormat::new(rate, 8, f));
            }
        }
        caps
    }

    #[test]
    fn picks_the_first_accepted_preference() {
        let caps = packed_only_device();
        assert_eq!(
            caps.choose(96_000, 2, &SampleFormat::PREFERENCE),
            Some(StreamFormat::new(96_000, 2, S24Packed))
        );
        assert_eq!(
            caps.choose(96_000, 2, &[S16, S24Packed]),
            Some(StreamFormat::new(96_000, 2, S16))
        );
    }

    #[test]
    fn never_substitutes_a_rate_channel_count_or_unlisted_format() {
        let caps = packed_only_device();
        assert_eq!(caps.choose(44_100, 2, &SampleFormat::PREFERENCE), None);
        assert_eq!(caps.choose(192_000, 8, &SampleFormat::PREFERENCE), None);
        assert_eq!(caps.choose(192_000, 1, &SampleFormat::PREFERENCE), None);
        assert_eq!(caps.choose(48_000, 2, &[S32, S24In32]), None);
        assert_eq!(caps.choose(48_000, 2, &[]), None);
    }

    #[test]
    fn asks_in_preference_order_once_each() {
        let mut asked = Vec::new();
        let got = choose_format(176_400, 2, &[S32, S24In32, S32, S24Packed, S16], |f| {
            asked.push(f.sample);
            f.sample == S24Packed
        });
        assert_eq!(got, Some(StreamFormat::new(176_400, 2, S24Packed)));
        assert_eq!(asked, [S32, S24In32, S24Packed]);
    }

    #[test]
    fn queries_describe_the_set() {
        let caps = packed_only_device();
        assert_eq!(caps.channel_counts(), [2, 8]);
        assert_eq!(caps.rates(2), [48_000, 96_000, 192_000]);
        assert_eq!(caps.rates(8), [48_000, 96_000]);
        assert_eq!(caps.formats(48_000, 2), [S16, S24Packed]);
        assert!(caps.formats(44_100, 2).is_empty());
        assert_eq!(caps.iter().count(), 10);
        let rebuilt: Capabilities = caps.iter().copied().collect();
        assert_eq!(rebuilt, caps);
    }
}
