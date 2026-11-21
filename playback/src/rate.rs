//! The rate policy: from what a file is, what the device takes exclusively
//! and the user's settings to the [`Route`] a track plays through, or the
//! reason none fits ([`NoRoute`]).
//!
//! [`choose_route`] is a pure function:
//!
//! - **Lossless PCM** plays at the file's own rate and channels, untouched.
//! - **Lossy files** (MP3, Vorbis, ...) are decoded to 24-bit PCM and
//!   routed as plain PCM, never lossless (the route says what it was
//!   decoded from, [`Route::decoded_from`]).
//!
//! A route fits when its output rate is within `max_rate` and the device
//! takes the file's channel count at that rate in a format that holds every
//! bit of the stream: at least the file's bits (24 for lossy and
//! floating-point files). Such a bit-perfect route always wins.
//!
//! When none fits and [`Settings::resample`] is on, the route gains a
//! [`Resample`] step ([`crate::resample`]) to a rate the device takes in a
//! 24-bit format within the cap, and is no longer lossless: the lowest such
//! rate above the file's rate, else the highest one below it.
//!
//! A route that is not bit-perfect carries the [`Settings::headroom`] at
//! its one requantization to 24 bits: inside the resampler, or where a
//! lossy or floating-point file is turned into 24-bit samples. A lossy
//! file's always has at least [`LOSSY_HEADROOM`] (its decoded signal goes
//! over full scale where the master was near it). A bit-perfect route never
//! has any.
//!
//! When nothing fits either way the answer is a [`NoRoute`] that says why.

use std::fmt;

use jodel_output::{Capabilities, OpenRequest, SampleFormat, StreamFormat};

use crate::source::TrackInfo;
use crate::stems::StemMix;

/// A gain below full scale for routes that are not bit-perfect, in tenths
/// of a dB (`Headroom(10)` is −1 dB): the samples are scaled by it before
/// the route's one requantization to 24 bits (in the resampler's `f64`
/// computation, or the source's for a lossy or floating-point file), so
/// overs of a near-full-scale signal (inter-sample ones, or a lossy
/// codec's) need not clip. It never touches a bit-perfect route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Headroom(pub u16);

impl Headroom {
    /// No headroom (the default).
    pub const NONE: Headroom = Headroom(0);

    /// The choices a player offers: none, 0.5, 1, 2 and 3 dB.
    pub const CHOICES: [Headroom; 5] = [
        Headroom(0),
        Headroom(5),
        Headroom(10),
        Headroom(20),
        Headroom(30),
    ];

    /// The headroom in dB.
    pub fn db(self) -> f64 {
        f64::from(self.0) / 10.0
    }

    /// The linear gain, `10^(−dB/20)` (1 for none).
    pub fn gain(self) -> f64 {
        if self.0 == 0 {
            1.0
        } else {
            10f64.powf(-self.db() / 20.0)
        }
    }

    /// The dB as written in a config file: `0`, `0.5`, `1`, `2`, `3`.
    pub fn value(self) -> String {
        if self.0 % 10 == 0 {
            format!("{}", self.0 / 10)
        } else {
            format!("{}.{}", self.0 / 10, self.0 % 10)
        }
    }

    /// Parses dB (`0`, `0.5`, `1`, `1 dB`, `-1`; at most 20 dB, in
    /// tenths).
    pub fn parse(text: &str) -> Option<Headroom> {
        let text = text.trim();
        let text = text.strip_suffix("dB").unwrap_or(text).trim();
        let text = text.strip_prefix('-').unwrap_or(text);
        let db: f64 = text.parse().ok()?;
        let tenths = (db * 10.0).round();
        ((0.0..=200.0).contains(&tenths) && (db * 10.0 - tenths).abs() < 1e-6)
            .then_some(Headroom(tenths as u16))
    }
}

impl fmt::Display for Headroom {
    /// `none`, or `-1 dB`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            f.write_str("none")
        } else {
            write!(f, "-{} dB", self.value())
        }
    }
}

/// The least headroom a lossy file's route has (−1.5 dB): a loud master
/// decoded from MP3 or AAC peaks up to about 1 dB over full scale.
pub const LOSSY_HEADROOM: Headroom = Headroom(15);

/// The headroom of `file`'s route when it is not bit-perfect.
fn headroom_for(file: &FileFacts, settings: &Settings) -> Headroom {
    if file.decoded_from.is_some() {
        settings.headroom.max(LOSSY_HEADROOM)
    } else {
        settings.headroom
    }
}

/// The user's choices that shape a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The highest output rate to use, if capped.
    pub max_rate: Option<u32>,
    /// Resample when no rate the device takes carries the stream
    /// bit-perfect. Off: such files have no route.
    pub resample: bool,
    /// The headroom of routes that are not bit-perfect ([`Headroom`]; a
    /// lossy file's is at least [`LOSSY_HEADROOM`]); bit-perfect routes
    /// never have any.
    pub headroom: Headroom,
    /// The stems heard: all of them plays the file on its usual route, any
    /// fewer on a [`Stage::Stems`] route.
    pub stems: StemMix,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_rate: None,
            resample: true,
            headroom: Headroom::NONE,
            stems: StemMix::ALL,
        }
    }
}

/// What the policy needs to know about a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileFacts {
    /// Frames per second.
    pub rate: u32,
    /// Channels per frame.
    pub channels: u16,
    /// Bits per sample (0: unknown, taken as 24).
    pub bits: u32,
    /// Floating-point samples.
    pub float: bool,
    /// For a lossy file, the codec it is decoded from (`MP3`, `Vorbis`,
    /// ...); `None` for lossless files.
    pub decoded_from: Option<&'static str>,
}

impl From<&TrackInfo> for FileFacts {
    fn from(info: &TrackInfo) -> Self {
        let lossy = info.is_lossy();
        FileFacts {
            rate: info.rate,
            channels: info.channels,
            bits: info.bits,
            float: info.float || lossy,
            decoded_from: lossy.then(|| info.codec.name()),
        }
    }
}

/// What runs between the file and the DAC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The file's PCM, unchanged.
    Passthrough,
    /// Separated into stems at 44.1 kHz ([`crate::stems`]) and the stems
    /// that are on summed.
    Stems,
}

impl Stage {
    /// A short name for logs and UIs.
    pub fn name(self) -> String {
        match self {
            Stage::Passthrough => "PCM".into(),
            Stage::Stems => "stems".into(),
        }
    }
}

/// A sample-rate conversion after the source: `from` Hz to `to` Hz, by
/// `up / down` in lowest terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resample {
    /// The rate going in.
    pub from: u32,
    /// The rate coming out (the device's).
    pub to: u32,
    /// The ratio's numerator: `to / gcd(from, to)`.
    pub up: u32,
    /// The ratio's denominator: `from / gcd(from, to)`.
    pub down: u32,
    /// The gain below full scale inside the conversion.
    pub headroom: Headroom,
}

impl Resample {
    /// The step from `from` to `to`, without headroom.
    ///
    /// # Panics
    ///
    /// If either rate is 0.
    pub fn new(from: u32, to: u32) -> Resample {
        let (up, down) = crate::resample::reduce(from, to);
        Resample {
            from,
            to,
            up,
            down,
            headroom: Headroom::NONE,
        }
    }

    /// The same step with `headroom`.
    pub fn with_headroom(self, headroom: Headroom) -> Resample {
        Resample { headroom, ..self }
    }

    /// The ratio, and the headroom when there is some: `160/147` or
    /// `160/147, -1 dB`.
    pub fn detail(&self) -> String {
        match self.headroom {
            Headroom::NONE => format!("{}/{}", self.up, self.down),
            headroom => format!("{}/{}, {headroom}", self.up, self.down),
        }
    }
}

/// The way a track plays: what runs, at which rate, in which format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    /// What runs between the file and the DAC.
    pub stage: Stage,
    /// The file's rate.
    pub input_rate: u32,
    /// The rate the device is opened at.
    pub output_rate: u32,
    /// The channels the device is opened with.
    pub channels: u16,
    /// The format the device is expected to accept (from its
    /// capabilities).
    pub format: StreamFormat,
    /// The formats to open with, `format.sample` first: every format the
    /// device takes at this rate that holds the stream's bits.
    pub formats: Vec<SampleFormat>,
    /// The sample-rate conversion after the source, when no rate the
    /// device takes carries the stream bit-perfect.
    pub resample: Option<Resample>,
    /// Whether the samples reach the device unchanged: false for resampled
    /// routes, for files with more than 24 bits or floating-point samples,
    /// and for lossy files.
    pub lossless: bool,
    /// For a lossy file, the codec its PCM is decoded from (`MP3`,
    /// `Vorbis`, ...).
    pub decoded_from: Option<&'static str>,
    /// The gain below full scale at the route's one requantization (as in
    /// [`Resample::headroom`] when it is resampled; else the source's, for
    /// a lossy or floating-point file). None on a bit-perfect route.
    pub headroom: Headroom,
    /// The better choice that was passed over, and why (a resampled route:
    /// why it is not bit-perfect).
    pub passed_over: Option<String>,
}

impl Route {
    /// The rate out of the source (or the stem mix): the output rate, or
    /// the rate going into the resampler.
    pub fn decoded_rate(&self) -> u32 {
        self.resample.map_or(self.output_rate, |r| r.from)
    }

    /// Whether an output already open at `format` can play this route
    /// without reopening: same rate and channels, and a sample format of
    /// [`Route::formats`].
    pub fn fits(&self, format: &StreamFormat) -> bool {
        format.rate == self.output_rate
            && format.channels == self.channels
            && self.formats.contains(&format.sample)
    }

    /// The request to open the device with.
    pub fn open_request(&self) -> OpenRequest {
        OpenRequest::new(self.output_rate, self.channels).formats(&self.formats)
    }

    /// The signal path, for example `96 kHz`, or `44.1 kHz → stems
    /// 44.1 kHz` for a stem mix.
    ///
    /// A resampled route ends with the step and its ratio, for example
    /// `44.1 kHz → resample 48 kHz (160/147)`, and its headroom when it has
    /// some (`(160/147, -1 dB)`). A lossy file's route starts
    /// with what it was decoded from, for example `44.1 kHz decoded from
    /// MP3`, and ends with its headroom (`(-1.5 dB)`).
    pub fn describe(&self) -> String {
        let stages = self.describe_stages();
        match self.resample {
            Some(r) => format!("{stages} → resample {} ({})", khz(r.to), r.detail()),
            None if self.headroom != Headroom::NONE => format!("{stages} ({})", self.headroom),
            None => stages,
        }
    }

    fn describe_stages(&self) -> String {
        let input = khz(self.input_rate);
        match self.stage {
            Stage::Passthrough => match self.decoded_from {
                Some(codec) => format!("{input} decoded from {codec}"),
                None => input,
            },
            Stage::Stems => format!("{input} → stems {}", khz(self.decoded_rate())),
        }
    }
}

/// Why no route fits a file on a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoRoute {
    /// A sentence for the user.
    pub reason: String,
}

impl fmt::Display for NoRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for NoRoute {}

fn khz(rate: u32) -> String {
    if rate % 1000 == 0 {
        format!("{} kHz", rate / 1000)
    } else {
        format!("{} kHz", f64::from(rate) / 1000.0)
    }
}

/// The lossless formats for a stream with `bits` valid bits, in the
/// default preference order.
fn lossless_formats(bits: u32) -> Vec<SampleFormat> {
    SampleFormat::PREFERENCE
        .iter()
        .copied()
        .filter(|f| f.valid_bits() >= bits)
        .collect()
}

/// What the device takes, for a reason text.
fn takes(caps: &Capabilities, channels: u16) -> String {
    let rates = caps.rates(channels);
    if rates.is_empty() {
        if caps.is_empty() {
            return "the device reported no exclusive formats".into();
        }
        return format!("the device takes no {channels}-channel stream exclusively");
    }
    let list: Vec<String> = rates.iter().map(|&r| khz(r)).collect();
    format!(
        "the device takes {} with {channels} channels",
        list.join(", ")
    )
}

/// Checks one rate against the cap and the device. `Ok` holds the format
/// the device would open and the open preference; `Err` says why not.
fn fit(
    rate: u32,
    channels: u16,
    wanted: &[SampleFormat],
    caps: &Capabilities,
    settings: &Settings,
) -> Result<(StreamFormat, Vec<SampleFormat>), String> {
    if let Some(cap) = settings.max_rate {
        if rate > cap {
            return Err(format!("{} is above the {} cap", khz(rate), khz(cap)));
        }
    }
    let accepted: Vec<SampleFormat> = wanted
        .iter()
        .copied()
        .filter(|&f| caps.supports(&StreamFormat::new(rate, channels, f)))
        .collect();
    match accepted.first() {
        Some(&first) => Ok((StreamFormat::new(rate, channels, first), accepted)),
        None if !caps.formats(rate, channels).is_empty() => Err(format!(
            "the device takes {} only in formats that would lose bits",
            khz(rate)
        )),
        None => Err(format!("the device does not take {}", khz(rate))),
    }
}

/// Where a stream at `rate` would be resampled to: the lowest rate the
/// device takes with `channels` channels in a 24-bit format within the cap
/// above `rate`, else the highest one below it. `None` when the device has
/// no such rate (other than `rate` itself).
fn resample_target(
    rate: u32,
    channels: u16,
    caps: &Capabilities,
    settings: &Settings,
) -> Option<(StreamFormat, Vec<SampleFormat>)> {
    let wanted = lossless_formats(24);
    let mut fits: Vec<(StreamFormat, Vec<SampleFormat>)> = caps
        .rates(channels)
        .into_iter()
        .filter(|&r| r != rate)
        .filter_map(|r| fit(r, channels, &wanted, caps, settings).ok())
        .collect();
    match fits.iter().position(|(f, _)| f.rate > rate) {
        Some(at) => Some(fits.swap_remove(at)),
        None => fits.pop(),
    }
}

/// The end of a reason when nothing fits bit-perfect.
fn resample_note(settings: &Settings) -> &'static str {
    if settings.resample {
        "; nor any rate to resample to in a 24-bit format"
    } else {
        " (resampling is off)"
    }
}

/// Picks how `file` plays on a device that accepts `caps` exclusively, with
/// `settings`. Pure: it opens nothing.
pub fn choose_route(
    file: &FileFacts,
    caps: &Capabilities,
    settings: &Settings,
) -> Result<Route, NoRoute> {
    if file.rate == 0 || file.channels == 0 {
        return Err(NoRoute {
            reason: "the file has no sample rate or channels".into(),
        });
    }
    if !settings.stems.is_full() {
        return stems_route(file, caps, settings);
    }
    pcm_route(file, caps, settings, Stage::Passthrough)
}

/// [`choose_route`], with the output open now (`open`): a stem route plays
/// at its rate and format when the device takes them there (resampling the
/// stems to it), so moving to stems and between tracks on stems keeps the
/// output open; stems are never bit-perfect, so they follow the output
/// rather than reopen it. Any other route is [`choose_route`]'s.
pub fn choose_route_near(
    file: &FileFacts,
    caps: &Capabilities,
    settings: &Settings,
    open: Option<&StreamFormat>,
) -> Result<Route, NoRoute> {
    if !settings.stems.is_full() && file.rate > 0 {
        if let Some(route) = open.and_then(|f| stems_at(file, caps, settings, f)) {
            return Ok(route);
        }
    }
    choose_route(file, caps, settings)
}

/// The stems of `file` at the open output's format, when the device takes
/// it there (and resampling is allowed, unless it is the stems' own rate).
pub(crate) fn stems_at(
    file: &FileFacts,
    caps: &Capabilities,
    settings: &Settings,
    open: &StreamFormat,
) -> Option<Route> {
    let rate = open.rate;
    if open.channels != 2 || (rate != crate::stems::RATE && !settings.resample) {
        return None;
    }
    let (_, formats) = fit(rate, 2, &lossless_formats(24), caps, settings).ok()?;
    if !formats.contains(&open.sample) {
        return None;
    }
    Some(Route {
        stage: Stage::Stems,
        input_rate: file.rate,
        output_rate: rate,
        channels: 2,
        format: *open,
        formats,
        resample: (rate != crate::stems::RATE)
            .then(|| Resample::new(crate::stems::RATE, rate).with_headroom(settings.headroom)),
        lossless: false,
        decoded_from: None,
        headroom: if rate != crate::stems::RATE {
            settings.headroom
        } else {
            Headroom::NONE
        },
        passed_over: None,
    })
}

/// The stems of `file`: stereo at the separator's rate, played as a
/// floating-point stream of that rate would be (resampled when the device
/// does not take it).
fn stems_route(
    file: &FileFacts,
    caps: &Capabilities,
    settings: &Settings,
) -> Result<Route, NoRoute> {
    let stems = FileFacts {
        rate: crate::stems::RATE,
        channels: 2,
        bits: 24,
        float: true,
        decoded_from: None,
    };
    let mut route = pcm_route(&stems, caps, settings, Stage::Stems)?;
    route.input_rate = file.rate;
    Ok(route)
}

/// The file's PCM, or its stem mix (`stage` says which).
fn pcm_route(
    file: &FileFacts,
    caps: &Capabilities,
    settings: &Settings,
    stage: Stage,
) -> Result<Route, NoRoute> {
    let lossless = !file.float && file.bits <= 24 && file.decoded_from.is_none();
    let bits = if !lossless || file.bits == 0 {
        24
    } else {
        file.bits
    };
    let wanted = lossless_formats(bits);
    let headroom = headroom_for(file, settings);
    match fit(file.rate, file.channels, &wanted, caps, settings) {
        Ok((format, formats)) => Ok(Route {
            stage,
            input_rate: file.rate,
            output_rate: file.rate,
            channels: file.channels,
            format,
            formats,
            resample: None,
            lossless,
            decoded_from: file.decoded_from,
            // The source's requantization (a stem mix's has none).
            headroom: if lossless || stage == Stage::Stems {
                Headroom::NONE
            } else {
                headroom
            },
            passed_over: None,
        }),
        Err(why) => {
            let target = settings
                .resample
                .then(|| resample_target(file.rate, file.channels, caps, settings))
                .flatten();
            match target {
                Some((format, formats)) => Ok(Route {
                    stage,
                    input_rate: file.rate,
                    output_rate: format.rate,
                    channels: file.channels,
                    format,
                    formats,
                    resample: Some(Resample::new(file.rate, format.rate).with_headroom(headroom)),
                    lossless: false,
                    decoded_from: file.decoded_from,
                    headroom,
                    passed_over: Some(format!("{}: {why}", stage.name())),
                }),
                None => Err(NoRoute {
                    reason: format!(
                        "no bit-perfect route for this {} {} file: {why}; {}{}",
                        khz(file.rate),
                        match stage {
                            Stage::Stems => "stem mix",
                            Stage::Passthrough => file.decoded_from.unwrap_or("PCM"),
                        },
                        takes(caps, file.channels),
                        resample_note(settings)
                    ),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stems::Stem;
    use jodel_output::PROBE_RATES;
    use SampleFormat::*;

    fn device(rates: &[u32], formats: &[SampleFormat], channels: &[u16]) -> Capabilities {
        let mut caps = Capabilities::new();
        for &rate in rates {
            for &ch in channels {
                for &f in formats {
                    caps.insert(StreamFormat::new(rate, ch, f));
                }
            }
        }
        caps
    }

    /// A USB DAC that takes 48/96/192 kHz, packed 24 and 16 bit only
    /// (stereo, and 8 channels at 48/96).
    fn gc7() -> Capabilities {
        let mut caps = device(&[48_000, 96_000, 192_000], &[S16, S24Packed], &[2]);
        for rate in [48_000, 96_000] {
            for f in [S16, S24Packed] {
                caps.insert(StreamFormat::new(rate, 8, f));
            }
        }
        caps
    }

    /// An HDMI output: 44.1/48 kHz, S16, S24_3 and S24-in-32.
    fn hdmi() -> Capabilities {
        device(&[44_100, 48_000], &[S16, S24Packed, S24In32], &[2])
    }

    /// A DAC that takes everything up to 384 kHz in S32.
    fn full() -> Capabilities {
        device(&PROBE_RATES[..8], &[S32], &[1, 2])
    }

    fn pcm(rate: u32, channels: u16, bits: u32) -> FileFacts {
        FileFacts {
            rate,
            channels,
            bits,
            float: false,
            decoded_from: None,
        }
    }

    /// Bit-perfect routes only.
    fn settings(max_rate: Option<u32>) -> Settings {
        Settings {
            max_rate,
            resample: false,
            headroom: Headroom::NONE,
            stems: StemMix::ALL,
        }
    }

    fn exact() -> Settings {
        settings(None)
    }

    /// With resampling when needed.
    fn resampling(max_rate: Option<u32>) -> Settings {
        Settings {
            resample: true,
            ..settings(max_rate)
        }
    }

    /// A lossy file's route has headroom where its decoded samples are
    /// turned into 24 bits (or in the resampler): at least
    /// [`LOSSY_HEADROOM`], more when the setting asks. A floating-point
    /// file's has the setting's; a bit-perfect one none.
    #[test]
    fn lossy_and_floating_point_files_get_headroom() {
        let mp3 = FileFacts {
            rate: 44_100,
            channels: 2,
            bits: 0,
            float: true,
            decoded_from: Some("MP3"),
        };
        let s = resampling(None);
        let route = choose_route(&mp3, &full(), &s).unwrap();
        assert_eq!(route.resample, None);
        assert_eq!(route.headroom, LOSSY_HEADROOM);
        assert_eq!(route.describe(), "44.1 kHz decoded from MP3 (-1.5 dB)");
        let route = choose_route(&mp3, &gc7(), &s).unwrap();
        assert_eq!(route.resample.unwrap().headroom, LOSSY_HEADROOM);
        assert_eq!(route.headroom, LOSSY_HEADROOM);
        let more = Settings {
            headroom: Headroom(30),
            ..s
        };
        assert_eq!(
            choose_route(&mp3, &full(), &more).unwrap().headroom,
            Headroom(30)
        );
        let float = FileFacts {
            bits: 32,
            decoded_from: None,
            ..mp3
        };
        assert_eq!(
            choose_route(&float, &full(), &s).unwrap().headroom,
            Headroom::NONE
        );
        let route = choose_route(&float, &full(), &more).unwrap();
        assert_eq!(route.headroom, Headroom(30));
        assert_eq!(route.describe(), "44.1 kHz (-3 dB)");
        assert_eq!(
            choose_route(&pcm(44_100, 2, 16), &full(), &more)
                .unwrap()
                .headroom,
            Headroom::NONE
        );
    }

    /// Headroom goes on resampled routes, never on bit-perfect ones.
    #[test]
    fn headroom_is_for_resampled_routes_only() {
        let one = Headroom(10);
        let s = Settings {
            headroom: one,
            ..resampling(None)
        };
        let gc7 = gc7();
        let route = choose_route(&pcm(44_100, 2, 16), &gc7, &s).unwrap();
        assert_eq!(route.resample.unwrap().headroom, one);
        assert_eq!(
            route.describe(),
            "44.1 kHz → resample 48 kHz (160/147, -1 dB)"
        );
        // Bit-perfect routes are the same as without headroom.
        for file in [pcm(48_000, 2, 24), pcm(96_000, 2, 16)] {
            let with = choose_route(&file, &gc7, &s).unwrap();
            assert_eq!(with.resample, None);
            assert_eq!(Ok(with), choose_route(&file, &gc7, &resampling(None)));
        }
    }

    #[test]
    fn headroom_values() {
        assert_eq!(Headroom::default(), Headroom::NONE);
        assert_eq!(Headroom::NONE.gain(), 1.0);
        assert!((Headroom(10).gain() - 0.891_250_938_133_745_5).abs() < 1e-15);
        assert!((Headroom(30).gain() - 0.707_945_784_384_137_9).abs() < 1e-15);
        let values: Vec<String> = Headroom::CHOICES.iter().map(|h| h.value()).collect();
        assert_eq!(values, ["0", "0.5", "1", "2", "3"]);
        for h in Headroom::CHOICES {
            assert_eq!(Headroom::parse(&h.value()), Some(h));
        }
        assert_eq!(Headroom::parse("-1 dB"), Some(Headroom(10)));
        assert_eq!(Headroom::parse("1.5"), Some(Headroom(15)));
        assert_eq!(Headroom::parse("0.25"), None);
        assert_eq!(Headroom::parse("loud"), None);
        assert_eq!(Headroom::parse("30"), None);
        assert_eq!(Headroom(5).to_string(), "-0.5 dB");
        assert_eq!(Headroom::NONE.to_string(), "none");
    }

    #[test]
    fn gc7_plays_the_48k_family_and_refuses_the_44k_family() {
        let caps = gc7();
        let def = exact();
        let route = choose_route(&pcm(192_000, 2, 24), &caps, &def).unwrap();
        assert_eq!(route.stage, Stage::Passthrough);
        assert_eq!(route.format, StreamFormat::new(192_000, 2, S24Packed));
        // S16 would lose bits: never offered.
        assert_eq!(route.formats, [S24Packed]);
        assert!(route.lossless);
        // 16-bit 48 kHz may use S16, 44.1 kHz cannot play.
        let r = choose_route(&pcm(48_000, 2, 16), &caps, &def).unwrap();
        assert_eq!(r.stage, Stage::Passthrough);
        assert_eq!(r.formats, [S24Packed, S16]);
        let none = choose_route(&pcm(44_100, 2, 16), &caps, &def).unwrap_err();
        assert!(none.reason.contains("48 kHz, 96 kHz, 192 kHz"), "{none}");
        assert!(choose_route(&pcm(176_400, 2, 24), &caps, &def).is_err());
        // 8 channels at 96 kHz is in the matrix, at 192 kHz it is not.
        assert!(choose_route(&pcm(96_000, 8, 24), &caps, &def).is_ok());
        assert!(choose_route(&pcm(192_000, 8, 24), &caps, &def).is_err());
    }

    #[test]
    fn hdmi_prefers_the_24_bit_container() {
        let caps = hdmi();
        for rate in [44_100, 48_000] {
            let route = choose_route(&pcm(rate, 2, 24), &caps, &exact()).unwrap();
            assert_eq!(route.output_rate, rate);
            assert_eq!(route.format.sample, S24In32);
            assert_eq!(route.formats, [S24In32, S24Packed]);
        }
        let r = choose_route(&pcm(44_100, 2, 16), &caps, &exact()).unwrap();
        assert_eq!(r.format.sample, S24In32);
        assert!(choose_route(&pcm(96_000, 2, 24), &caps, &exact()).is_err());
    }

    #[test]
    fn plain_pcm_plays_at_its_own_rate_and_channels() {
        let caps = full();
        let def = exact();
        let r = choose_route(&pcm(96_000, 1, 24), &caps, &def).unwrap();
        assert_eq!((r.output_rate, r.channels), (96_000, 1));
        assert_eq!(r.resample, None);
        // A rate outside the probe list is not in the matrix.
        assert!(choose_route(&pcm(32_000, 2, 16), &caps, &def).is_err());
        let caps32 = device(&[32_000], &[S32], &[2]);
        assert_eq!(
            choose_route(&pcm(32_000, 2, 24), &caps32, &def)
                .unwrap()
                .stage,
            Stage::Passthrough
        );
        // 32-bit and float files play, flagged as not lossless.
        let wide = pcm(44_100, 2, 32);
        let r = choose_route(&wide, &caps, &def).unwrap();
        assert!(!r.lossless);
        let float = FileFacts {
            float: true,
            ..pcm(44_100, 2, 32)
        };
        assert!(!choose_route(&float, &caps, &def).unwrap().lossless);
        // The cap applies.
        let capped = settings(Some(96_000));
        assert!(choose_route(&pcm(192_000, 2, 24), &caps, &capped)
            .unwrap_err()
            .reason
            .contains("above the 96 kHz cap"));
    }

    #[test]
    fn sixteen_bit_only_and_empty_devices() {
        let s16 = device(&PROBE_RATES, &[S16], &[2]);
        let def = exact();
        let r = choose_route(&pcm(44_100, 2, 16), &s16, &def).unwrap();
        assert_eq!(r.format.sample, S16);
        assert!(choose_route(&pcm(44_100, 2, 24), &s16, &def)
            .unwrap_err()
            .reason
            .contains("lose bits"));
        let empty = Capabilities::new();
        let none = choose_route(&pcm(44_100, 2, 16), &empty, &def).unwrap_err();
        assert!(none.reason.contains("no exclusive formats"), "{none}");
        assert!(choose_route(&pcm(48_000, 2, 24), &empty, &def).is_err());
    }

    /// Every rate, depth and cap against every single-rate device: a
    /// bit-perfect route is at the file's own rate, which the device takes
    /// in a format holding its bits, within the cap; there is none only
    /// when that does not fit.
    #[test]
    fn exhaustive_policy_invariants() {
        let caps_sets: Vec<Capabilities> = PROBE_RATES[..8]
            .iter()
            .map(|&r| device(&[r], &[S24In32], &[2]))
            .chain([full(), gc7(), hdmi(), Capabilities::new()])
            .collect();
        for &rate in &PROBE_RATES[..8] {
            for bits in [16, 24] {
                for cap in [None, Some(48_000), Some(96_000), Some(192_000)] {
                    let s = settings(cap);
                    for caps in &caps_sets {
                        let file = pcm(rate, 2, bits);
                        match choose_route(&file, caps, &s) {
                            Ok(route) => {
                                assert_eq!(route.output_rate, rate);
                                assert!(caps.supports(&route.format));
                                assert!(route.format.sample.valid_bits() >= bits);
                                assert!(cap.is_none_or(|c| route.output_rate <= c));
                                assert!(route.lossless && route.resample.is_none());
                            }
                            Err(none) => {
                                assert!(!none.reason.is_empty());
                                assert!(fit(rate, 2, &lossless_formats(bits), caps, &s).is_err());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn route_helpers() {
        let route = choose_route(&pcm(352_800, 2, 24), &full(), &exact()).unwrap();
        assert_eq!(route.describe(), "352.8 kHz");
        assert!(route.fits(&StreamFormat::new(352_800, 2, S32)));
        assert!(!route.fits(&StreamFormat::new(352_800, 2, S16)));
        assert!(!route.fits(&StreamFormat::new(176_400, 2, S32)));
        let req = route.open_request();
        assert_eq!((req.rate, req.channels), (352_800, 2));
        assert_eq!(req.formats, [S32]);
        let plain = choose_route(&pcm(96_000, 2, 24), &full(), &exact()).unwrap();
        assert_eq!(plain.describe(), "96 kHz");
    }

    /// (rate in, output rate, ratio).
    fn resampled(
        file: FileFacts,
        caps: &Capabilities,
        s: Settings,
    ) -> Option<(u32, u32, (u32, u32))> {
        choose_route(&file, caps, &s).ok().map(|r| {
            let step = r.resample.expect("a resampled route");
            assert_eq!(step.from, r.decoded_rate());
            assert_eq!(step.to, r.output_rate);
            assert!(!r.lossless);
            (step.from, r.output_rate, (step.up, step.down))
        })
    }

    #[test]
    fn resampling_is_on_by_default() {
        assert!(Settings::default().resample);
    }

    #[test]
    fn gc7_resamples_the_44k_family() {
        let caps = gc7();
        let def = Settings::default();
        // The next rate up, in a 24-bit format (never S16).
        let route = choose_route(&pcm(44_100, 2, 16), &caps, &def).unwrap();
        assert_eq!(route.describe(), "44.1 kHz → resample 48 kHz (160/147)");
        assert_eq!(route.formats, [S24Packed]);
        assert_eq!(route.stage, Stage::Passthrough);
        assert!(route.passed_over.is_some());
        for (rate, out, ratio) in [
            (22_050, 48_000, (320, 147)),
            (32_000, 48_000, (3, 2)),
            (88_200, 96_000, (160, 147)),
            (176_400, 192_000, (160, 147)),
            (352_800, 192_000, (80, 147)),
            (384_000, 192_000, (1, 2)),
            (768_000, 192_000, (1, 4)),
        ] {
            assert_eq!(
                resampled(pcm(rate, 2, 24), &caps, def),
                Some((rate, out, ratio)),
                "{rate}"
            );
        }
        // Capped: the highest rate within the cap.
        assert_eq!(
            resampled(pcm(176_400, 2, 24), &caps, resampling(Some(96_000))),
            Some((176_400, 96_000, (80, 147)))
        );
        // Eight channels exist at 48 and 96 kHz only.
        assert_eq!(
            resampled(pcm(44_100, 8, 24), &caps, def),
            Some((44_100, 48_000, (160, 147)))
        );
        assert_eq!(
            resampled(pcm(192_000, 8, 24), &caps, def),
            Some((192_000, 96_000, (1, 2)))
        );
        // The 48 kHz family stays bit-perfect.
        assert!(choose_route(&pcm(48_000, 2, 24), &caps, &def)
            .unwrap()
            .resample
            .is_none());
        // No mono at all: resampling cannot help.
        let none = choose_route(&pcm(44_100, 1, 16), &caps, &def).unwrap_err();
        assert!(
            none.reason.contains("nor any rate to resample to"),
            "{none}"
        );
        // Resampling off: no route.
        let none = choose_route(&pcm(44_100, 2, 16), &caps, &exact()).unwrap_err();
        assert!(none.reason.contains("(resampling is off)"), "{none}");
    }

    #[test]
    fn hdmi_resamples_high_rates_down() {
        let caps = hdmi();
        let def = Settings::default();
        assert_eq!(
            resampled(pcm(96_000, 2, 24), &caps, def),
            Some((96_000, 48_000, (1, 2)))
        );
        assert_eq!(
            resampled(pcm(88_200, 2, 24), &caps, def),
            Some((88_200, 48_000, (80, 147)))
        );
        // The S24-in-32 container comes first for a resampled stream too.
        let route = choose_route(&pcm(32_000, 2, 16), &caps, &def).unwrap();
        assert_eq!(route.output_rate, 44_100);
        assert_eq!(route.formats, [S24In32, S24Packed]);
    }

    #[test]
    fn resampling_to_keep_the_bits() {
        // 48 kHz only in S16, 96 kHz in S24: a 24-bit file goes to 96 kHz,
        // a 16-bit one plays bit-perfect at 48 kHz.
        let mut caps = device(&[48_000], &[S16], &[2]);
        caps.insert(StreamFormat::new(96_000, 2, S24In32));
        let def = Settings::default();
        assert_eq!(
            resampled(pcm(48_000, 2, 24), &caps, def),
            Some((48_000, 96_000, (2, 1)))
        );
        assert!(choose_route(&pcm(48_000, 2, 16), &caps, &def)
            .unwrap()
            .resample
            .is_none());
        // A 16-bit-only device has nothing to resample to.
        let s16 = device(&PROBE_RATES, &[S16], &[2]);
        let none = choose_route(&pcm(44_100, 2, 24), &s16, &def).unwrap_err();
        assert!(
            none.reason.contains("nor any rate to resample to"),
            "{none}"
        );
    }

    /// With resampling on, every file at every rate against every device
    /// and cap: a bit-perfect route is unchanged; otherwise the route
    /// resamples to the rate the rule names, in a 24-bit format the device
    /// takes within the cap; and there is no route only when the device
    /// takes no such rate at all.
    #[test]
    fn exhaustive_resampling_invariants() {
        let mut caps_sets: Vec<Capabilities> = PROBE_RATES[..8]
            .iter()
            .map(|&r| device(&[r], &[S24In32], &[2]))
            .chain([full(), gc7(), hdmi(), Capabilities::new()])
            .collect();
        caps_sets.push(device(&[48_000, 96_000], &[S24Packed], &[2]));
        caps_sets.push(device(&[44_100, 88_200, 176_400], &[S32], &[2]));
        caps_sets.push(device(&[32_000, 384_000], &[S24In32], &[2]));
        let files: Vec<FileFacts> = [
            8_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400,
        ]
        .into_iter()
        .chain(PROBE_RATES[4..].iter().copied())
        .map(|r| pcm(r, 2, 24))
        .collect();
        for file in &files {
            for cap in [None, Some(48_000), Some(96_000), Some(192_000)] {
                let on = resampling(cap);
                let off = settings(cap);
                for caps in &caps_sets {
                    let usable: Vec<u32> = caps
                        .rates(2)
                        .into_iter()
                        .filter(|&r| fit(r, 2, &lossless_formats(24), caps, &on).is_ok())
                        .collect();
                    let exact = choose_route(file, caps, &off);
                    let route = choose_route(file, caps, &on);
                    if let Ok(exact) = exact {
                        assert_eq!(route, Ok(exact));
                        continue;
                    }
                    let Ok(route) = route else {
                        assert!(usable.is_empty(), "{file:?} {usable:?}");
                        continue;
                    };
                    let step = route.resample.unwrap();
                    assert!(!route.lossless);
                    assert!(caps.supports(&route.format));
                    assert!(route.format.sample.holds_24_bits());
                    assert!(cap.is_none_or(|c| route.output_rate <= c));
                    assert_eq!(step, Resample::new(step.from, route.output_rate));
                    assert_ne!(step.from, step.to);
                    assert_eq!(step.from, file.rate);
                    let below = usable.iter().copied().rfind(|&r| r != file.rate);
                    let rule = usable.iter().copied().find(|&r| r > file.rate).or(below);
                    assert_eq!(Some(route.output_rate), rule);
                }
            }
        }
    }

    #[test]
    fn a_stem_mix_plays_on_a_stems_route_and_the_whole_mix_as_before() {
        let drums = |s: Settings| Settings {
            stems: StemMix::ALL.toggle(Stem::Vocals),
            ..s
        };
        let file = pcm(44_100, 2, 24);
        // All stems: the usual route.
        let route = choose_route(&file, &full(), &exact()).unwrap();
        assert_eq!(
            (route.stage, route.output_rate),
            (Stage::Passthrough, 44_100)
        );
        // A stem out: the stems at 44.1 kHz, never lossless.
        let route = choose_route(&file, &full(), &drums(exact())).unwrap();
        assert_eq!(route.stage, Stage::Stems);
        assert_eq!(
            (route.input_rate, route.output_rate, route.channels),
            (44_100, 44_100, 2)
        );
        assert!(route.resample.is_none() && !route.lossless);
        assert_eq!(route.describe(), "44.1 kHz → stems 44.1 kHz");
        // A 96 kHz mono file: its stems are stereo at 44.1 kHz.
        let route = choose_route(&pcm(96_000, 1, 24), &full(), &drums(exact())).unwrap();
        assert_eq!(
            (route.input_rate, route.output_rate, route.channels),
            (96_000, 44_100, 2)
        );
        // A device without 44.1 kHz resamples them, or has no route without
        // resampling.
        let resampled = drums(resampling(None));
        let route = choose_route(&pcm(44_100, 2, 16), &gc7(), &resampled).unwrap();
        assert_eq!((route.stage, route.output_rate), (Stage::Stems, 48_000));
        assert_eq!(
            route.resample.map(|r| (r.from, r.to)),
            Some((44_100, 48_000))
        );
        assert_eq!(
            route.describe(),
            "44.1 kHz → stems 44.1 kHz → resample 48 kHz (160/147)"
        );
        let none = choose_route(&pcm(44_100, 2, 16), &gc7(), &drums(exact())).unwrap_err();
        assert!(none.reason.contains("stem mix"), "{none}");
        // With an output open at 176.4 kHz the stems follow it, resampled.
        let open = StreamFormat::new(176_400, 2, S32);
        let near = |s: Settings| choose_route_near(&file, &full(), &s, Some(&open));
        let route = near(drums(resampling(None))).unwrap();
        assert_eq!((route.stage, route.output_rate), (Stage::Stems, 176_400));
        assert!(route.fits(&open));
        assert_eq!(
            route.resample.map(|r| (r.from, r.to)),
            Some((44_100, 176_400))
        );
        // Not without resampling, nor for the whole mix.
        assert_eq!(near(drums(exact())).unwrap().output_rate, 44_100);
        assert_eq!(near(resampling(None)).unwrap().stage, Stage::Passthrough);
    }
}
