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
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_rate: None,
            resample: true,
            headroom: Headroom::NONE,
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
    /// Separated into stems at 44.1 kHz ([`crate::stems`]) and summed, each
    /// faded by a transition's lanes, at the open output's format.
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
    /// 44.1 kHz` for a transition's stem mix.
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
    pcm_route(file, caps, settings)
}

/// The stems of `file` at the open output's format, when the device takes
/// it there (and resampling is allowed, unless it is the stems' own rate):
/// where a transition mixes its tracks, at the output open, never
/// bit-perfect.
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

/// The file's PCM.
fn pcm_route(file: &FileFacts, caps: &Capabilities, settings: &Settings) -> Result<Route, NoRoute> {
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
            stage: Stage::Passthrough,
            input_rate: file.rate,
            output_rate: file.rate,
            channels: file.channels,
            format,
            formats,
            resample: None,
            lossless,
            decoded_from: file.decoded_from,
            // The source's requantization.
            headroom: if lossless { Headroom::NONE } else { headroom },
            passed_over: None,
        }),
        Err(why) => {
            let target = settings
                .resample
                .then(|| resample_target(file.rate, file.channels, caps, settings))
                .flatten();
            match target {
                Some((format, formats)) => Ok(Route {
                    stage: Stage::Passthrough,
                    input_rate: file.rate,
                    output_rate: format.rate,
                    channels: file.channels,
                    format,
                    formats,
                    resample: Some(Resample::new(file.rate, format.rate).with_headroom(headroom)),
                    lossless: false,
                    decoded_from: file.decoded_from,
                    headroom,
                    passed_over: Some(format!("{}: {why}", Stage::Passthrough.name())),
                }),
                None => Err(NoRoute {
                    reason: format!(
                        "no bit-perfect route for this {} {} file: {why}; {}{}",
                        khz(file.rate),
                        file.decoded_from.unwrap_or("PCM"),
                        takes(caps, file.channels),
                        resample_note(settings)
                    ),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests;
