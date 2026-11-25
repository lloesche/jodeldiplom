//! The rate policy: from what a file is, the one format its target plays
//! at and the user's settings to the [`Route`] a track plays through, or
//! the reason none fits ([`NoRoute`]).
//!
//! A target plays at one format: on a desk the desk's, on a bare device
//! the system's mix format. [`choose_route`] is a pure function (a route
//! without headroom, then [`Route::with_headroom`]):
//!
//! - **At the target's rate** a file plays as it is: lossless (a straight
//!   copy) when it is PCM of at most 24 bits in the target's channels.
//! - **At another rate** the route gains a [`Resample`] step
//!   ([`crate::resample`]) to the target's rate, and is not lossless.
//! - **Lossy files** (MP3, Vorbis, ...) are decoded to 24-bit PCM and
//!   routed as plain PCM, never lossless (the route says what it was
//!   decoded from, [`Route::decoded_from`]).
//! - A file whose channels are not the target's, where it takes stereo,
//!   is made stereo on its way ([`crate::remix::Remix`]: mono doubled,
//!   more channels folded down), and is not lossless then.
//!
//! A route that is not lossless carries the [`Settings::headroom`] at
//! its one requantization to 24 bits: inside the resampler, or where a
//! lossy or floating-point file is turned into 24-bit samples. A lossy
//! file's always has at least [`LOSSY_HEADROOM`] (its decoded signal goes
//! over full scale where the master was near it). A lossless route never
//! has any.
//!
//! A target whose format holds fewer bits than the stream (24 for a route
//! that is not lossless), or whose channels the file cannot be made into,
//! has no route for it: a [`NoRoute`] says why.

use std::fmt;

use jodel_output::{OpenRequest, StreamFormat};

use crate::remix::Remix;
use crate::source::TrackInfo;

/// A gain below full scale for routes that are not lossless, in tenths
/// of a dB (`Headroom(10)` is −1 dB): the samples are scaled by it before
/// the route's one requantization to 24 bits (in the resampler's `f64`
/// computation, or the source's for a lossy or floating-point file), so
/// overs of a near-full-scale signal (inter-sample ones, or a lossy
/// codec's) need not clip. It never touches a lossless route.
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

/// The user's choices that shape a route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    /// The headroom of routes that are not lossless ([`Headroom`]; a
    /// lossy file's is at least [`LOSSY_HEADROOM`]); lossless routes
    /// never have any.
    pub headroom: Headroom,
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
        FileFacts {
            rate: info.rate,
            channels: info.channels,
            bits: info.bits,
            float: info.float,
            decoded_from: info.is_lossy().then(|| info.codec.name()),
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
    /// The file's rate.
    pub input_rate: u32,
    /// The file's channels ([`Route::channels`] where it is not made
    /// stereo on its way).
    pub input_channels: u16,
    /// The rate the output plays at (the target's).
    pub output_rate: u32,
    /// The channels the output plays (the target's).
    pub channels: u16,
    /// The target's one format, which the output opens at.
    pub format: StreamFormat,
    /// The sample-rate conversion after the source, when the file is not
    /// at the target's rate.
    pub resample: Option<Resample>,
    /// Whether the samples reach the output unchanged: false for resampled
    /// routes, for files with more than 24 bits or floating-point samples,
    /// for files made stereo, and for lossy files.
    pub lossless: bool,
    /// For a lossy file, the codec its PCM is decoded from (`MP3`,
    /// `Vorbis`, ...).
    pub decoded_from: Option<&'static str>,
    /// The gain below full scale at the route's one requantization (as in
    /// [`Resample::headroom`] when it is resampled; else the source's, for
    /// a lossy or floating-point file). None on a lossless route.
    pub headroom: Headroom,
}

impl Route {
    /// How the file is made stereo on its way, where it is.
    pub fn remix(&self) -> Option<Remix> {
        (self.channels != self.input_channels)
            .then(|| Remix::to_stereo(self.input_channels))
            .flatten()
    }

    /// The same route with the headroom `setting` asks for at its one
    /// requantization: none on a lossless route, at least
    /// [`LOSSY_HEADROOM`] on a lossy file's.
    pub fn with_headroom(self, setting: Headroom) -> Route {
        let headroom = match (self.lossless, self.decoded_from) {
            (true, _) => Headroom::NONE,
            (false, Some(_)) => setting.max(LOSSY_HEADROOM),
            (false, None) => setting,
        };
        Route {
            resample: self.resample.map(|r| r.with_headroom(headroom)),
            headroom,
            ..self
        }
    }

    /// Whether an output already open at `format` plays this route.
    pub fn fits(&self, format: &StreamFormat) -> bool {
        self.format == *format
    }

    /// The request to open the output with.
    pub fn open_request(&self) -> OpenRequest {
        OpenRequest::new(self.output_rate, self.channels).formats(&[self.format.sample])
    }

    /// The signal path, for example `96 kHz`, or `44.1 kHz → stems
    /// 44.1 kHz` for a transition's stem mix.
    ///
    /// A resampled route ends with the step and its ratio, for example
    /// `44.1 kHz → resample 48 kHz (160/147)`, and its headroom when it has
    /// some (`(160/147, -1 dB)`). A lossy file's route starts
    /// with what it was decoded from, for example `44.1 kHz decoded from
    /// MP3`, and ends with its headroom (`(-1.5 dB)`). A file made stereo
    /// says so after its rate: `48 kHz 5.1 → stereo`.
    pub fn describe(&self) -> String {
        let stages = self.describe_stages();
        match self.resample {
            Some(r) => format!("{stages} → resample {} ({})", khz(r.to), r.detail()),
            None if self.headroom != Headroom::NONE => format!("{stages} ({})", self.headroom),
            None => stages,
        }
    }

    fn describe_stages(&self) -> String {
        let mut input = khz(self.input_rate);
        if let Some(remix) = self.remix() {
            input = format!("{input} {}", remix.describe());
        }
        match self.decoded_from {
            Some(codec) => format!("{input} decoded from {codec}"),
            None => input,
        }
    }
}

/// Why no route fits a file on a target.
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

/// Picks how `file` plays at `format` (its target's one), with
/// `settings`. Pure: it opens nothing.
pub fn choose_route(
    file: &FileFacts,
    format: StreamFormat,
    settings: &Settings,
) -> Result<Route, NoRoute> {
    route(file, format).map(|r| r.with_headroom(settings.headroom))
}

/// How `file` plays at `format`, without headroom.
fn route(file: &FileFacts, format: StreamFormat) -> Result<Route, NoRoute> {
    let refuse = |why: String| NoRoute {
        reason: format!(
            "no route for this {} {} file: {why}",
            khz(file.rate),
            file.decoded_from.unwrap_or("PCM")
        ),
    };
    if file.rate == 0 || file.channels == 0 {
        return Err(NoRoute {
            reason: "the file has no sample rate or channels".into(),
        });
    }
    let channels = format.channels;
    if file.channels != channels && channels != 2 {
        let why = format!("the device takes {channels} channels only");
        return Err(refuse(why));
    }
    let resampled = file.rate != format.rate;
    let lossless = !resampled
        && !file.float
        && file.bits <= 24
        && file.decoded_from.is_none()
        && channels == file.channels;
    let bits = if lossless && file.bits > 0 {
        file.bits
    } else {
        24
    };
    if format.sample.valid_bits() < bits {
        let why = format!("the device takes {} bits only", format.sample.valid_bits());
        return Err(refuse(why));
    }
    Ok(Route {
        input_rate: file.rate,
        input_channels: file.channels,
        output_rate: format.rate,
        channels,
        format,
        resample: resampled.then(|| Resample::new(file.rate, format.rate)),
        lossless,
        decoded_from: file.decoded_from,
        headroom: Headroom::NONE,
    })
}

#[cfg(test)]
mod tests;
