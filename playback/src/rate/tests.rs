//! The rate policy against a target's one format: rates, depths, channels.

use super::*;
use jodel_output::SampleFormat::{self, *};

/// The 44.1 and 48 kHz families from 1× to 16×.
const RATE_FAMILIES: [u32; 10] = [
    44_100, 48_000, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000, 705_600, 768_000,
];

/// Stereo at `rate` in `sample`.
fn at(rate: u32, sample: SampleFormat) -> StreamFormat {
    StreamFormat::new(rate, 2, sample)
}

/// The system's mixer at 48 kHz, as a desk takes it.
fn desk() -> StreamFormat {
    at(48_000, F32)
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

fn mp3() -> FileFacts {
    FileFacts {
        rate: 44_100,
        channels: 2,
        bits: 0,
        float: true,
        decoded_from: Some("MP3"),
    }
}

fn with(headroom: Headroom) -> Settings {
    Settings { headroom }
}

/// At the target's rate a file plays as it is, a straight copy.
#[test]
fn at_the_targets_rate_the_file_plays_untouched() {
    for bits in [16, 24] {
        let route = choose_route(&pcm(48_000, 2, bits), desk(), &with(Headroom(30))).unwrap();
        assert!(route.lossless);
        assert_eq!(route.resample, None);
        assert_eq!(route.headroom, Headroom::NONE);
        assert_eq!(route.format, desk());
        assert_eq!(route.describe(), "48 kHz");
    }
    // 32-bit and float files play at the rate, not lossless.
    let wide = choose_route(&pcm(48_000, 2, 32), desk(), &Settings::default()).unwrap();
    assert!(!wide.lossless && wide.resample.is_none());
    let float = FileFacts {
        float: true,
        ..pcm(48_000, 2, 32)
    };
    let route = choose_route(&float, desk(), &with(Headroom(30))).unwrap();
    assert!(!route.lossless);
    assert_eq!(route.headroom, Headroom(30));
    assert_eq!(route.describe(), "48 kHz (-3 dB)");
}

/// At another rate the route resamples to the target's.
#[test]
fn at_another_rate_the_file_is_resampled_to_the_targets() {
    let def = Settings::default();
    for (rate, ratio) in [
        (22_050, (320, 147)),
        (32_000, (3, 2)),
        (44_100, (160, 147)),
        (88_200, (80, 147)),
        (96_000, (1, 2)),
        (192_000, (1, 4)),
        (768_000, (1, 16)),
    ] {
        let route = choose_route(&pcm(rate, 2, 24), desk(), &def).unwrap();
        let step = route.resample.expect("resampled");
        let got = (step.from, step.to, (step.up, step.down));
        assert_eq!(got, (rate, 48_000, ratio));
        assert_eq!(route.output_rate, 48_000);
        assert!(!route.lossless);
    }
    let route = choose_route(&pcm(44_100, 2, 16), desk(), &def).unwrap();
    assert_eq!(route.describe(), "44.1 kHz → resample 48 kHz (160/147)");
    let route = choose_route(&pcm(44_100, 2, 16), desk(), &with(Headroom(10))).unwrap();
    assert_eq!(route.resample.unwrap().headroom, Headroom(10));
    assert_eq!(
        route.describe(),
        "44.1 kHz → resample 48 kHz (160/147, -1 dB)"
    );
}

/// A lossy file's route has headroom where its decoded samples are turned
/// into 24 bits (or in the resampler): at least [`LOSSY_HEADROOM`], more
/// when the setting asks.
#[test]
fn lossy_files_get_headroom() {
    let route = choose_route(&mp3(), at(44_100, S32), &Settings::default()).unwrap();
    assert_eq!(route.resample, None);
    assert!(!route.lossless);
    assert_eq!(route.headroom, LOSSY_HEADROOM);
    assert_eq!(route.describe(), "44.1 kHz decoded from MP3 (-1.5 dB)");
    let route = choose_route(&mp3(), desk(), &Settings::default()).unwrap();
    assert_eq!(route.resample.unwrap().headroom, LOSSY_HEADROOM);
    let more = choose_route(&mp3(), desk(), &with(Headroom(30))).unwrap();
    assert_eq!(more.headroom, Headroom(30));
}

/// New headroom on a route already chosen is the route the settings choose
/// with it.
#[test]
fn headroom_on_a_route_is_the_route_with_it() {
    for file in [pcm(44_100, 2, 16), pcm(48_000, 2, 24), mp3()] {
        let plain = choose_route(&file, desk(), &Settings::default()).unwrap();
        for headroom in Headroom::CHOICES {
            let want = choose_route(&file, desk(), &with(headroom)).unwrap();
            assert_eq!(plain.clone().with_headroom(headroom), want);
        }
    }
}

/// Mono doubled, more channels folded down, on a stereo target.
#[test]
fn files_are_made_stereo_for_a_stereo_target() {
    let def = Settings::default();
    let mono = choose_route(&pcm(48_000, 1, 16), desk(), &def).unwrap();
    assert_eq!((mono.input_channels, mono.channels), (1, 2));
    assert!(!mono.lossless);
    assert_eq!(mono.describe(), "48 kHz mono → stereo");
    let surround = choose_route(&pcm(44_100, 6, 24), desk(), &def).unwrap();
    assert_eq!((surround.input_channels, surround.channels), (6, 2));
    assert!(surround.remix().is_some() && surround.resample.is_some());
}

/// A 16-bit target plays 16-bit files at its rate, and nothing that needs
/// more bits: a 24-bit file, a resampled or a lossy one.
#[test]
fn a_sixteen_bit_target_refuses_what_would_lose_bits() {
    let s16 = at(44_100, S16);
    let def = Settings::default();
    let route = choose_route(&pcm(44_100, 2, 16), s16, &def).unwrap();
    assert!(route.lossless);
    assert_eq!(route.format.sample, S16);
    for file in [pcm(44_100, 2, 24), pcm(48_000, 2, 16), mp3()] {
        let none = choose_route(&file, s16, &def).unwrap_err();
        assert!(none.reason.contains("16 bits only"), "{none}");
    }
    let zero = choose_route(&pcm(0, 2, 16), s16, &def).unwrap_err();
    assert!(zero.reason.contains("no sample rate"), "{zero}");
}

/// Every rate and depth against every rate of a target: lossless exactly
/// at the target's rate, else resampled to it; refused only where the
/// format holds too few bits.
#[test]
fn exhaustive_policy_invariants() {
    for &target in &RATE_FAMILIES {
        for sample in [S16, S24Packed, S24In32, S32, F32] {
            let format = at(target, sample);
            for &rate in RATE_FAMILIES.iter().chain(&[8_000, 22_050, 32_000]) {
                for bits in [16, 24] {
                    match choose_route(&pcm(rate, 2, bits), format, &Settings::default()) {
                        Ok(route) => {
                            assert_eq!(route.format, format);
                            assert_eq!(route.lossless, rate == target);
                            match route.resample {
                                Some(step) => {
                                    assert_eq!(step, Resample::new(rate, target));
                                    assert!(sample.holds_24_bits());
                                }
                                None => assert_eq!(rate, target),
                            }
                        }
                        Err(none) => {
                            let need = if rate == target { bits } else { 24 };
                            assert!(sample.valid_bits() < need, "{none}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn route_helpers() {
    let format = at(352_800, S32);
    let route = choose_route(&pcm(352_800, 2, 24), format, &Settings::default()).unwrap();
    assert_eq!(route.describe(), "352.8 kHz");
    assert!(route.fits(&format));
    assert!(!route.fits(&at(352_800, S16)));
    assert!(!route.fits(&at(176_400, S32)));
    let req = route.open_request();
    assert_eq!((req.rate, req.channels), (352_800, 2));
    assert_eq!(req.formats, [S32]);
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
