//! The rate policy against devices, rates, depths and caps.

use super::*;
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
    assert_eq!(route.format, StreamFormat::new(192_000, 2, S24Packed));
    // S16 would lose bits: never offered.
    assert_eq!(route.formats, [S24Packed]);
    assert!(route.lossless);
    // 16-bit 48 kHz may use S16, 44.1 kHz cannot play.
    let r = choose_route(&pcm(48_000, 2, 16), &caps, &def).unwrap();
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
    assert!(choose_route(&pcm(32_000, 2, 24), &caps32, &def).is_ok());
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
fn resampled(file: FileFacts, caps: &Capabilities, s: Settings) -> Option<(u32, u32, (u32, u32))> {
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
