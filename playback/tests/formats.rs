//! The formats beside FLAC and WAV: AIFF, CAF, ALAC and AAC in MP4, MP3,
//! MP2, Vorbis in Ogg, and the requantization of floating-point and 32-bit PCM.
//! The fixtures in `tests/fixtures/formats` are made by its `make.py` from
//! the two signals recomputed here ([`lossless`], [`lossy`]).

#![allow(
    clippy::disallowed_methods,
    reason = "tests: tracks in each format, a tagged one and junk"
)]

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::*;
use jodel_output::SampleFormat::*;
use jodel_output::{SimClock, StreamFormat};
use jodel_playback::{
    choose_route, probe, probe_header, Codec, Container, CoverOrigin, Event, FileFacts,
    MemoryTarget, PlayState, Player, PlayerOptions, Resampler, Settings, Source, LOSSY_HEADROOM,
};

const RATE: u32 = 44_100;
const LOSSLESS_FRAMES: usize = 3_000;
const LOSSY_FRAMES: usize = 20_000;
/// The average bit rate `aac.m4a`'s descriptor states (96 kbit/s asked).
const AAC_BITRATE: u32 = 96_808;

fn format_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/formats")
        .join(name)
}

/// The lossless signal (as in `make.py`): 24-bit integers.
fn lossless(i: usize, c: usize) -> i32 {
    let t = (i + 100 * c) % 400;
    let tri = if t < 200 { t } else { 400 - t } as i64 - 100;
    let hash = ((i * 7919 + c * 104_729) as u64).wrapping_mul(2_654_435_761) & 0xffff_ffff;
    let value = tri * 40_000 + (hash >> 24) as i64 - 128;
    (if c == 0 { value } else { -value.div_euclid(2) }) as i32
}

fn lossless_samples() -> Vec<i32> {
    (0..LOSSLESS_FRAMES)
        .flat_map(|i| [lossless(i, 0), lossless(i, 1)])
        .collect()
}

/// The lossy signal (as in `make.py`), in [-0.5, 0.5].
fn lossy(i: usize, c: usize) -> f64 {
    let t = i as f64 / f64::from(RATE);
    let length = LOSSY_FRAMES as f64 / f64::from(RATE);
    let phase = 2.0 * std::f64::consts::PI * (100.0 * t + 1_900.0 * t * t / (2.0 * length));
    if c == 1 {
        0.25 * (phase + std::f64::consts::FRAC_PI_2).sin()
    } else {
        0.5 * phase.sin()
    }
}

fn read_all(path: &Path) -> (Source, Vec<i32>) {
    let mut source = Source::open(path).unwrap();
    let mut all = Vec::new();
    while source.read(&mut all).unwrap() > 0 {}
    (source, all)
}

/// A lossy file's samples as its route plays them: turned down by its
/// headroom before the requantization.
fn read_lossy(path: &Path) -> Vec<i32> {
    let mut source = Source::open(path).unwrap();
    source.set_gain(LOSSY_HEADROOM.gain());
    let mut all = Vec::new();
    while source.read(&mut all).unwrap() > 0 {}
    all
}

fn read_from(path: &Path, frame: u64) -> Vec<i32> {
    let mut source = Source::open(path).unwrap();
    assert_eq!(source.seek(frame).unwrap(), frame);
    assert_eq!(source.position(), frame);
    let mut all = Vec::new();
    while source.read(&mut all).unwrap() > 0 {}
    all
}

/// An AIFF file of interleaved samples of `bits` bits (16 or 24).
fn write_aiff(path: &Path, rate: u32, channels: u16, bits: u16, samples: &[i32]) {
    let frames = samples.len() / usize::from(channels);
    let bytes = usize::from(bits / 8);
    let mut data = Vec::with_capacity(samples.len() * bytes);
    for &s in samples {
        data.extend_from_slice(&s.to_be_bytes()[4 - bytes..]);
    }
    // The rate as an 80-bit extended float.
    let exponent = 31 - rate.leading_zeros();
    let mut extended = [0u8; 10];
    extended[..2].copy_from_slice(&(16_383 + exponent as u16).to_be_bytes());
    extended[2..].copy_from_slice(&(u64::from(rate) << (63 - exponent)).to_be_bytes());
    let mut comm = Vec::new();
    comm.extend(channels.to_be_bytes());
    comm.extend((frames as u32).to_be_bytes());
    comm.extend(bits.to_be_bytes());
    comm.extend(extended);
    let mut form = b"AIFF".to_vec();
    form.extend(b"COMM");
    form.extend((comm.len() as u32).to_be_bytes());
    form.extend(comm);
    form.extend(b"SSND");
    form.extend(((data.len() + 8) as u32).to_be_bytes());
    form.extend([0u8; 8]);
    form.extend(data);
    let mut file = b"FORM".to_vec();
    file.extend((form.len() as u32).to_be_bytes());
    file.extend(form);
    std::fs::write(path, file).unwrap();
}

/// The lag (in frames, within `range`) at which `decoded` matches `want`
/// best (left channel), and the signal-to-error ratio there in dB.
fn best_lag(decoded: &[i32], want: impl Fn(usize) -> f64, range: i64) -> (i64, f64) {
    let frames = decoded.len() / 2;
    let got: Vec<f64> = decoded
        .chunks_exact(2)
        .map(|f| f64::from(f[0]) / 8_388_608.0)
        .collect();
    let reference: Vec<f64> = (0..frames).map(&want).collect();
    let mut best = (0, f64::MIN);
    for lag in -range..=range {
        let mut dot = 0.0;
        for (i, &g) in got.iter().enumerate() {
            let j = i as i64 + lag;
            if (0..frames as i64).contains(&j) {
                dot += g * reference[j as usize];
            }
        }
        if dot > best.1 {
            best = (lag, dot);
        }
    }
    let (signal, error) = got.iter().enumerate().fold((0.0, 0.0), |(s, e), (i, &g)| {
        let r = reference[i];
        (s + r * r, e + (g - r) * (g - r))
    });
    (best.0, 10.0 * (signal / error).log10())
}

#[test]
fn lossless_formats_decode_exactly() {
    let want = lossless_samples();
    let dir = temp_dir("aiff");
    let aiff = dir.join("lossless.aiff");
    write_aiff(&aiff, RATE, 2, 24, &want);
    let cases = [
        (
            format_fixture("lossless.flac"),
            Container::Flac,
            Codec::Flac,
        ),
        (format_fixture("alac.m4a"), Container::Mp4, Codec::Alac),
        (format_fixture("alac.caf"), Container::Caf, Codec::Alac),
        (aiff.clone(), Container::Aiff, Codec::Pcm),
    ];
    let (_, flac) = read_all(&cases[0].0);
    assert!(flac == want, "the FLAC fixture holds the signal");
    for (path, container, codec) in cases {
        let (source, got) = read_all(&path);
        let info = source.info();
        let name = path.display();
        assert_eq!((info.container, info.codec), (container, codec), "{name}");
        assert_eq!(
            (info.rate, info.channels, info.bits, info.float),
            (RATE, 2, 24, false),
            "{name}"
        );
        assert_eq!(info.frames, Some(LOSSLESS_FRAMES as u64), "{name}");
        assert!(!info.is_lossy() && info.fits_24_bits(), "{name}");
        assert_eq!(source.clipped(), 0);
        assert!(
            got == flac,
            "{name} decodes to the same PCM as the FLAC file"
        );
        // Sample-accurate seeks, across ALAC's 4096-frame packets too.
        for frame in [1, 1_234, 2_999] {
            assert!(
                read_from(&path, frame) == flac[2 * frame as usize..],
                "{name} @{frame}"
            );
        }
    }
    // 16 bits: shifted up exactly.
    let sixteen: Vec<i32> = want.iter().map(|s| s >> 8).collect();
    let aiff16 = dir.join("sixteen.aiff");
    write_aiff(&aiff16, 48_000, 2, 16, &sixteen);
    let (source, got) = read_all(&aiff16);
    assert_eq!((source.info().bits, source.info().rate), (16, 48_000));
    assert!(got == sixteen.iter().map(|s| s << 8).collect::<Vec<_>>());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn alac_in_mp4_has_tags_and_its_cover() {
    let info = probe(format_fixture("alac.m4a")).unwrap();
    assert_eq!(info.tags.title.as_deref(), Some("Format Test"));
    assert_eq!(info.tags.artist.as_deref(), Some("Fixture Band"));
    assert_eq!(info.tags.album.as_deref(), Some("Formats"));
    assert_eq!(info.format_name(), "ALAC");
    let cover = info.cover.unwrap();
    assert_eq!(cover.origin, CoverOrigin::Embedded);
    assert_eq!(cover.mime, "image/png");
    assert!(cover.data.starts_with(b"\x89PNG"));
}

#[test]
fn lossy_tracks_start_and_end_exactly() {
    // The AAC bit rate is the average its descriptor states.
    for (name, codec, bitrate) in [
        ("lossy.mp3", Codec::Mp3, 96_000),
        ("lossy.ogg", Codec::Vorbis, 64_000),
        ("aac.m4a", Codec::Aac, AAC_BITRATE),
    ] {
        let path = format_fixture(name);
        let (source, got) = read_all(&path);
        let info = source.info();
        assert_eq!(info.codec, codec, "{name}");
        assert!(
            info.is_lossy() && info.float && !info.fits_24_bits(),
            "{name}"
        );
        assert_eq!(info.bitrate, Some(bitrate), "{name}");
        // The encoder delay and padding are gone: the stated length, read
        // to the frame, and the signal's own length.
        assert_eq!(info.frames, Some(LOSSY_FRAMES as u64), "{name}");
        assert_eq!(got.len(), 2 * LOSSY_FRAMES, "{name}");
        assert_eq!(source.position(), LOSSY_FRAMES as u64, "{name}");
        // The first decoded frame is the first frame of the signal.
        let (lag, snr) = best_lag(&got, |i| lossy(i, 0), 2_000);
        assert_eq!(lag, 0, "{name} is shifted by {lag} frames");
        assert!(snr > 20.0, "{name}: {snr:.1} dB");
        // The tags and the cover (ID3v2 APIC, METADATA_BLOCK_PICTURE, MP4
        // covr).
        assert_eq!(info.tags.title.as_deref(), Some("Format Test"), "{name}");
        assert_eq!(info.tags.artist.as_deref(), Some("Fixture Band"), "{name}");
        assert_eq!(info.tags.album.as_deref(), Some("Formats"), "{name}");
        let cover = info.cover.as_ref().unwrap();
        assert_eq!(cover.origin, CoverOrigin::Embedded, "{name}");
        assert!(cover.data.starts_with(b"\x89PNG"), "{name}");
        assert_eq!(source.clipped(), 0, "{name}");
    }
}

#[test]
fn lossy_seeks_are_sample_accurate() {
    for name in ["lossy.mp3", "lossy.ogg", "aac.m4a"] {
        let path = format_fixture(name);
        let (_, full) = read_all(&path);
        // Within the first packets, on and around packet edges (1024 for
        // AAC, 1152 for MP3), deep in, the last frame.
        for frame in [1u64, 577, 1_023, 1_024, 1_025, 1_152, 4_321, 12_345, 19_999] {
            let from = read_from(&path, frame);
            assert_eq!(
                from.len(),
                full.len() - 2 * frame as usize,
                "{name} @{frame}"
            );
            assert!(from == full[2 * frame as usize..], "{name} @{frame}");
        }
        // Back to the start, and to the end.
        assert!(read_from(&path, 0) == full, "{name} @0");
        assert!(
            read_from(&path, LOSSY_FRAMES as u64).is_empty(),
            "{name} @end"
        );
        // A seek after reading, on the same source.
        let mut source = Source::open(&path).unwrap();
        let mut some = Vec::new();
        for _ in 0..5 {
            source.read(&mut some).unwrap();
        }
        source.seek(3_000).unwrap();
        let mut rest = Vec::new();
        while source.read(&mut rest).unwrap() > 0 {}
        assert!(rest == full[6_000..], "{name}: seek after reads");
    }
}

#[test]
fn mpeg_layer_2_decodes() {
    let (source, got) = read_all(&format_fixture("lossy.mp2"));
    let info = source.info();
    assert_eq!((info.container, info.codec), (Container::Mpeg, Codec::Mp2));
    assert_eq!((info.channels, info.bitrate), (1, Some(64_000)));
    // No gapless header: whole frames, at least the signal's length.
    assert_eq!(got.len() % 1152, 0);
    assert!(got.len() >= 11_025);
}

/// An ID3v2.3 tag with one `TXXX` frame.
fn id3_txxx(description: &str, value: &str) -> Vec<u8> {
    let mut body = vec![0u8];
    body.extend(description.as_bytes());
    body.push(0);
    body.extend(value.as_bytes());
    let mut frame = b"TXXX".to_vec();
    frame.extend((body.len() as u32).to_be_bytes());
    frame.extend([0, 0]);
    frame.extend(body);
    let size = frame.len() as u32;
    let mut tag = b"ID3\x03\x00\x00".to_vec();
    tag.extend((0..4).rev().map(|k| ((size >> (7 * k)) & 0x7f) as u8));
    tag.extend(frame);
    tag
}

#[test]
fn lossy_files_route_as_pcm_and_read_user_tags() {
    let dir = temp_dir("lossy-tags");
    // A user-defined ID3v2 frame in front of an MP3 stream is read.
    let tagged = dir.join("tagged.mp3");
    let mut bytes = id3_txxx("ORIGINALSAMPLERATE", "96000");
    bytes.extend(std::fs::read(format_fixture("lossy.mp3")).unwrap());
    std::fs::write(&tagged, bytes).unwrap();
    let info = probe(&tagged).unwrap();
    assert_eq!(
        info.tags.get("originalsamplerate"),
        Some("96000"),
        "the tag is read: {:?}",
        info.tags.all
    );
    assert_eq!(probe_header(&tagged).unwrap().tags, info.tags);
    // The rate policy routes lossy files as PCM, never lossless.
    let facts = FileFacts::from(&info);
    assert!(facts.float);
    assert_eq!(facts.decoded_from, Some("MP3"));
    let route = choose_route(
        &facts,
        StreamFormat::new(44_100, 2, S24In32),
        &Settings::default(),
    )
    .unwrap();
    assert!(!route.lossless);
    assert_eq!(route.headroom, LOSSY_HEADROOM);
    assert_eq!(route.describe(), "44.1 kHz decoded from MP3 (-1.5 dB)");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn headers_agree_with_the_full_open() {
    for name in [
        "lossless.flac",
        "alac.m4a",
        "alac.caf",
        "lossy.mp3",
        "lossy.ogg",
        "lossy.mp2",
        "aac.m4a",
    ] {
        let path = format_fixture(name);
        let full = probe(&path).unwrap();
        let header = probe_header(&path).unwrap();
        assert_eq!(header, full, "{name}");
    }
}

#[test]
fn unsupported_codecs_say_so() {
    // HE-AAC, signalled after an LC header (which the decoder would take
    // for LC) or not at all (an LC core at half the track's rate): refused,
    // not played without its upper band.
    for (name, says) in [
        ("he-aac.m4a", "HE-AAC (AAC with SBR) is not supported"),
        ("he-aac-implicit.m4a", "this looks like HE-AAC"),
    ] {
        let path = format_fixture(name);
        for error in [
            probe(&path).unwrap_err().to_string(),
            probe_header(&path).unwrap_err().to_string(),
        ] {
            assert!(error.contains(says), "{name}: {error}");
            assert!(error.contains("only AAC-LC plays"), "{name}: {error}");
        }
    }
    let dir = temp_dir("not-audio");
    let junk = dir.join("junk.mp3");
    std::fs::write(&junk, vec![0x55u8; 4096]).unwrap();
    let error = probe(&junk).unwrap_err().to_string();
    assert!(error.contains("not a FLAC, WAV, AIFF, CAF"), "{error}");
    std::fs::remove_dir_all(dir).unwrap();
}

fn wav(path: &Path, bits: u16, float: bool, samples: impl Iterator<Item = f64>) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: bits,
        sample_format: if float {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for s in samples {
        if float {
            writer.write_sample(s as f32).unwrap();
        } else {
            writer.write_sample(s as i32).unwrap();
        }
    }
    writer.finalize().unwrap();
}

#[test]
fn wide_pcm_is_requantized_once_and_exact_where_it_fits() {
    let dir = temp_dir("wide");
    let frames = 6_000usize;
    // 32-bit integers holding 24-bit audio (low byte zero): exact.
    let padded = dir.join("padded.wav");
    let values: Vec<i32> = (0..2 * frames)
        .map(|k| (k as i32 * 7_919) % 8_000_000 - 4_000_000)
        .collect();
    wav(
        &padded,
        32,
        false,
        values.iter().map(|&v| f64::from(v) * 256.0),
    );
    let (source, got) = read_all(&padded);
    assert_eq!((source.info().bits, source.info().float), (32, false));
    assert!(!source.info().fits_24_bits());
    assert!(got == values, "24-bit audio in 32 bits passes exactly");
    let route = choose_route(
        &FileFacts::from(&**source.info()),
        StreamFormat::new(48_000, 2, S32),
        &Settings::default(),
    )
    .unwrap();
    assert!(!route.lossless, "32-bit files are not lossless at 24 bits");
    assert_eq!(route.decoded_from, None);

    // Real 32-bit detail: dithered to 24 bits, within the dither's reach,
    // with the same samples from a seek point.
    let fine = dir.join("fine.wav");
    let wide: Vec<i64> = (0..2 * frames as i64)
        .map(|k| (k * 1_000_003) % 2_000_000_000 - 1_000_000_000)
        .collect();
    wav(&fine, 32, false, wide.iter().map(|&v| v as f64));
    let (source, got) = read_all(&fine);
    let mut off = 0;
    for (&q, &v) in got.iter().zip(&wide) {
        let exact = v as f64 / 256.0;
        assert!((f64::from(q) - exact).abs() < 2.0, "{q} vs {exact}");
        if f64::from(q) != exact.floor() {
            off += 1;
        }
    }
    assert!(off > got.len() / 4, "dithered, not truncated");
    assert_eq!(source.clipped(), 0);
    assert!(read_from(&fine, 4_100) == got[2 * 4_100..]);

    // Floating point beyond full scale: saturated and counted.
    let float = dir.join("float.wav");
    let loud: Vec<f64> = (0..2 * frames)
        .map(|k| {
            if k % 1_000 == 0 {
                1.5
            } else {
                (k as f64 * 0.001).sin() * 0.5
            }
        })
        .collect();
    wav(&float, 32, true, loud.iter().copied());
    let (source, got) = read_all(&float);
    assert!(source.info().float);
    assert_eq!(source.clipped(), (2 * frames / 1_000) as u64);
    assert_eq!(got[0], 8_388_607);
    for (&q, &y) in got.iter().zip(&loud).skip(1).take(500) {
        assert!((f64::from(q) - (y as f32 as f64) * 8_388_608.0).abs() < 2.0);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn lossy_options() -> PlayerOptions {
    PlayerOptions {
        position_interval: Duration::from_millis(1),
        ..PlayerOptions::default()
    }
}

fn started(events: &[Event]) -> Vec<&jodel_playback::TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

#[test]
fn lossy_tracks_play_gaplessly_and_never_lossless() {
    let mp3 = format_fixture("lossy.mp3");
    let ogg = format_fixture("lossy.ogg");
    let aac = format_fixture("aac.m4a");
    let target = MemoryTarget::new(caps(&[44_100], &[S24In32]), SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), lossy_options());
    player
        .load(vec![
            mp3.clone(),
            ogg.clone(),
            aac.clone(),
            aac.clone(),
            mp3.clone(),
        ])
        .unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Stopped)));
    drop(player);
    let starts = started(&seen);
    assert_eq!(starts.len(), 5);
    for start in &starts {
        assert!(!start.route.lossless);
    }
    assert_eq!(starts[0].route.decoded_from, Some("MP3"));
    assert_eq!(starts[1].route.decoded_from, Some("Vorbis"));
    assert_eq!(
        starts[1].route.describe(),
        "44.1 kHz decoded from Vorbis (-1.5 dB)"
    );
    assert_eq!(starts[2].route.decoded_from, Some("AAC"));
    // One stream, each track its exact length, back to back, turned down
    // by the headroom.
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    let a = read_lossy(&mp3);
    let b = read_lossy(&ogg);
    let c = read_lossy(&aac);
    let mut want = a.clone();
    want.extend(&b);
    want.extend(&c);
    want.extend(&c);
    want.extend(&a);
    assert_eq!(streams[0].1.len(), 5 * 2 * LOSSY_FRAMES * 4);
    assert!(streams[0].1 == device_bytes(&want, S24In32));
}

#[test]
fn a_lossy_track_seeks_exactly_while_playing() {
    for name in ["lossy.mp3", "aac.m4a"] {
        let path = format_fixture(name);
        let (target, captures, gate) = GatedTarget::new(caps(&[44_100], &[S24In32]));
        let (player, events) = Player::new(Box::new(target), lossy_options());
        player.load(vec![path.clone()]).unwrap();
        gate.allow(3_333);
        player.play(0).unwrap();
        gate.wait_written(3_333);
        player.seek(15_001.0 / 44_100.0).unwrap();
        collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(&events);
        drop(player);
        let full = read_lossy(&path);
        let mut want = full[..2 * 3_333].to_vec();
        want.extend(&full[2 * 15_001..]);
        assert!(
            captured(&captures)[0].1 == device_bytes(&want, S24In32),
            "{name}"
        );
    }
}

#[test]
fn a_resampled_lossy_track_is_quantized_once() {
    let ogg = format_fixture("lossy.ogg");
    let target = MemoryTarget::new(caps(&[48_000], &[S24In32]), SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), lossy_options());
    player.load(vec![ogg.clone()]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    let start = started(&seen)[0];
    assert_eq!(
        start.route.describe(),
        "44.1 kHz decoded from Vorbis → resample 48 kHz (160/147, -1.5 dB)"
    );
    // The resampler gets the decoded samples before any requantization.
    let mut source = Source::open(&ogg).unwrap();
    let mut unquantized = Vec::new();
    while source.read_unquantized(&mut unquantized).unwrap() > 0 {}
    assert_eq!(unquantized.len(), 2 * LOSSY_FRAMES);
    let mut resampler = Resampler::new(44_100, 48_000, 2);
    resampler.set_gain(LOSSY_HEADROOM.gain());
    let mut want = Vec::new();
    resampler.process_float(&unquantized, &mut want);
    resampler.finish(&mut want);
    assert_eq!(
        want.len(),
        2 * jodel_playback::resample::output_frames(44_100, 48_000, 20_000) as usize
    );
    let streams = captured(&captures);
    assert_eq!(streams[0].0, StreamFormat::new(48_000, 2, S24In32));
    assert!(streams[0].1 == device_bytes(&want, S24In32));
}
