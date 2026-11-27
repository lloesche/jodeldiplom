//! The shots' song being got ready ([`preparing`]): made-up results of
//! each step, told as the workers would tell them, and a made-up cover
//! taken apart ([`sample_cut`]).

use std::path::PathBuf;
use std::sync::Arc;

use jodeldiplom::core::{Action, Side, State};

/// A made-up song being got ready, queued after `state`'s and selected
/// on the queue, its work at `step` (`stems`, `key`, `beat`, `words`,
/// `cover`), each result told as its worker would.
pub fn preparing(state: &State, step: &str) -> State {
    use jodel_rhythm::key::Key;
    use jodel_rhythm::waveform::Lane;
    use jodeldiplom::core::prep::{BeatsFound, CoverCut, Preview, Separating, Story, Words};
    use jodeldiplom::core::{update, Facts, Heard, Job, LyricsFacts, RhythmFacts};
    let mut s = state.clone();
    s.preparing = vec![Job::Rhythm, Job::Lyrics, Job::Cover];
    s.side = Side::Queue;
    let path = PathBuf::from("/music/Album/09 Getting Ready.flac");
    s.details.insert(
        path.clone(),
        jodeldiplom::browser::Details {
            chip: "FLAC 44.1".into(),
            title: Some("Getting Ready".into()),
            artist: Some("The Fixtures".into()),
            seconds: Some(214.0),
            error: None,
        },
    );
    s.queue.paths.push(path.clone());
    s.queue.cursor = s.queue.paths.len() - 1;
    let tell = |s: &mut State, part: Heard| {
        update(
            s,
            Action::Heard {
                path: path.clone(),
                part,
            },
        );
    };
    let done = |s: &mut State, job: Job, facts: Facts| {
        update(
            s,
            Action::Ready {
                path: path.clone(),
                job,
                facts: Some(facts),
            },
        );
    };
    let work = |s: &mut State, job: Job, on: bool| {
        update(
            s,
            Action::Working {
                path: path.clone(),
                job,
                on,
            },
        );
    };
    // A song's loudness, frame by frame (100 a second): verses and a
    // chorus, a beat in it.
    let frames = 21_400;
    let level = |k: usize, stem: usize| -> u8 {
        let t = k as f64 / 100.0;
        let section = if (60.0..100.0).contains(&t) || t > 150.0 {
            1.0
        } else {
            0.7
        };
        let beat = (1.0 - (t * 2.0).fract()).powi(3);
        let v = match stem {
            0 => 0.55 + 0.25 * (t * 0.7).sin().abs(),
            1 => 0.35 + 0.6 * beat,
            2 => 0.5 + 0.2 * (t * 0.31).sin(),
            _ => {
                if (t * 0.25).sin() > -0.3 {
                    0.6 + 0.3 * (t * 1.3).sin().abs()
                } else {
                    0.1
                }
            }
        };
        (255.0 * (v * section).clamp(0.0, 1.0)) as u8
    };
    let lane = |stem: usize, upto: usize| Lane {
        peak: (0..upto).map(|k| level(k, stem)).collect(),
        rms: (0..upto).map(|k| level(k, stem) / 2).collect(),
    };
    let separated = |share: f64| {
        let upto = (share * frames as f64) as usize;
        let lanes = [lane(0, upto), lane(1, upto), lane(2, upto), lane(3, upto)];
        Separating::of([&lanes[0], &lanes[1], &lanes[2], &lanes[3]], frames)
    };
    work(&mut s, Job::Rhythm, true);
    // The stems separated alongside the decoding, before the key is.
    if step == "stems" {
        tell(&mut s, Heard::Separating(Arc::new(separated(0.45))));
        work(&mut s, Job::Cover, true);
        return s;
    }
    tell(&mut s, Heard::Key(Some(Key::on(2, true))));
    if step == "key" {
        return s;
    }
    tell(&mut s, Heard::Separating(Arc::new(separated(1.0))));
    tell(&mut s, Heard::Stems);
    let beats: Vec<f64> = (0..428).map(|k| 0.5 * f64::from(k)).collect();
    let analysis = jodel_rhythm::Analysis {
        key: Some(Key::on(2, true)),
        duration: 214.0,
        downbeats: (0..428).step_by(4).collect(),
        local_bpm: vec![123.0; beats.len()],
        summary: Some(jodel_rhythm::Summary {
            bpm: 123.0,
            low: 122.0,
            high: 124.0,
            steady: true,
        }),
        beats,
        ..jodel_rhythm::Analysis::default()
    };
    let mix = lane(1, frames);
    tell(
        &mut s,
        Heard::Beats(Arc::new(BeatsFound::of(&analysis, &mix))),
    );
    work(&mut s, Job::Rhythm, false);
    let facts = RhythmFacts::of(&analysis);
    done(
        &mut s,
        Job::Rhythm,
        Facts::Rhythm(RhythmFacts {
            stems: true,
            ..facts
        }),
    );
    work(&mut s, Job::Lyrics, true);
    if step == "beat" {
        tell(&mut s, Heard::Lyrics(Words::Looking));
        return s;
    }
    tell(
        &mut s,
        Heard::Lyrics(Words::Found(jodeldiplom::core::prep::Origin::Lrclib)),
    );
    tell(&mut s, Heard::Lyrics(Words::Timing));
    let mut lyrics = jodel_lyrics::Lyrics::parse(
        "[00:41.00]Pack the bags and wake the band
[00:44.00]every song is close at hand
",
        jodel_lyrics::Source::Lrclib,
    );
    for (l, line) in lyrics.lines.iter_mut().enumerate() {
        for (k, word) in line.words.iter_mut().enumerate() {
            let start = 41.0 + 3.0 * l as f64 + 0.42 * k as f64;
            (word.start, word.end) = (Some(start), Some(start + 0.36));
        }
    }
    let preview = Preview::of(&lyrics).expect("timed");
    tell(&mut s, Heard::Lyrics(Words::Timed(Arc::new(preview))));
    work(&mut s, Job::Lyrics, false);
    done(&mut s, Job::Lyrics, Facts::Lyrics(LyricsFacts::Timed));
    work(&mut s, Job::Cover, true);
    if step == "words" {
        return s;
    }
    tell(&mut s, Heard::Cover(CoverCut(Arc::new(sample_cut()))));
    work(&mut s, Job::Cover, false);
    done(&mut s, Job::Cover, Facts::Cover(true));
    if step == "cover" {
        return s;
    }
    for words in [
        "The band packs up at dawn; ",
        "the van hums the first bars ",
        "of the song",
    ] {
        tell(&mut s, Heard::Story(Story::Words(words.into())));
    }
    let scene = image::RgbaImage::from_fn(96, 96, |x, y| {
        image::Rgba([60 + x as u8, 40 + y as u8, 160, 255])
    });
    tell(
        &mut s,
        Heard::Story(Story::Scene {
            index: 0,
            of: 4,
            picture: Arc::new(scene),
        }),
    );
    s
}

/// A cover taken apart, made up: the generated cover, a disc of it as the
/// subject (its edge soft), the plate with the disc filled flat.
fn sample_cut() -> jodeldiplom::work::cover::Apart {
    use jodeldiplom::work::cover::{Apart, CoverLayers};
    use jodeldiplom::work::picture::{NearMap, Swatch};
    let side = 512;
    let cover = jodeldiplom::work::art::sample_cover(side);
    let (cx, cy, r) = (0.6 * side as f32, 0.45 * side as f32, 0.27 * side as f32);
    let inside = |x: u32, y: u32| {
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        ((r - d) / 6.0).clamp(0.0, 1.0)
    };
    let subject = image::RgbaImage::from_fn(side, side, |x, y| {
        let [r, g, b] = cover.get_pixel(x, y).0;
        image::Rgba([r, g, b, (255.0 * inside(x, y)) as u8])
    });
    let plate = image::RgbImage::from_fn(side, side, |x, y| {
        let a = inside(x, y);
        let p = cover.get_pixel(x, y).0;
        let fill = [70.0, 80.0, 150.0];
        image::Rgb(std::array::from_fn(|c| {
            (f32::from(p[c]) * (1.0 - a) + fill[c] * a) as u8
        }))
    });
    let near = NearMap {
        width: 2,
        height: 2,
        near: vec![20_000; 4],
    };
    Apart::new(
        "made-up".into(),
        CoverLayers {
            plate,
            subject: Some(subject),
            depth: near.clone(),
            plate_depth: near,
            outline: image::GrayImage::new(2, 2),
            palette: vec![Swatch {
                rgb: [70, 80, 150],
                share: 1.0,
            }],
        },
    )
}
