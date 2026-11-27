//! The shots' song being got ready ([`preparing`]): made-up results of
//! each step, told as the workers would tell them.

use std::path::PathBuf;
use std::sync::Arc;

use jodeldiplom::core::{Action, Side, State};

/// A made-up song being got ready, queued after `state`'s and selected
/// on the queue, its work at `step` (`stems`, `key`, `beat`, `words`,
/// `plan`, `story`), each result told as its worker would.
pub fn preparing(state: &State, step: &str) -> State {
    use jodel_rhythm::key::Key;
    use jodel_rhythm::waveform::Lane;
    use jodeldiplom::core::prep::{BeatsFound, Picture, Preview, Separating, Story, Words};
    use jodeldiplom::core::{update, Facts, Heard, Job, LyricsFacts, RhythmFacts, SettledLyrics};
    let mut s = state.clone();
    s.preparing = vec![Job::Rhythm, Job::Lyrics, Job::Story];
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
    done(
        &mut s,
        Job::Lyrics,
        Facts::Lyrics(SettledLyrics {
            how: LyricsFacts::Timed,
            print: 1,
        }),
    );
    if step == "words" {
        return s;
    }
    work(&mut s, Job::Story, true);
    tell(&mut s, Heard::Story(Story::Planning));
    for words in [
        "The band packs up at dawn; ",
        "the van hums the first bars ",
        "of the song",
    ] {
        tell(&mut s, Heard::Story(Story::Words(words.into())));
    }
    if step == "plan" {
        return s;
    }
    tell(&mut s, Heard::Story(Story::Reading));
    let of = 12;
    for index in 0..5 {
        tell(
            &mut s,
            Heard::Story(Story::Drawing(Picture::Scene(index + 1, of))),
        );
        let scene = image::RgbaImage::from_fn(96, 96, |x, y| {
            let shade = (40 * index) as u8;
            image::Rgba([60 + x as u8, 40 + y as u8, 160 - shade, 255])
        });
        tell(
            &mut s,
            Heard::Story(Story::Scene {
                index,
                of,
                picture: Arc::new(scene),
            }),
        );
    }
    tell(&mut s, Heard::Story(Story::Drawing(Picture::Scene(6, of))));
    s
}
