//! One frame of Karaoke mode for a song in the caches, as the GPU draws it
//! without the lane (the notes off, or a song without notes of its own: a
//! PNG of the stage's lyrics set in the middle over the plain backdrop; the
//! lines and how far each is sung printed): its lines at `SECONDS`, with
//! the notes of the line being sung from its melody. Only the caches are
//! read, never the song. `--size WxH`: the window's size (1600 × 900 by
//! default).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example karaoke_frame -- DATA_DIR LYRICS_KEY ANALYSIS_KEY SECONDS OUT.png
//! ```
//!
//! `LYRICS_KEY` names `lyrics/KEY.timed.lrc`, `ANALYSIS_KEY` names
//! `analyses/KEY.wave`.
//!
//! `--gpu FRAMES`: instead, the last `FRAMES` frames up to `SECONDS` at 60
//! a second as the GPU draws the mode with its lane (the stage over the
//! plain backdrop), each a PNG (`OUT` with the frame's number before
//! `.png`); the singer of `--sing` heard as they go.

use std::path::Path;

use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::present::Offscreen;
use jodeldiplom::gui::stage::gpu::StageFrame;
use jodeldiplom::gui::stage::{Song, Stage, Tint};
use jodeldiplom::view::karaoke::{sung_on, timed_lines, KaraokePicture, LineKind, Score};

/// A GPU drawing Karaoke mode offscreen at `w` × `h`, over the plain
/// backdrop (no cover).
fn plain_gpu((w, h): (u32, u32)) -> Offscreen {
    let mut gpu = Offscreen::new(false).expect("a GPU");
    gpu.set_plain_ground(w, h);
    gpu
}

/// `frame` drawn by `gpu` at `w` × `h` into the PNG `file`.
fn save_stage(gpu: &mut Offscreen, frame: &StageFrame, (w, h): (u32, u32), file: &Path) {
    gpu.set_stage(Some(frame), w, h);
    let shown = gpu.show_frame(w, h).expect("a frame");
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let p = shown[(y * w + x) as usize];
        image::Rgb([(p >> 16) as u8, (p >> 8) as u8, p as u8])
    });
    save_png(&img, file);
}

/// `img` as a PNG in `file`, written whole.
fn save_png(img: &image::RgbImage, file: &Path) {
    let mut png = durable::AtomicFile::create(file).expect("a PNG file");
    img.write_to(&mut png, image::ImageFormat::Png)
        .expect("a PNG");
    png.commit().expect("written");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&args[0]);
    let text = std::fs::read_to_string(data.join("lyrics").join(format!("{}.timed.lrc", args[1])))
        .expect("the song's timed lyrics");
    let body = jodeldiplom::store::cache::LYRICS_TIMED
        .body(&text)
        .unwrap_or(&text);
    let lyrics = jodel_lyrics::Lyrics::parse(body, jodel_lyrics::Source::Unknown);
    let wave = std::fs::read(data.join("analyses").join(format!("{}.wave", args[2])))
        .expect("the song's waveform");
    let waveform = jodel_rhythm::waveform::Waveform::from_bytes(&wave).expect("a waveform");
    let melody = waveform.melody.expect("a melody");
    let t: f64 = args[3].parse().unwrap();
    // `--size WxH`: the window's size (1600 × 900 by default).
    let size = args
        .iter()
        .position(|a| a == "--size")
        .and_then(|k| args.get(k + 1))
        .and_then(|s| s.split_once('x'))
        .map_or((1600, 900), |(w, h)| {
            (w.parse().expect("W"), h.parse().expect("H"))
        });
    let sung: Vec<&jodel_lyrics::Line> = lyrics
        .lines
        .iter()
        .filter(|l| !l.words.is_empty())
        .collect();
    let mut lines = timed_lines(&sung, t).expect("timed lines");
    // `--sing CENTS`: a singer who sings the melody that far off (in the
    // key they keep), heard up to `t`.
    let sing: Option<f32> = args
        .iter()
        .position(|a| a == "--sing")
        .and_then(|k| args.get(k + 1))
        .and_then(|v| v.parse().ok());
    let lyrics = std::sync::Arc::new(lyrics.clone());
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let gpu: Option<usize> = args
        .iter()
        .position(|a| a == "--gpu")
        .and_then(|k| args.get(k + 1))
        .and_then(|v| v.parse().ok());
    if let Some(frames) = gpu {
        gpu_frames(
            &args[4],
            &sung,
            &melody,
            &mut singing,
            (sing, size),
            t,
            frames,
        );
        return;
    }
    if let Some(off) = sing {
        let mut k = 0u32;
        while f64::from(k) / 100.0 <= t {
            let at = f64::from(k) / 100.0;
            let i = k as usize;
            let cents = (melody.cents.get(i).is_some_and(|&c| c > 0) && melody.sure[i] >= 128)
                .then(|| f32::from(melody.cents[i]) + off);
            singing.hear(at, cents);
            k += 1;
        }
        println!(
            "score {} ({} of {} words)",
            singing.tally.score, singing.tally.hit_words, singing.tally.scored_words,
        );
    }
    for l in lines.iter_mut().filter(|l| l.kind == LineKind::Current) {
        if let Some(i) = l.id {
            let notes = singing.notes.lines[i].clone();
            if sing.is_some() {
                sung_on(&singing, (true, 0.0), i, &notes, t, l);
            }
            l.notes = notes.notes;
        }
    }
    // `--words`: each word of the line sung, its times and what the
    // melody has under it (a frame a character: its semitone's last digit
    // where sure, `.` where not).
    if args.iter().any(|a| a == "--words") {
        if let Some(line) = lines
            .iter()
            .find(|l| l.kind == LineKind::Current)
            .and_then(|l| l.id)
            .and_then(|i| sung.get(i))
        {
            for w in &line.words {
                let (Some(s), Some(e)) = (w.start, w.end) else {
                    continue;
                };
                let frames: String = ((s * 100.0) as usize..(e * 100.0).ceil() as usize)
                    .map(|i| match (melody.cents.get(i), melody.sure.get(i)) {
                        (Some(&c), Some(&q)) if c > 0 && q >= 128 => {
                            char::from_digit(u32::from(c / 100) % 10, 10).unwrap_or('?')
                        }
                        (Some(&c), Some(_)) if c > 0 => ',',
                        _ => '.',
                    })
                    .collect();
                println!(
                    "{:>8} {s:6.2}-{e:6.2} conf {:3.0}  notes {:?}  {frames}",
                    w.text,
                    w.confidence * 100.0,
                    melody.notes_between(s, e)
                );
            }
        }
    }
    for l in &lines {
        if !l.notes.is_empty() {
            let notes: Vec<String> = l
                .notes
                .iter()
                .map(|n| format!("{:.1}-{:.1}@{:+.2}", n.from, n.to, n.height))
                .collect();
            println!("{}: {}", l.text, notes.join(" "));
        }
    }
    let picture = KaraokePicture {
        bar: 5.0 / 24.0,
        reach: 0.0,
        lines,
        count: None,
        cue: None,
        controls: None,
        next: None,
        before: Vec::new(),
        moved: 1.0,
        score: sing.map(|_| Score {
            points: singing.tally.score,
            streak: singing.tally.streak,
        }),
        result: None,
    };
    let mut gpu = plain_gpu(size);
    let song = Song::new(&singing, Vec::new(), None);
    let area = (0.0, 0.0, size.0 as f32, size.1 as f32);
    let mut stage = Stage::default();
    let frame = stage.scene((&song, false), &picture, area, (t, 0.0), &Tint::STAGE);
    save_stage(
        &mut gpu,
        &frame.expect("the lyrics"),
        size,
        Path::new(&args[4]),
    );
    // The lines, and how far each is sung.
    for line in &picture.lines {
        println!("{:>4.0} %  {}", line.fill * 100.0, line.text);
    }
}

/// `frames` frames up to `t` at 60 a second, as the GPU draws them, into
/// PNGs named after `out`.
#[allow(clippy::too_many_arguments)]
fn gpu_frames(
    out: &str,
    sung: &[&jodel_lyrics::Line],
    melody: &jodel_rhythm::waveform::Melody,
    singing: &mut Singing,
    (sing, size): (Option<f32>, (u32, u32)),
    t: f64,
    frames: usize,
) {
    let mut gpu = plain_gpu(size);
    let mut stage = Stage::default();
    let mut heard_to = 0usize;
    let mut before: Option<Vec<jodeldiplom::view::karaoke::LyricLine>> = None;
    let mut changed_at = f64::NEG_INFINITY;
    for f in 0..frames {
        let now = t - (frames - 1 - f) as f64 / 60.0;
        // The singer heard up to now (late as the microphone is).
        if let Some(off) = sing {
            while (heard_to as f64) / 100.0 <= now - 0.06 {
                let i = heard_to;
                let cents = (melody.cents.get(i).is_some_and(|&c| c > 0) && melody.sure[i] >= 128)
                    .then(|| f32::from(melody.cents[i]) + off);
                singing.hear(i as f64 / 100.0, cents);
                heard_to += 1;
            }
        }
        let lines = timed_lines(sung, now).unwrap_or_default();
        let ids: Vec<_> = lines.iter().map(|l| (l.id, l.kind)).collect();
        let was: Vec<_> = before
            .as_ref()
            .map(|b| b.iter().map(|l| (l.id, l.kind)).collect())
            .unwrap_or_default();
        if before.is_some() && ids != was && changed_at < now - 0.3 {
            changed_at = now;
        }
        let moved = ((now - changed_at) / 0.3).clamp(0.0, 1.0);
        let picture = KaraokePicture {
            bar: 5.0 / 24.0,
            reach: 0.0,
            lines: lines.clone(),
            count: None,
            cue: None,
            controls: None,
            next: None,
            before: if moved < 1.0 {
                before.clone().unwrap_or_default()
            } else {
                Vec::new()
            },
            moved,
            score: sing.map(|_| Score {
                points: singing.tally.score,
                streak: singing.tally.streak,
            }),
            result: None,
        };
        if moved >= 1.0 {
            before = Some(lines);
        }
        let song = Song::new(
            singing,
            sung.to_vec(),
            Some(melody.cents.len() as f64 / 100.0),
        );
        let area = (0.0, 0.0, size.0 as f32, size.1 as f32);
        let step = (now, 1.0 / 60.0);
        let frame = stage.scene((&song, true), &picture, area, step, &Tint::STAGE);
        let name = out.replace(".png", &format!("{f:03}.png"));
        save_stage(&mut gpu, &frame.expect("a stage"), size, Path::new(&name));
    }
    println!(
        "score {} ({} of {} words, streak {})",
        singing.tally.score,
        singing.tally.hit_words,
        singing.tally.scored_words,
        singing.tally.streak
    );
}
