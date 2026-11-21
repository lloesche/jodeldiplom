//! A song's singing kept by Karaoke mode (`karaoke_keep`, a
//! `recordings/sing-*` folder) heard again as the game hears it: its pitch
//! frame by frame, scored at each level; and what the detection and the
//! scoring made of it, to tune them on real singing.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example sing_replay -- FOLDER [--late SECONDS] [--words] [--auto]
//! ```
//!
//! `--late`: how late the voice comes, instead of what the game took then.
//! `--words`: each word, how it was sung and scored. `--auto`: the delay
//! estimated from the singing as the game does it
//! ([`jodeldiplom::core::lag::estimate`]), every 10 s of the song at each
//! level, and the scoring at what it settled on.

use std::path::Path;
use std::sync::Arc;

use jodel_rhythm::waveform::Melody;
use jodel_rhythm::yin::{Yin, WINDOW};
use jodeldiplom::core::lag::estimate::{self, Estimator};
use jodeldiplom::core::singing::{Hit, Level, Singing};
use jodeldiplom::keep;

/// A frame heard: when in the song, its pitch (none: not sung).
type Frame = (f64, Option<f32>);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let folder = Path::new(args.first().expect("a sing-* folder"));
    let read =
        |name: &str| std::fs::read(folder.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (voice, rate) = keep::read_wav(&read(keep::VOICE)).expect("the voice");
    let marks = keep::read_times(&String::from_utf8_lossy(&read(keep::TIMES)));
    let lyrics = jodel_lyrics::Lyrics::parse(
        &String::from_utf8_lossy(&read(keep::LYRICS)),
        jodel_lyrics::Source::Unknown,
    );
    let waveform =
        jodel_rhythm::waveform::Waveform::from_bytes(&read(keep::MELODY)).expect("the waveform");
    let melody = waveform.melody.expect("a melody");
    let about = String::from_utf8_lossy(&read(keep::ABOUT)).into_owned();
    let kept_late = about
        .lines()
        .filter_map(|l| l.strip_prefix("late "))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .next_back()
        .unwrap_or(jodeldiplom::core::lag::DEFAULT_LATE);
    let late = flag(&args, "--late").unwrap_or(kept_late);
    let song = about
        .lines()
        .find_map(|l| l.strip_prefix("song "))
        .unwrap_or("?");
    println!("{song}");
    println!(
        "{:.0} s of voice at {rate} Hz, heard {:.0} ms late (kept: {:.0} ms)",
        voice.len() as f64 / f64::from(rate),
        late * 1000.0,
        kept_late * 1000.0
    );

    // The pitch, frame by frame, at the time in the song it was sung.
    let frames = pitch(&voice, rate, &marks, late);
    let lyrics = Arc::new(lyrics);

    // How much of the singing is heard as a pitch, and how steadily.
    let mut singing = Singing::default();
    singing.prepare(Path::new("song"), &lyrics, &melody, 1, Level::Hard);
    let on_notes: Vec<&Frame> = frames
        .iter()
        .filter(|f| {
            singing
                .lines
                .iter()
                .any(|l| l.notes.iter().any(|n| f.0 >= n.start && f.0 < n.end))
        })
        .collect();
    let voiced = on_notes.iter().filter(|f| f.1.is_some()).count();
    println!(
        "\nDetection: {} frames on the notes, {:.0} % with a pitch",
        on_notes.len(),
        100.0 * voiced as f64 / on_notes.len().max(1) as f64
    );
    let mut steps: Vec<f64> = Vec::new();
    let (mut octaves, mut slips) = (0, 0);
    for pair in frames.windows(2) {
        if let ((ta, Some(a)), (tb, Some(b))) = (pair[0], pair[1]) {
            if tb - ta > 0.015 {
                continue;
            }
            let d = f64::from((b - a).abs());
            steps.push(d);
            if (d - 1200.0).abs() < 150.0 {
                octaves += 1;
            } else if d > 400.0 {
                slips += 1;
            }
        }
    }
    println!(
        "  frame to frame: median {:.0} cents, 90 % under {:.0}; octave jumps {octaves}, other jumps over 4 semitones {slips} ({:.1} a second sung)",
        quantile(&mut steps.clone(), 0.5),
        quantile(&mut steps, 0.9),
        f64::from(octaves + slips) / (voiced as f64 / 100.0).max(1.0)
    );

    // When the singer comes in against when the words do: each word
    // after a pause against the nearest start of singing (a pitch after a
    // tenth of a second without), within 0.4 s; the median of those.
    let mut onsets: Vec<f64> = Vec::new();
    let mut quiet = 0usize;
    for &(t, c) in &frames {
        if c.is_some() {
            if quiet >= 10 {
                onsets.push(t);
            }
            quiet = 0;
        } else {
            quiet += 1;
        }
    }
    let mut starts: Vec<f64> = Vec::new();
    let mut last_end = f64::NEG_INFINITY;
    for n in singing.lines.iter().flat_map(|l| l.notes.iter()) {
        if n.start - last_end >= 0.2 {
            starts.push(n.start);
        }
        last_end = last_end.max(n.end);
    }
    let mut lags: Vec<f64> = starts
        .iter()
        .filter_map(|&s| {
            let i = onsets.partition_point(|&o| o < s - 0.4);
            onsets[i..]
                .iter()
                .take_while(|&&o| o <= s + 0.4)
                .min_by(|a, b| (*a - s).abs().total_cmp(&(*b - s).abs()))
                .map(|&o| o - s)
        })
        .collect();
    if lags.is_empty() {
        println!("  timing: no start of singing near a word after a pause");
    } else {
        let n = lags.len();
        let lag = quantile(&mut lags, 0.5);
        println!(
            "  timing: of {n} words after a pause, the singing comes in {:.0} ms {} them (median; 0 is in time)",
            lag.abs() * 1000.0,
            if lag >= 0.0 { "after" } else { "before" }
        );
    }

    // Scored at each level, and how far off the notes the pitch was.
    println!("\nScoring at {:.0} ms:", late * 1000.0);
    score(
        &frames,
        &lyrics,
        &melody,
        args.iter().any(|a| a == "--words"),
    );
    if args.iter().any(|a| a == "--auto") {
        auto(&pitch(&voice, rate, &marks, 0.0), &lyrics, &melody);
    }
}

/// The delay estimated from `raw` (the frames as heard, no delay taken
/// off) at each level as the song goes, then the scoring at each level's.
fn auto(raw: &[Frame], lyrics: &Arc<jodel_lyrics::Lyrics>, melody: &Melody) {
    println!("\nThe delay estimated from the singing, in ms (every 10 s of the song):");
    let mut settled = Vec::new();
    for level in Level::ALL {
        let mut s = Singing::default();
        s.prepare(Path::new("song"), lyrics, melody, 1, level);
        let mut e = Estimator::default();
        let mut next = 10.0;
        let mut said = Vec::new();
        for &(t, c) in raw {
            e.hear(&s, t, c);
            if t >= next {
                said.push(
                    e.lag
                        .map_or_else(|| "-".to_owned(), |l| format!("{:.0}", l * 1000.0)),
                );
                next += 10.0;
            }
        }
        println!("  {:<6} {}", level.name(), said.join(" "));
        // What a measure costs (the window the game holds, every 2 s).
        let window = (estimate::WINDOW * 100.0) as usize;
        let step = (estimate::EVERY * 100.0) as usize;
        let mut took: Vec<f64> = (window..raw.len())
            .step_by(step)
            .map(|end| {
                let started = std::time::Instant::now();
                let _ = estimate::measure(&s, &raw[end - window..end]);
                started.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        println!(
            "         a measure takes {:.2} ms (median), {:.2} ms at most",
            quantile(&mut took.clone(), 0.5),
            quantile(&mut took, 1.0)
        );
        // The whole song at once: how well it fits at each delay, 0 to
        // 400 ms every 20, a share of the best.
        if let Some(curve) = estimate::measure(&s, raw) {
            let top = curve.iter().copied().fold(0.0, f64::max).max(1.0);
            let shares: Vec<String> = curve
                .iter()
                .map(|h| format!("{:.0}", 100.0 * h / top))
                .collect();
            println!("         whole song, % of the best: {}", shares.join(" "));
        }
        settled.push((level, e.lag));
    }
    for (level, lag) in settled {
        let Some(lag) = lag else {
            continue;
        };
        let frames: Vec<Frame> = raw.iter().map(|&(t, c)| (t - lag, c)).collect();
        println!(
            "\nScoring at {}'s estimate, {:.0} ms:",
            level.name(),
            lag * 1000.0
        );
        score(&frames, lyrics, melody, false);
    }
}

/// `frames` scored at each level: the words hit (and the share of a word
/// each asks), how far off the notes the pitch was; with `words_too`, each
/// word on medium.
fn score(frames: &[Frame], lyrics: &Arc<jodel_lyrics::Lyrics>, melody: &Melody, words_too: bool) {
    for level in Level::ALL {
        let mut s = Singing::default();
        s.prepare(Path::new("song"), lyrics, melody, 1, level);
        for &(t, c) in frames {
            s.hear(t, c);
        }
        s.hear(f64::MAX / 4.0, None);
        let mut offs: Vec<f64> = s.heard.iter().filter_map(|h| h.off.map(f64::abs)).collect();
        println!(
            "  {:<6} ({:>2.0} % of a word) {:>3} of {} words, {} half: {:.0} % · off its note: median {:.0} cents, 75 % under {:.0}, 90 % under {:.0}",
            level.name(),
            level.share() * 100.0,
            s.hit_words,
            s.scored_words,
            s.half_words,
            100.0 * f64::from(s.hit_words) / f64::from(s.scored_words.max(1)),
            quantile(&mut offs.clone(), 0.5),
            quantile(&mut offs.clone(), 0.75),
            quantile(&mut offs, 0.9),
        );
        if level == Level::Medium && words_too {
            words(&s, lyrics);
        }
    }
}

/// Each word: when, the share of its note time with a pitch, the share
/// hit, its median distance off.
fn words(s: &Singing, lyrics: &jodel_lyrics::Lyrics) {
    let sung: Vec<&jodel_lyrics::Line> = lyrics
        .lines
        .iter()
        .filter(|l| !l.words.is_empty())
        .collect();
    for (l, notes) in s.lines.iter().enumerate() {
        let mut k = 0;
        while k < notes.notes.len() {
            let w = notes.notes[k].word;
            let (start, mut end) = (notes.notes[k].start, notes.notes[k].end);
            while k < notes.notes.len() && notes.notes[k].word == w {
                end = notes.notes[k].end;
                k += 1;
            }
            let frames = ((end - start) * 100.0).round().max(1.0);
            let heard: Vec<_> = s.heard_on(l, start, end).collect();
            let full = heard.iter().filter(|h| h.hit == Some(Hit::Full)).count();
            let mut offs: Vec<f64> = heard.iter().filter_map(|h| h.off.map(f64::abs)).collect();
            let text = sung
                .get(l)
                .and_then(|line| line.words.get(w))
                .map_or("?", |w| w.text.as_str());
            println!(
                "    {start:7.2} {:<14} pitch {:>3.0} %  hit {:>3.0} %  off {:>4.0}",
                text,
                100.0 * heard.len() as f64 / frames,
                100.0 * full as f64 / frames,
                if offs.is_empty() {
                    f64::NAN
                } else {
                    quantile(&mut offs, 0.5)
                }
            );
        }
    }
}

/// The voice's pitch frame by frame, each at the song's time it was sung.
fn pitch(voice: &[f32], rate: u32, marks: &[(usize, f64)], late: f64) -> Vec<Frame> {
    let mut out = Vec::new();
    let mut yin = Yin::new(rate);
    let mut m = 0;
    for (i, &x) in voice.iter().enumerate() {
        while m + 1 < marks.len() && marks[m + 1].0 <= i {
            m += 1;
            // A gap: the pitch starts afresh.
            let (k, t) = marks[m];
            let (k0, t0) = marks[m - 1];
            if (t - (t0 + (k - k0) as f64 / f64::from(rate))).abs() > 0.05 {
                yin = Yin::new(rate);
            }
        }
        let (k, t0) = marks.get(m).copied().unwrap_or((0, 0.0));
        yin.push(&[x], |cents| {
            let t = t0 + (i - k) as f64 / f64::from(rate) - late - WINDOW / 2.0;
            out.push((t, cents));
        });
    }
    out
}

fn flag(args: &[String], name: &str) -> Option<f64> {
    let k = args.iter().position(|a| a == name)?;
    args.get(k + 1)?.parse().ok()
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}
