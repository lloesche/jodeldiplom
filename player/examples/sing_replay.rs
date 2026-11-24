//! A song's singing kept by Karaoke mode (`karaoke_keep`, a
//! `recordings/sing-*` folder) heard again as the game hears it: its pitch
//! frame by frame, scored; and what the detection and the scoring made
//! of it, to tune them on real singing: the
//! words whose note bubbles all end up full and burst
//! ([`jodeldiplom::core::bubbles`]), and the delay in force as the game
//! follows it through the song from the one it took then (the estimate
//! from the singing correcting a measured one).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example sing_replay -- FOLDER [--late SECONDS] [--words] [--auto]
//! ```
//!
//! `--late`: how late the voice comes, instead of what the game took then.
//! Sung where the song played in another key (its `key` lines), the voice
//! is moved back into the song's own by as many semitones.
//! `--words`: each word, how it was sung and scored. `--auto`: the delay
//! estimated from the singing as the game does it
//! ([`jodeldiplom::core::lag::estimate`]), every 10 s of the song, and the
//! scoring at what it settled on.

use std::path::Path;
use std::sync::Arc;

use jodel_rhythm::waveform::Melody;
use jodel_rhythm::yin::{Yin, WINDOW};
use jodeldiplom::core::bubbles;
use jodeldiplom::core::lag::estimate::{self, Estimator};
use jodeldiplom::core::lag::{self, Lag, Source};
use jodeldiplom::core::singing::Singing;
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

    // The pitch, frame by frame, at the time in the song it was sung, in
    // the song's own key.
    let keys = keep::read_keys(&about);
    let frames = pitch(&voice, rate, &marks, &keys, late);
    let lyrics = Arc::new(lyrics);

    // How much of the singing is heard as a pitch, and how steadily.
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let on_notes: Vec<&Frame> = frames
        .iter()
        .filter(|f| {
            singing
                .notes
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
    for n in singing.notes.lines.iter().flat_map(|l| l.notes.iter()) {
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

    // Scored, and how far off the notes the pitch was.
    println!("\nScoring at {:.0} ms:", late * 1000.0);
    score(
        &frames,
        &lyrics,
        &melody,
        args.iter().any(|a| a == "--words"),
    );
    // The delay in force through the song, as the game follows it from the
    // one it took then.
    let kept_source = about
        .lines()
        .filter_map(|l| l.strip_prefix("late "))
        .filter_map(|l| l.split_whitespace().nth(2))
        .next_back()
        .map_or(Source::Default, |k| match k {
            "measured" => Source::Measured,
            "echo" => Source::Echo,
            "singing" => Source::Singing,
            _ => Source::Default,
        });
    let raw = pitch(&voice, rate, &marks, &keys, 0.0);
    follow(&raw, &lyrics, &melody, (kept_late, kept_source));
    if args.iter().any(|a| a == "--auto") {
        auto(&raw, &lyrics, &melody);
    }
}

/// The delay estimated from `raw` (the frames as heard, no delay taken
/// off) as the song goes, then the scoring at it.
fn auto(raw: &[Frame], lyrics: &Arc<jodel_lyrics::Lyrics>, melody: &Melody) {
    println!("\nThe delay estimated from the singing, in ms (every 10 s of the song):");
    let mut s = Singing::default();
    s.prepare(
        Path::new("song"),
        lyrics,
        &jodeldiplom::core::singing::melodic(melody),
    );
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
    println!("  {}", said.join(" "));
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
        "  a measure takes {:.2} ms (median), {:.2} ms at most",
        quantile(&mut took.clone(), 0.5),
        quantile(&mut took, 1.0)
    );
    // The whole song at once: how well it fits at each delay, 0 to 400 ms
    // every 20, a share of the best.
    if let Some(curve) = estimate::measure(&s, raw) {
        let top = curve.iter().copied().fold(0.0, f64::max).max(1.0);
        let shares: Vec<String> = curve
            .iter()
            .map(|h| format!("{:.0}", 100.0 * h / top))
            .collect();
        println!("  whole song, % of the best: {}", shares.join(" "));
    }
    if let Some(lag) = e.lag {
        let frames: Vec<Frame> = raw.iter().map(|&(t, c)| (t - lag, c)).collect();
        println!("\nScoring at the estimate, {:.0} ms:", lag * 1000.0);
        score(&frames, lyrics, melody, false);
    }
}

/// `frames` scored: the words hit, how far off the notes the pitch was,
/// the words whose bubbles all end up full; with `words_too`, each word.
fn score(frames: &[Frame], lyrics: &Arc<jodel_lyrics::Lyrics>, melody: &Melody, words_too: bool) {
    let mut s = Singing::default();
    s.prepare(
        Path::new("song"),
        lyrics,
        &jodeldiplom::core::singing::melodic(melody),
    );
    for &(t, c) in frames {
        s.hear(t, c);
    }
    s.finish();
    let mut offs: Vec<f64> = s.heard.iter().filter_map(|h| h.off.map(f64::abs)).collect();
    println!(
        "  ({:>2.0} % of a word) {:>3} of {} words: {:.0} % · off its note: median {:.0} cents, 75 % under {:.0}, 90 % under {:.0}",
        jodeldiplom::core::singing::SHARE * 100.0,
        s.tally.hit_words,
        s.tally.scored_words,
        100.0 * f64::from(s.tally.hit_words) / f64::from(s.tally.scored_words.max(1)),
        quantile(&mut offs.clone(), 0.5),
        quantile(&mut offs.clone(), 0.75),
        quantile(&mut offs, 0.9),
    );
    if words_too {
        words(&s, lyrics);
    }
    // The words whose bubbles all end up full, and burst.
    let (burst, all) = bubbles::words_full(&s, f64::MAX);
    println!("  bursting words: {burst} of {all}");
}

/// The delay in force as the game follows it through the song from the one
/// `kept` (as `about.txt` has it: measured, or else): the
/// estimate from `raw` (the frames as heard) correcting it
/// ([`lag::correction`]), the value in force every 10 s.
fn follow(raw: &[Frame], lyrics: &Arc<jodel_lyrics::Lyrics>, melody: &Melody, kept: (f64, Source)) {
    let mut s = Singing::default();
    s.prepare(
        Path::new("song"),
        lyrics,
        &jodeldiplom::core::singing::melodic(melody),
    );
    let (base, source) = kept;
    let (measured, echo) = match source {
        Source::Measured => (Some(base), None),
        Source::Echo => (None, Some(base)),
        _ => (None, None),
    };
    // As the game followed it (no nudge then).
    let known = lag::Known {
        measured,
        echo,
        nudge: None,
    };
    let mut lag = Lag::default();
    lag.step(base, source, 0.0, true);
    if source == Source::Singing {
        lag.estimate.lag = Some(base);
    }
    let mut said = Vec::new();
    let mut next = 10.0;
    let mut moved_at = None;
    for &(t, c) in raw {
        lag.estimate.hear(&s, t, c);
        lag.settle(known, t, false);
        if moved_at.is_none() && (lag.late - base).abs() >= 0.0095 {
            moved_at = Some(t);
        }
        if t >= next {
            said.push(format!(
                "{:.0}({})",
                lag.late * 1000.0,
                lag.estimate
                    .lag
                    .map_or("-".to_owned(), |l| format!("{:.0}", l * 1000.0))
            ));
            next += 10.0;
        }
    }
    let first = s
        .notes
        .lines
        .iter()
        .filter_map(|l| l.notes.first().map(|n| n.start))
        .fold(f64::INFINITY, f64::min);
    println!(
        "\nThe delay in force from {:.0} ms ({}), sung from {first:.0} s, every 10 s (the estimate): {}",
        base * 1000.0,
        source.key(),
        said.join(" ")
    );
    println!(
        "  the estimate settles on {}; the delay in force moves 10 ms by {}",
        lag.estimate
            .lag
            .map_or_else(|| "nothing".to_owned(), |l| format!("{:.0} ms", l * 1000.0)),
        moved_at.map_or_else(|| "never".to_owned(), |t| format!("{t:.0} s"))
    );
}

/// Each word: when, the share of its note time with a pitch, the share
/// hit, its median distance off.
fn words(s: &Singing, lyrics: &jodel_lyrics::Lyrics) {
    let sung: Vec<&jodel_lyrics::Line> = lyrics
        .lines
        .iter()
        .filter(|l| !l.words.is_empty())
        .collect();
    for (l, notes) in s.notes.lines.iter().enumerate() {
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
            let full = heard.iter().filter(|h| h.hit).count();
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

/// The voice's pitch frame by frame, each at the song's time it was sung,
/// moved back by the key the song was heard in then (`keys`: from each
/// sample on, semitones).
fn pitch(
    voice: &[f32],
    rate: u32,
    marks: &[(usize, f64)],
    keys: &[(usize, i8)],
    late: f64,
) -> Vec<Frame> {
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
        let key = keys
            .iter()
            .rev()
            .find(|&&(at, _)| at <= i)
            .map_or(0, |&(_, s)| s);
        yin.push(&[x], |cents| {
            let t = t0 + (i - k) as f64 / f64::from(rate) - late - WINDOW / 2.0;
            out.push((t, cents.map(|c| c - 100.0 * f32::from(key))));
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
