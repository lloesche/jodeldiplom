//! Analyses the beats of audio files and reports them.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example beats -- [--methods flux,network] [--no-stems] [--beats] [--levels] [--presence] [--clicks OUT.wav] [--stems DIR] FILE...
//! cargo run --release -p jodeldiplom --example beats -- [--methods flux,network] [--no-stems] --truth LIST [--fit | --compare]
//! ```
//!
//! For each file: the tempo, how sure the analysis is, the candidate grid
//! it chose and how the beats group its pulse, and every candidate's
//! tempo and agreement. `--methods` picks the beat methods that vote
//! (default: all). `--beats` lists every beat (`|` marks a downbeat)
//! with its local tempo. `--levels` prints what each cue of the meter
//! said about each grouping, averaged over the track. `--clicks OUT.wav`
//! writes the (last) file as 16-bit WAV with a click on every beat, a
//! higher one on downbeats, to check the beats by ear. It plays nothing.
//! `--wave` prints, every 10 seconds, each lane's peak level (the mix and
//! each stem), as the Mix tab draws them. `--presence` shows, every 10 seconds, how present the drum and bass
//! stems are and how much the drum stem's candidate has a say there.
//!
//! The cues listen to stems separated on the GPU when there is one, as in
//! the player; `--no-stems` makes them listen to the mix. `--stems DIR`
//! writes each file's four stems (bass, drums, other, vocals) to DIR as
//! 16-bit WAV at 44.1 kHz, to hear the separation.
//!
//! `--truth LIST` analyses the tracks of a list and compares their tempo
//! with the one given: one track per line, a tempo, a tab and the path.
//! A plain tempo (`56`) was tapped by a listener, `?125` is believed but
//! not confirmed, and `-` only reports; `#` starts a comment. It prints
//! each track (ours, the truth, the verdict) and how many are right
//! (within 4 %), with the factor each wrong one is off by. `--fit` then
//! searches the meter's weights for the most right tempi (a confirmed
//! one counts three times a believed one) and reports with the best.
//! `--compare` scores the list without and with the drum stem's
//! candidate; it lists the tracks whose verdict changes and how the drum
//! stem's candidate did.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use jodel_gpu::GpuSeparator;
use jodel_playback::Source;
use jodel_playback::StereoFeed;
use jodel_rhythm::activation::Curve;
use jodel_rhythm::meter::{self, Weights};
use jodel_rhythm::stems::{self, Separator, Splitter, Stem};
use jodel_rhythm::vote::Candidate;
use jodel_rhythm::{vote, Analysis, Evidence, Method};
use jodeldiplom::core::RhythmRecipe;
use jodeldiplom::rhythm::listen_file;

fn main() -> ExitCode {
    let mut files = Vec::new();
    let (mut list, mut levels, mut fit) = (false, false, false);
    let (mut show_presence, mut compare, mut wave) = (false, false, false);
    let mut clicks: Option<PathBuf> = None;
    let mut truth: Option<PathBuf> = None;
    let mut stem_dir: Option<PathBuf> = None;
    let mut use_stems = true;
    let mut methods = Method::ALL.to_vec();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--beats" => list = true,
            "--levels" => levels = true,
            "--fit" => fit = true,
            "--compare" => compare = true,
            "--presence" => show_presence = true,
            "--wave" => wave = true,
            "--clicks" => clicks = args.next().map(PathBuf::from),
            "--stems" => stem_dir = args.next().map(PathBuf::from),
            "--no-stems" => use_stems = false,
            "--truth" => truth = args.next().map(PathBuf::from),
            "--methods" => {
                let ids = args.next().unwrap_or_default();
                match ids.split(',').map(|id| Method::parse(id.trim())).collect() {
                    Some(m) => methods = m,
                    None => {
                        eprintln!("--methods takes flux, network or both: {ids}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            _ => files.push(PathBuf::from(arg)),
        }
    }
    let recipe = RhythmRecipe {
        methods,
        stems: use_stems,
    };
    if let Some(list) = truth {
        let run = if compare {
            compare_setups(&list, &recipe)
        } else {
            score(&list, &recipe, fit)
        };
        return match run {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{}: {e}", list.display());
                ExitCode::FAILURE
            }
        };
    }
    if files.is_empty() {
        eprintln!(
            "usage: beats [--methods flux,network] [--no-stems] [--beats] [--levels] [--presence] [--clicks OUT.wav] [--stems DIR] FILE...\n       beats [--methods flux,network] [--no-stems] --truth LIST [--fit | --compare]"
        );
        return ExitCode::FAILURE;
    }
    let mut last = None;
    for file in &files {
        let started = Instant::now();
        match listen_file(file, &recipe) {
            Ok(listened) => {
                let weights = Weights::for_parts(listened.evidence.stems);
                if let Some(e) = &listened.stems_failed {
                    eprintln!(
                        "{}: stems failed ({e}); the cues listened to the mix",
                        file.display()
                    );
                }
                let a = listened.evidence.decide(&weights);
                report(file, &a, started.elapsed().as_secs_f64());
                if levels {
                    explain(&listened.evidence, &weights);
                }
                if show_presence {
                    presence(&listened.evidence);
                }
                if wave {
                    wave_levels(&listened.waveform);
                }
                if list {
                    beats(&a);
                }
                last = Some((file.clone(), a));
            }
            Err(e) => eprintln!("{}: {e}", file.display()),
        }
    }
    if let Some(dir) = stem_dir {
        for file in &files {
            match write_stems(file, &dir) {
                Ok(written) => println!("wrote {written}"),
                Err(e) => {
                    eprintln!("{}: {e}", file.display());
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    if let (Some(out), Some((file, a))) = (clicks, last) {
        if let Err(e) = write_clicks(&file, &a, &out) {
            eprintln!("{}: {e}", out.display());
            return ExitCode::FAILURE;
        }
        println!("wrote {}", out.display());
    }
    ExitCode::SUCCESS
}

fn report(file: &Path, a: &Analysis, seconds: f64) {
    let name = file
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let tempo = match a.summary {
        Some(s) if s.steady => format!("{:.2} BPM steady", s.bpm),
        Some(s) => format!("{:.1} BPM ({:.1}–{:.1})", s.bpm, s.low, s.high),
        None => "no beat".into(),
    };
    let bars = if a.downbeats.len() >= 2 {
        let spans: Vec<usize> = a.downbeats.windows(2).map(|w| w[1] - w[0]).collect();
        let fours = spans.iter().filter(|&&s| s == 4).count();
        let threes = spans.iter().filter(|&&s| s == 3).count();
        format!("bars: {fours}×4 {threes}×3")
    } else {
        "bars: –".into()
    };
    let weak = a.regions.iter().filter(|r| r.confidence < 0.5).count();
    println!(
        "{name}\n  {tempo}, confidence {:.2} (weak regions {weak}/{}), {bars}, {} beats in {:.0} s, analysed in {seconds:.2} s\n  chosen {} {}, cues from {}",
        a.confidence,
        a.regions.len(),
        a.beats.len(),
        a.duration,
        a.chosen,
        meter::describe(&a.groupings, a.duration),
        if a.stems { "stems" } else { "the mix" }
    );
    for c in &a.candidates {
        println!(
            "    {:14} {:7.2} BPM  agreement {:.2}",
            c.name, c.bpm, c.agreement
        );
    }
}

/// Every 10 seconds: each lane's highest peak there, in dBFS.
fn wave_levels(w: &jodel_rhythm::waveform::Waveform) {
    let db = |v: f32| {
        if v <= 0.0 {
            "    -".to_owned()
        } else {
            format!(
                "{:5.0}",
                f64::from(v) * -jodel_rhythm::waveform::FLOOR_DB + jodel_rhythm::waveform::FLOOR_DB
            )
        }
    };
    println!("       s    mix  drums   bass  other vocals   (peak dBFS)");
    let mut t = 0.0;
    while t < w.duration() {
        let end = t + 10.0;
        let mut row = format!("    {t:4.0}  {}", db(w.mix.peak_between(t, end)));
        if let Some(stems) = &w.stems {
            // Shown as drums, bass, other, vocals.
            for i in [1, 0, 2, 3] {
                row.push_str(&format!("  {}", db(stems[i].peak_between(t, end))));
            }
        }
        println!("{row}");
        t = end;
    }
}

/// Every 10 seconds: how present the drums and the bass are, and the say
/// of the drum stem's candidate there.
fn presence(evidence: &Evidence) {
    let say = |prefix: &str| {
        evidence
            .candidates
            .iter()
            .find(|c| c.name.starts_with(prefix))
            .and_then(|c| c.say.clone())
    };
    let drums_say = say("drums/");
    let (Some(drums), Some(bass)) = (&evidence.cues.drums_present, &evidence.cues.bass_present)
    else {
        println!("    no stems");
        return;
    };
    let mean = |c: &Option<Curve>, a: f64, b: f64| {
        c.as_ref()
            .map_or(0.0, |c| jodel_rhythm::presence::mean_between(c, a, b))
    };
    println!("       s  drums present say   bass present");
    let mut t = 0.0;
    while t < evidence.duration {
        let end = t + 10.0;
        println!(
            "    {t:4.0}        {:5.2} {:5.2}         {:5.2}",
            jodel_rhythm::presence::mean_between(drums, t, end),
            mean(&drums_say, t, end),
            jodel_rhythm::presence::mean_between(bass, t, end),
        );
        t = end;
    }
}

/// What each cue of the meter said about each grouping of the chosen
/// grid, averaged over the track, best total first.
fn explain(evidence: &Evidence, weights: &Weights) {
    let Some(outcome) = vote::vote(&evidence.candidates) else {
        return;
    };
    let grid = &evidence.candidates[outcome.chosen].beats;
    let mut rows = meter::explain(grid, &evidence.candidates, &evidence.cues, weights);
    rows.sort_by(|a, b| b.1.total(weights).total_cmp(&a.1.total(weights)));
    println!("    grouping  total  backbeat accent change candidates preference");
    for (g, s) in rows.iter().take(8) {
        println!(
            "    {:3} /{}   {:6.3}   {:6.3} {:6.3} {:6.3}     {:6.3}     {:6.3}",
            g.label(),
            g.phase,
            s.total(weights),
            s.backbeat,
            s.accent,
            s.change,
            s.candidates,
            s.preference
        );
    }
}

/// A line of a truth list.
struct Entry {
    name: String,
    /// The tempo, if one is given.
    tempo: Option<f64>,
    /// Whether it is believed rather than confirmed.
    believed: bool,
    evidence: Evidence,
}

/// The tracks of the truth list at `path`, listened to.
fn read_truth(path: &Path, recipe: &RhythmRecipe) -> Result<Vec<Entry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut entries = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((tempo, file)) = line.split_once('\t') else {
            return Err(format!("no tab in: {line}"));
        };
        let (believed, tempo) = match tempo.trim() {
            "-" => (false, None),
            t => match t.strip_prefix('?') {
                Some(t) => (true, t.parse::<f64>().ok()),
                None => (false, t.parse::<f64>().ok()),
            },
        };
        let file = PathBuf::from(file.trim());
        let name = file
            .file_stem()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        match listen_file(&file, recipe) {
            Ok(listened) => entries.push(Entry {
                name,
                tempo,
                believed,
                evidence: listened.evidence,
            }),
            Err(e) => eprintln!("  error  {name}: {e}"),
        }
    }
    Ok(entries)
}

/// `ours` against `truth` in words: `ok` within 4 %, else the factor it
/// is off by, or `off`.
fn verdict(ours: f64, truth: f64) -> &'static str {
    const FACTORS: [(f64, &str); 7] = [
        (1.0, "ok"),
        (2.0, "×2"),
        (0.5, "×½"),
        (3.0, "×3"),
        (1.0 / 3.0, "×⅓"),
        (4.0, "×4"),
        (0.25, "×¼"),
    ];
    let ratio = ours / truth;
    FACTORS
        .iter()
        .find(|(f, _)| (ratio / f - 1.0).abs() <= 0.04)
        .map_or("off", |(_, word)| word)
}

/// The tempo of an analysis, 0 without one.
fn tempo(a: &Analysis) -> f64 {
    a.summary.map_or(0.0, |s| s.bpm)
}

/// How a set of weights does on a truth list.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Judgement {
    /// Confirmed tempi right.
    confirmed: usize,
    /// Believed tempi right.
    believed: usize,
    /// The mean share of each track's time in its main grouping.
    steadiness: f64,
}

impl Judgement {
    /// A confirmed tempo counts three times a believed one.
    fn value(&self) -> usize {
        3 * self.confirmed + self.believed
    }

    /// More right tempi, then steadier groupings.
    fn beats(&self, other: &Judgement) -> bool {
        self.value() > other.value()
            || (self.value() == other.value() && self.steadiness > other.steadiness + 1e-9)
    }
}

/// The share of `a`'s time in its most common grouping.
fn steadiness(a: &Analysis) -> f64 {
    let mut shares: Vec<(usize, f64)> = Vec::new();
    for (i, s) in a.groupings.iter().enumerate() {
        let until = a.groupings.get(i + 1).map_or(a.duration, |n| n.start);
        let key = s.grouping.pulses;
        match shares.iter_mut().find(|(k, _)| *k == key) {
            Some((_, t)) => *t += until - s.start,
            None => shares.push((key, until - s.start)),
        }
    }
    let whole: f64 = shares.iter().map(|s| s.1).sum();
    let most = shares.iter().map(|s| s.1).fold(0.0, f64::max);
    if whole > 0.0 {
        most / whole
    } else {
        1.0
    }
}

/// How `entries` do with `weights`.
fn judge(entries: &[Entry], weights: &Weights) -> Judgement {
    let mut j = Judgement::default();
    for e in entries {
        let a = e.evidence.decide(weights);
        j.steadiness += steadiness(&a) / entries.len() as f64;
        let Some(truth) = e.tempo else { continue };
        if verdict(tempo(&a), truth) == "ok" {
            if e.believed {
                j.believed += 1;
            } else {
                j.confirmed += 1;
            }
        }
    }
    j
}

/// The weights fitted for the entries' parts (stems or the mix).
fn fitted(entries: &[Entry]) -> Weights {
    Weights::for_parts(entries.first().is_some_and(|e| e.evidence.stems))
}

/// The weights with the most right tempi (then the steadiest groupings),
/// by coordinate search from the fitted ones: each weight in turn tries a
/// few values, keeping the best, until a round changes nothing.
fn fit_weights(entries: &[Entry]) -> Weights {
    let mut best = fitted(entries);
    let mut best_j = judge(entries, &best);
    println!("defaults: {best_j:?}");
    type Field = fn(&mut Weights) -> &mut f64;
    let fields: [(&str, Field, &[f64]); 7] = [
        (
            "backbeat",
            |w| &mut w.backbeat,
            &[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        ),
        (
            "accent",
            |w| &mut w.accent,
            &[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        ),
        (
            "change",
            |w| &mut w.change,
            &[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        ),
        (
            "candidates",
            |w| &mut w.candidates,
            &[0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
        ),
        (
            "preference",
            |w| &mut w.preference,
            &[0.0, 0.15, 0.3, 0.6, 1.0, 1.5],
        ),
        (
            "center_bpm",
            |w| &mut w.center_bpm,
            &[90.0, 100.0, 110.0, 120.0, 130.0],
        ),
        (
            "switch",
            |w| &mut w.switch,
            &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        ),
    ];
    for round in 0..6 {
        let before = best_j;
        for (name, field, values) in &fields {
            for &v in *values {
                let mut w = best;
                *field(&mut w) = v;
                let got = judge(entries, &w);
                if got.beats(&best_j) {
                    best_j = got;
                    best = w;
                    println!("round {round}: {name} = {v} → {got:?}");
                }
            }
        }
        if best_j == before {
            break;
        }
    }
    println!("best: {best:?}");
    best
}

/// The evidence with only the candidates `keep` keeps.
fn keeping(evidence: &Evidence, keep: fn(&Candidate) -> bool) -> Evidence {
    let mut e = evidence.clone();
    e.candidates.retain(keep);
    e
}

/// A way to score a truth list: which candidates vote.
type Setup = (&'static str, fn(&Candidate) -> bool);

/// Scores the truth list at `path` without and with the drum stem's
/// candidate, lists the tracks whose result changes, and the drum stem
/// candidate's mean agreement and say.
fn compare_setups(path: &Path, recipe: &RhythmRecipe) -> Result<(), String> {
    let entries = read_truth(path, recipe)?;
    let base = fitted(&entries);
    let setups: [Setup; 2] = [
        ("no drum stem candidate", |c| !c.is_partial()),
        ("drum stem candidate", |_| true),
    ];
    let mut verdicts: Vec<Vec<String>> = Vec::new();
    for (label, keep) in setups {
        let weights = base;
        let chosen: Vec<Entry> = entries
            .iter()
            .map(|e| Entry {
                name: e.name.clone(),
                tempo: e.tempo,
                believed: e.believed,
                evidence: keeping(&e.evidence, keep),
            })
            .collect();
        let j = judge(&chosen, &weights);
        let (mut confidence, mut row) = (0.0, Vec::new());
        for e in &chosen {
            let a = e.evidence.decide(&weights);
            confidence += a.confidence / chosen.len() as f64;
            let word = e.tempo.map_or("", |t| verdict(tempo(&a), t));
            row.push(format!("{:7.1} {word:4} {}", tempo(&a), a.chosen));
        }
        println!(
            "{label:40} confirmed {} believed {} steadiness {:.3} mean confidence {confidence:.3}",
            j.confirmed, j.believed, j.steadiness
        );
        verdicts.push(row);
    }
    println!("changes from the first:");
    for (i, e) in entries.iter().enumerate() {
        if verdicts.iter().any(|row| row[i] != verdicts[0][i]) {
            println!("  {}", e.name);
            for (row, (label, _)) in verdicts.iter().zip(setups) {
                println!("    {:40} {}", label, row[i]);
            }
        }
    }
    // The drum stem's candidate: on how many tracks it voted, its mean
    // agreement there and its mean say over the track.
    let (mut tracks, mut agreement, mut say) = (0usize, 0.0, 0.0);
    for e in &entries {
        let a = e.evidence.decide(&base);
        for (c, r) in e.evidence.candidates.iter().zip(&a.candidates) {
            if c.is_partial() && c.beats.len() >= 4 {
                tracks += 1;
                say += c.say_between(0.0, e.evidence.duration);
                agreement += r.agreement;
            }
        }
    }
    let n = tracks.max(1) as f64;
    println!(
        "the drum stem voted on {tracks} of {} tracks: agreement {:.2}, say {:.2}",
        entries.len(),
        agreement / n,
        say / n
    );
    Ok(())
}

/// Compares the tempi of the truth list at `path` with ours, with the
/// default weights or (`fit`) the best found.
fn score(path: &Path, recipe: &RhythmRecipe, fit: bool) -> Result<(), String> {
    let entries = read_truth(path, recipe)?;
    let weights = if fit {
        fit_weights(&entries)
    } else {
        fitted(&entries)
    };
    let mut right = [0usize; 2];
    let mut total = [0usize; 2];
    let mut wrong = Vec::new();
    for e in &entries {
        let a = e.evidence.decide(&weights);
        let ours = tempo(&a);
        let (truth, word) = match e.tempo {
            Some(t) => {
                let word = verdict(ours, t);
                let k = usize::from(e.believed);
                total[k] += 1;
                if word == "ok" {
                    right[k] += 1;
                } else {
                    wrong.push(format!("{word:4} {}", e.name));
                }
                let mark = if e.believed { '?' } else { ' ' };
                (format!("{mark}{t:6.1}"), word)
            }
            None => ("      -".to_owned(), ""),
        };
        let grouping = meter::describe(&a.groupings, a.duration);
        let chosen = format!("{} {grouping}", a.chosen);
        println!("{ours:7.1} {truth} {word:4} {chosen:28} {}", e.name);
    }
    println!(
        "confirmed {}/{} right, believed {}/{} right",
        right[0], total[0], right[1], total[1]
    );
    for w in wrong {
        println!("  {w}");
    }
    Ok(())
}

fn beats(a: &Analysis) {
    for (i, (b, bpm)) in a.beats.iter().zip(&a.local_bpm).enumerate() {
        let mark = if a.is_downbeat(i) { '|' } else { ' ' };
        println!("  {mark} {b:9.3}  {bpm:6.1}");
    }
}

/// The file at its own rate with clicks on the beats, as 16-bit WAV.
fn write_clicks(file: &Path, a: &Analysis, out: &Path) -> Result<(), String> {
    let mut source = Source::open(file).map_err(|e| e.to_string())?;
    let (rate, channels) = (source.info().rate, usize::from(source.info().channels));
    let mut samples: Vec<f64> = Vec::new();
    let mut block = Vec::new();
    loop {
        block.clear();
        if source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        samples.extend(block.iter().map(|s| s / 8_388_608.0 * 0.7));
    }
    let sr = f64::from(rate);
    let length = (0.03 * sr) as usize;
    for (i, &b) in a.beats.iter().enumerate() {
        let (freq, gain) = if a.is_downbeat(i) {
            (2000.0, 0.5)
        } else {
            (1000.0, 0.35)
        };
        let start = (b * sr).round() as usize;
        for k in 0..length {
            let t = k as f64 / sr;
            let click = gain * (2.0 * PI * freq * t).sin() * (-t / 0.008).exp();
            for c in 0..channels {
                if let Some(s) = samples.get_mut((start + k) * channels + c) {
                    *s += click;
                }
            }
        }
    }
    write_wav(out, rate, channels, &samples)
}

/// A 16-bit WAV file written as the samples come; the sizes go into the
/// header at the end.
struct WavWriter {
    out: BufWriter<File>,
    bytes: u32,
}

impl WavWriter {
    fn create(path: &Path, rate: u32, channels: u16) -> Result<WavWriter, String> {
        let mut out = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
        let block_align = channels * 2;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&36u32.to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&channels.to_le_bytes());
        header.extend_from_slice(&rate.to_le_bytes());
        header.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        header.extend_from_slice(&block_align.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&0u32.to_le_bytes());
        out.write_all(&header).map_err(|e| e.to_string())?;
        Ok(WavWriter { out, bytes: 0 })
    }

    /// Writes interleaved samples (full scale ±1).
    fn write(&mut self, samples: impl Iterator<Item = f64>) -> Result<(), String> {
        for s in samples {
            let v = (s * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
            self.out
                .write_all(&v.to_le_bytes())
                .map_err(|e| e.to_string())?;
            self.bytes += 2;
        }
        Ok(())
    }

    /// Puts the sizes into the header and closes the file.
    fn finish(self) -> Result<(), String> {
        use std::io::{Seek, SeekFrom};
        let mut file = self.out.into_inner().map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(4)).map_err(|e| e.to_string())?;
        file.write_all(&(36 + self.bytes).to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(40)).map_err(|e| e.to_string())?;
        file.write_all(&self.bytes.to_le_bytes())
            .map_err(|e| e.to_string())
    }
}

/// `samples` (interleaved, full scale ±1) as a 16-bit WAV file.
fn write_wav(out: &Path, rate: u32, channels: usize, samples: &[f64]) -> Result<(), String> {
    let mut w = WavWriter::create(out, rate, channels as u16)?;
    w.write(samples.iter().copied())?;
    w.finish()
}

/// Separates `file` on the GPU and writes its stems to `dir` as
/// `NAME - STEM.wav`, block by block as they are finished; returns what
/// was written.
fn write_stems(file: &Path, dir: &Path) -> Result<String, String> {
    let gpu = GpuSeparator::shared().ok_or("stems need a GPU")?;
    let mut source = Source::open(file).map_err(|e| e.to_string())?;
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let name = file
        .file_stem()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let paths: Vec<PathBuf> = Stem::ALL
        .iter()
        .map(|stem| dir.join(format!("{name} - {}.wav", stem.id())))
        .collect();
    let mut writers = paths
        .iter()
        .map(|p| WavWriter::create(p, stems::RATE, 2))
        .collect::<Result<Vec<_>, _>>()?;
    let mut failed = None;
    let mut sink = |b: &stems::StemBlock| {
        for (w, stem) in writers.iter_mut().zip(&b.stems) {
            let frames = stem[0].iter().zip(stem[1]);
            let samples = frames.flat_map(|(l, r)| [f64::from(*l), f64::from(*r)]);
            if let Err(e) = w.write(samples) {
                failed.get_or_insert(e);
            }
        }
    };
    let mut splitter = Splitter::overlapped(gpu);
    let mut split_error = None;
    let mut put = |left: Vec<f32>, right: Vec<f32>| {
        if split_error.is_none() {
            if let Err(e) = splitter.push(&left, &right, &mut sink) {
                split_error = Some(e);
            }
        }
    };
    let mut feed = StereoFeed::new(rate);
    let mut block = Vec::new();
    loop {
        block.clear();
        if source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        feed.push(&block, channels, &mut put);
    }
    feed.finish(&mut put);
    if let Some(e) = split_error {
        return Err(e);
    }
    splitter.finish(&mut sink)?;
    if let Some(e) = failed {
        return Err(e);
    }
    for w in writers {
        w.finish()?;
    }
    let written: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
    Ok(format!(
        "{} on {}: {}",
        name,
        gpu.name(),
        written.join(", ")
    ))
}
