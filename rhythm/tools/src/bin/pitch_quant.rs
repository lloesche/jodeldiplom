//! Measures how well packed pitch weights keep the full precision
//! network's output on real voices (separated vocal stems, 16 kHz mono
//! 16-bit WAV files), and gathers the statistics `pitch_pack` weighs the
//! quantisation by.
//!
//! The voices are the lines `FROM..TO` of a list of file names (`--list`,
//! one name a line) in a folder (`--voices`).
//!
//! ```text
//! pitch_quant reference --weights F32 --refs DIR   VOICES
//! pitch_quant stats     --weights F32 --out FILE [--gram FILE] VOICES
//! pitch_quant eval      --refs DIR [--worst N]     VOICES WEIGHTS...
//! VOICES: --voices DIR --list FILE --from A --to B
//! ```
//!
//! - `reference` keeps each voice's pitch by the given weights as
//!   `DIR/NAME.pitch`: a frame's Hz and confidence, little-endian singles.
//! - `stats` writes, for each weight matrix, the mean square of each of
//!   its input columns over every frame (in the order of the tensor's
//!   values, `[out][in][k]`), as a packed weights file of single vectors
//!   named as the matrices; with `--gram`, also the mean product of each
//!   two input columns, a `[cols][cols]` matrix each, in the same order.
//! - `eval` runs each weights file on the voices and compares its pitch
//!   with the kept one: frames voiced in one only; the cents between them
//!   where both are (median, 99th percentile, the shares over 50 and over
//!   600); the frames whose note as the player shows it
//!   (`Melody::sung`) differs; the file's size and bits a weight.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use jodel_rhythm::nn::weights::{Kind, Weights};
use jodel_rhythm::pitch::{spectrogram, Pitch, PitchNet, RATE};
use jodel_rhythm::waveform::Melody;
use jodel_rhythm_tools::pack::{pack, PackedTensor};

/// The options and the words left.
struct Args {
    flags: HashMap<String, String>,
    rest: Vec<String>,
}

impl Args {
    fn parse(args: Vec<String>) -> Args {
        let mut flags = HashMap::new();
        let mut rest = Vec::new();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                let v = it
                    .next()
                    .unwrap_or_else(|| panic!("--{name} needs a value"));
                flags.insert(name.to_owned(), v);
            } else {
                rest.push(a);
            }
        }
        Args { flags, rest }
    }

    fn get(&self, name: &str) -> &str {
        self.flags
            .get(name)
            .unwrap_or_else(|| panic!("--{name} is needed"))
    }

    fn voices(&self) -> Vec<PathBuf> {
        let list = std::fs::read_to_string(self.get("list")).expect("the list");
        let from: usize = self.get("from").parse().expect("--from");
        let to: usize = self.get("to").parse().expect("--to");
        let dir = Path::new(self.get("voices"));
        list.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .skip(from)
            .take(to - from)
            .map(|name| dir.join(name))
            .collect()
    }
}

/// The samples of a 16-bit PCM mono WAV file at [`RATE`], as -1 to 1.
fn read_wav(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{}: not a WAV file",
        path.display()
    );
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32_at(at + 4) as usize;
        let body = at + 8;
        if id == b"fmt " {
            let (format, channels, rate, bits) = (
                u16_at(body),
                u16_at(body + 2),
                u32_at(body + 4),
                u16_at(body + 14),
            );
            assert!(
                format == 1 && channels == 1 && rate == RATE && bits == 16,
                "{}: not 16-bit PCM mono at {RATE} Hz",
                path.display()
            );
        } else if id == b"data" {
            let end = (body + len).min(bytes.len());
            return bytes[body..end]
                .chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
                .collect();
        }
        at = body + len + (len & 1);
    }
    panic!("{}: no data", path.display());
}

fn load_net(path: &str) -> PitchNet {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    PitchNet::load(Weights::parse(&bytes).expect("weights")).expect("the network")
}

fn pitch_of(net: &PitchNet, samples: &[f32]) -> Pitch {
    let (mel, frames) = spectrogram(samples);
    net.decode(&net.run(&mel, frames, false).0)
}

fn stem(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

fn write_pitch(path: &Path, p: &Pitch) {
    let mut out = Vec::with_capacity(p.hz.len() * 8);
    for (h, c) in p.hz.iter().zip(&p.confidence) {
        out.extend(h.to_le_bytes());
        out.extend(c.to_le_bytes());
    }
    durable::write_atomic(path, &out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn read_pitch(path: &Path) -> Pitch {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut p = Pitch::default();
    for b in bytes.chunks_exact(8) {
        p.hz.push(f32::from_le_bytes(b[..4].try_into().unwrap()));
        p.confidence
            .push(f32::from_le_bytes(b[4..].try_into().unwrap()));
    }
    p
}

fn reference(args: &Args) {
    let net = load_net(args.get("weights"));
    let dir = PathBuf::from(args.get("refs"));
    let started = Instant::now();
    let voices = args.voices();
    for (i, v) in voices.iter().enumerate() {
        let out = dir.join(format!("{}.pitch", stem(v)));
        if out.exists() {
            continue;
        }
        write_pitch(&out, &pitch_of(&net, &read_wav(v)));
        if i % 10 == 9 {
            println!("{} of {} ({:.0?})", i + 1, voices.len(), started.elapsed());
        }
    }
}

/// What a matrix multiplied, summed: a column's squares, the products of
/// each two columns (where asked), the rows.
#[derive(Default)]
struct Sums {
    squares: Vec<f64>,
    products: Vec<f64>,
    rows: u64,
}

/// `rows` (of `cols`): each column's sum of squares and, where `products`,
/// each two columns' sum of products (`[cols][cols]`, the upper triangle).
fn sums_of(rows: &[f32], cols: usize, products: bool) -> (Vec<f64>, Vec<f32>) {
    let mut squares = vec![0.0f64; cols];
    let mut prod = vec![0.0f32; if products { cols * cols } else { 0 }];
    for row in rows.chunks_exact(cols) {
        for (s, &v) in squares.iter_mut().zip(row) {
            *s += f64::from(v) * f64::from(v);
        }
        if products {
            for (i, &a) in row.iter().enumerate() {
                for (p, &b) in prod[i * cols + i..(i + 1) * cols].iter_mut().zip(&row[i..]) {
                    *p += a * b;
                }
            }
        }
    }
    (squares, prod)
}

fn stats(args: &Args) {
    let net = load_net(args.get("weights"));
    let gram = args.flags.get("gram");
    let all: Mutex<HashMap<String, Sums>> = Mutex::new(HashMap::new());
    let see = |name: &str, rows: &[f32]| {
        let cols = row_len(name);
        let (squares, products) = sums_of(rows, cols, gram.is_some());
        let mut all = all.lock().unwrap();
        let entry = all.entry(name.to_owned()).or_default();
        if entry.squares.is_empty() {
            entry.squares = vec![0.0; cols];
            entry.products = vec![0.0; products.len()];
        }
        for (s, p) in entry.squares.iter_mut().zip(squares) {
            *s += p;
        }
        for (s, &p) in entry.products.iter_mut().zip(&products) {
            *s += f64::from(p);
        }
        entry.rows += (rows.len() / cols) as u64;
    };
    let started = Instant::now();
    let voices = args.voices();
    for (i, v) in voices.iter().enumerate() {
        let (mel, frames) = spectrogram(&read_wav(v));
        net.run_seeing(&mel, frames, &see);
        if i % 10 == 9 {
            println!("{} of {} ({:.0?})", i + 1, voices.len(), started.elapsed());
        }
    }
    let mut means = Vec::new();
    let mut grams = Vec::new();
    for (name, s) in all.into_inner().unwrap() {
        let n = s.rows.max(1) as f64;
        let order = stored_order(&name, s.squares.len());
        let mean: Vec<f32> = order.iter().map(|&t| (s.squares[t] / n) as f32).collect();
        means.push(vector(&name, vec![mean.len()], mean));
        if gram.is_some() {
            let cols = order.len();
            let mut g = vec![0.0f32; cols * cols];
            for (a, &ta) in order.iter().enumerate() {
                for (b, &tb) in order.iter().enumerate() {
                    let (i, j) = (ta.min(tb), ta.max(tb));
                    g[a * cols + b] = (s.products[i * cols + j] / n) as f32;
                }
            }
            grams.push(vector(&name, vec![cols, cols], g));
        }
    }
    let out = args.get("out");
    write_tensors(out, means);
    if let Some(g) = gram {
        write_tensors(g, grams);
    }
}

/// A tensor of single values to pack.
fn vector(name: &str, shape: Vec<usize>, values: Vec<f32>) -> PackedTensor {
    PackedTensor {
        name: name.to_owned(),
        shape,
        values,
        kind: Kind::F32,
        importance: None,
        refine: false,
        gram: None,
    }
}

fn write_tensors(path: &str, mut tensors: Vec<PackedTensor>) {
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    durable::write_atomic(Path::new(path), &pack(&tensors).unwrap()).expect("the statistics");
    println!("{} matrices to {path}", tensors.len());
}
/// The length of the rows matrix `name` multiplies.
fn row_len(name: &str) -> usize {
    match name {
        "input_stack.0.weight" => 3 * 128,
        "input_stack.3.weight" => 3 * 512,
        "output_proj.weight" => 512,
        n if n.ends_with(".2.weight") => 512,
        n if n.ends_with(".6.weight") => 1024,
        n => panic!("no matrix {n}"),
    }
}

/// For each column of matrix `name` in the order of its values (`[in][k]`
/// for a convolution), the column as seen (`[k][in]`).
fn stored_order(name: &str, cols: usize) -> Vec<usize> {
    if !name.starts_with("input_stack.") {
        return (0..cols).collect();
    }
    let inputs = cols / 3;
    (0..cols).map(|s| (s % 3) * inputs + s / 3).collect()
}
/// How a candidate's pitch differs from the kept one, summed over voices.
#[derive(Default)]
struct Tally {
    frames: u64,
    voiced: u64,
    flips: u64,
    cents: Vec<f32>,
    notes: u64,
    sung: u64,
}

impl Tally {
    fn add(&mut self, got: &Pitch, want: &Pitch) {
        let n = want.hz.len().min(got.hz.len());
        for i in 0..n {
            let (a, b) = (got.hz[i], want.hz[i]);
            self.voiced += u64::from(b > 0.0);
            match (a > 0.0, b > 0.0) {
                (true, true) => self.cents.push((1200.0 * (a / b).log2()).abs()),
                (false, false) => {}
                _ => self.flips += 1,
            }
        }
        let (mg, mw) = (Melody::from_pitch(got), Melody::from_pitch(want));
        for i in 0..n {
            let w = mw.sung(i);
            self.sung += u64::from(w.is_some());
            self.notes += u64::from(mg.sung(i) != w);
        }
        self.frames += n as u64;
    }

    /// The share over 50 cents, for ranking voices.
    fn far(&self) -> f64 {
        self.cents.iter().filter(|&&c| c > 50.0).count() as f64 / self.cents.len().max(1) as f64
    }

    fn row(&mut self, name: &str, bytes: usize, weights: usize) -> String {
        self.cents.sort_by(f32::total_cmp);
        let at = |q: f64| {
            self.cents
                .get(((self.cents.len() as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let share = |c: f32| {
            100.0 * self.cents.iter().filter(|&&x| x > c).count() as f64
                / self.cents.len().max(1) as f64
        };
        let pct = |n: u64| 100.0 * n as f64 / self.frames.max(1) as f64;
        format!(
            "{name:28} {bytes:>9} {:>5.2} | {:>6.3} | {:>6.2} {:>7.2} | {:>6.3} {:>6.3} | {:>6.3}",
            bytes as f64 * 8.0 / weights as f64,
            pct(self.flips),
            at(0.5),
            at(0.99),
            share(50.0),
            share(600.0),
            pct(self.notes),
        )
    }
}

/// The weights in a network's matrices and depthwise kernels.
fn weight_count(w: &Weights) -> usize {
    w.shapes()
        .iter()
        .filter(|(_, s)| s.len() >= 2)
        .map(|(_, s)| s.iter().product::<usize>())
        .sum()
}

fn eval(args: &Args) {
    let refs = PathBuf::from(args.get("refs"));
    let worst: usize = args.flags.get("worst").map_or(0, |w| w.parse().unwrap());
    let files = &args.rest[1..];
    let mut nets = Vec::new();
    for f in files {
        let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        let w = Weights::parse(&bytes).expect("weights");
        let count = weight_count(&w);
        nets.push((PitchNet::load(w).expect("the network"), bytes.len(), count));
    }
    let mut tallies: Vec<Tally> = files.iter().map(|_| Tally::default()).collect();
    let mut per_voice: Vec<Vec<(f64, f64, String)>> = files.iter().map(|_| Vec::new()).collect();
    let started = Instant::now();
    let voices = args.voices();
    for (i, v) in voices.iter().enumerate() {
        let want = read_pitch(&refs.join(format!("{}.pitch", stem(v))));
        let (mel, frames) = spectrogram(&read_wav(v));
        for ((net, _, _), (tally, voice)) in nets
            .iter()
            .zip(tallies.iter_mut().zip(per_voice.iter_mut()))
        {
            let got = net.decode(&net.run(&mel, frames, false).0);
            let mut one = Tally::default();
            one.add(&got, &want);
            voice.push((
                one.far(),
                one.flips as f64 / one.frames.max(1) as f64,
                stem(v),
            ));
            tally.add(&got, &want);
        }
        if i % 25 == 24 {
            eprintln!("{} of {} ({:.0?})", i + 1, voices.len(), started.elapsed());
        }
    }
    let t = &tallies[0];
    println!(
        "{} voices, {} frames, {:.1} % voiced, {:.1} % with a note",
        voices.len(),
        t.frames,
        100.0 * t.voiced as f64 / t.frames.max(1) as f64,
        100.0 * t.sung as f64 / t.frames.max(1) as f64
    );
    println!(
        "{:28} {:>9} {:>5} | {:>6} | {:>6} {:>7} | {:>6} {:>6} | {:>6}",
        "weights", "bytes", "bpw", "flip%", "med", "99%", ">50%", ">600%", "note%"
    );
    for ((f, (_, bytes, count)), tally) in files.iter().zip(&nets).zip(&mut tallies) {
        let name = Path::new(f).file_stem().unwrap().to_string_lossy();
        println!("{}", tally.row(&name, *bytes, *count));
    }
    for (f, voice) in files.iter().zip(&mut per_voice) {
        if worst == 0 {
            break;
        }
        voice.sort_by(|a, b| b.0.total_cmp(&a.0));
        let list: Vec<String> = voice
            .iter()
            .take(worst)
            .map(|(far, flip, name)| format!("{name} {:.1}/{:.1}", 100.0 * far, 100.0 * flip))
            .collect();
        println!("{f}: worst (>50 % / flip %): {}", list.join(", "));
    }
}

fn main() {
    let args = Args::parse(std::env::args().skip(1).collect());
    match args.rest.first().map(String::as_str) {
        Some("reference") => reference(&args),
        Some("stats") => stats(&args),
        Some("eval") => eval(&args),
        _ => {
            eprintln!("usage: pitch_quant reference|stats|eval … (see the source)");
            std::process::exit(2);
        }
    }
}
