//! Packs the pitch network's weights (`tools/convert_fcpe.py` makes them at
//! full precision) into the file `jodel_rhythm::pitch` builds in.
//!
//! Vectors (and tensors under 4096 values) stay as they are. A matrix's
//! kind comes from rules, the later ones first: the defaults, Q4_K where
//! its rows are whole K blocks, halves otherwise; then each `PATTERN=KIND`
//! given, `KIND` one of `f32`, `f16`, `q4k`, `q5k`, `q8`, `PATTERN` a
//! tensor's name, `*` matching any run of characters.
//!
//! The K blocks (Q4_K, Q5_K) are fitted as `jodel_rhythm::net::q4k` says:
//! with `--stats FILE` (`pitch_quant stats`: each matrix's mean square
//! input a column) by those statistics to the power `--power` (1 if not
//! given); with `--gptq FILE` (`pitch_quant stats --gram`: each matrix's
//! mean product of each two inputs) by GPTQ; with `--refine`, their
//! scales refined.
//!
//! The built-in file, from the workspace (`fcpe-f32.bin` the full
//! precision weights, `gram-120.bin` the grams of the first 120 voices of
//! the calibration list, both kept out of this repository):
//!
//! ```text
//! cargo run --release -p jodel-rhythm --example pitch_pack -- \
//!     fcpe-f32.bin rhythm/models/pitch.bin --gptq gram-120.bin \
//!     '*=q5k' input_stack.0.weight=f16 '*.4.conv.weight=f16'
//! ```
//!
//! Every matrix whose rows are whole blocks is Q5_K by GPTQ; the first
//! convolution and the depthwise ones (rows of 384 and 31) are halves.

use jodel_rhythm::net::q4k::{BLOCK, Q8_BLOCK};
use jodel_rhythm::net::weights::{Kind, PackedTensor, Weights};

/// Whether `name` matches `pattern`, `*` any run of characters.
fn matches(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == name,
        Some((head, tail)) => {
            let Some(rest) = name.strip_prefix(head) else {
                return false;
            };
            (0..=rest.len())
                .filter(|&i| rest.is_char_boundary(i))
                .any(|i| matches(tail, &rest[i..]))
        }
    }
}

fn kind_of(word: &str) -> Kind {
    match word {
        "f32" => Kind::F32,
        "f16" => Kind::F16,
        "q4k" => Kind::Q4k,
        "q5k" => Kind::Q5k,
        "q8" => Kind::Q8,
        k => panic!("no kind {k}"),
    }
}

/// The default kind of a tensor of `shape` and `cols` a row.
fn default_kind(shape: &[usize], len: usize, cols: usize) -> Kind {
    if shape.len() < 2 || len < 4096 {
        Kind::F32
    } else if cols % BLOCK == 0 {
        Kind::Q4k
    } else {
        Kind::F16
    }
}

/// What the command line asks.
struct Options {
    input: String,
    output: String,
    stats: Option<Weights>,
    grams: Option<Weights>,
    power: f32,
    refine: bool,
    rules: Vec<(String, Kind)>,
}

fn options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut words = Vec::new();
    let (mut stats, mut grams, mut power, mut refine) = (None, None, 1.0, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--stats" => {
                let f = args.next().expect("--stats FILE");
                let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("{f}: {e}"));
                stats = Some(Weights::parse(&bytes).expect("a statistics file"));
            }
            "--gptq" => {
                let f = args.next().expect("--gptq FILE");
                let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("{f}: {e}"));
                grams = Some(Weights::parse(&bytes).expect("a gram file"));
            }
            "--power" => power = args.next().and_then(|p| p.parse().ok()).expect("--power P"),
            "--refine" => refine = true,
            _ => words.push(a),
        }
    }
    if words.len() < 2 {
        eprintln!("usage: pitch_pack IN_F32 OUT [--stats FILE] [--gptq FILE] [--power P] [--refine] [PATTERN=KIND]...");
        std::process::exit(2);
    }
    let rules = words[2..]
        .iter()
        .map(|r| {
            let (p, k) = r
                .split_once('=')
                .unwrap_or_else(|| panic!("{r}: not PATTERN=KIND"));
            (p.to_owned(), kind_of(k))
        })
        .collect();
    Options {
        input: words[0].clone(),
        output: words[1].clone(),
        stats,
        grams,
        power,
        refine,
        rules,
    }
}

fn main() {
    let o = options();
    let bytes = std::fs::read(&o.input).unwrap_or_else(|e| panic!("{}: {e}", o.input));
    let w = Weights::parse(&bytes).expect("a weights file");
    let mut tensors = Vec::new();
    for (name, shape) in w.shapes() {
        let values = w.get(name).unwrap().to_vec();
        let rows = shape.first().copied().unwrap_or(1);
        let cols = values.len() / rows.max(1);
        let kind = match default_kind(shape, values.len(), cols) {
            Kind::F32 => Kind::F32,
            default => o
                .rules
                .iter()
                .rev()
                .find(|(p, _)| matches(p, name))
                .map_or(default, |r| r.1),
        };
        let blocks = matches!(kind, Kind::Q4k | Kind::Q5k);
        let importance = o
            .stats
            .as_ref()
            .filter(|_| blocks)
            .and_then(|s| s.get(name))
            .map(|m| m.iter().map(|v| v.max(0.0).powf(o.power)).collect());
        assert!(
            !blocks || cols % BLOCK == 0,
            "{name}: rows of {cols} are not whole K blocks"
        );
        assert!(
            kind != Kind::Q8 || cols % Q8_BLOCK == 0,
            "{name}: rows of {cols} are not whole Q8_0 blocks"
        );
        let gram = o
            .grams
            .as_ref()
            .filter(|_| blocks)
            .and_then(|g| g.get(name))
            .map(<[f32]>::to_vec);
        println!(
            "{name:50} {shape:?} {kind:?}{}{}",
            if importance.is_some() {
                " by importance"
            } else {
                ""
            },
            if gram.is_some() { " by GPTQ" } else { "" }
        );
        tensors.push(PackedTensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            values,
            kind,
            importance,
            refine: o.refine && blocks,
            gram,
        });
    }
    let packed = Weights::pack(&tensors).expect("packed");
    std::fs::write(&o.output, &packed).unwrap_or_else(|e| panic!("{}: {e}", o.output));
    println!("{} bytes, from {}", packed.len(), bytes.len());
}
