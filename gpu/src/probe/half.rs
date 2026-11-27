//! f16 on this GPU, where it has it ([`probe`]): the range's edges (the
//! largest finite value, overflow to ∞, subnormals kept or flushed),
//! arithmetic past the range, the accuracy of `exp`, `tanh`, products and
//! sums in f16 ulps against the processor, and f16 storage: an
//! `array<f16>` read and written a value an invocation (two to a 32-bit
//! word, so a word is written by two invocations).

use super::elementwise::{self, Operation};
use super::gpu::{Bound, Gpu, Noise};
use super::maths::{judge, short, substitute, ulps_of, Verdict};
use super::text::{check_lines, table, ulps_cell, value};
use super::{Check, Miss};

/// An f32 rounded to the nearest f16 (ties to even), as its bits.
pub(super) fn to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // A subnormal f16 (or zero): the mantissa with its leading one,
        // shifted down and rounded.
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut h = m >> shift;
        if rest > half || (rest == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        // A carry into the exponent is right (up to ∞).
        h += 1;
    }
    sign | h as u16
}

/// An f16's bits as the f32 it is.
pub(super) fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f32::from(h & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        0x1f if mantissa == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// `x` through f16 and back.
pub(super) fn rounded(x: f64) -> f32 {
    from_f16(to_f16(x as f32))
}

/// The distance in f16 ulps of two values that are f16s ([`ulps_of`]).
pub(super) fn ulps16(a: f32, b: f32) -> u64 {
    ulps_of(a, b, |x| u32::from(to_f16(x)), 1 << 15)
}

/// An operation of the f16 kernel, on `a`, `b`, `c` made f16s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Convert,
    Mul,
    Add,
    Exp,
    Tanh,
    Fma,
}

const OPS: [Op; 6] = [Op::Convert, Op::Mul, Op::Add, Op::Exp, Op::Tanh, Op::Fma];

impl Operation for Op {
    fn code(self) -> u32 {
        self as u32
    }

    fn wgsl(self) -> &'static str {
        match self {
            Op::Convert => "a",
            Op::Mul => "a * b",
            Op::Add => "a + b",
            Op::Exp => "exp(a)",
            Op::Tanh => "tanh(a)",
            Op::Fma => "fma(a, b, c)",
        }
    }
}

impl Op {
    /// What f16 arithmetic gives: the f16 inputs' exact result rounded
    /// once to f16.
    fn reference(self, a: f32, b: f32, c: f32) -> f32 {
        let [a, b, c] = [a, b, c].map(|x| f64::from(rounded(f64::from(x))));
        rounded(match self {
            Op::Convert => a,
            Op::Mul => a * b,
            Op::Add => a + b,
            Op::Exp => a.exp(),
            Op::Tanh => a.tanh(),
            Op::Fma => a.mul_add(b, c),
        })
    }
}

/// One edge: the expression, f16's value, the GPU's, how they compare.
#[derive(Clone, Debug)]
struct Edge {
    shown: String,
    want: f32,
    got: f32,
    verdict: Verdict,
}

/// One operation's largest error over a range, in f16 ulps.
#[derive(Clone, Debug)]
struct Accuracy {
    op: Op,
    range: (f32, f32),
    ulps: u64,
}

/// What f16 does here.
#[derive(Clone, Debug)]
pub(super) struct Half {
    edges: Vec<Edge>,
    accuracy: Vec<Accuracy>,
    /// An `array<f16>` written a value an invocation.
    stored: Check,
}

/// The f16 storage kernel: each invocation one value of an `array<f16>`.
const STORAGE: &str = "enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    dst[g.x] = src[g.x] * f16(2.0) + f16(1.0);
}
";

type Element = elementwise::Element<Op>;

fn edge_elements() -> Vec<Element> {
    vec![
        (Op::Convert, 65504.0, 0.0, 0.0),
        (Op::Convert, 65519.0, 0.0, 0.0),
        (Op::Convert, 65520.0, 0.0, 0.0),
        (Op::Convert, 1e5, 0.0, 0.0),
        (Op::Convert, 6.1035156e-5, 0.0, 0.0),
        (Op::Convert, 6e-8, 0.0, 0.0),
        (Op::Convert, 1e-8, 0.0, 0.0),
        (Op::Convert, f32::NAN, 0.0, 0.0),
        (Op::Mul, 65504.0, 2.0, 0.0),
        (Op::Mul, 300.0, 300.0, 0.0),
        (Op::Mul, 6.1035156e-5, 0.01, 0.0),
        (Op::Exp, 11.0, 0.0, 0.0),
        (Op::Exp, 12.0, 0.0, 0.0),
        (Op::Exp, -20.0, 0.0, 0.0),
        (Op::Tanh, 20.0, 0.0, 0.0),
        (Op::Tanh, -20.0, 0.0, 0.0),
    ]
}

/// An accuracy range: the operation, the first input's range, the
/// second's.
type Span = (Op, (f32, f32), (f32, f32));

const SPANS: [Span; 5] = [
    (Op::Exp, (-10.0, 11.0), (0.0, 0.0)),
    (Op::Tanh, (-5.0, 5.0), (0.0, 0.0)),
    (Op::Mul, (-200.0, 200.0), (-200.0, 200.0)),
    (Op::Add, (-1000.0, 1000.0), (-1000.0, 1000.0)),
    (Op::Fma, (-10.0, 10.0), (-10.0, 10.0)),
];

const SAMPLES: usize = 8192;

/// The f32 ulps an f16 edge may be off and count as close: 16 f16 ulps
/// (an f16 value as an f32 has 13 bits more below its last one).
const CLOSE: u64 = 16 << 13;

/// f16 on `gpu` (one with `SHADER_F16`).
pub(super) fn probe(gpu: &Gpu) -> Result<Half, Miss> {
    let elements = edge_elements();
    let edges = elements
        .iter()
        .zip(elementwise::run(
            gpu,
            "enable f16;",
            "f16",
            &OPS,
            &elements,
        )?)
        .map(|(&(op, a, b, c), got)| {
            let want = op.reference(a, b, c);
            Edge {
                shown: format!(
                    "f16: {}",
                    substitute(op.wgsl(), &[short(a), short(b), short(c)])
                ),
                want,
                got,
                verdict: judge(want, got, CLOSE),
            }
        })
        .collect();
    let mut noise = Noise::new(0xf16);
    let mut elements = Vec::new();
    for &(op, (lo, hi), (lo2, hi2)) in &SPANS {
        for _ in 0..SAMPLES {
            let pick = |noise: &mut Noise, lo: f32, hi: f32| lo + (hi - lo) * noise.unit() as f32;
            let a = pick(&mut noise, lo, hi);
            let b = pick(&mut noise, lo2, hi2);
            let c = pick(&mut noise, lo2, hi2);
            elements.push((op, a, b, c));
        }
    }
    let got = elementwise::run(gpu, "enable f16;", "f16", &OPS, &elements)?;
    let accuracy = SPANS
        .iter()
        .enumerate()
        .map(|(s, &(op, range, _))| {
            let at = s * SAMPLES..(s + 1) * SAMPLES;
            let ulps = elements[at.clone()]
                .iter()
                .zip(&got[at])
                .filter(|(&(op, a, b, c), _)| op.reference(a, b, c).is_finite())
                .map(|(&(op, a, b, c), &g)| ulps16(op.reference(a, b, c), g))
                .max()
                .unwrap_or(0);
            Accuracy { op, range, ulps }
        })
        .collect();
    Ok(Half {
        edges,
        accuracy,
        stored: storage(gpu)?,
    })
}

/// f16 storage: values (two to a word) in, each doubled plus one out.
fn storage(gpu: &Gpu) -> Result<Check, String> {
    const VALUES: usize = 1 << 20;
    let kernel = gpu.kernel(STORAGE, "main")?;
    let values: Vec<u16> = (0..VALUES)
        .map(|k| to_f16((k % 1000) as f32 * 0.25))
        .collect();
    let words: Vec<u32> = values
        .chunks_exact(2)
        .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
        .collect();
    let src = gpu.words(&words)?;
    let dst = gpu.zeros(VALUES as u64 * 2)?;
    gpu.run(
        &kernel,
        &[Bound::All(&src), Bound::All(&dst)],
        [(VALUES / 256) as u32, 1, 1],
        1,
    )?;
    let out = gpu.read(&dst)?;
    let wrong = out
        .iter()
        .flat_map(|&w| [w as u16, (w >> 16) as u16])
        .zip(&values)
        .filter(|&(got, &v)| got != to_f16(from_f16(v) * 2.0 + 1.0))
        .count() as u64;
    Ok(Check {
        name: "array<f16> storage, a value an invocation (two to a word)",
        cases: VALUES as u64,
        wrong,
    })
}

/// The f16 findings as report lines.
pub(super) fn lines(h: &Half) -> Vec<String> {
    let mut rows = vec![vec![
        "expression".to_string(),
        "f16".into(),
        "got".into(),
        "".into(),
    ]];
    for e in &h.edges {
        rows.push(vec![
            e.shown.clone(),
            value(e.want),
            value(e.got),
            e.verdict.label().into(),
        ]);
    }
    let mut lines = table(&rows);
    lines.push(String::new());
    let mut rows = vec![vec![
        "f16 function".to_string(),
        "range".into(),
        "max f16 ulp".into(),
    ]];
    for a in &h.accuracy {
        rows.push(vec![
            a.op.wgsl().to_string(),
            format!("[{}, {}]", short(a.range.0), short(a.range.1)),
            ulps_cell(a.ulps),
        ]);
    }
    lines.extend(table(&rows));
    lines.extend(check_lines(std::slice::from_ref(&h.stored)));
    lines
}
