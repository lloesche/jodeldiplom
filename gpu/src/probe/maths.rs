//! The GPU's f32 maths at the edges and across ranges ([`edges`],
//! [`accuracy`]): `exp`, `exp2` and `tanh` at huge arguments (and the held
//! forms every program has, `exp_held`, `tanh_held`), `log` at 0 and
//! below, infinities and NaN through arithmetic, `min`, `max`, `clamp` and
//! `select`, division by zero, denormals; then each function's worst error
//! against the processor's f64 over a range. All inputs come from a
//! buffer, so nothing is folded by the compiler.

use super::elementwise::{self, Operation};
use super::gpu::{Gpu, Noise};
use super::text::{table, ulps_cell, value};
use super::Miss;

/// An operation of the maths kernel on its inputs `a`, `b`, `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    Exp,
    Exp2,
    Tanh,
    ExpHeld,
    TanhHeld,
    /// `exp(a - b)`: an online softmax's fade against -1e30.
    ExpDiff,
    Log,
    Sqrt,
    InverseSqrt,
    Sin,
    Cos,
    Pow,
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Clamp,
    /// `select(b, c, a < 1.0)`.
    SelectLess,
    /// `select(0.0, 1.0, a != a)`: NaN found by comparing.
    IsNan,
    Fma,
    /// `a`, stored again.
    Copy,
}

const OPS: [Op; 23] = [
    Op::Exp,
    Op::Exp2,
    Op::Tanh,
    Op::ExpHeld,
    Op::TanhHeld,
    Op::ExpDiff,
    Op::Log,
    Op::Sqrt,
    Op::InverseSqrt,
    Op::Sin,
    Op::Cos,
    Op::Pow,
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Min,
    Op::Max,
    Op::Clamp,
    Op::SelectLess,
    Op::IsNan,
    Op::Fma,
    Op::Copy,
];

impl Operation for Op {
    fn code(self) -> u32 {
        self as u32
    }

    /// The operation in WGSL, on `a`, `b`, `c`.
    fn wgsl(self) -> &'static str {
        match self {
            Op::Exp => "exp(a)",
            Op::Exp2 => "exp2(a)",
            Op::Tanh => "tanh(a)",
            Op::ExpHeld => "exp_held(a)",
            Op::TanhHeld => "tanh_held(a)",
            Op::ExpDiff => "exp(a - b)",
            Op::Log => "log(a)",
            Op::Sqrt => "sqrt(a)",
            Op::InverseSqrt => "inverseSqrt(a)",
            Op::Sin => "sin(a)",
            Op::Cos => "cos(a)",
            Op::Pow => "pow(a, b)",
            Op::Add => "a + b",
            Op::Sub => "a - b",
            Op::Mul => "a * b",
            Op::Div => "a / b",
            Op::Min => "min(a, b)",
            Op::Max => "max(a, b)",
            Op::Clamp => "clamp(a, b, c)",
            Op::SelectLess => "select(b, c, a < 1.0)",
            Op::IsNan => "select(0.0, 1.0, a != a)",
            Op::Fma => "fma(a, b, c)",
            Op::Copy => "a",
        }
    }
}

impl Op {
    /// What IEEE 754 arithmetic gives (in f64, rounded to f32 by the
    /// caller): `min` and `max` as `minNum` and `maxNum` (a NaN loses).
    pub(super) fn reference(self, a: f64, b: f64, c: f64) -> f64 {
        match self {
            Op::Exp => a.exp(),
            Op::Exp2 => a.exp2(),
            Op::Tanh => a.tanh(),
            Op::ExpHeld => a.max(-87.0).exp(),
            Op::TanhHeld => a.clamp(-10.0, 10.0).tanh(),
            Op::ExpDiff => ((a as f32 - b as f32) as f64).exp(),
            Op::Log => a.ln(),
            Op::Sqrt => a.sqrt(),
            Op::InverseSqrt => 1.0 / a.sqrt(),
            Op::Sin => a.sin(),
            Op::Cos => a.cos(),
            Op::Pow => a.powf(b),
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
            Op::Div => a / b,
            Op::Min => a.min(b),
            Op::Max => a.max(b),
            Op::Clamp => a.max(b).min(c),
            Op::SelectLess => {
                if a < 1.0 {
                    c
                } else {
                    b
                }
            }
            Op::IsNan => {
                if a.is_nan() {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Fma => a.mul_add(b, c),
            Op::Copy => a,
        }
    }

    /// The operation written with its inputs, for the report.
    fn shown(self, a: f32, b: f32, c: f32) -> String {
        substitute(self.wgsl(), &[short(a), short(b), short(c)])
    }

    /// The ulps WGSL allows a result of this operation on `a`, `b` (the
    /// `exp`s: 3 + 2|x| of the exponent `x`); others a few.
    fn allowed(self, a: f32, b: f32) -> u64 {
        let x = match self {
            Op::Exp | Op::Exp2 => a,
            Op::ExpHeld => a.max(-87.0),
            Op::ExpDiff => a - b,
            _ => return 8,
        };
        // A saturating cast: a huge exponent allows any finite result,
        // but its limit (∞ or 0) is judged first ([`judge`]).
        (3.0 + 2.0 * x.abs().ceil()) as u64
    }
}

/// `template` with each of the names `a`, `b`, `c` standing alone (not a
/// part of a longer name) replaced by `values`' first, second, third.
pub(super) fn substitute(template: &str, values: &[String; 3]) -> String {
    let chars: Vec<char> = template.chars().collect();
    let part = |c: Option<&char>| c.is_some_and(|c| c.is_alphanumeric() || *c == '_');
    let mut out = String::new();
    for (k, &ch) in chars.iter().enumerate() {
        let alone = !part(k.checked_sub(1).and_then(|p| chars.get(p))) && !part(chars.get(k + 1));
        match ch {
            'a' | 'b' | 'c' if alone => out += &values[(ch as u8 - b'a') as usize],
            _ => out.push(ch),
        }
    }
    out
}

/// A short form of an input: `1e30`, `-88`, `NaN`, `inf`, `1e-40`.
pub(super) fn short(x: f32) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else if x != 0.0 && (x.abs() >= 1e5 || x.abs() < 1e-3) {
        format!("{x:e}")
    } else {
        format!("{x}")
    }
}

/// How a result compares with IEEE's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The same value (NaN for NaN).
    Same,
    /// Within the ulps allowed (what WGSL allows, or a few).
    Close,
    /// A denormal (or a value that small) given as zero.
    Flushed,
    Differs,
}

impl Verdict {
    /// The verdict for the report.
    pub(super) fn label(self) -> &'static str {
        match self {
            Verdict::Same => "same",
            Verdict::Close => "close",
            Verdict::Flushed => "flushed to 0",
            Verdict::Differs => "DIFFERS",
        }
    }
}

/// The distance in ulps of two floats of a format, each `bits` of it as a
/// word whose bit `sign` is the sign (0 for two NaNs, the most for one
/// NaN; +0 and -0 are one value).
pub(super) fn ulps_of(a: f32, b: f32, bits: impl Fn(f32) -> u32, sign: u32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let ordered = |x: f32| -> i64 {
        let word = bits(x);
        let magnitude = i64::from(word & (sign - 1));
        if word & sign != 0 {
            -magnitude
        } else {
            magnitude
        }
    };
    ordered(a).abs_diff(ordered(b))
}

/// The distance in ulps of two f32s ([`ulps_of`]).
pub(super) fn ulps(a: f32, b: f32) -> u64 {
    ulps_of(a, b, f32::to_bits, 1 << 31)
}

/// How `got` compares with `want`, `allowed` ulps counting as close. A
/// limit (∞, 0) must be met exactly.
pub(super) fn judge(want: f32, got: f32, allowed: u64) -> Verdict {
    if want.is_nan() || got.is_nan() {
        return if want.is_nan() && got.is_nan() {
            Verdict::Same
        } else {
            Verdict::Differs
        };
    }
    if want == got {
        return Verdict::Same;
    }
    if want.is_infinite() || got.is_infinite() || want == 0.0 {
        return Verdict::Differs;
    }
    if want.abs() < f32::MIN_POSITIVE && got == 0.0 {
        return Verdict::Flushed;
    }
    if ulps(want, got) <= allowed {
        Verdict::Close
    } else {
        Verdict::Differs
    }
}

/// The maths kernel's prelude: the held functions every program has
/// (`common.wgsl`).
fn prelude() -> Result<String, String> {
    let common = include_str!("../common.wgsl");
    Ok(["tanh_held", "exp_held"]
        .iter()
        .map(|name| function(common, name).ok_or(format!("{name} not in common.wgsl")))
        .collect::<Result<Vec<_>, _>>()?
        .join("\n"))
}

/// The text of WGSL function `name` in `source`: from `fn name(` to its
/// closing brace at the start of a line.
pub(super) fn function<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let start = source.find(&format!("fn {name}("))?;
    let end = source[start..].find("\n}")? + start + 2;
    Some(&source[start..end])
}

/// One element of the kernel's input.
type Element = elementwise::Element<Op>;

/// One edge: the operation and inputs, IEEE's value, the GPU's.
#[derive(Clone, Debug)]
pub(super) struct Edge {
    pub(super) op: Op,
    pub(super) inputs: [f32; 3],
    pub(super) want: f32,
    pub(super) got: f32,
    pub(super) verdict: Verdict,
}

/// The edge cases.
fn edge_elements() -> Vec<Element> {
    let big = [
        1e30, -1e30, 1e4, -1e4, 100.0, -100.0, 89.0, -89.0, 88.0, -88.0, 10.0, -10.0,
    ];
    let mut e = Vec::new();
    for op in [Op::Exp, Op::Exp2, Op::Tanh] {
        e.extend(big.iter().map(|&x| (op, x, 0.0, 0.0)));
    }
    for x in [-1e30, 1e30, -100.0, 100.0] {
        e.push((Op::ExpHeld, x, 0.0, 0.0));
        e.push((Op::TanhHeld, x, 0.0, 0.0));
    }
    for (a, b) in [(-1e30, -1e30), (-1e30, 0.0), (0.0, -1e30)] {
        e.push((Op::ExpDiff, a, b, 0.0));
    }
    let (inf, nan, tiny) = (f32::INFINITY, f32::NAN, 1e-40_f32);
    for x in [0.0, -0.0, -1.0, tiny, inf] {
        e.push((Op::Log, x, 0.0, 0.0));
    }
    e.extend([
        (Op::Sqrt, -1.0, 0.0, 0.0),
        (Op::InverseSqrt, 0.0, 0.0, 0.0),
        (Op::InverseSqrt, -1.0, 0.0, 0.0),
        (Op::Add, inf, 1.0, 0.0),
        (Op::Sub, inf, inf, 0.0),
        (Op::Mul, inf, 0.0, 0.0),
        (Op::Add, nan, 1.0, 0.0),
        (Op::Mul, nan, 0.0, 0.0),
        (Op::Div, 1.0, 0.0, 0.0),
        (Op::Div, -1.0, 0.0, 0.0),
        (Op::Div, 0.0, 0.0, 0.0),
        (Op::Min, nan, 1.0, 0.0),
        (Op::Min, 1.0, nan, 0.0),
        (Op::Max, nan, 1.0, 0.0),
        (Op::Max, 1.0, nan, 0.0),
        (Op::Clamp, nan, 0.0, 1.0),
        (Op::SelectLess, nan, 2.0, 3.0),
        (Op::IsNan, nan, 0.0, 0.0),
        (Op::IsNan, 1.0, 0.0, 0.0),
        (Op::Fma, inf, 0.0, 1.0),
        (Op::Copy, tiny, 0.0, 0.0),
        (Op::Mul, tiny, 1.0, 0.0),
        (Op::Mul, f32::MIN_POSITIVE, 0.5, 0.0),
        (Op::Add, tiny, 0.0, 0.0),
        (Op::Mul, tiny, 1e10, 0.0),
    ]);
    e
}

/// The edge cases on `gpu`.
pub(super) fn edges(gpu: &Gpu) -> Result<Vec<Edge>, Miss> {
    let elements = edge_elements();
    let got = elementwise::run(gpu, &prelude()?, "f32", &OPS, &elements)?;
    Ok(elements
        .iter()
        .zip(got)
        .map(|(&(op, a, b, c), got)| {
            let want = op.reference(f64::from(a), f64::from(b), f64::from(c)) as f32;
            Edge {
                op,
                inputs: [a, b, c],
                want,
                got,
                verdict: judge(want, got, op.allowed(a, b)),
            }
        })
        .collect())
}

/// Whether an `exp`, `exp2` or `tanh` edge broke: its result differs
/// ([`judge`]), that is it missed IEEE's limit (∞, 0, ±1, met exactly)
/// or WGSL's bound (3 + 2|x| ulps for the `exp`s, a few for `tanh`). A
/// denormal flushed to 0 is no break.
pub(super) fn breaks(e: &Edge) -> bool {
    matches!(e.op, Op::Exp | Op::Exp2 | Op::Tanh | Op::ExpDiff) && e.verdict == Verdict::Differs
}

/// Whether any edge broke ([`breaks`]).
pub(super) fn fast_maths_breaks(edges: &[Edge]) -> bool {
    edges.iter().any(breaks)
}

/// The edge cases as report lines.
pub(super) fn edge_lines(edges: &[Edge]) -> Vec<String> {
    let mut rows = vec![vec![
        "expression".to_string(),
        "IEEE".into(),
        "got".into(),
        "".into(),
    ]];
    for e in edges {
        let [a, b, c] = e.inputs;
        rows.push(vec![
            e.op.shown(a, b, c),
            value(e.want),
            value(e.got),
            e.verdict.label().into(),
        ]);
    }
    let mut lines = table(&rows);
    let differ = edges
        .iter()
        .filter(|e| e.verdict == Verdict::Differs)
        .count();
    let broken: Vec<String> = edges
        .iter()
        .filter(|e| breaks(e))
        .map(|e| {
            let [a, b, c] = e.inputs;
            e.op.shown(a, b, c)
        })
        .collect();
    lines.push(format!(
        "verdict: {differ} of {} differ from IEEE",
        edges.len()
    ));
    lines.push(if broken.is_empty() {
        "exp, exp2 and tanh meet their limits and WGSL's bound here".into()
    } else {
        format!(
            "these miss their limits or WGSL's bound here (fast maths): hold the arguments: {}",
            broken.join(", ")
        )
    });
    lines
}

/// How inputs are spread over a range.
#[derive(Clone, Copy, Debug)]
enum Spread {
    Linear,
    /// Evenly in the logarithm (a positive range).
    Log,
}

/// One function's error over a range.
#[derive(Clone, Debug)]
pub(super) struct Accuracy {
    op: Op,
    range: (f64, f64),
    /// The largest error in ulps (where the result is a normal f32), the
    /// largest absolute error, and the input with the most ulps.
    ulps: u64,
    absolute: f64,
    worst: f32,
    /// What the WGSL specification allows.
    allowed: &'static str,
}

/// Inputs a range.
const SAMPLES: usize = 16384;

/// The ranges: (operation, range, spread, the second input's range, what
/// WGSL allows).
type Span = (Op, (f64, f64), Spread, Option<(f64, f64)>, &'static str);

const SPANS: [Span; 11] = [
    (Op::Exp, (-10.0, 10.0), Spread::Linear, None, "3 + 2|x| ulp"),
    (Op::Exp, (-87.0, 88.0), Spread::Linear, None, "3 + 2|x| ulp"),
    (
        Op::Exp2,
        (-126.0, 127.0),
        Spread::Linear,
        None,
        "3 + 2|x| ulp",
    ),
    (
        Op::Tanh,
        (-10.0, 10.0),
        Spread::Linear,
        None,
        "as sinh/cosh",
    ),
    (
        Op::Log,
        (1e-30, 1e30),
        Spread::Log,
        None,
        "3 ulp; 2^-21 abs in [0.5, 2]",
    ),
    (
        Op::Sqrt,
        (1e-30, 1e30),
        Spread::Log,
        None,
        "as 1/inverseSqrt",
    ),
    (Op::InverseSqrt, (1e-30, 1e30), Spread::Log, None, "2 ulp"),
    (
        Op::Sin,
        (-std::f64::consts::PI, std::f64::consts::PI),
        Spread::Linear,
        None,
        "2^-11 abs",
    ),
    (
        Op::Sin,
        (-1e4, 1e4),
        Spread::Linear,
        None,
        "none outside [-pi, pi]",
    ),
    (
        Op::Cos,
        (-std::f64::consts::PI, std::f64::consts::PI),
        Spread::Linear,
        None,
        "2^-11 abs",
    ),
    (
        Op::Pow,
        (0.1, 10.0),
        Spread::Linear,
        Some((-8.0, 8.0)),
        "as exp2(b * log2(a))",
    ),
];

/// Each function's error over its ranges on `gpu`.
pub(super) fn accuracy(gpu: &Gpu) -> Result<Vec<Accuracy>, Miss> {
    let mut noise = Noise::new(0x5eed);
    let mut elements = Vec::with_capacity(SPANS.len() * SAMPLES);
    for &(op, (lo, hi), spread, second, _) in &SPANS {
        for k in 0..SAMPLES {
            let t = (k as f64 + noise.unit()) / SAMPLES as f64;
            let a = match spread {
                Spread::Linear => lo + (hi - lo) * t,
                Spread::Log => (lo.ln() + (hi.ln() - lo.ln()) * t).exp(),
            };
            let b = second.map_or(0.0, |(lo, hi)| lo + (hi - lo) * noise.unit());
            elements.push((op, a as f32, b as f32, 0.0));
        }
    }
    let got = elementwise::run(gpu, &prelude()?, "f32", &OPS, &elements)?;
    Ok(SPANS
        .iter()
        .enumerate()
        .map(|(s, &(op, range, _, _, allowed))| {
            let at = s * SAMPLES..(s + 1) * SAMPLES;
            let (ulps, absolute, worst) = worst(&elements[at.clone()], &got[at]);
            Accuracy {
                op,
                range,
                ulps,
                absolute,
                worst,
                allowed,
            }
        })
        .collect())
}

/// The largest error in ulps (over results that are normal f32s), the
/// largest absolute error, and the input with the most ulps, of `got`
/// against the f64 reference of `elements`.
fn worst(elements: &[Element], got: &[f32]) -> (u64, f64, f32) {
    let mut most = (0, 0.0, 0.0);
    for (&(op, a, b, c), &g) in elements.iter().zip(got) {
        let exact = op.reference(f64::from(a), f64::from(b), f64::from(c));
        let want = exact as f32;
        let absolute = (exact - f64::from(g)).abs();
        let absolute = if absolute.is_nan() {
            f64::INFINITY
        } else {
            absolute
        };
        if absolute > most.1 {
            most.1 = absolute;
        }
        if want.is_normal() {
            let u = ulps(want, g);
            if u > most.0 {
                most.0 = u;
                most.2 = a;
            }
        }
    }
    most
}

/// The accuracy as report lines.
pub(super) fn accuracy_lines(accuracy: &[Accuracy]) -> Vec<String> {
    let mut rows = vec![vec![
        "function".to_string(),
        "range".into(),
        "max ulp".into(),
        "max abs error".into(),
        "worst at".into(),
        "WGSL allows".into(),
    ]];
    for a in accuracy {
        rows.push(vec![
            a.op.wgsl().to_string(),
            format!("[{}, {}]", short(a.range.0 as f32), short(a.range.1 as f32)),
            ulps_cell(a.ulps),
            format!("{:.2e}", a.absolute),
            short(a.worst),
            a.allowed.into(),
        ]);
    }
    table(&rows)
}
