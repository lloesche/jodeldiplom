//! What an index past the end of a storage binding does on this GPU
//! ([`probe`]): a read gives zero, the binding's last element (clamped),
//! the bytes that are really there, or something else; a write is
//! dropped, lands on the last element, or goes past the binding. For
//! `array<f32>` and `array<vec4<f32>>`, the binding the first 777 words of
//! a buffer of 1024 (3108 bytes: 194 whole vectors and three words of a
//! 195th), and a buffer of exactly 777 words bound whole.

use super::gpu::{bits, Bound, Gpu};
use super::text::{table, value};
use super::Miss;

/// The words in the binding, and in the buffer under it.
pub(super) const BOUND: u32 = 777;
const BUFFER: u32 = 1024;

/// The value the probe's buffer holds at word `k`.
pub(super) fn held(k: u32) -> f32 {
    1000.0 + k as f32
}

/// The marker an out-of-bounds write stores.
const MARK: f32 = -7777.0;

/// What an out-of-bounds read gave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Read {
    /// The index was in bounds and the value right (the controls).
    Right,
    Zero,
    /// The binding's last element (wgpu's clamp).
    Last,
    /// The value really in the buffer there, past the binding.
    Real,
    Other(f32),
}

/// What an out-of-bounds write did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Wrote {
    Dropped,
    /// Onto the binding's last element.
    Last,
    /// Past the binding, where the index says.
    Past,
    /// Somewhere else.
    Elsewhere,
}

/// One read: the element type, how the buffer is bound, the index, what
/// came back (a word each, four for a vector), and what that is.
#[derive(Clone, Debug)]
pub(super) struct ReadCase {
    vector: bool,
    exact: bool,
    index: u32,
    got: Vec<f32>,
    read: Read,
}

/// One write: the element type, the index, what it did.
#[derive(Clone, Debug)]
pub(super) struct WriteCase {
    vector: bool,
    index: u32,
    wrote: Wrote,
}

/// What the probe found.
#[derive(Clone, Debug)]
pub(super) struct Bounds {
    reads: Vec<ReadCase>,
    writes: Vec<WriteCase>,
}

/// The indices read: the last in bounds first (a control), then past it.
const F32_READS: [u32; 5] = [776, 777, 800, 1023, 100_000];
const VEC4_READS: [u32; 5] = [193, 194, 200, 255, 100_000];
const F32_WRITES: [u32; 2] = [777, 1000];
const VEC4_WRITES: [u32; 2] = [194, 200];

fn read_source(vector: bool) -> String {
    let ty = if vector { "vec4<f32>" } else { "f32" };
    format!(
        "@group(0) @binding(0) var<storage, read> data: array<{ty}>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    out[g.x] = data[idx[g.x]];
}}
"
    )
}

fn write_source(vector: bool) -> String {
    let (ty, mark) = if vector {
        ("vec4<f32>", format!("vec4<f32>({MARK:?})"))
    } else {
        ("f32", format!("{MARK:?}"))
    };
    format!(
        "@group(0) @binding(0) var<storage, read_write> data: array<{ty}>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@compute @workgroup_size(1)
fn main() {{
    data[idx[0]] = {mark};
}}
"
    )
}

/// Every read and write on `gpu`.
pub(super) fn probe(gpu: &Gpu) -> Result<Bounds, Miss> {
    let full: Vec<f32> = (0..BUFFER).map(held).collect();
    let mut reads = Vec::new();
    for vector in [false, true] {
        let kernel = gpu.kernel(&read_source(vector), "main")?;
        let indices: &[u32] = if vector { &VEC4_READS } else { &F32_READS };
        for exact in [false, true] {
            let data = if exact {
                gpu.words(&bits(&full[..BOUND as usize]))?
            } else {
                gpu.words(&bits(&full))?
            };
            let idx = gpu.words(indices)?;
            let width = if vector { 4 } else { 1 };
            let out = gpu.zeros(indices.len() as u64 * 4 * width)?;
            gpu.run(
                &kernel,
                &[
                    Bound::First(&data, u64::from(BOUND) * 4),
                    Bound::All(&idx),
                    Bound::All(&out),
                ],
                [indices.len() as u32, 1, 1],
                1,
            )?;
            let got = gpu.read_f32(&out)?;
            for (n, &index) in indices.iter().enumerate() {
                let got = got[n * width as usize..(n + 1) * width as usize].to_vec();
                reads.push(ReadCase {
                    vector,
                    exact,
                    index,
                    read: judge_read(vector, exact, index, &got),
                    got,
                });
            }
        }
    }
    let mut writes = Vec::new();
    for vector in [false, true] {
        let kernel = gpu.kernel(&write_source(vector), "main")?;
        let indices: &[u32] = if vector { &VEC4_WRITES } else { &F32_WRITES };
        for &index in indices {
            let data = gpu.words(&bits(&full))?;
            let idx = gpu.words(&[index])?;
            gpu.run(
                &kernel,
                &[Bound::First(&data, u64::from(BOUND) * 4), Bound::All(&idx)],
                [1, 1, 1],
                1,
            )?;
            let after = gpu.read_f32(&data)?;
            writes.push(WriteCase {
                vector,
                index,
                wrote: judge_write(vector, index, &after),
            });
        }
    }
    Ok(Bounds { reads, writes })
}

/// The elements of a binding of [`BOUND`] words as `f32`s or `vec4`s.
fn elements(vector: bool) -> u32 {
    if vector {
        BOUND / 4
    } else {
        BOUND
    }
}

/// What a read of element `index` that gave `got` is: in bounds and
/// right, or past the end and zero, the last element, the real bytes
/// (where the buffer has them past the binding: not `exact`), or other.
pub(super) fn judge_read(vector: bool, exact: bool, index: u32, got: &[f32]) -> Read {
    let width = if vector { 4 } else { 1 };
    let words =
        |element: u32| -> Vec<f32> { (0..width).map(|w| held(element * width + w)).collect() };
    if index < elements(vector) {
        return if got == words(index).as_slice() {
            Read::Right
        } else {
            Read::Other(got[0])
        };
    }
    if got.iter().all(|&v| v == 0.0) {
        Read::Zero
    } else if got == words(elements(vector) - 1).as_slice() {
        Read::Last
    } else if !exact && (index + 1) * width <= BUFFER && got == words(index).as_slice() {
        Read::Real
    } else if exact && vector && index == elements(vector) && got[0] == held(index * width) {
        // The partial vector's one word that is in the buffer.
        Read::Real
    } else {
        Read::Other(got.iter().copied().find(|&v| v != 0.0).unwrap_or(0.0))
    }
}

/// What a write of the marker at element `index` did, from the buffer's
/// words `after`.
pub(super) fn judge_write(vector: bool, index: u32, after: &[f32]) -> Wrote {
    let width = if vector { 4 } else { 1 };
    let changed: Vec<u32> = (0..after.len() as u32)
        .filter(|&k| after[k as usize] != held(k))
        .collect();
    let span = |element: u32| -> Vec<u32> { (element * width..(element + 1) * width).collect() };
    if changed.is_empty() {
        Wrote::Dropped
    } else if changed == span(elements(vector) - 1) {
        Wrote::Last
    } else if changed == span(index) {
        Wrote::Past
    } else {
        Wrote::Elsewhere
    }
}

/// Whether an out-of-bounds read gave the binding's last element.
pub(super) fn clamps(b: &Bounds) -> bool {
    b.reads.iter().any(|r| r.read == Read::Last)
}

/// The probe's findings as report lines.
pub(super) fn lines(b: &Bounds) -> Vec<String> {
    let mut rows = vec![vec![
        "read of".to_string(),
        "binding".into(),
        "index".into(),
        "got".into(),
        "which is".into(),
    ]];
    for r in &b.reads {
        let ty = if r.vector {
            "array<vec4<f32>>"
        } else {
            "array<f32>"
        };
        let binding = if r.exact {
            "a 777-word buffer, whole"
        } else {
            "777 of 1024 words"
        };
        let last = elements(r.vector) - 1;
        let index = if r.index <= last {
            format!("{} (last in bounds)", r.index)
        } else {
            r.index.to_string()
        };
        let got = r
            .got
            .iter()
            .map(|&v| value(v))
            .collect::<Vec<_>>()
            .join(", ");
        let which = match r.read {
            Read::Right => "right".into(),
            Read::Zero => "zero".into(),
            Read::Last => format!("the last element ({last}): clamped"),
            Read::Real => "the bytes really there".into(),
            Read::Other(v) => format!("something else ({})", value(v)),
        };
        rows.push(vec![ty.into(), binding.into(), index, got, which]);
    }
    let mut lines = table(&rows);
    lines.push(String::new());
    let mut rows = vec![vec![
        "write to".to_string(),
        "index".into(),
        "what it did".into(),
    ]];
    for w in &b.writes {
        let ty = if w.vector {
            "array<vec4<f32>>"
        } else {
            "array<f32>"
        };
        let did = match w.wrote {
            Wrote::Dropped => "dropped".to_string(),
            Wrote::Last => format!(
                "wrote the last element ({}): clamped",
                elements(w.vector) - 1
            ),
            Wrote::Past => "wrote past the binding, where it said".into(),
            Wrote::Elsewhere => "wrote somewhere else".into(),
        };
        rows.push(vec![ty.into(), w.index.to_string(), did]);
    }
    lines.extend(table(&rows));
    lines.push(if clamps(b) {
        "verdict: an index past the end reads the LAST element here: a binding read as vectors must hold whole vectors".into()
    } else {
        "verdict: no read past the end gave the last element here".into()
    });
    lines
}
