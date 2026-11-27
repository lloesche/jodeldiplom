//! One operation an element on the GPU ([`run`]), as the maths and f16
//! probes take them: each element an [`Operation`] and three inputs (f32s,
//! made the kernel's scalar type), one result out as an f32. The inputs
//! come from a buffer, so the compiler folds nothing.

use super::gpu::{Bound, Gpu};

/// An operation of an elementwise kernel: its code in the kernel, and its
/// expression on `a`, `b`, `c`.
pub(super) trait Operation: Copy {
    fn code(self) -> u32;
    fn wgsl(self) -> &'static str;
}

/// An element: its operation and inputs `a`, `b`, `c`.
pub(super) type Element<O> = (O, f32, f32, f32);

/// The kernel: `prelude` (enables, functions), the inputs made `scalar`s,
/// and a case for each of `ops`; an unknown code gives 0.
fn source<O: Operation>(prelude: &str, scalar: &str, ops: &[O]) -> String {
    let cases: String = ops
        .iter()
        .map(|op| format!("        case {}u: {{ r = {}; }}\n", op.code(), op.wgsl()))
        .collect();
    format!(
        "{prelude}
@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {{
    let at = g.x + g.y * n.x * 256u;
    if (at >= arrayLength(&out)) {{ return; }}
    let e = input[at];
    let a = {scalar}(bitcast<f32>(e.y));
    let b = {scalar}(bitcast<f32>(e.z));
    let c = {scalar}(bitcast<f32>(e.w));
    var r = {scalar}(0.0);
    switch e.x {{
{cases}        default: {{ r = {scalar}(0.0); }}
    }}
    out[at] = f32(r);
}}
"
    )
}

/// The results of `elements` (each an operation of `ops` and its inputs
/// `a`, `b`, `c`) on `gpu`.
pub(super) fn run<O: Operation>(
    gpu: &Gpu,
    prelude: &str,
    scalar: &str,
    ops: &[O],
    elements: &[Element<O>],
) -> Result<Vec<f32>, String> {
    let kernel = gpu.kernel(&source(prelude, scalar, ops), "main")?;
    let words: Vec<u32> = elements
        .iter()
        .flat_map(|&(op, a, b, c)| [op.code(), a.to_bits(), b.to_bits(), c.to_bits()])
        .collect();
    let input = gpu.words(&words)?;
    let out = gpu.zeros(elements.len() as u64 * 4)?;
    let groups = (elements.len() as u32).div_ceil(256);
    gpu.run(
        &kernel,
        &[Bound::All(&input), Bound::All(&out)],
        [groups.min(65535), groups.div_ceil(65535), 1],
        1,
    )?;
    gpu.read_f32(&out)
}
