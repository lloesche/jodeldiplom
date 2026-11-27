//! Barriers and atomics ([`probe`]): message passing across a barrier
//! (each invocation writes its value, then after `workgroupBarrier` or
//! `storageBarrier` reads another's), a flag set with an atomic after the
//! data, and `atomicAdd`: counts exact, every old value handed out once,
//! a workgroup's count added on.

use super::gpu::{Bound, Gpu};
use super::{Check, Miss};

const GROUPS: u32 = 4096;
const RUNS: usize = 4;

/// The other invocation an invocation reads.
const OTHER: &str = "(i * 97u + 13u) % 256u";

fn workgroup_source() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> bad: array<atomic<u32>>;
var<workgroup> s: array<u32, 256>;
var<workgroup> data: u32;
var<workgroup> flag: atomic<u32>;
fn seed(w: u32, i: u32) -> u32 {{ return (w * 2654435761u) ^ (i * 40503u) ^ 0x9e37u; }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {{
    s[i] = seed(w.x, i);
    if (i == 0u) {{
        data = seed(w.x, 999u);
        atomicStore(&flag, 1u);
    }}
    workgroupBarrier();
    let o = {OTHER};
    if (s[o] != seed(w.x, o)) {{ atomicAdd(&bad[0], 1u); }}
    if (atomicLoad(&flag) != 1u || data != seed(w.x, 999u)) {{ atomicAdd(&bad[1], 1u); }}
}}
"
    )
}

fn storage_source() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> bad: array<atomic<u32>>;
fn seed(w: u32, i: u32) -> u32 {{ return (w * 2654435761u) ^ (i * 40503u) ^ 0x9e37u; }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {{
    buf[w.x * 256u + i] = seed(w.x, i);
    storageBarrier();
    let o = {OTHER};
    if (buf[w.x * 256u + o] != seed(w.x, o)) {{ atomicAdd(&bad[0], 1u); }}
}}
"
    )
}

/// The atomics kernel (`here` starts at 0: wgpu zeroes workgroup memory
/// for the probe's kernels unless told not to).
const ATOMICS: &str = "@group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> handed: array<u32>;
var<workgroup> here: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {
    let old = atomicAdd(&counts[0], 1u);
    handed[old % arrayLength(&handed)] = w.x * 256u + i + 1u;
    atomicAdd(&here, 1u);
    workgroupBarrier();
    if (i == 0u) {
        atomicAdd(&counts[1], atomicLoad(&here));
    }
}
";

/// Every check on `gpu`.
pub(super) fn probe(gpu: &Gpu) -> Result<Vec<Check>, Miss> {
    let total = u64::from(GROUPS) * 256;
    let runs = RUNS as u64;
    let check = |name, cases, wrong| Check { name, cases, wrong };
    let mut checks = Vec::new();

    let kernel = gpu.kernel(&workgroup_source(), "main")?;
    let bad = gpu.zeros(8)?;
    for _ in 0..RUNS {
        gpu.run(&kernel, &[Bound::All(&bad)], [GROUPS, 1, 1], 1)?;
    }
    let found = gpu.read(&bad)?;
    checks.push(check(
        "workgroup memory across workgroupBarrier",
        total * runs,
        u64::from(found[0]),
    ));
    checks.push(check(
        "data then an atomic flag, across workgroupBarrier",
        total * runs,
        u64::from(found[1]),
    ));

    let kernel = gpu.kernel(&storage_source(), "main")?;
    let buf = gpu.zeros(total * 4)?;
    let bad = gpu.zeros(4)?;
    for _ in 0..RUNS {
        gpu.clear(&buf)?;
        gpu.run(
            &kernel,
            &[Bound::All(&buf), Bound::All(&bad)],
            [GROUPS, 1, 1],
            1,
        )?;
    }
    let found = gpu.read(&bad)?;
    checks.push(check(
        "a storage buffer across storageBarrier",
        total * runs,
        u64::from(found[0]),
    ));

    let kernel = gpu.kernel(ATOMICS, "main")?;
    let counts = gpu.zeros(8)?;
    let handed = gpu.zeros(total * 4)?;
    gpu.run(
        &kernel,
        &[Bound::All(&counts), Bound::All(&handed)],
        [GROUPS, 1, 1],
        1,
    )?;
    let c = gpu.read(&counts)?;
    let h = gpu.read(&handed)?;
    checks.push(check(
        "atomicAdd: the count",
        total,
        u64::from(c[0]).abs_diff(total),
    ));
    checks.push(check(
        "atomicAdd: workgroup counts added",
        total,
        u64::from(c[1]).abs_diff(total),
    ));
    checks.push(check("atomicAdd: each old value once", total, unhanded(&h)));
    Ok(checks)
}

/// The slots of `handed` no invocation got, or got twice (each holds the
/// one that got it, from 1; 0 is none).
pub(super) fn unhanded(handed: &[u32]) -> u64 {
    let mut seen = vec![false; handed.len() + 1];
    let mut wrong = 0;
    for &h in handed {
        let h = h as usize;
        if h == 0 || h > handed.len() || seen[h] {
            wrong += 1;
        } else {
            seen[h] = true;
        }
    }
    wrong
}
