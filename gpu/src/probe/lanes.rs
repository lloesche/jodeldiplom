//! Whether writing different lanes of one vector from different
//! invocations loses values on this GPU ([`probe`]): in workgroup memory
//! and in a storage buffer, for `vec4<f32>`, `vec2<f32>` and `vec4<u32>`,
//! each lane chosen by index or by name, the lanes of a vector written by
//! neighbouring invocations (one subgroup) or by invocations a workgroup's
//! vectors apart (64 for a `vec4`, 128 for a `vec2`: different subgroups);
//! and, as the control, the rules' own way (an array of `f32`s, each
//! written whole).

use super::gpu::{Bound, Gpu};
use super::text::{share, table};
use super::Miss;

/// Where the vectors are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Memory {
    Workgroup,
    Storage,
}

/// The vector written a lane at a time (`F32` is the control: no vector).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Vector {
    Vec4F32,
    Vec2F32,
    Vec4U32,
    F32,
}

impl Vector {
    fn lanes(self) -> u32 {
        match self {
            Vector::Vec4F32 | Vector::Vec4U32 => 4,
            Vector::Vec2F32 => 2,
            Vector::F32 => 1,
        }
    }

    fn wgsl(self) -> &'static str {
        match self {
            Vector::Vec4F32 => "vec4<f32>",
            Vector::Vec2F32 => "vec2<f32>",
            Vector::Vec4U32 => "vec4<u32>",
            Vector::F32 => "f32",
        }
    }

    /// The scalar of a value `v` (a `u32` expression) as stored, and the
    /// stored scalar `s` as the `u32` read back.
    fn put(self, v: &str) -> String {
        match self {
            Vector::Vec4U32 => v.to_string(),
            _ => format!("f32({v})"),
        }
    }

    fn get(self, s: &str) -> String {
        match self {
            Vector::Vec4U32 => s.to_string(),
            _ => format!("u32({s})"),
        }
    }
}

/// Which invocations write the lanes of a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// Neighbours: invocation `i` writes lane `i % n` of vector `i / n`.
    Adjacent,
    /// Far apart: invocation `i` writes lane `i / V` of vector `i % V`
    /// (`V` the vectors of a workgroup: 64 for a `vec4`), so each vector's
    /// lanes come from different subgroups.
    Strided,
}

/// How a lane is named in the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Store {
    /// `v[j][l] = x`.
    Index,
    /// `v[j].x = x` and so on, through a `switch` on the lane.
    Name,
    /// The control: `v[j * n + l] = x` into an array of scalars.
    Whole,
}

/// One case and what it lost.
#[derive(Clone, Debug)]
pub(super) struct Case {
    memory: Memory,
    vector: Vector,
    layout: Layout,
    store: Store,
    /// Lanes written over all the repetitions, and those read back wrong.
    lanes: u64,
    wrong: u64,
}

/// Invocations a workgroup, workgroups a run, runs a case.
const INVOCATIONS: u32 = 256;
const GROUPS: u32 = 2048;
const RUNS: usize = 4;

/// Every case on `gpu`.
pub(super) fn probe(gpu: &Gpu) -> Result<Vec<Case>, Miss> {
    let mut cases = Vec::new();
    for memory in [Memory::Workgroup, Memory::Storage] {
        for vector in [Vector::Vec4F32, Vector::Vec2F32, Vector::Vec4U32] {
            for layout in [Layout::Adjacent, Layout::Strided] {
                cases.push(run(gpu, memory, vector, layout, Store::Index)?);
            }
        }
        for layout in [Layout::Adjacent, Layout::Strided] {
            cases.push(run(gpu, memory, Vector::Vec4F32, layout, Store::Name)?);
        }
        cases.push(run(
            gpu,
            memory,
            Vector::F32,
            Layout::Adjacent,
            Store::Whole,
        )?);
    }
    Ok(cases)
}

/// The vector and lane invocation `i` writes (`j`, `l`: WGSL expressions).
fn place(vector: Vector, layout: Layout) -> (String, String) {
    let n = vector.lanes();
    let count = INVOCATIONS / n;
    match layout {
        Layout::Adjacent => (format!("(i / {n}u)"), format!("(i % {n}u)")),
        Layout::Strided => (format!("(i % {count}u)"), format!("(i / {count}u)")),
    }
}

/// The store of `x` into lane `l` of vector `j` of `v`.
fn store(vector: Vector, s: Store, j: &str, l: &str, x: &str) -> String {
    let n = vector.lanes();
    match s {
        Store::Index => format!("v[{j}][{l}] = {x};"),
        Store::Name => {
            let names = ["x", "y", "z", "w"];
            let mut text = format!("switch {l} {{");
            for (k, name) in names.iter().enumerate().take(n as usize) {
                if k + 1 == n as usize {
                    text += &format!(" default: {{ v[{j}].{name} = {x}; }}");
                } else {
                    text += &format!(" case {k}u: {{ v[{j}].{name} = {x}; }}");
                }
            }
            text + " }"
        }
        Store::Whole => format!("v[{j} * {n}u + {l}] = {x};"),
    }
}

/// The kernel of a case.
///
/// The stores into one lane of a vector here are what the project's
/// kernels must never do (gpu/README.md, "Rules for kernels"), written on
/// purpose to see whether this GPU loses lanes. They are made as text when
/// the probe runs, in this Rust file and not in a `.wgsl` one, so
/// `scripts/check-rules.sh` (which holds every tracked WGSL file to the
/// rule) holds the project's kernels to it without an exception for these.
fn source(memory: Memory, vector: Vector, layout: Layout, s: Store) -> String {
    let n = vector.lanes();
    let count = INVOCATIONS / n;
    let (j, l) = place(vector, layout);
    let ty = vector.wgsl();
    let value = vector.put("value");
    let read = |at: &str| {
        let lane = match s {
            Store::Whole => format!("v[{at} * {n}u + {l}]"),
            _ => format!("v[{at}][{l}]"),
        };
        vector.get(&lane)
    };
    // The control's array holds scalars, a lane each.
    let (elem, elements) = if s == Store::Whole {
        ("f32", INVOCATIONS)
    } else {
        (ty, count)
    };
    match memory {
        Memory::Workgroup => {
            let put = store(vector, s, &j, &l, &value);
            let got = read(&j);
            format!(
                "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
var<workgroup> v: array<{elem}, {elements}>;
@compute @workgroup_size({INVOCATIONS})
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {{
    let value = w.x * {INVOCATIONS}u + i + 1u;
    {put}
    workgroupBarrier();
    out[w.x * {INVOCATIONS}u + i] = {got};
}}
"
            )
        }
        Memory::Storage => {
            let at = format!("(w.x * {elements}u + {j})");
            let put = store(vector, s, &at, &l, &value);
            format!(
                "@group(0) @binding(0) var<storage, read_write> v: array<{elem}>;
@compute @workgroup_size({INVOCATIONS})
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {{
    let value = w.x * {INVOCATIONS}u + i + 1u;
    {put}
}}
"
            )
        }
    }
}

/// What lane `k` of the buffer of a storage case should hold: the value
/// of the invocation that writes it.
pub(super) fn expected(vector: Vector, layout: Layout, k: u32) -> u32 {
    let n = vector.lanes();
    let count = INVOCATIONS / n;
    let (group, within) = (k / INVOCATIONS, k % INVOCATIONS);
    let (j, l) = (within / n, within % n);
    let i = match layout {
        Layout::Adjacent => j * n + l,
        Layout::Strided => l * count + j,
    };
    group * INVOCATIONS + i + 1
}

/// The word read back as the value it stands for (a float holds the
/// probe's values exactly; `Vec4U32` is a plain word).
fn word(vector: Vector, w: u32) -> u32 {
    match vector {
        Vector::Vec4U32 => w,
        _ => {
            let f = f32::from_bits(w);
            if f.is_finite() && f >= 0.0 {
                f as u32
            } else {
                u32::MAX
            }
        }
    }
}

fn run(
    gpu: &Gpu,
    memory: Memory,
    vector: Vector,
    layout: Layout,
    s: Store,
) -> Result<Case, String> {
    let kernel = gpu.kernel(&source(memory, vector, layout, s), "main")?;
    let total = GROUPS * INVOCATIONS;
    let buffer = gpu.zeros(u64::from(total) * 4)?;
    let mut wrong = 0;
    for _ in 0..RUNS {
        gpu.clear(&buffer)?;
        gpu.run(&kernel, &[Bound::All(&buffer)], [GROUPS, 1, 1], 1)?;
        let words = gpu.read(&buffer)?;
        // A workgroup case's buffer holds what each invocation read back,
        // as a u32; a storage case's the lanes themselves.
        let as_read = match memory {
            Memory::Workgroup => Vector::Vec4U32,
            Memory::Storage if s == Store::Whole => Vector::F32,
            Memory::Storage => vector,
        };
        wrong += (0..total)
            .zip(&words)
            .filter(|&(k, &w)| {
                let want = match memory {
                    // Each invocation reads back its own lane.
                    Memory::Workgroup => k + 1,
                    Memory::Storage if s == Store::Whole => k + 1,
                    Memory::Storage => expected(vector, layout, k),
                };
                word(as_read, w) != want
            })
            .count() as u64;
    }
    Ok(Case {
        memory,
        vector,
        layout,
        store: s,
        lanes: u64::from(total) * RUNS as u64,
        wrong,
    })
}

/// Whether any case but the control lost lanes.
pub(super) fn races(cases: &[Case]) -> bool {
    cases.iter().any(|c| c.store != Store::Whole && c.wrong > 0)
}

/// The cases as report lines.
pub(super) fn lines(cases: &[Case]) -> Vec<String> {
    let mut rows = vec![vec![
        "memory".to_string(),
        "type".into(),
        "lanes from".into(),
        "store".into(),
        "lanes lost".into(),
    ]];
    for c in cases {
        rows.push(vec![
            format!("{:?}", c.memory).to_lowercase(),
            c.vector.wgsl().to_string(),
            match c.layout {
                Layout::Adjacent => "neighbours".into(),
                Layout::Strided => format!("{} apart", INVOCATIONS / c.vector.lanes()),
            },
            match c.store {
                Store::Index => "v[j][l] = x",
                Store::Name => "v[j].x = x",
                Store::Whole => "f32s, whole (control)",
            }
            .into(),
            share(c.wrong, c.lanes),
        ]);
    }
    let mut lines = table(&rows);
    lines.push(if races(cases) {
        "verdict: lane writes RACE here (a store to one lane writes the whole vector back)".into()
    } else {
        "verdict: no lane was lost here".into()
    });
    lines
}
