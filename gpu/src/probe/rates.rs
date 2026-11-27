//! Throughput micro-benchmarks ([`probe`]), each about 40 ms: f32 and f16
//! multiply-adds, `dot4I8Packed` against the same done by hand (four
//! bytes unpacked and multiplied), workgroup memory read, and device
//! memory read, written and copied; and what wgpu's zeroing of workgroup
//! memory costs a dispatch ([`zeroing`]). They say what the hardware does
//! well, so kernels can be chosen per GPU. Every benchmark stores what it
//! computed, each chain of it from its own seeds, so a compiler can
//! neither drop nor merge the work counted.

use std::time::Duration;

use super::gpu::{Bound, Gpu};
use super::text::{si, table};
use super::{needs, Found, Miss};

/// Workgroups of 256 and loop rounds of the arithmetic benchmarks.
const GROUPS: u32 = 1024;
const ROUNDS: u32 = 1024;

/// How long each rate is measured for.
pub(super) const TARGET: Duration = Duration::from_millis(40);

/// What a rate counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    F32Fma,
    F16Fma,
    Dot4I8,
    Dot4ByHand,
    WorkgroupRead,
    DeviceRead,
    DeviceWrite,
    DeviceCopy,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::F32Fma => "f32 fma (vec4)",
            Kind::F16Fma => "f16 fma (vec4)",
            Kind::Dot4I8 => "dot4I8Packed",
            Kind::Dot4ByHand => "the same, bytes unpacked by hand",
            Kind::WorkgroupRead => "workgroup memory read",
            Kind::DeviceRead => "device memory read",
            Kind::DeviceWrite => "device memory write",
            Kind::DeviceCopy => "device memory copy (read + write)",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Kind::F32Fma | Kind::F16Fma => "FLOP/s",
            Kind::Dot4I8 | Kind::Dot4ByHand => "MAC/s",
            _ => "B/s",
        }
    }
}

/// The rates measured, and the device memory benchmarks' buffers (bytes).
#[derive(Clone, Debug)]
pub(super) struct Rates {
    rates: Vec<(Kind, Found<f64>)>,
    device_bytes: u64,
}

impl Rates {
    /// The rate of `kind`, where it was measured.
    pub(super) fn of(&self, kind: Kind) -> Option<f64> {
        self.rates
            .iter()
            .find(|(k, _)| *k == kind)
            .and_then(|(_, rate)| rate.measured().copied())
    }

    /// The rates as report lines, with how large the device memory's
    /// buffers were.
    pub(super) fn lines(&self) -> Vec<String> {
        let rows: Vec<Vec<String>> = self
            .rates
            .iter()
            .map(|(kind, rate)| vec![kind.name().to_string(), rate.cell(|r| si(*r, kind.unit()))])
            .collect();
        let mut lines = table(&rows);
        lines.push(format!(
            "(device memory over buffers of {} MB)",
            self.device_bytes >> 20
        ));
        lines
    }
}

/// The multiply-add benchmark of `ty` (`f32` or `f16`): eight chains of
/// vec4 multiply-adds an invocation, every lane of every chain from a seed
/// of its own (the invocation's and its place), so no two of the 32
/// recurrences are the same.
fn fma_source(ty: &str) -> String {
    let enable = if ty == "f16" { "enable f16;\n" } else { "" };
    let mut init = String::new();
    let mut step = String::new();
    let mut sum = String::from("a0");
    for k in 0..8 {
        let seed = |lane: u32| format!("s + {ty}({:.3})", f64::from(4 * k + lane) * 0.003);
        init += &format!(
            "    var a{k} = vec4<{ty}>({}, {}, {}, {});\n",
            seed(0),
            seed(1),
            seed(2),
            seed(3)
        );
        // a·a + c: no closed form a compiler could fold the rounds into
        // (a·m + c has one, through m^n); with c = 0.2 it settles at
        // 0.276, far from f16's subnormals.
        step += &format!("        a{k} = fma(a{k}, a{k}, c);\n");
        if k > 0 {
            sum += &format!(" + a{k}");
        }
    }
    format!(
        "{enable}@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let s = {ty}(f32(g.x % 7u) * 0.01);
    // From the buffer (zeros), so the compiler knows nothing of it.
    let c = vec4<{ty}>({ty}(0.2 + out[2]));
{init}    for (var k = 0u; k < {ROUNDS}u; k++) {{
{step}    }}
    let t = {sum};
    out[3u + (g.x & 7u)] = f32(t.x + t.y + t.z + t.w);
}}
"
    )
}

/// The packed dot benchmark: eight chains of `dot4I8Packed` (or the same
/// by hand) an invocation, each from its own seed.
fn dot_source(by_hand: bool) -> String {
    let mut init = String::new();
    let mut step = String::new();
    let mut sum = String::from("a0");
    for k in 0..8 {
        init += &format!("    var a{k} = i32(g.x) + {k};\n");
        step += &format!("        a{k} += dot4(bitcast<u32>(a{k}), q);\n");
        if k > 0 {
            sum += &format!(" + a{k}");
        }
    }
    let dot = if by_hand {
        "fn bytes(a: u32) -> vec4<i32> {
    let s = bitcast<i32>(a);
    return vec4<i32>((s << 24u) >> 24u, (s << 16u) >> 24u, (s << 8u) >> 24u, s >> 24u);
}
fn dot4(a: u32, b: u32) -> i32 { return dot(bytes(a), bytes(b)); }"
    } else {
        "fn dot4(a: u32, b: u32) -> i32 { return dot4I8Packed(a, b); }"
    };
    format!(
        "{dot}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let q = 0x05fb7f81u ^ g.x;
{init}    for (var k = 0u; k < {ROUNDS}u; k++) {{
{step}    }}
    out[3u + (g.x & 7u)] = f32({sum});
}}
"
    )
}

/// The workgroup memory benchmark: 16 KB of vec4s, every one written
/// before the barrier (so wgpu need not zero them: it is built without),
/// then read round by round.
fn workgroup_source() -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> t: array<vec4<f32>, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32) {{
    for (var k = i; k < 1024u; k += 256u) {{ t[k] = vec4<f32>(f32(k)); }}
    workgroupBarrier();
    var acc = vec4<f32>(0.0);
    var j = i;
    for (var k = 0u; k < {ROUNDS}u; k++) {{
        acc += t[j];
        j = (j + 256u + (k & 1u)) & 1023u;
    }}
    out[3u + (i & 7u)] = acc.x + acc.y + acc.z + acc.w;
}}
"
    )
}

/// Device memory groups: each invocation takes every `stride`th vec4.
const DEVICE_GROUPS: u32 = 16384;

/// The device memory benchmarks over `n` vec4s: read, write or copy.
fn device_source(kind: Kind, n: u64) -> String {
    let stride = DEVICE_GROUPS * 256;
    let each = format!("for (var k = g.x; k < {n}u; k += {stride}u)");
    let (bindings, body) = match kind {
        Kind::DeviceRead => (
            "@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;",
            format!(
                "var acc = vec4<f32>(0.0);
    {each} {{ acc += src[k]; }}
    out[g.x & 7u] = acc.x + acc.y + acc.z + acc.w;"
            ),
        ),
        Kind::DeviceWrite => (
            "@group(0) @binding(0) var<storage, read_write> dst: array<vec4<f32>>;",
            format!("{each} {{ dst[k] = vec4<f32>(f32(k)); }}"),
        ),
        _ => (
            "@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;",
            format!("{each} {{ dst[k] = src[k]; }}"),
        ),
    };
    format!(
        "{bindings}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    {body}
}}
"
    )
}

/// The device memory buffers' size: 256 MB where a binding takes it.
fn device_bytes(gpu: &Gpu) -> u64 {
    (256 << 20).min(gpu.largest_binding()) / 16 * 16
}

/// Every rate on `gpu`.
pub(super) fn probe(gpu: &Gpu) -> Rates {
    let invocations = f64::from(GROUPS * 256);
    let rounds = f64::from(ROUNDS);
    let mut rates = Vec::new();
    let mut measure = |kind: Kind, rate: Result<f64, Miss>| rates.push((kind, Found::of(rate)));

    // Without wgpu's zeroing of workgroup memory: only the workgroup read
    // has any (16 KB, written before it is read), and its zeroing is
    // [`zeroing`]'s to time.
    let arithmetic = |source: String, work: f64| -> Result<f64, Miss> {
        let out = gpu.zeros(64)?;
        let kernel = gpu.kernel_zeroing(&source, "main", false)?;
        Ok(gpu.rate(&kernel, &[Bound::All(&out)], [GROUPS, 1, 1], work, TARGET)?)
    };
    // Eight chains of vec4 fmas: 64 flops a round; eight dots: 32 MACs.
    measure(
        Kind::F32Fma,
        arithmetic(fma_source("f32"), invocations * rounds * 64.0),
    );
    measure(
        Kind::F16Fma,
        needs(gpu, wgpu::Features::SHADER_F16, "SHADER_F16")
            .and_then(|_| arithmetic(fma_source("f16"), invocations * rounds * 64.0)),
    );
    measure(
        Kind::Dot4I8,
        arithmetic(dot_source(false), invocations * rounds * 32.0),
    );
    measure(
        Kind::Dot4ByHand,
        arithmetic(dot_source(true), invocations * rounds * 32.0),
    );
    measure(
        Kind::WorkgroupRead,
        arithmetic(workgroup_source(), invocations * rounds * 16.0),
    );

    let bytes = device_bytes(gpu);
    let n = bytes / 16;
    let device = |kind: Kind| -> Result<f64, Miss> {
        let kernel = gpu.kernel(&device_source(kind, n), "main")?;
        let a = gpu.zeros(bytes)?;
        let groups = [DEVICE_GROUPS, 1, 1];
        Ok(match kind {
            Kind::DeviceRead => {
                let out = gpu.zeros(64)?;
                gpu.rate(
                    &kernel,
                    &[Bound::All(&a), Bound::All(&out)],
                    groups,
                    bytes as f64,
                    TARGET,
                )?
            }
            Kind::DeviceWrite => {
                gpu.rate(&kernel, &[Bound::All(&a)], groups, bytes as f64, TARGET)?
            }
            _ => {
                let b = gpu.zeros(bytes)?;
                gpu.rate(
                    &kernel,
                    &[Bound::All(&a), Bound::All(&b)],
                    groups,
                    2.0 * bytes as f64,
                    TARGET,
                )?
            }
        })
    };
    for kind in [Kind::DeviceRead, Kind::DeviceWrite, Kind::DeviceCopy] {
        measure(kind, device(kind));
    }
    Rates {
        rates,
        device_bytes: bytes,
    }
}

/// What wgpu's zeroing of workgroup memory costs: a dispatch of
/// [`ZERO_GROUPS`] workgroups of 256, each with a 16 KB `var<workgroup>`
/// array it barely uses, its wall time per dispatch with the zeroing
/// (wgpu's default) and without.
#[derive(Clone, Debug)]
pub(super) struct Zeroing {
    zeroed: f64,
    unzeroed: f64,
}

/// The zeroing benchmark's workgroups a dispatch, and dispatches a run.
const ZERO_GROUPS: u32 = 1024;
const ZERO_DISPATCHES: u32 = 64;

/// The zeroing benchmark: 4096 f32s (16 KB) of workgroup memory, as
/// Gemma's `staged`; each invocation writes one and reads another.
const ZEROING: &str = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> staged: array<f32, 4096>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) w: vec3<u32>) {
    staged[i * 16u] = f32(w.x + i);
    workgroupBarrier();
    let v = staged[((i + 1u) % 256u) * 16u];
    if (v == -1.0) { out[0] = v; }
}
";

/// The zeroing's cost on `gpu`: seconds a dispatch with and without.
pub(super) fn zeroing(gpu: &Gpu) -> Result<Zeroing, Miss> {
    let out = gpu.zeros(64)?;
    let per_dispatch = |zeroed: bool| -> Result<f64, String> {
        let kernel = gpu.kernel_zeroing(ZEROING, "main", zeroed)?;
        let groups = [ZERO_GROUPS, 1, 1];
        gpu.run(&kernel, &[Bound::All(&out)], groups, 1)?;
        // The least of three runs: the one least disturbed.
        let mut least = f64::INFINITY;
        for _ in 0..3 {
            let took = gpu.run(&kernel, &[Bound::All(&out)], groups, ZERO_DISPATCHES)?;
            least = least.min(took.as_secs_f64() / f64::from(ZERO_DISPATCHES));
        }
        Ok(least)
    };
    Ok(Zeroing {
        zeroed: per_dispatch(true)?,
        unzeroed: per_dispatch(false)?,
    })
}

/// The zeroing's cost as report lines.
pub(super) fn zeroing_lines(z: &Zeroing) -> Vec<String> {
    let us = |s: f64| format!("{:.1} µs", s * 1e6);
    vec![
        format!(
            "a dispatch of {ZERO_GROUPS} workgroups with 16 KB of var<workgroup>: {} zeroed (wgpu's default), {} not",
            us(z.zeroed),
            us(z.unzeroed)
        ),
        format!(
            "the zeroing costs {} a dispatch ({:.1} ns a workgroup)",
            us(z.zeroed - z.unzeroed),
            (z.zeroed - z.unzeroed) * 1e9 / f64::from(ZERO_GROUPS)
        ),
    ]
}
