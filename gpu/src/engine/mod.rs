//! What every network here does the same way on its device.
//!
//! - A program ([`Program`], `program.rs`): one shader module of kernels on
//!   one bind group layout (the buffers read at bindings 0 and 1, the one
//!   written at 2, the dispatch's parameters at 3, any more from 4 on, as
//!   its [`Kernels`] say), storage buffers checked against the device's
//!   limits, and dispatches recorded: their parameters packed into one
//!   uniform buffer, a bind group for each set of buffers they bind
//!   ([`Program::record`]). The runs go in short submissions
//!   ([`crate::pace::dispatch`]), waited for ([`Program::run`]) or left
//!   running while the processor does more ([`Program::run_left`]); a
//!   failure on the device is an `Err` ([`guarded`]).
//! - A network planned once per input size and kept on a device of its own
//!   (the separator, the beat network, the speech model) is a [`Model`]
//!   (`model.rs`): its plans recorded when first run, each run one plan and
//!   one read back; its dispatches are [`ModelOp`]s (three buffers and the
//!   [`PARAM_WORDS`] numbers of `bindings.wgsl`). With its weights packed
//!   from the CPU model it is a [`PackedNet`] (`network.rs`).
//! - Every matrix multiply is planned as a [`Mul`] (`mul.rs`): what it
//!   reads and writes, the weights, what becomes of the products.
//! - Every network is held as a [`Kept`] (`kept.rs`): set up when first
//!   needed, and again when its device is lost (each says so: [`OnDevice`]);
//!   a failed setup is tried again a while later. The process keeps the
//!   separators and the beat network itself ([`KeptNet`]); the player
//!   keeps the others. A device the parts of several models share (the
//!   cover's networks', the picture models') is a [`SharedDevice`]
//!   (`shared.rs`), set up and set up again the same way.

mod kept;
mod model;
mod mul;
mod network;
mod program;
mod retrying;
mod shared;

use std::hash::Hash;

pub use kept::{Kept, OnDevice, SetUp, UNKNOWN};
pub(crate) use model::{buffers_footprint, Model};
pub(crate) use mul::{After, Matrix, Mul, Tiled, TiledMul, ARGMAX, BIAS, MUL_NUMBERS, SWIGLU};
pub(crate) use network::{KeptNet, PackedNet};
pub(crate) use program::{using, InFlight, Input, Output, Program, Read, Recorded, Write};
pub(crate) use shared::SharedDevice;

/// Bytes between the dispatches' parameters (the uniform offset
/// alignment every device offers).
pub(crate) const PARAM_STRIDE: u64 = 256;

/// The numbers of a dispatch of the kernels on `bindings.wgsl` (the
/// separator's, the beat network's, the speech model's and the
/// recogniser's: its `Params`, `f0` to `f11`).
pub(crate) const PARAM_WORDS: usize = 12;

/// The bytes of those numbers ([`Kernels::PARAM_BYTES`] of those kernels).
pub(crate) const PARAM_WORDS_BYTES: u64 = (PARAM_WORDS * 4) as u64;

/// A dispatch of a [`Model`]: its kernel, its three buffers (`src`, `aux`,
/// `dst`), its [`PARAM_WORDS`] numbers and its workgroups.
pub(crate) type ModelOp<K, B> = Op<K, [B; 3], PARAM_WORDS>;

/// How a kernel binds a buffer after its parameters (binding 4 on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
}

/// A program's kernels and the interface they share: every kernel and its
/// entry point, the bytes of a dispatch's parameters, and the buffers
/// bound after them.
pub(crate) trait Kernels: Copy + Eq + Hash + 'static {
    const ALL: &'static [Self];
    const PARAM_BYTES: u64;
    const EXTRA: &'static [Access];
    fn entry(self) -> &'static str;

    /// Whether the kernel is in the program on a device with `features`
    /// (a program makes the pipelines of these only; every one by default).
    fn available(self, _features: wgpu::Features) -> bool {
        true
    }
}

/// One dispatch as planned: its kernel, the buffers it binds (`B`, a key
/// its network turns into buffers: [`Program::record`]), its `P` numbers
/// and its workgroups.
#[derive(Clone, Debug)]
pub(crate) struct Op<K, B, const P: usize> {
    pub(crate) kernel: K,
    pub(crate) buffers: B,
    pub(crate) params: [usize; P],
    pub(crate) groups: [u32; 3],
}

/// One dispatch as recorded: its kernel, its bind group (by index), its
/// parameters' offset in their buffer, its workgroups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dispatch<K> {
    pub(crate) kernel: K,
    pub(crate) group: usize,
    pub(crate) offset: u64,
    pub(crate) groups: [u32; 3],
}

/// A value the kernels read as 32 bits.
pub(crate) trait Word: Copy {
    fn le(self) -> [u8; 4];
}

impl Word for f32 {
    fn le(self) -> [u8; 4] {
        self.to_le_bytes()
    }
}

impl Word for u32 {
    fn le(self) -> [u8; 4] {
        self.to_le_bytes()
    }
}

/// `values` as little-endian bytes.
pub(crate) fn bytes<T: Word>(values: &[T]) -> Vec<u8> {
    values.iter().flat_map(|v| v.le()).collect()
}

/// Groups for `count` invocations of 256, spread over x and y.
pub(crate) fn spread(count: usize) -> [u32; 3] {
    let groups = count.div_ceil(256).max(1);
    let x = groups.min(65_535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

/// A dispatch's numbers: `values` first, then zeros (each must fit 32 bits
/// when recorded).
pub(crate) fn params<const P: usize>(values: &[usize]) -> [usize; P] {
    let mut p = [0; P];
    for (slot, &v) in p.iter_mut().zip(values) {
        *slot = v;
    }
    p
}

/// A head's rotary frequencies: its pairs (half its width `dim`), the
/// first `turned` of them turning at `1 / base^(2i / dim)`, the rest not.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Freqs {
    pub(crate) base: f64,
    pub(crate) dim: usize,
    pub(crate) turned: usize,
}

/// The separator's and the beat network's heads of 32
/// (every pair turned, base 10⁴).
pub(crate) const HEAD_32: Freqs = Freqs {
    base: 10_000.0,
    dim: 32,
    turned: 16,
};

/// How a rotary table is reckoned (each model's as its reference does).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Angles {
    /// The frequencies, angles, cosines and sines in f32 (the separator's,
    /// the beat network's, the depth network's and the text encoder's).
    F32,
    /// The frequencies and angles in f32, their cosines and sines in f64
    /// (the language model: its reference's f32 angles, the functions to
    /// the last bit).
    F32ThenF64,
    /// All in f64 (the picture transformer, as its pipeline; the
    /// recogniser's decoder).
    F64,
}

/// A rotary table: per position of `positions`, each pair's cosine and
/// sine (as f32s, side by side).
pub(crate) fn rotary(
    positions: impl IntoIterator<Item = usize>,
    freqs: Freqs,
    angles: Angles,
) -> Vec<f32> {
    // Each pair's frequency, in the precision `angles` takes it (an f32
    // one held exactly as an f64).
    let freq = |i: usize| -> f64 {
        if i >= freqs.turned {
            return 0.0;
        }
        let (base, dim) = (freqs.base, freqs.dim as f64);
        match angles {
            Angles::F64 => 1.0 / base.powf((2 * i) as f64 / dim),
            Angles::F32 | Angles::F32ThenF64 => {
                f64::from(1.0 / (base as f32).powf((2 * i) as f32 / dim as f32))
            }
        }
    };
    let freqs: Vec<f64> = (0..freqs.dim / 2).map(freq).collect();
    // A position's turn at a pair's frequency: its cosine and sine.
    let turn: fn(usize, f64) -> (f32, f32) = match angles {
        Angles::F32 => |p, f| {
            let a = p as f32 * f as f32;
            (a.cos(), a.sin())
        },
        Angles::F32ThenF64 => |p, f| {
            let a = f64::from(p as f32 * f as f32);
            (a.cos() as f32, a.sin() as f32)
        },
        Angles::F64 => |p, f| {
            let a = p as f64 * f;
            (a.cos() as f32, a.sin() as f32)
        },
    };
    let mut out = Vec::new();
    for p in positions {
        for &f in &freqs {
            let (c, s) = turn(p, f);
            out.extend([c, s]);
        }
    }
    out
}

/// WGSL's declarations of `values` as `u32` constants, so the kernels'
/// flags and places are written once, in Rust.
pub(crate) fn wgsl_consts(values: &[(&str, usize)]) -> String {
    values
        .iter()
        .map(|(name, v)| format!("const {name}: u32 = {v}u;\n"))
        .collect()
}

/// WGSL's readers of a dispatch's numbers by name: for each name and its
/// place, `fn name() -> u32 { return p.{field}{place}; }` (`field` the
/// prefix of the parameters' fields, `f` or `n`).
pub(crate) fn wgsl_numbers<S: AsRef<str>>(
    field: &str,
    names: impl IntoIterator<Item = (S, usize)>,
) -> String {
    names
        .into_iter()
        .map(|(name, i)| format!("fn {}() -> u32 {{ return p.{field}{i}; }}\n", name.as_ref()))
        .collect()
}

/// Runs `work` inside error scopes for validation, out-of-memory and
/// internal errors, so a failure on the device is an `Err` (not the
/// default handler's panic).
pub(crate) fn guarded<T>(
    device: &wgpu::Device,
    work: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let scopes = [
        wgpu::ErrorFilter::Internal,
        wgpu::ErrorFilter::OutOfMemory,
        wgpu::ErrorFilter::Validation,
    ]
    .map(|f| device.push_error_scope(f));
    let result = work();
    let mut error = None;
    // The innermost first.
    for scope in scopes.into_iter().rev() {
        if let Some(e) = pollster::block_on(scope.pop()) {
            error.get_or_insert(e.to_string());
        }
    }
    match (result, error) {
        (Ok(_), Some(e)) => Err(e),
        (result, _) => result,
    }
}

/// The kernels' source for the separator, the beat network, the speech
/// model and the recogniser: their prelude (`bindings.wgsl`), the helpers
/// every model shares (`common.wgsl`), the kernels and helpers these four
/// share (`shared.wgsl`), then `more` (each network's own).
pub(crate) fn shared_source(more: &[&str]) -> String {
    let mut parts = vec![
        include_str!("../bindings.wgsl"),
        include_str!("../common.wgsl"),
        include_str!("../shared.wgsl"),
    ];
    parts.extend(more);
    parts.join("\n")
}

/// Every kernel of `source` parses and validates, and Metal's and DirectX's
/// writers take it (as wgpu would on a Mac and through DirectX 12), on any
/// machine; each of `entries` is in both outputs.
#[cfg(test)]
pub(crate) fn check_source(
    source: &str,
    capabilities: naga::valid::Capabilities,
    entries: &[&str],
) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{e}"));
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    let (msl, _) = naga::back::msl::write_string(
        &module,
        &info,
        &naga::back::msl::Options::default(),
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut hlsl = String::new();
    naga::back::hlsl::Writer::new(
        &mut hlsl,
        &naga::back::hlsl::Options::default(),
        &naga::back::hlsl::PipelineOptions::default(),
    )
    .write(&module, &info, None)
    .unwrap_or_else(|e| panic!("{e}"));
    for e in entries {
        assert!(msl.contains(e), "{e} in Metal");
        assert!(hlsl.contains(e), "{e} in HLSL");
    }
}

#[cfg(test)]
mod tests {
    /// The separator's, the beat network's and the speech model's kernels
    /// (with the shared prelude and helpers) validate and write for Metal
    /// and DirectX.
    #[test]
    fn the_kernels_validate_and_write_for_metal_and_directx() {
        for (more, entry) in [
            (include_str!("../separator.wgsl"), "attention"),
            (include_str!("../net.wgsl"), "net_attention"),
            (include_str!("../speech.wgsl"), "sp_attention"),
        ] {
            super::check_source(
                &super::shared_source(&[more]),
                naga::valid::Capabilities::empty(),
                &[
                    "matmul",
                    "rmsnorm",
                    "rope",
                    "transpose",
                    "gelu_in_place",
                    entry,
                ],
            );
        }
    }
}
