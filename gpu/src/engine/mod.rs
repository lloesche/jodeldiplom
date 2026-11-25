//! What every network here does the same way on its device.
//!
//! - A program ([`Program`], `program.rs`): one shader module of kernels on
//!   one bind group layout (the buffers read at bindings 0 and 1, the one
//!   written at 2, the dispatch's parameters at 3, any more from 4 on, as
//!   its [`Kernels`] say), storage buffers checked against the device's
//!   limits, and dispatches recorded: their parameters packed into one
//!   uniform buffer, a bind group for each set of buffers they bind
//!   ([`Program::record`]). The runs go in short submissions
//!   ([`crate::pace::dispatch`]).
//! - A network planned once per input size and kept on a device of its own
//!   (the separator, the beat network, the speech model) is a [`Model`]
//!   (`model.rs`): its plans recorded when first run, each run one plan and
//!   one read back.

mod model;
mod program;

use std::hash::Hash;

pub(crate) use model::Model;
pub(crate) use program::{Program, Recorded};

/// Bytes between the dispatches' parameters (the uniform offset
/// alignment every device offers).
pub(crate) const PARAM_STRIDE: u64 = 256;

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

/// The rotary table: per position, per pair of a head of 32, its cosine
/// and sine (in single precision, as the CPU's tables).
pub(crate) fn rotary_table(positions: usize) -> Vec<f32> {
    let mut table = Vec::with_capacity(positions * 32);
    for p in 0..positions {
        for i in 0..16 {
            let freq = 1.0f32 / 10_000f32.powf((2 * i) as f32 / 32.0);
            let angle = p as f32 * freq;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    table
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
/// model and the recogniser: their prelude (`bindings.wgsl`), the shared
/// helpers (`common.wgsl`), the separator's kernels (`kernels.wgsl`), then
/// `more`.
pub(crate) fn kernels_source(more: &[&str]) -> String {
    let mut parts = vec![
        include_str!("../bindings.wgsl"),
        include_str!("../common.wgsl"),
        include_str!("../kernels.wgsl"),
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
        for more in [
            &[][..],
            &[include_str!("../net.wgsl")][..],
            &[include_str!("../speech.wgsl")][..],
        ] {
            super::check_source(
                &super::kernels_source(more),
                naga::valid::Capabilities::empty(),
                &["matmul", "rmsnorm", "attention", "geglu"],
            );
        }
    }
}
