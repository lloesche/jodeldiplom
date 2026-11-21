//! The stem separator of [`jodel_rhythm::stems`] on the GPU, through
//! `wgpu` (Vulkan, DirectX 12 or Metal); the beat network of
//! [`jodel_rhythm::net`] ([`GpuNetwork`], `net.rs`); and the speech model of
//! [`jodel_lyrics::speech`] ([`GpuSpeech`], `speech.rs`).
//!
//! [`GpuSeparator`] runs the same network as the built-in CPU separator,
//! on the same weights, and gives the same masks to within rounding. A
//! chunk runs entirely on the GPU: its STFT goes up, the band split, the
//! 16 transformers and the mask layers run as one list of dispatches of
//! six kernels (`kernels.wgsl`: a tiled matrix multiply, RMSNorm, the
//! rotary positions, attention with an online softmax, the gated GELU and
//! a transpose), and the masks come back.
//!
//! ```no_run
//! use jodel_gpu::GpuSeparator;
//! use jodel_rhythm::stems;
//!
//! if let Some(gpu) = GpuSeparator::shared() {
//!     let (left, right) = (vec![0.0; 44_100], vec![0.0; 44_100]);
//!     let parts = stems::parts(gpu, &left, &right).unwrap();
//! }
//! ```

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Mutex, OnceLock};

mod asr;
mod net;
pub mod pace;
mod speech;
pub use asr::{GpuAsr, MAX_NEW, MAX_SECONDS};
pub use net::GpuNetwork;
pub use speech::GpuSpeech;

use jodel_rhythm::stems::{self, Separator, BAND_BINS, DIM, STEMS};

/// The most frames a chunk may have (6 s at 44.1 kHz, hop 512).
pub const MAX_FRAMES: usize = stems::CHUNK / 512 + 1;
/// Values per STFT frame: bins × channels × (real, imaginary).
const FRAME: usize = 1025 * 4;
const REGISTERS: usize = 4;
const HIDDEN: usize = 768;
const DEPTH: usize = 8;
/// Bytes between the dispatches' parameters (the uniform offset
/// alignment every device offers).
const PARAM_STRIDE: u64 = 256;
/// Bytes of one dispatch's parameters.
const PARAM_SIZE: u64 = 48;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Norm,
    Rope,
    Attention,
    Geglu,
    Transpose,
}

impl Kernel {
    const ALL: [Kernel; 6] = [
        Kernel::Matmul,
        Kernel::Norm,
        Kernel::Rope,
        Kernel::Attention,
        Kernel::Geglu,
        Kernel::Transpose,
    ];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Norm => "rmsnorm",
            Kernel::Rope => "rope",
            Kernel::Attention => "attention",
            Kernel::Geglu => "geglu",
            Kernel::Transpose => "transpose",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Weights,
    Table,
    Stft,
    Band,
    Grid,
    ByBand,
    Normed,
    Queries,
    KeysValues,
    Attended,
    Gate,
    Up,
    Final,
    Masks,
}

/// One weight layer's place in the weights buffer.
#[derive(Clone, Copy, Debug, Default)]
struct Layer {
    weight: u32,
    bias: u32,
}

/// One transformer layer's weights.
#[derive(Clone, Copy, Debug, Default)]
struct BlockAt {
    attention_norm: u32,
    q: u32,
    k: u32,
    v: u32,
    o: u32,
    mlp_norm: u32,
    gate: u32,
    up: u32,
    down: u32,
}

/// Where everything is in the weights buffer (in f32s).
struct Offsets {
    split_norm: Vec<u32>,
    split: Vec<Layer>,
    blocks: Vec<[BlockAt; 2]>,
    registers: u32,
    final_norm: u32,
    masks: Vec<Vec<Layer>>,
}

/// The weights as one buffer's contents, the layers transposed to
/// [input][output].
fn pack_weights() -> Result<(Vec<f32>, Offsets), String> {
    let mut w = stems::weights();
    let mut data: Vec<f32> = Vec::with_capacity(18_000_000);
    let mut put = |values: Vec<f32>| -> u32 {
        let at = data.len() as u32;
        data.extend(values);
        at
    };
    let linear = |w: &mut jodel_rhythm::net::weights::Weights,
                  put: &mut dyn FnMut(Vec<f32>) -> u32,
                  name: &str,
                  inputs: usize,
                  outputs: usize,
                  bias: bool|
     -> Result<Layer, String> {
        let raw = w.take(&format!("{name}.weight"), &[outputs, inputs])?;
        let mut t = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                t[i * outputs + o] = raw[o * inputs + i];
            }
        }
        let weight = put(t);
        let bias = if bias {
            put(w.take(&format!("{name}.bias"), &[outputs])?)
        } else {
            0
        };
        Ok(Layer { weight, bias })
    };
    let mut split_norm = Vec::new();
    let mut split = Vec::new();
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        let name = format!("model.band_split.to_features.{b}");
        split_norm.push(put(w.take(&format!("{name}.0.weight"), &[4 * bins])?));
        split.push(linear(
            &mut w,
            &mut put,
            &format!("{name}.1"),
            4 * bins,
            DIM,
            true,
        )?);
    }
    let mut blocks = Vec::new();
    for l in 0..DEPTH {
        let mut pair = [BlockAt::default(); 2];
        for (which, at) in pair.iter_mut().enumerate() {
            let name = format!("model.layers.{l}.{which}.layers.0");
            let kv = 128;
            *at = BlockAt {
                attention_norm: put(w.take(&format!("{name}.input_layernorm.weight"), &[DIM])?),
                q: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.q_proj"),
                    DIM,
                    DIM,
                    false,
                )?
                .weight,
                k: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.k_proj"),
                    DIM,
                    kv,
                    false,
                )?
                .weight,
                v: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.v_proj"),
                    DIM,
                    kv,
                    false,
                )?
                .weight,
                o: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.o_proj"),
                    DIM,
                    DIM,
                    false,
                )?
                .weight,
                mlp_norm: put(w.take(&format!("{name}.post_attention_layernorm.weight"), &[DIM])?),
                gate: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.gate_proj"),
                    DIM,
                    HIDDEN,
                    false,
                )?
                .weight,
                up: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.up_proj"),
                    DIM,
                    HIDDEN,
                    false,
                )?
                .weight,
                down: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.down_proj"),
                    HIDDEN,
                    DIM,
                    false,
                )?
                .weight,
            };
        }
        blocks.push(pair);
    }
    let registers = put(w.take("model.register_tokens", &[REGISTERS, REGISTERS, DIM])?);
    let final_norm = put(w.take("model.final_norm.weight", &[DIM])?);
    let mut masks = Vec::new();
    for s in 0..STEMS {
        let mut bands = Vec::new();
        for (b, &bins) in BAND_BINS.iter().enumerate() {
            let name = format!("mask_estimators.{s}.to_freq_mlps.{b}");
            bands.push(linear(&mut w, &mut put, &name, DIM, 4 * bins, true)?);
        }
        masks.push(bands);
    }
    if !w.left().is_empty() {
        return Err(format!("unused weights: {:?}", w.left()));
    }
    Ok((
        data,
        Offsets {
            split_norm,
            split,
            blocks,
            registers,
            final_norm,
            masks,
        },
    ))
}

/// The rotary table: per position, per pair of a head of 32, its cosine
/// and sine (in single precision, as the CPU table).
fn rotary_table(positions: usize) -> Vec<f32> {
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

/// One dispatch.
struct Op {
    kernel: Kernel,
    /// src, aux, dst.
    buffers: [Buf; 3],
    params: [u32; 12],
    groups: [u32; 3],
}

/// Groups for `count` invocations of 256, spread over x and y.
fn spread(count: usize) -> [u32; 3] {
    let groups = count.div_ceil(256).max(1);
    let x = groups.min(65_535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

/// The dispatches of a chunk of `frames` frames.
fn plan(o: &Offsets, frames: usize) -> Vec<Op> {
    let bands = BAND_BINS.len();
    let (rows, cols) = (frames + REGISTERS, bands + REGISTERS);
    let tokens = rows * cols;
    let d = DIM as u32;
    let mut ops = Vec::new();
    let matmul = |ops: &mut Vec<Op>,
                  a: Buf,
                  c: Buf,
                  m: usize,
                  n: usize,
                  k: usize,
                  a_off: usize,
                  lda: usize,
                  w: u32,
                  c_off: usize,
                  ldc: usize,
                  bias: Option<u32>,
                  add: bool| {
        let flags = u32::from(bias.is_some()) | (u32::from(add) << 1);
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [a, Buf::Weights, c],
            params: [
                m as u32,
                n as u32,
                k as u32,
                a_off as u32,
                lda as u32,
                w,
                c_off as u32,
                ldc as u32,
                bias.unwrap_or(0),
                flags,
                0,
                0,
            ],
            groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
        });
    };
    let norm = |ops: &mut Vec<Op>,
                x: Buf,
                y: Buf,
                rows: usize,
                dim: usize,
                x_off: usize,
                ldx: usize,
                w: u32,
                ldy: usize| {
        ops.push(Op {
            kernel: Kernel::Norm,
            buffers: [x, Buf::Weights, y],
            params: [
                rows as u32,
                dim as u32,
                x_off as u32,
                ldx as u32,
                w,
                0,
                ldy as u32,
                0,
                0,
                0,
                0,
                0,
            ],
            groups: spread(rows),
        });
    };
    let transpose = |ops: &mut Vec<Op>, from: Buf, to: Buf, rows: usize, cols: usize| {
        ops.push(Op {
            kernel: Kernel::Transpose,
            buffers: [from, Buf::Weights, to],
            params: [rows as u32, cols as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            groups: spread(rows * cols * DIM),
        });
    };
    let block = |ops: &mut Vec<Op>, x: Buf, at: &BlockAt, len: usize, turned: usize| {
        norm(
            ops,
            x,
            Buf::Normed,
            tokens,
            DIM,
            0,
            DIM,
            at.attention_norm,
            DIM,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Queries,
            tokens,
            DIM,
            DIM,
            0,
            DIM,
            at.q,
            0,
            DIM,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::KeysValues,
            tokens,
            128,
            DIM,
            0,
            DIM,
            at.k,
            0,
            DIM,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::KeysValues,
            tokens,
            128,
            DIM,
            0,
            DIM,
            at.v,
            128,
            DIM,
            None,
            false,
        );
        for (buf, heads) in [(Buf::Queries, 8u32), (Buf::KeysValues, 4)] {
            ops.push(Op {
                kernel: Kernel::Rope,
                buffers: [Buf::Table, Buf::Weights, buf],
                params: [
                    tokens as u32,
                    len as u32,
                    turned as u32,
                    heads,
                    d,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                groups: spread(tokens * heads as usize * 16),
            });
        }
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Queries, Buf::KeysValues, Buf::Attended],
            params: [len as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            groups: [len.div_ceil(64) as u32, 8, (tokens / len) as u32],
        });
        matmul(
            ops,
            Buf::Attended,
            x,
            tokens,
            DIM,
            DIM,
            0,
            DIM,
            at.o,
            0,
            DIM,
            None,
            true,
        );
        norm(ops, x, Buf::Normed, tokens, DIM, 0, DIM, at.mlp_norm, DIM);
        matmul(
            ops,
            Buf::Normed,
            Buf::Gate,
            tokens,
            HIDDEN,
            DIM,
            0,
            DIM,
            at.gate,
            0,
            HIDDEN,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Up,
            tokens,
            HIDDEN,
            DIM,
            0,
            DIM,
            at.up,
            0,
            HIDDEN,
            None,
            false,
        );
        ops.push(Op {
            kernel: Kernel::Geglu,
            buffers: [Buf::Up, Buf::Weights, Buf::Gate],
            params: [(tokens * HIDDEN) as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            groups: spread(tokens * HIDDEN),
        });
        matmul(
            ops,
            Buf::Gate,
            x,
            tokens,
            DIM,
            HIDDEN,
            0,
            HIDDEN,
            at.down,
            0,
            DIM,
            None,
            true,
        );
    };
    // The band split, into the grid (cleared, registers copied in before).
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        let width = 4 * bins;
        norm(
            &mut ops,
            Buf::Stft,
            Buf::Band,
            frames,
            width,
            4 * start,
            FRAME,
            o.split_norm[b],
            width,
        );
        matmul(
            &mut ops,
            Buf::Band,
            Buf::Grid,
            frames,
            DIM,
            width,
            0,
            width,
            o.split[b].weight,
            b * DIM,
            cols * DIM,
            Some(o.split[b].bias),
            false,
        );
        start += bins;
    }
    for pair in &o.blocks {
        transpose(&mut ops, Buf::Grid, Buf::ByBand, rows, cols);
        block(&mut ops, Buf::ByBand, &pair[0], rows, frames);
        transpose(&mut ops, Buf::ByBand, Buf::Grid, cols, rows);
        block(&mut ops, Buf::Grid, &pair[1], cols, bands);
    }
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        norm(
            &mut ops,
            Buf::Grid,
            Buf::Final,
            frames,
            DIM,
            b * DIM,
            cols * DIM,
            o.final_norm,
            DIM,
        );
        for (s, stem) in o.masks.iter().enumerate() {
            let c_off = s * frames * FRAME + 4 * start;
            matmul(
                &mut ops,
                Buf::Final,
                Buf::Masks,
                frames,
                4 * bins,
                DIM,
                0,
                DIM,
                stem[b].weight,
                c_off,
                FRAME,
                Some(stem[b].bias),
                false,
            );
        }
        start += bins;
    }
    ops
}

/// A recorded chunk size: its dispatches' bind groups and parameters.
struct Plan {
    ops: Vec<(Kernel, usize, u64, [u32; 3])>,
    /// The dispatches' parameters, kept while their bind groups use them.
    _params: wgpu::Buffer,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    buffers: HashMap<Buf, wgpu::Buffer>,
    staging: wgpu::Buffer,
    offsets: Offsets,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_index: HashMap<([Buf; 3], u64), usize>,
    plans: HashMap<usize, Plan>,
    pieces: pace::Pieces,
}

/// The separator network on the GPU.
pub struct GpuSeparator {
    name: String,
    inner: Mutex<Inner>,
}

/// `values` as little-endian bytes.
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The best adapter: a high-performance one preferred, a software
/// renderer refused.
fn adapter() -> Result<wgpu::Adapter, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("no GPU: {e}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("{} is a software renderer", info.name));
    }
    Ok(adapter)
}

/// The name of the GPU the separator would use, without setting it up;
/// `None` without a usable one.
pub fn gpu_name() -> Option<String> {
    std::panic::catch_unwind(|| adapter().ok().map(|a| a.get_info().name))
        .ok()
        .flatten()
}

/// What the GPU side finds on this machine, as lines for a diagnostic
/// report: every adapter `wgpu` sees on every backend with the limits the
/// separator needs, the one it would use (or why none), whether it sets
/// up (or the error), how long a 6 s chunk of silence takes on it, and
/// how that time splits ([`Profile`]: each kernel on its own, the matrix
/// multiplies by layer, the upload and the readback, whole chunks in a
/// row). Sets up a separator of its own (not [`GpuSeparator::shared`]'s).
pub fn report() -> Vec<String> {
    let mut lines = Vec::new();
    let mb = |bytes: u64| format!("{} MB", bytes / (1024 * 1024));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
    }));
    match adapters {
        Err(_) => lines.push("adapters: listing them panicked".into()),
        Ok(adapters) if adapters.is_empty() => {
            lines.push("adapters: none on any backend (Vulkan, Metal, DirectX 12, OpenGL)".into())
        }
        Ok(adapters) => {
            lines.push("adapters:".into());
            for a in adapters {
                let info = a.get_info();
                let limits = a.limits();
                lines.push(format!(
                    "  [{:?}] {} ({:?}), vendor 0x{:04x} device 0x{:04x}, driver {:?} {:?}",
                    info.backend,
                    info.name,
                    info.device_type,
                    info.vendor,
                    info.device,
                    info.driver,
                    info.driver_info
                ));
                lines.push(format!(
                    "      max buffer {}, max storage binding {}, workgroup storage {} KB, {} invocations per workgroup",
                    mb(limits.max_buffer_size),
                    mb(limits.max_storage_buffer_binding_size),
                    limits.max_compute_workgroup_storage_size / 1024,
                    limits.max_compute_invocations_per_workgroup
                ));
            }
        }
    }
    match std::panic::catch_unwind(adapter) {
        Ok(Ok(a)) => {
            let info = a.get_info();
            lines.push(format!(
                "the separator would use: {} ({:?})",
                info.name, info.backend
            ));
        }
        Ok(Err(e)) => lines.push(format!("the separator would use: none ({e})")),
        Err(_) => lines.push("the separator would use: none (looking panicked)".into()),
    }
    let started = std::time::Instant::now();
    let separator = match std::panic::catch_unwind(GpuSeparator::new) {
        Ok(Ok(s)) => {
            lines.push(format!(
                "separator set up: yes, in {} ms",
                started.elapsed().as_millis()
            ));
            s
        }
        Ok(Err(e)) => {
            lines.push(format!("separator set up: failed: {e}"));
            lines.extend(net::report());
            return lines;
        }
        Err(_) => {
            lines.push("separator set up: panicked".into());
            lines.extend(net::report());
            return lines;
        }
    };
    let stft = vec![0.0f32; MAX_FRAMES * FRAME];
    let mut times = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        match separator.masks(MAX_FRAMES, &stft) {
            Ok(_) => times.push(started.elapsed().as_millis()),
            Err(e) => {
                lines.push(format!("a 6 s chunk: failed: {e}"));
                lines.extend(net::report());
                return lines;
            }
        }
    }
    lines.push(format!(
        "a 6 s chunk of silence: {} ms, then {} ms",
        times[0], times[1]
    ));
    match separator.profile(MAX_FRAMES, &stft) {
        Ok(profile) => lines.extend(profile.lines()),
        Err(e) => lines.push(format!("profile: failed: {e}")),
    }
    lines.extend(net::report());
    lines
}

/// Runs `work` inside error scopes for validation, out-of-memory and
/// internal errors, so a failure on the device is an `Err` (not the
/// default handler's panic).
fn guarded<T>(
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

impl GpuSeparator {
    /// The separator on the best GPU, set up on first use; `None` without
    /// a usable one (a software renderer does not count), or if setting it
    /// up failed in any way.
    pub fn shared() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<Option<GpuSeparator>> = OnceLock::new();
        GPU.get_or_init(|| std::panic::catch_unwind(GpuSeparator::new).ok()?.ok())
            .as_ref()
    }

    /// A separator of its own for background work (analyses, getting
    /// lyrics timed), so that it never waits for, nor holds up, the stems
    /// of the track playing ([`GpuSeparator::shared`]'s); that one where a
    /// second does not fit on the GPU.
    pub fn background() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<Option<GpuSeparator>> = OnceLock::new();
        GPU.get_or_init(|| {
            let gpu = std::panic::catch_unwind(GpuSeparator::new).ok()?.ok()?;
            gpu.inner.lock().ok()?.pieces = pace::Pieces::resting();
            Some(gpu)
        })
        .as_ref()
        .or_else(GpuSeparator::shared)
    }

    /// Sets up the separator on the best GPU.
    pub fn new() -> Result<GpuSeparator, String> {
        let adapter = adapter()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jodel-gpu"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", info.name))?;
        let probe = device.clone();
        let inner = guarded(&probe, || Inner::new(device, queue, &limits))
            .map_err(|e| format!("{}: {e}", info.name))?;
        Ok(GpuSeparator {
            name: info.name,
            inner: Mutex::new(inner),
        })
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
    ) -> Result<Inner, String> {
        let (weights, offsets) = pack_weights()?;
        let tokens = (MAX_FRAMES + REGISTERS) * (BAND_BINS.len() + REGISTERS);
        let sizes = [
            (Buf::Weights, weights.len()),
            (Buf::Table, (MAX_FRAMES + REGISTERS) * 32),
            (Buf::Stft, MAX_FRAMES * FRAME),
            (Buf::Band, MAX_FRAMES * 4 * 129),
            (Buf::Grid, tokens * DIM),
            (Buf::ByBand, tokens * DIM),
            (Buf::Normed, tokens * DIM),
            (Buf::Queries, tokens * DIM),
            (Buf::KeysValues, tokens * DIM),
            (Buf::Attended, tokens * DIM),
            (Buf::Gate, tokens * HIDDEN),
            (Buf::Up, tokens * HIDDEN),
            (Buf::Final, MAX_FRAMES * DIM),
            (Buf::Masks, STEMS * MAX_FRAMES * FRAME),
        ];
        for &(buf, len) in &sizes {
            let size = (len * 4) as u64;
            if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
                return Err(format!(
                    "the GPU's buffers are too small for {buf:?} ({size} bytes)"
                ));
            }
        }
        let buffers: HashMap<Buf, wgpu::Buffer> = sizes
            .iter()
            .map(|&(buf, len)| {
                let b = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jodel-gpu"),
                    size: (len.max(1) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                (buf, b)
            })
            .collect();
        queue.write_buffer(&buffers[&Buf::Weights], 0, &bytes(&weights));
        queue.write_buffer(
            &buffers[&Buf::Table],
            0,
            &bytes(&rotary_table(MAX_FRAMES + REGISTERS)),
        );
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu masks"),
            size: (STEMS * MAX_FRAMES * FRAME * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodel-gpu"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(PARAM_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodel-gpu"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodel-gpu kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        let pipelines = Kernel::ALL
            .iter()
            .map(|&k| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(k.entry()),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(k.entry()),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (k, pipeline)
            })
            .collect();
        Ok(Inner {
            device,
            queue,
            pipelines,
            layout,
            buffers,
            staging,
            offsets,
            bind_groups: Vec::new(),
            bind_index: HashMap::new(),
            plans: HashMap::new(),
            pieces: pace::Pieces::default(),
        })
    }

    /// The bind group of `buffers` with the parameters `params`.
    fn bind_group(&mut self, buffers: [Buf; 3], params: &wgpu::Buffer, plan: u64) -> usize {
        if let Some(&i) = self.bind_index.get(&(buffers, plan)) {
            return i;
        }
        let entry = |binding: u32, buf: Buf| wgpu::BindGroupEntry {
            binding,
            resource: self.buffers[&buf].as_entire_binding(),
        };
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodel-gpu"),
            layout: &self.layout,
            entries: &[
                entry(0, buffers[0]),
                entry(1, buffers[1]),
                entry(2, buffers[2]),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: params,
                        offset: 0,
                        size: NonZeroU64::new(PARAM_SIZE),
                    }),
                },
            ],
        });
        self.bind_groups.push(group);
        self.bind_index
            .insert((buffers, plan), self.bind_groups.len() - 1);
        self.bind_groups.len() - 1
    }

    /// The plan for chunks of `frames` frames, recorded on first use.
    fn plan(&mut self, frames: usize) {
        if self.plans.contains_key(&frames) {
            return;
        }
        let ops = plan(&self.offsets, frames);
        let mut data = vec![0u8; ops.len() * PARAM_STRIDE as usize];
        for (i, op) in ops.iter().enumerate() {
            for (j, v) in op.params.iter().enumerate() {
                let at = i * PARAM_STRIDE as usize + 4 * j;
                data[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu params"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params, 0, &data);
        let recorded = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = self.bind_group(op.buffers, &params, frames as u64);
                (op.kernel, group, i as u64 * PARAM_STRIDE, op.groups)
            })
            .collect();
        self.plans.insert(
            frames,
            Plan {
                ops: recorded,
                _params: params,
            },
        );
    }

    fn masks(&mut self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        if frames == 0 || frames > MAX_FRAMES || stft.len() != frames * FRAME {
            return Err(format!("{frames} frames, {} values", stft.len()));
        }
        let device = self.device.clone();
        guarded(&device, || {
            self.plan(frames);
            self.run(frames, stft)
        })
    }

    /// One chunk through the recorded plan, in short submissions
    /// ([`pace::dispatch`]).
    fn run(&mut self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.queue
            .write_buffer(&self.buffers[&Buf::Stft], 0, &bytes(stft));
        let rows = frames + REGISTERS;
        let cols = BAND_BINS.len() + REGISTERS;
        let (buffers, offsets, staging) = (&self.buffers, &self.offsets, &self.staging);
        let size = (STEMS * frames * FRAME * 4) as u64;
        pace::dispatch(
            &self.device,
            &self.queue,
            "jodel-gpu",
            &mut self.pieces,
            &self.pipelines,
            &self.bind_groups,
            &self.plans[&frames].ops,
            |encoder| {
                let grid = &buffers[&Buf::Grid];
                encoder.clear_buffer(grid, 0, Some((rows * cols * DIM * 4) as u64));
                for i in 0..REGISTERS {
                    let from = (offsets.registers as usize + i * REGISTERS * DIM) * 4;
                    let to = ((frames + i) * cols + BAND_BINS.len()) * DIM * 4;
                    encoder.copy_buffer_to_buffer(
                        &buffers[&Buf::Weights],
                        from as u64,
                        grid,
                        to as u64,
                        (REGISTERS * DIM * 4) as u64,
                    );
                }
            },
            |encoder| {
                encoder.copy_buffer_to_buffer(&buffers[&Buf::Masks], 0, staging, 0, size);
            },
        )?;
        let slice = self.staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let out: Vec<f32> = slice
            .get_mapped_range()
            .map_err(|e| e.to_string())?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        self.staging.unmap();
        Ok(out)
    }
}

/// How long the parts of one chunk take on the GPU
/// ([`GpuSeparator::profile`]).
#[derive(Clone, Debug, Default)]
pub struct Profile {
    /// Frames in the chunk.
    pub frames: usize,
    /// Sending the chunk's STFT up (and waiting for it).
    pub upload: std::time::Duration,
    /// Copying the masks back and reading them.
    pub readback: std::time::Duration,
    /// Whole chunks one after the other, as the separator runs them.
    pub chunks: Vec<std::time::Duration>,
    /// Each kind of dispatch on its own: (name, dispatches, workgroups,
    /// time, the best of three).
    pub parts: Vec<(String, usize, u64, std::time::Duration)>,
}

impl Profile {
    /// The profile as lines for a report.
    pub fn lines(&self) -> Vec<String> {
        let ms = |d: std::time::Duration| format!("{:.1} ms", d.as_secs_f64() * 1e3);
        let mut lines = vec![
            format!(
                "profile of one chunk ({} frames); each part on its own, the best of three:",
                self.frames
            ),
            format!(
                "  upload {}, readback {}",
                ms(self.upload),
                ms(self.readback)
            ),
            format!(
                "  whole chunks in a row: {}",
                self.chunks
                    .iter()
                    .map(|&d| ms(d))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ];
        let width = self.parts.iter().map(|p| p.0.len()).max().unwrap_or(0);
        let mut kernels = std::time::Duration::ZERO;
        for (name, dispatches, groups, time) in &self.parts {
            if !name.starts_with("  ") {
                kernels += *time;
            }
            lines.push(format!(
                "  {name:width$}  {dispatches:5} dispatches {groups:9} workgroups  {}",
                ms(*time)
            ));
        }
        lines.push(format!("  the kernels' times added up: {}", ms(kernels)));
        lines
    }
}

impl GpuSeparator {
    /// Times the parts of a chunk of `frames` frames of `stft` (see
    /// [`Profile`]): for finding what is slow on a GPU.
    pub fn profile(&self, frames: usize, stft: &[f32]) -> Result<Profile, String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU separator failed before".to_owned())?;
        if frames == 0 || frames > MAX_FRAMES || stft.len() != frames * FRAME {
            return Err(format!("{frames} frames, {} values", stft.len()));
        }
        let device = inner.device.clone();
        guarded(&device, || inner.profile(frames, stft))
    }
}

/// What a matrix multiply of the plan computes, by its shape.
fn matmul_part(params: &[u32; 12]) -> &'static str {
    let (n, k) = (params[1] as usize, params[2] as usize);
    match (n, k) {
        (DIM, DIM) => "  matmul q, o (attention)",
        (128, DIM) => "  matmul k, v (attention)",
        (HIDDEN, DIM) => "  matmul gate, up (feed-forward)",
        (DIM, HIDDEN) => "  matmul down (feed-forward)",
        (DIM, _) => "  matmul band split",
        _ => "  matmul masks",
    }
}

impl Inner {
    /// Waits for everything sent to the GPU so far.
    fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// Runs the dispatches of the plan for `frames` that `keep` picks (by
    /// index) on their own, three times; the best time.
    fn time_dispatches(
        &mut self,
        frames: usize,
        keep: &dyn Fn(usize) -> bool,
    ) -> Result<std::time::Duration, String> {
        let mut best = std::time::Duration::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jodel-gpu profile"),
                });
            {
                let plan = &self.plans[&frames];
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("jodel-gpu profile"),
                    timestamp_writes: None,
                });
                for (i, &(kernel, group, offset, [x, y, z])) in plan.ops.iter().enumerate() {
                    if keep(i) {
                        pass.set_pipeline(&self.pipelines[&kernel]);
                        pass.set_bind_group(0, &self.bind_groups[group], &[offset as u32]);
                        pass.dispatch_workgroups(x, y, z);
                    }
                }
            }
            self.queue.submit([encoder.finish()]);
            self.wait()?;
            best = best.min(started.elapsed());
        }
        Ok(best)
    }

    fn profile(&mut self, frames: usize, stft: &[f32]) -> Result<Profile, String> {
        self.plan(frames);
        let mut profile = Profile {
            frames,
            ..Profile::default()
        };
        // Whole chunks first, on a GPU that has not run much yet.
        for _ in 0..5 {
            let started = std::time::Instant::now();
            self.run(frames, stft)?;
            profile.chunks.push(started.elapsed());
        }
        let started = std::time::Instant::now();
        self.queue
            .write_buffer(&self.buffers[&Buf::Stft], 0, &bytes(stft));
        self.queue.submit([]);
        self.wait()?;
        profile.upload = started.elapsed();
        let size = (STEMS * frames * FRAME * 4) as u64;
        let started = std::time::Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jodel-gpu profile"),
            });
        encoder.copy_buffer_to_buffer(&self.buffers[&Buf::Masks], 0, &self.staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait()?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let read: f64 = slice
            .get_mapped_range()
            .map_err(|e| e.to_string())?
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .sum();
        std::hint::black_box(read);
        self.staging.unmap();
        profile.readback = started.elapsed();
        // Each kernel on its own, the matrix multiplies also by layer.
        let ops = plan(&self.offsets, frames);
        for kernel in Kernel::ALL {
            let picked: Vec<usize> = (0..ops.len())
                .filter(|&i| ops[i].kernel == kernel)
                .collect();
            if picked.is_empty() {
                continue;
            }
            let groups: u64 = picked
                .iter()
                .map(|&i| ops[i].groups.iter().map(|&g| u64::from(g)).product::<u64>())
                .sum();
            let time = self.time_dispatches(frames, &|i| ops[i].kernel == kernel)?;
            profile
                .parts
                .push((kernel.entry().to_owned(), picked.len(), groups, time));
            if kernel == Kernel::Matmul {
                let mut names: Vec<&'static str> = picked
                    .iter()
                    .map(|&i| matmul_part(&ops[i].params))
                    .collect();
                names.sort_unstable();
                names.dedup();
                for name in names {
                    let mine: Vec<usize> = picked
                        .iter()
                        .copied()
                        .filter(|&i| matmul_part(&ops[i].params) == name)
                        .collect();
                    let groups: u64 = mine
                        .iter()
                        .map(|&i| ops[i].groups.iter().map(|&g| u64::from(g)).product::<u64>())
                        .sum();
                    let time = self.time_dispatches(frames, &|i| {
                        ops[i].kernel == Kernel::Matmul && matmul_part(&ops[i].params) == name
                    })?;
                    profile
                        .parts
                        .push((name.to_owned(), mine.len(), groups, time));
                }
            }
        }
        Ok(profile)
    }
}

impl Separator for GpuSeparator {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.inner
            .lock()
            .map_err(|_| "the GPU separator failed before".to_owned())?
            .masks(frames, stft)
    }
}
