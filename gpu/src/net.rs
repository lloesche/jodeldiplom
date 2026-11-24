//! The beat network of [`jodel_rhythm::net`] (Beat This!, small) on the
//! GPU, with the same built-in weights: [`GpuNetwork`] is a
//! [`NetRunner`], so the analysis runs a piece's layers here and only the
//! two logits per frame come back.
//!
//! A piece is one list of dispatches: the spectrogram in, the stem's
//! convolution (as patches through the tiled matrix multiply), the three
//! blocks (attention and feed-forward across the bands of each frame, then
//! across the frames of each band, a convolution halving the bands), the
//! projection, six transformer layers, the final RMSNorm and the head.
//! The matrix multiply, RMSNorm and rotary kernels are the separator's
//! (`kernels.wgsl`, in the beat network's modes); the rest are in
//! `net.wgsl`.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Mutex, OnceLock};

use jodel_rhythm::net::mel::BANDS;
use jodel_rhythm::net::weights::Weights;
use jodel_rhythm::net::{built_in_weights, NetRunner, PIECE_FRAMES};

use super::{bytes, guarded, rotary_table, spread, Turned, PARAM_SIZE, PARAM_STRIDE};

/// The front end's channels after the stem, and the transformer's width.
const STEM: usize = 32;
const DIM: usize = 128;
const LAYERS: usize = 6;
/// A head's width.
const HEAD: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Norm,
    Rope,
    In,
    Patches,
    AffineGelu,
    Gelu,
    Transpose,
    Flatten,
    Attention,
}

impl Kernel {
    const ALL: [Kernel; 10] = [
        Kernel::Matmul,
        Kernel::Norm,
        Kernel::Rope,
        Kernel::In,
        Kernel::Patches,
        Kernel::AffineGelu,
        Kernel::Gelu,
        Kernel::Transpose,
        Kernel::Flatten,
        Kernel::Attention,
    ];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Norm => "rmsnorm",
            Kernel::Rope => "rope",
            Kernel::In => "net_in",
            Kernel::Patches => "net_patches",
            Kernel::AffineGelu => "net_affine_gelu",
            Kernel::Gelu => "net_gelu",
            Kernel::Transpose => "net_transpose",
            Kernel::Flatten => "net_flatten",
            Kernel::Attention => "net_attention",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Weights,
    Table,
    Spect,
    X,
    Y,
    Normed,
    Qkv,
    Gates,
    Attended,
    Hidden,
    Patches,
    Logits,
}

/// An attention's weights (offsets in f32s).
#[derive(Clone, Copy, Debug, Default)]
struct AttnAt {
    norm: u32,
    qkv: u32,
    gates: u32,
    gates_bias: u32,
    out: u32,
}

/// A feed-forward step's weights.
#[derive(Clone, Copy, Debug, Default)]
struct FfAt {
    norm: u32,
    up: u32,
    up_bias: u32,
    down: u32,
    down_bias: u32,
}

/// A front-end block's weights.
#[derive(Clone, Copy, Debug, Default)]
struct BlockAt {
    attn_f: AttnAt,
    ff_f: FfAt,
    attn_t: AttnAt,
    ff_t: FfAt,
    conv: u32,
    scale: u32,
    shift: u32,
}

/// Where everything is in the weights buffer.
#[derive(Clone, Debug, Default)]
struct Offsets {
    in_scale: u32,
    in_shift: u32,
    stem: u32,
    stem_scale: u32,
    stem_shift: u32,
    blocks: Vec<BlockAt>,
    project: u32,
    project_bias: u32,
    layers: Vec<(AttnAt, FfAt)>,
    norm: u32,
    head: u32,
    head_bias: u32,
}

/// Takes the weights out of the file into one buffer's contents.
struct Packer {
    w: Weights,
    data: Vec<f32>,
}

impl Packer {
    fn put(&mut self, values: Vec<f32>) -> u32 {
        let at = self.data.len() as u32;
        self.data.extend(values);
        at
    }

    fn vector(&mut self, name: &str, dim: usize) -> Result<u32, String> {
        let v = self.w.take(name, &[dim])?;
        Ok(self.put(v))
    }

    /// A layer, transposed to [input][output]; and its bias (0 without).
    fn linear(
        &mut self,
        name: &str,
        inputs: usize,
        outputs: usize,
        bias: bool,
    ) -> Result<(u32, u32), String> {
        let raw = self.w.take(&format!("{name}.weight"), &[outputs, inputs])?;
        let mut t = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                t[i * outputs + o] = raw[o * inputs + i];
            }
        }
        let weight = self.put(t);
        let bias = if bias {
            self.vector(&format!("{name}.bias"), outputs)?
        } else {
            0
        };
        Ok((weight, bias))
    }

    /// A convolution of `kf` bands by 3 frames as a layer over its patches:
    /// [out][in][kf][kt] to [(kf, kt, in)][out].
    fn conv(
        &mut self,
        name: &str,
        inputs: usize,
        outputs: usize,
        kf: usize,
    ) -> Result<u32, String> {
        let raw = self
            .w
            .take(&format!("{name}.weight"), &[outputs, inputs, kf, 3])?;
        let mut t = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                for f in 0..kf {
                    for k in 0..3 {
                        t[((f * 3 + k) * inputs + i) * outputs + o] =
                            raw[((o * inputs + i) * kf + f) * 3 + k];
                    }
                }
            }
        }
        Ok(self.put(t))
    }

    /// A per-channel scale and shift.
    fn affine(&mut self, name: &str, dim: usize) -> Result<(u32, u32), String> {
        let scale = self.vector(&format!("{name}.scale"), dim)?;
        let shift = self.vector(&format!("{name}.shift"), dim)?;
        Ok((scale, shift))
    }

    fn attention(&mut self, name: &str, dim: usize) -> Result<AttnAt, String> {
        let heads = dim / HEAD;
        let norm = self.vector(&format!("{name}.norm.gamma"), dim)?;
        let (qkv, _) = self.linear(&format!("{name}.to_qkv"), dim, 3 * dim, false)?;
        let (gates, gates_bias) = self.linear(&format!("{name}.to_gates"), dim, heads, true)?;
        let (out, _) = self.linear(&format!("{name}.to_out.0"), dim, dim, false)?;
        Ok(AttnAt {
            norm,
            qkv,
            gates,
            gates_bias,
            out,
        })
    }

    fn feed_forward(&mut self, name: &str, dim: usize) -> Result<FfAt, String> {
        let norm = self.vector(&format!("{name}.net.0.gamma"), dim)?;
        let (up, up_bias) = self.linear(&format!("{name}.net.1"), dim, 4 * dim, true)?;
        let (down, down_bias) = self.linear(&format!("{name}.net.4"), 4 * dim, dim, true)?;
        Ok(FfAt {
            norm,
            up,
            up_bias,
            down,
            down_bias,
        })
    }
}

/// The weights as one buffer's contents and where each tensor is; every
/// tensor of the file is used.
fn pack_weights() -> Result<(Vec<f32>, Offsets), String> {
    let mut p = Packer {
        w: built_in_weights()?,
        data: Vec::with_capacity(2_000_000),
    };
    let mut o = Offsets::default();
    (o.in_scale, o.in_shift) = p.affine("frontend.stem.bn1d", BANDS)?;
    o.stem = p.conv("frontend.stem.conv2d", 1, STEM, 4)?;
    (o.stem_scale, o.stem_shift) = p.affine("frontend.stem.bn2d", STEM)?;
    let mut c = STEM;
    for b in 0..3 {
        let n = format!("frontend.blocks.{b}");
        let attn_f = p.attention(&format!("{n}.partial.attnF"), c)?;
        let ff_f = p.feed_forward(&format!("{n}.partial.ffF"), c)?;
        let attn_t = p.attention(&format!("{n}.partial.attnT"), c)?;
        let ff_t = p.feed_forward(&format!("{n}.partial.ffT"), c)?;
        let conv = p.conv(&format!("{n}.conv2d"), c, 2 * c, 2)?;
        let (scale, shift) = p.affine(&format!("{n}.norm"), 2 * c)?;
        o.blocks.push(BlockAt {
            attn_f,
            ff_f,
            attn_t,
            ff_t,
            conv,
            scale,
            shift,
        });
        c *= 2;
    }
    let bands_out = BANDS / 4 / 8;
    (o.project, o.project_bias) = p.linear("frontend.linear", c * bands_out, DIM, true)?;
    for l in 0..LAYERS {
        let n = format!("transformer_blocks.layers.{l}");
        let attn = p.attention(&format!("{n}.0"), DIM)?;
        let ff = p.feed_forward(&format!("{n}.1"), DIM)?;
        o.layers.push((attn, ff));
    }
    o.norm = p.vector("transformer_blocks.norm.gamma", DIM)?;
    (o.head, o.head_bias) = p.linear("task_heads.beat_downbeat_lin", DIM, 2, true)?;
    if !p.w.left().is_empty() {
        return Err(format!("unused weights: {:?}", p.w.left()));
    }
    Ok((p.data, o))
}

/// One dispatch.
struct Op {
    kernel: Kernel,
    /// src, aux, dst.
    buffers: [Buf; 3],
    params: [u32; 12],
    groups: [u32; 3],
}

fn params(values: &[usize]) -> [u32; 12] {
    let mut p = [0u32; 12];
    for (slot, &v) in p.iter_mut().zip(values) {
        *slot = v as u32;
    }
    p
}

/// The dispatches of a piece of `frames` frames.
fn plan(o: &Offsets, frames: usize) -> Vec<Op> {
    let f = frames;
    let mut ops: Vec<Op> = Vec::new();
    let elementwise = |ops: &mut Vec<Op>, kernel, buffers, p: &[usize], count: usize| {
        ops.push(Op {
            kernel,
            buffers,
            params: params(p),
            groups: spread(count),
        });
    };
    // dst (+)= a · weights, see the kernel.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        ops: &mut Vec<Op>,
        a: Buf,
        c: Buf,
        m: usize,
        n: usize,
        k: usize,
        w: u32,
        bias: Option<u32>,
        add: bool,
    ) {
        let flags = usize::from(bias.is_some()) | (usize::from(add) << 1);
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [a, Buf::Weights, c],
            params: params(&[
                m,
                n,
                k,
                0,
                k,
                w as usize,
                0,
                n,
                bias.unwrap_or(0) as usize,
                flags,
            ]),
            groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
        });
    }
    fn norm(ops: &mut Vec<Op>, x: Buf, y: Buf, rows: usize, dim: usize, w: u32) {
        ops.push(Op {
            kernel: Kernel::Norm,
            buffers: [x, Buf::Weights, y],
            params: params(&[rows, dim, 0, dim, w as usize, 0, dim, 1]),
            groups: spread(rows),
        });
    }
    // Attention over sequences of `len` tokens of `dim`, added to `x`.
    fn attention(ops: &mut Vec<Op>, x: Buf, tokens: usize, len: usize, dim: usize, at: AttnAt) {
        let heads = dim / HEAD;
        norm(ops, x, Buf::Normed, tokens, dim, at.norm);
        matmul(
            ops,
            Buf::Normed,
            Buf::Qkv,
            tokens,
            3 * dim,
            dim,
            at.qkv,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Gates,
            tokens,
            heads,
            dim,
            at.gates,
            Some(at.gates_bias),
            false,
        );
        for col in [0, dim] {
            ops.push(Op {
                kernel: Kernel::Rope,
                buffers: [Buf::Table, Buf::Weights, Buf::Qkv],
                params: params(&[tokens, len, len, heads, 3 * dim, col, 1]),
                groups: spread(tokens * heads * 16),
            });
        }
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Qkv, Buf::Gates, Buf::Attended],
            params: params(&[len, dim, heads]),
            groups: [len.div_ceil(64) as u32, heads as u32, (tokens / len) as u32],
        });
        matmul(ops, Buf::Attended, x, tokens, dim, dim, at.out, None, true);
    }
    // The feed-forward step, added to `x`.
    fn feed_forward(ops: &mut Vec<Op>, x: Buf, tokens: usize, dim: usize, at: FfAt) {
        norm(ops, x, Buf::Normed, tokens, dim, at.norm);
        matmul(
            ops,
            Buf::Normed,
            Buf::Hidden,
            tokens,
            4 * dim,
            dim,
            at.up,
            Some(at.up_bias),
            false,
        );
        let count = tokens * 4 * dim;
        ops.push(Op {
            kernel: Kernel::Gelu,
            buffers: [Buf::Table, Buf::Weights, Buf::Hidden],
            params: params(&[count]),
            groups: spread(count),
        });
        matmul(
            ops,
            Buf::Hidden,
            x,
            tokens,
            dim,
            4 * dim,
            at.down,
            Some(at.down_bias),
            true,
        );
    }
    // The stem: each band scaled, patches of 4 bands by 3 frames, to 32
    // channels, scaled, GELU.
    elementwise(
        &mut ops,
        Kernel::In,
        [Buf::Spect, Buf::Weights, Buf::X],
        &[f, BANDS, o.in_scale as usize, o.in_shift as usize],
        f * BANDS,
    );
    let mut bands = BANDS / 4;
    elementwise(
        &mut ops,
        Kernel::Patches,
        [Buf::X, Buf::Weights, Buf::Patches],
        &[f, 1, 4, bands],
        bands * f * 12,
    );
    matmul(
        &mut ops,
        Buf::Patches,
        Buf::Y,
        bands * f,
        STEM,
        12,
        o.stem,
        None,
        false,
    );
    elementwise(
        &mut ops,
        Kernel::AffineGelu,
        [Buf::Table, Buf::Weights, Buf::Y],
        &[
            bands * f * STEM,
            STEM,
            o.stem_scale as usize,
            o.stem_shift as usize,
        ],
        bands * f * STEM,
    );
    // The blocks, the activations in `cur` as [band][frame][channel].
    let (mut cur, mut other) = (Buf::Y, Buf::X);
    let mut c = STEM;
    for block in &o.blocks {
        let tokens = bands * f;
        // Across the bands of each frame: [frame][band][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [cur, Buf::Weights, other],
            &[bands, f, c],
            tokens * c,
        );
        attention(&mut ops, other, tokens, bands, c, block.attn_f);
        feed_forward(&mut ops, other, tokens, c, block.ff_f);
        // Across the frames of each band: back to [band][frame][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [other, Buf::Weights, cur],
            &[f, bands, c],
            tokens * c,
        );
        attention(&mut ops, cur, tokens, f, c, block.attn_t);
        feed_forward(&mut ops, cur, tokens, c, block.ff_t);
        // Patches of 2 bands by 3 frames, to twice the channels.
        let out_bands = bands / 2;
        elementwise(
            &mut ops,
            Kernel::Patches,
            [cur, Buf::Weights, Buf::Patches],
            &[f, c, 2, out_bands],
            out_bands * f * 6 * c,
        );
        matmul(
            &mut ops,
            Buf::Patches,
            other,
            out_bands * f,
            2 * c,
            6 * c,
            block.conv,
            None,
            false,
        );
        let count = out_bands * f * 2 * c;
        elementwise(
            &mut ops,
            Kernel::AffineGelu,
            [Buf::Table, Buf::Weights, other],
            &[count, 2 * c, block.scale as usize, block.shift as usize],
            count,
        );
        std::mem::swap(&mut cur, &mut other);
        bands = out_bands;
        c *= 2;
    }
    // Each frame's channels × bands, channel-major, to the width.
    elementwise(
        &mut ops,
        Kernel::Flatten,
        [cur, Buf::Weights, Buf::Patches],
        &[f, bands, c],
        f * c * bands,
    );
    matmul(
        &mut ops,
        Buf::Patches,
        other,
        f,
        DIM,
        c * bands,
        o.project,
        Some(o.project_bias),
        false,
    );
    for &(attn, ff) in &o.layers {
        attention(&mut ops, other, f, f, DIM, attn);
        feed_forward(&mut ops, other, f, DIM, ff);
    }
    norm(&mut ops, other, Buf::Normed, f, DIM, o.norm);
    matmul(
        &mut ops,
        Buf::Normed,
        Buf::Logits,
        f,
        2,
        DIM,
        o.head,
        Some(o.head_bias),
        false,
    );
    ops
}

/// A recorded piece size: its dispatches' bind groups and parameters.
struct Plan {
    ops: Vec<(Kernel, usize, u64, [u32; 3])>,
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
    bind_index: HashMap<([Buf; 3], usize), usize>,
    plans: HashMap<usize, Plan>,
    pieces: crate::pace::Pieces,
}

/// The beat network on the GPU.
pub struct GpuNetwork {
    name: String,
    inner: Turned<Mutex<Inner>>,
}

impl GpuNetwork {
    /// The network on the best GPU, set up on first use; `None` without a
    /// usable one, or if setting it up failed.
    pub fn shared() -> Option<&'static GpuNetwork> {
        static NET: OnceLock<Option<GpuNetwork>> = OnceLock::new();
        NET.get_or_init(|| std::panic::catch_unwind(GpuNetwork::new).ok()?.ok())
            .as_ref()
    }

    /// Sets the network up on the best GPU.
    pub fn new() -> Result<GpuNetwork, String> {
        let (inner, name) = crate::device::open("jodel-gpu network", |device, queue, limits| {
            Inner::new(device, queue, limits)
        })?;
        Ok(GpuNetwork {
            name,
            inner: Turned::new(Mutex::new(inner)),
        })
    }
}

/// Lines for a diagnostic report: whether the network sets up on the GPU
/// (a network of its own, not [`GpuNetwork::shared`]'s), and how long a
/// 30 s piece takes on it.
pub(crate) fn report() -> Vec<String> {
    let started = std::time::Instant::now();
    let network = match std::panic::catch_unwind(GpuNetwork::new) {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return vec![format!("beat network set up: failed: {e}")],
        Err(_) => return vec!["beat network set up: panicked".into()],
    };
    let mut lines = vec![format!(
        "beat network set up: yes, in {} ms",
        started.elapsed().as_millis()
    )];
    // A 30 s piece of a quiet spectrogram.
    let spect = vec![-5.0f32; PIECE_FRAMES * BANDS];
    let mut times = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        if let Err(e) = network.forward(&spect, PIECE_FRAMES) {
            lines.push(format!("a 30 s piece of the beat network: failed: {e}"));
            return lines;
        }
        times.push(started.elapsed().as_millis());
    }
    lines.push(format!(
        "a 30 s piece of the beat network: {} ms, then {} ms",
        times[0], times[1]
    ));
    lines
}

impl NetRunner for GpuNetwork {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn forward(&self, spect: &[f32], frames: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU network failed before".to_owned())?;
        if frames == 0 || frames > PIECE_FRAMES || spect.len() != frames * BANDS {
            return Err(format!("{frames} frames, {} values", spect.len()));
        }
        let device = inner.device.clone();
        guarded(&device, || {
            inner.plan(frames);
            inner.run(frames, spect)
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
        let f = PIECE_FRAMES;
        let wide = 4 * STEM * BANDS / 4 * f;
        let sizes = [
            (Buf::Weights, weights.len()),
            (Buf::Table, f * 32),
            (Buf::Spect, f * BANDS),
            (Buf::X, wide / 4),
            (Buf::Y, wide / 4),
            (Buf::Normed, wide / 4),
            (Buf::Qkv, 3 * wide / 4),
            (Buf::Gates, BANDS / 4 * f),
            (Buf::Attended, wide / 4),
            (Buf::Hidden, wide),
            (Buf::Patches, 3 * wide / 4),
            (Buf::Logits, 2 * f),
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
                    label: Some("jodel-gpu network"),
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
        queue.write_buffer(&buffers[&Buf::Table], 0, &bytes(&rotary_table(f)));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu network logits"),
            size: (2 * f * 4) as u64,
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
            label: Some("jodel-gpu network"),
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
            label: Some("jodel-gpu network"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let source = crate::kernels_source(&[include_str!("net.wgsl")]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodel-gpu network kernels"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
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
            pieces: crate::pace::Pieces::resting(),
        })
    }

    /// The bind group of `buffers` with the parameters of plan `plan`.
    fn bind_group(&mut self, buffers: [Buf; 3], params: &wgpu::Buffer, plan: usize) -> usize {
        if let Some(&i) = self.bind_index.get(&(buffers, plan)) {
            return i;
        }
        let entry = |binding: u32, buf: Buf| wgpu::BindGroupEntry {
            binding,
            resource: self.buffers[&buf].as_entire_binding(),
        };
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodel-gpu network"),
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

    /// The plan for pieces of `frames` frames, recorded on first use.
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
            label: Some("jodel-gpu network params"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params, 0, &data);
        let recorded = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = self.bind_group(op.buffers, &params, frames);
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

    /// One piece through the recorded plan: the beat and downbeat logits.
    fn run(&mut self, frames: usize, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.queue
            .write_buffer(&self.buffers[&Buf::Spect], 0, &bytes(spect));
        let size = (2 * frames * 4) as u64;
        let (buffers, staging) = (&self.buffers, &self.staging);
        crate::pace::dispatch(
            &self.device,
            &self.queue,
            "jodel-gpu network",
            &mut self.pieces,
            &self.pipelines,
            &self.bind_groups,
            &self.plans[&frames].ops,
            |_| {},
            |encoder| {
                encoder.copy_buffer_to_buffer(&buffers[&Buf::Logits], 0, staging, 0, size);
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
        let downbeat: Vec<f32> = out.chunks(2).map(|o| o[1]).collect();
        let beat = out.chunks(2).map(|o| o[0] + o[1]).collect();
        Ok((beat, downbeat))
    }
}
