//! The speech recogniser of [`jodel_lyrics::asr`] (Qwen3-ASR-0.6B) on the
//! GPU, its large weights at 4 bits (`asr.wgsl` with `kernels.wgsl`'s
//! helpers): the encoder over the whole voice at once, then the decoder
//! with a cache of keys and values, a token a step, the next one chosen on
//! the GPU (the steps go a few at a time between reads).
//!
//! The voice is heard in parts ([`jodel_lyrics::asr::parts_of`], at most
//! [`MAX_SECONDS`]), each on its own: over minutes the decoder may loop or
//! stop early. With `JODEL_ASR_TIMES` set, each
//! piece's encoder and prompt time and its tokens' time go to stderr.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Mutex;

use jodel_lyrics::asr::packed::{PackedModel, Tensor};
use jodel_lyrics::asr::{
    audio_steps, loop_start, mel, piece_steps, sinusoids, text, transcribe_parts, Transcript, CONV,
    DIM, ENC_DIM, ENC_FF, ENC_HEADS, ENC_LAYERS, FF, HEAD, HEADS, KV_HEADS, LAYERS, PIECE,
    ROPE_BASE, VOCAB, WINDOW_PIECES,
};
use jodel_lyrics::VOCALS_RATE;

use super::{adapter, guarded, setting_up, spread, Turned, PARAM_SIZE, PARAM_STRIDE};

/// The longest part of a voice heard at once, in seconds.
pub const MAX_SECONDS: usize = jodel_lyrics::asr::PART_SECONDS as usize;
/// The most tokens written for one part.
pub const MAX_NEW: usize = 384;
/// Steps decoded between reads of the tokens.
const STEPS_A_READ: usize = 8;
/// Pieces of a second through the convolutions at once.
const CONV_BATCH: usize = 4;
/// The most dispatches in one run.
const MAX_OPS: usize = 8192;

const QKV_ENC: usize = 3 * ENC_DIM;
const QKV_DEC: usize = (HEADS + 2 * KV_HEADS) * HEAD;
const KV: usize = 2 * KV_HEADS * HEAD;
const PATCH: usize = 9 * CONV;
const BANDS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Matvec,
    Conv1,
    Patches,
    Flatten,
    Place,
    LayerNorm,
    Gelu,
    EncAttention,
    RmsNorm,
    HeadNormRope,
    Copy,
    DecAttention,
    StepPart,
    StepJoin,
    SwiGlu,
    Embed,
    Argmax,
    ColSq,
}

impl Kernel {
    const ALL: [Kernel; 19] = [
        Kernel::Matmul,
        Kernel::Matvec,
        Kernel::Conv1,
        Kernel::Patches,
        Kernel::Flatten,
        Kernel::Place,
        Kernel::LayerNorm,
        Kernel::Gelu,
        Kernel::EncAttention,
        Kernel::RmsNorm,
        Kernel::HeadNormRope,
        Kernel::Copy,
        Kernel::DecAttention,
        Kernel::StepPart,
        Kernel::StepJoin,
        Kernel::SwiGlu,
        Kernel::Embed,
        Kernel::Argmax,
        Kernel::ColSq,
    ];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "qa_matmul",
            Kernel::Matvec => "qa_matvec",
            Kernel::Conv1 => "qa_conv1",
            Kernel::Patches => "qa_patches",
            Kernel::Flatten => "qa_flatten",
            Kernel::Place => "qa_place",
            Kernel::LayerNorm => "qa_layernorm",
            Kernel::Gelu => "qa_gelu",
            Kernel::EncAttention => "qa_enc_attention",
            Kernel::RmsNorm => "qa_rmsnorm",
            Kernel::HeadNormRope => "qa_head_norm_rope",
            Kernel::Copy => "qa_copy",
            Kernel::DecAttention => "qa_dec_attention",
            Kernel::StepPart => "qa_step_part",
            Kernel::StepJoin => "qa_step_join",
            Kernel::SwiGlu => "qa_swiglu",
            Kernel::Embed => "qa_embed",
            Kernel::Argmax => "qa_argmax",
            Kernel::ColSq => "qa_colsq",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    /// The 4-bit matrices (u32).
    Weights,
    /// Biases, norms, the first convolution, positions, the rotary table.
    Small,
    Pieces,
    Conv1,
    Patches,
    Conv2,
    Conv3,
    Flat,
    ConvOut,
    Hidden,
    Normed,
    Qkv,
    Attn,
    Up,
    Audio,
    X,
    Xn,
    QkvD,
    AttnD,
    Gu,
    Act,
    Cache,
    Scratch,
    Last,
    Logits,
    Tokens,
    /// Calibration: each matrix's input columns' sums of squares.
    Stats,
}

/// Where each tensor went: a 4-bit matrix in `Weights`, else values in
/// `Small` (offsets in u32s or f32s).
struct Packed {
    weights: Vec<u32>,
    small: Vec<f32>,
    at: HashMap<String, usize>,
    /// The matrices not at 4 bits, and their format's flag ([`WIDE`],
    /// [`HALVES`]).
    wide: HashMap<String, usize>,
    /// Each matrix: its name, where it is, its columns.
    matrices: Vec<(String, usize, usize)>,
}

/// The packed model's tensors laid out for the GPU, `rope` positions of
/// rotary angles.
fn pack(model: &PackedModel, rope: usize) -> Packed {
    let mut p = Packed {
        weights: Vec::new(),
        small: Vec::new(),
        at: HashMap::new(),
        wide: Default::default(),
        matrices: Vec::new(),
    };
    for (name, t) in &model.tensors {
        match t {
            Tensor::Values(v) => {
                p.at.insert(name.clone(), p.small.len());
                p.small.extend_from_slice(v);
            }
            Tensor::Q4 { data, cols, .. } => {
                p.at.insert(name.clone(), p.weights.len());
                p.matrices.push((name.clone(), p.weights.len(), *cols));
                p.weights.extend_from_slice(data);
            }
            Tensor::Q8 { data, cols, .. } => {
                p.at.insert(name.clone(), p.weights.len());
                p.matrices.push((name.clone(), p.weights.len(), *cols));
                p.wide.insert(name.clone(), WIDE);
                p.weights.extend_from_slice(data);
            }
            Tensor::F16 { data, cols, .. } => {
                p.at.insert(name.clone(), p.weights.len());
                p.matrices.push((name.clone(), p.weights.len(), *cols));
                p.wide.insert(name.clone(), HALVES);
                p.weights.extend_from_slice(data);
            }
        }
    }
    // A tied head is the embeddings.
    if !p.at.contains_key("head") {
        if let Some(&e) = p.at.get("embed") {
            p.at.insert("head".into(), e);
            if let Some(&flag) = p.wide.get("embed") {
                p.wide.insert("head".into(), flag);
            }
        }
    }
    p.at.insert("positions".into(), p.small.len());
    p.small.extend(sinusoids(piece_steps(PIECE), ENC_DIM));
    p.at.insert("rope".into(), p.small.len());
    let half = HEAD / 2;
    for pos in 0..rope {
        for i in 0..half {
            let a = pos as f64 / ROPE_BASE.powf(2.0 * i as f64 / HEAD as f64);
            p.small.push(a.cos() as f32);
            p.small.push(a.sin() as f32);
        }
    }
    p
}
/// One dispatch.
#[derive(Clone)]
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

/// Flags of a matrix multiply.
const BIAS: usize = 1;
const ADD: usize = 2;
const GELU: usize = 4;
/// The matrix is at 8 bits.
const WIDE: usize = 8;
/// The matrix is of halves.
const HALVES: usize = 16;

/// A 4-bit matrix multiply: `dst` rows (`m × n` at `c_off`, `ldc` apart)
/// (+)= `src` rows (`k` from `a_off`, `lda` apart) times the matrix at `w`
/// (+ the bias at `bias`), GELU after if asked. One row goes through the
/// matrix-vector kernel.
#[allow(clippy::too_many_arguments)]
fn matmul(
    ops: &mut Vec<Op>,
    (src, dst): (Buf, Buf),
    (m, n, k): (usize, usize, usize),
    (a_off, lda): (usize, usize),
    (w, wide): (usize, usize),
    (c_off, ldc): (usize, usize),
    bias: Option<usize>,
    flags: usize,
) {
    let flags = if bias.is_some() { flags | BIAS } else { flags } | wide;
    let values = [m, n, k, a_off, lda, w, c_off, ldc, bias.unwrap_or(0), flags];
    if m == 1 {
        let blocks = n.div_ceil(8);
        let x = blocks.min(65_535);
        ops.push(Op {
            kernel: Kernel::Matvec,
            buffers: [src, Buf::Small, dst],
            params: params(&values),
            groups: [x as u32, blocks.div_ceil(x) as u32, 1],
        });
    } else {
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [src, Buf::Small, dst],
            params: params(&values),
            groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
        });
    }
}

/// A kernel with a workgroup a row.
fn per_row(ops: &mut Vec<Op>, kernel: Kernel, buffers: [Buf; 3], rows: usize, values: &[usize]) {
    let x = rows.clamp(1, 65_535);
    ops.push(Op {
        kernel,
        buffers,
        params: params(values),
        groups: [x as u32, rows.div_ceil(x).max(1) as u32, 1],
    });
}

/// An elementwise or per-row kernel over `count` invocations.
fn each(ops: &mut Vec<Op>, kernel: Kernel, buffers: [Buf; 3], count: usize, values: &[usize]) {
    ops.push(Op {
        kernel,
        buffers,
        params: params(values),
        groups: spread(count),
    });
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    buffers: HashMap<Buf, wgpu::Buffer>,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    at: HashMap<String, usize>,
    wide: HashMap<String, usize>,
    /// Each matrix by where it is: its name, where its statistics go, its
    /// columns, and the rows seen while calibrating.
    stats_at: HashMap<usize, (String, usize, usize)>,
    rows_seen: HashMap<usize, u64>,
    stats_len: usize,
    calibrating: bool,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_index: HashMap<[Buf; 3], usize>,
    pieces: crate::pace::Pieces,
}

/// Qwen3-ASR-0.6B on the GPU.
pub struct GpuAsr {
    name: String,
    vocab: text::Vocab,
    inner: Turned<Mutex<Inner>>,
}

/// The sizes a piece of at most [`MAX_SECONDS`] needs.
struct Sizes {
    frames: usize,
    pieces: usize,
    steps: usize,
    rows: usize,
    positions: usize,
}

fn sizes() -> Sizes {
    let frames = MAX_SECONDS * VOCALS_RATE as usize / mel::HOP;
    let steps = audio_steps(frames);
    let rows = text::prompt(steps).len();
    Sizes {
        frames,
        pieces: frames.div_ceil(PIECE),
        steps,
        rows,
        positions: rows + MAX_NEW + STEPS_A_READ,
    }
}

impl GpuAsr {
    /// The packed model on the best GPU.
    pub fn new(model: &PackedModel) -> Result<GpuAsr, String> {
        let _turn = setting_up();
        let adapter = adapter()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jodel-gpu asr"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", info.name))?;
        let probe = device.clone();
        let inner = guarded(&probe, || Inner::new(device, queue, &limits, model))
            .map_err(|e| format!("{}: {e}", info.name))?;
        Ok(GpuAsr {
            name: info.name,
            vocab: model.vocab.clone(),
            inner: Turned::new(Mutex::new(inner)),
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What is sung in `voice` (16 kHz mono): its parts each heard on
    /// their own ([`jodel_lyrics::asr::transcribe_parts`]).
    pub fn transcribe(&self, voice: &[f32]) -> Result<Transcript, String> {
        transcribe_parts(voice, &self.vocab, |part| self.tokens(part))
    }

    /// What is sung in short glimpses of `voice` (16 kHz mono), each heard
    /// on its own ([`jodel_lyrics::glimpses`]).
    pub fn glimpses(&self, voice: &[f32]) -> Result<Vec<jodel_lyrics::asr::Part>, String> {
        jodel_lyrics::glimpses::hear_glimpses(voice, |piece| Ok(self.transcribe(piece)?.text))
    }

    /// Starts (afresh) or stops collecting calibration statistics: each
    /// matrix's input columns' sums of squares over what is heard.
    pub fn calibrate(&self, on: bool) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech recogniser failed before".to_owned())?;
        if on {
            let zeros = vec![0.0f32; inner.stats_len];
            inner
                .queue
                .write_buffer(&inner.buffers[&Buf::Stats], 0, &super::bytes(&zeros));
            inner.rows_seen.clear();
        }
        inner.calibrating = on;
        Ok(())
    }

    /// The statistics collected since [`GpuAsr::calibrate`]: each matrix's
    /// name, the rows its input had, and its columns' sums of squares.
    pub fn statistics(&self) -> Result<Vec<(String, u64, Vec<f32>)>, String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech recogniser failed before".to_owned())?;
        let len = inner.stats_len;
        let sums = inner.run(&[], Some((Buf::Stats, 0, len)))?;
        let mut out: Vec<(String, u64, Vec<f32>)> = inner
            .stats_at
            .iter()
            .map(|(w, (name, at, cols))| {
                let rows = inner.rows_seen.get(w).copied().unwrap_or(0);
                (name.clone(), rows, sums[*at..*at + *cols].to_vec())
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// The encoder's output for one piece (`[steps][1024]`), for checks.
    pub fn audio(&self, voice: &[f32]) -> Result<Vec<f32>, String> {
        let (features, frames) = mel::features(voice, 8);
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech recogniser failed before".to_owned())?;
        let device = inner.device.clone();
        guarded(&device, || {
            inner.upload(&features, frames);
            let ops = inner.encoder_ops(frames);
            inner.run(&ops, Some((Buf::Audio, 0, audio_steps(frames) * DIM)))
        })
    }

    /// The tokens written for one part (at most [`MAX_SECONDS`]), a loop cut
    /// short ([`jodel_lyrics::asr::loop_start`]).
    pub fn tokens(&self, voice: &[f32]) -> Result<Vec<u32>, String> {
        let (features, frames) = mel::features(voice, 8);
        if frames == 0 {
            return Ok(Vec::new());
        }
        if frames > sizes().frames {
            return Err(format!("{frames} frames, longer than a part"));
        }
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech recogniser failed before".to_owned())?;
        let device = inner.device.clone();
        guarded(&device, || inner.hear(&features, frames))
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        model: &PackedModel,
    ) -> Result<Inner, String> {
        let s = sizes();
        let packed = pack(model, s.positions);
        let width = PIECE;
        let s1 = width.div_ceil(2);
        let s2 = s1.div_ceil(2);
        let s3 = s2.div_ceil(2);
        let b = CONV_BATCH;
        let f32s = [
            (Buf::Small, packed.small.len()),
            (Buf::Pieces, s.pieces * mel::MELS * width),
            (Buf::Conv1, b * 64 * s1 * CONV),
            (Buf::Patches, b * 32 * s2 * PATCH),
            (Buf::Conv2, b * 32 * s2 * CONV),
            (Buf::Conv3, b * BANDS * s3 * CONV),
            (Buf::Flat, b * s3 * BANDS * CONV),
            (Buf::ConvOut, b * s3 * ENC_DIM),
            (Buf::Hidden, s.steps * ENC_DIM),
            (Buf::Normed, s.steps * ENC_DIM),
            (Buf::Qkv, s.steps * QKV_ENC),
            (Buf::Attn, s.steps * ENC_DIM),
            (Buf::Up, s.steps * ENC_FF),
            (Buf::Audio, s.steps * DIM),
            (Buf::X, s.rows * DIM),
            (Buf::Xn, s.rows * DIM),
            (Buf::QkvD, s.rows * QKV_DEC),
            (Buf::AttnD, s.rows * HEADS * HEAD),
            (Buf::Gu, s.rows * 2 * FF),
            (Buf::Act, s.rows * FF),
            (Buf::Cache, LAYERS * s.positions * KV),
            (Buf::Scratch, HEADS * s.positions.div_ceil(128) * 130),
            (Buf::Last, DIM),
            (Buf::Logits, VOCAB),
            (Buf::Tokens, s.positions),
            (Buf::Stats, packed.matrices.iter().map(|m| m.2).sum()),
        ];
        let mut all: Vec<(Buf, usize)> = f32s.to_vec();
        all.push((Buf::Weights, packed.weights.len()));
        for &(buf, len) in &all {
            let size = (len * 4) as u64;
            if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
                return Err(format!(
                    "the GPU's buffers are too small for {buf:?} ({size} bytes)"
                ));
            }
        }
        let buffers: HashMap<Buf, wgpu::Buffer> = all
            .iter()
            .map(|&(buf, len)| {
                let b = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jodel-gpu asr"),
                    size: (len.max(1) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                (buf, b)
            })
            .collect();
        queue.write_buffer(
            &buffers[&Buf::Weights],
            0,
            &packed
                .weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        queue.write_buffer(&buffers[&Buf::Small], 0, &super::bytes(&packed.small));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu asr params"),
            size: MAX_OPS as u64 * PARAM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu asr tokens"),
            size: (s.positions.max(VOCAB).max(s.steps * DIM) * 4) as u64,
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
            label: Some("jodel-gpu asr"),
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
                storage(4, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodel-gpu asr"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let source = format!(
            "{}\n{}",
            include_str!("kernels.wgsl"),
            include_str!("asr.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodel-gpu asr kernels"),
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
            params,
            staging,
            at: packed.at,
            wide: packed.wide,
            stats_len: packed.matrices.iter().map(|m| m.2).sum(),
            stats_at: {
                let mut at = 0;
                packed
                    .matrices
                    .iter()
                    .map(|(name, w, cols)| {
                        at += cols;
                        (*w, (name.clone(), at - cols, *cols))
                    })
                    .collect()
            },
            rows_seen: HashMap::new(),
            calibrating: false,
            bind_groups: Vec::new(),
            bind_index: HashMap::new(),
            pieces: crate::pace::Pieces::resting(),
        })
    }

    fn w(&self, name: &str) -> usize {
        self.at[name]
    }

    /// A matrix: where it is, and its format's flag.
    fn m(&self, name: &str) -> (usize, usize) {
        (self.at[name], self.wide.get(name).copied().unwrap_or(0))
    }

    /// A matrix's bits a value.
    fn bits(&self, name: &str) -> usize {
        match self.wide.get(name) {
            Some(&HALVES) => 16,
            Some(_) => 8,
            None => 4,
        }
    }

    /// The bind group of `buffers` (and the weights, and the parameters).
    fn bind_group(&mut self, buffers: [Buf; 3]) -> usize {
        if let Some(&i) = self.bind_index.get(&buffers) {
            return i;
        }
        let entry = |binding: u32, buf: Buf| wgpu::BindGroupEntry {
            binding,
            resource: self.buffers[&buf].as_entire_binding(),
        };
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodel-gpu asr"),
            layout: &self.layout,
            entries: &[
                entry(0, buffers[0]),
                entry(1, buffers[1]),
                entry(2, buffers[2]),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: 0,
                        size: NonZeroU64::new(PARAM_SIZE),
                    }),
                },
                entry(4, Buf::Weights),
            ],
        });
        self.bind_groups.push(group);
        self.bind_index.insert(buffers, self.bind_groups.len() - 1);
        self.bind_groups.len() - 1
    }

    /// Runs `ops` (their parameters written first); `read` copies `count`
    /// f32s of a buffer from an offset back.
    fn run(&mut self, ops: &[Op], read: Option<(Buf, usize, usize)>) -> Result<Vec<f32>, String> {
        let calibrated;
        let ops = if self.calibrating {
            calibrated = self.with_statistics(ops);
            &calibrated[..]
        } else {
            ops
        };
        assert!(ops.len() <= MAX_OPS, "{} dispatches", ops.len());
        let mut data = vec![0u8; ops.len() * PARAM_STRIDE as usize];
        for (i, op) in ops.iter().enumerate() {
            for (j, v) in op.params.iter().enumerate() {
                let at = i * PARAM_STRIDE as usize + 4 * j;
                data[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        self.queue.write_buffer(&self.params, 0, &data);
        let recorded: Vec<crate::pace::Op<Kernel>> = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = self.bind_group(op.buffers);
                (op.kernel, group, i as u64 * PARAM_STRIDE, op.groups)
            })
            .collect();
        let (buffers, staging) = (&self.buffers, &self.staging);
        crate::pace::dispatch(
            &self.device,
            &self.queue,
            "jodel-gpu asr",
            &mut self.pieces,
            &self.pipelines,
            &self.bind_groups,
            &recorded,
            |_| {},
            |encoder| {
                if let Some((buf, from, count)) = read {
                    encoder.copy_buffer_to_buffer(
                        &buffers[&buf],
                        (from * 4) as u64,
                        staging,
                        0,
                        (count * 4) as u64,
                    );
                }
            },
        )?;
        let Some((_, _, count)) = read else {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| e.to_string())?;
            return Ok(Vec::new());
        };
        let size = (count * 4) as u64;
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

    /// `ops` with each matrix multiply's input columns' sums of squares
    /// added to the statistics first, its rows counted.
    fn with_statistics(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut out = Vec::with_capacity(ops.len() * 2);
        for op in ops {
            if matches!(op.kernel, Kernel::Matmul | Kernel::Matvec) {
                let [m, _, k, a_off, lda, w] = [0, 1, 2, 3, 4, 5].map(|i| op.params[i] as usize);
                if let Some(&(_, s_off, _)) = self.stats_at.get(&w) {
                    *self.rows_seen.entry(w).or_default() += m as u64;
                    out.push(Op {
                        kernel: Kernel::ColSq,
                        buffers: [op.buffers[0], Buf::Small, Buf::Stats],
                        params: params(&[m, k, a_off, lda, s_off]),
                        groups: spread(k),
                    });
                }
            }
            out.push(op.clone());
        }
        out
    }

    /// The encoder's dispatches for `frames` frames of features (already in
    /// `Pieces`): the audio's steps in `Audio`.
    fn encoder_ops(&self, frames: usize) -> Vec<Op> {
        let mut ops = Vec::new();
        let pieces = frames.div_ceil(PIECE);
        let width = frames.min(PIECE);
        let (s1, s2, s3) = (
            width.div_ceil(2),
            width.div_ceil(2).div_ceil(2),
            piece_steps(width),
        );
        let steps = audio_steps(frames);
        for first in (0..pieces).step_by(CONV_BATCH) {
            let b = (pieces - first).min(CONV_BATCH);
            each(
                &mut ops,
                Kernel::Conv1,
                [Buf::Pieces, Buf::Small, Buf::Conv1],
                b * 64 * s1 * CONV,
                &[b, width, s1, self.w("conv1.w"), self.w("conv1.b"), first],
            );
            each(
                &mut ops,
                Kernel::Patches,
                [Buf::Conv1, Buf::Small, Buf::Patches],
                b * 32 * s2 * PATCH,
                &[b, 64, s1, 32, s2],
            );
            matmul(
                &mut ops,
                (Buf::Patches, Buf::Conv2),
                (b * 32 * s2, CONV, PATCH),
                (0, PATCH),
                self.m("conv2.w"),
                (0, CONV),
                Some(self.w("conv2.b")),
                GELU,
            );
            each(
                &mut ops,
                Kernel::Patches,
                [Buf::Conv2, Buf::Small, Buf::Patches],
                b * BANDS * s3 * PATCH,
                &[b, 32, s2, BANDS, s3],
            );
            matmul(
                &mut ops,
                (Buf::Patches, Buf::Conv3),
                (b * BANDS * s3, CONV, PATCH),
                (0, PATCH),
                self.m("conv3.w"),
                (0, CONV),
                Some(self.w("conv3.b")),
                GELU,
            );
            each(
                &mut ops,
                Kernel::Flatten,
                [Buf::Conv3, Buf::Small, Buf::Flat],
                b * s3 * BANDS * CONV,
                &[b, s3],
            );
            matmul(
                &mut ops,
                (Buf::Flat, Buf::ConvOut),
                (b * s3, ENC_DIM, BANDS * CONV),
                (0, BANDS * CONV),
                self.m("conv_out.w"),
                (0, ENC_DIM),
                None,
                0,
            );
            let last = first + b == pieces;
            let keep = if last {
                piece_steps(frames - (pieces - 1) * PIECE)
            } else {
                s3
            };
            each(
                &mut ops,
                Kernel::Place,
                [Buf::ConvOut, Buf::Small, Buf::Hidden],
                b * s3 * ENC_DIM,
                &[b, s3, keep, first * s3, self.w("positions")],
            );
        }
        let window = s3 * WINDOW_PIECES;
        for l in 0..ENC_LAYERS {
            let n = |what: &str| self.w(&format!("enc{l}.{what}"));
            let nm = |what: &str| self.m(&format!("enc{l}.{what}"));
            each(
                &mut ops,
                Kernel::LayerNorm,
                [Buf::Hidden, Buf::Small, Buf::Normed],
                steps,
                &[steps, ENC_DIM, n("ln.w"), n("ln.b")],
            );
            matmul(
                &mut ops,
                (Buf::Normed, Buf::Qkv),
                (steps, QKV_ENC, ENC_DIM),
                (0, ENC_DIM),
                nm("qkv.w"),
                (0, QKV_ENC),
                Some(n("qkv.b")),
                0,
            );
            ops.push(Op {
                kernel: Kernel::EncAttention,
                buffers: [Buf::Qkv, Buf::Small, Buf::Attn],
                params: params(&[steps, window]),
                groups: [steps.div_ceil(64) as u32, ENC_HEADS as u32, 1],
            });
            matmul(
                &mut ops,
                (Buf::Attn, Buf::Hidden),
                (steps, ENC_DIM, ENC_DIM),
                (0, ENC_DIM),
                nm("o.w"),
                (0, ENC_DIM),
                Some(n("o.b")),
                ADD,
            );
            each(
                &mut ops,
                Kernel::LayerNorm,
                [Buf::Hidden, Buf::Small, Buf::Normed],
                steps,
                &[steps, ENC_DIM, n("fln.w"), n("fln.b")],
            );
            matmul(
                &mut ops,
                (Buf::Normed, Buf::Up),
                (steps, ENC_FF, ENC_DIM),
                (0, ENC_DIM),
                nm("up.w"),
                (0, ENC_FF),
                Some(n("up.b")),
                GELU,
            );
            matmul(
                &mut ops,
                (Buf::Up, Buf::Hidden),
                (steps, ENC_DIM, ENC_FF),
                (0, ENC_FF),
                nm("down.w"),
                (0, ENC_DIM),
                Some(n("down.b")),
                ADD,
            );
        }
        each(
            &mut ops,
            Kernel::LayerNorm,
            [Buf::Hidden, Buf::Small, Buf::Normed],
            steps,
            &[steps, ENC_DIM, self.w("post.w"), self.w("post.b")],
        );
        matmul(
            &mut ops,
            (Buf::Normed, Buf::Attn),
            (steps, ENC_DIM, ENC_DIM),
            (0, ENC_DIM),
            self.m("proj1.w"),
            (0, ENC_DIM),
            Some(self.w("proj1.b")),
            GELU,
        );
        matmul(
            &mut ops,
            (Buf::Attn, Buf::Audio),
            (steps, DIM, ENC_DIM),
            (0, ENC_DIM),
            self.m("proj2.w"),
            (0, DIM),
            Some(self.w("proj2.b")),
            0,
        );
        ops
    }

    /// The decoder's dispatches for `rows` rows of `X` at positions from
    /// `start`, then the next token (its id at `Tokens[start + rows]`).
    fn decoder_ops(&self, ops: &mut Vec<Op>, rows: usize, start: usize) {
        let s = sizes();
        let rope = self.w("rope");
        for l in 0..LAYERS {
            let n = |what: &str| self.w(&format!("dec{l}.{what}"));
            let nm = |what: &str| self.m(&format!("dec{l}.{what}"));
            per_row(
                ops,
                Kernel::RmsNorm,
                [Buf::X, Buf::Small, Buf::Xn],
                rows,
                &[rows, DIM, n("norm"), 0, DIM],
            );
            matmul(
                ops,
                (Buf::Xn, Buf::QkvD),
                (rows, QKV_DEC, DIM),
                (0, DIM),
                nm("qkv.w"),
                (0, QKV_DEC),
                None,
                0,
            );
            each(
                ops,
                Kernel::HeadNormRope,
                [Buf::Small, Buf::Small, Buf::QkvD],
                rows * HEADS,
                &[rows, QKV_DEC, 0, HEADS, n("q_norm"), start, rope],
            );
            each(
                ops,
                Kernel::HeadNormRope,
                [Buf::Small, Buf::Small, Buf::QkvD],
                rows * KV_HEADS,
                &[
                    rows,
                    QKV_DEC,
                    HEADS * HEAD,
                    KV_HEADS,
                    n("k_norm"),
                    start,
                    rope,
                ],
            );
            let cache = l * s.positions * KV;
            each(
                ops,
                Kernel::Copy,
                [Buf::QkvD, Buf::Small, Buf::Cache],
                rows * KV,
                &[rows, KV, HEADS * HEAD, QKV_DEC, cache + start * KV, KV],
            );
            if rows == 1 {
                let chunks = (start + 1).div_ceil(128);
                ops.push(Op {
                    kernel: Kernel::StepPart,
                    buffers: [Buf::QkvD, Buf::Cache, Buf::Scratch],
                    params: params(&[start, cache, 0, chunks]),
                    groups: [chunks as u32, HEADS as u32, 1],
                });
                ops.push(Op {
                    kernel: Kernel::StepJoin,
                    buffers: [Buf::Scratch, Buf::Small, Buf::AttnD],
                    params: params(&[0, chunks]),
                    groups: [HEADS as u32, 1, 1],
                });
            } else {
                ops.push(Op {
                    kernel: Kernel::DecAttention,
                    buffers: [Buf::QkvD, Buf::Cache, Buf::AttnD],
                    params: params(&[rows, start, cache]),
                    groups: [rows.div_ceil(64) as u32, HEADS as u32, 1],
                });
            }
            matmul(
                ops,
                (Buf::AttnD, Buf::X),
                (rows, DIM, HEADS * HEAD),
                (0, HEADS * HEAD),
                nm("o.w"),
                (0, DIM),
                None,
                ADD,
            );
            per_row(
                ops,
                Kernel::RmsNorm,
                [Buf::X, Buf::Small, Buf::Xn],
                rows,
                &[rows, DIM, n("post_norm"), 0, DIM],
            );
            matmul(
                ops,
                (Buf::Xn, Buf::Gu),
                (rows, 2 * FF, DIM),
                (0, DIM),
                nm("gate_up.w"),
                (0, 2 * FF),
                None,
                0,
            );
            each(
                ops,
                Kernel::SwiGlu,
                [Buf::Gu, Buf::Small, Buf::Act],
                rows * FF,
                &[rows],
            );
            matmul(
                ops,
                (Buf::Act, Buf::X),
                (rows, DIM, FF),
                (0, FF),
                nm("down.w"),
                (0, DIM),
                None,
                ADD,
            );
        }
        per_row(
            ops,
            Kernel::RmsNorm,
            [Buf::X, Buf::Small, Buf::Last],
            1,
            &[1, DIM, self.w("norm"), (rows - 1) * DIM, DIM],
        );
        matmul(
            ops,
            (Buf::Last, Buf::Logits),
            (1, VOCAB, DIM),
            (0, DIM),
            self.m("head"),
            (0, VOCAB),
            None,
            0,
        );
        ops.push(Op {
            kernel: Kernel::Argmax,
            buffers: [Buf::Logits, Buf::Small, Buf::Tokens],
            params: params(&[VOCAB, start + rows]),
            groups: [1, 1, 1],
        });
    }

    /// The features as pieces of a second, the last padded with zeros (as
    /// all are, to the longest: a whole piece unless the voice is shorter).
    fn upload(&self, features: &[f32], frames: usize) {
        let width = frames.min(PIECE);
        let pieces = frames.div_ceil(PIECE);
        let mut x = vec![0.0f32; pieces * mel::MELS * width];
        for p in 0..pieces {
            let start = p * PIECE;
            let len = (frames - start).min(width);
            for b in 0..mel::MELS {
                let row = &mut x[(p * mel::MELS + b) * width..][..len];
                row.copy_from_slice(&features[b * frames + start..][..len]);
            }
        }
        self.queue
            .write_buffer(&self.buffers[&Buf::Pieces], 0, &super::bytes(&x));
    }

    /// The tokens the model writes for `frames` frames of features.
    fn hear(&mut self, features: &[f32], frames: usize) -> Result<Vec<u32>, String> {
        self.upload(features, frames);
        let steps = audio_steps(frames);
        let prompt = text::prompt(steps);
        let ids: Vec<f32> = prompt.iter().map(|&i| i as f32).collect();
        self.queue
            .write_buffer(&self.buffers[&Buf::Tokens], 0, &super::bytes(&ids));
        let mut ops = self.encoder_ops(frames);
        let rows = prompt.len();
        each(
            &mut ops,
            Kernel::Embed,
            [Buf::Tokens, Buf::Small, Buf::X],
            rows * DIM,
            &[rows, 0, self.w("embed"), 0, self.bits("embed")],
        );
        let audio_at = prompt
            .iter()
            .position(|&i| i == text::AUDIO_PAD)
            .unwrap_or(0);
        each(
            &mut ops,
            Kernel::Copy,
            [Buf::Audio, Buf::Small, Buf::X],
            steps * DIM,
            &[steps, DIM, 0, DIM, audio_at * DIM, DIM],
        );
        self.decoder_ops(&mut ops, rows, 0);
        let t0 = std::time::Instant::now();
        let enc = self.encoder_ops(frames).len();
        let first = self.run(&ops, Some((Buf::Tokens, rows, 1)))?;
        let prefill = t0.elapsed();
        let t1 = std::time::Instant::now();
        let mut out: Vec<u32> = first.iter().map(|&v| v as u32).collect();
        let s = sizes();
        while !out.last().is_some_and(|&t| text::is_end(t))
            && out.len() < MAX_NEW
            && rows + out.len() + STEPS_A_READ < s.positions
        {
            let mut ops = Vec::new();
            let pos = rows + out.len() - 1;
            for k in 0..STEPS_A_READ {
                each(
                    &mut ops,
                    Kernel::Embed,
                    [Buf::Tokens, Buf::Small, Buf::X],
                    DIM,
                    &[1, pos + k, self.w("embed"), 0, self.bits("embed")],
                );
                self.decoder_ops(&mut ops, 1, pos + k);
            }
            let got = self.run(&ops, Some((Buf::Tokens, pos + 1, STEPS_A_READ)))?;
            for v in got {
                let t = v as u32;
                out.push(t);
                if text::is_end(t) {
                    break;
                }
                if let Some(keep) = loop_start(&out) {
                    out.truncate(keep);
                    out.push(text::IM_END);
                    break;
                }
            }
        }
        if std::env::var_os("JODEL_ASR_TIMES").is_some() {
            eprintln!("asr: {steps} steps, {rows} rows, {enc} encoder ops, encoder+prefill {prefill:.2?}, {} tokens in {:.2?}", out.len(), t1.elapsed());
        }
        while out.last().is_some_and(|&t| text::is_end(t)) {
            out.pop();
        }
        out.truncate(MAX_NEW);
        Ok(out)
    }
}
