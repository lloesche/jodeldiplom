//! The speech model of [`jodel_lyrics::speech`] (wav2vec2-base-960h) on the
//! GPU: the same network, weights and pieces as the CPU's, as one list of
//! dispatches a piece (`speech.wgsl` with `kernels.wgsl`'s matrix
//! multiply), giving the same logits to within rounding.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Mutex;

use jodel_lyrics::ctc::{Acoustic, Emissions};
use jodel_lyrics::speech::{
    emissions_in_pieces, Levels, Wav2Vec2, CONTEXT_SECONDS, CONV, HOP, PIECE_SECONDS, RECEPTIVE,
};
use jodel_lyrics::{Maker, VOCALS_RATE};

use super::{adapter, bytes, guarded, spread, PARAM_SIZE, PARAM_STRIDE};

const CHANNELS: usize = 512;
const DIM: usize = 768;
const QKV: usize = 3 * DIM;
const FF: usize = 3072;
const LAYERS: usize = 12;
const POS: usize = 128 * 48;
const TOKENS: usize = 32;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Conv0,
    ChanNormGelu,
    Gelu,
    LayerNorm,
    PosPatches,
    PosAdd,
    Attention,
}

impl Kernel {
    const ALL: [Kernel; 8] = [
        Kernel::Matmul,
        Kernel::Conv0,
        Kernel::ChanNormGelu,
        Kernel::Gelu,
        Kernel::LayerNorm,
        Kernel::PosPatches,
        Kernel::PosAdd,
        Kernel::Attention,
    ];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Conv0 => "sp_conv0",
            Kernel::ChanNormGelu => "sp_chan_norm_gelu",
            Kernel::Gelu => "sp_gelu",
            Kernel::LayerNorm => "sp_layernorm",
            Kernel::PosPatches => "sp_pos_patches",
            Kernel::PosAdd => "sp_pos_add",
            Kernel::Attention => "sp_attention",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Weights,
    Audio,
    ConvA,
    ConvB,
    Stats,
    Hidden,
    Qkv,
    Attended,
    Up,
    Patches,
    Pos,
    Logits,
}

/// The most samples a piece has: its frames and the context either side.
fn longest_piece() -> usize {
    let frames_of = |s: f64| (s * f64::from(VOCALS_RATE)) as usize / HOP;
    let frames = frames_of(PIECE_SECONDS) + 2 * frames_of(CONTEXT_SECONDS);
    (frames - 1) * HOP + RECEPTIVE
}

/// The frames after each convolution of `samples` samples.
fn conv_frames(samples: usize) -> [usize; 7] {
    let mut out = [0; 7];
    let mut n = samples;
    for (l, &(k, s)) in CONV.iter().enumerate() {
        n = if n < k { 0 } else { (n - k) / s + 1 };
        out[l] = n;
    }
    out
}

/// The weights as one buffer's contents, and where each is (in f32s).
fn pack(model: &Wav2Vec2) -> (Vec<f32>, HashMap<String, u32>) {
    let mut data: Vec<f32> = Vec::with_capacity(95_000_000);
    let mut at = HashMap::new();
    let mut parts: HashMap<String, Vec<f32>> = HashMap::new();
    model.visit(&mut |name, values| {
        // Each layer's queries, keys and values go side by side (below).
        let fused = [".q.", ".k.", ".v."].iter().any(|p| name.contains(p));
        if fused {
            parts.insert(name.to_owned(), values.to_vec());
        } else {
            at.insert(name.to_owned(), data.len() as u32);
            data.extend_from_slice(values);
        }
    });
    for l in 0..LAYERS {
        let mut take = |what: &str| parts.remove(&format!("{l}.{what}")).unwrap_or_default();
        let (q, k, v) = (take("q.w"), take("k.w"), take("v.w"));
        at.insert(format!("{l}.qkv.w"), data.len() as u32);
        for i in 0..DIM {
            for w in [&q, &k, &v] {
                data.extend_from_slice(&w[i * DIM..(i + 1) * DIM]);
            }
        }
        at.insert(format!("{l}.qkv.b"), data.len() as u32);
        for b in ["q.b", "k.b", "v.b"] {
            data.extend(take(b));
        }
    }
    (data, at)
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

/// `dst` (`m × n` at `c_off`, rows `ldc` apart) (+)= `src` rows (`k` values
/// from `a_off`, `lda` apart) times the weight at `w` (+ the bias at `bias`).
#[allow(clippy::too_many_arguments)]
fn matmul(
    ops: &mut Vec<Op>,
    src: Buf,
    dst: Buf,
    (m, n, k): (usize, usize, usize),
    lda: usize,
    w: u32,
    (c_off, ldc): (usize, usize),
    bias: Option<u32>,
    add: bool,
) {
    let flags = usize::from(bias.is_some()) | (usize::from(add) << 1);
    ops.push(Op {
        kernel: Kernel::Matmul,
        buffers: [src, Buf::Weights, dst],
        params: params(&[
            m,
            n,
            k,
            0,
            lda,
            w as usize,
            c_off,
            ldc,
            bias.unwrap_or(0) as usize,
            flags,
        ]),
        groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
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

/// The dispatches of a piece of `samples` samples; its frames.
fn plan(at: &HashMap<String, u32>, samples: usize) -> (Vec<Op>, usize) {
    let w = |name: &str| at[name] as usize;
    let wu = |name: &str| at[name];
    let frames = conv_frames(samples);
    let mut ops = Vec::new();
    let t0 = frames[0];
    each(
        &mut ops,
        Kernel::Conv0,
        [Buf::Audio, Buf::Weights, Buf::ConvA],
        t0 * CHANNELS,
        &[t0, w("conv0")],
    );
    // The GroupNorm at the whole voice's statistics, written to `Stats`
    // before each piece.
    each(
        &mut ops,
        Kernel::ChanNormGelu,
        [Buf::Stats, Buf::Weights, Buf::ConvA],
        t0 * CHANNELS,
        &[t0 * CHANNELS, CHANNELS, w("gn.w"), w("gn.b")],
    );
    let (mut from, mut to) = (Buf::ConvA, Buf::ConvB);
    for (l, &(k, s)) in CONV.iter().enumerate().skip(1) {
        let m = frames[l];
        matmul(
            &mut ops,
            from,
            to,
            (m, CHANNELS, k * CHANNELS),
            s * CHANNELS,
            wu(&format!("conv{l}")),
            (0, CHANNELS),
            None,
            false,
        );
        each(
            &mut ops,
            Kernel::Gelu,
            [Buf::Stats, Buf::Weights, to],
            m * CHANNELS,
            &[m * CHANNELS],
        );
        std::mem::swap(&mut from, &mut to);
    }
    let f = frames[6];
    let norm = |ops: &mut Vec<Op>, buf: Buf, dim: usize, name: &str| {
        each(
            ops,
            Kernel::LayerNorm,
            [Buf::Stats, Buf::Weights, buf],
            f,
            &[f, dim, w(&format!("{name}.w")), w(&format!("{name}.b"))],
        );
    };
    norm(&mut ops, from, CHANNELS, "fp.ln");
    matmul(
        &mut ops,
        from,
        Buf::Hidden,
        (f, DIM, CHANNELS),
        CHANNELS,
        wu("proj.w"),
        (0, DIM),
        Some(wu("proj.b")),
        false,
    );
    for g in 0..16 {
        each(
            &mut ops,
            Kernel::PosPatches,
            [Buf::Hidden, Buf::Weights, Buf::Patches],
            f * POS,
            &[f, g],
        );
        matmul(
            &mut ops,
            Buf::Patches,
            Buf::Pos,
            (f, 48, POS),
            POS,
            wu(&format!("pos{g}")),
            (g * 48, DIM),
            None,
            false,
        );
    }
    each(
        &mut ops,
        Kernel::PosAdd,
        [Buf::Pos, Buf::Weights, Buf::Hidden],
        f * DIM,
        &[f * DIM, w("pos.b")],
    );
    norm(&mut ops, Buf::Hidden, DIM, "enc.ln");
    for l in 0..LAYERS {
        let n = |what: &str| wu(&format!("{l}.{what}"));
        matmul(
            &mut ops,
            Buf::Hidden,
            Buf::Qkv,
            (f, QKV, DIM),
            DIM,
            n("qkv.w"),
            (0, QKV),
            Some(n("qkv.b")),
            false,
        );
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Qkv, Buf::Weights, Buf::Attended],
            params: params(&[f]),
            groups: [f.div_ceil(64) as u32, 12, 1],
        });
        matmul(
            &mut ops,
            Buf::Attended,
            Buf::Hidden,
            (f, DIM, DIM),
            DIM,
            n("o.w"),
            (0, DIM),
            Some(n("o.b")),
            true,
        );
        norm(&mut ops, Buf::Hidden, DIM, &format!("{l}.ln"));
        matmul(
            &mut ops,
            Buf::Hidden,
            Buf::Up,
            (f, FF, DIM),
            DIM,
            n("up.w"),
            (0, FF),
            Some(n("up.b")),
            false,
        );
        each(
            &mut ops,
            Kernel::Gelu,
            [Buf::Stats, Buf::Weights, Buf::Up],
            f * FF,
            &[f * FF],
        );
        matmul(
            &mut ops,
            Buf::Up,
            Buf::Hidden,
            (f, DIM, FF),
            FF,
            n("down.w"),
            (0, DIM),
            Some(n("down.b")),
            true,
        );
        norm(&mut ops, Buf::Hidden, DIM, &format!("{l}.fln"));
    }
    matmul(
        &mut ops,
        Buf::Hidden,
        Buf::Logits,
        (f, TOKENS, DIM),
        DIM,
        wu("head.w"),
        (0, TOKENS),
        Some(wu("head.b")),
        false,
    );
    (ops, f)
}

/// A recorded piece length: its dispatches' bind groups and parameters.
struct Plan {
    ops: Vec<(Kernel, usize, u64, [u32; 3])>,
    frames: usize,
    _params: wgpu::Buffer,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    buffers: HashMap<Buf, wgpu::Buffer>,
    staging: wgpu::Buffer,
    at: HashMap<String, u32>,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_index: HashMap<([Buf; 3], usize), usize>,
    plans: HashMap<usize, Plan>,
    pieces: crate::pace::Pieces,
}

/// wav2vec2-base-960h on the GPU.
pub struct GpuSpeech {
    name: String,
    /// The first convolution's weights, for a voice's [`Levels`].
    conv0: Vec<f32>,
    inner: Mutex<Inner>,
}

impl GpuSpeech {
    /// The model on the best GPU, from the CPU's (its weights as they are).
    pub fn new(model: &Wav2Vec2) -> Result<GpuSpeech, String> {
        let adapter = adapter()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jodel-gpu speech"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", info.name))?;
        let probe = device.clone();
        let inner = guarded(&probe, || Inner::new(device, queue, &limits, model))
            .map_err(|e| format!("{}: {e}", info.name))?;
        let mut conv0 = Vec::new();
        model.visit(&mut |name, values| {
            if name == "conv0" {
                conv0 = values.to_vec();
            }
        });
        Ok(GpuSpeech {
            name: info.name,
            conv0,
            inner: Mutex::new(inner),
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The logits (`frames × 32`) of a voice at 16 kHz heard on its own, in
    /// one piece (at most a piece and its context long): normalised at its
    /// own levels, then the network.
    pub fn logits(&self, voice: &[f32]) -> Result<Vec<f32>, String> {
        self.logits_at(voice, &Levels::of(&self.conv0, voice))
    }

    /// The logits (`frames × 32`) of a piece of a voice at 16 kHz (at most
    /// a piece and its context long), heard at the whole voice's `levels`.
    pub fn logits_at(&self, piece: &[f32], levels: &Levels) -> Result<Vec<f32>, String> {
        if piece.len() > longest_piece() {
            return Err(format!("{} samples, longer than a piece", piece.len()));
        }
        if Wav2Vec2::frames_of(piece.len()) == 0 {
            return Ok(Vec::new());
        }
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech model failed before".to_owned())?;
        let device = inner.device.clone();
        let x = levels.normalise(piece);
        guarded(&device, || {
            inner.plan(x.len());
            inner.run(&x, &levels.channels(&x))
        })
    }
}

impl Acoustic for GpuSpeech {
    fn maker(&self) -> Maker {
        jodel_lyrics::speech::maker()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        let levels = Levels::of(&self.conv0, vocals_16k);
        emissions_in_pieces(vocals_16k, |piece| self.logits_at(piece, &levels).ok())
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        model: &Wav2Vec2,
    ) -> Result<Inner, String> {
        let (weights, at) = pack(model);
        let samples = longest_piece();
        let frames = conv_frames(samples);
        let f = frames[6];
        let sizes = [
            (Buf::Weights, weights.len()),
            (Buf::Audio, samples),
            (Buf::ConvA, frames[0] * CHANNELS),
            (Buf::ConvB, frames[1] * CHANNELS),
            (Buf::Stats, 2 * CHANNELS),
            (Buf::Hidden, f * DIM),
            (Buf::Qkv, f * QKV),
            (Buf::Attended, f * DIM),
            (Buf::Up, f * FF),
            (Buf::Patches, f * POS),
            (Buf::Pos, f * DIM),
            (Buf::Logits, f * TOKENS),
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
                    label: Some("jodel-gpu speech"),
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
        drop(weights);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu speech logits"),
            size: (f * TOKENS * 4) as u64,
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
            label: Some("jodel-gpu speech"),
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
            label: Some("jodel-gpu speech"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let source = format!(
            "{}\n{}",
            include_str!("kernels.wgsl"),
            include_str!("speech.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodel-gpu speech kernels"),
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
            at,
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
            label: Some("jodel-gpu speech"),
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

    /// The plan for pieces of `samples` samples, recorded on first use.
    fn plan(&mut self, samples: usize) {
        if self.plans.contains_key(&samples) {
            return;
        }
        let (ops, frames) = plan(&self.at, samples);
        let mut data = vec![0u8; ops.len() * PARAM_STRIDE as usize];
        for (i, op) in ops.iter().enumerate() {
            for (j, v) in op.params.iter().enumerate() {
                let at = i * PARAM_STRIDE as usize + 4 * j;
                data[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodel-gpu speech params"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params, 0, &data);
        let recorded = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = self.bind_group(op.buffers, &params, samples);
                (op.kernel, group, i as u64 * PARAM_STRIDE, op.groups)
            })
            .collect();
        // Pieces come in a few lengths (the last of a song's is shorter):
        // older plans go when there are many.
        if self.plans.len() > 8 {
            self.plans.clear();
            self.bind_groups.clear();
            self.bind_index.clear();
            return self.plan(samples);
        }
        self.plans.insert(
            samples,
            Plan {
                ops: recorded,
                frames,
                _params: params,
            },
        );
    }

    /// One piece through its recorded plan, its first convolution's
    /// channels at `channels` (each one's mean and 1/σ): the logits.
    fn run(&mut self, x: &[f32], channels: &[f32]) -> Result<Vec<f32>, String> {
        self.queue
            .write_buffer(&self.buffers[&Buf::Audio], 0, &bytes(x));
        self.queue
            .write_buffer(&self.buffers[&Buf::Stats], 0, &bytes(channels));
        let plan = &self.plans[&x.len()];
        let size = (plan.frames * TOKENS * 4) as u64;
        let (buffers, staging) = (&self.buffers, &self.staging);
        crate::pace::dispatch(
            &self.device,
            &self.queue,
            "jodel-gpu speech",
            &mut self.pieces,
            &self.pipelines,
            &self.bind_groups,
            &plan.ops,
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
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_piece_fits_the_frames() {
        let frames = conv_frames(longest_piece());
        assert_eq!(frames[6], Wav2Vec2::frames_of(longest_piece()));
        assert_eq!(frames[6], 1700);
    }
}
