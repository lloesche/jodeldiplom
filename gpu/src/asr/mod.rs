//! The speech recogniser of [`jodel_lyrics::asr`] (Qwen3-ASR-0.6B) on the
//! GPU, its large weights at 4 bits (`asr.wgsl` and `fused.wgsl` with
//! `kernels.wgsl`'s helpers): the encoder over the whole voice at once, then
//! the decoder with a cache of keys and values, a token a step, the next
//! one chosen on the GPU (the steps go a few at a time between reads).
//!
//! The voice is heard in parts ([`jodel_lyrics::asr::parts_of`], at most
//! [`MAX_SECONDS`]), each on its own: over minutes the decoder may loop or
//! stop early. Pieces of the same length (a voice's glimpses) are heard
//! together ([`GpuAsr::tokens_of`], [`batch`]): the encoder over all of
//! them, the decoder a row each, every row's sums as if it were heard
//! alone, so each piece gets the tokens it gets on its own. A step of the
//! decoder is five dispatches a layer ([`decoder`]). With
//! `JODEL_ASR_TIMES` set, each run's encoder and prompt time and its
//! tokens' time go to stderr.

mod batch;
mod calibration;
mod decoder;
mod encoder;
mod ops;
mod pack;
mod shape;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Mutex;

use jodel_lyrics::asr::packed::PackedModel;
use jodel_lyrics::asr::{
    mel, text, transcribe_parts, Transcript, CONV, DIM, ENC_DIM, HEAD, HEADS, KV_HEADS, VOCAB,
};

use self::calibration::Stats;
use self::ops::Op;
use self::pack::{pack, Layout};
use self::shape::{longest, Room, Shape};
use super::{guarded, spread, Turned, PARAM_SIZE, PARAM_STRIDE};

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
/// The most pieces heard together.
const MOST_AT_ONCE: usize = 16;
/// The tokens a piece heard with others has room for (with its first, from
/// the prompt, and reads of four steps, it may write 49); one that has not
/// ended by then is heard again on its own, where it may write
/// [`MAX_NEW`].
const NEW_AT_ONCE: usize = 48;
/// The most positions a piece's decoder attends over (`fused.wgsl` keeps
/// eight chunks of 128).
const MOST_POSITIONS: usize = 1024;
/// The most bytes of keys and values the pieces heard together keep.
const MOST_CACHE: u64 = 1 << 29;
/// The workgroup memory the kernels take at most: `qa_matmul`'s tiles of
/// 128 × 32 and 32 × 64 values (`qa_rows` takes 22.9 KB).
const WORKGROUP_MEMORY: u32 = (128 + 64) * 32 * 4;

const QKV_ENC: usize = 3 * ENC_DIM;
const QKV_DEC: usize = (HEADS + 2 * KV_HEADS) * HEAD;
const KV: usize = 2 * KV_HEADS * HEAD;
const PATCH: usize = 9 * CONV;
const BANDS: usize = 16;
/// A row's attention chunks (`qa_dec_attend`): 8 for each head, each its
/// largest score, its sum, and 128 values.
const PARTS: usize = HEADS * 8 * 130;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Rows,
    Conv1,
    Patches,
    Flatten,
    Place,
    LayerNorm,
    EncAttention,
    RmsNorm,
    HeadNormRope,
    Copy,
    DecAttention,
    DecAttend,
    SwiGlu,
    Embed,
    ArgmaxEmbed,
    ColSq,
}

impl Kernel {
    const ALL: [Kernel; 17] = [
        Kernel::Matmul,
        Kernel::Rows,
        Kernel::Conv1,
        Kernel::Patches,
        Kernel::Flatten,
        Kernel::Place,
        Kernel::LayerNorm,
        Kernel::EncAttention,
        Kernel::RmsNorm,
        Kernel::HeadNormRope,
        Kernel::Copy,
        Kernel::DecAttention,
        Kernel::DecAttend,
        Kernel::SwiGlu,
        Kernel::Embed,
        Kernel::ArgmaxEmbed,
        Kernel::ColSq,
    ];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "qa_matmul",
            Kernel::Rows => "qa_rows",
            Kernel::Conv1 => "qa_conv1",
            Kernel::Patches => "qa_patches",
            Kernel::Flatten => "qa_flatten",
            Kernel::Place => "qa_place",
            Kernel::LayerNorm => "qa_layernorm",
            Kernel::EncAttention => "qa_enc_attention",
            Kernel::RmsNorm => "qa_rmsnorm",
            Kernel::HeadNormRope => "qa_head_norm_rope",
            Kernel::Copy => "qa_copy",
            Kernel::DecAttention => "qa_dec_attention",
            Kernel::DecAttend => "qa_dec_attend",
            Kernel::SwiGlu => "qa_swiglu",
            Kernel::Embed => "qa_embed",
            Kernel::ArgmaxEmbed => "qa_argmax_embed",
            Kernel::ColSq => "qa_colsq",
        }
    }
}

/// The kernels' source: the shared helpers, the recogniser's kernels, the
/// fused ones.
fn source() -> String {
    crate::kernels_source(&[include_str!("asr.wgsl"), include_str!("fused.wgsl")])
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
    /// Each piece's keys and values, then the attention chunks' sums.
    Cache,
    Last,
    Logits,
    /// Each row's largest logit of each of the head's workgroups.
    Best,
    Tokens,
    /// Calibration: each matrix's input columns' sums of squares.
    Stats,
    /// The second buffer written of a kernel that writes one.
    Spare,
}

/// Which runs a run is, each kind paced on its own ([`crate::pace`]
/// learns how many dispatches make a submission of a few milliseconds:
/// the encoder's and the prompt's are heavier than a step's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Runs {
    Prompt,
    Steps,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    buffers: HashMap<Buf, wgpu::Buffer>,
    room: Room,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Where the model's tensors are.
    tensors: Layout,
    stats: Stats,
    /// The tokens a piece heard with others has room for ([`NEW_AT_ONCE`];
    /// fewer in tests).
    new_at_once: usize,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_index: HashMap<[Buf; 4], usize>,
    prompt_pieces: crate::pace::Pieces,
    step_pieces: crate::pace::Pieces,
}

/// Qwen3-ASR-0.6B on the GPU.
pub struct GpuAsr {
    name: String,
    vocab: text::Vocab,
    inner: Turned<Mutex<Inner>>,
}

impl GpuAsr {
    /// The packed model on the best GPU.
    pub fn new(model: &PackedModel) -> Result<GpuAsr, String> {
        let (inner, name) = crate::device::open("jodel-gpu asr", |device, queue, limits| {
            Inner::new(device, queue, limits, model)
        })?;
        Ok(GpuAsr {
            name,
            vocab: model.vocab.clone(),
            inner: Turned::new(Mutex::new(inner)),
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, String> {
        self.inner
            .lock()
            .map_err(|_| "the GPU speech recogniser failed before".to_owned())
    }

    /// What is sung in `voice` (16 kHz mono): its parts each heard on
    /// their own ([`jodel_lyrics::asr::transcribe_parts`]).
    pub fn transcribe(&self, voice: &[f32]) -> Result<Transcript, String> {
        transcribe_parts(voice, &self.vocab, |part| self.tokens(part))
    }

    /// What is sung in short glimpses of `voice` (16 kHz mono), each heard
    /// on its own ([`jodel_lyrics::glimpses`]) as one part of a transcript
    /// (its words, the language left out), but all together
    /// ([`GpuAsr::tokens_of`]).
    pub fn glimpses(&self, voice: &[f32]) -> Result<Vec<jodel_lyrics::asr::Part>, String> {
        jodel_lyrics::glimpses::hear_glimpses_at_once(voice, |pieces| {
            Ok(self
                .tokens_of(pieces)?
                .iter()
                .map(|ids| text::split_language(&self.vocab.decode(ids)).1)
                .collect())
        })
    }

    /// The encoder's output for one piece (`[steps][1024]`), for checks.
    pub fn audio(&self, voice: &[f32]) -> Result<Vec<f32>, String> {
        let (features, frames) = mel::features(voice, 8);
        if frames == 0 || frames > longest() {
            return Err(format!("{frames} frames"));
        }
        let mut inner = self.lock()?;
        let device = inner.device.clone();
        guarded(&device, || {
            let shape = Shape::new(1, frames);
            inner.upload(&[&features], &shape);
            let ops = inner.encoder_ops(&shape);
            inner.run(&ops, Some((Buf::Audio, 0, shape.steps * DIM)), Runs::Prompt)
        })
    }

    /// The tokens written for one part (at most [`MAX_SECONDS`]), a loop cut
    /// short ([`jodel_lyrics::asr::loop_start`]).
    pub fn tokens(&self, voice: &[f32]) -> Result<Vec<u32>, String> {
        Ok(self.tokens_of(&[voice])?.pop().unwrap_or_default())
    }

    /// The tokens of each of `pieces` (each at most [`MAX_SECONDS`]), as
    /// [`GpuAsr::tokens`] writes them, those of the same length heard
    /// together (as many at a time as the buffers hold, at most 16; one at
    /// a time while calibrating).
    pub fn tokens_of(&self, pieces: &[&[f32]]) -> Result<Vec<Vec<u32>>, String> {
        let features = mel::features_of(pieces);
        if let Some(&(_, frames)) = features.iter().find(|f| f.1 > longest()) {
            return Err(format!("{frames} frames, longer than a part"));
        }
        let mut out = vec![Vec::new(); pieces.len()];
        let mut by_length: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, &(_, frames)) in features.iter().enumerate() {
            if frames > 0 {
                by_length.entry(frames).or_default().push(i);
            }
        }
        for (frames, all) in by_length {
            let mut left: &[usize] = &all;
            while !left.is_empty() {
                let mut inner = self.lock()?;
                let n = if inner.stats.on {
                    1
                } else {
                    inner.room.at_once(frames).min(left.len())
                };
                let (now, rest) = left.split_at(n);
                left = rest;
                let device = inner.device.clone();
                let chosen: Vec<&[f32]> = now.iter().map(|&i| &features[i].0[..]).collect();
                let heard = guarded(&device, || inner.hear(&chosen, frames))?;
                for (&i, ids) in now.iter().zip(heard) {
                    out[i] = match ids {
                        Some(ids) => ids,
                        // Not done in the room the pieces heard together
                        // have: again, on its own.
                        None => guarded(&device, || inner.hear(&[&features[i].0], frames))?
                            .pop()
                            .flatten()
                            .unwrap_or_default(),
                    };
                }
            }
        }
        Ok(out)
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        model: &PackedModel,
    ) -> Result<Inner, String> {
        let alone = Shape::new(1, longest());
        if alone.positions > MOST_POSITIONS {
            return Err(format!("{} positions", alone.positions));
        }
        if limits.max_compute_workgroup_storage_size < WORKGROUP_MEMORY {
            return Err(format!(
                "{} bytes of workgroup memory, {WORKGROUP_MEMORY} needed",
                limits.max_compute_workgroup_storage_size
            ));
        }
        let packed = pack(model, alone.positions)?;
        let stats = Stats::new(&packed.matrices);
        let most_binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let room = Room::new(MOST_AT_ONCE, MOST_CACHE.min(most_binding), stats.len);
        let mut all: Vec<(Buf, usize)> = room.buffers().collect();
        all.push((Buf::Small, packed.small.len()));
        all.push((Buf::Stats, stats.len));
        all.push((Buf::Spare, 1));
        all.push((Buf::Weights, packed.weights.len()));
        for &(buf, len) in &all {
            let size = (len * 4) as u64;
            if size > most_binding {
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
            size: (room.staging * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (layout, pipelines) = pipelines(&device);
        Ok(Inner {
            device,
            queue,
            pipelines,
            layout,
            buffers,
            room,
            params,
            staging,
            tensors: packed.layout,
            stats,
            new_at_once: NEW_AT_ONCE,
            bind_groups: Vec::new(),
            bind_index: HashMap::new(),
            prompt_pieces: crate::pace::Pieces::resting(),
            step_pieces: crate::pace::Pieces::resting(),
        })
    }

    /// The bind group of `buffers` (and the weights, and the parameters).
    fn bind_group(&mut self, buffers: [Buf; 4]) -> usize {
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
                entry(5, buffers[3]),
            ],
        });
        self.bind_groups.push(group);
        self.bind_index.insert(buffers, self.bind_groups.len() - 1);
        self.bind_groups.len() - 1
    }

    /// Runs `ops` (their parameters written first), paced as `runs`;
    /// `read` copies `count` f32s of a buffer from an offset back.
    fn run(
        &mut self,
        ops: &[Op],
        read: Option<(Buf, usize, usize)>,
        runs: Runs,
    ) -> Result<Vec<f32>, String> {
        let counted;
        let ops = if self.stats.on {
            counted = self.stats.counted(ops);
            &counted[..]
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
        let pieces = match runs {
            Runs::Prompt => &mut self.prompt_pieces,
            Runs::Steps => &mut self.step_pieces,
        };
        let (buffers, staging) = (&self.buffers, &self.staging);
        crate::pace::dispatch(
            &self.device,
            &self.queue,
            "jodel-gpu asr",
            pieces,
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
}

/// The bind group layout every kernel takes (src, aux, dst, the
/// parameters, the weights, a second dst) and each kernel's pipeline.
fn pipelines(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    HashMap<Kernel, wgpu::ComputePipeline>,
) {
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
            storage(5, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("jodel-gpu asr"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jodel-gpu asr kernels"),
        source: wgpu::ShaderSource::Wgsl(source().into()),
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
    (layout, pipelines)
}
