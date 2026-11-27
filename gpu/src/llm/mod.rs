//! Gemma 4 E4B-it's text decoder on the GPU ([`Gemma`]): the karaoke story
//! mode's language model, writing a song's storyboard and look from its
//! lyrics, greedily (the same prompt always gives the same answer).
//!
//! Its weights are packed once from the checkpoint ([`pack()`], `pack.rs`:
//! every matrix at 4 bits, in groups of 32 with a fitted scale and offset)
//! and kept packed on the GPU, 4.4 GB. Loading them ([`Gemma::open`]) and
//! the room for a context (the cache of keys and values and the working
//! buffers, [`Gemma::reserve`]) are separate, and either goes when asked
//! ([`Gemma::release`], or the model dropped); what each takes is known
//! before anything is loaded ([`Memory::of`]).
//!
//! The architecture is exactly the reference's (Hugging Face's
//! `Gemma4TextModel`): 42 layers of width 2560, five sliding-window layers
//! (a window of 512, heads of 256 turned at base 10⁴) to each global one
//! (heads of 512, a quarter of each head turned at base 10⁶); 8 query heads
//! on 2 key/value heads, the queries and keys RMS-normed per head (with
//! weights; the values without), no scaling of the scores; the last 18
//! layers using the keys and values of the last earlier layer of their kind;
//! norms after attention and after the feed-forward as well as before
//! (GELU-gated, 10 240 wide); per-layer inputs (each token's own
//! embedding of 256 for each layer, plus its embedding's projection,
//! normed) gating a residual of their own in each layer; each layer's
//! output scaled; the embeddings tied to the head, its logits capped at 30.
//!
//! A prompt goes in chunks of up to 512 rows ([`plan`]): the 24 layers with
//! keys and values of their own over all the rows (the tiled matrix
//! multiplies, flash attention), the rest for its last row only. Then a
//! token a step, the step's dispatches recorded once: each kernel reads the
//! position from the GPU, and the head's largest logit is the next token
//! there, put after it. Steps go eight at a time between reads, the next
//! eight made while those run. The matrix multiplies take their rows at 8
//! bits where the GPU multiplies bytes four at a time (`dot4I8Packed`),
//! as f32s where that would be emulated (Metal): [`Precision`].
//!
//! Text only for now: a picture (the cover's questions) would come in at
//! the prompt's embeddings, its soft tokens in the rows of its placeholders
//! (and the per-layer inputs taking the padding token's there), as the
//! reference's `Gemma4Model` does; the vision tower is a network of its own.

mod pack;
mod plan;
#[cfg(test)]
mod tests;
mod tokens;

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub use self::pack::pack;
use self::pack::{Layout, Reader};
pub use self::plan::Precision;
use self::plan::{Bind, Plan, Sizes, CHUNK, MOST_POSITIONS};
pub use self::tokens::{Tokenizer, STOP};
use crate::engine::{bytes, guarded, Access, Dispatch, Kernels, Program, Recorded};
use crate::Turned;

/// The width of the residual stream.
const DIM: usize = 2560;
const LAYERS: usize = 42;
/// The feed-forward's width.
const FF: usize = 10_240;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
/// A layer's per-layer input.
const PER_LAYER: usize = 256;
/// Every layer's.
const PER_LAYERS: usize = LAYERS * PER_LAYER;
const VOCAB: usize = 262_144;
/// A sliding layer's window: its own position and the 511 before.
const WINDOW: usize = 512;
/// The layers with keys and values of their own (the first 24); the rest
/// use those of the last of these of their kind.
const OWN_KV: usize = 24;
/// The rotary bases of sliding and global layers, and the pairs a global
/// head turns (a quarter of its 512 values: the other pairs stay).
const LOCAL_BASE: f32 = 10_000.0;
const GLOBAL_BASE: f32 = 1_000_000.0;
const GLOBAL_TURNED: usize = 64;
/// The embeddings' scale: √2560 as the reference has it, rounded to
/// bfloat16.
const EMBED_SCALE: f32 = 50.5;
/// The per-layer embeddings' scale (√256) and the projection's (1/√2560).
const PER_LAYER_EMBED_SCALE: f32 = 16.0;
const PER_LAYER_PROJECTION_SCALE: f32 = 0.019_764_235;
/// Steps between reads of the tokens.
const STEPS_A_READ: usize = 8;

/// Whether layer `l` is a global one (every sixth).
fn is_global(l: usize) -> bool {
    l % 6 == 5
}

/// The width of layer `l`'s heads.
fn head_of(l: usize) -> usize {
    if is_global(l) {
        512
    } else {
        256
    }
}

/// Whether layer `l` uses an earlier layer's keys and values.
fn shares(l: usize) -> bool {
    l >= OWN_KV
}

/// The layer whose keys and values layer `l` uses: its own, or the last
/// with its own of its kind.
fn kv_of(l: usize) -> usize {
    if !shares(l) {
        return l;
    }
    (0..OWN_KV)
        .rev()
        .find(|&e| is_global(e) == is_global(l))
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Embed,
    Quant,
    Residual,
    PerLayer,
    Argmax,
    GemvQ8,
    GemvF32,
    GemmQ8,
    GemmF32,
    Prep,
    Attend,
    Join,
    Flash,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Embed,
        Kernel::Quant,
        Kernel::Residual,
        Kernel::PerLayer,
        Kernel::Argmax,
        Kernel::GemvQ8,
        Kernel::GemvF32,
        Kernel::GemmQ8,
        Kernel::GemmF32,
        Kernel::Prep,
        Kernel::Attend,
        Kernel::Join,
        Kernel::Flash,
    ];
    const PARAM_BYTES: u64 = 64;
    /// After the parameters: the 4-bit weights (read), the ints.
    const EXTRA: &'static [Access] = &[Access::Read, Access::Write];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Embed => "embed",
            Kernel::Quant => "quant",
            Kernel::Residual => "residual",
            Kernel::PerLayer => "per_layer",
            Kernel::Argmax => "argmax",
            Kernel::GemvQ8 => "gemv_q8",
            Kernel::GemvF32 => "gemv_f32",
            Kernel::GemmQ8 => "gemm_q8",
            Kernel::GemmF32 => "gemm_f32",
            Kernel::Prep => "prep",
            Kernel::Attend => "attend",
            Kernel::Join => "join",
            Kernel::Flash => "flash",
        }
    }
}

/// The kernels' source.
fn source() -> String {
    [
        include_str!("prelude.wgsl"),
        include_str!("matmul.wgsl"),
        include_str!("attention.wgsl"),
        include_str!("rows.wgsl"),
    ]
    .join("\n")
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    /// The norms and scalars.
    Small,
    /// The embeddings and the per-layer projection.
    Shared,
    /// Layers' matrices ([`pack::LAYERS_A_BUFFER`] layers a buffer).
    Layers(usize),
    /// The rotary tables.
    Rope,
    /// The residual stream's rows.
    X,
    /// A matrix multiply's products.
    H,
    /// The embeddings' per-layer projection.
    PerLayer,
    /// The keys and values, and the queries under way.
    Cache,
    /// A step's attention chunks.
    Parts,
    Logits,
    Ints,
    /// Bound where a kernel writes nothing to dst.
    Spare,
}

/// What a model takes on the GPU, in bytes: its weights (while it is
/// loaded), and the room for a context (while it is reserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Memory {
    pub weights: u64,
    pub context: u64,
}

impl Memory {
    /// What the model takes with room for `context` positions (the prompt
    /// and what it writes): the cache of keys and values, the working
    /// buffers of a prompt's chunk and a step, the recorded step's numbers
    /// and the reads.
    pub fn of(context: usize) -> Memory {
        let layout = Layout::new();
        let sizes = Sizes::new(context + 2 * STEPS_A_READ);
        let plan = Plan {
            layout: &layout,
            sizes,
            precision: Precision::Q8,
        };
        let words: usize = sizes.buffers().iter().map(|b| b.1).sum();
        let step = plan.step().len();
        let chunks = context.div_ceil(CHUNK);
        let prompt = chunks * plan.chunk(0, CHUNK, false).len() + step;
        Memory {
            weights: layout.bytes(),
            context: words as u64 * 4
                + ((step + prompt) as u64 * crate::engine::PARAM_STRIDE)
                + 2 * staging_bytes(),
        }
    }

    /// Both.
    pub fn total(&self) -> u64 {
        self.weights + self.context
    }
}

/// The bytes of a staging buffer (the logits, or a read's tokens).
fn staging_bytes() -> u64 {
    (VOCAB.max(STEPS_A_READ) * 4) as u64
}

/// What a generation did: the tokens it wrote (its stop token not among
/// them), the prompt's tokens and time (to its first token), the steps
/// and their time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generated {
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub prompt_time: Duration,
    pub steps: usize,
    pub step_time: Duration,
    /// It ended on a stop token (else at its limit).
    pub stopped: bool,
}

/// Gemma 4 E4B-it's text decoder on the GPU (see the module's doc).
pub struct Gemma {
    name: String,
    tokens: Tokenizer,
    inner: Turned<Mutex<Inner>>,
}

impl Gemma {
    /// The packed model at `path` ([`pack()`]'s) on the best GPU: its
    /// weights read a buffer at a time and kept there. Its context is not
    /// reserved yet ([`Gemma::reserve`]).
    pub fn open(path: &Path) -> Result<Gemma, String> {
        let mut reader = Reader::open(path)?;
        let (inner, name) = crate::device::open("jodel-gpu gemma", |device, queue, limits| {
            Inner::new(device, queue, limits, &mut reader)
        })?;
        Ok(Gemma {
            name,
            tokens: reader.tokens,
            inner: Turned::new(Mutex::new(inner)),
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its tokeniser.
    pub fn tokens(&self) -> &Tokenizer {
        &self.tokens
    }

    /// How its matrices take their rows on this GPU.
    pub fn precision(&self) -> Result<Precision, String> {
        Ok(self.lock()?.precision)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, String> {
        self.inner
            .lock()
            .map_err(|_| "the GPU language model failed before".to_owned())
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is opened instead.
    pub fn is_lost(&self) -> bool {
        self.lock().map_or(true, |inner| inner.program.lost())
    }

    /// Room for `context` positions (a prompt and what it writes), made
    /// now (any room it had goes first): [`Memory::of`]`(context)`'s
    /// context bytes.
    pub fn reserve(&self, context: usize) -> Result<(), String> {
        let mut inner = self.lock()?;
        inner.context = None;
        let device = inner.program.device.clone();
        guarded(&device, || inner.reserve(context))
    }

    /// Lets the context's room go (the weights stay).
    pub fn release(&self) -> Result<(), String> {
        self.lock()?.context = None;
        Ok(())
    }

    /// What it holds on the GPU now.
    pub fn memory(&self) -> Result<Memory, String> {
        let inner = self.lock()?;
        Ok(Memory {
            weights: inner.weights_bytes,
            context: inner.context.as_ref().map_or(0, |c| c.bytes),
        })
    }

    /// The tokens it writes after `prompt` (its ids, the chat template's:
    /// [`Tokenizer::chat`]), greedily, until a stop token ([`STOP`]) or
    /// `most` of them; `seen` is shown them as they are read (eight at a
    /// time). The context must have room for the prompt and `most`.
    pub fn generate(
        &self,
        prompt: &[u32],
        most: usize,
        seen: impl FnMut(&[u32]),
    ) -> Result<Generated, String> {
        let mut inner = self.lock()?;
        let device = inner.program.device.clone();
        guarded(&device, || inner.generate(prompt, most, seen))
    }

    /// The logits after `prompt` (capped, as the head writes them), for
    /// checks.
    pub fn logits(&self, prompt: &[u32]) -> Result<Vec<f32>, String> {
        let mut inner = self.lock()?;
        let device = inner.program.device.clone();
        guarded(&device, || {
            let running = inner.prompt(prompt, Buf::Logits)?;
            inner.read(running)
        })
    }
}

/// The room for a context: its buffers, the step recorded, the reads.
struct Context {
    sizes: Sizes,
    buffers: HashMap<Buf, wgpu::Buffer>,
    step: Recorded<Kernel>,
    /// The step's dispatches [`STEPS_A_READ`] times.
    steps: Vec<Dispatch<Kernel>>,
    staging: [wgpu::Buffer; 2],
    bytes: u64,
}

struct Inner {
    program: Program<Kernel>,
    weights: HashMap<Buf, wgpu::Buffer>,
    weights_bytes: u64,
    layout: Layout,
    precision: Precision,
    context: Option<Context>,
    prompt_pieces: crate::pace::Pieces,
    step_pieces: crate::pace::Pieces,
}

/// A run made and maybe still running: its last submission (`None`:
/// done), the staging buffer it reads into, how many words, and whether
/// it was a prompt's.
struct Running {
    index: Option<wgpu::SubmissionIndex>,
    staging: usize,
    count: usize,
    prompt: bool,
}

/// How the matrices take their rows on the chosen GPU: as f32s on Metal
/// (whose `dot4I8Packed` is emulated), else at 8 bits;
/// `JODEL_GEMMA_PRECISION` (`q8` or `plain`) says otherwise.
fn precision_here() -> Precision {
    match std::env::var("JODEL_GEMMA_PRECISION").as_deref() {
        Ok("plain") => return Precision::Plain,
        Ok("q8") => return Precision::Q8,
        _ => {}
    }
    let backend = crate::chosen()
        .adapter
        .as_ref()
        .map(|a| a.get_info().backend);
    if backend == Ok(wgpu::Backend::Metal) {
        Precision::Plain
    } else {
        Precision::Q8
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        reader: &mut Reader,
    ) -> Result<Inner, String> {
        let layout = Layout::new();
        let program = Program::new(device, queue, limits, "jodel-gpu gemma", &source());
        let mut weights = HashMap::new();
        // Each section's copy done before the next is read: the queue's
        // staging of it let go.
        let put = |b: &wgpu::Buffer, at: usize, data: &[u8]| {
            program.queue.write_buffer(b, (at * 4) as u64, data);
            program.queue.submit([]);
            program.wait()
        };
        for (buf, words) in [(Buf::Small, layout.small), (Buf::Shared, layout.shared)]
            .into_iter()
            .chain((0..).zip(layout.groups()).map(|(g, w)| (Buf::Layers(g), w)))
        {
            weights.insert(buf, program.storage(&format!("gemma {buf:?}"), words)?);
        }
        put(&weights[&Buf::Small], 0, &reader.small)?;
        put(&weights[&Buf::Shared], 0, &reader.next(layout.shared)?)?;
        for layer in &layout.layers {
            let b = &weights[&Buf::Layers(layer.group)];
            put(b, layer.base, &reader.next(layer.words)?)?;
        }
        let weights_bytes = weights.values().map(wgpu::Buffer::size).sum();
        Ok(Inner {
            program,
            weights,
            weights_bytes,
            layout,
            precision: precision_here(),
            context: None,
            prompt_pieces: crate::pace::Pieces::resting(),
            step_pieces: crate::pace::Pieces::resting(),
        })
    }

    /// Makes room for `context` positions (and the steps run past a
    /// stop), the step recorded.
    fn reserve(&mut self, context: usize) -> Result<(), String> {
        let positions = context + 2 * STEPS_A_READ;
        if positions > MOST_POSITIONS {
            return Err(format!(
                "a context of {context}: at most {} positions",
                MOST_POSITIONS - 2 * STEPS_A_READ
            ));
        }
        let sizes = Sizes::new(positions);
        let buffers = self.program.buffers(&sizes.buffers())?;
        self.program
            .queue
            .write_buffer(&buffers[&Buf::Rope], 0, &bytes(&sizes.rope()));
        let plan = Plan {
            layout: &self.layout,
            sizes,
            precision: self.precision,
        };
        let step = self.record(&buffers, &plan.step())?;
        let steps = (0..STEPS_A_READ)
            .flat_map(|_| step.dispatches.iter().copied())
            .collect();
        let staging = [
            self.program.staging(VOCAB.max(STEPS_A_READ)),
            self.program.staging(VOCAB.max(STEPS_A_READ)),
        ];
        let bytes = buffers.values().map(wgpu::Buffer::size).sum::<u64>()
            + step.dispatches.len() as u64 * crate::engine::PARAM_STRIDE
            + 2 * staging_bytes();
        self.context = Some(Context {
            sizes,
            buffers,
            step,
            steps,
            staging,
            bytes,
        });
        Ok(())
    }

    /// `ops` recorded on the weights and `buffers`.
    fn record(
        &self,
        buffers: &HashMap<Buf, wgpu::Buffer>,
        ops: &[plan::Op],
    ) -> Result<Recorded<Kernel>, String> {
        let get = |b: Buf| self.weights.get(&b).unwrap_or_else(|| &buffers[&b]);
        self.program.record(ops, |bind: Bind| {
            vec![
                get(bind.src),
                get(Buf::Small),
                get(bind.dst),
                get(bind.wq),
                get(Buf::Ints),
            ]
        })
    }

    /// `prompt` through the decoder, its first token chosen; it reads
    /// that token back (or the logits, with `read` `Logits`).
    fn prompt(&mut self, prompt: &[u32], read: Buf) -> Result<Running, String> {
        let context = room(&self.context)?;
        let positions = context.sizes.positions;
        let n = prompt.len();
        if n == 0 || n + 2 * STEPS_A_READ > positions {
            return Err(format!(
                "a prompt of {n} tokens in a context of {}",
                positions - 2 * STEPS_A_READ
            ));
        }
        if prompt.iter().any(|&t| t as usize >= VOCAB) {
            return Err("a token past the vocabulary".into());
        }
        let ints = &context.buffers[&Buf::Ints];
        let at = plan::TOKENS;
        let mut state = vec![0u32; at];
        state[0] = (n - 1) as u32;
        state.extend_from_slice(prompt);
        self.program.queue.write_buffer(ints, 0, &bytes(&state));
        let plan = Plan {
            layout: &self.layout,
            sizes: context.sizes,
            precision: self.precision,
        };
        let mut ops = Vec::new();
        for first in (0..n).step_by(CHUNK) {
            let rows = CHUNK.min(n - first);
            ops.extend(plan.chunk(first, rows, first + rows == n));
        }
        let recorded = self.record(&context.buffers, &ops)?;
        let (from, first, count) = match read {
            Buf::Logits => (&context.buffers[&Buf::Logits], 0, VOCAB),
            _ => (ints, at + n, 1),
        };
        let index = self.program.submit(
            &mut self.prompt_pieces,
            (&recorded.groups, &recorded.dispatches),
            |_| {},
            Some((from, first, count)),
            &context.staging[0],
            true,
        )?;
        Ok(Running {
            index,
            staging: 0,
            count,
            prompt: true,
        })
    }

    /// [`STEPS_A_READ`] steps from the decoder's position `pos` (the
    /// tokens after it read back into the staging buffer `staging`).
    fn steps(&mut self, pos: usize, staging: usize) -> Result<Running, String> {
        let context = room(&self.context)?;
        let index = self.program.submit(
            &mut self.step_pieces,
            (&context.step.groups, &context.steps),
            |_| {},
            Some((
                &context.buffers[&Buf::Ints],
                plan::TOKENS + pos + 1,
                STEPS_A_READ,
            )),
            &context.staging[staging],
            true,
        )?;
        Ok(Running {
            index,
            staging,
            count: STEPS_A_READ,
            prompt: false,
        })
    }

    /// What `running` read, once it is done.
    fn read(&mut self, running: Running) -> Result<Vec<f32>, String> {
        if let Some(index) = running.index {
            let pieces = if running.prompt {
                &mut self.prompt_pieces
            } else {
                &mut self.step_pieces
            };
            crate::pace::finish(pieces, &self.program.device, index)?;
        }
        let context = room(&self.context)?;
        crate::device::read_back(
            &self.program.device,
            &context.staging[running.staging],
            running.count,
        )
    }

    /// [`Gemma::generate`]'s work.
    fn generate(
        &mut self,
        prompt: &[u32],
        most: usize,
        mut seen: impl FnMut(&[u32]),
    ) -> Result<Generated, String> {
        let positions = room(&self.context)?.sizes.positions;
        let n = prompt.len();
        if n + most + 2 * STEPS_A_READ > positions {
            return Err(format!(
                "{n} tokens and {most} more in a context of {}",
                positions - 2 * STEPS_A_READ
            ));
        }
        let started = Instant::now();
        let mut flying = VecDeque::from([self.prompt(prompt, Buf::Ints)?]);
        // The decoder's position after the runs made so far.
        let mut pos = n;
        let mut out = Generated {
            prompt_tokens: n,
            ..Generated::default()
        };
        let mut first_read = None;
        let mut runs = 0;
        loop {
            // The tokens the runs made so far write: the prompt's first,
            // then each step's.
            let made = 1 + (pos - n);
            if !out.stopped && flying.len() < 2 && made < most {
                runs += 1;
                flying.push_back(self.steps(pos, runs % 2)?);
                pos += STEPS_A_READ;
                out.steps += STEPS_A_READ;
                continue;
            }
            let Some(running) = flying.pop_front() else {
                break;
            };
            let got: Vec<u32> = self.read(running)?.iter().map(|v| v.to_bits()).collect();
            first_read.get_or_insert_with(Instant::now);
            if out.stopped || out.tokens.len() >= most {
                continue;
            }
            let before = out.tokens.len();
            for t in got {
                if STOP.contains(&t) {
                    out.stopped = true;
                    break;
                }
                out.tokens.push(t);
                if out.tokens.len() == most {
                    break;
                }
            }
            seen(&out.tokens[before..]);
        }
        let first = first_read.unwrap_or(started);
        out.prompt_time = first - started;
        out.step_time = first.elapsed();
        Ok(out)
    }
}

/// The room for a context, where there is one.
fn room(context: &Option<Context>) -> Result<&Context, String> {
    context
        .as_ref()
        .ok_or_else(|| "the language model has no room for a context".to_owned())
}
