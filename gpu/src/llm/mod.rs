//! Gemma 4 E4B-it's text decoder on the GPU ([`Gemma`]): the karaoke story
//! mode's language model, writing a song's storyboard and look from its
//! lyrics, greedily (the same prompt always gives the same answer).
//!
//! Its weights are read from Unsloth's GGUF of it (Q4_K_M: most matrices
//! in Q4_K, some in Q6_K, the per-layer embeddings in Q5_K, the per-layer
//! projection in BF16, its small matrices and norms in F32; `model.rs`)
//! and kept on the GPU as the file has them, 4.7 GB, the blocks unpacked
//! by the kernels that read them (`blocks.wgsl`); its tokeniser is the
//! file's too. Loading them ([`Gemma::open`]) and the room for a context
//! (the cache of keys and values and the working buffers,
//! [`Gemma::reserve`]) are separate, and the room goes when asked
//! ([`Resident::release`], or the model dropped); what each takes is
//! known before anything is loaded ([`Gemma::planned`]).
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
//! eight made while those run. With [`Precision::Fast`] the matrix
//! multiplies of quantised matrices take their rows at 8 bits where the
//! GPU multiplies bytes four at a time natively (`dot4I8Packed`; not
//! Metal, where it is emulated), else as f32s; those of BF16 and F32 ones
//! always as f32s.
//!
//! Text only for now: a picture (the cover's questions) would come in at
//! the prompt's embeddings, its soft tokens in the rows of its placeholders
//! (and the per-layer inputs taking the padding token's there), as the
//! reference's `Gemma4Model` does; the vision tower is a network of its own.

mod kernels;
mod model;
mod plan;
#[cfg(test)]
mod tests;
mod tokens;

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use self::kernels::{source, Kernel};
use self::model::Layout;
pub use self::plan::MOST_CONTEXT;
use self::plan::{Bind, ContextPlan, Plan, CHUNK};
pub use self::tokens::{Stream, Tokenizer};
use crate::engine::{self, bytes, guarded, Dispatch, Program, Recorded};
use crate::{Footprint, Precision, Resident, Turned};
use jodel_tensors::gguf::Gguf;

/// The width of the residual stream.
const DIM: usize = 2560;
const LAYERS: usize = 42;
/// The feed-forward's width.
const FF: usize = 10_240;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
/// The heads of sliding layers and of global ones.
const LOCAL_HEAD: usize = 256;
const GLOBAL_HEAD: usize = 512;
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
        GLOBAL_HEAD
    } else {
        LOCAL_HEAD
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

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Buf {
    /// The norms and scalars.
    Small,
    /// The embeddings and the per-layer projection.
    Shared,
    /// Layers' matrices ([`model::LAYERS_A_BUFFER`] layers a buffer).
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

/// What a generation wrote as text ([`Gemma::generate_text`]), and the
/// generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Text {
    pub text: String,
    pub generated: Generated,
}

/// Gemma 4 E4B-it's text decoder on the GPU (see the module's doc).
pub struct Gemma {
    name: String,
    tokens: Tokenizer,
    inner: Turned<Mutex<Inner>>,
}

impl Gemma {
    /// What the model in the GGUF at `path` takes with room for `context`
    /// positions (the prompt and what it writes), before it is loaded
    /// (only the file's header read): its weights as the file holds them,
    /// and the context's room at its most (a prompt of all of it under
    /// way).
    pub fn planned(path: &Path, context: usize) -> Result<Footprint, String> {
        let layout = Layout::of(&Gguf::open(path)?)?;
        let room = ContextPlan::new(&layout, context)?;
        Ok(Footprint {
            resident: layout.bytes(),
            working: room.peak_bytes(),
        })
    }

    /// The model in the GGUF at `path` (Unsloth's
    /// `gemma-4-E4B-it-Q4_K_M.gguf`) on the best GPU, reckoning as
    /// `precision` says: its weights read a tensor at a time and kept
    /// there, its tokeniser read from the file. Its context is not
    /// reserved yet ([`Gemma::reserve`]).
    pub fn open(path: &Path, precision: Precision) -> Result<Gemma, String> {
        let mut file = Gguf::open(path)?;
        let tokens = Tokenizer::from_gguf(&file)?;
        let layout = Layout::of(&file)?;
        let (inner, name) = crate::device::open("jodel-gpu gemma", |device, queue, limits| {
            Inner::new(device, queue, limits, &mut file, layout, precision)
        })?;
        Ok(Gemma {
            name,
            tokens,
            inner: Turned::new(Mutex::new(inner)),
        })
    }

    /// Its tokeniser.
    pub fn tokens(&self) -> &Tokenizer {
        &self.tokens
    }

    /// How it reckons on this GPU: [`Precision::Fast`] where its rows go at
    /// 8 bits.
    pub fn precision(&self) -> Precision {
        if self.peek().q8 {
            Precision::Fast
        } else {
            Precision::Full
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner>, String> {
        self.inner
            .lock()
            .map_err(|_| "the GPU language model failed before".to_owned())
    }

    /// The model for what it says of itself, even after a use failed.
    fn peek(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Room for `context` positions (a prompt and what it writes), made
    /// now (any room it had goes first): [`Gemma::planned`]'s working
    /// bytes, less a prompt's dispatches (recorded as it runs).
    pub fn reserve(&self, context: usize) -> Result<(), String> {
        let mut inner = self.lock()?;
        inner.context = None;
        let device = inner.program.device.clone();
        let made = guarded(&device, || inner.reserve(context));
        if made.is_err() {
            // No room, and none to make again: the next use asks for one.
            inner.reserved = None;
        }
        made
    }

    /// The bytes the GPU's allocator holds for it now (where the backend
    /// tells), as the picture models say theirs.
    pub fn reserved(&self) -> Option<u64> {
        self.peek()
            .program
            .device
            .generate_allocator_report()
            .map(|r| r.total_reserved_bytes)
    }

    /// The tokens it writes after `prompt` (its ids, the chat template's:
    /// [`Tokenizer::chat`]), greedily, until a stop token ([`Tokenizer::stop`]) or
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
        guarded(&device, || {
            inner.generate(prompt, (most, self.tokens.stop()), seen)
        })
    }

    /// [`Gemma::generate`]'s answer as text: `seen` is shown each piece as
    /// it is read (whole characters only: each token decoded once,
    /// [`Tokenizer::stream`]).
    pub fn generate_text(
        &self,
        prompt: &[u32],
        most: usize,
        mut seen: impl FnMut(&str),
    ) -> Result<Text, String> {
        let mut stream = self.tokens.stream();
        let mut text = String::new();
        let mut take = |piece: String| {
            if !piece.is_empty() {
                seen(&piece);
                text.push_str(&piece);
            }
        };
        let generated = self.generate(prompt, most, |tokens| {
            for &t in tokens {
                take(stream.push(t));
            }
        })?;
        take(stream.finish());
        Ok(Text { text, generated })
    }

    /// The logits after `prompt` (capped, as the head writes them), for
    /// checks.
    pub fn logits(&self, prompt: &[u32]) -> Result<Vec<f32>, String> {
        let mut inner = self.lock()?;
        let device = inner.program.device.clone();
        guarded(&device, || {
            inner.ready()?;
            let running = inner.prompt(prompt, Read::Logits)?;
            inner.read(running)
        })
    }
}

impl Gemma {
    /// What it holds on the GPU now: its weights, and the room for a
    /// context while it has one.
    pub fn held(&self) -> Footprint {
        let inner = self.peek();
        Footprint {
            resident: inner.weights_bytes,
            working: inner.context.as_ref().map_or(0, |c| c.bytes),
        }
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is opened instead.
    pub fn is_lost(&self) -> bool {
        self.peek().program.lost()
    }

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.name
    }
}

impl Resident for Gemma {
    /// Its weights, and the room for the context last reserved at its most
    /// (none before one is).
    fn footprint(&self) -> Footprint {
        let inner = self.peek();
        Footprint {
            resident: inner.weights_bytes,
            working: inner.reserved.map_or(0, |(_, bytes)| bytes),
        }
    }

    /// Lets the context's room go (the weights stay); the next generation
    /// makes it again, as large.
    fn release(&self) {
        self.peek().context = None;
    }
}

/// The room for a context: its plan, its buffers, the step recorded, the
/// reads.
struct Context {
    plan: ContextPlan,
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
    /// Whether its quantised matrices take their rows at 8 bits.
    q8: bool,
    context: Option<Context>,
    /// The context last reserved, and its room at its most (made again by
    /// a use after it was let go).
    reserved: Option<(usize, u64)>,
    prompt_pieces: crate::pace::Pieces,
    step_pieces: crate::pace::Pieces,
}

/// What a prompt's run reads back: the token chosen, or the logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    Token,
    Logits,
}

/// What a run was: a prompt's, or a batch of steps'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prompt,
    Steps,
}

/// A run made and maybe still running: its last submission (`None`:
/// done), the staging buffer it reads into, how many words, and what it
/// was.
struct Running {
    index: Option<wgpu::SubmissionIndex>,
    staging: usize,
    count: usize,
    phase: Phase,
}

/// Whether the matrices take their rows at 8 bits with `precision` on
/// `device`: where it is fast and `dot4I8Packed` is native (not Metal).
fn q8_here(device: &wgpu::Device, precision: Precision) -> bool {
    precision == Precision::Fast && device.adapter_info().backend != wgpu::Backend::Metal
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        file: &mut Gguf,
        layout: Layout,
        precision: Precision,
    ) -> Result<Inner, String> {
        let q8 = q8_here(&device, precision);
        let program = Program::new(device, queue, limits, "jodel-gpu gemma", &source());
        let mut weights = HashMap::new();
        for (buf, words) in [(Buf::Small, layout.small), (Buf::Shared, layout.shared)]
            .into_iter()
            .chain((0..).zip(layout.groups()).map(|(g, w)| (Buf::Layers(g), w)))
        {
            weights.insert(buf, program.storage(&format!("gemma {buf:?}"), words)?);
        }
        let mut upload = program.upload();
        model::load(file, &layout.pieces, |buf, at, data| {
            upload.write(&weights[&buf], (at * 4) as u64, data)
        })?;
        upload.finish()?;
        let weights_bytes = weights.values().map(wgpu::Buffer::size).sum();
        Ok(Inner {
            program,
            weights,
            weights_bytes,
            layout,
            q8,
            context: None,
            reserved: None,
            prompt_pieces: crate::pace::Pieces::resting(),
            step_pieces: crate::pace::Pieces::resting(),
        })
    }

    /// A plan of dispatches for the context `plan`.
    fn plan<'a>(&'a self, plan: &ContextPlan) -> Plan<'a> {
        Plan {
            layout: &self.layout,
            sizes: plan.sizes,
            q8: self.q8,
        }
    }

    /// Makes room for `context` positions (and the steps run past a
    /// stop), the step recorded.
    fn reserve(&mut self, context: usize) -> Result<(), String> {
        let room = ContextPlan::new(&self.layout, context)?;
        let buffers = self.program.buffers(&room.sizes.buffers())?;
        self.program
            .queue
            .write_buffer(&buffers[&Buf::Rope], 0, &bytes(&room.sizes.rope()));
        let step = self.record(&buffers, &self.plan(&room).step())?;
        let steps = (0..STEPS_A_READ)
            .flat_map(|_| step.dispatches.iter().copied())
            .collect();
        let staging = [0, 1].map(|_| self.program.staging(plan::STAGING_WORDS));
        let bytes = room.held_bytes();
        debug_assert_eq!(
            bytes,
            buffers.values().map(wgpu::Buffer::size).sum::<u64>()
                + step.dispatches.len() as u64 * engine::PARAM_STRIDE
                + staging.iter().map(wgpu::Buffer::size).sum::<u64>(),
            "the room as planned"
        );
        self.reserved = Some((context, room.peak_bytes()));
        self.context = Some(Context {
            plan: room,
            buffers,
            step,
            steps,
            staging,
            bytes,
        });
        Ok(())
    }

    /// The room for the context last reserved made again, where it was
    /// let go.
    fn ready(&mut self) -> Result<(), String> {
        match (&self.context, self.reserved) {
            (None, Some((context, _))) => self.reserve(context),
            _ => Ok(()),
        }
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
    /// back what `read` says.
    fn prompt(&mut self, prompt: &[u32], read: Read) -> Result<Running, String> {
        let context = room(&self.context)?;
        let n = prompt.len();
        if n == 0 || n > context.plan.context {
            return Err(format!(
                "a prompt of {n} tokens in a context of {}",
                context.plan.context
            ));
        }
        if prompt.iter().any(|&t| t as usize >= VOCAB) {
            return Err("a token past the vocabulary".into());
        }
        let ints = &context.buffers[&Buf::Ints];
        let at = kernels::TOKENS;
        let mut state = vec![0u32; at];
        state[0] = (n - 1) as u32;
        state.extend_from_slice(prompt);
        self.program.queue.write_buffer(ints, 0, &bytes(&state));
        let plan = self.plan(&context.plan);
        let mut ops = Vec::new();
        for first in (0..n).step_by(CHUNK) {
            let rows = CHUNK.min(n - first);
            ops.extend(plan.chunk(first, rows, first + rows == n));
        }
        let recorded = self.record(&context.buffers, &ops)?;
        let (from, first, count) = match read {
            Read::Logits => (&context.buffers[&Buf::Logits], 0, VOCAB),
            Read::Token => (ints, at + n, 1),
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
            phase: Phase::Prompt,
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
                kernels::TOKENS + pos + 1,
                STEPS_A_READ,
            )),
            &context.staging[staging],
            true,
        )?;
        Ok(Running {
            index,
            staging,
            count: STEPS_A_READ,
            phase: Phase::Steps,
        })
    }

    /// What `running` read, once it is done.
    fn read(&mut self, running: Running) -> Result<Vec<f32>, String> {
        if let Some(index) = running.index {
            let pieces = match running.phase {
                Phase::Prompt => &mut self.prompt_pieces,
                Phase::Steps => &mut self.step_pieces,
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

    /// [`Gemma::generate`]'s work: at most `most` tokens, until one of
    /// `stop`.
    fn generate(
        &mut self,
        prompt: &[u32],
        (most, stop): (usize, &[u32]),
        mut seen: impl FnMut(&[u32]),
    ) -> Result<Generated, String> {
        self.ready()?;
        let context = room(&self.context)?.plan.context;
        let n = prompt.len();
        if n + most > context {
            return Err(format!(
                "{n} tokens and {most} more in a context of {context}"
            ));
        }
        let started = Instant::now();
        let mut flying = VecDeque::from([self.prompt(prompt, Read::Token)?]);
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
                if stop.contains(&t) {
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
