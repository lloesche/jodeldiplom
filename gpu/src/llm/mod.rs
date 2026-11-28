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
//! known before anything is loaded ([`Gemma::planned`]), and so is the
//! largest context a budget has room for ([`Gemma::fitting`]). A sliding
//! layer keeps the keys and values of its window only, in a ring (with a
//! chunk's rows: 1024 positions), so only the global layers' grow with the
//! context, each in a buffer of its own (32 KB a position in all; with
//! the rotary tables, 33.5 KB): up to the 128K positions the model was
//! trained for ([`MOST_CONTEXT`]), each buffer well within a binding.
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
//! eight made while those run. A conversation goes on from what its cache
//! holds ([`Gemma::go_on`]): the next turns run from where the last
//! answer ended, the prompt and the answer before not run again. With
//! [`Precision::Fast`] the matrix
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
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use self::kernels::{source, Kernel, Sums};
use self::model::Layout;
use self::plan::{Bind, ContextPlan, Plan, CHUNK};
pub use self::plan::{LEAST_CONTEXT, MOST_CONTEXT};
pub use self::tokens::{Stream, Take, Thinking, Tokenizer};
use crate::engine::{bytes, using, InFlight, Program, Read, Recorded};
use crate::pace::{Paces, Phase};
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
const fn is_global(l: usize) -> bool {
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
    /// The sliding layers' keys and values (a ring each), and the queries
    /// under way of their rows.
    Ring,
    /// A global layer's keys and values (the `n`th of those with their
    /// own), and the queries under way of its rows.
    Global(usize),
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

    /// The largest context the model in the GGUF at `path` takes within
    /// `room` bytes (its weights and the context's room at its most, as
    /// [`Gemma::planned`] says them), each of its buffers within what the
    /// process's GPU binds, no larger than `need` positions take: in whole
    /// steps of 2048 positions, from [`LEAST_CONTEXT`] (where none fits, or
    /// no room is told) to [`MOST_CONTEXT`] (only the file's header read);
    /// an error without a GPU.
    pub fn fitting(path: &Path, room: Option<u64>, need: usize) -> Result<usize, String> {
        let layout = Layout::of(&Gguf::open(path)?)?;
        let binding = crate::device::binding()?;
        let sizes = (room, binding);
        Ok(ContextPlan::fitting(&layout, layout.bytes(), sizes, need))
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
        let (inner, name) = crate::device::open_with(
            "jodel-gpu gemma",
            |adapter| {
                (
                    kernels::wanted(adapter),
                    wgpu::ExperimentalFeatures::disabled(),
                    (),
                )
            },
            |device, queue, limits, ()| {
                Inner::new(device, queue, limits, &mut file, layout, precision)
            },
        )?;
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
        if self.peek().shape.q8 {
            Precision::Fast
        } else {
            Precision::Full
        }
    }

    /// `work` on the model, a failure on its device an `Err` ([`using`]).
    fn using<R>(&self, work: impl FnOnce(&mut Inner) -> Result<R, String>) -> Result<R, String> {
        using(
            &self.inner,
            "the GPU language model failed before",
            |inner| &inner.program,
            work,
        )
    }

    /// The model for what it says of itself, even after a use failed.
    fn peek(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Room for `context` positions (a prompt and what it writes), made
    /// now (any room it had goes first): [`Gemma::planned`]'s working
    /// bytes, less a prompt's dispatches (recorded as it runs).
    pub fn reserve(&self, context: usize) -> Result<(), String> {
        let made = self.using(|inner| {
            inner.context = None;
            inner.reserve(context)
        });
        if made.is_err() {
            // No room, and none to make again: the next use asks for one.
            // Out here, as a failure on the device is told only once the
            // work is done (`guarded`), not inside it.
            self.peek().reserved = None;
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
    /// [`Tokenizer::chat`]), greedily, until a stop token ([`Tokenizer::stop`]),
    /// `most` of them, or `seen` breaks; `seen` is shown them as they are
    /// read (eight at a time), and those it was shown when it broke are
    /// the last kept (not [`Generated::stopped`]: they did not end by
    /// themselves). The context must have room for the prompt and `most`.
    pub fn generate(
        &self,
        prompt: &[u32],
        most: usize,
        seen: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> Result<Generated, String> {
        self.using(|inner| inner.generate(prompt, 0, (most, self.tokens.stop()), seen))
    }

    /// [`Gemma::generate`] going on in the conversation of the last
    /// generation (or of the last going on), which holds `after` tokens,
    /// its last `last` (a cheap check that it is that conversation):
    /// its prompt and the tokens it wrote stay in the cache as they are,
    /// and only `more` (the next turns as the chat template writes them, its
    /// end of the model's turn first: [`Tokenizer::next_turn`]) goes
    /// through the decoder, after the last token written (run again, as its
    /// keys and values may not have been made yet), so a turn after the
    /// first costs its own tokens alone. [`Generated::prompt_tokens`] are
    /// those run. An error, the conversation kept, where the context has no
    /// room for it and `most`; an error where there is no conversation of
    /// `after` tokens ending in `last` to go on (none written yet, another
    /// generation since, or its room let go or made again).
    pub fn go_on(
        &self,
        (after, last): (usize, u32),
        more: &[u32],
        most: usize,
        seen: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> Result<Generated, String> {
        self.using(|inner| {
            let context = room(&inner.context)?;
            let said = context.said.as_deref().unwrap_or_default();
            if said.len() != after || said.last() != Some(&last) {
                return Err(format!(
                    "no conversation of {after} tokens to go on (the cache holds {})",
                    said.len()
                ));
            }
            let room = context.plan.context;
            if after + more.len() + most > room {
                return Err(format!(
                    "{after} tokens, {} more and {most} to write in a context of {room}",
                    more.len()
                ));
            }
            let said = (inner.context.as_mut())
                .and_then(|c| c.said.take())
                .expect("checked");
            let from = said.len() - 1;
            let mut all = said;
            all.extend_from_slice(more);
            inner.generate(&all, from, (most, self.tokens.stop()), seen)
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
            ControlFlow::Continue(())
        })?;
        take(stream.finish());
        Ok(Text { text, generated })
    }

    /// The logits after `prompt` (capped, as the head writes them), for
    /// checks.
    pub fn logits(&self, prompt: &[u32]) -> Result<Vec<f32>, String> {
        self.using(|inner| {
            inner.ready()?;
            let in_flight = inner.prompt(prompt, 0, Wanted::Logits)?;
            inner.read(in_flight, Phase::Prompt)
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

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.name
    }
}

impl crate::OnDevice for Gemma {
    fn lost(&self) -> bool {
        self.peek().program.lost()
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

/// The room for a context: its plan, its buffers, the steps recorded (a
/// step's dispatches [`STEPS_A_READ`] times), the reads, and the
/// conversation its cache holds.
struct Context {
    plan: ContextPlan,
    buffers: HashMap<Buf, wgpu::Buffer>,
    steps: Recorded<Kernel, Bind>,
    staging: [wgpu::Buffer; 2],
    bytes: u64,
    /// The last generation's prompt and the tokens it wrote (its stop
    /// token not among them), in the cache from the first position: what
    /// [`Gemma::go_on`] goes on from. None before a generation, and once
    /// a prompt overwrote it.
    said: Option<Vec<u32>>,
}

struct Inner {
    program: Program<Kernel>,
    weights: HashMap<Buf, wgpu::Buffer>,
    weights_bytes: u64,
    layout: Layout,
    /// How it is shaped for its GPU.
    shape: Shape,
    context: Option<Context>,
    /// The context last reserved, and its room at its most (made again by
    /// a use after it was let go).
    reserved: Option<(usize, u64)>,
    paces: Paces,
}

/// What a prompt's run reads back: the token chosen, or the logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wanted {
    Token,
    Logits,
}

/// Whether [`Precision::Fast`] takes the rows at 8 bits on `backend`:
/// every one but Metal, where `dot4I8Packed` is emulated (the probe,
/// [`crate::probe`], measures it).
pub(crate) fn q8_on(backend: wgpu::Backend) -> bool {
    backend != wgpu::Backend::Metal
}

/// How the decoder is shaped for its GPU, chosen once ([`Shape::of`]).
/// `wgpu` tells nothing of a GPU's cores or how it fills them, so this is
/// chosen by backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    /// Whether the quantised matrices take their rows at 8 bits.
    pub(super) q8: bool,
    /// A global layer's chunks of a step's attention for one more split.
    pub(super) per: usize,
    /// How the f32 GEMVs add their sums.
    pub(super) sums: Sums,
}

impl Shape {
    /// The shape on the adapter `info` tells of, with the device
    /// `features`, reckoning as `precision` says: the rows at 8 bits where
    /// it is fast and `dot4I8Packed` is native ([`q8_on`]); a global layer's
    /// splits spread on Metal (Apple's GPUs take many workgroups to fill),
    /// elsewhere kept as they were so the plans come out token for token as
    /// before ([`plan::KEPT_CHUNKS_A_SPLIT`]); the f32 path's GEMV sums
    /// across subgroups where they are 32 wide and the device has them (the
    /// 8-bit path's through workgroup memory, its answers token for token
    /// as before).
    pub(super) fn of(
        info: &wgpu::AdapterInfo,
        features: wgpu::Features,
        precision: Precision,
    ) -> Shape {
        let q8 = precision == Precision::Fast && q8_on(info.backend);
        let subgroups =
            features.contains(wgpu::Features::SUBGROUP) && crate::device::subgroups_of_32(info);
        Shape {
            q8,
            per: if info.backend == wgpu::Backend::Metal {
                plan::SPREAD_CHUNKS_A_SPLIT
            } else {
                plan::KEPT_CHUNKS_A_SPLIT
            },
            sums: if subgroups && !q8 {
                Sums::Subgroup
            } else {
                Sums::Shared
            },
        }
    }
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
        let shape = Shape::of(&device.adapter_info(), device.features(), precision);
        let source = source(shape.sums);
        let program = Program::new(device, queue, limits, "jodel-gpu gemma", &source);
        let mut weights = HashMap::new();
        let mut upload = program.upload(layout.bytes());
        for (buf, words) in [(Buf::Small, layout.small), (Buf::Shared, layout.shared)]
            .into_iter()
            .chain((0..).zip(layout.groups()).map(|(g, w)| (Buf::Layers(g), w)))
        {
            let buffer = upload.storage(&format!("gemma {buf:?}"), words, true)?;
            weights.insert(buf, buffer);
        }
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
            shape,
            context: None,
            reserved: None,
            paces: Paces::resting(),
        })
    }

    /// A plan of dispatches for the context `plan`.
    fn plan<'a>(&'a self, plan: &ContextPlan) -> Plan<'a> {
        Plan {
            layout: &self.layout,
            sizes: plan.sizes,
            shape: self.shape,
        }
    }

    /// Makes room for `context` positions (and the steps run past a
    /// stop), the step recorded.
    fn reserve(&mut self, context: usize) -> Result<(), String> {
        let room = ContextPlan::new(&self.layout, context)?;
        // The context zeroed and the rotary tables put up, paced (the cache
        // is gigabytes on a large card).
        let rope = bytes(&room.sizes.rope());
        let buffers = (self.program).room(&room.sizes.buffers(), &[(Buf::Rope, &rope)])?;
        let mut steps = self.record(&buffers, &self.plan(&room).step())?;
        steps.repeat(STEPS_A_READ);
        let staging = [0, 1].map(|_| self.program.staging(plan::STAGING_WORDS));
        let bytes = room.held_bytes();
        debug_assert_eq!(
            bytes,
            buffers.values().map(wgpu::Buffer::size).sum::<u64>()
                + steps.params_bytes()
                + staging.iter().map(wgpu::Buffer::size).sum::<u64>(),
            "the room as planned"
        );
        self.reserved = Some((context, room.peak_bytes()));
        self.context = Some(Context {
            plan: room,
            buffers,
            steps,
            staging,
            bytes,
            said: None,
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
    ) -> Result<Recorded<Kernel, Bind>, String> {
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

    /// `prompt` through the decoder from its position `from` (those before
    /// it in the cache already, as they are), its first token chosen; it
    /// reads back what `read` says. The conversation the cache held is
    /// gone.
    fn prompt(&mut self, prompt: &[u32], from: usize, wanted: Wanted) -> Result<InFlight, String> {
        if let Some(context) = &mut self.context {
            context.said = None;
        }
        let context = room(&self.context)?;
        let n = prompt.len();
        if from >= n || n > context.plan.context {
            return Err(format!(
                "a prompt of {n} tokens from its {from}th in a context of {}",
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
        self.program.queue.write_buffer(ints, 0, &bytes(&state));
        let tokens = &bytes(&prompt[from..]);
        self.program
            .queue
            .write_buffer(ints, ((at + from) * 4) as u64, tokens);
        let plan = self.plan(&context.plan);
        let mut ops = Vec::new();
        for first in (from..n).step_by(CHUNK) {
            let rows = CHUNK.min(n - first);
            ops.extend(plan.chunk(first, rows, first + rows == n));
        }
        let recorded = self.record(&context.buffers, &ops)?;
        let (from, first, count) = match wanted {
            Wanted::Logits => (&context.buffers[&Buf::Logits], 0, VOCAB),
            Wanted::Token => (ints, at + n, 1),
        };
        let read = Read {
            from,
            first,
            count,
            staging: &context.staging[0],
        };
        let pieces = self.paces.of(Phase::Prompt);
        self.program.run_left(pieces, &recorded, Some(read))
    }

    /// [`STEPS_A_READ`] steps from the decoder's position `pos` (the
    /// tokens after it read back into the staging buffer `staging`).
    fn steps(&mut self, pos: usize, staging: usize) -> Result<InFlight, String> {
        let context = room(&self.context)?;
        let read = Read {
            from: &context.buffers[&Buf::Ints],
            first: kernels::TOKENS + pos + 1,
            count: STEPS_A_READ,
            staging: &context.staging[staging],
        };
        let pieces = self.paces.of(Phase::Steps);
        self.program.run_left(pieces, &context.steps, Some(read))
    }

    /// What `in_flight` (a run of `phase`) read, once it is done.
    fn read(&mut self, in_flight: InFlight, phase: Phase) -> Result<Vec<f32>, String> {
        self.program.finish(in_flight, self.paces.of(phase))
    }

    /// [`Gemma::generate`]'s work: at most `most` tokens, until one of
    /// `stop`, after `prompt` run from its position `from` (those before it
    /// in the cache already); `prompt` and the tokens written are the
    /// conversation the cache holds then.
    fn generate(
        &mut self,
        prompt: &[u32],
        from: usize,
        (most, stop): (usize, &[u32]),
        mut seen: impl FnMut(&[u32]) -> ControlFlow<()>,
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
        let first_run = self.prompt(prompt, from, Wanted::Token)?;
        let mut flying = VecDeque::from([(first_run, Phase::Prompt)]);
        // The decoder's position after the runs made so far.
        let mut pos = n;
        let mut out = Generated {
            prompt_tokens: n - from,
            ..Generated::default()
        };
        let mut first_read = None;
        let mut runs = 0;
        // Ended: on a stop token, at `most`, or by `seen`.
        let mut ended = false;
        loop {
            // The tokens the runs made so far write: the prompt's first,
            // then each step's.
            let made = 1 + (pos - n);
            if !ended && flying.len() < 2 && made < most {
                runs += 1;
                flying.push_back((self.steps(pos, runs % 2)?, Phase::Steps));
                pos += STEPS_A_READ;
                out.steps += STEPS_A_READ;
                continue;
            }
            let Some((in_flight, phase)) = flying.pop_front() else {
                break;
            };
            let got: Vec<u32> = self
                .read(in_flight, phase)?
                .iter()
                .map(|v| v.to_bits())
                .collect();
            first_read.get_or_insert_with(Instant::now);
            if ended {
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
            let broke = seen(&out.tokens[before..]).is_break();
            ended = out.stopped || out.tokens.len() == most || broke;
        }
        let first = first_read.unwrap_or(started);
        out.prompt_time = first - started;
        out.step_time = first.elapsed();
        if let Some(context) = &mut self.context {
            context.said = Some([prompt, &out.tokens].concat());
        }
        Ok(out)
    }
}

/// The room for a context, where there is one.
fn room(context: &Option<Context>) -> Result<&Context, String> {
    context
        .as_ref()
        .ok_or_else(|| "the language model has no room for a context".to_owned())
}
