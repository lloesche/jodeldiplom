//! The text encoder ([`TextEncoder`]): Qwen3-4B from Qwen's FP8 release
//! (its matrices FP8 with a scale per 128 × 128 block), its first 27
//! layers (the embeddings take the hidden states after layers 9, 18 and
//! 27). The prompt's 512 tokens go through in one plan: per layer RMSNorm,
//! the queries, keys and values (one matrix), the heads' norms and
//! rotary positions (halves, θ = 10⁶), causal attention of 32 query heads
//! on 8 key heads (the padding's keys left out, its queries kept: they are
//! part of the embeddings), the output added; RMSNorm, the gate and up as
//! SwiGLU's pairs, the down added. The token embeddings are looked up on
//! the processor (the table stays off the GPU).

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use jodel_tensors::half::bf16_to_f32;
use jodel_tensors::{DType, Shards};

use super::gpu::{Loaded, Made};
use super::plan::{At, Buf, Mat, Plan, Rope, Tail};
use super::text::{Tokenizer, TOKENS};
use super::weights::{Fp8, Packing, Source};
use crate::{Footprint, Precision, Resident};

const DIM: usize = 2560;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD: usize = 128;
const FF: usize = 9728;
const VOCAB: usize = 151_936;
const QKV: usize = (HEADS + 2 * KV_HEADS) * HEAD;
/// The layers run: the last hidden state taken is layer 27's.
const LAYERS: usize = 27;
/// The hidden states taken, after these layers (1-based), side by side.
const TAKEN: [usize; 3] = [9, 18, 27];
/// The embeddings' width.
pub const WIDTH: usize = 3 * DIM;
const EPS: f32 = 1e-6;
const THETA: f64 = 1e6;
/// The token embeddings' tensor.
const TABLE: &str = "model.embed_tokens.weight";
/// The checkpoint's index of its files.
const INDEX_FILE: &str = "model.safetensors.index.json";

/// The files [`TextEncoder::load`] opens in Qwen's FP8 release
/// (`Qwen/Qwen3-4B-FP8`): the tokenizer's, the index, then the weights'
/// (those the index names; another is refused).
pub const ENCODER_FILES: &[&str] = &[
    super::text::VOCAB_FILE,
    super::text::MERGES_FILE,
    super::text::CONFIG_FILE,
    INDEX_FILE,
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

/// A prompt's embeddings: 512 tokens of [`WIDTH`] (the hidden states after
/// layers 9, 18 and 27, side by side), the padding's too.
#[derive(Clone, Debug, PartialEq)]
pub struct Embeds {
    pub values: Vec<f32>,
}

/// One layer's weights.
struct Layer {
    input_norm: usize,
    post_norm: usize,
    q_norm: usize,
    k_norm: usize,
    qkv: Mat,
    o: Mat,
    gate_up: Mat,
    down: Mat,
}

/// The name of layer `i`'s tensor `n`.
fn layer(i: usize, n: &str) -> String {
    format!("model.layers.{i}.{n}")
}

/// The layers' weights from `src` into `pack`, and the rotary table's
/// place.
fn pack<'a>(src: Source<'a>, pack: &mut Packing<'a>) -> Result<(Vec<Layer>, usize), String> {
    let mut layers = Vec::with_capacity(LAYERS);
    for i in 0..LAYERS {
        let fp8 = |n: &str, shape| Fp8::per_block(src, &layer(i, n), shape);
        let q = fp8("self_attn.q_proj", (HEADS * HEAD, DIM))?;
        let k = fp8("self_attn.k_proj", (KV_HEADS * HEAD, DIM))?;
        let v = fp8("self_attn.v_proj", (KV_HEADS * HEAD, DIM))?;
        let o = fp8("self_attn.o_proj", (DIM, HEADS * HEAD))?;
        let gate = fp8("mlp.gate_proj", (FF, DIM))?;
        let up = fp8("mlp.up_proj", (FF, DIM))?;
        let down = fp8("mlp.down_proj", (DIM, FF))?;
        let vector =
            |pack: &mut Packing<'a>, n: &str, len| pack.bf16_vector(src, &layer(i, n), len);
        layers.push(Layer {
            input_norm: vector(pack, "input_layernorm.weight", DIM)?,
            post_norm: vector(pack, "post_attention_layernorm.weight", DIM)?,
            q_norm: vector(pack, "self_attn.q_norm.weight", HEAD)?,
            k_norm: vector(pack, "self_attn.k_norm.weight", HEAD)?,
            qkv: pack.fp8_stack(&[q, k, v]),
            o: pack.fp8(&o),
            gate_up: pack.fp8_pairs(&gate, &up),
            down: pack.fp8(&down),
        });
    }
    let rope = pack.vector(&rope_table());
    Ok((layers, rope))
}

/// Qwen3-4B's first 27 layers on the GPU.
pub struct TextEncoder {
    tokenizer: Tokenizer,
    /// The token embeddings (bf16 rows), on the processor.
    table: Vec<u8>,
    loaded: Loaded,
    layers: Vec<Layer>,
    rope: usize,
    precision: Precision,
    /// The plan for prompts of a length, made on the first (let go by
    /// [`Resident::release`]).
    made: Mutex<Option<(usize, Slots, Made)>>,
}

impl TextEncoder {
    /// What it takes on the GPU while loaded, and while it encodes, before
    /// it is loaded: its weights packed as they would be, nothing read.
    pub fn planned() -> Footprint {
        let mut packing = Packing::new();
        let (layers, rope) = pack(Source::Shapes, &mut packing).expect("shapes alone do not fail");
        Footprint {
            resident: packing.bytes(),
            working: plan(&layers, rope, false, TOKENS).0.bytes(),
        }
    }

    /// Qwen3-4B from Qwen's FP8 release in `dir` (its tokenizer too), on
    /// the GPU.
    pub fn load(dir: &Path, precision: Precision) -> Result<TextEncoder, String> {
        let tokenizer = Tokenizer::open(dir)?;
        let mut names = vec![TABLE.to_owned()];
        names.extend((0..LAYERS).map(|i| layer(i, "input_layernorm.weight")));
        let files = Shards::open(dir, INDEX_FILE, &names, ENCODER_FILES)?;
        let src = Source::Checkpoint(&files);
        let table = src
            .bytes(TABLE, DType::Bf16, &[VOCAB, DIM])?
            .expect("a checkpoint's bytes")
            .to_vec();
        let mut packing = Packing::new();
        let (layers, rope) = pack(src, &mut packing)?;
        let loaded = packing.upload("jodel-gpu text encoder")?;
        Ok(TextEncoder {
            tokenizer,
            table,
            loaded,
            layers,
            rope,
            precision,
            made: Mutex::new(None),
        })
    }

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.loaded.gpu.name
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is loaded instead.
    pub fn is_lost(&self) -> bool {
        self.loaded.lost()
    }

    /// What it holds on the GPU now: its weights, and its plan's buffers
    /// (made on the first prompt, kept until released).
    pub fn held(&self) -> Footprint {
        let made = self.made.lock().unwrap_or_else(PoisonError::into_inner);
        Footprint {
            resident: self.loaded.bytes(),
            working: made.as_ref().map_or(0, |(_, _, m)| m.bytes()),
        }
    }

    /// The bytes the GPU's allocator holds for every part now (where the
    /// backend tells).
    pub fn reserved(&self) -> Option<u64> {
        self.loaded.gpu.reserved()
    }

    /// The tokens of `prompt` as the encoder takes it, and how many are
    /// the prompt's.
    pub fn tokens(&self, prompt: &str) -> (Vec<u32>, usize) {
        self.tokenizer.prompt(prompt)
    }

    /// `prompt`'s embeddings.
    pub fn encode(&mut self, prompt: &str) -> Result<Embeds, String> {
        let (ids, real) = self.tokens(prompt);
        let x: Vec<f32> = ids
            .iter()
            .flat_map(|&id| {
                let row = &self.table[id as usize * DIM * 2..(id as usize + 1) * DIM * 2];
                row.chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            })
            .collect();
        // The plan depends on the prompt's length (the keys that count).
        let slot = self.made.get_mut().unwrap_or_else(PoisonError::into_inner);
        if slot.as_ref().is_none_or(|(n, _, _)| *n != real) {
            *slot = None;
            let fast = self.loaded.fast(self.precision);
            let (p, slots) = plan(&self.layers, self.rope, fast, real);
            *slot = Some((real, slots, Made::new(&self.loaded, &p, TOKENS * WIDTH)?));
        }
        let (_, slots, made) = slot.as_mut().expect("made");
        let values = made.run(
            &self.loaded,
            &[(slots.x, 0, &x)],
            (slots.out, 0, TOKENS * WIDTH),
        )?;
        Ok(Embeds { values })
    }
}

impl crate::OnDevice for TextEncoder {
    fn lost(&self) -> bool {
        self.is_lost()
    }
}

impl Resident for TextEncoder {
    /// Its weights, and its plan's buffers (every prompt's are as large).
    fn footprint(&self) -> Footprint {
        Footprint {
            resident: self.loaded.bytes(),
            working: plan(&self.layers, self.rope, false, TOKENS).0.bytes(),
        }
    }

    /// Lets the plan's buffers go (the weights stay); the next prompt
    /// makes them again.
    fn release(&self) {
        *self.made.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// The plan's slots: the hidden state, its norm, the queries' keys' and
/// values' rows, the heads, the attention's output (then SwiGLU's), the
/// embeddings.
struct Slots {
    x: u8,
    h: u8,
    rows: u8,
    heads: u8,
    mid: u8,
    out: u8,
}

/// The rotary table for 512 positions: per position, 64 pairs' cosine and
/// sine (θ = 10⁶ over halves of 128), the angles in f32 as the reference
/// makes them.
fn rope_table() -> Vec<f32> {
    let freqs = crate::engine::Freqs {
        base: THETA,
        dim: HEAD,
        turned: HEAD / 2,
    };
    crate::engine::rotary(0..TOKENS, freqs, crate::engine::Angles::F32)
}

/// The encoder's plan over 512 tokens, the first `valid` the prompt's (the
/// padding's keys left out), and its slots.
fn plan(layers: &[Layer], rope: usize, fast: bool, valid: usize) -> (Plan, Slots) {
    let mut p = Plan::new(fast);
    let n = TOKENS;
    let s = Slots {
        x: p.slot(n * DIM),
        h: p.slot(n * DIM),
        rows: p.slot(n * QKV),
        heads: p.slot(n * QKV),
        mid: p.slot(n * FF),
        out: p.slot(n * WIDTH),
    };
    let x = At::new(s.x, 0);
    let h = At::new(s.h, 0);
    let rows = At::new(s.rows, 0);
    let heads = At::new(s.heads, 0);
    let mid = At::new(s.mid, 0);
    let qkv = (
        heads,
        heads.plus(HEADS * n * HEAD),
        heads.plus((HEADS + KV_HEADS) * n * HEAD),
    );
    let table = Some((Rope::Halves, Buf::Small, rope));
    for (i, l) in layers.iter().enumerate() {
        p.rmsnorm(n, DIM, (x, DIM), (h, DIM), l.input_norm, EPS);
        p.gemm((h, DIM), n, &l.qkv, (rows, QKV), Tail::Store);
        p.heads(
            (n, HEADS, KV_HEADS),
            (rows, QKV),
            (qkv, n, 0),
            (l.q_norm, l.k_norm),
            table,
            EPS,
        );
        p.attention(
            (n, n, valid),
            (HEADS, KV_HEADS),
            qkv,
            (mid, HEADS * HEAD),
            true,
        );
        p.gemm((mid, HEADS * HEAD), n, &l.o, (x, DIM), Tail::Add);
        p.rmsnorm(n, DIM, (x, DIM), (h, DIM), l.post_norm, EPS);
        p.gemm((h, DIM), n, &l.gate_up, (mid, FF), Tail::SwiGlu);
        p.gemm((mid, FF), n, &l.down, (x, DIM), Tail::Add);
        if let Some(j) = TAKEN.iter().position(|&t| t == i + 1) {
            p.copy(n, DIM, (x, DIM), (At::new(s.out, j * DIM), WIDTH));
        }
    }
    (p, s)
}
