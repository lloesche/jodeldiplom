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

use std::path::{Path, PathBuf};

use jodel_tensors::half::bf16_to_f32;
use jodel_tensors::json::Json;
use jodel_tensors::SafeTensors;

use super::gpu::{Loaded, Made};
use super::plan::{At, Buf, Mat, Plan, Rope, Tail};
use super::text::{Tokenizer, TOKENS};
use super::weights::{bf16_bytes, Fp8, Packing};
use super::{Footprint, Precision};

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
const THETA: f32 = 1e6;

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

/// Qwen3-4B's first 27 layers on the GPU.
pub struct TextEncoder {
    tokenizer: Tokenizer,
    /// The token embeddings (bf16 rows), on the processor.
    table: Vec<u8>,
    loaded: Loaded,
    layers: Vec<Layer>,
    rope: usize,
    precision: Precision,
    /// The plan for prompts of a length, made on the first.
    made: Option<(usize, Made)>,
}

/// The checkpoint's files that hold `names`, opened (from its index).
fn shards(dir: &Path, names: &[String]) -> Result<Vec<SafeTensors>, String> {
    let index = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
    let map = Json::parse(&text)
        .and_then(|j| j.get("weight_map").cloned())
        .ok_or("no weight map in the index")?;
    let mut files: Vec<PathBuf> = names
        .iter()
        .map(|n| {
            map.get(n)
                .and_then(Json::as_str)
                .map(|f| dir.join(f))
                .ok_or_else(|| format!("{n} is in no file of the index"))
        })
        .collect::<Result<_, _>>()?;
    files.sort();
    files.dedup();
    files.iter().map(|f| SafeTensors::open(f)).collect()
}

impl TextEncoder {
    /// The bytes it takes on the GPU while loaded, and while it encodes.
    pub fn footprint() -> Footprint {
        let fp8 = (LAYERS * (QKV * DIM + DIM * HEADS * HEAD + 2 * FF * DIM + DIM * FF)) as u64;
        let scales = fp8 / (64 * 128) * 4;
        let small = ((LAYERS * (2 * DIM + 2 * HEAD)) * 4 + TOKENS * 128 * 4) as u64;
        Footprint {
            resident: fp8 + scales + small,
            working: plan(&[], 0, false, TOKENS).bytes(),
        }
    }

    /// Qwen3-4B from Qwen's FP8 release in `dir` (its tokenizer too), on
    /// the GPU.
    pub fn load(dir: &Path, precision: Precision) -> Result<TextEncoder, String> {
        let tokenizer = Tokenizer::open(dir)?;
        let layer = |i: usize, n: &str| format!("model.layers.{i}.{n}");
        let mut names = vec!["model.embed_tokens.weight".to_owned()];
        names.extend((0..LAYERS).map(|i| layer(i, "input_layernorm.weight")));
        let files = shards(dir, &names)?;
        let find = |name: &str| {
            files
                .iter()
                .find(|f| f.has(name))
                .ok_or(format!("no tensor {name}"))
        };
        let table = bf16_bytes(find(&names[0])?, &names[0], &[VOCAB, DIM])?.to_vec();
        let mut pack = Packing::new();
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let t = find(&layer(i, "input_layernorm.weight"))?;
            let fp8 = |n: &str, shape| Fp8::per_block(t, &layer(i, n), shape);
            let q = fp8("self_attn.q_proj", (HEADS * HEAD, DIM))?;
            let k = fp8("self_attn.k_proj", (KV_HEADS * HEAD, DIM))?;
            let v = fp8("self_attn.v_proj", (KV_HEADS * HEAD, DIM))?;
            let o = fp8("self_attn.o_proj", (DIM, HEADS * HEAD))?;
            let gate = fp8("mlp.gate_proj", (FF, DIM))?;
            let up = fp8("mlp.up_proj", (FF, DIM))?;
            let down = fp8("mlp.down_proj", (DIM, FF))?;
            layers.push(Layer {
                input_norm: pack.bf16_vector(t, &layer(i, "input_layernorm.weight"), DIM)?,
                post_norm: pack.bf16_vector(
                    t,
                    &layer(i, "post_attention_layernorm.weight"),
                    DIM,
                )?,
                q_norm: pack.bf16_vector(t, &layer(i, "self_attn.q_norm.weight"), HEAD)?,
                k_norm: pack.bf16_vector(t, &layer(i, "self_attn.k_norm.weight"), HEAD)?,
                qkv: pack.fp8_stack(&[q, k, v]),
                o: pack.fp8(&o),
                gate_up: pack.fp8_pairs(&gate, &up),
                down: pack.fp8(&down),
            });
        }
        let rope = pack.vector(&rope_table());
        let loaded = pack.upload("jodel-gpu text encoder")?;
        Ok(TextEncoder {
            tokenizer,
            table,
            loaded,
            layers,
            rope,
            precision,
            made: None,
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

    /// What it takes on the GPU now: its weights, and its plan's buffers
    /// (made on the first prompt, kept until [`TextEncoder::release`]).
    pub fn bytes(&self) -> Footprint {
        Footprint {
            resident: self.loaded.bytes(),
            working: self.made.as_ref().map_or(0, |(_, m)| m.bytes()),
        }
    }

    /// The bytes the GPU's allocator holds for every part now (where the
    /// backend tells).
    pub fn reserved(&self) -> Option<u64> {
        self.loaded.gpu.reserved()
    }

    /// Lets the plan's buffers go (the weights stay).
    pub fn release(&mut self) {
        self.made = None;
    }

    /// The tokens of `prompt` as the encoder takes it, and how many are
    /// the prompt's.
    pub fn tokens(&self, prompt: &str) -> Result<(Vec<u32>, usize), String> {
        self.tokenizer.prompt(prompt)
    }

    /// `prompt`'s embeddings.
    pub fn encode(&mut self, prompt: &str) -> Result<Embeds, String> {
        let (ids, real) = self.tokens(prompt)?;
        let x: Vec<f32> = ids
            .iter()
            .flat_map(|&id| {
                let row = &self.table[id as usize * DIM * 2..(id as usize + 1) * DIM * 2];
                row.chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            })
            .collect();
        // The plan depends on the prompt's length (the keys that count).
        if self.made.as_ref().is_none_or(|(n, _)| *n != real) {
            self.made = None;
            let fast = self.precision == Precision::Fast && self.loaded.gpu.coop;
            let p = plan(&self.layers, self.rope, fast, real);
            self.made = Some((real, Made::new(&self.loaded, &p, TOKENS * WIDTH)?));
        }
        let (_, made) = self.made.as_mut().expect("made");
        let values = made.run(&self.loaded, &[(X, 0, &x)], (OUT, 0, TOKENS * WIDTH))?;
        Ok(Embeds { values })
    }
}

/// The plan's slots: the hidden state, its norm, the queries' keys' and
/// values' rows, the heads, the attention's output (then SwiGLU's), the
/// embeddings.
const X: u8 = 0;
const H: u8 = 1;
const QKV_ROWS: u8 = 2;
const HEADS_SLOT: u8 = 3;
const MID: u8 = 4;
const OUT: u8 = 5;

/// The rotary table for 512 positions: per position, 64 pairs' cosine and
/// sine (θ = 10⁶ over halves of 128), in f32 as the reference makes it.
fn rope_table() -> Vec<f32> {
    let mut out = Vec::with_capacity(TOKENS * 128);
    for pos in 0..TOKENS {
        for i in 0..64 {
            let inv = 1.0f32 / THETA.powf((2 * i) as f32 / 128.0);
            let angle = pos as f32 * inv;
            out.push(angle.cos());
            out.push(angle.sin());
        }
    }
    out
}

/// The encoder's plan over 512 tokens, the first `valid` the prompt's (the
/// padding's keys left out); with no layers, its slots only (for its
/// footprint).
fn plan(layers: &[Layer], rope: usize, fast: bool, valid: usize) -> Plan {
    let mut p = Plan::new(fast);
    let n = TOKENS;
    p.need(X, n * DIM);
    p.need(H, n * DIM);
    p.need(QKV_ROWS, n * QKV);
    p.need(HEADS_SLOT, n * QKV);
    p.need(MID, n * FF);
    p.need(OUT, n * WIDTH);
    let x = At::new(X, 0);
    let h = At::new(H, 0);
    let rows = At::new(QKV_ROWS, 0);
    let heads = At::new(HEADS_SLOT, 0);
    let mid = At::new(MID, 0);
    let (q, k, v) = (
        heads,
        heads.plus(HEADS * n * HEAD),
        heads.plus((HEADS + KV_HEADS) * n * HEAD),
    );
    let table = Some((Rope::Halves, Buf::Small, rope));
    for (i, l) in layers.iter().enumerate() {
        p.rmsnorm(n, DIM, (x, DIM), (h, DIM), l.input_norm, EPS);
        p.gemm((h, DIM), n, &l.qkv, (rows, QKV), Tail::Store);
        p.qk_prep(
            (n, HEADS),
            (rows, QKV),
            (q, n, 0),
            Some(l.q_norm),
            table,
            EPS,
        );
        p.qk_prep(
            (n, KV_HEADS),
            (rows.plus(HEADS * HEAD), QKV),
            (k, n, 0),
            Some(l.k_norm),
            table,
            EPS,
        );
        p.qk_prep(
            (n, KV_HEADS),
            (rows.plus((HEADS + KV_HEADS) * HEAD), QKV),
            (v, n, 0),
            None,
            None,
            EPS,
        );
        p.attention(
            (n, n, valid),
            (HEADS, KV_HEADS),
            (q, k, v),
            (mid, HEADS * HEAD),
            true,
        );
        p.gemm((mid, HEADS * HEAD), n, &l.o, (x, DIM), Tail::Add);
        p.rmsnorm(n, DIM, (x, DIM), (h, DIM), l.post_norm, EPS);
        p.gemm((h, DIM), n, &l.gate_up, (mid, FF), Tail::SwiGlu);
        p.gemm((mid, FF), n, &l.down, (x, DIM), Tail::Add);
        if let Some(j) = TAKEN.iter().position(|&t| t == i + 1) {
            p.copy(n, DIM, (x, DIM), (At::new(OUT, j * DIM), WIDTH));
        }
    }
    p
}
