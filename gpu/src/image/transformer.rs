//! FLUX.2 klein 4B's transformer (BFL's FP8 checkpoint): the prompt's 512
//! tokens and the picture's (the noisy latents' 2 × 2 patches, then each
//! reference picture's) in one sequence, the text first. Per step: the
//! timestep's embedding and from it every block's modulation (shift,
//! scale and gate, for all four steps at once before the first); the
//! inputs embedded; 5 double-stream blocks (text and picture each with
//! their own matrices, attending together) and 20 single-stream ones (the
//! queries, keys, values and the feed-forward's gate and up as one matrix,
//! the attention's output and SwiGLU's as one input to the last); the
//! final norm and projection of the picture's latent tokens (not the
//! references'), and the sampler's step on them.

use super::plan::{At, Buf, Mat, Plan, Rope, Tail};
use super::weights::{Fp8, Packing, Source};

pub(crate) const DIM: usize = 3072;
const HEADS: usize = 24;
const HEAD: usize = 128;
const FF: usize = 9216;
pub(crate) const CHANNELS: usize = 128;
const TEXT: usize = super::encoder::WIDTH;
const DOUBLE: usize = 5;
const SINGLE: usize = 20;
const EPS: f32 = 1e-6;
/// The timestep's sinusoids.
const TIME: usize = 256;
/// The modulation's widths: double-stream (two sets of shift, scale and
/// gate), single-stream (one), final (shift and scale).
const MOD2: usize = 6 * DIM;
const MOD1: usize = 3 * DIM;
const MODF: usize = 2 * DIM;

/// One stream's matrices of a double-stream block, and its heads' norms.
struct Stream {
    qkv: Mat,
    proj: Mat,
    mlp_in: Mat,
    mlp_out: Mat,
    q_norm: usize,
    k_norm: usize,
}

struct Single {
    qkv: Mat,
    mlp_in: Mat,
    out: Mat,
    q_norm: usize,
    k_norm: usize,
}

/// The transformer's weights in a part's buffers.
pub(crate) struct Transformer {
    img_in: Mat,
    txt_in: Mat,
    time_in: Mat,
    time_out: Mat,
    mod_img: Mat,
    mod_txt: Mat,
    mod_single: Mat,
    mod_final: Mat,
    final_out: Mat,
    double: Vec<(Stream, Stream)>,
    single: Vec<Single>,
}

impl Transformer {
    /// The weights from BFL's checkpoint into `pack`.
    pub(crate) fn pack<'a>(src: Source<'a>, pack: &mut Packing<'a>) -> Result<Transformer, String> {
        let bf = |pack: &mut Packing<'a>, name: &str, shape| {
            pack.bf16(src, &format!("{name}.weight"), shape)
        };
        let stream = |pack: &mut Packing<'a>, b: &str| -> Result<Stream, String> {
            let fp8 = |n: &str, shape| Fp8::per_tensor(src, &format!("{b}_{n}"), shape);
            let mlp = fp8("mlp.0", (2 * FF, DIM))?;
            Ok(Stream {
                qkv: pack.fp8(&fp8("attn.qkv", (3 * DIM, DIM))?),
                proj: pack.fp8(&fp8("attn.proj", (DIM, DIM))?),
                mlp_in: pack.fp8_pairs(&mlp.rows(0, FF), &mlp.rows(FF, FF)),
                mlp_out: pack.fp8(&fp8("mlp.2", (DIM, FF))?),
                q_norm: pack.bf16_vector(src, &format!("{b}_attn.norm.query_norm.scale"), HEAD)?,
                k_norm: pack.bf16_vector(src, &format!("{b}_attn.norm.key_norm.scale"), HEAD)?,
            })
        };
        let double = (0..DOUBLE)
            .map(|i| {
                Ok((
                    stream(pack, &format!("double_blocks.{i}.img"))?,
                    stream(pack, &format!("double_blocks.{i}.txt"))?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let single = (0..SINGLE)
            .map(|i| {
                let b = format!("single_blocks.{i}");
                let l1 = Fp8::per_tensor(src, &format!("{b}.linear1"), (3 * DIM + 2 * FF, DIM))?;
                Ok(Single {
                    qkv: pack.fp8(&l1.rows(0, 3 * DIM)),
                    mlp_in: pack.fp8_pairs(&l1.rows(3 * DIM, FF), &l1.rows(3 * DIM + FF, FF)),
                    out: pack.fp8(&Fp8::per_tensor(
                        src,
                        &format!("{b}.linear2"),
                        (DIM, DIM + FF),
                    )?),
                    q_norm: pack.bf16_vector(src, &format!("{b}.norm.query_norm.scale"), HEAD)?,
                    k_norm: pack.bf16_vector(src, &format!("{b}.norm.key_norm.scale"), HEAD)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Transformer {
            img_in: bf(pack, "img_in", (DIM, CHANNELS))?,
            txt_in: bf(pack, "txt_in", (DIM, TEXT))?,
            time_in: bf(pack, "time_in.in_layer", (DIM, TIME))?,
            time_out: bf(pack, "time_in.out_layer", (DIM, DIM))?,
            mod_img: bf(pack, "double_stream_modulation_img.lin", (MOD2, DIM))?,
            mod_txt: bf(pack, "double_stream_modulation_txt.lin", (MOD2, DIM))?,
            mod_single: bf(pack, "single_stream_modulation.lin", (MOD1, DIM))?,
            mod_final: bf(pack, "final_layer.adaLN_modulation.1", (MODF, DIM))?,
            final_out: bf(pack, "final_layer.linear", (CHANNELS, DIM))?,
            double,
            single,
        })
    }
}

/// A picture's tokens: the prompt's, the latents' (`h × w` patches) and
/// each reference's (its patches' rows and columns).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tokens {
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) refs: Vec<(usize, usize)>,
}

impl Tokens {
    pub(crate) fn text(&self) -> usize {
        super::text::TOKENS
    }

    /// The latents' tokens.
    pub(crate) fn latent(&self) -> usize {
        self.h * self.w
    }

    /// The latents' and the references' tokens.
    pub(crate) fn picture(&self) -> usize {
        self.latent() + self.refs.iter().map(|&(h, w)| h * w).sum::<usize>()
    }

    pub(crate) fn all(&self) -> usize {
        self.text() + self.picture()
    }

    /// Each token's place (time, row, column, text position) as the
    /// pipeline gives them: the prompt's (0, 0, 0, l), the latents' (0, y,
    /// x, 0), the k-th reference's (10·(k + 1), y, x, 0).
    fn ids(&self) -> Vec<[usize; 4]> {
        let mut out: Vec<[usize; 4]> = (0..self.text()).map(|l| [0, 0, 0, l]).collect();
        let grid = |t: usize, h: usize, w: usize| (0..h * w).map(move |i| [t, i / w, i % w, 0]);
        out.extend(grid(0, self.h, self.w));
        for (k, &(h, w)) in self.refs.iter().enumerate() {
            out.extend(grid(10 * (k + 1), h, w));
        }
        out
    }

    /// The rotary table: per token, 64 pairs' cosine and sine (16 per
    /// place's axis, θ = 2000 over 32), reckoned in f64 as the pipeline
    /// does, then f32.
    pub(crate) fn rope(&self) -> Vec<f32> {
        let freqs = crate::engine::Freqs {
            base: 2000.0,
            dim: 32,
            turned: 16,
        };
        crate::engine::rotary(
            self.ids().into_iter().flatten(),
            freqs,
            crate::engine::Angles::F64,
        )
    }
}

/// The timestep's sinusoids (cosines then sines, 128 each, as the model
/// embeds a timestep `t` of 0..1000).
pub(crate) fn time_sinusoids(t: f32) -> Vec<f32> {
    let half = TIME / 2;
    let freqs = (0..half).map(|i| (-(10_000f32.ln()) * i as f32 / half as f32).exp());
    let args: Vec<f32> = freqs.map(|f| t * f).collect();
    args.iter()
        .map(|a| a.cos())
        .chain(args.iter().map(|a| a.sin()))
        .collect()
}

/// The transformer's slots: the hidden state, its norm, the queries' keys'
/// and values' rows, the heads, the attention's output beside the
/// feed-forward's; the latents and the references' latents (tokens of
/// 128), the prompt's embeddings, the steps' sinusoids (and their SiLUs),
/// the timesteps' embeddings, the modulations, the rotary table, the
/// prediction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slots {
    pub(crate) x: u8,
    pub(crate) h: u8,
    pub(crate) qkv: u8,
    pub(crate) heads: u8,
    pub(crate) cat: u8,
    pub(crate) lat: u8,
    pub(crate) txt: u8,
    pub(crate) times: u8,
    temb: u8,
    mods: u8,
    pub(crate) rope: u8,
    pub(crate) pred: u8,
}

/// Where the four steps' modulations lie in their slot: double-stream
/// picture's, text's, single-stream, final.
fn mods(steps: usize) -> [usize; 4] {
    let img = 0;
    let txt = img + steps * MOD2;
    let single = txt + steps * MOD2;
    let fin = single + steps * MOD1;
    [img, txt, single, fin]
}

impl Slots {
    /// The slots `tokens` takes over `steps`, handed out by `p`.
    pub(crate) fn new(p: &mut Plan, tokens: &Tokens, steps: usize) -> Slots {
        let l = tokens.all();
        Slots {
            x: p.slot(l * DIM),
            h: p.slot(l * DIM),
            qkv: p.slot(l * 3 * DIM),
            heads: p.slot(3 * l * DIM),
            cat: p.slot(l * (DIM + FF)),
            lat: p.slot(tokens.picture() * CHANNELS),
            txt: p.slot(tokens.text() * TEXT),
            times: p.slot(steps * (TIME + 2 * DIM)),
            temb: p.slot(steps * 2 * DIM),
            mods: p.slot(mods(steps)[3] + steps * MODF),
            rope: p.slot(l * 128),
            pred: p.slot(tokens.latent() * CHANNELS),
        }
    }

    /// The large ones, which the autoencoder takes in turns with it.
    pub(crate) fn large(&self) -> [u8; 5] {
        [self.cat, self.qkv, self.heads, self.h, self.x]
    }
}

impl Transformer {
    /// The steps' modulations, from their sinusoids (in `times`).
    pub(crate) fn plan_mods(&self, p: &mut Plan, s: &Slots, steps: usize) {
        let times = At::new(s.times, 0);
        let s1 = times.plus(steps * TIME);
        let s2 = s1.plus(steps * DIM);
        let t1 = At::new(s.temb, 0);
        let temb = t1.plus(steps * DIM);
        p.gemm((times, TIME), steps, &self.time_in, (t1, DIM), Tail::Store);
        p.silu(steps * DIM, t1, s1);
        p.gemm((s1, DIM), steps, &self.time_out, (temb, DIM), Tail::Store);
        p.silu(steps * DIM, temb, s2);
        let [img, txt, single, fin] = mods(steps);
        for (w, at, width) in [
            (&self.mod_img, img, MOD2),
            (&self.mod_txt, txt, MOD2),
            (&self.mod_single, single, MOD1),
            (&self.mod_final, fin, MODF),
        ] {
            p.gemm(
                (s2, DIM),
                steps,
                w,
                (At::new(s.mods, at), width),
                Tail::Store,
            );
        }
    }

    /// Step `step` of `steps`: the prediction for the latents (in `lat`),
    /// and the sampler's step of `dt` on them.
    pub(crate) fn plan_step(
        &self,
        p: &mut Plan,
        (s, tokens): (&Slots, &Tokens),
        (step, steps): (usize, usize),
        dt: f32,
    ) {
        let (t, n) = (tokens.text(), tokens.picture());
        let l = t + n;
        let x = At::new(s.x, 0);
        let h = At::new(s.h, 0);
        let qkv = At::new(s.qkv, 0);
        let heads = At::new(s.heads, 0);
        let qkv_heads = (heads, heads.plus(l * DIM), heads.plus(2 * l * DIM));
        let cat = At::new(s.cat, 0);
        let wide = DIM + FF;
        let rope = Some((Rope::Pairs, Buf::Slot(s.rope), 0));
        let [img_at, txt_at, single_at, fin_at] = mods(steps);
        let m =
            |at: usize, width: usize, part: usize| At::new(s.mods, at + step * width + part * DIM);
        // The inputs.
        p.gemm(
            (At::new(s.lat, 0), CHANNELS),
            n,
            &self.img_in,
            (x.plus(t * DIM), DIM),
            Tail::Store,
        );
        p.gemm(
            (At::new(s.txt, 0), TEXT),
            t,
            &self.txt_in,
            (x, DIM),
            Tail::Store,
        );
        // A stream's rows: (first row, rows, its modulation, its weights).
        for (img, txt) in &self.double {
            let streams = [(t, n, img_at, img), (0, t, txt_at, txt)];
            for &(r, rows, at, _) in &streams {
                let rx = x.plus(r * DIM);
                p.layernorm_mod(
                    rows,
                    DIM,
                    (rx, DIM),
                    (h.plus(r * DIM), DIM),
                    m(at, MOD2, 0),
                    m(at, MOD2, 1),
                    EPS,
                );
            }
            for &(r, rows, _, w) in &streams {
                let src = qkv.plus(r * 3 * DIM);
                p.gemm(
                    (h.plus(r * DIM), DIM),
                    rows,
                    &w.qkv,
                    (src, 3 * DIM),
                    Tail::Store,
                );
                p.heads(
                    (rows, HEADS, HEADS),
                    (src, 3 * DIM),
                    (qkv_heads, l, r),
                    (w.q_norm, w.k_norm),
                    rope,
                    EPS,
                );
            }
            p.attention((l, l, l), (HEADS, HEADS), qkv_heads, (cat, wide), false);
            for &(r, rows, at, w) in &streams {
                let rx = x.plus(r * DIM);
                let rc = cat.plus(r * wide);
                p.gemm(
                    (rc, wide),
                    rows,
                    &w.proj,
                    (rx, DIM),
                    Tail::Gated(m(at, MOD2, 2)),
                );
                p.layernorm_mod(
                    rows,
                    DIM,
                    (rx, DIM),
                    (h.plus(r * DIM), DIM),
                    m(at, MOD2, 3),
                    m(at, MOD2, 4),
                    EPS,
                );
                p.gemm(
                    (h.plus(r * DIM), DIM),
                    rows,
                    &w.mlp_in,
                    (rc.plus(DIM), wide),
                    Tail::SwiGlu,
                );
                p.gemm(
                    (rc.plus(DIM), wide),
                    rows,
                    &w.mlp_out,
                    (rx, DIM),
                    Tail::Gated(m(at, MOD2, 5)),
                );
            }
        }
        for w in &self.single {
            p.layernorm_mod(
                l,
                DIM,
                (x, DIM),
                (h, DIM),
                m(single_at, MOD1, 0),
                m(single_at, MOD1, 1),
                EPS,
            );
            p.gemm((h, DIM), l, &w.qkv, (qkv, 3 * DIM), Tail::Store);
            p.gemm((h, DIM), l, &w.mlp_in, (cat.plus(DIM), wide), Tail::SwiGlu);
            p.heads(
                (l, HEADS, HEADS),
                (qkv, 3 * DIM),
                (qkv_heads, l, 0),
                (w.q_norm, w.k_norm),
                rope,
                EPS,
            );
            p.attention((l, l, l), (HEADS, HEADS), qkv_heads, (cat, wide), false);
            p.gemm(
                (cat, wide),
                l,
                &w.out,
                (x, DIM),
                Tail::Gated(m(single_at, MOD1, 2)),
            );
        }
        // The latents' tokens only.
        let lat = tokens.latent();
        p.layernorm_mod(
            lat,
            DIM,
            (x.plus(t * DIM), DIM),
            (h, DIM),
            m(fin_at, MODF, 0),
            m(fin_at, MODF, 1),
            EPS,
        );
        let pred = At::new(s.pred, 0);
        p.gemm(
            (h, DIM),
            lat,
            &self.final_out,
            (pred, CHANNELS),
            Tail::Store,
        );
        p.euler(lat * CHANNELS, At::new(s.lat, 0), pred, dt);
    }
}
