//! FLUX.2's autoencoder (diffusers' `AutoencoderKLFlux2`, bf16): its
//! encoder for the reference pictures (to latents, their 2 × 2 patches
//! normalized by its batch norm: the transformer's tokens) and its decoder
//! for the picture. Every convolution is a matrix multiply reading its
//! patches where they lie, its group norm and SiLU taken on the way (the
//! norm's statistics first, in two small dispatches); the middle block's
//! attention (one head of 512 over every latent pixel) as two matrix
//! multiplies around a softmax, a band of queries at a time.
//!
//! Its buffers are five large slots taken in turns (`Slots`): a resnet
//! block's input, its first convolution's output, and (where the channels
//! change) its shortcut's; the attention's queries, keys and values, the
//! values transposed, the scores and the output.

use jodel_tensors::SafeTensors;

use super::plan::{At, Conv, Mat, Norm, Plan, Tail};
use super::weights::Packing;

const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
const BN_EPS: f32 = 1e-4;
pub(crate) const LATENT: usize = 32;
/// The encoder's channels by block (the decoder's reversed).
const BLOCKS: [usize; 4] = [128, 256, 512, 512];
const MID: usize = 512;
/// The attention's queries a band.
const BAND: usize = 2048;

/// A resnet block: two group norms with SiLU and 3 × 3 convolutions, and a
/// 1 × 1 shortcut where the channels change.
struct Resnet {
    cin: usize,
    cout: usize,
    norm1: (usize, usize),
    conv1: Mat,
    norm2: (usize, usize),
    conv2: Mat,
    shortcut: Option<Mat>,
}

/// The middle block's attention: its group norm, queries, keys, values
/// and output (1 × 1 convolutions, with biases).
struct Attention {
    norm: (usize, usize),
    q: Mat,
    k: Mat,
    v: Mat,
    out: Mat,
}

struct Middle {
    first: Resnet,
    attention: Attention,
    second: Resnet,
}

struct Encoder {
    conv_in: Mat,
    /// Each block's resnets and its downsampling (3 × 3, stride 2).
    blocks: Vec<(Vec<Resnet>, Option<Mat>)>,
    middle: Middle,
    norm_out: (usize, usize),
    conv_out: Mat,
    quant: Mat,
}

struct Decoder {
    post_quant: Mat,
    conv_in: Mat,
    middle: Middle,
    /// Each block's resnets and its upsampling's convolution.
    blocks: Vec<(Vec<Resnet>, Option<Mat>)>,
    norm_out: (usize, usize),
    conv_out: Mat,
}

/// The autoencoder's weights in a part's buffers.
pub(crate) struct Vae {
    encoder: Encoder,
    decoder: Decoder,
    /// The batch norm's means and spreads (√(var + ε)) over the latents'
    /// 128 patch channels.
    bn: (usize, usize),
}

/// What makes the weights: a convolution's (name, cin, cout, size), a
/// linear layer's as a 1 × 1 convolution, a vector's (name, length).
trait Source {
    fn conv(&mut self, name: &str, shape: (usize, usize, usize)) -> Result<Mat, String>;
    fn linear(&mut self, name: &str, shape: (usize, usize)) -> Result<Mat, String>;
    fn vector(&mut self, name: &str, len: usize) -> Result<usize, String>;
    fn values(&mut self, values: &[f32]) -> usize;
    fn get(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String>;
}

/// The checkpoint into a part's packing.
struct Checkpoint<'p, 'a> {
    t: &'a SafeTensors,
    pack: &'p mut Packing<'a>,
}

/// A convolution's input channels as the kernels read them (a multiple of
/// 4: the picture's 3 padded).
fn padded(cin: usize) -> usize {
    cin.next_multiple_of(4)
}

impl Source for Checkpoint<'_, '_> {
    fn conv(
        &mut self,
        name: &str,
        (cin, cout, size): (usize, usize, usize),
    ) -> Result<Mat, String> {
        self.pack.conv(self.t, name, (cin, cout, size), padded(cin))
    }

    fn linear(&mut self, name: &str, (cin, cout): (usize, usize)) -> Result<Mat, String> {
        self.pack.linear(self.t, name, (cin, cout))
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<usize, String> {
        self.pack.bf16_vector(self.t, name, len)
    }

    fn values(&mut self, values: &[f32]) -> usize {
        self.pack.vector(values)
    }

    fn get(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        self.t.get(name, &[len])
    }
}

/// The weights' shapes alone (for a plan's footprint before loading), and
/// the bytes they would take.
struct Shapes {
    bytes: u64,
}

impl Source for Shapes {
    fn conv(&mut self, _: &str, (cin, cout, size): (usize, usize, usize)) -> Result<Mat, String> {
        let k = size * size * padded(cin);
        self.bytes += (cout * k.next_multiple_of(32) * 2 + cout * 4) as u64;
        Ok(Mat {
            buf: super::plan::Buf::W(0),
            at: 0,
            n: cout,
            k,
            ldw: k.next_multiple_of(32),
            format: super::plan::Format::Bf16,
            bias: Some(0),
        })
    }

    fn linear(&mut self, name: &str, (cin, cout): (usize, usize)) -> Result<Mat, String> {
        self.conv(name, (cin, cout, 1))
    }

    fn vector(&mut self, _: &str, len: usize) -> Result<usize, String> {
        self.bytes += (len * 4) as u64;
        Ok(0)
    }

    fn values(&mut self, values: &[f32]) -> usize {
        self.bytes += (values.len() * 4) as u64;
        0
    }

    fn get(&mut self, _: &str, len: usize) -> Result<Vec<f32>, String> {
        Ok(vec![1.0; len])
    }
}

fn norm(s: &mut dyn Source, name: &str, c: usize) -> Result<(usize, usize), String> {
    Ok((
        s.vector(&format!("{name}.weight"), c)?,
        s.vector(&format!("{name}.bias"), c)?,
    ))
}

fn resnet(s: &mut dyn Source, name: &str, cin: usize, cout: usize) -> Result<Resnet, String> {
    Ok(Resnet {
        cin,
        cout,
        norm1: norm(s, &format!("{name}.norm1"), cin)?,
        conv1: s.conv(&format!("{name}.conv1"), (cin, cout, 3))?,
        norm2: norm(s, &format!("{name}.norm2"), cout)?,
        conv2: s.conv(&format!("{name}.conv2"), (cout, cout, 3))?,
        shortcut: if cin != cout {
            Some(s.conv(&format!("{name}.conv_shortcut"), (cin, cout, 1))?)
        } else {
            None
        },
    })
}

fn middle(s: &mut dyn Source, name: &str) -> Result<Middle, String> {
    let a = format!("{name}.attentions.0");
    Ok(Middle {
        first: resnet(s, &format!("{name}.resnets.0"), MID, MID)?,
        attention: Attention {
            norm: norm(s, &format!("{a}.group_norm"), MID)?,
            q: s.linear(&format!("{a}.to_q"), (MID, MID))?,
            k: s.linear(&format!("{a}.to_k"), (MID, MID))?,
            v: s.linear(&format!("{a}.to_v"), (MID, MID))?,
            out: s.linear(&format!("{a}.to_out.0"), (MID, MID))?,
        },
        second: resnet(s, &format!("{name}.resnets.1"), MID, MID)?,
    })
}

impl Vae {
    /// The weights from the checkpoint into `pack`.
    pub(crate) fn pack<'a>(t: &'a SafeTensors, pack: &mut Packing<'a>) -> Result<Vae, String> {
        Vae::from(&mut Checkpoint { t, pack })
    }

    /// The weights' shapes alone, and the bytes they take.
    pub(crate) fn shapes() -> (Vae, u64) {
        let mut s = Shapes { bytes: 0 };
        let vae = Vae::from(&mut s).expect("shapes alone do not fail");
        (vae, s.bytes)
    }

    fn from(s: &mut dyn Source) -> Result<Vae, String> {
        let mut blocks = Vec::new();
        let mut cin = BLOCKS[0];
        for (i, &c) in BLOCKS.iter().enumerate() {
            let b = format!("encoder.down_blocks.{i}");
            let resnets = (0..2)
                .map(|j| {
                    resnet(
                        s,
                        &format!("{b}.resnets.{j}"),
                        if j == 0 { cin } else { c },
                        c,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            let down = if i + 1 < BLOCKS.len() {
                Some(s.conv(&format!("{b}.downsamplers.0.conv"), (c, c, 3))?)
            } else {
                None
            };
            blocks.push((resnets, down));
            cin = c;
        }
        let encoder = Encoder {
            conv_in: s.conv("encoder.conv_in", (3, BLOCKS[0], 3))?,
            blocks,
            middle: middle(s, "encoder.mid_block")?,
            norm_out: norm(s, "encoder.conv_norm_out", MID)?,
            conv_out: s.conv("encoder.conv_out", (MID, 2 * LATENT, 3))?,
            quant: s.conv("quant_conv", (2 * LATENT, 2 * LATENT, 1))?,
        };
        let mut blocks = Vec::new();
        let mut cin = MID;
        for (i, &c) in BLOCKS.iter().rev().enumerate() {
            let b = format!("decoder.up_blocks.{i}");
            let resnets = (0..3)
                .map(|j| {
                    resnet(
                        s,
                        &format!("{b}.resnets.{j}"),
                        if j == 0 { cin } else { c },
                        c,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            let up = if i + 1 < BLOCKS.len() {
                Some(s.conv(&format!("{b}.upsamplers.0.conv"), (c, c, 3))?)
            } else {
                None
            };
            blocks.push((resnets, up));
            cin = c;
        }
        let decoder = Decoder {
            post_quant: s.conv("post_quant_conv", (LATENT, LATENT, 1))?,
            conv_in: s.conv("decoder.conv_in", (LATENT, MID, 3))?,
            middle: middle(s, "decoder.mid_block")?,
            blocks,
            norm_out: norm(s, "decoder.conv_norm_out", BLOCKS[0])?,
            conv_out: s.conv("decoder.conv_out", (BLOCKS[0], 3, 3))?,
        };
        let mean = s.get("bn.running_mean", 4 * LATENT)?;
        let spread: Vec<f32> = s
            .get("bn.running_var", 4 * LATENT)?
            .iter()
            .map(|v| (v + BN_EPS).sqrt())
            .collect();
        let bn = (s.values(&mean), s.values(&spread));
        Ok(Vae {
            encoder,
            decoder,
            bn,
        })
    }
}

/// The autoencoder's slots: five large ones taken in turns, the group
/// norm's partial sums and statistics.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slots {
    pub(crate) large: [u8; 5],
    pub(crate) part: u8,
    pub(crate) stats: u8,
}

/// A feature map: its slot, rows, columns and channels.
#[derive(Clone, Copy, Debug)]
struct Map {
    slot: u8,
    h: usize,
    w: usize,
    c: usize,
}

impl Map {
    fn at(&self) -> At {
        At::new(self.slot, 0)
    }

    fn len(&self) -> usize {
        self.h * self.w * self.c
    }
}

/// Plans the autoencoder's layers into `p`.
struct Planner<'p> {
    p: &'p mut Plan,
    s: Slots,
}

impl Planner<'_> {
    /// A feature map in `slot` (the slot made large enough).
    fn map(&mut self, slot: u8, h: usize, w: usize, c: usize) -> Map {
        let m = Map { slot, h, w, c };
        self.p.need(slot, m.len());
        m
    }

    /// The group norm's statistics of `x` (to the stats slot from `at`).
    fn stats(&mut self, x: Map, at: usize) -> At {
        let stats = At::new(self.s.stats, at);
        self.p.need(self.s.stats, at + 2 * GROUPS);
        self.p.gn_stats(
            (x.h * x.w, x.c),
            x.at(),
            At::new(self.s.part, 0),
            stats,
            EPS,
        );
        stats
    }

    fn norm(&self, stats: At, (weight, bias): (usize, usize), c: usize, silu: bool) -> Norm {
        Norm {
            stats,
            weight,
            bias,
            group: c / GROUPS,
            silu,
        }
    }

    /// `w` over `x` into `out` (its slot made large enough).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: Map,
        w: &Mat,
        (size, stride, pad, up): (usize, usize, (usize, usize), bool),
        out: u8,
        tail: Tail,
        norm: Option<Norm>,
    ) -> Map {
        let conv = Conv {
            input: x.at(),
            h: x.h,
            w: x.w,
            cin: padded(x.c),
            size,
            stride,
            pad: pad.0,
            trail: pad.1,
            up,
        };
        let (oh, ow) = conv.out();
        let y = self.map(out, oh, ow, w.n);
        self.p.conv(conv, w, y.at(), tail, norm);
        y
    }

    /// A resnet block on `x`, with `b` and (where the channels change) `c`
    /// free: the slot of its output.
    fn resnet(&mut self, r: &Resnet, x: Map, b: u8, c: u8) -> Map {
        assert_eq!(x.c, r.cin);
        let s1 = self.stats(x, 0);
        let n1 = self.norm(s1, r.norm1, r.cin, true);
        let h = self.conv(x, &r.conv1, (3, 1, (1, 1), false), b, Tail::Store, Some(n1));
        let s2 = self.stats(h, 2 * GROUPS);
        let n2 = self.norm(s2, r.norm2, r.cout, true);
        let out = match &r.shortcut {
            Some(sc) => self.conv(x, sc, (1, 1, (0, 0), false), c, Tail::Store, None),
            None => x,
        };
        self.conv(
            h,
            &r.conv2,
            (3, 1, (1, 1), false),
            out.slot,
            Tail::Add,
            Some(n2),
        );
        out
    }

    /// The middle block's attention on `x` (added to it), with four slots
    /// free.
    fn attention(&mut self, a: &Attention, x: Map, [qkv, vt, scores, out]: [u8; 4]) {
        let l = x.h * x.w;
        assert_eq!(l % 32, 0, "the attention's keys in rows of 32");
        let stats = self.stats(x, 0);
        let n = self.norm(stats, a.norm, x.c, false);
        let cv = |planner: &mut Planner, w: &Mat, at: usize| {
            let conv = Conv {
                input: x.at(),
                h: x.h,
                w: x.w,
                cin: x.c,
                size: 1,
                stride: 1,
                pad: 0,
                trail: 0,
                up: false,
            };
            planner.p.need(qkv, at + l * MID);
            planner
                .p
                .conv(conv, w, At::new(qkv, at), Tail::Store, Some(n));
        };
        cv(self, &a.q, 0);
        cv(self, &a.k, l * MID);
        cv(self, &a.v, 2 * l * MID);
        self.p.need(vt, MID * l);
        self.p.transpose(
            (l, MID),
            (At::new(qkv, 2 * l * MID), MID),
            (At::new(vt, 0), l),
        );
        let keys = Mat::of_slot(At::new(qkv, l * MID), l, MID, MID);
        let values = Mat::of_slot(At::new(vt, 0), MID, l, l);
        self.p.need(out, l * MID);
        for r0 in (0..l).step_by(BAND) {
            let rows = BAND.min(l - r0);
            self.p.need(scores, rows * l);
            let s = At::new(scores, 0);
            self.p.gemm(
                (At::new(qkv, r0 * MID), MID),
                rows,
                &keys,
                (s, l),
                Tail::Store,
            );
            self.p.softmax(rows, l, (s, l), 1.0 / (MID as f32).sqrt());
            self.p.gemm(
                (s, l),
                rows,
                &values,
                (At::new(out, r0 * MID), MID),
                Tail::Store,
            );
        }
        self.p
            .gemm((At::new(out, 0), MID), l, &a.out, (x.at(), MID), Tail::Add);
    }

    fn middle(&mut self, m: &Middle, x: Map, free: [u8; 4]) {
        self.resnet(&m.first, x, free[0], free[1]);
        self.attention(&m.attention, x, free);
        self.resnet(&m.second, x, free[0], free[1]);
    }
}

impl Vae {
    /// The encoder over a picture of `h × w` (multiples of 16, its pixels'
    /// channels 4: RGB in -1..1 and a zero) at `image`: its latent tokens
    /// to `tokens` (`h/16 × w/16` of 128).
    pub(crate) fn plan_encode(
        &self,
        p: &mut Plan,
        s: Slots,
        image: At,
        (h, w): (usize, usize),
        tokens: At,
    ) {
        let e = &self.encoder;
        let [a, b, c, d, f] = s.large;
        let mut pl = Planner { p, s };
        let input = Map {
            slot: image.slot,
            h,
            w,
            c: 4,
        };
        assert_eq!(image.off, 0);
        let mut x = pl.conv(
            input,
            &e.conv_in,
            (3, 1, (1, 1), false),
            a,
            Tail::Store,
            None,
        );
        // x lives in `a` through the blocks, `b` and `c` free.
        for (resnets, down) in &e.blocks {
            for r in resnets {
                let y = pl.resnet(r, x, b, c);
                if y.slot != x.slot {
                    // The shortcut's slot holds it now: copied back keeps
                    // `a` its home.
                    x = pl.conv_copy(y, a);
                } else {
                    x = y;
                }
            }
            if let Some(dw) = down {
                let y = pl.conv(x, dw, (3, 2, (0, 1), false), b, Tail::Store, None);
                x = pl.conv_copy(y, a);
            }
        }
        pl.middle(&e.middle, x, [b, c, d, f]);
        let stats = pl.stats(x, 0);
        let n = pl.norm(stats, e.norm_out, x.c, true);
        let moments = pl.conv(
            x,
            &e.conv_out,
            (3, 1, (1, 1), false),
            b,
            Tail::Store,
            Some(n),
        );
        let q = pl.conv(
            moments,
            &e.quant,
            (1, 1, (0, 0), false),
            c,
            Tail::Store,
            None,
        );
        pl.p.patchify((h / 16, w / 16, 2 * LATENT), q.at(), tokens, self.bn);
    }

    /// The decoder over `h/16 × w/16` latent tokens at `tokens`: the
    /// picture (`h × w × 3`, in -1..1) to `image`.
    pub(crate) fn plan_decode(
        &self,
        p: &mut Plan,
        s: Slots,
        tokens: At,
        (h, w): (usize, usize),
        image: At,
    ) {
        let d = &self.decoder;
        let [a, b, c, e, f] = s.large;
        let mut pl = Planner { p, s };
        let lat = pl.map(e, h / 8, w / 8, LATENT);
        pl.p.unpatchify((h / 16, w / 16), tokens, lat.at(), self.bn);
        let post = pl.conv(
            lat,
            &d.post_quant,
            (1, 1, (0, 0), false),
            f,
            Tail::Store,
            None,
        );
        let mut x = pl.conv(
            post,
            &d.conv_in,
            (3, 1, (1, 1), false),
            a,
            Tail::Store,
            None,
        );
        pl.middle(&d.middle, x, [b, c, e, f]);
        // The blocks: x moves between `a`, `b` and `c`.
        for (resnets, up) in &d.blocks {
            for r in resnets {
                let others: Vec<u8> = [a, b, c].into_iter().filter(|&s| s != x.slot).collect();
                x = pl.resnet(r, x, others[0], others[1]);
            }
            if let Some(u) = up {
                let to = [a, b, c]
                    .into_iter()
                    .find(|&s| s != x.slot)
                    .expect("three slots");
                x = pl.conv(x, u, (3, 1, (1, 1), true), to, Tail::Store, None);
            }
        }
        let stats = pl.stats(x, 0);
        let n = pl.norm(stats, d.norm_out, x.c, true);
        assert_eq!(image.off, 0);
        pl.conv(
            x,
            &d.conv_out,
            (3, 1, (1, 1), false),
            image.slot,
            Tail::Store,
            Some(n),
        );
    }
}

impl Planner<'_> {
    /// `x` copied to `slot`.
    fn conv_copy(&mut self, x: Map, slot: u8) -> Map {
        let y = self.map(slot, x.h, x.w, x.c);
        self.p.copy(x.h * x.w, x.c, (x.at(), x.c), (y.at(), y.c));
        y
    }
}
