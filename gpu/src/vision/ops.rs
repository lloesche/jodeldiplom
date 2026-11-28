//! A network's plan: its buffers and its dispatches, each written by a
//! helper that knows its kernel's numbers (`conv.wgsl`, `ops.wgsl`,
//! `attention.wgsl`).

use super::Kernel;
use crate::engine::{self, params, spread};

/// One dispatch: its kernel, its buffers (the plan's: src, src2, dst), its
/// numbers and its workgroups.
pub(crate) type Op = engine::Op<Kernel, [usize; 3], 24>;

/// An image (or rows of tokens) in a buffer: `views` of `h × w` pixels,
/// each pixel's `c` channels from `off`, pixels `ld` apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Img {
    pub(crate) buf: usize,
    pub(crate) off: usize,
    pub(crate) views: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) c: usize,
    pub(crate) ld: usize,
}

impl Img {
    /// At the start of `buf`, its channels side by side.
    pub(crate) fn new(buf: usize, views: usize, h: usize, w: usize, c: usize) -> Img {
        Img {
            buf,
            off: 0,
            views,
            h,
            w,
            c,
            ld: c,
        }
    }

    /// Rows of tokens: `rows` of `c` features.
    pub(crate) fn rows(buf: usize, rows: usize, c: usize) -> Img {
        Img::new(buf, 1, 1, rows, c)
    }

    /// The f32s it spans.
    pub(crate) fn len(&self) -> usize {
        self.views * self.h * self.w * self.ld
    }

    /// The pixels of all views.
    pub(crate) fn pixels(&self) -> usize {
        self.views * self.h * self.w
    }

    /// Channels `from..from + count` of it.
    pub(crate) fn channels(self, from: usize, count: usize) -> Img {
        assert!(from + count <= self.c);
        Img {
            off: self.off + from,
            c: count,
            ..self
        }
    }

    /// Views `from..from + count` of it.
    pub(crate) fn views(self, from: usize, count: usize) -> Img {
        assert!(from + count <= self.views);
        Img {
            off: self.off + from * self.h * self.w * self.ld,
            views: count,
            ..self
        }
    }

    /// Rows `from..from + count` of it taken as one view of `pixels × 1`
    /// rows (tokens).
    pub(crate) fn tokens(self, from: usize, count: usize) -> Img {
        Img {
            off: self.off + from * self.ld,
            views: 1,
            h: 1,
            w: count,
            ..self
        }
    }

    /// The same values seen as `views × h × w`.
    pub(crate) fn shaped(self, views: usize, h: usize, w: usize) -> Img {
        assert_eq!(views * h * w, self.pixels());
        Img {
            views,
            h,
            w,
            ..self
        }
    }
}

/// A convolution's weights in the packed buffer: `cout × cin × k × k` as
/// `conv` reads them, a bias maybe; its stride and padding; `up > 1` for a
/// transposed convolution with its stride as its size (`cout` its outputs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvW {
    pub(crate) at: usize,
    pub(crate) bias: Option<usize>,
    pub(crate) cin: usize,
    pub(crate) cout: usize,
    pub(crate) k: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    pub(crate) up: usize,
}

/// A band of a convolution's rows ([`Plan::conv_band`]): its output from
/// row `out_row` of the picture, its input from `in_row`, the input picture
/// `rows_in_all` rows high.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Band {
    pub(crate) out_row: usize,
    pub(crate) in_row: usize,
    pub(crate) rows_in_all: usize,
}

/// What a convolution does on the way in and out.
pub(crate) const ADD: u32 = 2;
pub(crate) const RELU_IN: u32 = 4;
pub(crate) const GELU: u32 = 8;
pub(crate) const RELU: u32 = 16;
pub(crate) const UP_IN: u32 = 32;

/// The resize modes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Resize {
    Bilinear,
    Corners,
    Nearest,
}

/// Activations in place.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Act {
    /// 1 / (1 + e^-x) (the kernel also has GELU and ReLU, which the
    /// convolutions do on their way out here).
    Sigmoid,
}

/// An attention's operands (see `attention.wgsl`).
pub(crate) struct Attend {
    pub(crate) q: Img,
    pub(crate) k: Img,
    pub(crate) v: Img,
    pub(crate) out: Img,
    /// Groups (a window or a view each), queries and keys a group.
    pub(crate) groups: usize,
    pub(crate) nq: usize,
    pub(crate) nk: usize,
    pub(crate) heads: usize,
    pub(crate) d: usize,
    /// Swin's bias: the table's offset, the window's size and shift, and
    /// the windows across and down.
    pub(crate) swin: Option<(usize, usize, usize, usize, usize)>,
}

/// A network's buffers and dispatches.
pub(crate) struct Plan {
    /// Each buffer's length in f32s; buffer 0 is a spare one for bindings
    /// a kernel does not read.
    pub(crate) sizes: Vec<usize>,
    pub(crate) ops: Vec<Op>,
    /// Scratch for instance norms' sums: (partials, stats).
    norm: Option<(usize, usize)>,
    /// Scratch for attention's scores.
    scores: Option<usize>,
    /// Buffers whose contents are no longer needed, for later ones.
    free: Vec<usize>,
}

const SPARE: usize = 0;

/// The pixels an instance norm's chunk spans.
pub(crate) const NORM_CHUNK: usize = 1024;

impl Plan {
    pub(crate) fn new() -> Plan {
        Plan {
            sizes: vec![4],
            ops: Vec::new(),
            norm: None,
            scores: None,
            free: Vec::new(),
        }
    }

    /// A buffer of `len` f32s: a released one where there is one (the
    /// smallest that holds it, else the largest, made to hold it), else a
    /// new one.
    pub(crate) fn buffer(&mut self, len: usize) -> usize {
        let len = len.max(1);
        let fits = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, &b)| self.sizes[b] >= len)
            .min_by_key(|(_, &b)| self.sizes[b])
            .map(|(i, _)| i);
        let pick = fits.or_else(|| {
            self.free
                .iter()
                .enumerate()
                .max_by_key(|(_, &b)| self.sizes[b])
                .map(|(i, _)| i)
        });
        if let Some(i) = pick {
            let buf = self.free.swap_remove(i);
            self.grow(buf, len);
            return buf;
        }
        self.sizes.push(len);
        self.sizes.len() - 1
    }

    /// `bufs`' contents are no longer needed: later [`Plan::buffer`]s may
    /// take them.
    pub(crate) fn release(&mut self, bufs: &[usize]) {
        for &b in bufs {
            assert!(b != SPARE && !self.free.contains(&b));
            self.free.push(b);
        }
    }

    /// Makes buffer `buf` at least `len` long.
    pub(crate) fn grow(&mut self, buf: usize, len: usize) {
        self.sizes[buf] = self.sizes[buf].max(len);
    }

    fn push(
        &mut self,
        kernel: Kernel,
        src: usize,
        src2: usize,
        dst: usize,
        p: [usize; 24],
        groups: [u32; 3],
    ) {
        assert!(
            src != dst && src2 != dst,
            "{kernel:?} reads the buffer it writes"
        );
        self.ops.push(Op {
            kernel,
            buffers: [src, src2, dst],
            params: p,
            groups,
            cost: crate::pace::Cost::UNESTIMATED,
        });
    }

    /// `out` (+)= the convolution of `x` by `w` (`flags`: [`ADD`],
    /// [`RELU_IN`], [`GELU`], [`RELU`], [`UP_IN`]: `x` at half the size it
    /// is taken at). `out`'s size is what the convolution makes.
    pub(crate) fn conv(&mut self, x: Img, w: &ConvW, out: Img, flags: u32) {
        self.conv_band(x, w, out, flags, Band::default());
    }

    /// [`Plan::conv`] for a band of the output (`out` its rows from
    /// `band.out_row`, at `out.off`) from a band of the input (`x` the
    /// picture's rows from `band.in_row`, of a picture `band.rows_in_all`
    /// rows high): rows outside the input band read as 0, as the picture's
    /// edges.
    pub(crate) fn conv_band(&mut self, x: Img, w: &ConvW, out: Img, flags: u32, band: Band) {
        let banded = band != Band::default();
        let (in_h, in_w) = if flags & UP_IN != 0 {
            (2 * x.h, 2 * x.w)
        } else {
            (x.h, x.w)
        };
        assert_eq!(x.c, w.cin, "the input's channels");
        let (out_h, out_w) = if w.up > 1 {
            assert_eq!((out.h, out.w), (in_h * w.up, in_w * w.up));
            (in_h, in_w)
        } else {
            let all = if banded { band.rows_in_all } else { in_h };
            let oh = (all + 2 * w.pad - w.k) / w.stride + 1;
            let ow = (in_w + 2 * w.pad - w.k) / w.stride + 1;
            assert_eq!(out.w, ow, "the output's width");
            assert!(band.out_row + out.h <= oh, "the output's rows");
            assert!(banded || out.h == oh, "the output's size");
            (out.h, ow)
        };
        assert_eq!(out.c, w.cout, "the output's channels");
        assert_eq!(out.views, x.views);
        let n = w.cout * w.up * w.up;
        let rows = x.views * out_h * out_w;
        let mut f = flags;
        if w.bias.is_some() {
            f |= 1;
        }
        let p = params(&[
            rows,
            n,
            w.k * w.k * w.cin,
            x.off,
            w.at,
            w.bias.unwrap_or(0),
            out.off,
            out.ld,
            f as usize,
            w.cin,
            x.ld,
            in_w,
            in_h,
            out_w,
            out_h,
            w.k | (w.stride << 8) | (w.pad << 16) | (w.up << 24),
            band.out_row,
            band.in_row,
        ]);
        assert!(rows.div_ceil(64) <= 65_535, "{rows} rows");
        self.push(
            Kernel::Conv,
            x.buf,
            SPARE,
            out.buf,
            p,
            [n.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1],
        );
    }

    /// `out` (+)= `x`'s rows through a linear layer (`[in][out]` at `w`).
    pub(crate) fn linear(&mut self, x: Img, w: usize, bias: Option<usize>, out: Img, flags: u32) {
        let rows = x.pixels();
        let cw = ConvW {
            at: w,
            bias,
            cin: x.c,
            cout: out.c,
            k: 1,
            stride: 1,
            pad: 0,
            up: 1,
        };
        assert_eq!(rows, out.pixels());
        let x = Img {
            views: 1,
            h: 1,
            w: rows,
            ..x
        };
        let out = Img {
            views: 1,
            h: 1,
            w: rows,
            ..out
        };
        self.conv(x, &cw, out, flags);
    }

    /// LayerNorm of each pixel's channels of `x` into `out` (which may be
    /// `x`: in place), by the weights at `w` and `b`.
    pub(crate) fn layernorm(&mut self, x: Img, out: Img, w: usize, b: usize, eps: f32) {
        assert_eq!(x.pixels(), out.pixels());
        assert_eq!(x.c, out.c);
        self.layernorm_rows(
            (x.buf, x.off, x.ld, 0, 1),
            (out.buf, out.off, out.ld, 0),
            x.pixels(),
            x.c,
            (w, b, eps),
        );
    }

    /// LayerNorm of `rows` rows of `dim` values: row r at off + (r / per)·
    /// outer + (r % per)·inner in each of `from` (buf, off, outer, inner,
    /// per) and `to` (buf, off, outer, inner); the same buffer: in place.
    pub(crate) fn layernorm_rows(
        &mut self,
        from: (usize, usize, usize, usize, usize),
        to: (usize, usize, usize, usize),
        rows: usize,
        dim: usize,
        (w, b, eps): (usize, usize, f32),
    ) {
        let in_place = from.0 == to.0;
        let p = params(&[
            rows,
            dim,
            from.1,
            from.2,
            from.3,
            from.4,
            to.1,
            to.2,
            to.3,
            w,
            b,
            eps.to_bits() as usize,
            usize::from(in_place),
        ]);
        let x = rows.clamp(1, 65_535);
        let src = if in_place { SPARE } else { from.0 };
        self.push(
            Kernel::LayerNorm,
            src,
            SPARE,
            to.0,
            p,
            [x as u32, rows.div_ceil(x) as u32, 1],
        );
    }

    /// An activation over `x` (its channels side by side), in place.
    pub(crate) fn act(&mut self, x: Img, act: Act) {
        assert_eq!(x.ld, x.c, "an activation's image is packed");
        let count = x.len();
        let mode = match act {
            Act::Sigmoid => 2,
        };
        self.push(
            Kernel::Act,
            SPARE,
            SPARE,
            x.buf,
            params(&[count, mode, x.off]),
            spread(count),
        );
    }

    /// An instance norm of `x` (each view's channels over its pixels, eps
    /// 1e-5, no weights), then GELU if `gelu`, in place.
    pub(crate) fn instance_norm(&mut self, x: Img, gelu: bool) {
        let pixels = x.h * x.w;
        let total = pixels.div_ceil(NORM_CHUNK);
        self.norm_stats(x, 0, total);
        self.norm_merge(x.views, pixels, x.c);
        self.norm_apply(x, gelu);
    }

    /// The instance norm's scratch (partials, stats) for `views` of `c`
    /// channels in `chunks` chunks.
    fn norm_scratch(&mut self, views: usize, chunks: usize, c: usize) -> (usize, usize) {
        let partial_len = views * chunks * c * 2;
        let stats_len = views * c * 2;
        let (partials, stats) = match self.norm {
            Some(n) => n,
            None => {
                let n = (self.buffer(partial_len), self.buffer(stats_len));
                self.norm = Some(n);
                n
            }
        };
        self.grow(partials, partial_len);
        self.grow(stats, stats_len);
        (partials, stats)
    }

    /// Each chunk's mean and spread of `x` (packed; a band of a picture
    /// whose chunks before it number `first`, `total` in all), into the
    /// instance norm's partials.
    pub(crate) fn norm_stats(&mut self, x: Img, first: usize, total: usize) {
        assert_eq!(x.ld, x.c, "an instance norm's image is packed");
        let pixels = x.h * x.w;
        assert!(
            first == 0 || pixels.is_multiple_of(NORM_CHUNK),
            "a band of whole chunks"
        );
        let chunks = pixels.div_ceil(NORM_CHUNK);
        let (partials, stats) = self.norm_scratch(x.views, total, x.c);
        let p = params(&[pixels, x.c, x.views, NORM_CHUNK, x.off, first, total]);
        let groups = [(x.views * chunks) as u32, x.c.div_ceil(64) as u32, 1];
        self.push(Kernel::ChanSums, x.buf, stats, partials, p, groups);
    }

    /// The chunks' partials merged into each channel's mean and 1/σ (eps
    /// 1e-5), for `views` of `pixels` (in all) and `c` channels.
    pub(crate) fn norm_merge(&mut self, views: usize, pixels: usize, c: usize) {
        let (partials, stats) = self.norm_scratch(views, pixels.div_ceil(NORM_CHUNK), c);
        let eps = 1e-5f32.to_bits() as usize;
        let p = params(&[pixels, c, views, NORM_CHUNK, eps]);
        self.push(
            Kernel::ChanReduce,
            partials,
            SPARE,
            stats,
            p,
            spread(views * c),
        );
    }

    /// `x` (packed) normed by the merged stats, then GELU if `gelu`, in
    /// place.
    pub(crate) fn norm_apply(&mut self, x: Img, gelu: bool) {
        assert_eq!(x.ld, x.c, "an instance norm's image is packed");
        let (_, stats) = self.norm.expect("stats merged first");
        let p = params(&[x.h * x.w, x.c, x.views, 0, x.off, usize::from(gelu)]);
        self.push(Kernel::ChanApply, stats, SPARE, x.buf, p, spread(x.len()));
    }

    /// `out` (+)= `x` resized to `out`'s size.
    pub(crate) fn resize(&mut self, x: Img, out: Img, mode: Resize, add: bool) {
        assert_eq!((x.views, x.c), (out.views, out.c));
        let m = match mode {
            Resize::Bilinear => 0,
            Resize::Corners => 1,
            Resize::Nearest => 2,
        };
        let p = params(&[
            x.views,
            x.c,
            x.h,
            x.w,
            out.h,
            out.w,
            x.off,
            out.off,
            m,
            usize::from(add),
            x.ld,
            out.ld,
        ]);
        self.push(
            Kernel::Resize,
            x.buf,
            SPARE,
            out.buf,
            p,
            spread(out.pixels() * out.c),
        );
    }

    /// `out` = `x` pooled to `out`'s size (adaptive average pooling).
    pub(crate) fn pool(&mut self, x: Img, out: Img) {
        assert_eq!((x.views, x.c), (out.views, out.c));
        let p = params(&[
            x.views, x.c, x.h, x.w, out.h, out.w, x.off, out.off, 0, 0, x.ld, out.ld,
        ]);
        self.push(
            Kernel::Pool,
            x.buf,
            SPARE,
            out.buf,
            p,
            spread(out.pixels() * out.c),
        );
    }

    /// A strided copy (see `copy` in `ops.wgsl`): counts `n`, the source
    /// (buffer, offset, three strides), the destination likewise; added if
    /// `add`.
    pub(crate) fn copy(
        &mut self,
        n: [usize; 4],
        from: (usize, usize, [usize; 3]),
        to: (usize, usize, [usize; 3]),
        add: bool,
    ) {
        let p = params(&[
            n[0],
            n[1],
            n[2],
            n[3],
            from.1,
            from.2[0],
            from.2[1],
            from.2[2],
            to.1,
            to.2[0],
            to.2[1],
            to.2[2],
            usize::from(add),
        ]);
        self.push(
            Kernel::Copy,
            from.0,
            SPARE,
            to.0,
            p,
            spread(n.iter().product()),
        );
    }

    /// `out` (+)= `x`, image to image (the same shape).
    pub(crate) fn copy_img(&mut self, x: Img, out: Img, add: bool) {
        assert_eq!((x.pixels(), x.c), (out.pixels(), out.c));
        self.copy(
            [1, 1, x.pixels(), x.c],
            (x.buf, x.off, [0, 0, x.ld]),
            (out.buf, out.off, [0, 0, out.ld]),
            add,
        );
    }

    /// The quarters of `x` (views of 2h × 2w) as four views each (`out`,
    /// views · 4 of h × w; view 4·v + 2·qy + qx), or back (`back`).
    pub(crate) fn quarters(&mut self, x: Img, out: Img, back: bool) {
        let (big, small) = if back { (out, x) } else { (x, out) };
        let (h, w, c) = (small.h, small.w, small.c);
        assert_eq!((big.h, big.w, big.c), (2 * h, 2 * w, c));
        for v in 0..big.views {
            for q in 0..4 {
                let (qy, qx) = (q / 2, q % 2);
                let b_off = big.off + (v * big.h * big.w + qy * h * big.w + qx * w) * big.ld;
                let s_off = small.off + (4 * v + q) * h * w * small.ld;
                let b = (big.buf, b_off, [0, big.w * big.ld, big.ld]);
                let s = (small.buf, s_off, [0, w * small.ld, small.ld]);
                let (from, to) = if back { (s, b) } else { (b, s) };
                self.copy([1, h, w, c], from, to, false);
            }
        }
    }

    /// Swin's windows of `x` (packed) into `out` (views · windows of size²
    /// tokens), padded to whole windows and rolled back by `shift`; or
    /// (`back`) the windows in `x` added back into the image `out`.
    pub(crate) fn windows(&mut self, x: Img, out: Img, size: usize, shift: usize, back: bool) {
        let image = if back { out } else { x };
        assert_eq!(image.ld, image.c);
        let (hp, wp) = (
            image.h.next_multiple_of(size),
            image.w.next_multiple_of(size),
        );
        let p = params(&[image.views, image.h, image.w, image.c, size, shift, hp, wp]);
        // Both kernels index the image and the windows from their starts.
        assert_eq!((x.off, out.off), (0, 0));
        if back {
            self.push(
                Kernel::WinBack,
                x.buf,
                SPARE,
                out.buf,
                p,
                spread(image.len()),
            );
        } else {
            self.push(
                Kernel::WinPart,
                x.buf,
                SPARE,
                out.buf,
                p,
                spread(image.views * hp * wp * image.c),
            );
        }
    }

    /// 2-D rotary positions on the queries and keys (heads of 64) of the
    /// qkv rows `qkv` (`tokens` of 3·C, C = 64·heads), from the table at
    /// `table`, a grid `cols` wide; every patch at (1, 1) if `flat`.
    pub(crate) fn rope(&mut self, qkv: Img, heads: usize, cols: usize, flat: bool, table: usize) {
        assert_eq!(qkv.off, 0);
        let tokens = qkv.pixels();
        let p = params(&[
            tokens,
            heads,
            qkv.ld,
            cols,
            heads * 64,
            usize::from(flat),
            table,
        ]);
        self.push(
            Kernel::Rope,
            SPARE,
            SPARE,
            qkv.buf,
            p,
            spread(tokens * heads * 64),
        );
    }

    /// `count` values of the table at `table` into `buf` from `off`
    /// (repeating every `period`), set or added.
    pub(crate) fn table(
        &mut self,
        buf: usize,
        off: usize,
        count: usize,
        table: usize,
        period: usize,
        add: bool,
    ) {
        let p = params(&[count, off, table, usize::from(add), period]);
        self.push(Kernel::Table, SPARE, SPARE, buf, p, spread(count));
    }

    /// Four views `x` (packed, the quarters of a picture) each times the
    /// one-channel map `map` (h × w over the whole picture, at half its
    /// resolution).
    pub(crate) fn mulmap(&mut self, x: Img, map: Img) {
        assert_eq!((x.views, x.ld), (4, x.c));
        assert_eq!((map.h, map.w, map.c), (x.h, x.w, 1));
        let p = params(&[x.h, x.w, x.c, map.off, x.off]);
        self.push(Kernel::MulMap, map.buf, SPARE, x.buf, p, spread(x.len()));
    }

    /// A batched matrix multiply (see `bmm` in `attention.wgsl`): `z`
    /// batches of `heads`; A (src) and B (src2) and C (dst) each a buffer,
    /// an offset and their strides.
    #[allow(clippy::too_many_arguments)]
    fn bmm(
        &mut self,
        (m, n, k): (usize, usize, usize),
        (z, heads): (usize, usize),
        a: (usize, usize, [usize; 3]),
        b: (usize, usize, [usize; 4]),
        c: (usize, usize, [usize; 3]),
        scale: f32,
    ) {
        let p = params(&[
            m,
            n,
            k,
            heads,
            a.1,
            a.2[0],
            a.2[1],
            a.2[2],
            b.1,
            b.2[0],
            b.2[1],
            b.2[2],
            b.2[3],
            c.1,
            c.2[0],
            c.2[1],
            c.2[2],
            scale.to_bits() as usize,
        ]);
        assert!(z <= 65_535 && m.div_ceil(64) <= 65_535);
        self.push(
            Kernel::Bmm,
            a.0,
            b.0,
            c.0,
            p,
            [n.div_ceil(64) as u32, m.div_ceil(64) as u32, z as u32],
        );
    }

    /// Attention (see [`Attend`]): the scores of a chunk of groups (or of a
    /// group's queries) at a time, at most [`SCORES`] of them.
    pub(crate) fn attend(&mut self, a: &Attend) {
        /// The most scores held at once.
        const SCORES: usize = 1 << 24;
        let (table, s, shift, nwx, nwy) = a.swin.unwrap_or((0, 0, 0, 0, 0));
        let per_group = a.heads * a.nq * a.nk;
        let (group_chunk, query_chunk) = if per_group <= SCORES {
            ((SCORES / per_group).min(a.groups).max(1), a.nq)
        } else {
            assert!(a.swin.is_none(), "Swin's windows are small");
            (1, (SCORES / (a.heads * a.nk)).max(64) / 64 * 64)
        };
        let len = group_chunk * a.heads * query_chunk.min(a.nq) * a.nk;
        let scores = match self.scores {
            Some(buf) => buf,
            None => {
                let buf = self.buffer(len);
                self.scores = Some(buf);
                buf
            }
        };
        self.grow(scores, len);
        let scale = (a.d as f32).powf(-0.5);
        let mut g0 = 0;
        while g0 < a.groups {
            let gs = group_chunk.min(a.groups - g0);
            let mut q0 = 0;
            while q0 < a.nq {
                let qs = query_chunk.min(a.nq - q0);
                let z = gs * a.heads;
                let s_strides = [a.heads * qs * a.nk, qs * a.nk, a.nk];
                // Scores: q · kᵀ · scale.
                self.bmm(
                    (qs, a.nk, a.d),
                    (z, a.heads),
                    (
                        a.q.buf,
                        a.q.off + (g0 * a.nq + q0) * a.q.ld,
                        [a.nq * a.q.ld, a.d, a.q.ld],
                    ),
                    (
                        a.k.buf,
                        a.k.off + g0 * a.nk * a.k.ld,
                        [a.nk * a.k.ld, a.d, 1, a.k.ld],
                    ),
                    (scores, 0, s_strides),
                    scale,
                );
                let rows = z * qs;
                let x = rows.clamp(1, 65_535);
                let p = params(&[
                    rows,
                    a.nk,
                    0,
                    qs,
                    a.heads,
                    usize::from(a.swin.is_some()),
                    s | (shift << 8) | (nwx << 16) | (nwy << 24),
                    table,
                    g0,
                ]);
                self.push(
                    Kernel::Softmax,
                    SPARE,
                    SPARE,
                    scores,
                    p,
                    [x as u32, rows.div_ceil(x) as u32, 1],
                );
                // Out: scores · v.
                self.bmm(
                    (qs, a.d, a.nk),
                    (z, a.heads),
                    (scores, 0, s_strides),
                    (
                        a.v.buf,
                        a.v.off + g0 * a.nk * a.v.ld,
                        [a.nk * a.v.ld, a.d, a.v.ld, 1],
                    ),
                    (
                        a.out.buf,
                        a.out.off + (g0 * a.nq + q0) * a.out.ld,
                        [a.nq * a.out.ld, a.d, a.out.ld],
                    ),
                    1.0,
                );
                q0 += qs;
            }
            g0 += gs;
        }
    }
}
