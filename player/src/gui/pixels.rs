//! The window's pixels as words: a colour packed into `0x00RRGGBB` (the
//! window buffer's own layout), what shows through it in the top byte, the
//! mixing of colours over them, rects of pixels, and the targets painted
//! into ([`Target`], and a rect of one, [`Pixels`]).

/// A pixmap word is premultiplied over what the GPU draws under the
/// pixmap (Karaoke mode's backdrop, [`super::backdrop`]): its colour is
/// the pixmap's own share, its top byte how much of what is under still
/// shows (255: all of it, the colour then black). A colour packed
/// ([`pack`]) covers what is under; drawing over a word goes through
/// [`over`] and [`dim`], which keep the two together. This is a word
/// that is all see-through.
pub const SEE_THROUGH: u32 = 0xff00_0000;

/// `word` with `colour` over it by `a` (0 to 1): the colour mixed into its
/// own as [`mix`] does, and what shows through it less by as much.
pub fn over(word: u32, colour: [u8; 3], a: f32) -> u32 {
    let a = a.clamp(0.0, 1.0);
    let through = ((word >> 24) as f32 * (1.0 - a)).round() as u32;
    pack(mix(unpack(word), colour, a)) | through << 24
}

/// `word` darkened to `keep` (0 to 1) of itself: its own colour and what
/// shows through it alike.
pub fn dim(word: u32, keep: f32) -> u32 {
    let keep = keep.clamp(0.0, 1.0);
    let scale = |v: u32| (v as f32 * keep).round() as u32;
    let [r, g, b] = unpack(word).map(|v| scale(u32::from(v)) as u8);
    pack([r, g, b]) | scale(word >> 24) << 24
}

/// Word `a` towards word `b` by `t` (0 to 1): their colours and what shows
/// through them alike (so `b` drawn over the ground faded in by `t`).
pub fn between(a: u32, b: u32, t: f32) -> u32 {
    let through = |w: u32| [(w >> 24) as u8; 3];
    pack(mix(unpack(a), unpack(b), t)) | u32::from(mix(through(a), through(b), t)[0]) << 24
}

/// A colour as a pixmap word, `0x00RRGGBB`.
pub fn pack([r, g, b]: [u8; 3]) -> u32 {
    u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
}

/// A pixmap word's colour.
pub fn unpack(word: u32) -> [u8; 3] {
    [(word >> 16) as u8, (word >> 8) as u8, word as u8]
}

/// `a` towards `b` by `t` (0 to 1).
pub fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (f32::from(a[i]) * (1.0 - t) + f32::from(b[i]) * t).round() as u8)
}

/// A rect of pixels: x, y, width, height.
pub type PixelRect = (usize, usize, usize, usize);

/// The rect around `a` (if any) and `b`.
pub fn union(a: Option<PixelRect>, b: PixelRect) -> PixelRect {
    match a {
        None => b,
        Some((x, y, w, h)) => {
            let (x0, y0) = (x.min(b.0), y.min(b.1));
            let (x1, y1) = ((x + w).max(b.0 + b.2), (y + h).max(b.1 + b.3));
            (x0, y0, x1 - x0, y1 - y0)
        }
    }
}

/// Where to draw into: the pixmap, its width and height.
pub struct Target<'a> {
    /// `0x00RRGGBB` words, row by row.
    pub pixels: &'a mut [u32],
    /// Its width.
    pub width: usize,
    /// Its height.
    pub height: usize,
}

impl Target<'_> {
    /// The pixels from (`x0`, `y0`) up to (`x1`, `y1`) in one colour,
    /// cut to the target.
    pub fn fill(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]) {
        let (x1, word) = (x1.min(self.width), pack(color));
        if x0 >= x1 {
            return;
        }
        for y in y0..y1.min(self.height) {
            let row = y * self.width;
            self.pixels[row + x0..row + x1].fill(word);
        }
    }

    /// `color` over the pixel at (`x`, `y`) by `alpha` ([`over`]), where it
    /// is in the target.
    pub fn put_over(&mut self, x: usize, y: usize, color: [u8; 3], alpha: u8) {
        if x < self.width && y < self.height {
            let i = y * self.width + x;
            self.pixels[i] = over(self.pixels[i], color, f32::from(alpha) / 255.0);
        }
    }

    /// The pixels of `rect` alone, in its own coordinates.
    pub fn sub(&mut self, (x0, y0, w, h): PixelRect) -> Pixels<'_> {
        Pixels {
            data: self.pixels,
            stride: self.width,
            x0,
            y0,
            w,
            h,
        }
    }
}

/// The pixels of one rect of a target, clipped to it.
pub struct Pixels<'a> {
    data: &'a mut [u32],
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

impl Pixels<'_> {
    fn index(&self, x: usize, y: usize) -> Option<usize> {
        let i = (self.y0 + y) * self.stride + self.x0 + x;
        (x < self.w && y < self.h && i < self.data.len()).then_some(i)
    }

    /// The pixel at (`x`, `y`) in `c`.
    pub fn set(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if let Some(i) = self.index(x, y) {
            self.data[i] = pack(c);
        }
    }

    /// The pixel at (`x`, `y`) towards `c` by `t` ([`mix`]).
    pub fn blend(&mut self, x: usize, y: usize, c: [u8; 3], t: f32) {
        if let Some(i) = self.index(x, y) {
            self.data[i] = pack(mix(unpack(self.data[i]), c, t));
        }
    }

    /// The rows of the rect inside the target, as (first index, width).
    fn rows(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let w = self.w.min(self.stride.saturating_sub(self.x0));
        let (stride, x0, y0) = (self.stride, self.x0, self.y0);
        let h = self
            .h
            .min((self.data.len() / stride.max(1)).saturating_sub(y0));
        (0..h).map(move |y| ((y0 + y) * stride + x0, w))
    }

    /// The whole rect in one colour.
    pub fn fill(&mut self, c: [u8; 3]) {
        let word = pack(c);
        for (at, w) in self.rows() {
            self.data[at..at + w].fill(word);
        }
    }

    /// The rect's pixels moved `k` columns left (right where negative); the
    /// columns that leaves behind keep what they had.
    pub fn shift(&mut self, k: i64) {
        let n = k.unsigned_abs() as usize;
        for (at, w) in self.rows() {
            if n >= w {
                continue;
            }
            let row = &mut self.data[at..at + w];
            if k > 0 {
                row.copy_within(n.., 0);
            } else {
                row.copy_within(..w - n, n);
            }
        }
    }

    /// Column `x`, each pixel the colour `color_at` gives for its row.
    pub fn column(&mut self, x: usize, mut color_at: impl FnMut(usize) -> u32) {
        if x >= self.w || self.x0 + x >= self.stride {
            return;
        }
        for (y, (at, _)) in self.rows().enumerate() {
            self.data[at + x] = color_at(y);
        }
    }

    /// A square of `side` pixels from (`x`, `y`) in `c`.
    pub fn square(&mut self, x: usize, y: usize, side: usize, c: [u8; 3]) {
        for dy in 0..side {
            for dx in 0..side {
                self.set(x + dx, y + dy, c);
            }
        }
    }

    /// A round dot centred at (`cx`, `cy`), its edge smoothed.
    pub fn disc(&mut self, cx: f64, cy: f64, radius: f64, c: [u8; 3]) {
        let reach = radius + 1.0;
        let (x_from, x_to) = ((cx - reach).floor().max(0.0), (cx + reach).ceil());
        let (y_from, y_to) = ((cy - reach).floor().max(0.0), (cy + reach).ceil());
        for y in y_from as usize..y_to as usize {
            for x in x_from as usize..x_to as usize {
                let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
                let cover = (radius + 0.5 - d).clamp(0.0, 1.0) as f32;
                if cover > 0.0 {
                    self.blend(x, y, c, cover);
                }
            }
        }
    }
}
