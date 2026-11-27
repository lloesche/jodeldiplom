//! The window's pixels as words: a colour packed into `0x00RRGGBB` (the
//! window buffer's own layout), the mixing of colours over them, rects of
//! pixels, and the targets painted into ([`Target`], and a rect of one,
//! [`Pixels`]).

/// `word` with `colour` over it by `a` (0 to 1), as [`mix`] does.
pub fn over(word: u32, colour: [u8; 3], a: f32) -> u32 {
    pack(mix(unpack(word), colour, a))
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

/// A window's pixels as its points land on them: this many pixels a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap(pub f32);

impl Snap {
    /// `v` points on the window's pixels.
    pub fn snap(self, v: f32) -> f32 {
        (v * self.0).round() / self.0
    }

    /// `rect` with its edges on the window's pixels (none less than empty).
    pub fn rect(self, rect: iced::Rectangle) -> iced::Rectangle {
        let (x, y) = (self.snap(rect.x), self.snap(rect.y));
        iced::Rectangle::new(
            iced::Point::new(x, y),
            iced::Size::new(
                (self.snap(rect.x + rect.width) - x).max(0.0),
                (self.snap(rect.y + rect.height) - y).max(0.0),
            ),
        )
    }

    /// `rect` (in points) in the window's pixels.
    pub fn pixels(self, rect: iced::Rectangle) -> PixelRect {
        let px = |v: f32| (v * self.0).round().max(0.0) as usize;
        (
            px(rect.x),
            px(rect.y),
            px(rect.x + rect.width) - px(rect.x),
            px(rect.y + rect.height) - px(rect.y),
        )
    }
}

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
