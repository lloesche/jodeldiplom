//! The player's icon, drawn from its shapes at any size: a dark rounded
//! square with a spectrum, as the player shows one, from the core's colour
//! to the render's. Large sizes draw the curve dotted, as the player does;
//! small ones as a line, which stays legible at 16 pixels.
//!
//! Two styles: [`Style::Windows`] fills the canvas; [`Style::MacOs`] sits
//! on Apple's icon grid (the square 824 of 1024 pixels, centred) with a
//! soft shadow under it, as macOS icons do.
//!
//! The window sets it at run time (on macOS, the Dock icon);
//! `assets/jodeldiplom.ico` (Windows executable) and
//! `assets/jodeldiplom.icns` (macOS app bundle) hold the sizes each system
//! asks for, written by `cargo run -p jodeldiplom --example icon` and
//! checked by the tests.

use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

/// The sizes in `assets/jodeldiplom.ico`.
pub const ICO_SIZES: [u32; 7] = [16, 20, 24, 32, 48, 64, 256];

/// The entries in `assets/jodeldiplom.icns`: type and pixel size (the
/// `ic11`..`ic14` ones are the Retina versions of 16, 32, 128 and 256).
pub const ICNS_ENTRIES: [(&[u8; 4], u32); 11] = [
    (b"icp4", 16),
    (b"icp5", 32),
    (b"icp6", 64),
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic13", 256),
    (b"ic14", 512),
];

/// How the icon sits on its canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// The square fills the canvas (Windows, Linux window icons).
    Windows,
    /// The square on Apple's grid, with a shadow (macOS).
    MacOs,
}

impl Style {
    /// The square's side relative to the Windows one: Apple's 824 of 1024
    /// against 96 % of the canvas.
    fn scale(self) -> f32 {
        match self {
            Style::Windows => 1.0,
            Style::MacOs => (824.0 / 1024.0) / (1.0 - 2.0 * MARGIN),
        }
    }
}

/// The background: the player's background, a little lighter at the top.
const BG_TOP: [f32; 3] = [38.0, 38.0, 48.0];
const BG_BOTTOM: [f32; 3] = [17.0, 17.0, 21.0];
/// The square's edge (the player's border colour).
const EDGE: [f32; 3] = [64.0, 64.0, 74.0];
/// The curve: the core's colour at the left, the render's at the right.
const LEFT: [f32; 3] = [0.0, 205.0, 225.0];
const RIGHT: [f32; 3] = [240.0, 90.0, 170.0];
/// The faint line under the curve (the player's grid colour).
const GRID: [f32; 3] = [78.0, 78.0, 92.0];

/// The icon at `size` x `size` pixels in `style`, as RGBA rows (straight
/// alpha).
pub fn rgba(style: Style, size: u32) -> Vec<u8> {
    let shape = Shape::new(style, size);
    // Subsamples per pixel side: fewer where pixels are small against the
    // shapes.
    let sub: u32 = if size > 256 { 2 } else { 4 };
    let n = (sub * sub) as f32;
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            // Premultiplied sums over the subsamples.
            let mut sum = [0.0f32; 4];
            for sy in 0..sub {
                for sx in 0..sub {
                    let x = (px as f32 + (sx as f32 + 0.5) / sub as f32) / size as f32;
                    let y = (py as f32 + (sy as f32 + 0.5) / sub as f32) / size as f32;
                    let (c, a) = shape.sample(x, y);
                    for i in 0..3 {
                        sum[i] += c[i] * a;
                    }
                    sum[3] += a;
                }
            }
            let alpha = sum[3] / n;
            for value in &sum[..3] {
                let c = if sum[3] > 0.0 { value / sum[3] } else { 0.0 };
                out.push(c.round().clamp(0.0, 255.0) as u8);
            }
            out.push((alpha * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// The icon at `size` pixels in `style`, as a PNG file.
pub fn png(style: Style, size: u32) -> Vec<u8> {
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(&rgba(style, size), size, size, ExtendedColorType::Rgba8)
        .expect("encoding into memory does not fail");
    out
}

/// The Windows icon file: every size in [`ICO_SIZES`], each as a 32-bit
/// bitmap with its alpha (what every Windows version reads).
pub fn ico() -> Vec<u8> {
    let images: Vec<(u32, Vec<u8>)> = ICO_SIZES.iter().map(|&s| (s, dib(s))).collect();
    let mut out = Vec::new();
    // ICONDIR: reserved, type 1 (icon), count.
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * images.len() as u32;
    for (size, data) in &images {
        // ICONDIRENTRY: width and height (0 is 256), colours, reserved,
        // planes, bits per pixel, bytes, offset.
        let side = if *size >= 256 { 0 } else { *size as u8 };
        out.extend_from_slice(&[side, side, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += data.len() as u32;
    }
    for (_, data) in &images {
        out.extend_from_slice(data);
    }
    out
}

/// The macOS icon file: every entry in [`ICNS_ENTRIES`] as a PNG (each
/// size drawn once).
pub fn icns() -> Vec<u8> {
    let mut drawn: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut body = Vec::new();
    for (kind, size) in ICNS_ENTRIES {
        if !drawn.iter().any(|(s, _)| *s == size) {
            drawn.push((size, png(Style::MacOs, size)));
        }
        let data = &drawn.iter().find(|(s, _)| *s == size).unwrap().1;
        body.extend_from_slice(kind);
        body.extend_from_slice(&(data.len() as u32 + 8).to_be_bytes());
        body.extend_from_slice(data);
    }
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(b"icns");
    out.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// One Windows icon image as a device-independent bitmap: the header, the
/// BGRA rows bottom up, then the 1-bit mask (all shown; the alpha decides).
fn dib(size: u32) -> Vec<u8> {
    let rgba = rgba(Style::Windows, size);
    let mask_row = size.div_ceil(32) * 4;
    let mut out = Vec::new();
    // BITMAPINFOHEADER; the height counts the colour rows and the mask.
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(size as i32).to_le_bytes());
    out.extend_from_slice(&(2 * size as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&(size * size * 4 + mask_row * size).to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for y in (0..size).rev() {
        for x in 0..size {
            let i = ((y * size + x) * 4) as usize;
            out.extend_from_slice(&[rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]]);
        }
    }
    out.extend(std::iter::repeat_n(0u8, (mask_row * size) as usize));
    out
}

/// The shapes, in units of the Windows icon's side (0 to 1, y down); the
/// macOS style maps its canvas onto these.
struct Shape {
    style: Style,
    /// The curve as points from left to right.
    curve: Vec<(f32, f32)>,
    /// Dots along the curve (large sizes), else none: a line.
    dots: Vec<(f32, f32)>,
    /// The line's half width, or the dots' radius.
    half: f32,
    /// One pixel, in units.
    pixel: f32,
}

/// The square's margin and corner radius, and the curve's span.
const MARGIN: f32 = 0.02;
const RADIUS: f32 = 0.22;
const LEFT_X: f32 = 0.17;
const RIGHT_X: f32 = 0.83;
/// The line under the curve.
const FLOOR_Y: f32 = 0.76;
/// The macOS shadow: how far down it falls, how soft it is and how dark.
const SHADOW_DROP: f32 = 0.018;
const SHADOW_BLUR: f32 = 0.05;
const SHADOW_ALPHA: f32 = 0.35;

impl Shape {
    fn new(style: Style, size: u32) -> Shape {
        // The square's size in pixels decides the detail.
        let square = size as f32 * style.scale();
        let pixel = 1.0 / square;
        let curve: Vec<(f32, f32)> = (0..=240)
            .map(|i| {
                let t = i as f32 / 240.0;
                (LEFT_X + t * (RIGHT_X - LEFT_X), level_y(t))
            })
            .collect();
        let (dots, half) = if square >= 46.0 {
            (evenly_along(&curve, 15), 0.027)
        } else {
            // At least about a pixel and a half wide.
            (Vec::new(), (0.05f32).max(0.75 * pixel))
        };
        Shape {
            style,
            curve,
            dots,
            half,
            pixel,
        }
    }

    /// The colour and coverage at (x, y) on the canvas.
    fn sample(&self, x: f32, y: f32) -> ([f32; 3], f32) {
        let scale = self.style.scale();
        let (x, y) = (0.5 + (x - 0.5) / scale, 0.5 + (y - 0.5) / scale);
        let inside = rounded_square(x, y);
        if inside <= 0.0 {
            return match self.style {
                Style::Windows => ([0.0; 3], 0.0),
                Style::MacOs => {
                    // Under the square, a little lower: a soft shadow.
                    let out = -rounded_square(x, y - SHADOW_DROP);
                    let fade = (1.0 - out / SHADOW_BLUR).clamp(0.0, 1.0);
                    ([0.0; 3], SHADOW_ALPHA * fade * fade)
                }
            };
        }
        // The background, with a one-pixel edge.
        let edge_width = self.pixel.max(0.012);
        let mut colour = mix(BG_TOP, BG_BOTTOM, y);
        if inside < edge_width {
            colour = mix(colour, EDGE, 0.8);
        }
        // The floor line, faint.
        if (y - FLOOR_Y).abs() <= self.pixel.max(0.008) * 0.6 && (LEFT_X..=RIGHT_X).contains(&x) {
            colour = mix(colour, GRID, 0.7);
        }
        let along = ((x - LEFT_X) / (RIGHT_X - LEFT_X)).clamp(0.0, 1.0);
        let curve_colour = mix(LEFT, RIGHT, along);
        // Under the curve: a glow that fades towards the floor.
        if (LEFT_X..=RIGHT_X).contains(&x) {
            let top = level_y(along);
            if y > top && y < FLOOR_Y {
                let fade = 1.0 - (y - top) / (FLOOR_Y - top);
                colour = mix(colour, curve_colour, 0.28 * fade);
            }
        }
        // The curve itself.
        let on_curve = if self.dots.is_empty() {
            self.near_line(x, y)
        } else {
            self.dots
                .iter()
                .any(|&(cx, cy)| (x - cx).powi(2) + (y - cy).powi(2) <= self.half.powi(2))
        };
        if on_curve {
            colour = curve_colour;
        }
        (colour, 1.0)
    }

    /// Whether (x, y) is within the line's half width of the curve.
    fn near_line(&self, x: f32, y: f32) -> bool {
        let reach = self.half;
        self.curve.windows(2).any(|w| {
            let ((ax, ay), (bx, by)) = (w[0], w[1]);
            if x < ax.min(bx) - reach || x > ax.max(bx) + reach {
                return false;
            }
            let (dx, dy) = (bx - ax, by - ay);
            let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let (px, py) = (ax + t * dx, ay + t * dy);
            (x - px).powi(2) + (y - py).powi(2) <= reach * reach
        })
    }
}

/// `count` points spaced evenly along `curve` (by its length), the first
/// and the last at its ends.
fn evenly_along(curve: &[(f32, f32)], count: usize) -> Vec<(f32, f32)> {
    let mut lengths = vec![0.0f32];
    for w in curve.windows(2) {
        let step = ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
        lengths.push(lengths.last().unwrap() + step);
    }
    let total = *lengths.last().unwrap();
    (0..count)
        .map(|i| {
            let want = total * i as f32 / (count - 1) as f32;
            let j = lengths
                .windows(2)
                .position(|l| l[1] >= want)
                .unwrap_or(curve.len() - 2);
            let span = (lengths[j + 1] - lengths[j]).max(f32::EPSILON);
            let t = ((want - lengths[j]) / span).clamp(0.0, 1.0);
            let (a, b) = (curve[j], curve[j + 1]);
            (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
        })
        .collect()
}

/// The curve's height at `t` (0 at the left, 1 at the right) as a y: one
/// smooth line, a level that falls gently, rolls off past two thirds, as
/// a band edge does, and lands on a flat noise floor.
fn level_y(t: f32) -> f32 {
    let level = 0.60 - 0.10 * t * t;
    let rolloff = 1.0 / (1.0 + ((t - 0.64) / 0.07).exp());
    let low = 0.24;
    let height = low + (level - low) * rolloff;
    FLOOR_Y - height * 0.85
}

/// How far (x, y) is inside the rounded square (0 or less: outside; the
/// distance outside, negated).
fn rounded_square(x: f32, y: f32) -> f32 {
    let (lo, hi) = (MARGIN + RADIUS, 1.0 - MARGIN - RADIUS);
    let qx = x.clamp(lo, hi);
    let qy = y.clamp(lo, hi);
    let corner = ((x - qx).powi(2) + (y - qy).powi(2)).sqrt();
    if corner > 0.0 {
        RADIUS - corner
    } else {
        (x - MARGIN)
            .min(1.0 - MARGIN - x)
            .min(y - MARGIN)
            .min(1.0 - MARGIN - y)
    }
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
