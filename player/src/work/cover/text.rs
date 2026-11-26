//! Where a cover's lettering is: our own inference of PP-OCRv5 mobile det,
//! PaddleOCR's text detector (by the PaddlePaddle authors, Apache-2.0, see
//! `models/LICENSE-pp-ocr`), folded for inference by
//! `tools/convert_text_det.py` and built in.
//!
//! The cover, at 960 × 960 (as the detector's own preprocessing sizes it,
//! its longer side 960) in blue, green, red order and normalised as the
//! authors do, goes through a PP-LCNetV3 backbone (re-parameterised
//! depthwise and pointwise convolutions with the hard swish, squeeze-
//! excitation in its last stage), a residual squeeze-excitation FPN, and
//! DB's head (a convolution and two transposed ones back to the full size,
//! the sigmoid): each pixel's probability of being text ([`TextMap`]).

use std::sync::OnceLock;

use image::RgbImage;
use jodel_rhythm::nn::weights::Weights;

use super::nets::{hard_swish, Conv, Excite, Map, Up2};

/// The weights, built in (`tools/convert_text_det.py`).
static WEIGHTS: &[u8] = include_bytes!("../../../models/text-det.bin");

/// The side the cover is looked at.
pub const SIDE: usize = 960;

/// The backbone's stages: each layer's stride (the kernel and whether it
/// excites are in its weights).
const STRIDES: [&[usize]; 5] = [&[1], &[2, 1], &[2, 1], &[2, 1, 1, 1, 1], &[2, 1, 1, 1]];

/// Where a cover's lettering is: a probability per pixel of the cover at
/// `side` × `side`.
#[derive(Clone, Debug, PartialEq)]
pub struct TextMap {
    /// Its size (square).
    pub side: usize,
    /// Each pixel's probability of being text, row by row.
    pub prob: Vec<f32>,
}

impl TextMap {
    /// No lettering anywhere.
    pub fn none() -> TextMap {
        TextMap {
            side: 1,
            prob: vec![0.0],
        }
    }
}

/// A convolution with its activation's scale and shift after the hard
/// swish (none on a stride-2 depthwise one).
struct Act {
    conv: Conv,
    act: Option<[f32; 2]>,
}

impl Act {
    fn load(w: &mut Weights, name: &str, stride: usize) -> Result<Act, String> {
        let conv = Conv::load(w, name, stride)?;
        let act = match w.get(&format!("{name}.act")) {
            Some(_) => {
                let a = w.take(&format!("{name}.act"), &[2])?;
                Some([a[0], a[1]])
            }
            None => None,
        };
        Ok(Act { conv, act })
    }

    fn apply(&self, x: &Map, threads: usize) -> Map {
        let y = self.conv.apply(x, threads);
        match self.act {
            Some([s, b]) => y.each(|v| s * hard_swish(v) + b),
            None => y,
        }
    }
}

/// A backbone layer: depthwise, squeeze-excitation where it has one,
/// pointwise.
struct Layer {
    dw: Act,
    se: Option<Excite>,
    pw: Act,
}

/// A convolution and a residual squeeze-excitation (the neck's).
struct Rse {
    conv: Conv,
    se: Excite,
}

impl Rse {
    fn load(w: &mut Weights, name: &str) -> Result<Rse, String> {
        Ok(Rse {
            conv: Conv::load(w, name, 1)?,
            se: Excite::load(w, &format!("{name}.se1"), &format!("{name}.se2"))?,
        })
    }

    fn apply(&self, x: &Map, threads: usize) -> Map {
        let y = self.conv.apply(x, threads);
        let s = self.se.scales(&y, |v| (0.2 * v + 0.5).clamp(0.0, 1.0));
        let excited = y.clone().scaled(&s);
        y.plus(&excited)
    }
}

/// The text detector.
pub struct TextNet {
    stem: Conv,
    stages: Vec<Vec<Layer>>,
    lateral: Vec<Conv>,
    insert: Vec<Rse>,
    input: Vec<Rse>,
    down: Conv,
    up: Up2,
    last: Up2,
}

impl TextNet {
    /// The network of `w`.
    pub fn load(mut w: Weights) -> Result<TextNet, String> {
        let w = &mut w;
        let stem = Conv::load(w, "stem", 2)?;
        let mut stages = Vec::new();
        for (s, strides) in STRIDES.iter().enumerate() {
            let mut layers = Vec::new();
            for (l, &stride) in strides.iter().enumerate() {
                let n = format!("s{s}.{l}");
                let se = if w.get(&format!("{n}.se.c1.weight")).is_some() {
                    Some(Excite::load(
                        w,
                        &format!("{n}.se.c1"),
                        &format!("{n}.se.c2"),
                    )?)
                } else {
                    None
                };
                layers.push(Layer {
                    dw: Act::load(w, &format!("{n}.dw"), stride)?,
                    se,
                    pw: Act::load(w, &format!("{n}.pw"), 1)?,
                });
            }
            stages.push(layers);
        }
        let net = TextNet {
            stem,
            stages,
            lateral: (0..4)
                .map(|i| Conv::load(w, &format!("layer{i}"), 1))
                .collect::<Result<_, _>>()?,
            insert: (0..4)
                .map(|i| Rse::load(w, &format!("neck.insert{i}")))
                .collect::<Result<_, _>>()?,
            input: (0..4)
                .map(|i| Rse::load(w, &format!("neck.input{i}")))
                .collect::<Result<_, _>>()?,
            down: Conv::load(w, "head.down", 1)?,
            up: Up2::load(w, "head.up")?,
            last: Up2::load(w, "head.final")?,
        };
        if !w.left().is_empty() {
            return Err(format!("weights left over: {:?}", w.left()));
        }
        Ok(net)
    }

    /// The network with the built-in weights, made once.
    pub fn built_in() -> &'static TextNet {
        static NET: OnceLock<TextNet> = OnceLock::new();
        NET.get_or_init(|| {
            Weights::parse(WEIGHTS)
                .and_then(TextNet::load)
                .expect("the built-in text detector's weights load")
        })
    }

    /// Where `image`'s lettering is, at [`SIDE`].
    pub fn find(&self, image: &RgbImage) -> TextMap {
        let threads = jodel_tensors::kernels::threads();
        let small = image::imageops::resize(
            image,
            SIDE as u32,
            SIDE as u32,
            image::imageops::FilterType::Triangle,
        );
        // Blue, green, red, each as the authors normalise it.
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let data = small
            .pixels()
            .flat_map(|p| {
                [2, 1, 0]
                    .into_iter()
                    .enumerate()
                    .map(move |(i, c)| (f32::from(p[c]) / 255.0 - mean[i]) / std[i])
            })
            .collect();
        let x = Map {
            h: SIDE,
            w: SIDE,
            c: 3,
            data,
        };
        let prob = self.run(&x, threads);
        TextMap { side: SIDE, prob }
    }

    /// The probability map of the normalised picture `x` (its sides a
    /// multiple of 32).
    fn run(&self, x: &Map, threads: usize) -> Vec<f32> {
        let mut h = self.stem.apply(x, threads);
        let mut features = Vec::new();
        for (s, layers) in self.stages.iter().enumerate() {
            for layer in layers {
                h = layer.dw.apply(&h, threads);
                if let Some(se) = &layer.se {
                    let scale = se.scales(&h, |v| (v + 3.0).clamp(0.0, 6.0) / 6.0);
                    h = h.scaled(&scale);
                }
                h = layer.pw.apply(&h, threads);
            }
            if s >= 1 {
                features.push(h.clone());
            }
        }
        let mut fused: Vec<Map> = features
            .iter()
            .zip(&self.lateral)
            .zip(&self.insert)
            .map(|((f, lateral), insert)| insert.apply(&lateral.apply(f, threads), threads))
            .collect();
        for i in (0..3).rev() {
            let above = fused[i + 1].up(2);
            fused[i] = std::mem::replace(&mut fused[i], Map::new(0, 0, 0)).plus(&above);
        }
        let levels: Vec<Map> = fused
            .iter()
            .zip(&self.input)
            .enumerate()
            .map(|(i, (f, input))| {
                let y = input.apply(f, threads);
                if i == 0 {
                    y
                } else {
                    y.up(1 << i)
                }
            })
            .collect();
        let joined = Map::concat(&[&levels[3], &levels[2], &levels[1], &levels[0]]);
        let h = self.down.apply(&joined, threads).each(|v| v.max(0.0));
        let h = self.up.apply(&h, threads).each(|v| v.max(0.0));
        let h = self.last.apply(&h, threads);
        h.data
            .into_iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lettering_is_found_and_a_plain_picture_has_none() {
        let net = TextNet::built_in();
        // A plain grey cover: nothing.
        let plain = RgbImage::from_pixel(320, 320, image::Rgb([120, 120, 120]));
        let map = net.find(&plain);
        assert_eq!(map.prob.len(), SIDE * SIDE);
        assert!(map.prob.iter().all(|&p| p < 0.3));
        // A line of large dark letters on white: found there, not below.
        let mut cover = RgbImage::from_pixel(640, 640, image::Rgb([250, 250, 250]));
        let (x0, y0, x1, y1) = letters(&mut cover, "KARAOKE", 80.0, 60.0, 300.0);
        let map = net.find(&cover);
        let scale = SIDE as f32 / 640.0;
        let (mut inside, mut found) = (0, 0);
        for y in (y0 * scale) as usize..(y1 * scale) as usize {
            for x in (x0 * scale) as usize..(x1 * scale) as usize {
                inside += 1;
                found += usize::from(map.prob[y * SIDE + x] > 0.3);
            }
        }
        assert!(found * 3 > inside, "{found} of {inside} around the letters");
        let below = &map.prob[SIDE * 3 / 4 * SIDE..];
        assert!(below.iter().all(|&p| p < 0.3));
    }

    /// Draws `text` in black at `px` pixels, from (`x`, `baseline`); the
    /// box its ink fills.
    fn letters(
        cover: &mut RgbImage,
        text: &str,
        px: f32,
        x: f32,
        baseline: f32,
    ) -> (f32, f32, f32, f32) {
        use ab_glyph::{Font, FontRef, ScaleFont};
        let font = FontRef::try_from_slice(crate::gui::fonts::FONT).unwrap();
        let scaled = font.as_scaled(px);
        let (mut pen, mut bounds) = (x, (f32::MAX, f32::MAX, 0.0f32, 0.0f32));
        for c in text.chars() {
            let glyph = scaled.scaled_glyph(c);
            let advance = scaled.h_advance(glyph.id);
            let glyph = glyph
                .id
                .with_scale_and_position(px, ab_glyph::point(pen, baseline));
            if let Some(outline) = font.outline_glyph(glyph) {
                let b = outline.px_bounds();
                bounds = (
                    bounds.0.min(b.min.x),
                    bounds.1.min(b.min.y),
                    bounds.2.max(b.max.x),
                    bounds.3.max(b.max.y),
                );
                outline.draw(|gx, gy, cov| {
                    let (px, py) = (b.min.x as u32 + gx, b.min.y as u32 + gy);
                    if px < cover.width() && py < cover.height() && cov > 0.5 {
                        cover.put_pixel(px, py, image::Rgb([10, 10, 10]));
                    }
                });
            }
            pen += advance;
        }
        bounds
    }
}
