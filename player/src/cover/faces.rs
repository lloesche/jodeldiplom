//! Where a cover's faces are: our own inference of YuNet (Shiqi Yu and
//! others, OpenCV Zoo's face detector, MIT, see `models/LICENSE-yunet`),
//! read from its export by `tools/convert_yunet.py` and built in.
//!
//! The cover, at 640 × 640 in blue, green, red order (values 0 to 255, as
//! OpenCV gives it), goes through YuNet's backbone (pairs of a 1 × 1 and a
//! depthwise 3 × 3 convolution with ReLU, 2 × 2 max pools) and its small
//! FPN; heads at strides 8, 16 and 32 give each cell a class and an
//! objectness score and a box. A cell's score is the geometric mean of the
//! two; its box is centred at the cell plus the offset, its sides the
//! stride times their exponentials. Boxes over [`MIN_SCORE`] are kept, the
//! weaker of two that overlap by more than [`OVERLAP`] dropped.

use std::sync::OnceLock;

use image::RgbImage;
use jodel_rhythm::nn::weights::Weights;

use super::nets::{Conv, Map};

/// The weights, built in (`tools/convert_yunet.py`).
static WEIGHTS: &[u8] = include_bytes!("../../models/yunet.bin");

/// The side the cover is looked at.
const SIDE: usize = 640;
/// The least score a face is kept with.
pub const MIN_SCORE: f32 = 0.5;
/// How much two faces may overlap (intersection over union) before the
/// weaker goes.
const OVERLAP: f32 = 0.3;
/// The heads' strides.
const STRIDES: [usize; 3] = [8, 16, 32];

/// A face on a cover: its box in the cover's units (0 to 1 across and
/// down) and how sure the detector is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    /// Its left edge.
    pub x: f32,
    /// Its top edge.
    pub y: f32,
    /// Its width.
    pub w: f32,
    /// Its height.
    pub h: f32,
    /// The detector's score, 0 to 1.
    pub score: f32,
}

/// A 1 × 1 convolution and a depthwise 3 × 3 one, ReLU after.
struct Pair {
    pw: Conv,
    dw: Conv,
}

impl Pair {
    fn load(w: &mut Weights, name: &str) -> Result<Pair, String> {
        Ok(Pair {
            pw: Conv::load(w, &format!("{name}.pw"), 1)?,
            dw: Conv::load(w, &format!("{name}.dw"), 1)?,
        })
    }

    fn apply(&self, x: &Map, threads: usize) -> Map {
        let y = self.pw.apply(x, threads);
        self.dw.apply(&y, threads).each(|v| v.max(0.0))
    }
}

/// A head: a 1 × 1 convolution, then a 3 × 3 one.
struct Head {
    c1: Conv,
    c2: Conv,
}

impl Head {
    fn apply(&self, x: &Map, threads: usize) -> Map {
        self.c2.apply(&self.c1.apply(x, threads), threads)
    }
}

/// The face detector.
pub struct FaceNet {
    stem: Conv,
    blocks: Vec<Vec<Pair>>,
    lateral: Vec<Pair>,
    /// Class, box, objectness at each stride.
    heads: Vec<[Head; 3]>,
}

impl FaceNet {
    /// The network of `w`.
    pub fn load(mut w: Weights) -> Result<FaceNet, String> {
        let w = &mut w;
        let stem = Conv::load(w, "stem", 2)?;
        let mut blocks = vec![vec![Pair::load(w, "b0.0")?]];
        for b in 1..6 {
            blocks.push(vec![
                Pair::load(w, &format!("b{b}.0"))?,
                Pair::load(w, &format!("b{b}.1"))?,
            ]);
        }
        let lateral = (0..3)
            .map(|l| Pair::load(w, &format!("lateral{l}")))
            .collect::<Result<_, _>>()?;
        let mut heads = Vec::new();
        for level in 0..3 {
            let head = |w: &mut Weights, kind: &str| -> Result<Head, String> {
                Ok(Head {
                    c1: Conv::load(w, &format!("{kind}{level}.c1"), 1)?,
                    c2: Conv::load(w, &format!("{kind}{level}.c2"), 1)?,
                })
            };
            heads.push([head(w, "cls")?, head(w, "bbox")?, head(w, "obj")?]);
            // The landmarks are not used.
            for name in ["c1", "c2"] {
                for part in ["weight", "bias"] {
                    let n = format!("kps{level}.{name}.{part}");
                    let s = super::nets::shape(w, &n)?;
                    w.take(&n, &s)?;
                }
            }
        }
        if !w.left().is_empty() {
            return Err(format!("weights left over: {:?}", w.left()));
        }
        Ok(FaceNet {
            stem,
            blocks,
            lateral,
            heads,
        })
    }

    /// The network with the built-in weights, made once.
    pub fn built_in() -> &'static FaceNet {
        static NET: OnceLock<FaceNet> = OnceLock::new();
        NET.get_or_init(|| {
            Weights::parse(WEIGHTS)
                .and_then(FaceNet::load)
                .expect("the built-in face detector's weights load")
        })
    }

    /// The faces on `image`, the surest first.
    pub fn find(&self, image: &RgbImage) -> Vec<Face> {
        let threads = jodel_tensors::kernels::threads();
        let small = image::imageops::resize(
            image,
            SIDE as u32,
            SIDE as u32,
            image::imageops::FilterType::Triangle,
        );
        let data = small
            .pixels()
            .flat_map(|p| [f32::from(p[2]), f32::from(p[1]), f32::from(p[0])])
            .collect();
        let x = Map {
            h: SIDE,
            w: SIDE,
            c: 3,
            data,
        };
        let mut faces = Vec::new();
        for (level, out) in self.run(&x, threads).iter().enumerate() {
            let stride = STRIDES[level] as f32;
            let [cls, bbox, obj] = out;
            for r in 0..cls.h {
                for c in 0..cls.w {
                    let i = r * cls.w + c;
                    let sig = |v: f32| (1.0 / (1.0 + (-v).exp())).clamp(0.0, 1.0);
                    let score = (sig(cls.data[i]) * sig(obj.data[i])).sqrt();
                    if score < MIN_SCORE {
                        continue;
                    }
                    let b = &bbox.data[i * 4..][..4];
                    let (cx, cy) = ((c as f32 + b[0]) * stride, (r as f32 + b[1]) * stride);
                    let (w, h) = (b[2].exp() * stride, b[3].exp() * stride);
                    let side = SIDE as f32;
                    faces.push(Face {
                        x: (cx - w / 2.0) / side,
                        y: (cy - h / 2.0) / side,
                        w: w / side,
                        h: h / side,
                        score,
                    });
                }
            }
        }
        keep_best(faces)
    }

    /// Class, box and objectness at each stride.
    fn run(&self, x: &Map, threads: usize) -> Vec<[Map; 3]> {
        let mut h = self.stem.apply(x, threads).each(|v| v.max(0.0));
        // A pool after block 0 and before blocks 3, 4 and 5, whose outputs
        // are at strides 8, 16 and 32.
        let mut kept = Vec::new();
        for (b, block) in self.blocks.iter().enumerate() {
            if b >= 3 {
                h = h.max_pool();
            }
            for pair in block {
                h = pair.apply(&h, threads);
            }
            if b == 0 {
                h = h.max_pool();
            }
            if b >= 3 {
                kept.push(h.clone());
            }
        }
        let (s8, s16, s32) = (&kept[0], &kept[1], &kept[2]);
        let p32 = self.lateral[2].apply(s32, threads);
        let p16 = self.lateral[1].apply(&s16.clone().plus(&p32.up(2)), threads);
        let p8 = self.lateral[0].apply(&s8.clone().plus(&p16.up(2)), threads);
        [p8, p16, p32]
            .iter()
            .zip(&self.heads)
            .map(|(p, [cls, bbox, obj])| {
                [
                    cls.apply(p, threads),
                    bbox.apply(p, threads),
                    obj.apply(p, threads),
                ]
            })
            .collect()
    }
}

/// `faces` without the weaker of any two that overlap too much, the surest
/// first.
fn keep_best(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in faces {
        if kept.iter().all(|k| overlap(k, &f) <= OVERLAP) {
            kept.push(f);
        }
    }
    kept
}

/// Intersection over union.
fn overlap(a: &Face, b: &Face) -> f32 {
    let w = ((a.x + a.w).min(b.x + b.w) - a.x.max(b.x)).max(0.0);
    let h = ((a.y + a.h).min(b.y + b.h) - a.y.max(b.y)).max(0.0);
    let inter = w * h;
    inter / (a.w * a.h + b.w * b.h - inter).max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_faces_keep_the_surest_and_a_plain_cover_has_none() {
        let face = |x: f32, score: f32| Face {
            x,
            y: 0.2,
            w: 0.2,
            h: 0.2,
            score,
        };
        let kept = keep_best(vec![face(0.1, 0.6), face(0.12, 0.9), face(0.6, 0.7)]);
        assert_eq!(kept, vec![face(0.12, 0.9), face(0.6, 0.7)]);
        let plain = RgbImage::from_pixel(300, 300, image::Rgb([90, 140, 200]));
        assert!(FaceNet::built_in().find(&plain).is_empty());
    }
}
