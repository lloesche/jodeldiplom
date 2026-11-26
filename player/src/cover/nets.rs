//! The layers of a cover's two small detectors (its lettering, `text.rs`,
//! and its faces, `faces.rs`), on the processor: feature maps as rows ×
//! columns × channels of `f32`, convolutions as rows of patches through
//! jodel-rhythm's matrix multiply ([`Linear`]), depthwise ones as plain
//! loops the compiler vectorises, each spread over jodel-rhythm's share of
//! the threads.

use jodel_rhythm::nn::layers::Linear;
use jodel_rhythm::nn::weights::Weights;
use jodel_tensors::kernels::par_chunks;

/// A feature map: `h` rows of `w` pixels of `c` channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Map {
    /// Its rows.
    pub h: usize,
    /// Its columns.
    pub w: usize,
    /// Its channels.
    pub c: usize,
    /// Its values, row by row, a pixel's channels together.
    pub data: Vec<f32>,
}

impl Map {
    /// A map of zeros.
    pub fn new(h: usize, w: usize, c: usize) -> Map {
        Map {
            h,
            w,
            c,
            data: vec![0.0; h * w * c],
        }
    }

    /// Applies `f` to every value.
    pub fn each(mut self, f: impl Fn(f32) -> f32) -> Map {
        self.data.iter_mut().for_each(|v| *v = f(*v));
        self
    }

    /// `self + other` (the same shape).
    pub fn plus(mut self, other: &Map) -> Map {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += b);
        self
    }

    /// Each pixel repeated `s` × `s` times (nearest upsampling).
    pub fn up(&self, s: usize) -> Map {
        let mut out = Map::new(self.h * s, self.w * s, self.c);
        let c = self.c;
        for y in 0..out.h {
            for x in 0..out.w {
                let from = ((y / s) * self.w + x / s) * c;
                out.data[(y * out.w + x) * c..][..c].copy_from_slice(&self.data[from..][..c]);
            }
        }
        out
    }

    /// The largest of each 2 × 2 (the last odd row or column dropped).
    pub fn max_pool(&self) -> Map {
        let mut out = Map::new(self.h / 2, self.w / 2, self.c);
        let c = self.c;
        for y in 0..out.h {
            for x in 0..out.w {
                let o = &mut out.data[(y * out.w + x) * c..][..c];
                o.fill(f32::MIN);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let i = ((2 * y + dy) * self.w + 2 * x + dx) * c;
                    for (a, &b) in o.iter_mut().zip(&self.data[i..][..c]) {
                        *a = a.max(b);
                    }
                }
            }
        }
        out
    }

    /// Each channel's mean.
    pub fn means(&self) -> Vec<f32> {
        let mut m = vec![0.0f32; self.c];
        for px in self.data.chunks(self.c) {
            m.iter_mut().zip(px).for_each(|(a, b)| *a += b);
        }
        let n = (self.h * self.w).max(1) as f32;
        m.iter_mut().for_each(|v| *v /= n);
        m
    }

    /// Each channel scaled by `scale`'s value for it.
    pub fn scaled(mut self, scale: &[f32]) -> Map {
        for px in self.data.chunks_mut(self.c) {
            px.iter_mut().zip(scale).for_each(|(a, s)| *a *= s);
        }
        self
    }

    /// The maps' channels side by side (the same size).
    pub fn concat(maps: &[&Map]) -> Map {
        let c = maps.iter().map(|m| m.c).sum();
        let mut out = Map::new(maps[0].h, maps[0].w, c);
        for p in 0..out.h * out.w {
            let mut at = p * c;
            for m in maps {
                out.data[at..][..m.c].copy_from_slice(&m.data[p * m.c..][..m.c]);
                at += m.c;
            }
        }
        out
    }
}

/// The shape of tensor `name` in `w`.
pub fn shape(w: &Weights, name: &str) -> Result<Vec<usize>, String> {
    w.shapes()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.to_vec())
        .ok_or_else(|| format!("no tensor {name}"))
}

/// The hard swish, `x · relu6(x + 3) / 6`.
pub fn hard_swish(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) / 6.0
}

/// A k × k convolution, padded by k / 2, with a bias: across all channels
/// (`dense`) or each channel on its own (depthwise).
pub struct Conv {
    k: usize,
    stride: usize,
    cin: usize,
    cout: usize,
    kind: Kind,
}

enum Kind {
    /// Over patches `[ky][kx][channel]`.
    Dense(Linear),
    /// Weights `[ky][kx][channel]`, and the bias.
    Depthwise(Vec<f32>, Vec<f32>),
}

/// Patches at most this many values at once (a band of output rows).
const PATCHES: usize = 1 << 22;

impl Conv {
    /// Convolution `name` of `w` (`NAME.weight` `[out][in / groups][k][k]`,
    /// `NAME.bias` if there is one) with `stride`; depthwise when its
    /// weight has one input channel a group and the map more.
    pub fn load(w: &mut Weights, name: &str, stride: usize) -> Result<Conv, String> {
        let s = shape(w, &format!("{name}.weight"))?;
        let [cout, per, k, k2] = s[..] else {
            return Err(format!("{name}: weight of shape {s:?}"));
        };
        if k != k2 {
            return Err(format!("{name}: a {k} × {k2} kernel"));
        }
        let raw = w.take(&format!("{name}.weight"), &s)?;
        let bias = if w.get(&format!("{name}.bias")).is_some() {
            w.take(&format!("{name}.bias"), &[cout])?
        } else {
            vec![0.0; cout]
        };
        Ok(Conv::new(raw, bias, cout, per, k, stride))
    }

    /// The convolution of `raw` (`[out][per][k][k]`, `per` 1 and `out`
    /// channels in for a depthwise one) and `bias`.
    pub fn new(
        raw: Vec<f32>,
        bias: Vec<f32>,
        cout: usize,
        per: usize,
        k: usize,
        stride: usize,
    ) -> Conv {
        if per == 1 && cout > 1 {
            let mut weight = vec![0.0f32; raw.len()];
            for c in 0..cout {
                for t in 0..k * k {
                    weight[t * cout + c] = raw[c * k * k + t];
                }
            }
            return Conv {
                k,
                stride,
                cin: cout,
                cout,
                kind: Kind::Depthwise(weight, bias),
            };
        }
        // [out][in][ky][kx] → [out][ky][kx][in].
        let mut weight = vec![0.0f32; raw.len()];
        for o in 0..cout {
            for i in 0..per {
                for t in 0..k * k {
                    weight[(o * k * k + t) * per + i] = raw[(o * per + i) * k * k + t];
                }
            }
        }
        Conv {
            k,
            stride,
            cin: per,
            cout,
            kind: Kind::Dense(Linear::new(weight, Some(bias), k * k * per, cout)),
        }
    }

    /// `x` through it.
    pub fn apply(&self, x: &Map, threads: usize) -> Map {
        debug_assert_eq!(x.c, self.cin);
        let pad = self.k / 2;
        let oh = (x.h + 2 * pad - self.k) / self.stride + 1;
        let ow = (x.w + 2 * pad - self.k) / self.stride + 1;
        match &self.kind {
            Kind::Dense(layer) if self.k == 1 && self.stride == 1 => Map {
                h: oh,
                w: ow,
                c: self.cout,
                data: layer.apply(&x.data, threads),
            },
            Kind::Dense(layer) => {
                let patch = self.k * self.k * x.c;
                let band = (PATCHES / (ow * patch)).clamp(1, oh);
                let mut out = Map::new(oh, ow, self.cout);
                let mut patches = vec![0.0f32; band * ow * patch];
                for y0 in (0..oh).step_by(band) {
                    let rows = band.min(oh - y0);
                    let p = &mut patches[..rows * ow * patch];
                    par_chunks(p, ow * patch, threads, |r, row| {
                        self.gather(x, y0 + r, row);
                    });
                    let y = layer.apply(p, threads);
                    out.data[y0 * ow * self.cout..][..y.len()].copy_from_slice(&y);
                }
                out
            }
            Kind::Depthwise(weight, bias) => {
                let c = x.c;
                let mut out = Map::new(oh, ow, c);
                par_chunks(&mut out.data, ow * c, threads, |oy, row| {
                    for ox in 0..ow {
                        let o = &mut row[ox * c..][..c];
                        o.copy_from_slice(bias);
                        for ky in 0..self.k {
                            let Some(iy) = (oy * self.stride + ky).checked_sub(pad) else {
                                continue;
                            };
                            if iy >= x.h {
                                continue;
                            }
                            for kx in 0..self.k {
                                let Some(ix) = (ox * self.stride + kx).checked_sub(pad) else {
                                    continue;
                                };
                                if ix >= x.w {
                                    continue;
                                }
                                let wk = &weight[(ky * self.k + kx) * c..][..c];
                                let xi = &x.data[(iy * x.w + ix) * c..][..c];
                                for ((a, &wv), &xv) in o.iter_mut().zip(wk).zip(xi) {
                                    *a += wv * xv;
                                }
                            }
                        }
                    }
                });
                out
            }
        }
    }

    /// Output row `oy`'s patches into `row` (`[x][ky][kx][channel]`, 0
    /// outside the map).
    fn gather(&self, x: &Map, oy: usize, row: &mut [f32]) {
        let (k, c, pad) = (self.k, x.c, self.k / 2);
        for (ox, p) in row.chunks_mut(k * k * c).enumerate() {
            for ky in 0..k {
                for kx in 0..k {
                    let dst = &mut p[(ky * k + kx) * c..][..c];
                    let iy = (oy * self.stride + ky)
                        .checked_sub(pad)
                        .filter(|&v| v < x.h);
                    let ix = (ox * self.stride + kx)
                        .checked_sub(pad)
                        .filter(|&v| v < x.w);
                    match (iy, ix) {
                        (Some(iy), Some(ix)) => {
                            dst.copy_from_slice(&x.data[(iy * x.w + ix) * c..][..c]);
                        }
                        _ => dst.fill(0.0),
                    }
                }
            }
        }
    }
}

/// A 2 × 2 transposed convolution of stride 2 (each pixel becomes four),
/// with a bias.
pub struct Up2 {
    cout: usize,
    /// Rows `(dy·2 + dx)·out + o`.
    layer: Linear,
}

impl Up2 {
    /// `name` of `w`: `NAME.weight` `[in][out][2][2]`, `NAME.bias`.
    pub fn load(w: &mut Weights, name: &str) -> Result<Up2, String> {
        let s = shape(w, &format!("{name}.weight"))?;
        let [cin, cout, 2, 2] = s[..] else {
            return Err(format!("{name}: weight of shape {s:?}"));
        };
        let raw = w.take(&format!("{name}.weight"), &s)?;
        let bias = w.take(&format!("{name}.bias"), &[cout])?;
        let mut weight = vec![0.0f32; raw.len()];
        let mut b = vec![0.0f32; 4 * cout];
        for q in 0..4 {
            for o in 0..cout {
                b[q * cout + o] = bias[o];
                for i in 0..cin {
                    weight[(q * cout + o) * cin + i] = raw[(i * cout + o) * 4 + q];
                }
            }
        }
        Ok(Up2 {
            cout,
            layer: Linear::new(weight, Some(b), cin, 4 * cout),
        })
    }

    /// `x` through it: twice the size.
    pub fn apply(&self, x: &Map, threads: usize) -> Map {
        let y = self.layer.apply(&x.data, threads);
        let c = self.cout;
        let mut out = Map::new(x.h * 2, x.w * 2, c);
        for py in 0..x.h {
            for px in 0..x.w {
                let from = &y[(py * x.w + px) * 4 * c..][..4 * c];
                for q in 0..4 {
                    let (dy, dx) = (q / 2, q % 2);
                    let at = ((2 * py + dy) * out.w + 2 * px + dx) * c;
                    out.data[at..][..c].copy_from_slice(&from[q * c..][..c]);
                }
            }
        }
        out
    }
}

/// A squeeze-excitation: the channels' means through two 1 × 1
/// convolutions (a ReLU between), each channel scaled by `gate` of the
/// result.
pub struct Excite {
    down: Linear,
    up: Linear,
}

impl Excite {
    /// `{down}` and `{up}` of `w`, each `.weight` `[out][in][1][1]` and
    /// `.bias`.
    pub fn load(w: &mut Weights, down: &str, up: &str) -> Result<Excite, String> {
        let linear = |w: &mut Weights, name: &str| -> Result<Linear, String> {
            let s = shape(w, &format!("{name}.weight"))?;
            let weight = w.take(&format!("{name}.weight"), &s)?;
            let bias = w.take(&format!("{name}.bias"), &[s[0]])?;
            Ok(Linear::new(weight, Some(bias), s[1], s[0]))
        };
        Ok(Excite {
            down: linear(w, down)?,
            up: linear(w, up)?,
        })
    }

    /// The scale of each of `x`'s channels.
    pub fn scales(&self, x: &Map, gate: impl Fn(f32) -> f32) -> Vec<f32> {
        let mid = self.down.apply(&x.means(), 1);
        let mid: Vec<f32> = mid.into_iter().map(|v| v.max(0.0)).collect();
        self.up.apply(&mid, 1).into_iter().map(gate).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dense_convolution_sums_its_patch_and_pads_with_zeros() {
        // One input channel, a 3 × 3 kernel of ones, bias 1: each output
        // is 1 plus the sum of its neighbourhood.
        let conv = Conv::new(vec![1.0; 9], vec![1.0], 1, 1, 3, 1);
        let x = Map {
            h: 3,
            w: 3,
            c: 1,
            data: (1..=9).map(|v| v as f32).collect(),
        };
        let y = conv.apply(&x, 2);
        assert_eq!(y.data[4], 46.0);
        assert_eq!(y.data[0], 1.0 + 1.0 + 2.0 + 4.0 + 5.0);
        // Stride 2: the corners.
        let conv = Conv::new(vec![1.0; 9], vec![0.0], 1, 1, 3, 2);
        let y = conv.apply(&x, 2);
        assert_eq!((y.h, y.w), (2, 2));
        assert_eq!(y.data, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn a_depthwise_convolution_keeps_the_channels_apart() {
        // Two channels: the first's kernel picks the centre, the second's
        // the pixel to the left.
        let mut raw = vec![0.0; 18];
        raw[4] = 2.0;
        raw[9 + 3] = 1.0;
        let conv = Conv::new(raw, vec![0.0, 0.5], 2, 1, 3, 1);
        let x = Map {
            h: 1,
            w: 3,
            c: 2,
            data: vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        };
        let y = conv.apply(&x, 1);
        assert_eq!(y.data, vec![2.0, 0.5, 4.0, 10.5, 6.0, 20.5]);
    }

    #[test]
    fn the_maps_helpers_shape_as_they_say() {
        let x = Map {
            h: 2,
            w: 2,
            c: 1,
            data: vec![1.0, 5.0, 3.0, 2.0],
        };
        assert_eq!(x.max_pool().data, vec![5.0]);
        assert_eq!(x.up(2).data[..4], [1.0, 1.0, 5.0, 5.0]);
        assert_eq!(x.means(), vec![2.75]);
        let both = Map::concat(&[&x, &x.clone().each(|v| -v)]);
        assert_eq!(both.data[..4], [1.0, -1.0, 5.0, -5.0]);
        assert_eq!(hard_swish(3.0), 3.0);
        assert_eq!(hard_swish(-3.0), 0.0);
    }
}
