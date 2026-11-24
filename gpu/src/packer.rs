//! A network's weights and constants as one buffer's contents ([`Packer`]),
//! each tensor laid out as the kernels read it.

/// A linear layer's weight (`[out][in]`, as PyTorch keeps it) as
/// `[in][out]`, as the matrix multiplies read it.
pub(crate) fn transposed(w: &[f32], outs: usize, ins: usize) -> Vec<f32> {
    assert_eq!(w.len(), outs * ins);
    let mut t = vec![0.0; w.len()];
    for o in 0..outs {
        for i in 0..ins {
            t[i * outs + o] = w[o * ins + i];
        }
    }
    t
}

/// Values packed one after another, each push's offset (in f32s) given
/// back; offsets count from `origin` (a packer for the part of a buffer
/// after another's).
#[derive(Clone, Debug, Default)]
pub(crate) struct Packer {
    /// Where the first value sits in the buffer.
    pub(crate) origin: usize,
    pub(crate) data: Vec<f32>,
}

impl Packer {
    /// A packer whose values start at `origin` of their buffer.
    pub(crate) fn at(origin: usize) -> Packer {
        Packer {
            origin,
            data: Vec::new(),
        }
    }

    /// `values` appended; their offset.
    pub(crate) fn push(&mut self, values: &[f32]) -> usize {
        let at = self.origin + self.data.len();
        self.data.extend_from_slice(values);
        at
    }

    /// A linear layer's weight (`[out][in]`, as PyTorch keeps it) as
    /// `[in][out]`.
    pub(crate) fn linear(&mut self, w: &[f32], outs: usize, ins: usize) -> usize {
        self.push(&transposed(w, outs, ins))
    }

    /// A convolution's weight (`[out][in][k][k]`) as `[(ky·k + kx)·in +
    /// ci][out]`.
    pub(crate) fn conv(&mut self, w: &[f32], outs: usize, ins: usize, k: usize) -> usize {
        assert_eq!(w.len(), outs * ins * k * k);
        let mut t = vec![0.0; w.len()];
        for o in 0..outs {
            for i in 0..ins {
                for ky in 0..k {
                    for kx in 0..k {
                        t[((ky * k + kx) * ins + i) * outs + o] =
                            w[((o * ins + i) * k + ky) * k + kx];
                    }
                }
            }
        }
        self.push(&t)
    }

    /// A transposed convolution's weight (`[in][out][s][s]`, its stride its
    /// size) as `[in][(dy·s + dx)·out + o]`.
    pub(crate) fn deconv(&mut self, w: &[f32], ins: usize, outs: usize, s: usize) -> usize {
        assert_eq!(w.len(), outs * ins * s * s);
        let mut t = vec![0.0; w.len()];
        for i in 0..ins {
            for o in 0..outs {
                for dy in 0..s {
                    for dx in 0..s {
                        t[i * (s * s * outs) + (dy * s + dx) * outs + o] =
                            w[((i * outs + o) * s + dy) * s + dx];
                    }
                }
            }
        }
        self.push(&t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_count_from_the_origin_and_layers_are_transposed() {
        let mut p = Packer::at(10);
        assert_eq!(p.push(&[1.0, 2.0]), 10);
        // [out 2][in 3] → [in 3][out 2].
        assert_eq!(p.linear(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3), 12);
        assert_eq!(&p.data[2..], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
