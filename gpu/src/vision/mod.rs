//! The cover's vision networks on the GPU: Depth Anything 3 Small (its
//! depth, [`GpuDepth`], `depth.rs`) and BEN2 Base (its subject's matte,
//! [`GpuMatte`], `matte.rs` with the Swin backbone in `swin.rs`).
//!
//! Both run on one small set of kernels (`prelude.wgsl` and
//! `../common.wgsl`, then `conv.wgsl`, `ops.wgsl`, `attention.wgsl`): a
//! convolution as a matrix multiply that reads its patches where they lie
//! (a linear layer is a 1 × 1 one, a transposed one with its stride as its
//! size writes its pixels spread out; a band of output rows at a time where
//! asked), LayerNorm, attention as two batched matrix multiplies around a
//! softmax (Swin's relative positions and shifted windows' mask added in
//! it), an instance norm from chunks' means and spreads, resizing as
//! PyTorch's interpolate does, adaptive pooling, strided copies, Swin's
//! windows in and out, 2-D rotary positions, tables, GELU (with the error
//! function, as PyTorch's), ReLU and the sigmoid. Images lie as views ×
//! rows × columns × channels; everything is f32, the weights as the
//! checkpoints hold them.
//!
//! A network is planned for its input's size ([`Plan`]: its buffers and
//! dispatches), made on the device ([`net::Net`], on the one GPU both share,
//! [`net::Gpu`]) and run in short submissions that give way to the window
//! ([`crate::pace`]).

pub mod check;
mod depth;
mod matte;
mod net;
mod ops;
pub mod prep;
mod swin;
#[cfg(test)]
mod tests;

use jodel_tensors::SafeTensors;

use crate::engine::{Access, Kernels};

pub use depth::{Depth, GpuDepth};
pub use matte::{GpuMatte, Matte};
pub(crate) use ops::Plan;

pub(crate) use crate::packer::{Named, Packer};

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Conv,
    LayerNorm,
    Act,
    ChanSums,
    ChanReduce,
    ChanApply,
    Resize,
    Pool,
    Copy,
    WinPart,
    WinBack,
    Rope,
    Table,
    MulMap,
    Bmm,
    Softmax,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Conv,
        Kernel::LayerNorm,
        Kernel::Act,
        Kernel::ChanSums,
        Kernel::ChanReduce,
        Kernel::ChanApply,
        Kernel::Resize,
        Kernel::Pool,
        Kernel::Copy,
        Kernel::WinPart,
        Kernel::WinBack,
        Kernel::Rope,
        Kernel::Table,
        Kernel::MulMap,
        Kernel::Bmm,
        Kernel::Softmax,
    ];
    const PARAM_BYTES: u64 = 96;
    /// After the parameters: the second buffer read.
    const EXTRA: &'static [Access] = &[Access::Read];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Conv => "conv",
            Kernel::LayerNorm => "layernorm",
            Kernel::Act => "act",
            Kernel::ChanSums => "chan_sums",
            Kernel::ChanReduce => "chan_reduce",
            Kernel::ChanApply => "chan_apply",
            Kernel::Resize => "resize",
            Kernel::Pool => "pool",
            Kernel::Copy => "copy",
            Kernel::WinPart => "win_part",
            Kernel::WinBack => "win_back",
            Kernel::Rope => "rope",
            Kernel::Table => "table",
            Kernel::MulMap => "mulmap",
            Kernel::Bmm => "bmm",
            Kernel::Softmax => "softmax",
        }
    }
}

/// The kernels' source.
pub(crate) fn source() -> String {
    [
        include_str!("prelude.wgsl"),
        include_str!("../common.wgsl"),
        include_str!("conv.wgsl"),
        include_str!("ops.wgsl"),
        include_str!("attention.wgsl"),
    ]
    .join("\n")
}

/// A checkpoint's tensors read into the packer by name, laid out as the
/// vision kernels read them.
impl Named {
    pub(crate) fn vector(&mut self, t: &SafeTensors, name: &str, len: usize) -> Result<(), String> {
        let v = t.get(name, &[len])?;
        self.put(name, &v);
        Ok(())
    }

    /// A convolution's weight (`[out][in][k][k]`) under `name`.
    pub(crate) fn put_conv(
        &mut self,
        name: &str,
        w: &[f32],
        (ins, outs, k): (usize, usize, usize),
    ) {
        let at = self.packer.conv(w, outs, ins, k);
        self.name(name, at);
    }

    /// A linear layer's weight (`[out][in]`) under `name`, as `[in][out]`.
    pub(crate) fn put_linear(&mut self, name: &str, w: &[f32], outs: usize, ins: usize) {
        let at = self.packer.linear(w, outs, ins);
        self.name(name, at);
    }

    /// A linear layer (`name.weight`, `name.bias` if `bias`), scaled by a
    /// layer scale (`scale`, folded into it).
    pub(crate) fn linear(
        &mut self,
        t: &SafeTensors,
        name: &str,
        (ins, outs): (usize, usize),
        bias: bool,
        scale: Option<&[f32]>,
    ) -> Result<(), String> {
        let mut w = t.get(&format!("{name}.weight"), &[outs, ins])?;
        let mut b = if bias {
            t.get(&format!("{name}.bias"), &[outs])?
        } else {
            Vec::new()
        };
        if let Some(g) = scale {
            for o in 0..outs {
                if bias {
                    b[o] *= g[o];
                }
                for v in &mut w[o * ins..(o + 1) * ins] {
                    *v *= g[o];
                }
            }
        }
        self.put_linear(&format!("{name}.weight"), &w, outs, ins);
        if bias {
            self.put(&format!("{name}.bias"), &b);
        }
        Ok(())
    }

    /// A convolution (`name.weight` of `[outs][ins][k][k]`, its bias).
    pub(crate) fn conv(
        &mut self,
        t: &SafeTensors,
        name: &str,
        (ins, outs, k): (usize, usize, usize),
        bias: bool,
    ) -> Result<(), String> {
        let w = t.get(&format!("{name}.weight"), &[outs, ins, k, k])?;
        let at = self.packer.conv(&w, outs, ins, k);
        self.name(&format!("{name}.weight"), at);
        if bias {
            self.vector(t, &format!("{name}.bias"), outs)?;
        }
        Ok(())
    }

    /// A transposed convolution, its stride its size `s`.
    pub(crate) fn deconv(
        &mut self,
        t: &SafeTensors,
        name: &str,
        ch: usize,
        s: usize,
    ) -> Result<(), String> {
        let w = t.get(&format!("{name}.weight"), &[ch, ch, s, s])?;
        let at = self.packer.deconv(&w, ch, ch, s);
        self.name(&format!("{name}.weight"), at);
        self.vector(t, &format!("{name}.bias"), ch)
    }

    /// The convolution `name` (`cin`, `cout`, size), its stride and padding.
    pub(crate) fn cw(
        &self,
        name: &str,
        (cin, cout, k): (usize, usize, usize),
        stride: usize,
        pad: usize,
        bias: bool,
    ) -> ops::ConvW {
        ops::ConvW {
            at: self.at(&format!("{name}.weight")),
            bias: bias.then(|| self.at(&format!("{name}.bias"))),
            cin,
            cout,
            k,
            stride,
            pad,
            up: 1,
        }
    }
}
