// The prelude of the vision networks' kernels (the cover's depth and its
// subject's matte). Every one reads `src` (and `src2`) and the weights and
// constants in `aux`, and writes `dst`; `p` holds its numbers, named per
// kernel. Buffers hold f32 images as views × rows × columns × channels (a
// pixel's channels side by side), and tokens as rows of features.

struct Params {
    f0: u32, f1: u32, f2: u32, f3: u32,
    f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32,
    f12: u32, f13: u32, f14: u32, f15: u32,
    f16: u32, f17: u32, f18: u32, f19: u32,
    f20: u32, f21: u32, f22: u32, f23: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> src2: array<f32>;
