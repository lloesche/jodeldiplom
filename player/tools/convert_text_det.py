#!/usr/bin/env python3
"""Converts PP-OCRv5 mobile det (PaddleOCR's text detector) into the weights
file the player embeds for keeping lettering out of a cover's subject
(`player/src/cover/text.rs`).

    python -I tools/convert_text_det.py model.safetensors models/text-det.bin

The checkpoint is read as plain bytes (a safetensors file: a JSON header and
raw little-endian arrays), with nothing but the standard library. The
network is folded for inference as its authors' own export folds it: each
of the backbone's re-parameterisable layers (four k×k branches, a 1×1
branch, an identity BatchNorm, each with its BatchNorm, and the learnable
scale and shift after their sum) becomes one convolution with a bias; every
other BatchNorm is folded into the convolution before it. The scale and
shift after a layer's activation are kept as NAME.act = [scale, shift].

The output is the format of rhythm/tools/convert_beat_this.py (b"BTW1",
half precision):

    stem                   the 3×3 stride-2 stem (no activation)
    s{S}.{L}.dw            stage S, layer L: the depthwise k×k convolution
    s{S}.{L}.dw.act        (stride 1 only) its activation's scale, shift
    s{S}.{L}.se.c1, .c2    (where there is one) its squeeze-excitation
    s{S}.{L}.pw, .pw.act   the pointwise convolution, its activation's
    layer{I}               the 1×1 convolutions from stages 2 to 5 to the neck
    neck.insert{I}, neck.input{I}, each with .se1 and .se2
    head.down, head.up, head.final

Used for models/text-det.bin: model.safetensors of
https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors at
c5041d225cf951ff06900ab81a3c7d543c45e2ad (Apache-2.0, see
models/LICENSE-pp-ocr), SHA-256
c49eea87d579f931c79bf29ac1ffb1b3166f86f6811b8e46b982bb66e3f0f287.
"""

import argparse
import array
import json
import struct
import sys
from collections import OrderedDict

EPS = 1e-5
# The backbone's stages: (kernel, stride, squeeze-excitation) of each layer.
STAGES = [
    [(3, 1, False)],
    [(3, 2, False), (3, 1, False)],
    [(3, 2, False), (3, 1, False)],
    [(3, 2, False), (5, 1, False), (5, 1, False), (5, 1, False), (5, 1, False)],
    [(5, 2, True), (5, 1, True), (5, 1, False), (5, 1, False)],
]


class Tensor:
    def __init__(self, shape, data):
        self.shape = list(shape)
        self.data = list(data)


def read(path):
    raw = open(path, "rb").read()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    body = raw[8 + length :]
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise SystemExit(f"{name}: {info['dtype']}, not F32")
        start, end = info["data_offsets"]
        a = array.array("f")
        a.frombytes(body[start:end])
        if sys.byteorder != "little":
            a.byteswap()
        out[name] = Tensor(info["shape"], a)
    return out


def size(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def bn(w, name):
    """A BatchNorm's scale and shift per channel."""
    g, b = w[f"{name}.weight"].data, w[f"{name}.bias"].data
    m, v = w[f"{name}.running_mean"].data, w[f"{name}.running_var"].data
    scale = [gi / (vi + EPS) ** 0.5 for gi, vi in zip(g, v)]
    shift = [bi - mi * si for bi, mi, si in zip(b, m, scale)]
    return scale, shift


def folded(w, conv, norm):
    """Convolution `conv` (no bias) with BatchNorm `norm` folded in."""
    t = w[f"{conv}.weight"]
    scale, shift = bn(w, norm)
    per = size(t.shape[1:])
    data = [v * scale[i // per] for i, v in enumerate(t.data)]
    return Tensor(t.shape, data), shift


def rep(w, name, k):
    """A re-parameterisable layer as one k×k convolution and bias."""
    total, bias = None, None

    def add(t, b):
        nonlocal total, bias
        if total is None:
            total, bias = Tensor(t.shape, t.data), list(b)
        else:
            total.data = [x + y for x, y in zip(total.data, t.data)]
            bias = [x + y for x, y in zip(bias, b)]

    branches = 0
    while f"{name}.conv_symmetric.{branches}.convolution.weight" in w:
        branches += 1
    first = w[f"{name}.conv_symmetric.0.convolution.weight"]
    out_c, in_g = first.shape[0], first.shape[1]
    for i in range(branches):
        b = f"{name}.conv_symmetric.{i}"
        add(*folded(w, f"{b}.convolution", f"{b}.normalization"))
    centre = k // 2
    if f"{name}.conv_small_symmetric.convolution.weight" in w:
        t, b = folded(
            w,
            f"{name}.conv_small_symmetric.convolution",
            f"{name}.conv_small_symmetric.normalization",
        )
        padded = [0.0] * (out_c * in_g * k * k)
        for o in range(out_c):
            for i in range(in_g):
                padded[((o * in_g + i) * k + centre) * k + centre] = t.data[o * in_g + i]
        add(Tensor([out_c, in_g, k, k], padded), b)
    if f"{name}.identity.running_mean" in w:
        scale, shift = bn(w, f"{name}.identity")
        ident = [0.0] * (out_c * in_g * k * k)
        for o in range(out_c):
            i = 0 if in_g == 1 else o
            ident[((o * in_g + i) * k + centre) * k + centre] = scale[o]
        add(Tensor([out_c, in_g, k, k], ident), shift)
    s = w[f"{name}.lab.scale"].data[0]
    sh = w[f"{name}.lab.bias"].data[0]
    total.data = [v * s for v in total.data]
    bias = [v * s + sh for v in bias]
    return total, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--f32", action="store_true")
    args = parser.parse_args()
    w = read(args.checkpoint)
    out = OrderedDict()

    def put(name, t, b=None):
        out[f"{name}.weight"] = (t.shape, t.data)
        if b is not None:
            out[f"{name}.bias"] = ([len(b)], b)

    def plain(name, source):
        put(name, w[f"{source}.weight"], w[f"{source}.bias"].data if f"{source}.bias" in w else None)

    enc = "model.backbone.encoder"
    put("stem", *folded(w, f"{enc}.convolution.convolution", f"{enc}.convolution.normalization"))
    for s, layers in enumerate(STAGES):
        for l, (k, stride, se) in enumerate(layers):
            src = f"{enc}.blocks.{s}.layers.{l}"
            dst = f"s{s}.{l}"
            put(f"{dst}.dw", *rep(w, f"{src}.depthwise_convolution", k))
            if stride == 1:
                a = f"{src}.depthwise_convolution.act.lab"
                out[f"{dst}.dw.act"] = ([2], [w[f"{a}.scale"].data[0], w[f"{a}.bias"].data[0]])
            if se:
                m = f"{src}.squeeze_excitation_module.convolutions"
                plain(f"{dst}.se.c1", f"{m}.0")
                plain(f"{dst}.se.c2", f"{m}.2")
            put(f"{dst}.pw", *rep(w, f"{src}.pointwise_convolution", 1))
            a = f"{src}.pointwise_convolution.act.lab"
            out[f"{dst}.pw.act"] = ([2], [w[f"{a}.scale"].data[0], w[f"{a}.bias"].data[0]])
    for i in range(4):
        plain(f"layer{i}", f"model.layer.{i}")
        for part, src in (("insert", "insert_conv"), ("input", "input_conv")):
            n = f"model.neck.{src}.{i}"
            put(f"neck.{part}{i}", w[f"{n}.in_conv.weight"])
            plain(f"neck.{part}{i}.se1", f"{n}.squeeze_excitation_block.conv1")
            plain(f"neck.{part}{i}.se2", f"{n}.squeeze_excitation_block.conv2")
    put("head.down", *folded(w, "head.conv_down.convolution", "head.conv_down.norm"))
    # The transposed convolution's weight is [in][out][2][2]: its BatchNorm
    # scales each output.
    up = w["head.conv_up.convolution.weight"]
    scale, shift = bn(w, "head.conv_up.norm")
    cin, cout = up.shape[0], up.shape[1]
    data = [v * scale[(i // 4) % cout] for i, v in enumerate(up.data)]
    bias = [b * s + t for b, s, t in zip(w["head.conv_up.convolution.bias"].data, scale, shift)]
    put("head.up", Tensor(up.shape, data), bias)
    plain("head.final", "head.conv_final")

    width = 4 if args.f32 else 2
    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", width, len(out)))
        count = 0
        for name, (shape, data) in out.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(shape)) + struct.pack(f"<{len(shape)}I", *shape))
            f.write(struct.pack(f"<{len(data)}{'f' if width == 4 else 'e'}", *data))
            count += len(data)
    print(f"{len(out)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
