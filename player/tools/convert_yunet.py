#!/usr/bin/env python3
"""Converts YuNet (OpenCV Zoo's face detector) into the weights file the
player embeds for finding faces on covers (`player/src/cover/faces.rs`).

    python -I tools/convert_yunet.py face_detection_yunet_2023mar.onnx models/yunet.bin

The ONNX file is read as plain protocol-buffer bytes with nothing but the
standard library: the graph's convolutions in their order, each with its
weight and bias (its BatchNorm is already folded in the export). Nothing in
the file is run.

The output is the format of rhythm/tools/convert_beat_this.py (b"BTW1",
single precision; the network is small), the convolutions named by what
they are:

    stem                         3×3 stride 2
    b{B}.{P}.pw, b{B}.{P}.dw     backbone block B (0 to 5), pair P: a 1×1
                                 convolution, then a depthwise 3×3 (ReLU after)
    lateral{L}.pw, lateral{L}.dw the neck, strides 8, 16, 32
    {cls,bbox,obj,kps}{L}.c1, .c2  the heads at stride 8, 16, 32

Used for models/yunet.bin: face_detection_yunet_2023mar.onnx of
https://huggingface.co/opencv/face_detection_yunet at
3cc26e7f1014a5ee5d74a42acee58bafc9d0a310 (MIT, see models/LICENSE-yunet),
SHA-256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4.
"""

import argparse
import array
import struct
import sys


def varint(b, p):
    v, shift = 0, 0
    while True:
        c = b[p]
        p += 1
        v |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return v, p


def fields(b):
    """The (number, wire type, value) of each field of message `b`."""
    p = 0
    while p < len(b):
        key, p = varint(b, p)
        num, wire = key >> 3, key & 7
        if wire == 0:
            v, p = varint(b, p)
        elif wire == 1:
            v, p = b[p : p + 8], p + 8
        elif wire == 2:
            n, p = varint(b, p)
            v, p = b[p : p + n], p + n
        elif wire == 5:
            v, p = b[p : p + 4], p + 4
        else:
            raise SystemExit(f"wire type {wire}")
        yield num, wire, v


def tensor(b):
    dims, name, raw, kind, floats = [], None, None, None, None
    for num, wire, v in fields(b):
        if num == 1:
            if wire == 2:
                q = 0
                while q < len(v):
                    d, q = varint(v, q)
                    dims.append(d)
            else:
                dims.append(v)
        elif num == 2:
            kind = v
        elif num == 4:
            floats = array.array("f", v)
        elif num == 8:
            name = v.decode()
        elif num == 9:
            raw = v
    if kind != 1:
        # Not a weight (the graph's shapes are kept as integers).
        return name, dims, None
    data = floats if raw is None else array.array("f", raw)
    if sys.byteorder != "little":
        data.byteswap()
    return name, dims, list(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("output")
    args = parser.parse_args()
    model = open(args.onnx, "rb").read()
    graph = next(v for num, _, v in fields(model) if num == 7)
    inits, convs = {}, []
    for num, _, v in fields(graph):
        if num == 5:
            name, dims, data = tensor(v)
            inits[name] = (dims, data)
        elif num == 1:
            inputs, op = [], None
            for n2, _, v2 in fields(v):
                if n2 == 1:
                    inputs.append(v2.decode())
                elif n2 == 4:
                    op = v2.decode()
            if op == "Conv":
                convs.append(inputs)
    names = ["stem", "b0.0"]
    for block in range(1, 6):
        names += [f"b{block}.0", f"b{block}.1"]
    names += ["lateral2", "lateral1", "lateral0"]
    pairs = []
    for name in names:
        pairs += [f"{name}.pw", f"{name}.dw"] if name != "stem" else ["stem"]
    for head in ("cls", "bbox", "obj", "kps"):
        for level in range(3):
            pairs += [f"{head}{level}.c1", f"{head}{level}.c2"]
    if len(pairs) != len(convs):
        raise SystemExit(f"{len(convs)} convolutions, expected {len(pairs)}")
    out = []
    for name, inputs in zip(pairs, convs):
        out.append((f"{name}.weight", *inits[inputs[1]]))
        out.append((f"{name}.bias", *inits[inputs[2]]))
    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", 4, len(out)))
        count = 0
        for name, dims, data in out:
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(dims)) + struct.pack(f"<{len(dims)}I", *dims))
            f.write(struct.pack(f"<{len(data)}f", *data))
            count += len(data)
    print(f"{len(out)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
