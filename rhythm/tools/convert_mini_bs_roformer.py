#!/usr/bin/env python3
"""Converts the Mini-BS-RoFormer weights into the file jodel-rhythm embeds.

    python -I tools/convert_mini_bs_roformer.py model.safetensors models/mini_bs_roformer.bin [--f16]

The input is a safetensors file: an 8-byte little-endian header length, a
JSON header naming each tensor's type, shape and byte range, and the raw
data. It is read with the standard library alone.

The output is the format of convert_beat_this.py, little-endian:

    b"BTW1", u32 element size in bytes (2: f16, 4: f32), u32 tensor count
    per tensor: u16 name length, name (UTF-8), u8 rank, u32 per dimension,
                the elements in row-major order

Names are kept as they are; the rotary frequencies (`rotary_emb.inv_freq`)
are left out, since our code computes them.

Used for models/mini_bs_roformer.bin: model.safetensors of
https://huggingface.co/HiDolen/Mini-BS-RoFormer-18M, SHA-256
61b54904c702c3a4ff52f761bab8ad364554179f493df468439b4fc3a3930d6e.
"""

import argparse
import array
import json
import struct
import sys

LEFT_OUT = ("model.rotary_emb.inv_freq",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("safetensors")
    parser.add_argument("output")
    parser.add_argument("--f16", action="store_true", help="half precision")
    args = parser.parse_args()

    with open(args.safetensors, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
        data = f.read()
    header.pop("__metadata__", None)

    tensors = []
    for name in sorted(header):
        if name in LEFT_OUT:
            continue
        info = header[name]
        if info["dtype"] != "F32":
            sys.exit(f"{name}: {info['dtype']}, expected F32")
        start, end = info["data_offsets"]
        values = array.array("f")
        values.frombytes(data[start:end])
        if sys.byteorder != "little":
            values.byteswap()
        tensors.append((name, info["shape"], values))

    width, code = (2, "e") if args.f16 else (4, "f")
    count = 0
    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", width, len(tensors)))
        for name, shape, values in tensors:
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(shape)) + struct.pack(f"<{len(shape)}I", *shape))
            f.write(struct.pack(f"<{len(values)}{code}", *values))
            count += len(values)
    print(f"{len(tensors)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
