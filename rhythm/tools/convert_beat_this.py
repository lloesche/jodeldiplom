#!/usr/bin/env python3
"""Converts a Beat This! checkpoint into the weights file jodel-rhythm embeds.

    python -I tools/convert_beat_this.py small0.ckpt models/beat_this_small.bin [--f32]

The checkpoint is a PyTorch Lightning file: a zip holding a pickle and the
raw tensor data. It is read without torch and without running anything in
it: the unpickler rebuilds only dicts and tensors and refuses every other
class.

The output, little-endian:

    b"BTW1", u32 element size in bytes (2: f16, 4: f32), u32 tensor count
    per tensor: u16 name length, name (UTF-8), u8 rank, u32 per dimension,
                the elements in row-major order

Names are the model's own without the "model." prefix. Each BatchNorm is
folded into NAME.scale = weight / sqrt(running_var + 1e-5) and
NAME.shift = bias - running_mean * scale.

Used for models/beat_this_small.bin: small0.ckpt from the authors' share
(https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp), SHA-256
6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b.
"""

import argparse
import array
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict

EPS = 1e-5


class Storage:
    def __init__(self, kind, key, count):
        self.kind = kind
        self.key = key
        self.count = count


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def rebuild_tensor(storage, offset, size, stride, *_):
    return Tensor(storage, offset, size, stride)


class Kind:
    def __init__(self, name):
        self.name = name


ALLOWED = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
    ("torch", "FloatStorage"): Kind("float"),
    ("torch", "LongStorage"): Kind("long"),
}


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing {module}.{name}") from None

    def persistent_load(self, pid):
        tag, kind, key, _location, count = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {tag}")
        return Storage(kind, key, count)


def values(archive, root, t):
    """The tensor's elements in row-major order, as floats."""
    if t.storage.kind.name != "float":
        raise ValueError("only float tensors are converted")
    data = array.array("f")
    data.frombytes(archive.read(f"{root}/data/{t.storage.key}"))
    if sys.byteorder != "little":
        data.byteswap()
    out = []

    def walk(dim, at):
        if dim == len(t.size):
            out.append(data[at])
            return
        for i in range(t.size[dim]):
            walk(dim + 1, at + i * t.stride[dim])

    walk(0, t.offset)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--f32", action="store_true", help="single precision")
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.checkpoint)
    pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    root = pkl.rsplit("/", 1)[0]
    checkpoint = Reader(archive.open(pkl)).load()
    hparams = checkpoint.get("hyper_parameters", {})
    print("hyper parameters:", {k: hparams[k] for k in sorted(hparams)}, file=sys.stderr)

    state = OrderedDict()
    for name, t in checkpoint["state_dict"].items():
        name = name.removeprefix("model.").replace("_orig_mod.", "")
        if name.endswith("num_batches_tracked") or name.endswith("rotary_embed.freqs"):
            continue
        state[name] = (t.size, values(archive, root, t))

    # Fold every BatchNorm (it has a running mean) into a scale and shift.
    tensors = OrderedDict()
    for name, (size, data) in state.items():
        base, _, leaf = name.rpartition(".")
        if f"{base}.running_mean" in state:
            if leaf != "weight":
                continue
            weight = data
            bias = state[f"{base}.bias"][1]
            mean = state[f"{base}.running_mean"][1]
            var = state[f"{base}.running_var"][1]
            scale = [w / (v + EPS) ** 0.5 for w, v in zip(weight, var)]
            shift = [b - m * s for b, m, s in zip(bias, mean, scale)]
            tensors[f"{base}.scale"] = (size, scale)
            tensors[f"{base}.shift"] = (size, shift)
        else:
            tensors[name] = (size, data)

    code, width = ("f", 4) if args.f32 else ("e", 2)
    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", width, len(tensors)))
        count = 0
        for name, (size, data) in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(size)) + struct.pack(f"<{len(size)}I", *size))
            f.write(struct.pack(f"<{len(data)}{code}", *data))
            count += len(data)
            print(f"{name:60} {list(size)}", file=sys.stderr)
    print(f"{len(tensors)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
