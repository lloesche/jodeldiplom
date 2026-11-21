#!/usr/bin/env python3
"""Converts MusicalKeyCNN's KeyNet checkpoint into the weights file
jodel-rhythm embeds for finding a song's key (`jodel_rhythm::key`).

    python -I tools/convert_keynet.py keynet.pt models/keynet.bin

The checkpoint (torch.save of the model's state dict) is read as
convert_beat_this.py reads its: without torch, and without running anything
in it. The output is the same format (b"BTW1", single precision here); each
BatchNorm is folded into NAME.scale and NAME.shift likewise. Names are the
model's own (conv1.conv.weight, conv1.bn.scale, …).

Used for models/keynet.bin: checkpoints/keynet.pt of
https://github.com/a1ex90/MusicalKeyCNN at 61d8fc1 (MIT, see
models/LICENSE-musical-key-cnn), SHA-256
fc92072f1b9b19552ce3b74a9c8ce0cecb97633a12ae524ac0d93c05800e354e.
"""

import argparse
import importlib.util
import struct
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

# The safe reader of convert_beat_this.py, loaded from beside this file (the
# interpreter's isolated mode leaves this folder off the import path).
_spec = importlib.util.spec_from_file_location(
    "convert_beat_this", Path(__file__).with_name("convert_beat_this.py")
)
beat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(beat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.checkpoint)
    pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    root = pkl.rsplit("/", 1)[0]
    checkpoint = beat.Reader(archive.open(pkl)).load()

    state = OrderedDict()
    for name, t in checkpoint.items():
        if name.endswith("num_batches_tracked"):
            continue
        state[name] = (t.size, beat.values(archive, root, t))

    tensors = OrderedDict()
    for name, (size, data) in state.items():
        base, _, leaf = name.rpartition(".")
        if f"{base}.running_mean" in state:
            if leaf != "weight":
                continue
            bias = state[f"{base}.bias"][1]
            mean = state[f"{base}.running_mean"][1]
            var = state[f"{base}.running_var"][1]
            scale = [w / (v + beat.EPS) ** 0.5 for w, v in zip(data, var)]
            shift = [b - m * s for b, m, s in zip(bias, mean, scale)]
            tensors[f"{base}.scale"] = (size, scale)
            tensors[f"{base}.shift"] = (size, shift)
        else:
            tensors[name] = (size, data)

    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", 4, len(tensors)))
        count = 0
        for name, (size, data) in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(size)) + struct.pack(f"<{len(size)}I", *size))
            f.write(struct.pack(f"<{len(data)}f", *data))
            count += len(data)
            print(f"{name:40} {list(size)}", file=sys.stderr)
    print(f"{len(tensors)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
