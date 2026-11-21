#!/usr/bin/env python3
"""Converts FCPE's bundled checkpoint into a weights file at full precision,
which `examples/pitch_pack.rs` quantises into the file jodel-rhythm embeds
for following a voice's pitch (`jodel_rhythm::pitch`).

    python -I tools/convert_fcpe.py fcpe_c_v001.pt fcpe-f32.bin

The checkpoint is read as convert_beat_this.py reads its: without torch, and
without running anything in it. The output is the same format (b"BTW1",
single precision). The output layer's weight norm is folded into
output_proj.weight (g · v / |v|, each output's row). Left out: what
inference does not use (the per-layer norms of the attention this model was
trained without, and a training mask). Names are the model's own.

Used for models/pitch.bin: torchfcpe/assets/fcpe_c_v001.pt of the torchfcpe
0.0.4 wheel (https://github.com/CNChTu/FCPE, MIT, see models/LICENSE-fcpe),
SHA-256 b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015.
"""

import argparse
import hashlib
import importlib.util
import math
import struct
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

# The safe reader of convert_beat_this.py, loaded from beside this file.
_spec = importlib.util.spec_from_file_location(
    "convert_beat_this", Path(__file__).with_name("convert_beat_this.py")
)
beat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(beat)

UNUSED = ("gaussian_blurred_cent_mask",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    raw = Path(args.checkpoint).read_bytes()
    print("SHA-256", hashlib.sha256(raw).hexdigest(), file=sys.stderr)
    archive = zipfile.ZipFile(args.checkpoint)
    pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    root = pkl.rsplit("/", 1)[0]
    checkpoint = beat.Reader(archive.open(pkl)).load()
    config = checkpoint["config_dict"]["model"]
    if not config.get("conv_only"):
        raise SystemExit("this converter is for the convolution-only model")
    state = checkpoint["model"]

    tensors = OrderedDict()
    for name, t in state.items():
        if name in UNUSED or (".encoder_layers." in name and ".norm." in name):
            continue
        if name.startswith("output_proj.weight_"):
            continue
        tensors[name] = (t.size, beat.values(archive, root, t))
    g = beat.values(archive, root, state["output_proj.weight_g"])
    v_t = state["output_proj.weight_v"]
    v = beat.values(archive, root, v_t)
    rows, cols = v_t.size
    folded = []
    for r in range(rows):
        row = v[r * cols:(r + 1) * cols]
        scale = g[r] / math.sqrt(sum(x * x for x in row))
        folded.extend(x * scale for x in row)
    tensors["output_proj.weight"] = ((rows, cols), folded)

    with open(args.output, "wb") as f:
        f.write(b"BTW1" + struct.pack("<II", 4, len(tensors)))
        count = 0
        for name, (size, data) in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<B", len(size)) + struct.pack(f"<{len(size)}I", *size))
            f.write(struct.pack(f"<{len(data)}f", *data))
            count += len(data)
            print(f"{name:50} {list(size)}", file=sys.stderr)
    print(f"{len(tensors)} tensors, {count} values", file=sys.stderr)


if __name__ == "__main__":
    main()
