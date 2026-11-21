"""Writes the source signals of the format fixtures and encodes them.

Run from this folder: `python make.py` (needs ffmpeg with libmp3lame and
libvorbis on PATH). The intermediate WAV files and the cover are deleted at
the end. The tests recompute both signals, so the formulas here and in
`playback/tests/formats.rs` must stay the same.
"""

import base64
import math
import os
import struct
import subprocess
import wave
import zlib

RATE = 44_100
LOSSLESS_FRAMES = 3_000
LOSSY_FRAMES = 20_000


def lossless(i, c):
    """24-bit integers: a triangle wave plus a small hashed noise."""
    t = (i + 100 * c) % 400
    tri = (t if t < 200 else 400 - t) - 100
    noise = (((i * 7919 + c * 104_729) * 2_654_435_761) % (1 << 32)) >> 24
    value = tri * 40_000 + noise - 128
    return value if c == 0 else -(value // 2)


def lossy(i, c):
    """A chirp from 100 Hz to 2 kHz at half scale (left), and the same
    chirp a quarter period later at quarter scale (right)."""
    t = i / RATE
    length = LOSSY_FRAMES / RATE
    phase = 2 * math.pi * (100 * t + (2_000 - 100) * t * t / (2 * length))
    if c == 1:
        return 0.25 * math.sin(phase + math.pi / 2)
    return 0.5 * math.sin(phase)


def write_wav(path, frames, bits, sample):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(RATE)
        data = bytearray()
        for i in range(frames):
            for c in range(2):
                data += sample(i, c).to_bytes(bits // 8, "little", signed=True)
        w.writeframes(bytes(data))


def png_1x1():
    def chunk(kind, body):
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xc0\x40\x20")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def flac_picture(png):
    """A FLAC METADATA_BLOCK_PICTURE (front cover), base64 for a comment."""
    mime = b"image/png"
    body = struct.pack(">II", 3, len(mime)) + mime + struct.pack(">I", 0)
    body += struct.pack(">IIIII", 1, 1, 24, 0, len(png)) + png
    return base64.b64encode(body).decode()


def strip_flac_padding(path):
    """Drops the PADDING metadata blocks the encoder leaves."""
    with open(path, "rb") as f:
        data = f.read()
    at, blocks = 4, []
    while True:
        last = data[at] & 0x80
        kind = data[at] & 0x7F
        length = int.from_bytes(data[at + 1 : at + 4], "big")
        if kind != 1:
            blocks.append(bytearray(data[at : at + 4 + length]))
        at += 4 + length
        if last:
            break
    for block in blocks:
        block[0] &= 0x7F
    blocks[-1][0] |= 0x80
    with open(path, "wb") as f:
        f.write(b"fLaC" + b"".join(blocks) + data[at:])


def ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


def main():
    write_wav("lossless.wav", LOSSLESS_FRAMES, 24, lossless)
    write_wav("lossy.wav", LOSSY_FRAMES, 16, lambda i, c: round(32_767 * lossy(i, c)))
    png = png_1x1()
    with open("cover.png", "wb") as f:
        f.write(png)
    tags = [
        "-metadata", "title=Format Test",
        "-metadata", "artist=Fixture Band",
        "-metadata", "album=Formats",
    ]
    bare = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]
    cover = ["-i", "cover.png", "-map", "0:a", "-map", "1:v", "-c:v", "copy",
             "-disposition:v", "attached_pic"]
    ffmpeg("-i", "lossless.wav", *bare, "-c:a", "flac", "lossless.flac")
    strip_flac_padding("lossless.flac")
    ffmpeg("-i", "lossless.wav", *cover, *bare, *tags, "-c:a", "alac", "alac.m4a")
    ffmpeg("-i", "lossless.wav", *bare, "-c:a", "alac", "alac.caf")
    ffmpeg("-i", "lossy.wav", *cover, *bare, *tags, "-id3v2_version", "3",
           "-c:a", "libmp3lame", "-b:a", "96k", "lossy.mp3")
    ffmpeg("-i", "lossy.wav", *bare, *tags,
           "-metadata", "METADATA_BLOCK_PICTURE=" + flac_picture(png),
           "-c:a", "libvorbis", "-q:a", "0", "lossy.ogg")
    ffmpeg("-i", "lossy.wav", *bare, "-ac", "1", "-t", "0.25",
           "-c:a", "mp2", "-b:a", "64k", "lossy.mp2")
    ffmpeg("-i", "lossy.wav", *bare, "-ac", "1", "-t", "0.1",
           "-c:a", "aac", "-b:a", "32k", "aac.m4a")
    for name in ["lossless.wav", "lossy.wav", "cover.png"]:
        os.remove(name)


if __name__ == "__main__":
    main()
