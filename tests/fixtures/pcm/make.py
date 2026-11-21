"""Writes the two PCM fixtures of the engine and player tests.

Run from this folder: `python make.py` (needs ffmpeg on PATH). Each file is
8,193 stereo frames of integer ramps (an odd length, so the last block
is short), encoded losslessly as FLAC; the intermediate WAV files are
deleted at the end.
"""

import os
import subprocess
import wave

FRAMES = 8_193


def ramp(i, c, bits):
    """A sawtooth at half scale, the right channel a third of a period on."""
    span = 1 << (bits - 1)
    step = 97 if bits == 16 else 24_593
    return ((i * step + c * span // 3) % span) - span // 2


def write_wav(path, rate, bits):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        data = bytearray()
        for i in range(FRAMES):
            for c in range(2):
                data += ramp(i, c, bits).to_bytes(bits // 8, "little", signed=True)
        w.writeframes(bytes(data))


def main():
    for rate, bits in [(44_100, 16), (48_000, 24)]:
        name = f"ramp-{rate}-{bits}bit"
        write_wav(name + ".wav", rate, bits)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", name + ".wav", "-map_metadata", "-1",
             "-fflags", "+bitexact", "-flags:a", "+bitexact",
             "-c:a", "flac", name + ".flac"],
            check=True,
        )
        os.remove(name + ".wav")


if __name__ == "__main__":
    main()
