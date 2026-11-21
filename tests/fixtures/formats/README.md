# Format fixtures

Small files for `playback/tests/formats.rs` and `player/tests/browser.rs`,
made by `make.py` in this folder (`python make.py`, with FFmpeg 9.0 on
`PATH`, built with libmp3lame and libvorbis). The script writes two source
signals, encodes them and deletes its intermediate files; the tests
recompute both signals from the same formulas.

- **Lossless signal**: 3,000 stereo frames at 44.1 kHz, 24 bits, a
  triangle wave plus a small hashed noise (integers only).
- **Lossy signal**: 20,000 stereo frames at 44.1 kHz (not a multiple of
  any codec's frame), a chirp from 100 Hz to 2 kHz, written as 16 bits.

| File | Bytes | Content |
|---|---|---|
| `lossless.flac` | 9,123 | the lossless signal, FLAC (padding block removed) |
| `alac.m4a` | 9,688 | the lossless signal, ALAC in MP4, with title, artist, album and a 1×1 PNG cover (`covr`) |
| `alac.caf` | 8,993 | the lossless signal, ALAC in CAF |
| `lossy.mp3` | 6,447 | the lossy signal, LAME 96 kbit/s CBR with its Xing/LAME header (encoder delay and padding), ID3v2.3 tags and the cover (`APIC`) |
| `lossy.ogg` | 5,452 | the lossy signal, libvorbis q0 (64 kbit/s nominal), Vorbis comments and the cover (`METADATA_BLOCK_PICTURE`) |
| `lossy.mp2` | 2,089 | the first 0.25 s of the lossy signal, mono, MP2 64 kbit/s |
| `aac.m4a` | 1,321 | the first 0.1 s of the lossy signal, mono, AAC: for the "not supported" message only |
