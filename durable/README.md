# durable

Every file this workspace writes goes through here, so they all behave
alike: a crash or a power cut leaves the old version or the new one whole,
never a torn mix, and nothing half written ever has a file's name.

| | |
|---|---|
| `write_atomic(path, bytes)` | the bytes to a temporary beside the file, `fsync`, renamed over it, the folder `fsync`ed |
| `AtomicFile` | a file written piece by piece (and seeked, for a header filled in last) under a temporary name; `commit()` makes its bytes durable, then gives it its name, then makes the folder durable; dropped without, the temporary goes |
| `rename_durable(from, to)` | a rename, both folders made durable |
| `sync_dir(dir)` | a folder's entries made durable (a directory handle's flush on Windows) |
| `is_temporary(name)`, `sweep(dir, older_than)` | the temporaries a crash left (`NAME.PID-N.tmp`, and older writers' `.partial`, `.moving`, `.recording`, `.tmpPID`), removed once old |

Users: the player's settings, session, queues, grids, caches (analyses,
waveforms, lyrics), setup's downloads, stamps and packed model, the data
folder's move journal; the stems files (`jodel-playback`); the karaoke
recordings and `FileOutput::to_file` (`jodel-output`); `PackedModel::write`
(`jodel-lyrics`). A running log (the
player's trace) is the one file written as it goes, by design.

`cargo test -p durable`: a file replaced whole with nothing left beside it,
a file written in pieces and seeked has its name only once committed (and
none when dropped), temporaries told apart and swept only once old.
