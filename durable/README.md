# durable

Every file this workspace writes goes through here, so they all behave
alike: a crash or a power cut leaves the old version or the new one whole,
never a torn mix, and nothing half written ever has a file's name.

| | |
|---|---|
| `write_atomic(path, bytes)` | the bytes to a temporary beside the file in one unbuffered write, `fsync`, renamed over it, the folder `fsync`ed |
| `AtomicFile` | a file written piece by piece (and seeked, for a header filled in last) under a temporary name; `commit()` makes its bytes durable, then gives it its name, then makes the folder durable; dropped without, or the commit failed, the temporary goes |
| `AtomicDir` | a folder of files filled under a temporary name; `commit()` makes its entries durable, then gives it its name whole (a rename refused for a while, by a scanner on Windows, tried again for about two seconds), then makes the folder it is in durable; dropped without, or the commit failed, the temporary goes with what is in it |
| `rename_durable(from, to)` | a rename, both folders made durable |
| `create_dir_all(dir)` | the folders missing made, each new one's entry made durable in its parent (the writers above make their folders this way) |
| `sync_dir(dir)` | a folder's entries made durable (a directory handle's flush on Windows; nothing more where a file system cannot flush a folder) |
| `is_temporary(name)`, `sweep(dir, older_than)`, `STALE` | the temporaries a crash left (`NAME.PID-N.tmp` files and folders, and the player's own copies under way, `.partial` and `.moving`), removed once old (a day, at the players' sweeps); a folder is as old as the youngest file directly in it |

Users: the player's settings, session, queues, grids, caches (analyses,
waveforms, lyrics), setup's downloads, stamps and packed model, the data
folder's move journal, the karaoke's kept singing (a folder a song); the
stems files (`jodel-playback`); the karaoke recordings (`jodel-output`);
`PackedModel::write` (`jodel-lyrics`). A running log (the player's trace)
is the one file written as it goes, by design.

`cargo test -p durable`: a file replaced whole with nothing left beside it,
a file written in pieces and seeked has its name only once committed (and
none when dropped), a failed commit leaves no temporary, a folder filled
has its name only once committed (none when dropped or its name is taken;
a rename refused for a while waited out, on Windows), folders made level
by level, temporaries told apart and swept only once old.
