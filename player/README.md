# jodeldiplom

A music player for PCM files (FLAC, WAV, AIFF, CAF, ALAC in
M4A/MP4), and for AAC-LC (in M4A/MP4), MP3 (MP2, MP1) and Ogg Vorbis, on top of
[`jodel-playback`](../playback) and [`jodel-output`](../output): output
through the system's mixer, beside every other application (a call's
screen share hears it), a plain file browser instead of a library database, in
a window: `jodeldiplom` (iced on wgpu, its CPU renderer without a GPU:
see [The window](#the-window); on macOS the app
`Jodeldiplom.app`). A Windows release build opens no console window;
started from a terminal, it attaches to that one, so `--help`, `--diag`,
`--list-devices`, `--clean-cache`, `--clean-diag` and errors print there
(a console prompt does not wait for a window program: pipe its output,
`jodeldiplom --diag | more`, or run it through `make`, to have it wait).

## Running

```sh
cargo run --release -p jodeldiplom -- [options] [path]
```

To build it for the system you are on, run `make` (or `make jodeldiplom`)
in the repository's root: `jodeldiplom` in `target/release` (or
`$CARGO_TARGET_DIR/release`), and on macOS the app `Jodeldiplom.app`
there too (signed ad hoc, through `macos/make-app.sh`).
The build is the whole player: the speech models are not part of it (the
player gets them itself, see Setup below), so a build can be shared as it is.
`make
diag` builds the player and runs its `--diag`. `make
clean-cache` forgets the beat analyses (an analysis and its waveform per
track) and the lyrics (the words found and our timings of them), so each
track is analysed, looked up and timed again when it is wanted. `make
clean-diag` forgets the record of the last analyses that `--diag` shows
(all it keeps of its own). Neither touches the settings, the beat grids
fixed by hand, the lyrics edited by hand or the speech model. `make clean` does both and removes the
build output; `make CARGO="cargo +1.100.0"` picks another cargo.

The player needs Rust 1.93 or newer; `jodel-playback`, `jodel-output` and
`jodel-rhythm` build with Rust 1.85.

`path` is a folder to start the browser in, or a file to play at once
(the browser opens on its folder); without it, the current folder (the
macOS app, which Finder starts in `/`, opens `~/Music`, or the home folder
without one).

| Option | |
|---|---|
| `--device ID\|NAME` | the output device (an id or a unique part of a name from `--list-devices`); default: the system default |
| `--headroom DB` | routes that change the samples play this many dB below full scale (`0`, `0.5`, `1`, `2`, `3`; default `0`), so overs need not clip: resampled routes (inter-sample overs) and floating-point files; lossy files (MP3, AAC, Vorbis) always at least 1.5 dB, as their decoded signal goes over full scale where the master was near it; files played as they are are never touched |
| `--config FILE` | the settings file (default: `config.toml` in the settings folder, see Folders and setup) |
| `--list-devices` | list the devices and what each takes (its rate and format, as the system mixes it) |
| `--clean-cache` | forget the beat analyses, the stems kept, the lyrics, the songs' loudness, the original singer's phrases and the covers taken apart in the data folder, and in the one a move is still taking it from (each made again when wanted); the models, the lyrics edited by hand and the best scores stay (`make clean-cache`) |
| `--clean-diag` | forget the record of the last analyses `--diag` shows (`make clean-diag`) |
| `--diag` | print what the player finds on this system, for a bug report: the build, the OS and processor, the settings in force, the cache folders, every output device with what it takes, and the GPU side (every adapter on every backend with the limits the separator needs, whether the separator sets up on the one it picks or the error, a 6 s chunk's time and how it splits between the kernels, the upload and the readback, and whether the beat analysis will listen to stems), then the last three beat analyses the player made (`recent-analyses.txt` in the analysis cache: when, the file's name without its folders, what the cues heard, the tempo found, and how long decoding, the stems, listening and deciding took; analyses taken from the cache are not counted). Plays nothing. The Windows window binary has no console: run it with `jodeldiplom` |

Options override the settings file for one run; settings changed in the
player are saved to it (only what was changed there, not the options).
The file holds `device`, `headroom`, `bpm`, `beat_voting`, `beat_methods = "flux, network"`
`beat_method = "network"`, `stems`, `stem_switch = "bar"|"instant"`, `skip_mix = "cut"|"mix"`, `skip_preset = "transition-tab"|"fade"|"drums-first"|"bass-swap"|"vocals-last"`
(what the next track key does), `transition_preset` and `transition_tempo =
"varispeed"|"stretch"` (the Transition tab's transition preset and tempo mode,
as last chosen), `transition_lyrics` (the lyrics on its vocals lane, `L`), and
the karaoke settings. A setting the player does not know is reported
and skipped; the rest of the file still loads.
For tests there are two options not in the usage: `--null` plays into a
null output (no device is opened) and `--exit-after SECONDS` quits; the
window also takes `--save-frame FILE.png`.

## Folders and setup

The player keeps two folders (`dirs.rs`):

| Folder | What | Where |
|---|---|---|
| Settings | `config.toml`, the session and the saved queues, the beat grids fixed by hand, the karaoke recordings, the record of a data move under way | fixed: `%APPDATA%\Jodeldiplom`, `~/Library/Application Support/Jodeldiplom`, `~/.config/Jodeldiplom` (`$XDG_CONFIG_HOME`); with `--config FILE`, that file's folder |
| Data | what the player makes or gets and could make again: `models` (the speech models), `analyses` (beat analyses and waveforms), `stems` (the stems kept), `lyrics` (the words found and our timings), `loudness` (the songs' loudness), `shine` (the original singer's phrases), `covers` (the covers taken apart); and the user's own: `lyric-edits` (the lyrics edited by hand: text, words moved, words not to be sung, the voice's level over them) and `scores.txt`, the best scores sung | `Jodeldiplom Data` in a folder of your choice (the settings row Data folder, `data_folder` in the file); by default in `%LOCALAPPDATA%`, `~/Library/Application Support`, `~/.local/share` (`$XDG_DATA_HOME`) |

Both are decided once, when the player starts, and every part of it is
given them: a player started with `--config FILE` keeps its session,
queues, grids and move record beside that file and its data where that
file says, and touches nothing of the usual player's.

**Data folder** (the settings' last row) opens a folder list: `enter` or
`→` opens a folder, `backspace` or `←` goes up (to the drives on Windows),
and the first row, **move the data folder here**, chooses the folder shown;
the data folder is made in it (or taken, if it is there already). Then
everything in the old one moves there (`mover.rs`), with a bar on the row:

- The move is written down first (`moving.txt` in the settings folder),
  then each file goes over alone: renamed where both are on one disk, else
  copied beside where it goes, made durable, renamed to its name, and only
  then removed where it was. Interrupted (the player quit, the system
  rebooted), the next start finds the record and goes on; nothing is lost
  and nothing done twice. A file already where it goes is the newer, and
  the old copy goes.
- A full disk stops the move with what is left, said on the settings page;
  `enter` on the row goes on. A file held open (Windows) is left for a
  later pass.
- While it moves, the player writes into the new folder and reads what has
  not come over yet where it still is: nothing is analysed, separated,
  timed or taken apart again, and the best scores load. That the copy
  already in the new folder is the newer holds because what the player
  writes there is either made afresh (an analysis, lyrics, a cover) or,
  for a file it adds to, made from what it read through the move: the
  best scores are written with every best the file held, wherever it
  was, so a best sung mid-move keeps all the earlier ones. A file the
  player removes goes from both folders, so an old copy never comes over.

**Setup.** Most of what the player needs is inside it; two models are too
big for that: the speech model that times karaoke's words
(`wav2vec2-base-960h`, 378 MB) and the speech recogniser that checks
fetched lyrics belong to the song (Qwen3-ASR-0.6B, packed at 4 bits to
589 MB). Their names, lengths and SHA-256s are built in (`setup.rs`). At
every start the player looks for them in `models` beside its executable
(or a macOS app's `Contents/Resources/models`), then in the data folder's
`models`; a file is taken once it matched its checksum (hashed the first
time, a stamp of its length and date in `checked.txt` sparing later starts
the hashing). With both there, it starts at once and never goes online for
them.

Otherwise a setup screen comes first, and cannot be skipped: each step says what it does and why
(downloading the word-timing model; downloading the recogniser's 1.9 GB
checkpoint, its vocabulary and tokenizer; packing it at 4 bits with the
calibration built in, then checking the packed file against its checksum:
packing gives the same bytes on every machine; removing the download), with
a bar for the step (its MB and speed, the time left) and one for the whole.
Before downloading it checks the disk has room, before packing that the
computer has the memory (about 7.5 GB). A failure says what went wrong and
what to do, with **Try again** (`enter`) and **Quit** (`q`): a part
downloaded whole is kept. About 45 s on a fast line. To spare a computer
the downloads, put the `models` folder beside the executable.

## Screen
One list pane with **Browser / Queue / Transition / Karaoke / Mixer**
tabs on its top edge (click one or press `tab`, which goes round all
five; the queue's tab carries its length); the browser shows where it
is above its rows, and the filter line at their foot. The command
palette (`:`) opens over everything, the window dimmed under it: a click
on a command runs it, a click beside it closes it. The right column has
the cover and under it, in line with it, the title, the artist and the
album; what the file is (the status light, `PCM`) beside the tempo and
key, a metronome before them lit on each beat; and the signal path as
tags joined by arrows, for example `file 48 → DAC`.
The output device is opened once, as the player starts, shared, through
the system's mixer: other applications play beside it, and a call's
screen share or "share sound" hears the music and the karaoke. It runs at
the rate the system runs the device at (set in the system's sound
settings, Audio MIDI Setup on macOS; 48 kHz as a rule), in the system's
mix format (32-bit float as a rule), and stays open until the player
quits: songs, stems, the preview and the game's sounds are all played at
it, so nothing is ever cut short by the device opening again between
songs, and the echo canceller keeps what it has learnt of the room. When
the system starts running the device at another rate (its format changed
in the sound settings), the song playing stops and the next one played
is routed to the new rate. A file at that rate plays as it is; any other
is resampled and the path shows the step in the
warning colour, for example `file 44.1 → resample 48 (160/147) → DAC`
(with headroom: `resample 48 (160/147, -1 dB)`); samples the resampler had
to clip are counted there too. The path says nothing of bit-perfection:
after the device's name comes the system's mixer, which may resample and
sets the volume. A lossy file plays 1.5 dB down (or the headroom
setting's, when more), as its decoded signal goes over full scale where a
loud master was near it: `file 44.1 decoded from MP3 (-1.5 dB)`. Its
path ends with what it was decoded from, for example `file 44.1 → DAC,
decoded from MP3`, and the status light names the codec and its bit rate
(`○ MP3 320 kbps · lossy`). Lossy files play gapless (encoder delay and
padding trimmed).
The right column's width is set by the window's alone (a third of a wide
one), so it stays put from track to track; where the path does not fit
it wraps before a `→` step or before what follows the device (`decoded
from MP3`), never inside a tag. Long titles, artists and albums end in
`…`. The title, artist
and album are links (underlined under the pointer): the title and the album show the track in its folder, the artist
the folder above it with the album selected (`T`, `B`, `R`).
At the bottom, the bar: the spectrum of what is heard (the mix the device plays,
the preview and its fades too, at the DAC's clock), a curve over a faint
grid with the level in dB on the left and the frequency under it; the
seek bar between the time heard and the length (press it and drag: the
pointer moves it from anywhere, and it seeks where it is let go of); the
transport buttons, the stems' switches (with a GPU), the message, and
the settings and the help. The wheel over it scrolls the list.
Clicking the cover (or `f`) opens Now playing: the cover as large as the
window holds it, with the title, artist and album (links, as in the right
column), the file's format and rate, the status light's word and the
tempo over its bottom; `esc`, a click elsewhere or `f` returns. Clicking
the device name opens its setting. `,` (or the button) opens the
settings, in sections (output, tempo and beats, stems, next track, data):
each row says what it is set to, with a switch, a check or a choice of a
few to click, and a click on a row does what `enter` does on it. The
output device (with what it takes: the rate the system runs it at and its
format) and the headroom (for what is resampled) come first. The device's rate is not a
setting of the player's: the system's sound settings set it, and a file
at another rate is resampled to it. A settings file from before, with a
`rate."<id>"` or a `resample` line, says once that it is an unknown
setting, and is written again without it.
Another device chosen opens it, the queue and the place carried over. The changes show at once and are saved at once; the engine gets them 400 ms after the last one, so stepping through
values reroutes the track once, without a break. A change never
stops playback: when the track has no route with the new settings, or its
new output does not open, it keeps playing as it was and a message says
so.

A setting that moves the track playing to another route (resampling,
the headroom) does so without a break, about half a second later (the new
route is decoded ahead and takes over at an exact frame, crossfaded over
20 ms). Below 72 columns the right column becomes a two-line summary; below
18 rows the spectrum goes.

`alt+↑` and `alt+↓` move the player's own volume a step of 1 dB, never
above 0 dB (also the palette's "Volume: up / down", and `↑` `↓` or the
wheel on the Mixer tab's master), and say where it is: the master's gain,
over everything the desk plays (the song, the preview, the game's
sounds), on every system. The system's volume is the system's: its keys
and sliders.

The stems of the track playing can be switched on and off: `alt+1`
drums, `alt+2` bass, `alt+3` vocals, `alt+4` other, `alt+0` all on (`option` on a Mac; also the palette's "Stems: …"
commands, and, with a GPU, the `drums` `bass` `vocals` `other` buttons
after the transport buttons, `D B V O` where the row is short; a stem
that is off is crossed out). Each stem is a strip of the Mixer tab inside
the player's (below), and these keys mute its strip or unmute it; `alt+0`
puts every stem's strip back as it starts (unmuted, 0 dB, not soloed).
With every stem's strip as it starts the file plays as it is; with
one touched the desk plays the stems in place of the file (a 10 ms
crossfade, the stems being the same music). A stem switched is heard within the
device's buffer: the stems are made beside the file as it plays, from the
stems kept (the queue keeps them as it gets tracks ready) or else
separated on the GPU, while a stem's strip is touched or the Mixer, Karaoke
or Transition tab shows (elsewhere none are made, so the GPU and the
processor rest). Without a GPU only the stems kept play. With the Stem
switching setting at **on the next bar** (the default) a key's change
lands on the first beat of a bar of the playing track, to the frame: the
first bar at least 0.1 s ahead (0.75 s while the stems are not there yet
and the GPU must separate them first, none being kept, so they are there
for it), during a transition too, and the message
says which bar; pressed
again before that bar, the change is taken back. The bar is held to when
it is heard: looping a bar, the change comes at the loop's end. Where the
playing track's bars are not known (the tempo off, the analysis not in
yet), and at **instantly**, the change is at once. The strips stay as
they are for the next tracks and seeks; nothing of the desk is kept, so
the player always starts with the file as it is. If the stems fail, the
track goes on as its file whatever the strips say, and a message says so.

Under the status light is the tempo of the track playing:
- `♩ 128 BPM` when it is steady;
- `♩ 98 BPM · 92–104` when it varies (the tempo where it is playing, then
  the range);
- `♩ … BPM` while it is analysed;
- `♩ – BPM` where the beat is unclear.

After the tempo comes the key, as its name and its place on the Camelot
wheel (`· E major (12B)`), the same on the Transition tab's header and
the Karaoke tab. It is heard from the first ten minutes of the track
([`jodel_rhythm::key`](../rhythm#the-key-key)); a key in the file's tags
(`INITIALKEY`, ID3's `TKEY`) is taken instead.

The `♩` lights up on every beat where the beat is clear, in the accent
colour on the first beat of a bar. The beats of the track playing and the
next two are analysed in the background with [`jodel-rhythm`](../rhythm),
in about 1.5 s per four-minute track with spectral flux alone and 4.5 s
with the network too; with stems separated on the GPU (through
[`jodel-gpu`](../gpu)), about 10 s on an RTX 5080. At start the player only
asks which GPU there is; the separator is set up the first time stems run. The results are kept in
`analyses` in the data folder (see Folders and setup below; one small text file per track and
choice of methods and stems, recomputed when the file or the analysis
changes), so a
track is analysed once. `M` (palette: "Tempo and beat pulse on / off")
turns all of this off and on; it is saved as `bpm` in the settings file,
and while it is off nothing is analysed.

After those, the whole queue is got ready in the background, from the
track playing to the end and then from the start: each track analysed
(beats, bars, stems and waveforms, the stems' audio kept) and its lyrics
looked up and, with the speech model there, timed word by word, all into
the caches, so the Transition and Karaoke tabs have them at once when it comes
up, and playing it on stems separates nothing. In the queue, left of
each track's format, five icons are grey until the track has that thing
and light up once it has, in the order the work finds them, so they
light one after the other as the queue is got ready: `♯` its key (found
as soon as the file is decoded), `▥` its stems (separated alongside, on
the GPU), `♩` its tempo (decided last, from the beats and the stems),
`¶` its lyrics (looked up and timed alongside, by another worker; a song
no source has lyrics for is heard from its voice instead, once a run, and
the icon lights once it is) and `◐` its cover taken apart (for Karaoke
mode's background). A step's last icon (the tempo, the lyrics, the cover)
lights only once that step is done, so with every icon lit nothing is left
to do for the track. They show where the row has room for its title too. Left of its number, a
spinner turns while work on the track is on (its analysis, its lyrics
looked up or timed) and a dot shows while some is still to come. What is in the cache is handed over at once
when it is wanted, even while another track is being analysed.

The settings (`,`) choose how beats are found, below the engine's rows:

| Row | What it does |
|---|---|
| Tempo and beats | the same switch as `M` |
| Beat voting | on: the checked methods vote on one beat grid; off: one method decides |
| Spectral flux | onsets through two trackers; `[x]` / `[ ]` while voting, `(●)` / `( )` without |
| Beat This! network | our own inference of a neural network that knows tempo levels, on the GPU where there is one (else on half the processors); the same marks |
| Stems | on: the meter's cues listen to stems (bass, drums, other, vocals) separated on the GPU, and the drum stem's flux votes where the drums play; off: the cues listen to the mix. Without a GPU it shows `unavailable` |
| Stem switching | on the next bar: `alt+1` to `alt+4` (and the stem buttons) are heard from a bar's first beat; instantly: as the key is pressed (above) |
| Next track (N) | what `N` does: cut, the next track starting at once, or mix into it, beat-matched; `X` switches it too |
| Its transition | the preset `N` mixes by: the Transition tab's as it is set (the default), or one of its own (fade, drums first, bass swap, vocals last) |
| Data folder | where the models, analyses, stems and lyrics are kept: `enter` opens a folder list to move them into another folder (see Folders and setup); while they move, a bar |

While voting, at least one method stays checked. Without voting the
methods are a choice of one, and the checked ones are kept for when voting
comes back. Both methods voting is the default and the surest; flux alone
is the fastest, but often reads rock and hip-hop at twice their tempo.
Stems are on by default where there is a GPU; if the GPU fails on a
track, that track's cues listen to the mix and a message says so.

Keys: `?` shows them all, `:` runs any command by name; both are generated
from the one keymap table (`src/core/keymap/table.rs`). On the help, `/` searches
it: only the keys whose name or text contain what you type stay, under
their section's title (`volume` leaves `alt+↑` and `alt+↓`); `enter` keeps
the search, `esc` clears it, and a second `esc` leaves. `space` play/pause, `←→`
seek, `shift+←→` previous/next track, `↑↓` move, `enter` opens a folder and
queues a file or playlist (the queue is never replaced by it),
`shift+enter` queues one and plays it at once (on a folder: queues it all),
`alt+enter` plays a file with its whole folder (or a folder or playlist)
in place of the queue; a double click does what `enter` does (opens a
folder, queues a file or playlist, plays a queue row), a right click
queues. `backspace` goes up (on Windows a drive's root has the list of
drives above it), `/` filters the folder as you
type; `enter` then goes to the folder, file or drive typed (`D:`,
`..\Other`, `~/Music`) if it exists, and otherwise searches every
subfolder for it. `ctrl+v` (`cmd+v` on a Mac) pastes the clipboard's
text into the filter or the palette; on the list it opens the filter with
it, so a copied path is one `enter` away. Line breaks and tabs in what is
pasted become spaces. `tab` always switches between Browser and Queue. `N`/`P` next/previous (`X`: whether
`N` cuts or mixes into the next track, below), `T`/`B`/`R`
go to the playing track, its album or its artist folder, `Q` or `ctrl+c`
quit.

Holding `ctrl` in the browser previews, as tuning a radio: once it has
been held alone for 0.3 s (so `ctrl+s`, `ctrl+l`, `ctrl+v` and `ctrl+c`
do what they do and preview nothing) the file selected plays from a third
of its length (its start when the length is not known), fading in from
its first sound while the song playing fades out under it (0.2 s, equal
power) and runs on unheard. While `ctrl` stays down, `↑↓`,
`pgup`/`pgdn`, `home`/`end`, a click, the wheel and the pointer over a row
select another file, which crossfades in from its third in turn; a
folder selected is silence. Fast moves never pile up: only the file
selected last is opened. Releasing `ctrl` fades the preview out and the
song back in where it has got to (a song paused stays paused). Each file
plays on a player of its own, a guest on the mixer that holds the device,
at its format (resampled when it must be); each crossfade starts both its
sides on the frame of the new file's first sound, on the mixer's clock.
The song's player, its queue and its place in it are not touched; `ctrl+enter` does what `enter` does (queues the file)
and ends the preview, as any other key or click does first. The status
line says `Preview: <title> from a third in · release Ctrl to go back`
meanwhile.

In the browser and the queue, typing jumps to names as in a file explorer:
lowercase letters, digits and other name characters select the first name
(as shown, the title once read) starting with what you type (a pause of a
second starts over; the same letter again moves to the next such name).
While a name is being typed, `space` is part of it (`02 b`); otherwise it
plays and pauses. Any other key or click, and opening or leaving a folder,
starts the next name afresh.
Lowercase letters never run commands there; commands use capitals (`Q`,
`N`, `P`, `S`, `F`, `T`, `B`, `R`, `M`, `G`, `X`, `J`, `K`) and the other keys.
The Transition tab has no names to type, so its transition keys are lowercase.

In the queue, `del` or `backspace` takes the selected track out (the one
playing gives way to the next), and `K` / `J` (or `shift+↑` / `shift+↓`)
move it up and down, as does dragging it with the mouse; playback goes on
as it was, a transition whose next track changes is called off.
`ctrl+s` saves the queue as a playlist (M3U8, which other players read)
among the saved queues, named after its first album and its length
(`Evolve – 12 tracks`); `ctrl+l` shows them in the browser, where `enter`
queues one and `alt+enter` plays it in place of the queue. Any `.m3u` or
`.m3u8` in the browser works the same way.

The session comes back at the next start: the queue (with the track that
was playing, or the row selected in it, selected; nothing plays) and the
browser's folder and row (unless a folder or file is given). It is saved a
couple of seconds after it changes and on quitting, as `session.m3u8`
in the settings folder; the saved queues are in `queues` there.

### The Transition tab

**Transition** takes the whole width above the transport bar, with no cover,
track text or spectrum, on a panel under the tabs. It shows the track
playing (**Now**) and below it the next one in the queue (**Next**), each
with:

- **a header:** the block's name (in the accent, with a mark, on the track
  the keys move), the title, and its facts as chips at the right: the
  tempo, the key, the bar the cursor is in (`bar 23 / 96`), `no stems`,
  `grid fixed by hand` where it was, and on the playing track the
  transition set up (below: its preset and tempo mode, `armed` or `mixing`
  lit, what to do next, the next track's tempo and the glide); in a narrow
  window the facts that matter least go first;
- **an overview:** the whole track in one row, with the playhead, the
  span the lanes below show (the rest of the track shaded, when they show
  less than all of it), and the selection;
- **a ruler:** the bar numbers (in the window, each starts at its bar's
  line, to the pixel, so they scroll with the waveforms), with the cursor
  as `▼` and the track's transition cue as `◆` in front of its number;
- **the lanes:** `song`, and with stems `drums`, `bass`, `other` and
  `vocals`, each labelled by a small tag in its colour (lit while its fade
  is edited), as tall as both tracks fit (18 to 64 points a lane; the next
  track's block only where it fits with 18). Bars start where a lane is shaded (a line in the
  window), the selection is tinted, the playhead is the accent colour,
  stretches where the beat is unclear are dimmer, and a stem turned off
  (`alt+1` to `alt+4`) is grey. The `vocals` lane carries the song's
  lyrics across its middle once they are timed (see Karaoke): each word
  where it is sung, each letter in the cell where the aligner heard it and
  a line where a sound is held (`Mi───ster`, `go────`; a word whose letters
  were not heard spread by syllables), the word heard now in the accent
  colour; where words crowd (zoomed out) those with no room are left out,
  and the words are laid out from the song's start, so scrolling never
  moves them. `L` turns them off and on (`transition_lyrics`); `z`, or the
  `fit lyrics` button at the right of the tab's footer, zooms the
  track selected (`↑` `↓`) to the widest zoom that leaves out none of its
  words; the other track keeps its zoom.

The waveforms come from the beat analysis, so the Transition tab needs the tempo
on (`M`). The stem lanes need stems (a GPU and the Stems setting); without
them there is the song lane alone. The lanes measure the peak and RMS of
each 10 ms, on a scale from −60 dBFS to full scale.

The window paints every lane and bar numbers once, in tiles, and shows
them from there (on the GPU, or by the processor without one): while the
playing track's view follows the playhead it scrolls them at every refresh
of the display by itself, a view moved by hand shows the tiles it already
has, and a mark that moves (the cursor going along, a cue, the fades)
paints only the tiles it is in.
On the developer's machine the Transition tab goes by at about 130 frames a
second on 12 % of a core.

Both tracks show the same span of time, so their tempi compare at a glance:
1, 2, 4, 8, 16 (to start with), 32 or 64 bars of the playing track, or both
tracks whole. The playing track's lanes follow what is heard (it sits a
third from the left) until the cursor moves; `.` follows again. The next
track's start at its first bar; while a transition is armed or mixing, they
follow the next track too, with its playhead (once it is heard) right
under the playing one's. When the next track takes over, its block slides
up into the playing track's place, and the one after it comes in below.

| Key | Transition tab |
|---|---|
| `↑` `↓` | the playing track / the next one |
| `←` `→` | the cursor a bar back / on (`alt`: a beat) |
| `shift+←` `shift+→` | select bars back / on |
| `pgup` `pgdn` | eight bars |
| `+` `-` | zoom in / out |
| `z` | the track selected to the widest zoom that leaves out no word of its lyrics |
| `enter` | the transition's cue on this track: the bars selected, else eight from the cursor's bar (playback stays as it is) |
| `.` | follow the playhead again |
| `L` | the lyrics on the vocals lane off/on |
| `1` … `9` | when the bar heard ends, jump to a bar's start: `5` the bar heard (it plays once more), `6` the next, `7` the one after, `4` the one before, … |
| `shift+1` … `9` | the same at once (`shift+5`: the bar heard starts again) |
| `[` `]` | the bars start a beat earlier / later |
| `{` `}` | every beat 5 ms earlier / later |
| `<` `>` | half / double the tempo |
| `D` | a bar starts on the cursor's beat (bars of four from there) |
| `U` | undo the grid fixes: the beats as the analysis found them |
| `p` | a transition: the next preset (named in the header; its fades show on the tracks cued) |
| `m` | a transition: varispeed / time-stretch |
| `r` | a transition's beats paired by the tempi, 1:1, 2:1 (the next track at half tempo), 1:2 (at double) |
| `e` | a transition: edit its fades (below) |
| `g` | arm the transition at the two cues, or call it off (not once it is mixing: it plays to its end) |
| `esc` | stop editing, else clear the selection, else call off a transition armed, else clear this track's cue (it never leaves the tab: `tab` does) |

A bar jump is heard some 30–60 ms after the key, crossfaded over 20 ms
from the bar that would have come next, never with a gap: the engine
keeps 24 s of the playing track either side of what is heard (gigabytes
are not needed: some 20 MB at 44.1 kHz, 150 MB at 384 kHz). With a stem
off the same: the stems are made beside the file, so a jump takes them
along.
Striking `5`
again and again stutters the bar. `shift` with a digit counts on any
keyboard layout. Not while a transition is set up.

The footer shows the zoom of the track selected (or, while a fade is edited,
which and at which bar) and the main keys as key caps; those that do not
fit go from the end.

`space` still plays and pauses, and `←` `→` move the cursor here instead
of seeking (the progress bar still seeks). With the mouse, a click on a
lane puts the cursor there, a double click sets the cue there (as
`enter`), a drag selects bars (following the pointer anywhere, until it is
let go of), the wheel zooms the track under the
pointer around it (`+` `-` zoom both), a drag with
the right button moves the view (the waveform goes with the pointer; over
the overview, the span shown does), and over the overview the wheel
scrolls and a click seeks there on the playing track (its view following
the playhead again) and moves the view there on the next one.

#### Transitions

A transition crossfades from the playing track into the next one, stem by
stem, with the next track's beats on the playing one's. It runs between
two cues, one on each track, set with `enter`: click (or move the cursor)
in the playing track and press `enter`, the same in the next track, then
`g`. A cue covers the bars selected (drag, or `shift+←` `shift+→`), else
eight bars from the cursor's bar (while the playing track's cursor
follows the playhead, from the first bar at least 4.5 s ahead, the
soonest the engine can get the tracks ready for). The window starts at
the playing track's cue and the next track comes in at its own; where
the two cues differ in length the window lasts as long as the longer one
(a cue selected outweighs one of the default eight bars). `enter` never
moves playback. `p` goes through the presets, named in the header; their
fades show on each track once it has a cue:

| Preset | |
|---|---|
| fade | everything crosses over evenly |
| drums first | the next drums come in over the first quarter, the bass swaps on the middle bar, the other instruments cross in the second half, and the vocals swap on the last bar |
| bass swap | everything fades, but the bass swaps hard in the middle |
| vocals last | the playing track's vocals stay to the end, the next one's come in after |

`m` chooses how the tempi meet: **varispeed** plays a track faster or
slower (its pitch moves with it), **time-stretch** changes its tempo and
keeps its pitch. The tempo glides across the window: the playing track
starts it at its own tempo, the next track follows its beats, and by the
window's end the playing track is at the next one's tempo and the next
one at its own (the glide changes the speed by 30 % at most; the rest
the next track makes up after the window). It then plays as it is
again where it did before.

Where the two tempi are far apart, the beats pair two to one rather than
one to one, whichever needs the smaller change (the tempi measured from
the beats where the tracks meet): a track at 60 BPM into one at 120 is
not sped up; one of its beats falls on every other beat of the other,
and its bars on every other bar. The cursor on the faster track then
moves two bars at a time, a window covers whole bars of the slower one,
and the header shows the pairing (`next 2:1 +3.1 %`) with the glide
(`tempo 128 → 132 across the window`). `r` sets the pairing by hand
(`(set)`), round to by the tempi again.

The window is tinted on both tracks, each lane is as bright as its gain,
and a line over it draws the fade. The header says the preset, the mode,
`(edited)`, `enter sets the cues`, `g arms`, `armed` or `mixing`, and how far apart the tempi are
(`next −7.2 %`).

`e` edits the fades by hand, bar by bar, on the eight lanes (the playing
track's four stems, then the next one's):

| Key | Editing the fades |
|---|---|
| `↑` `↓` | the lane |
| `←` `→` | the bar |
| `+` `-` | the gain there a quarter up / down |
| `0` `1` | off / full there |
| `c` | a hard cut there, or a fade again |
| `esc` `e` | done (the edits stay; a preset starts afresh) |

`g` arms it, once both tracks have a cue and the playing track's has not
been heard yet: it runs when the playing track reaches its cue. While it
is armed, edits, `m` and a new cue go to the engine at once; `g` again
(or `esc`) calls it off, as do a seek and a new queue, and the cues stay
for another go. Called off inside the window, the playing track goes on
alone.
Without stems every lane is the whole track, so every preset is a
beat-matched fade. The desk mixes the two tracks: the next one plays on
an input of its own on the same strips, locked to the playing one's
frames, and the fades are the desk's automation on each. The stems'
strips act on both tracks as they are set (a stem muted stays out of
both, a fader's level goes into both fades, a stem key lands on its bar
during the mix too), and nothing jumps where the next track goes back
onto its file after the mix. The device stays at the rate the mix played
at for the tracks after it, gaplessly, until you pick a track or stop
(or a track cannot play at that rate).

The next track key (`N`, `shift+→`, `▶▶`) can mix too: `X`, anywhere,
switches what it does, **cut** (the default, as ever) or **mix**.
Mixing, `N` arms a transition into the next track from the first bar at
least 4.5 s ahead, over eight bars, in the tempo mode `m` set, by the
Transition tab's preset as it is set then (`p`, its fades as edited), or
by a preset of its own chosen in the settings' row under it, Its
transition; `N` again before it starts cuts at once, and `N` while it
mixes cuts into the next track where it is heard (the playing track
stops there, the next one plays on alone). Where it cannot (no analysis
yet, no bars), it cuts and says why. Both are saved (`skip_mix`,
`skip_preset` in the settings file) and are rows of the settings, Next
track (N) and Its transition. The Transition tab's own preset (`p`) and
tempo mode (`m`) are saved too, so the next run starts from them.

The grid fixes are kept per track, by its path and size, in `grids`
in the settings folder, not with the cached analyses in the data folder, so a
new analysis never throws them away. The tempo, the beat pulse and the
Transition tab all use the fixed grid. The waveforms are cached with the
analyses (a `.wave` file beside each).

### Karaoke

The Karaoke tab (`tab` to it) sets karaoke up: the voice's level (off by
default) and the microphone, saved in the settings file (`karaoke_voice`,
`karaoke_mic_device`). The Voice slider is on the faders' taper (the
Mixer tab) from −∞ up to 0 dB, the voice at its own level, and shows its
level in dB; the settings file keeps where it is along its travel, 0 to
100 % (25 %: −30 dB, 50 %: −17 dB, 75 %: −8 dB).
On the tab, as in the mode, the voice plays at its
level (a gain on the vocals' lane on the desk, over the VOCALS strip,
which stays as you set it; heard at once), so what you hear, the meter
and a recording there are as in the mode; on other tabs that lane is at
full again. Without a GPU only songs whose stems are kept have their
voice turned down.
`esc` on the tab does nothing. The tab also shows what was found of the
lyrics of the song and the next two, and the microphone with a meter of
the singer. Under it, where the other tabs have the spectrum, is the
playing song's voice (its stem, as on the Transition tab's `vocals` lane) with
its words on it as they are sung, the playhead a third of the way in, each
word where it is sung. Its zoom is fixed for each song: the widest at which
no word is left out, so every word can be read (a song sung fast goes by
faster). In the window the words are two
cells tall, and the lane goes by at a frame for every refresh of the display
(60 a second where its rate is not known) on a clock eased
from the positions reported, so it moves evenly. With the GPU showing the window, the lane is painted once, in tiles 512
pixels wide, and the GPU scrolls it between frames by itself (to a share
of a pixel), the rest of the window drawn at 30 a second: following it
costs little (on the developer's machine 12 % of a core with the song
playing, from 37 %).

In the window the lane also shows the notes sung: a bar for each, on a row
a semitone apart across the song's range, the rows of its key's scale
lined faintly across the lane (the tonic's a little brighter). The pitch
is heard in the voice's stem as it is separated, by
[`jodel_rhythm::pitch`](../rhythm#the-pitch-pitch) (about 1.5 s a song on
the processor), and kept with the track's waveforms. A note is the median
of the frames around it the network is sure of, to the nearest semitone.

The microphone (the system's default input, or the one `i` picks) is open
while the tab or the mode shows (and on the Mixer tab while its strip is
selected), to hear the singer: it is never played.
Nothing of it reaches the speakers, so what plays is exactly what it
would be without it, and nothing can feed back. It sees what the device
plays (a tap at the output's last step that only reads), so the music's
echo can be taken out of what it hears (`karaoke_echo = true` in the
settings file, the default): an echo canceller learns from the music how
what plays comes back from the speakers, left and right each its own way
(within a second or so), and subtracts it, and takes nothing else out, so
the singer always passes whole. What it cannot predict of the echo (on a
laptop's own speakers, most of what they play apart and their limiter) is
then turned down where it outweighs the voice, for the pitch the game
hears; a singer much quieter than that leftover loses a few dB there,
which the pitch does not mind. The meter shows the singer isolated from
that voice: a small
network trained on singing takes out what the canceller leaves of the band
(through a laptop's speakers, most of it) and the room's own sounds (a
fire, a fan), so only the voice moves it. The tab says how many dB of the echo
go, and how late it comes, once it has learnt the room (it finds echoes up
to a second late, from a wireless speaker too). The voice is kept block by
block with where in the song it was sung, for scoring the singing. With
`JODELDIPLOM_TRACE` set, the trace gets a line a second on the microphone
and the canceller. `r` (on the tab or in the mode) records 30 s of what
the canceller works with, as a WAV file of five channels of 32-bit float
(the microphone, the music going to the left and the right speaker, the
voice the canceller gives, and that voice as the game hears it, its
leftover turned down; the extensible header with a channel layout, which
every player reads), in
`recordings` in the settings folder: sing over the music while it
records, to tune the canceller to a room. On macOS the system asks once
whether the player (the app, or the terminal it runs in) may use the
microphone; until it may, the microphone gives only silence, and the tab
says so.

Lyrics come from the file's tags, a `.lrc` file beside it, or LRCLIB
(lrclib.net; `karaoke_online = false` turns that off), and are cached in
`lyrics` in the data folder. Only their words are taken: whatever
times they come with (LRC line times, Enhanced LRC word times, LRCLIB's
synced lyrics) are dropped and never kept. Every line and word is timed
by us, against the voice: with the speech model
(`facebook/wav2vec2-base-960h`, Apache 2.0, 378 MB, got by setup with the
speech recogniser, `qwen3-asr-0.6b-q4.jdlasr`, `jodel_gpu::GpuAsr`), the song's voice is
separated on the GPU and the whole text aligned to it at once
(`jodel_gpu::GpuSpeech`, about a second a song; without a GPU
`jodel_lyrics::speech` on half the processor's threads, about half a
minute). Before that a text from LRCLIB is checked against what the
speech recogniser hears in the voice (on the GPU, about 4 s a song; in
parts of 30 s): when under 35 % of its words were heard, in order, LRCLIB's
other texts of the song are compared too, and the best that fits is the
one timed and kept instead (`jodel_lyrics::prepare`, `jodel_lyrics::fit`). A text
none of whose fellows fits is kept as it is: the recogniser also misses
some texts that fit (German heard as English, heavily processed voices).
The file's own texts (tags, `.lrc`) are heard the same way, for their fit
and to pin their words (below), but never swapped. The fit is kept with the
timed lyrics (a `[fit:78]` line, `[fit:64 swapped]` for a text swapped in) and the
Karaoke tab says it: `fits the song (78 % heard)`, `its first text did not
fit the song, this one does`, or `may not fit the song (only 31 % heard)`. The result is cached as Enhanced LRC, with where each letter was heard (a `[#:letters ...]` line other players pass over; a song timed before that is timed again once). A word fills in Karaoke mode letter by letter as each is sung, so a held vowel holds the fill. A line is sung from its
first word to its last as timed. Until a song is timed its first lines
show, untimed. `cargo run --release -p jodeldiplom --example lyrics_align
-- MODEL FOLDER` measures how close the aligned line starts come to songs'
own LRC times (those are only compared against, never used).

A song none of those has lyrics for is heard instead: the speech
recogniser writes down what the voice sings, the speech model times it as
any other text, and its lines are made where the timing says the singer
breaks (`jodel_lyrics::lines`): the most likely breaks over the whole song,
from the pauses, how long the word before lasts against the song's words,
how likely each line's length is (in words and seconds, 12 words at most)
and, weighing most, the recogniser's stops and commas. Learnt from 33
songs with real lines (`--example line_bench`), its weights tuned on 31
real transcripts of songs whose lines are known (`--example line_asr`,
half tuning, half scoring), it finds the real breaks at F1 0.64 (0.75
within a word; cutting at stops, else every 9 words: 0.60 and 0.71).
Keeping lines to the bars helps rap whose lines are a bar long, but
nothing overall; rhymes added nothing measurable. Such lyrics come from no trusted source and may well be wrong, and
the player says so everywhere: the Karaoke tab names them `heard from the
voice by machine learning (no trusted source)` in orange with a warning
line under them, Karaoke mode shows the warning under the song's title as
it starts, and the queue's `¶` is lit orange rather than yellow. They are
cached marked as heard (`[source:transcribed qwen3-asr-0.6b-q4 2]`). A
song in which too little is heard to make lines of is kept as having no
words.

Two things keep a quiet voice from misleading the timing. The speech model
brings each piece it hears to one level, so that a soft verse is heard as
well as a loud chorus; but a piece is never lifted more than 15 dB above
the level of the song's singing, or an intro holding nothing but the
separation's leftovers would be lifted to speech and heard as letters.
And a word may be held for as long as the voice sings on (a long note is
heard as little more than its first letters), but its letters are not
spread across a pause in the voice; between words a pause is free.

Two more keep the lines in their place when something else in the voice
sounds like them (an intro's chant, a sample, a spoken part). Where the
speech recogniser heard a line, in the parts of 20 to 30 s it hears a
song in for the fit check, the line costs to start outside that part
(`jodel_lyrics::anchors`): only runs of three words heard together count,
not a line whose words were heard in another part too (a chorus), and
not a part where the recogniser fell into a loop. And the song is aligned
twice: the second time, the words before the first four heard surely in a
row are pulled toward them, so a first line heard weakly stays near the
lines after it instead of going to anything sung earlier.

The parts say where a line is to half a minute; the recogniser also hears
the voice in glimpses of 3 s every 1.5 s (`jodel_lyrics::glimpses`, about
half a minute a song on the GPU; a glimpse where the voice is silent is
not heard), which say it to a second or two, and where nothing with words
is sung at all. The speech model hears letters in sounds without words: a
chant's "yo" as "you", a choir's "oh", a synth left in the voice. In the
second alignment each line's first word costs to start outside where the
glimpses near it heard it start (those that heard its first words, less
the middles of those that heard none of them), and a line costs for every
second it would go on over a stretch where the glimpses heard no words.
A line's first words heard too weakly to time are put just before its
first word heard surely, at a sung pace, not stretched back toward the
line before. On Kate Bush's "Running Up That Hill" (a "yeah, yeah, yo"
before each verse line, a choir's pads between the verses) this took the
lines within 1 s of LRCLIB's from 40 of 46 to 43, and the words no longer
run on over the pads. On 153 songs (`--example lyrics_set`) the lines
within 1 s went from 81.4 % to 82.6 % (English) and stayed at 81.3 %
(German), the first lines over 3 s off stayed as they were, and the lines
running on over 1 s past the next one's start went from 466 to 420
(English).

A text often writes a chorus or hook once that is sung more often; with
nothing in the text for the second time, the words around it were
stretched over it and the lines after it drifted. Where the recogniser
heard a run of words the text has nothing for, and those words are lines
the text has elsewhere, sung up to four times in a row, the lines go in
there again before the timing (`jodel_lyrics::repeats`), and are shown as
sung. On 152 songs it filled in 22, DMX's "Ruff Ryders' Anthem" its hook
seven times over where its text has it once.

A song's lyrics can be edited by hand (`core::lyric_edit`, on the lyrics
editor's screen: `e` or Edit lyrics on the Karaoke tab): a word moved, made longer or shorter, marked as
not to be sung (a backing vocal, an ad-lib), the voice's level set over
it, a word or a line replaced, put in or taken out, every change undone
and redone. The edits are the user's own (`lyric-edits` in the data
folder, `lyrics::edits`): their text is the song's from then on, and the
song is timed again with them, the words moved by hand held exactly where
they were set and the rest aligned around them (`jodel_lyrics::ctc`); a
new aligner keeps them so. A word not to be sung gets no bubble and is
not scored; in the mode the voice is turned up over it, 6 dB over the
Voice slider's level and to at least −6 dB, so it is heard even with the
voice all the way down (smoothed over 50 ms; a level set over a word
takes its place): the vocals part's automation on the desk, timed on the
playing head's frames (`core::voice_curve`).

The **lyrics editor** shows the song's words as boxes over its voice
stem's waveform (the whole song's where it has no stems), where they are
sung, the time across the top and the voice's level over each word under
it; the word selected, its line, and the bottom bar under it all. A word
set by hand carries a pin; one not to be sung is dimmed. The song plays
from the word under it (`space`), the view paging after the playhead, so
the timing is heard while it is set; the song timed again with each
change comes back to the open editor. Revert (asked first) forgets every
edit: the text as found, timed again.

| Key | Lyrics editor |
|---|---|
| `←` `→` / a click on a word | the word before, the next |
| `↑` `↓` | the line above, below |
| `shift+←` `shift+→` / drag the left edge | its start 20 ms earlier, later (set by hand) |
| `alt+←` `alt+→` / drag the right edge | its end 20 ms earlier, later |
| `ctrl+←` `ctrl+→` / drag the middle | the whole word 20 ms earlier, later |
| `space` | play from just before the word, again: pause |
| `n` | not to be sung, or to be sung again |
| `+` `-` `0` | the voice over it in Karaoke mode 3 dB louder, quieter, as the Voice slider |
| `enter` / a double click, `shift+enter` | write the word, its line (enter keeps it, `esc` leaves it) |
| `del` | take the word out |
| `ctrl+z` `ctrl+y` | undo, redo |
| the wheel, the secondary button dragged | zoom round the pointer, move the view |
| `esc` / Done | close (every change is kept as it is made) |
| Key | Karaoke tab |
|---|---|
| `enter` / the button | Karaoke mode |
| `shift+enter` / shift+click on the button | Karaoke mode where the song is: no rewind, no count-in |
| `↑` `↓` | the row the arrows move: the voice, the delay, the key |
| `←` `→` | its value down, up: the voice 1 dB (at −45 dB or below: off; from off: −45 dB; `shift`: 1 % of its travel), the delay 10 ms (its nudge), the key a semitone; a click, drag or the wheel on a slider too |
| `-` `+` (or `=`) `0` | the key a semitone down, up, the song's own (also on the Mixer tab) |
| `s` / a click on sync | sync: measure how late your voice comes (a microphone open) |
| `i` | the next microphone |
| `r` | record 30 s of what the echo canceller works with (also in the mode) |
| `g` / a click on Game | the game on or off: the singing scored, or just sung |
| `p` / a click on Notes | the song's notes shown or hidden (hidden, the game's bubbles and their sounds go too; its diploma stays) |
| a click on Keep | your singing of each song kept, or not |

The **key** (the tab's Key row, −6 to +6 semitones, `0` the song's own)
plays the song higher or lower, its tempo kept, to sing it where it suits
your voice: every stem but the drums is shifted (pitch-shifting drums
smears their hits), the whole mix where there are no stems (no GPU and
none kept). A change is heard within about a tenth of a second, without a
click; back at 0 the file plays bit for bit again. The song's notes and
the scoring move with the key heard, so singing in the new key scores as
singing in the song's own did. The key stays for the songs the queue runs
into, and for the song it is on started again (START in the mode); another
song picked plays in its own key; a key set while nothing plays (a song
held for START) is the next song's. Another output or the engine started
again keeps it. No transition is armed while the key is shifted, and the
key stays while one is; new settings wait for the song's own key or the
next song, and bar jumps are refused meanwhile (each says so).

The game is set on the tab only, not in the mode: a song is sung at one
level from its start to its end, and only a song whose words were nearly
all heard wins a medal or a best (one sought into, or sung from late on,
does not).

Your voice reaches the game a little after the music it is sung to (the
output's buffers, your ears, the microphone's): the **delay**, one value
for the scoring and for what Keep records. It is the first there is of:
the delay **measured** by a sync for the output and microphone in use;
the echo canceller's, once **steady** (within 10 ms of itself for 3 s;
with a headset there is no echo, so never); the delay **estimated from
your singing** (the last 20 s of it held against the song's notes every
2 s, at each delay from 0 to 400 ms, averaged over about half a minute:
where the most of each word scores, held to the notes a little more
strictly than the game, once clearly better than 150 ms); or 150 ms. Your singing
corrects a delay measured or the echo's too: where its estimate (all your
singing averaged, the delay nearest the one measured that fits within
0.5 % of the best, once the best fits 1 % better) is within 60 ms of it,
that is taken (a sync's "ta" comes a little sooner than a sung word does);
silence and noise move nothing, and a sync again starts the correction
over. Your nudge wins: nudged (even to 0), the delay is the one measured
and your nudge, whatever the singing says; a sync drops the nudge, and so
does `0` on the Delay row (it is "not nudged" again). The tab's **Delay** row shows the value in force
(0 to 600 ms) and its sync row where it comes from (`measured · your
singing +30 ms`). Mid-song it
moves at most 20 ms a second towards a new value, so the scoring never
jumps; `←` `→` on the Delay row (or the pointer) nudge it, at once
(−200 to +400 ms, kept per output and microphone). Another output or
microphone starts the echo's wait and the estimate over.

**Sync** (`s`, with a microphone open) stops the song and plays 8 clicks
at 90 BPM after a second of silence: say "ta" (or clap) with each click.
Each sound's start is timed in the very frames the voice is scored in, so
this measures the whole way, output, ears and microphone. The dots light
up as the clicks play, gold once answered; at the end, your delay (the
median of the answers within 150 ms of the median, at least 5 of 8, else
"again?"): keep (`enter`) keeps it for the output and microphone it was
measured with (in `delays.txt` beside the settings, the nudge dropped),
again (`r`), cancel (`esc`), each a button too. The clicks play on the
player's strip: with it muted, far down or another soloed on the Mixer tab,
there is no sync, and the tab says why.

With **Keep** on, each song sung in the mode is saved when it ends (or
the mode is left), unless there is no voice in it (under a second with a
pitch: the microphone muted or silent, nothing sung): a folder `recordings/sing-<seconds>` beside the
settings file with the voice as the game heard it, the echo cancelled and
its leftover turned down (`voice.wav`), where in the song each part was
sung (`times.txt`), the
lyrics with our word times and the song's melody as the game held the
singing against them, and the delay the game took (and where it came
from), and the key the song was heard in where it was not its own
(`about.txt`'s `key` lines). The `sing_replay` example hears it again as
the game does (in the song's own key) and
tells how the pitch was detected (how much of the notes had a pitch, how
steadily, the jumps), when the singing came in against the words, and
what each level scored, word by word with `--words`, and how many words
would burst at each; the delay in force as the game follows it from the
one it took then (every 10 s, and where the estimate settles); with
`--auto`, the delay the game estimates from the singing as the song goes
at each level, and the scoring at it: for tuning the game on real
singing.

The song's notes show (`p` hides them): from the voice's melody, each
lead word on its own (backing words in parentheses none; a word without
a pitch of its own between its neighbours').

**In the window** (where it draws with the GPU) they run as a lane
across the upper half of the screen towards a playhead a little left of
the middle: each word a pill as high as its longest note, as long as it
is sung (its text made smaller to fit before the next word), 4 s of the
song ahead of the playhead. Your voice is an arrow at the playhead,
gliding to your pitch (a spring that settles in about 40 ms, at any frame
rate); behind it the words you hit fill gold (a paler gold where close),
your line runs where you did not, and sparks fly off the arrow and drift
by with the lane, more on the note, a burst on a word hit and a bigger one
as the multiplier goes up. The line sung (filling word by word) and the
next are at the bottom, and along the bottom edge the song's progress,
its sung stretches apart from the breaks and instrumental parts (4 s or
more without singing). All of it is drawn by the GPU each frame (shapes
as signed distances in one shader, the text from an atlas), over the
blurred cover; nothing of it is painted by the processor frame by frame.
Without a GPU there are no notes: the lines show as text in the middle,
the one being sung filling in the accent.

As you sing, the microphone's pitch (YIN, 100 frames a second, at the
time in the song it was sung: the output frame playing as it was caught,
less the delay) is held against the notes, and with the game on (`g`)
the words you hit count. There is one way of scoring, generous: it
follows the key you sing in (how far you sing from the song, octaves
aside, over the last 3 s) and wants the melody's shape, within 3
semitones. A word counts when a tenth of its note time is hit (a pitch
is heard on only about half of a word sung well, its consonants and
breaths having none): 100 points, times the multiplier, shown top right.
The points roll up there as on an odometer: each figure that changes
turns to its new one, the one going out sliding up as the next slides
in, in about a third of a second (a big jump a little longer, never
more than 0.6 s behind); on a jackpot the low columns spin whole laps
in a blur, and the number glows and swells while it counts. A lower
score (a seek back, another song) shows at once.
Words hit in a row are a streak, climbing the multiplier's rungs, each
named after a tier: 8 make it ×2 (Good), 20 ×4 (Great), 40 ×8 (Awesome),
75 ×16 (Amazing), 120 ×32 (Diva). It shows over the playhead as
"×16", popping as it goes up a rung, and left of it a dark pill with the
rung's name in its tier's colour over the streak ("Amazing" over "82 in
a row"; the meter is right of the playhead); the points
top right have the rung under them ("Amazing ×16"). A word missed ends
it. Your
line snaps onto the note where it hits; off the notes it is drawn as you
sing it, smoothed (a median over 50 ms, then a glide half way each
frame).

Each **phrase** (a sung line) also **shines**: as it ends it is measured
against the original singer's same phrase (the song's vocals, heard as
your voice is) and scored as a share of it: how much of each word you
sang while it is sung (half as much as the original is heard on it is
all of it), times how near the notes your
pitch was, your timing, how steady your held notes, how long you held
the long ones and whether you reached its top and bottom, each against
the original's. The **shine meter** at the top fills as you sing a
phrase and springs to its shine as it ends, its tier popping over it
on a badge (drawn over the light, so the bloom never washes it out):
**Good** (60 % of the original), **Great** (72 %), **Awesome** (80 %),
**Amazing** (87 %), **Diva** (94 %, the pitch at 92 % of the original's
too: the original from its second phrase on, a good take on a few
phrases a song). Diva
has to be earned in a row: a phrase at Diva's bars counts as Diva only
right after a phrase that was Amazing or better, else as Amazing, so the
first phrase you sing is never Diva (nor after starting over). Each
phrase's bars are set by what the original itself reaches on it, heard every
everyday way (another rate, the delay off, octave errors, pitch lost),
so the original singer reaches Diva's bars on every phrase (and is Diva
from its second on), and a rough original
makes Diva easier; a phrase the original reaches under 70 % of itself on
(so rough that half of it sung could be Diva) has no shine. A phrase the
original is heard too little on (fewer than two words, or under 15 % of
its notes) only fills its bubbles. The
original's phrases are worked out off the audio thread once the words
are timed (a few seconds, from the song's kept stems) and kept in
`shine` in the data folder; until then the song has no shine. The tier
lights the stage: Good is a rainbow streak (bands fanning out from the
playhead behind the pills, a swirl round your arrow, the multiplier in
rainbow; each tier up throws stars, notes and rainbow bits from the
playhead; a phrase with no tier collapses it into the playhead), Great
adds a glow round the lane, and from there it grows like a jackpot, each
tier keeping what the one below has:

- **Awesome** lights half the lane in the rainbow and sets the screen's
  edges smouldering (low flames along the bottom, shorter up the sides,
  flaring on each bar's first beat and breathing with the bass). Its
  rung (×8) pays out: a fountain of gold coins shot up from the playhead
  as it is reached, and a coin flicked up out of each word hit while it
  holds.
- **Amazing** lights all of the lane, flames blazing off its edges, the
  edges' fire higher and the line being sung warmed by low flames at its
  letters' feet; at its rung (×16) marquee bulbs light up round the
  lane, chasing a bulb a beat.
- **Diva** has angels descending, the edges ablaze and the sung line
  burning hotter. A phrase reaching it fires a jackpot of coins and
  stars from the meter and slams "DIVA!" down onto the top of the lane
  (on a bar's first beat, with a squash, a shake of the screen and a
  ring of dust). Reaching its rung (×32) slams "×32" down, rains coins
  over the screen in bursts on the beats with the fountain, and every
  other bulb flashes, the two halves in turn. Each slammed word is sung
  too, by an announcer on the game's synth, in the song's key and landing
  with it: "di-VA!", "times thir-ty-TWO!" (softer while a line is sung).

The lyrics and the score stay readable through all of it: the edges'
fire shrinks back from the lyric band and the score's corner and shows
only faintly inside them, the bulbs' glow ends well before the
line, the slammed word lands between the top of the screen and the lane
(never over the notes or the lyrics), the coins cross the line in a few
hundredths of a second, and the line on fire keeps its letters' shapes
and brightness. Each of these is an effect the tier's row lists in the
tier table (`tiers.rs`, a row a tier: its bars, its rung, its colours,
glow, sounds and sticker; the effects are `gui::fx`): held while the
singing is at the tier (eased between the rows as the glow eases),
fired by a phrase reaching it, fired by reaching its rung, or held while
at its rung; a row can have an effect move with the music (its beat and
bars, the stems' levels, the pitches). The line sung frosted is made too
and no row lists it: the jackpot only heats up.
Where the song's beat grid is unsure they run freely, off the beat.

The game comes in a **theme** (`karaoke_theme` in the settings file:
`"classic"`, the default, or `"spooky"`; `theme.rs`). A theme overrides
by key what the tiers and their effects look like and what the sounds
are made of, and may hold effects throughout and grade the backdrop:
the tiers' colours (their badge, verdict, marks and stickers), the
count-in's colour, an effect's look, the synth's patch for one of the
sounds' atoms. Classic is the game as made. **Spooky** is its
Halloween: the tiers in slime green, pumpkin, witch's purple, a ghostly
mint and moonlight; the fire green (the screen's edges, the lane's
flames, the line sung); the marquee's bulbs jack-o'-lanterns grinning,
their candles lit as the chase runs; the coins candy, each shower a
sweet's colour; ghosts drifting down where the angels would; the
backdrop graded cold, and fog creeping in at the edges, where now and
then a pair of eyes opens and blinks. The pops bubble like a cauldron
over a skeleton's xylophone, the clinks are sweets in a jar, the chimes
toll as bells, the choir wails a ghost's "ooh", and a streak lost slides
down like a theremin.
The light blooms (the GPU draws it
apart, at high range, blurs it and tone maps it over the frame). Along the bottom the song's progress shows
each phrase sung in its tier's colour, a mark a phrase popping as its
tier is decided (a little jackpot on Diva), and the time left in a ring
that drains as the song goes, in the rainbow while you shine. In the
picture (no stage) the bar under the word you sing gets a rainbow edge,
and a smaller burst a step up.

The notes are **bubbles** that fill with a glowing liquid as you hit them
(a gap of a consonant under 60 ms bridged), poured in from the note's
start to a level: how much of it you hit. The liquid's front ripples and
sloshes, and a light shimmers through it. The moment a word counts all
its bubbles fill, the part already sung and the part to come: a quick
sweep to full (0.2 s). In the last stretch the bubble strains, bulging a
little and wobbling; full, the whole of it bursts (0.5 s) once its word
is over, its end past the playhead: it overfills,
bulging to about 1.3 times in 80 ms, then tears, its liquid splashing
out in drops that arc out and fall, torn bits of its skin and a few
sparkles flying with them, and it is gone: a faint wet sheen fades where
it was, then the spot is empty (the word keeps its sung colour). A word
you hold on after it counts (a long note) keeps its bubble full to its
end: it pumps, slowly growing a little thicker and breathing, glowing
more, then bursts the bigger the longer you held it (up to twice the
drops, flying further, and a louder pop). A
bubble only partly filled never bursts: it keeps its liquid at the level
you sang. Each bubble bursts by your singing's own time
(the song's less the delay, so you see the whole sweep and burst; a paused song holds them); sought back before what
filled it, it shows again as it is. In the window the GPU draws the
liquid and the bursts (a record a bubble, its surface, wobble, drops and
bits worked out in its shaders), so they cost the processor nothing.
When a song ends its result is revealed, beat by beat over a few seconds,
each beat with a sound: the title drops in, the points count up (faster
and faster), the words hit fill their bar, the longest streak pops up,
each phrase pops in as a gem in its tier's colour (Diva's sparkling),
then a medal by the words (bronze for half of them, silver 70 %, gold
85 %) is stamped down, or a star cheers you on; a new best rains
confetti, else the best before shows. Every song played to its end gets
its result, the last of the queue too, and one where nothing was scored
asks, with a wink, whether the microphone fell asleep (no medal, no
best). START during the reveal shows it all at once; the next START
sings the next song (after the last, the same one again: "Again"). The
screen takes its light from the cover's colours where the cover was
taken apart; it twinkles a few seconds once settled, then holds still
(drawing nothing) until START. Without a GPU the result shows as plain lines. The best
score of each song is kept (`scores.txt` in the data folder).

A song is sung only once its lyrics are ready (found, and their words
timed where they can be): until then, and for a song with none, the
button into the mode (and START's pill in it) is greyed out, with why
beside it (still looking for its lyrics, its words still being timed,
none found, an instrumental), and does nothing but say so. A big folder
just queued is got ready a track at a time, so its songs come singable
one by one.

Karaoke mode fills the window (fullscreen): the song's cover blurred
under a dark layer, its title and, once they are known, its lyrics. With
the game or the notes on, the whole cover sits sharp in the middle over
that blur, lit by the music (on the GPU, once it is taken apart: its
subject cut out, the person or the thing that stands out, never its
lettering, which two small built-in detectors of text and faces keep out;
a close-up face the cut breaks up is grown from the depth instead; a
cover whose cut is not sure of itself stays whole); with
neither (plain lyrics karaoke) the blur alone fills the window, and
switching either mid-song fades the cover out or in. The
voice plays at its level (as on the tab). Going in stops the song and
holds it at its start (with shift, it goes on where it is). Each song
waits for START (`enter`, `space`, or a click on START's gold pill low
in the middle, there whenever a song waits, from the moment the mode
opens, and under a song's result with the song it plays named before it;
the only START there is to click; without a GPU a small "enter: START"
says it): "3", "2", "1" are counted a second each, numbers over the
lines that shrink and fade, and "1" ends as the first word is sung.
While a song waits, and for 2 s after START, its title and artist show
alone, then fade out over a second; the first line then shows alone
until it is sung, the lines after it coming up with it. A song with an
intro plays at once, the title over its start and the count over the
rest; where the singing starts sooner than the title and the count need
(6 s), the music waits that much in silence. Until its words are timed,
a song is counted in before its start, its title staying. The count
follows what is heard, so it lands whatever the output's latency. It
comes again whatever started the song (shift into the mode, `N`, a
seek): before the first sung line as large, and after a break of 8 s or
more without singing, half as large in the same place at the top, apart
from the lines and never moving with them. A lyric line that only marks
a break (a note, an empty timed line, `[Instrumental]`, `(Guitar solo)`,
`*Interlude*` and the like) is kept as a break, several in a row as one,
and written as `♪` in our timing file; it is not sung, and the mode
shows one `♪` through each long pause. Only whole: heard from less than
three seconds before the words (the mode entered there, a seek), there
is none. A song's end holds the next one for START; `N` skips to the
next one, held the same way; `space` pauses, and a paused song goes on
after the same count-in: nothing plays in the mode without one. The
words light up a little before they are sung, `karaoke_lead` in the
settings file (200 ms by default), so they can be read first; lines move
up into their places as the song goes on. The controls (a translucent
bar rising out of the picture in a soft gradient: previous, pause and
next, the time heard, the seek bar and the length as on the other
screens, the voice slider, the way out, and the keys under them) come up
over the bottom of the picture, never moving the lines or START: only while the
pointer is near the bottom (the bottom 15 % of the screen, never less
than the bar and two lines) and while a slider or the seek bar is held
(no key brings them up), fading in and out over 0.2 s, and going 1.5 s
after the pointer leaves the bottom (or the window). The song's progress
along the bottom gives way to them: it sinks and fades in the first half
of their fade, and they come in the second (their seek bar and times
stand for it), so the two never show at once. The pointer hides as in a
video player: after 2.5 s without a move, the controls up or not; any
move shows it at once, and leaving the mode does too. It hides only over
the window, on Windows, macOS and Linux alike (iced's hidden pointer).
The seek bar does nothing
while a song is held for START or counting in. `esc` is the only way
out: back to the tab, fullscreen ends, the voice still at its level,
playback where it is (a count-in under way is called off).

| Key | Karaoke mode |
|---|---|
| `enter` / `space` / a click on START | START (`space` pauses a song playing) |
| `N` `P` | the next or previous song, held for START |
| `←` `→` | the voice 1 dB down, up (`shift`: 1 % of its travel) |
| `i` / `r` | the next microphone / record 30 s for the echo canceller |
| hold `p` | the demo sings: a perfect singer in place of the microphone, to show the game off ("AUTO" in the corner); that song's score is no best and its singing is not kept |
| `esc` | back to the tab |

The browser lists folders and files ending in `.flac`, `.wav`, `.aif`,
`.aiff`, `.aifc`, `.caf`, `.m4a`, `.mp4`, `.mp3`, `.mp2`, `.mp1`, `.ogg` and
`.oga`. Each file's row shows a format chip, read from the file's headers
without decoding audio: `FLAC 44.1`,
`ALAC 96`, `WAV 48`, `AIFF 96`, `CAF 192`, and for lossy files the codec
and bit rate, `MP3 320k` (a VBR file's average), `AAC 256k`, `VORBIS 160k`. An M4A file
holding HE-AAC is listed with `?` and the reason (only AAC-LC plays).

### The Mixer tab

The Mixer tab (`tab` to it, the fifth of the six) shows the mixer as a desk,
the whole width: a channel strip for the player, a narrower one for each
of its stems beside it (DRUMS, BASS, VOCALS, OTHER), one for the preview,
one for the karaoke game's sound effects (EFFECTS, where the window is
wide enough to take it without narrowing the others), the microphone's
strip, and the master at the right. The tall sections
line up as on a desk. Where the window is narrow the strips narrow (the
stems' to 56 points, VOCALS as VOX; the preview's without a scale of its
own), then the microphone's strip goes first, then the preview's, then
the stems'; in a window 640 points wide all but the microphone's show.
The arrow keys go through the strips drawn.

- **EFFECTS**: the karaoke game's sounds (a bubble's pop climbing a
  ladder in the song's key, the streak up a rung or lost, the rescue
  shot, a phrase's tier and the top's jackpot), played live by the
  player's own synth in the song's tempo and key, each started on the
  frame of the song its moment is seen on (a rung's, a phrase's on the
  next beat, the jackpot on a downbeat clear of the singing); and the
  result's (its paper, pen, stamps and tunes, its count-up, gems and
  points, by the same synth). With the notes hidden (no bubbles shown) the
  game's own are not played, the result's are. They follow the song's
  loudness: a loud song takes them a little higher, a quiet one a little
  lower, so they sit the same over each (below). Its fader is the user's
  own on top, 0 dB as calibrated; it and the mute are kept in the
  settings (`karaoke_effects`, `karaoke_effects_mute`); down to −∞ or
  muted, none is played. While one sounds, the output is mixed; the
  microphone's echo canceller hears them as it hears the song.
  - *The calibration*: the song's loudness is what the karaoke plays of
    it (its stems with the voice at the Karaoke tab's default, the
    instrumental; the file as it is where no stems are kept), integrated
    over the whole song as ITU-R BS.1770 has it (K-weighted, gated at
    −70 LUFS and 10 LU under the rest). It is measured once, on a thread
    of its own, as a song is first sung with the game on, and kept in
    `loudness` in the data folder (a small file a song; measured again
    when its stems come, or the voice's default changes). The synth's
    bus is set for −13.8 LUFS, and moves 0.27 dB for every LU the song is
    louder or quieter, 4 dB either way at most, before the synth's
    limiter (so nothing clips), gliding to a new one over 2 s while a
    sound plays; a song not measured yet is taken as at −13.8 LUFS.
    Fitted on six songs set by ear, from −22.2 to −9.4 LUFS as played:
    each lands within 1 dB of its setting, five within 0.6.
- **PLAYER** and **PREVIEW**: the name on a scribble strip (lit when
  selected), what the strip carries (the format; the player's LEAD row,
  lit while the track on air is heard, and while a transition mixes, or
  the track it left plays out what it holds, a NEXT row for the other
  one, each with the gain the transition's fade puts on it now; the key
  the song on air plays in, beside LEAD, while it is not its own; how full
  the ring is, and `↓` with the frames the next track dropped where it
  fell behind; the preview's two
  players A and B with the gain of each in dB, so a crossfade can be
  watched, `—` while there are none), a fader beside an L/R meter, the
  level, and `[M]` mute and `[S]` solo (while a strip is soloed, the
  others are silent). The fader is tapered to what is heard: +6 dB at the
  top, 0 dB at four fifths of the way up, then −6, −12, −20 and −30 dB,
  each 0.15 of the travel below the one before, −45 dB near the bottom and
  −∞ at the bottom, so most of the travel is where a change is heard
  (below −30 dB a stem is masked by the rest and a whole mix is faint, so
  −30 dB to −∞ gets only the last fifth). The meter beside it is on
  the same ruler, so the fader's marks read it in dBFS; above 0 dBFS (only
  with the fader above 0 dB) it is red: the overs. A level other than 0 dB
  shows in orange: the strip adds gain, or takes it away. Every change is
  a 20 ms ramp in the mixer, so nothing clicks.
- **DRUMS**, **BASS**, **VOCALS**, **OTHER**: the song's stems, strips
  inside the player's (its fader and mute act on them as on the file; a
  stem soloed silences the other stems, the preview and a sync's clicks,
  and the file once the stems are heard in its place: while the file
  stands for them, with none there yet or in a transition, it plays on;
  soloing PLAYER keeps them all). Each shows `muted`, or the Karaoke
  voice's gain on its lane, where that adds to its fader beside its meter
  on the player's ruler and its `[M]` `[S]` (`M S` where it is narrow). A bracket over them says how the stems
  stand: `● STEMS heard` (the desk plays them in place of the file: a
  stem's strip is touched), `○ STEMS ready` (made, the file heard),
  `separating…`, `loading…` (kept, being read) or `no stems`. With every
  stem's strip as it starts the file is heard as it is; the stem keys
  (above) are these strips' mutes.
- **MIC**: listened to, never played: no fader (`LISTEN ONLY`), but its
  meters as captured (IN) and with the singer isolated (VOICE), its port
  (hearing what the device plays, waiting for an output, or why it cannot
  hear), the echo canceller as an insert (`[ECHO]`, lit while on; a click
  or `e` turns it on or off: `karaoke_echo`, saved), how much echo it removes,
  how late the echo comes and whether it has learnt its way, and the
  clock bridge's drift (ppm), fill and dropouts. The lamps PITCH and REC
  are the game and the keeping of the singing (`karaoke_game`,
  `karaoke_keep`): yellow while set, green while they hear (in Karaoke
  mode). The microphone is open while this strip is selected (`←` `→`
  to it, or a click), and closed again when another is; until then the
  strip says it is closed. Without one it says so.
- **MAIN**: the device, its backend (shared, as every one is) and format,
  the LIMIT lamp (the limiter turned the gain down, held lit for a
  second), the samples clipped and the desk's clock, a big L/R meter, and
  VOLUME: the player's own volume (the master's gain over every strip, 0
  dB at most; `↑` `↓` and the wheel move it, as `alt+↑` `alt+↓` do
  anywhere; this run only). The master never adds gain above unity.

The meters are peak meters, the microphone's and the master's over −60
to 0 dBFS: green below −18, yellow to −6, red above; they rise at once and fall 20 dB a second, a tick
holds each peak for 1.5 s, and a lamp over each lights for 1.5 s when a
peak reaches full scale. They move by themselves while the tab shows, so
the window keeps drawing.

The file as it is stays the default: the desk adds no gain until a fader is
moved, and nothing of it is kept: every start is at 0 dB, unmuted, nothing
soloed. A new mixer (another device, an engine restarted) gets the faders
as they were set this run.

The tab is drawn in the window's own widgets (iced, [The
window](#the-window)): brushed panels, backlit scribble strips, buttons
that glow when on, LED-segment meters with their peak ticks, the caps
gliding to a level set by a key. With the mouse, a fader's cap follows
the pointer up and down while held, to the pixel (the pointer may leave
the track; a cap taken off its middle does not jump); a press elsewhere
on the track brings the cap there; within 5 pixels of 0 dB it is at
exactly 0 dB, as in a desk's detent (a key step across 0 dB stops there
too); a double click is 0 dB;
the wheel over a strip moves its fader by 1 dB (over the master, the
player's volume); a click selects a strip, and `[M]`, `[S]` and `[ECHO]`
are buttons.

| Key | Mixer tab |
|---|---|
| `←` `→` | the strip to the left, to the right (player, its stems, preview, microphone, master) |
| `↑` `↓` | the fader 1 dB up, down (at −45 dB or below: off; from off: −45 dB; on the master, the player's volume, never above 0 dB) |
| `pgup` `pgdn` | the fader 6 dB up, down |
| `home` | the fader to 0 dB: nothing added |
| `end` | the fader off (−∞) |
| `m` | mute the strip, or not |
| `s` | solo the strip, or not |
| `e` | the microphone's echo canceller on or off |
| `i` | the next microphone |
| `-` `+` (or `=`) `0` | the key a semitone down, up, the song's own (shown on the PLAYER strip) |

The capitals keep their commands here (`M` the beat pulse, `S` stop);
`esc` does nothing on the tab.

### The Synth tab

The Synth tab (`tab` to it, the last of the six) is a workbench for the
karaoke game's sounds. On the left every sound the game makes, by moment:
a word's pop, a run of words up to the first rung, each rung reached (its
stinger, its call, its payout), the rescue shot, a streak lost, each
tier's chime, the payout and the jackpot alone, the announcer's calls,
the diploma's count-up, gems, points and best streaks, and its sheet's
own (the paper, the pen, the stamps, the medals). A click plays one at
once on the Effects strip, as the game plays it, in the key and at the
tempo the header shows: chosen there (a step either way, or a click on
the value for the song's again), else the song's, else no key and 120.
The theme compared (Classic or Spooky) is the header's too.

In the middle, the sound's patches (the atoms it is made of, one shown,
an edited one marked) and the one shown: its engine (a segmented
choice of the synth's ten), its envelope (its curve beside its four
sliders) and its parameters in their groups, each a slider over its
range, a choice or a switch, a value off the built-in one in the accent.
The next play is heard with the edit (the game's sounds too, in that
theme). Keep writes the patch to `patches` in the data folder, loaded
over the built-in one at every start; As built brings the built-in one
back and forgets what was kept. The layers a tier switches on for the
sound are shown over it.

At the foot a keyboard of two octaves from C4 plays the patch shown: a
press holds the key under the pointer (sliding from key to key), and the
computer's keys play it too, each note held as long as its key, the
letters written on the keys they play.

| Key | Synth tab |
|---|---|
| `↑` `↓` | the sound above, below |
| `pgup` `pgdn` | the moment before, after |
| `home` `end` | the first, the last sound |
| `enter` | play the sound |
| `←` `→` | the sound's patch before, after |
| `a` `s` `d` `f` `g` `h` `j` `k` `l` `;` `'` | the keyboard's white keys (held as long as the key) |
| `w` `e` `t` `y` `u` `o` `p` | its black keys |
| `z` `x` | the computer's keys an octave down, up |
| `c` | the theme compared: Classic or Spooky |
| `[` `]` `\` | the sounds' key a step down, up, the song's |
| `-` `+` (or `=`) `0` | the sounds' tempo 5 down, up, the song's |
| `K` | keep the patch shown |
| `del` (or `backspace`) | the patch shown as built |

## Code

| Module | |
|---|---|
| `core` | state, `Action`, `update(state, action) -> effects`, keymap; no geometry: what the window's size does to the keys and the pointer comes in as `core::Fit`; the Transition tab (`core::transition::tab`, the lanes it shares with Karaoke in `core::lanes`), the grid fixes (`core::grid`), the transitions' presets and plans (`core::transition`), the Karaoke tab and mode (`core::karaoke`, its sliders in `core::sliders`), the key the song plays in (`core::key`), the Mixer tab's desk (`core::desk`: its strips, settings, the fader's taper, the meters' ballistics, the actions its widgets send) and the stems on it (`core::stems`: the stem keys as strip mutes on a bar of the head on air, when the engine makes stems, the Karaoke voice's lanes on both heads). Pure. |
| `view` | the presentation models the window draws (the list, the info, the bottom bar with the spectrum's plot, the Transition tab with `view::lanes`, which the Karaoke tab shares, the Karaoke tab and Karaoke mode's picture, the settings, the help, Now playing) |
| `app` | the running player: the engine and its desk, background workers, the spectrum (the mixer's monitor), cover; carries out the effects |
| `desk_wiring` | the desk the engine plays on: a mixer that holds the device (`jodel_output::mixer`), the player's strip with its stems' strips inside it, the preview's, the engine on two lane inputs of them (its two heads: a transition's next track plays on the other); the Mixer tab's settings made on it, its look for a frame |
| `preview` | the preview's players, guests on the mixer (one per file, two at most): only the latest file loaded, crossfades on the mixer's clock, the song ducked and back |
| `browser` | folder listing, file details read lazily for the rows on screen (cached for the session), recursive search, folder walk |
| `config`, `cli` | the settings file and the command line |
| `spectrum`, `art` | FFT of the tapped output, cover decoding and fitting |
| `rhythm` | the beat analysis in the background (decode, mono, resample to 22.05 kHz, [`jodel-rhythm`](../rhythm)), the waveforms, and their cache |
| `grids` | the beat grids fixed by hand, kept per track in the settings folder |
| `patches` | the game's sounds' patches kept on the Synth tab (`patches/` in the data folder), a file a theme's atom, each parameter by its name (`core::synth_tab` the tab, `sfx::bench` its sounds by moment, `gui::screens::synth` its widgets) |
| `cache` | the schemas of what is kept on disk between runs, and their versions |
| `stems_cache` | the tracks' stems kept (`stems/`): their files' names, the store the engine reads them from, pruning to 30 GB |
| `loudness` | each song's loudness as the karaoke plays it (ITU-R BS.1770, `jodel_rhythm::loudness`), measured once on a thread of its own and kept (`loudness/`), for the game's sounds to follow (`sfx::conductor::calibration`) |
| `gui` | the window, an iced application (`gui::window`); every screen but Karaoke mode in iced's widgets (`gui::screens`, the main screen laid out from the measures in `gui::metrics`; the Mixer tab `gui::mixer`; the covers `gui::widgets::cover`, the lanes `gui::lanes`); Karaoke mode in the window's own widgets too (`gui::screens::karaoke::mode`), its stage a widget of its own on the GPU (`gui::stage_view`, `gui::present`, `gui::stage`, `gui::stage_gpu`, a pipeline a group of shapes; what keeps it fast, held by tests, in PERFORMANCE.md) and its effects (`gui::fx`: an effect a module, registered by one line, its strength eased from the singing and moved by the music on a board, drawing at its points of the stage, its shaders held to its maths) |
| `tiers` | the phrase tiers, one table: a row a tier, all it takes and shows (its bars, colours, glow, sounds, sticker and the effects it lists) |

The window's font is Sarasa Fixed J (Iosevka's Latin with Source Han Sans
for Japanese, Chinese and Korean, each of those exactly two cells wide),
with Noto Emoji for emoji (monochrome, in the text colour), and Atkinson
Hyperlegible Next SemiBold for lyrics where they are drawn large (the
Karaoke tab's voice lane, Karaoke mode: a font made to be read at a
glance, proportional, set by its own spacing); all SIL Open Font License
1.1. They are built into `jodeldiplom` (about 16 MB), so
it is one file. They are in
`fonts`, with their licences and where they come from
([fonts/README.md](fonts/README.md)); `build.rs` takes them from there (or
from the folder `JODELDIPLOM_FONTS` names). Built
without them, the window uses the embedded Cascadia Mono
(`assets/CascadiaMono-OFL.txt`); `jodeldiplom --help` says which.
Characters none of these fonts has come from the system's fonts (Yu
Gothic, Microsoft YaHei, Malgun Gothic … on Windows, Hiragino and the
like on macOS, Noto CJK or whatever fontconfig picks on Linux). The size
follows the window's scale factor.

### The window

The window is an iced application (`gui::window`, iced's development
version pinned to a commit on the same wgpu as the rest of the player, so
there is one GPU stack): `core::State` is its state, keys go through
`core::keymap`, and the frames come at every refresh of the display while
something moves smoothly, 30 a second while only the spectrum and the
clock move, four a second at rest; the player's threads wake it. Without
a GPU (or with `JODELDIPLOM_SOFTWARE=1` set) iced draws on the processor.
Karaoke mode makes it fullscreen.

Every screen but Karaoke mode is drawn in iced's own widgets
(`gui::screens`, what shows chosen there; `gui::pacer` around it all ticks
at the display's refresh from inside each redraw). The main screen is laid
out by iced in rows and columns from a few named measures in points
(`gui::metrics`): the tabs' header, the side pane (the browser or the
queue, the Transition, Karaoke or Mixer tab) with the right column beside
it (or the track's summary under it in a narrow window), the Karaoke tab's
voice lane, and the bottom bar; what the window's size does to the keys
(the list's page, the help's, the desk's strips, the lanes' letters) is
reported to the core from the same measures (`core::Fit`). The covers are
iced's own images in the places the screens make for them
(`gui::widgets::cover`: the art is scaled to the pixels it shows at once
for each art and size); the lanes are a widget in their part of the screen
(`gui::lanes`).

Karaoke mode is the window's own too (`gui::screens::karaoke::mode`): a
widget of its own fills the window (`gui::stage_view`, iced's shader
widget: the backdrop as the ground and the stage over it, `gui::stage`:
the lane, the lyrics, the count-in, the score, START, the result; START's
pill takes the press), the controls' bar in iced's widgets over it; on the
processor the lines as text over the cover's blur.

The lanes draw on the processor only what they must: their columns sit on
a grid of whole pixels from the track's start (the view's start snapped to
it, within a pixel), so a view that scrolled is shifted by whole columns,
and only the columns new to it or whose marks changed (the playhead, the
cursor, a selection, a fade's curve) are painted, each in one pass; the
bar numbers are painted again only when they moved. A lane that follows
the playhead scrolls on the GPU, so most frames of a playing Karaoke tab
send nothing but where it stands. `cargo run --release -p jodeldiplom
--example frames -- [WIDTH HEIGHT SCALE] [FRAMES]` times what the window's
frames do beside iced's view headless, per screen: what the window's size
does to the keys and the lanes (default 3024 × 1890 at scale 2, a 16-inch
MacBook Pro's window maximised). The tests check that lanes framed one
after another give the pixels each would have painted afresh, on the
Transition tab through scrolling, seeking, a held view, a selection, a
transition's curves and a zoom, and, on a GPU, that the lanes and Karaoke
mode's frames drawn through the window's path off screen and read back are
those painted on the processor.

The icon (a dotted spectrum rolling off from one colour to another) is drawn by `src/icon.rs`: the window sets it at run time (on
macOS the Dock icon), on Windows `build.rs` builds `assets/jodeldiplom.ico`
into `jodeldiplom.exe` with `windres` (the target's
`*-w64-mingw32-windres` when cross compiling, or `WINDRES`), and the macOS
app carries `assets/jodeldiplom.icns` (the square on Apple's icon grid,
with a shadow). After changing the drawing, rewrite both files with
`cargo run -p jodeldiplom --example icon [PREVIEW_DIR]`; tests check they
match.

`macos/make-app.sh` builds the app (`Info.plist`, the binary, the icon and
the fonts' licences) and signs it ad hoc; `ffmpeg/build.sh` calls it for the macOS
bundle. The app is not notarised: after the first attempt to open it,
allow it under System Settings, Privacy & Security (Open Anyway).

### What is kept on disk

Each kind of file the player keeps between runs has a schema with a
version, in `src/cache.rs`:

- analyses and waveforms (`rhythm/`);
- the tracks' stems (`stems/`, beside `rhythm/`), written by the analysis
  so a track is separated once: everything that plays stems (the stems
  made beside the file for the desk, from any frame: a seek, a jump; a
  transition; Karaoke) and the lyrics timing reads them back; only a
  track without them is separated live. About 26–28 MB a minute of music (16 bits with a scale
  per block, coded losslessly; `jodel_playback::stem_file`), kept under
  30 GB, the least recently played going first;
- lyrics as found, "none found" records, timed lyrics and lyrics as
  edited, timed (`lyrics/`);
- lyrics edited by hand (`lyric-edits/`, beside `lyrics/`);
- grids fixed by hand (`grids/`).

A text file starts with a header naming its schema, its version and the
player that wrote it (`jodeldiplom-cache analysis 1 by jodeldiplom 0.1.0`).
The waveforms and the stems carry their version in their own first
bytes.

The contract:

- A cache file is read only in the schema version this player writes.
  Any other is made again; the player's own version plays no part.
- A change to what a schema's files hold, or how, raises that version.
  `tests/schemas.rs` holds every version's sample in `tests/schemas/`,
  byte for byte, and fails on any change to what is written. The fix is
  to raise the version, then run the tests with `UPDATE_SCHEMAS=1`; that
  adds the new version's sample and never touches an old one.
- While a version stays, every player reads the files of it, whoever
  wrote them.
- Files from before there were headers count as version 1 for the
  user's own files (beat grids, scores, delays, lyric edits); a cache
  without a header counts as missing and is made again.
- Grids and lyric edits are the user's work, so a new version of either
  still reads all the older ones.

How a file's contents were made is versioned apart from its schema:

- an analysis by `jodel_rhythm::VERSION`, which is in its file's name, so a
  new analysis is made afresh;
- timed lyrics by their aligner, which the file names; lyrics as edited,
  timed, by their aligner too and by the edits they were made with (a
  hash of their words and times set by hand).

## Finding what holds it up

With `JODELDIPLOM_TRACE=FILE` set, the player writes a timing trace to
`FILE`: the window's start (iced's, the app's), and every update or frame
that took longer than 30 ms, with what it was. Slow work belongs on a
thread of its own; the trace says which. Every 5 s it also writes a line
on the window's frames: how many, how many came late (the frame before
the last was not through the GPU by the next one), the longest gap, how
long drawing and handing the changes to the GPU took, and how
background GPU work gave way (frames late while it ran, its rests, the
share of the GPU it keeps to now).

The window comes first on the GPU: every network runs in submissions of
about 5 ms (`jodel_gpu::pace`), and background work (the analyses, the
lyrics' timing, the queue got ready) rests between them while something
on screen moves, to a share that falls when frames come late, and waits
whenever the stems of the tracks playing are on the GPU. Those have their
own separator and never rest.

`--diag` prints the build, the system, the settings, the output devices,
the microphones (and two seconds of what comes in from karaoke's one), the
speech model, the GPU, and the last analyses.

## Testing

`cargo test -p jodeldiplom` needs no device and no window: the core
(update, keymap, the window's size as the core hears it at several
sizes), the screens' models as text outlines compared with
`tests/snapshots/` (rewrite them with `UPDATE_SNAPSHOTS=1` after an
intended change), the screens in iced's simulator, the browser on a temporary
folder tree, the whole app playing the PCM fixtures (`tests/fixtures/pcm`) into a null
output, and the window's pixels drawn headless.

`cargo run --release -p jodeldiplom --example stems_kept -- FILE [OUT.stems]`
keeps a track's stems as the queue does and reports what it costs: the
time, the file's size, how fast it reads back, and how far it is from
separating the track again live.

`cargo run --release -p jodeldiplom --example shots -- OUT_DIR` renders every
screen to `gui-*.png` as the window draws it (`Gui::view` through iced's
headless renderer: iced's widgets, and in Karaoke mode the stage
the GPU draws), at 1280 × 800 at scale 1 and as
`gui-*-hidpi.png` at 1512 × 982 at scale 2, and the outline of its models
to `gui-*.txt`, with a fixture track playing into the null output.

`cargo run --release -p jodeldiplom --example view_cost -- [FRAMES] [--budget MEDIAN_MS WORST_MS]`
times what building the window's view costs a frame (`Gui::view`, laid out
and drawn headless at 1512 × 982 at scale 2), for the screens drawn by
iced's widgets: the main screen (also on a folder of 2,000 files: only
the rows in view are made), the queue and the Mixer tab.

`cargo run --release -p jodeldiplom --example beats -- [--methods flux,network] [--beats] [--levels] [--presence] [--clicks OUT.wav] FILE...`
analyses files as the player does (with the methods given, default both).
For each file it prints the tempo, the
confidence, the regions where the beat is unclear, the bars found, the
grid chosen and how the beats group it (`×⅓`: a beat every three of its
beats), and every candidate's tempo and agreement. `--beats` lists the
beats with their local tempo (`|` marks the first beat of a bar).
`--no-stems` makes the cues listen to the mix; `--stems DIR` writes each
file's four stems to DIR as WAV, to hear the separation.
`--levels` prints what each cue of the meter said about each grouping.
`--presence` shows every 10 seconds how present the drum and bass stems
are and how much the drum stem's candidate has a say there; `--wave`
prints every 10 seconds each lane's peak level, as the Transition tab draws it.
`--clicks` writes the last file as WAV with a click on every beat,
higher on the first beat of a bar, so the beats can be checked by ear.

`... --example beats -- --truth LIST [--fit | --compare]` compares the
tempi of a list of tracks with ones a listener tapped (a tempo, a tab and
a path per line; `?125` for a believed tempo, `-` to only report) and,
with `--fit`, searches the meter's weights for the most right ones.
`--compare` scores the list without and with the drum stem's candidate
and lists the tracks whose result changes.

`cargo run --release -p jodeldiplom --example transition -- FROM INTO OUT.wav [--preset fade|drums|bass|vocals] [--stretch] [--bars N] [--rate HZ]`
plays a transition from one track into another into a memory output in
real time (no device) and writes it to a 24-bit WAV file, from twelve
seconds before the window to ten after the next track is back at its own
tempo. The window is `--bars` bars (default 8) about half a minute before
FROM's end; INTO comes in at its first bar. It then analyses the file and
says how far its beats are from where they belong: before the window
(FROM alone: how close the analysis itself gets), in the window (FROM's
beats) and after it (INTO's, along the time map).

`cargo run --release -p jodeldiplom --example stems_at -- FILE [--from SECONDS]`
plays a stereo FILE from `--from` (default 60 s) through a desk (the
player's strip with its stems' strips inside it) into a memory output in
real time (at the file's rate), its stems made on the GPU beside it; 4 s
in it mutes the drums' strip on the first bar at least 0.1 s on (the
stem keys' way) and unmutes it on the first bar a second after that, and
says how far from those bars the changes were heard (found in what was
played less the file).

`cargo run --release -p jodeldiplom --example voice_set -- LIST OUT` makes
the training material of the karaoke meter's voice isolator
(`jodel_output::isolate`): for each song in LIST (one path a line), its voice
and its band (the mix less the voice), separated on the GPU, mono at
16 kHz, 16-bit, as `OUT/{id}-voice.wav` and `OUT/{id}-band.wav`, with a line
of `OUT/index.tsv` each; a run stopped goes on where it was. The network is
trained from them outside the repo.
