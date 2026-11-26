# The looks

Every song in Karaoke mode can have a look of its own: a stack of shader
passes over whatever the backdrop draws (the cover today, a song's scenes
later), under the stage and the lyrics. The passes are a library
(`looks/`, the `jodel-looks` crate); a look is data (JSON), written by a
person, a preset, or a language model that picks from the catalogue and
never writes shader code. The catalogue the model is given is generated
from the passes themselves: [`looks-catalogue.md`](looks-catalogue.md)
(`cargo run -p jodel-looks --example catalogue`, and a test holds it to
the code).

## A look

```json
{
  "id": "35mm-night-drive",
  "name": "35mm night drive",
  "tags": ["synth-pop", "night and city", "70s film stock"],
  "layers": [
    {"pass": "split_tone", "knobs": {"shadows": 0.53, "lights": 0.07, "amount": 0.6}},
    {"pass": "halation", "knobs": {"amount": 0.6},
     "drive": [{"knob": "amount", "by": "bass", "amount": 0.5}]},
    {"pass": "grain"},
    {"pass": "kaleidoscope", "mix": 0, "drive": [{"knob": "mix", "by": "instrumental", "amount": 0.8}]}
  ]
}
```

- `layers` draw in order, each over what the one before drew.
- `knobs` by name; a knob left out is at its default.
- `drive` moves a knob (or the layer's `mix`) with the music: `by` a
  driver, `amount` a share of the knob's swing, -1 to 1 (the swing is the
  most the music may move that knob). Left out: the pass's own drivers
  (every pass moves sensibly by itself: grain breathes with the bass,
  bloom follows the voice, glitches hit on the snares, the weather
  thickens in the choruses). `[]`: still.
- `mix` how much of the layer shows over what is under it (0 to 1).
- `keep`: the budget never drops it.

A song's look is a preset as it is, a preset with its knobs moved, or a
stack of its own; all are checked the same way before they draw.

## Where it draws

The backdrop is drawn as before; then each pass of the look draws the
whole window once, a pipeline a pass, each reading what the one before
drew, into textures the window's size (half floats, so the feedback's
echoes fade smoothly); the window's ground reads the last. The stage, the
lyrics and the effects are drawn over the look, never under it, so the
game stays readable whatever the look does. With no look nothing more is
drawn: Classic and Spooky have none, and their frames are byte for byte
as before. The theme in force carries its look as a preset's id
(`Theme::look`); a song's own look (owned data with its own id:
`KaraokeState::look`) is drawn instead where it has one; the Shader tab's
while the tab shows (`State::look`).

## The knobs

- **Few, meaningful knobs.** A pass has two to seven, each a thing a
  person would ask for ("how coarse the grain", "where the light comes
  from"), not a raw coefficient.
- **Even to the eye.** A knob's range is mapped inside the pass so equal
  steps look like equal changes (grain's amount squared, a dot's
  coverage as its area, contrast as a power about a pivot).
- **Subtle to strong, never broken.** At either end of every knob, and
  every corner of them, the picture is still a picture: no black or white
  frame, no number that is none (a test draws every pass at its corners).
- **Units a person can read.** `share` (0 to 1 of the effect), `px`
  (pixels at 1080 lines: a look looks the same on every screen), `turns`
  (of the colour wheel or a circle: they go round), `steps` (a count).
- **One line each for the model:** what it does, its range, default,
  unit and how far the music may move it, in the catalogue.

## The guards

Held whatever a look asks (`Look::check`, `jodel_looks::guard`):

- **Ranges.** Every knob, mix and amount held to its range; a pass or a
  knob there is none of is left out and said.
- **Flashes.** Each knob says how much of the screen's mean light its
  whole range moves (weighted by how much of the screen it lights; a test
  measures every knob and holds it to what it says). The fast drivers
  (pulses, levels, phases) together may move at most 0.08 of the light,
  under the tenth WCAG counts a flash from, however fast they beat; the
  slow ones (eased over half a second or more) 0.35. A stack asking more
  has its drives scaled down to it. On the frames themselves the flashes
  are counted as WCAG counts them (`guard::flashes`); the Shader tab shows
  the count live, and the tests hold the looks to at most three a second.
- **The lyric band.** In the band the lyrics are set in, each pass takes
  out its share of itself (the busy ones half) and the light is held
  under 0.55 (encoded), or what the picture had there if brighter: white
  words keep at least 3 to 1 before their dark outline.
- **The budget.** Each pass declares its cost (a whole-screen pass reading
  the picture once is 1); a stack may cost 8 in at most 6 passes. Over it,
  the dearest pass not marked `keep` goes (the later of equals), always
  the same one.
- **Determinism.** Nothing is random but by a seed made of the look's id
  and the song; the same look, song and time give the same frame.

## The music

Everything is read at the song's time as heard: the same time the stage,
the lyrics and the bubbles are drawn at (`App::heard_for_frame`, the
engine's position as heard, eased between its reports). The song's
curves are worked out once before it plays (`jodel_looks::Curves`, from
its analysis: the beats and bars, the waveform's lanes of the mix and
each stem, the sections, the key, the sung lines) on the song's own
frames, 100 a second, read between them; its moments (beats, kicks,
snares, bass notes, lines) as lists, each pulse worked out from its
time, rising just before the moment and peaking on it. So a look follows
seeks, pauses and changes of speed exactly, at any frame rate. What only
the singing knows is set live on the singing's clock (the voice's lag
taken off): the pitch sung, how many notes hit lately, the game's tier
and streak.

| driver | speed | what it is, what it is good for |
|---|---|---|
| `beat` | fast | a pulse on every beat, peaking on it: pumping, ticking |
| `downbeat` | fast | a pulse on each bar's first beat: flares, light leaks, accents |
| `beat_phase` | fast | where it is in the beat, 0 to 1: spinning and scrolling on the beat |
| `bar_phase` | fast | where it is in the bar: slower cycles locked to the bar |
| `tempo` | slow | the tempo, 1 at 120 a minute: faster songs move faster |
| `level` | fast | how loud the whole music is now |
| `drums`, `bass`, `vocals`, `other` | fast | each stem's level |
| `kick`, `snare` | fast | a pulse on each kick-like (beats 1, 3) and snare-like (2, 4) drum hit |
| `bass_note` | fast | a pulse on each bass note's start |
| `phrase` | fast | a pulse as each sung line starts |
| `low`, `mid`, `high` | fast | the low end (bass, kicks), the middle (voice, instruments), the top (the drums' bright hits) |
| `minor`, `key_hue` | slow | the key's mode, and its place on the circle of fifths |
| `section` | fast | a swell as a section starts |
| `energy`, `chorus`, `part` | slow | the section's loudness, 1 in the choruses, a new value each section |
| `singing`, `instrumental`, `quiet` | slow | a line being sung, a break without singing, the quiet parts |
| `pitch`, `melody` | fast | the pitch sung (live), the song's melody |
| `accuracy`, `tier`, `streak` | slow | how well it is sung lately, the game's tier and multiplier (live) |
| `progress` | slow | how far the song has gone |

Honest limits: the analysis has no spectrum, so `low`, `mid` and `high`
are made from the stems; kick and snare are told apart by where a drum
hit falls in the bar; there is no chord analysis yet, so `part` (a new
value each section) stands for harmony changes.

## The visualizer

The feedback pass keeps the frame before and draws it under this one,
zoomed, turned, swirled and flowing, fading by the second (not the frame);
it forgets it on a seek or a new look. The music's shapes (a scope's
line, a spectrum's ring and bars, rings on the beat, sparks from the
hits, a Lissajous figure) are drawn from the curves at the time heard.
A look mixes the scene and the visualizer in any proportion: a faint trail
over the picture, a ring framing it, or (the feedback's `scene` low, its
`mix` driven by `instrumental`) the visualizer filling the breaks.

## The Shader tab

The seventh tab: the playing song's cover drawn as Karaoke mode draws it,
through the tab's look, moved by the song as in the game (nothing
playing: still). The presets by family on the left (`/` searches them);
the stack in the middle: its passes (selected by a click or `[` `]`, moved
by `,` `.`, taken out by `del`, added from the list under them), the
selected pass's knobs on sliders with their ranges and meanings, and
each knob's driver (`‹ ›`) with its amount; under the picture what the
guards measure: flashes a second, the lyrics' contrast, the stack's cost
against the budget. `C` copies the stack as JSON, `K` keeps it in the data
folder (`looks/`) as a preset of your own, listed first from then on.
