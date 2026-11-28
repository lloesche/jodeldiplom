# The looks

Every song in Karaoke mode can have a look of its own: a stack of shader
passes over whatever the backdrop draws (the blurred cover, or with the
game the song's story scenes),
under the stage and the lyrics. The passes are a library
(`looks/`, the `jodel-looks` crate); a look is data (JSON), written by a
person, a preset, or a language model that picks from the catalogue and
never writes shader code. The catalogue the model is given is generated
from the passes themselves: [`looks-catalogue.md`](looks-catalogue.md)
(`cargo run -p jodel-looks --example catalogue`, and a test holds it to
the code).

## First: a look dances

A look must visibly dance to the song. A still filter over a cover is not
a look. Every pass moves by itself with the music unless a look says
otherwise (its natural drivers: grain and halation breathing with the
bass, bloom following the voice, the halftone's dots growing with the
drums, glitches and tape tracking hitting on the snares, colour and hue
moving on to a new place each section, the weather thickening in the
choruses, the vignette closing in the quiet parts), and every preset
keeps them, often with a motion of its own on top (a colour split on the
kicks, a warp on the bass, the film's weave on the bass notes). Motion is
preferred over light: warps, offsets, scale, rotation, the feedback's
zoom and turn, dot and grain size and the colours' split can move far
without a flash, where brightness pulsing is held small by the flash
guard.

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
  (of the colour wheel or a circle, 0 to 1: they go round), `steps` (a
  count, a whole number), `ratio` (one size over another).
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
  out its share of itself (the busy ones half) and the light (relative
  luminance, sRGB decoded, as the flashes are measured) is held under
  0.24, or what the picture had there if brighter: white words keep at
  least 3.5 to 1 before their dark outline.
- **The budget.** Each pass declares its cost (a whole-screen pass reading
  the picture once is 1); a stack may cost 8 in at most 6 passes. Over it,
  the dearest pass not marked `keep` goes (the later of equals), always
  the same one.
- **Washing out and blacking out.** Over a whole frame at most 15 % of the
  pixels may be clipped (nearly white or nearly black) beyond what the
  picture itself has, and its mean light stays between 0.08 and 0.75 (or
  within a tenth of the picture's own). A look whose frames over the
  song's picture would not, at 64 moments through the song (each drawn as
  the window draws it: a look that feeds back drawn for two seconds
  before it, its echoes built up), is drawn less strongly (its passes'
  mixes scaled down in steps to the strongest that keeps within), once
  for the look, the song and the picture, on a thread of its own: the
  same strength the whole song. Until it is found the look is not drawn
  (the picture as it is), or, where it is the same look or an edit of it
  (its id the same) over the same picture (the song's curves new: its
  waveform come, its grid fixed by hand, another song without a cover; a
  knob moved on the Shader tab), drawn at the strength found for it over
  that picture; then it eases to the one found over half a second (down
  in a tenth). Leaving the mode or the tab keeps a look whose strength
  is found, so coming back finds it again only for a new look, song or
  picture. The feedback cannot run away: what it carries
  over loses light every second and passes a soft knee (rising ever more
  slowly toward 0.7, never to white), and the drawn shapes (scope, rings,
  bars, sparks) add their light as a screen does, never past white.
- **All the stage's text.** The stage's text (the count-in, the song's
  title and artist, the lines set in the middle, the score's) has the
  lyrics' dark outline too, look or none.
- **Determinism.** Nothing is random but by a seed made of the look's id
  and the song; the same look, song and time give the same frame.

## The music

Everything is read at the song's time as heard: the same time the stage,
the lyrics and the bubbles are drawn at (`App::heard_for_frame`, the
engine's position as heard, eased between its reports). The music is
read in one place, `jodel_looks::drive`, for the looks and the stage's
effects alike: the beat from the song's grid (the one the sounds are
played on; its pulses and phases fade where the grid is unsure), the
lanes' levels against how loud they get, the melody, a section's swell.
The song's curves are worked out once before it plays
(`jodel_looks::Curves`, from its analysis: the grid, the waveform's lanes
of the mix and each stem, the sections, the key, the melody, the sung
lines), each driver from its own source (`drivers!` says which come from
the song and which live, and a test holds every song driver set): curves
on the song's own frames, 100 a second, read between them; its moments
(kicks, snares, bass notes, lines) as lists, each pulse worked out from
its time, rising just before the moment and peaking on it. So a look
follows seeks, pauses and changes of speed exactly, at any frame rate.
What only the singing knows is set live on the singing's clock (the
voice's lag taken off, `core::karaoke::live`): the pitch sung, how many
notes hit lately, the game's tier and streak.

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

## The presets

Some 380 presets, from the genre × theme × era matrix: each genre's own
look (from rock, punk, grunge, doom, black and power metal, shoegaze,
pop, synth-pop, disco, funk, soul, K-pop and city pop, musicals and
Schlager, through house, techno, trance, dubstep, drum and bass, ambient
and chiptune, boom bap, trap, lo-fi, dub, Afrobeat and reggaeton, jazz,
blues, country, folk, Celtic music, gospel and chanson, orchestral music and
opera, salsa, flamenco and Bollywood, to children's, Christmas and
Halloween songs), the same in its era and for the themes its songs
often sing of; each era (1920s sepia and silent film, 1950s
Technicolor, 60s psychedelia, 70s film stock, 80s neon and tape, 90s
camcorder, Y2K chrome and glitch, today) alone and with themes; each
theme (love, heartbreak, longing, party, protest, the road, nostalgia,
family, grief, faith, nature, the sea, mountains, night and the city,
space, dreams, psychedelia, money, war, home, summer, winter, horror,
humour, empowerment, the seasons) alone and on film; moods, media and
visualizers. One JSON file a family in `looks/presets/`, each saying its
kind (`genre`, `era`, `theme`, `mood`, `media`, `visualizer`): the
matrix is made from the kinds, an era's or a theme's key being its
one-tag look's tag. The matrix itself,
genre by genre with each cell's preset, is in the catalogue (its last
section), as worked examples for the model. A test draws every preset
and every pass through a whole song, frame by frame, over a dark and a
light picture, and holds it to every guard; another does the same over
real songs' curves where the machine keeps analyses.

Respectful by design: no sacred or ceremonial style is used as
decoration (light from above is light, not a symbol; colours are chosen
for the music, not for a people's flags or faiths).

## The Shader tab

The seventh tab: the playing song drawn as Karaoke mode draws it (its
story's scenes wherever it has them, the game on or off; else its cover
sharp and still; else a built-in picture, never the dark ground), through the tab's
look, moved by the song as in the game (nothing playing: still). On the
left a browser of every look: the presets by
family, then every pass as a look of its own under its pass family
(`/` searches them all, `↑` `↓` steps through them);
the stack in the middle: its passes (selected by a click or `[` `]`, moved
by `,` `.`, taken out by `del`, added from the list under them), the
selected pass's knobs on sliders with their ranges and meanings, and
each knob's driver (`‹ ›`) with its amount; under the picture what the
guards measure: flashes a second, the lyrics' contrast, the stack's cost
against the budget, the frame's exposure (clipped, mean light) and the
strength the look draws at. `C` copies the stack as JSON, `K` keeps it in the data
folder (`looks/`) as a preset of your own, listed first from then on.
