# The looks' catalogue

Made from the passes' own definitions by `cargo run -p jodel-looks --example catalogue`; a test holds it to them. How to use it: `docs/looks.md`.

A look is JSON: an `id`, and `layers`, each a `pass` by name with the `knobs` it sets (by name; left out: the default), how the music moves them (`drive`: a list of `{"knob", "by", "amount"}`, the amount a share of the knob's swing from -1 to 1; left out: the pass's own drivers; `[]`: still), its `mix` over what is under it (0 to 1; `mix` can be driven too) and `keep` (the budget never drops it).

Rules held whatever is asked: every number to its range; at most 6 passes costing 8 in all (over it, the dearest pass not kept goes); the fast drivers together may move 0.08 of the screen's light, the slow ones 0.35; the lyric band's light under 0.24 (encoded).

## Passes

### grade

#### `lift_gamma_gain` (cost 1)

the classic three-way grade: the blacks raised or crushed, the middle brighter or darker, the whites brighter or dimmer.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `lift` | -1 to 1 | 0 | share | 0.3 | raises the blacks (+, a faded film) or crushes them (-, deep shadows) |
| `gamma` | -1 to 1 | 0 | share | 0.4 | the midtones brighter (+) or darker (-), the ends kept |
| `gain` | -1 to 1 | 0 | share | 0.4 | the whites brighter (+) or dimmer (-) |

Moved by itself: `gain` by `kick` (0.8), `gamma` by `energy` (0.5).

Goes well with: `curve`, `grain`, `vignette`.

#### `white_balance` (cost 1)

white balance and tint, the light kept: daylight to tungsten, a green or a magenta cast.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `warmth` | -1 to 1 | 0 | share | 0.6 | warmer, orange (+), or cooler, blue (-) |
| `tint` | -1 to 1 | 0 | share | 0.4 | magenta (+) or green (-) |

Moved by itself: `warmth` by `part` (0.7), `tint` by `bass` (0.5).

Goes well with: `curve`, `grain`.

#### `saturation` (cost 1)

how vivid the colours are: saturation for all of them, vibrance for the dull ones only (skin and sky kept).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `saturation` | 0 to 2 | 1 | share | 0.6 | 0 grey, 1 as it is, 2 twice as vivid |
| `vibrance` | -1 to 1 | 0 | share | 0.6 | the dull colours more vivid (+) or duller (-), the vivid ones kept |

Moved by itself: `saturation` by `chorus` (0.6), `vibrance` by `vocals` (0.7).

Goes well with: `curve`, `duotone`.

#### `duotone` (cost 1)

the picture in two colours, dark to light (the cover's own, or two hues): posters, pop, moods.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dark` | 0 to 1 | 0.7 | turns | 0.1 | the shadows' hue |
| `light` | 0 to 1 | 0.1 | turns | 0.15 | the lights' hue |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the two colours |
| `cover` | 0 to 1 | 0 | share | 0 | from the two hues (0) to the cover's own colours (1) |
| `contrast` | -1 to 1 | 0 | share | 0.5 | flatter (-) or harder (+) from dark to light |

Moved by itself: `light` by `part` (0.7), `contrast` by `kick` (0.7).

Goes well with: `grain`, `halftone`, `curve`.

#### `tritone` (cost 1)

the picture in three colours, shadows, midtones and lights: screen prints, neon nights, sunsets.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dark` | 0 to 1 | 0.75 | turns | 0.1 | the shadows' hue |
| `middle` | 0 to 1 | 0.95 | turns | 0.15 | the midtones' hue |
| `light` | 0 to 1 | 0.12 | turns | 0.1 | the lights' hue |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the three colours |

Moved by itself: `middle` by `part` (0.8), `light` by `vocals` (0.5).

Goes well with: `grain`, `halftone`, `glow`.

#### `split_tone` (cost 1)

the shadows tinted one way and the lights another (teal and orange, the film look).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `shadows` | 0 to 1 | 0.52 | turns | 0.1 | the shadows' hue |
| `lights` | 0 to 1 | 0.08 | turns | 0.1 | the lights' hue |
| `amount` | 0 to 1 | 0.5 | share | 0.4 | how strongly tinted |
| `balance` | -1 to 1 | 0 | share | 0.5 | where shadows end and lights begin: lower (-) or higher (+) |

Moved by itself: `amount` by `vocals` (0.7), `balance` by `bass` (0.6).

Goes well with: `curve`, `grain`, `halation`.

#### `curve` (cost 1)

a tone curve: more or less contrast about a pivot, and the matte fade of printed film.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `contrast` | -1 to 1 | 0 | share | 0.4 | flatter (-) or punchier (+) |
| `pivot` | 0.2 to 0.8 | 0.5 | share | 0.1 | the tone the contrast turns about |
| `fade` | 0 to 1 | 0 | share | 0.4 | the blacks lifted and the whites lowered, matte |

Moved by itself: `contrast` by `kick` (0.7), `fade` by `quiet` (0.6).

Goes well with: `split_tone`, `grain`.

#### `gradient_map` (cost 1)

the tones mapped onto the cover's three colours, dark to light: a picture made the cover's palette.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.4 | how much of the cover's palette |

Moved by itself: `amount` by `energy` (0.6).

Goes well with: `grain`, `halftone`.

#### `technicolor` (cost 1)

the three-strip 1950s colour: each primary separated and rich, reds deep, skies cyan.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.8 | share | 0.3 | how much of the look |
| `punch` | 0 to 1 | 0.5 | share | 0.5 | how far the dyes separate |

Moved by itself: `punch` by `chorus` (0.8), `amount` by `vocals` (0.4).

Goes well with: `grain`, `halation`, `curve`.

#### `bleach_bypass` (cost 1)

the silver left in the print: harsh contrast, washed-out colour, metal and war films.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much of the look |
| `contrast` | 0 to 1 | 0.5 | share | 0.5 | how harsh |

Moved by itself: `contrast` by `drums` (0.7).

Goes well with: `grain`, `vignette`.

#### `cross_process` (cost 1)

slide film in the wrong chemistry: yellow-green lights, blue shadows, hard reds (90s indie, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.4 | how much of the look |

Moved by itself: `amount` by `energy` (0.6).

Goes well with: `grain`, `vignette`, `light_leak`.

#### `infrared` (cost 1)

infrared film: greens turn pink and red, skies deep (or in black and white, foliage glowing white).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.8 | share | 0.4 | how much of the look |
| `mono` | 0 to 1 | 0 | share | 0 | colour infrared (0) or black-and-white infrared (1) |

Moved by itself: `amount` by `vocals` (0.6).

Goes well with: `glow`, `grain`.

#### `sepia` (cost 1)

an old photograph: warm sepia brown or a daguerreotype's cool silver, faded with age.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.9 | share | 0.3 | how much of the look |
| `tone` | -1 to 1 | 1 | share | 0.4 | a daguerreotype's silver (-1) to warm sepia (+1) |
| `fade` | 0 to 1 | 0.3 | share | 0.4 | how faded with age |

Moved by itself: `fade` by `quiet` (0.6), `tone` by `part` (0.5).

Goes well with: `film_damage`, `vignette`, `gate_weave`.

### film

#### `grain` (cost 1)

film grain: fine to coarse, mono or coloured, strongest in the midtones, new every film frame.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.35 | share | 0.5 | how much grain |
| `size` | 0.6 to 3 | 1.2 | px | 1 | the grains' size |
| `chroma` | 0 to 1 | 0.15 | share | 0.2 | mono (0) to coloured grain (1) |

Moved by itself: `amount` by `bass` (0.8), `size` by `drums` (0.6).

Goes well with: `halation`, `vignette`, `curve`, `gate_weave`.

#### `halation` (cost 3)

halation: the red-orange glow film gets round bright lights, the 35 mm look at night.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how strong the glow |
| `radius` | 2 to 40 | 12 | px | 10 | how far it spreads |
| `warmth` | 0 to 1 | 0.6 | share | 0.2 | pinkish (0) to deep red-orange (1) |

Moved by itself: `amount` by `bass` (0.8), `radius` by `kick` (0.6).

Goes well with: `grain`, `split_tone`, `vignette`.

#### `glow` (cost 3)

bloom: the bright parts glowing round, tinted if wanted (dreamy, neon, stage light).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how strong the glow |
| `threshold` | 0.3 to 0.95 | 0.65 | share | 0.1 | how bright a part must be to glow |
| `radius` | 4 to 60 | 20 | px | 16 | how far it spreads |
| `hue` | 0 to 1 | 0.1 | turns | 0.1 | the glow's tint |
| `tint` | 0 to 1 | 0.2 | share | 0.2 | white (0) to the hue (1) |

Moved by itself: `amount` by `vocals` (0.8), `radius` by `bass` (0.6).

Goes well with: `diffusion`, `neon_edges`, `lift_gamma_gain`.

#### `diffusion` (cost 2.5)

a diffusion filter: soft focus, lights blooming gently, blacks a little lifted (dreamy, romantic, 70s).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how soft |
| `radius` | 2 to 30 | 8 | px | 8 | how far the softness reaches |

Moved by itself: `amount` by `vocals` (0.7), `radius` by `bass` (0.5).

Goes well with: `glow`, `curve`, `grain`.

#### `gate_weave` (cost 1)

gate weave: the whole frame drifting a little, as film does in an old projector.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.6 | how far it drifts (up to 2.5 pixels) |
| `speed` | 0.2 to 3 | 1 | share | 0.5 | how fast it drifts |

Moved by itself: `amount` by `bass_note` (0.9).

Goes well with: `grain`, `film_damage`, `sepia`.

#### `light_leak` (cost 1)

a light leak: warm light spilling in from an edge of the frame, drifting (summer, nostalgia, instant cameras).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the leak |
| `hue` | 0 to 1 | 0.06 | turns | 0.1 | its colour (orange, red, magenta ...) |
| `place` | 0 to 1 | 0.15 | turns | 0.3 | where round the frame it comes in |
| `size` | 0 to 1 | 0.5 | share | 0.3 | how far into the frame it reaches |

Moved by itself: `amount` by `downbeat` (0.7), `place` by `part` (0.6).

Goes well with: `grain`, `polaroid`, `cross_process`.

#### `vignette` (cost 1)

a vignette: the edges darkened, round as a lens does or along the frame, soft or hard.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how dark the edges |
| `softness` | 0 to 1 | 0.5 | share | 0.3 | how far in the darkening starts |
| `shape` | 0 to 1 | 0 | share | 0 | round (0) to the frame's shape (1) |

Moved by itself: `amount` by `quiet` (0.8), `softness` by `kick` (0.6).

Goes well with: `grain`, `curve`, `sepia`.

#### `polaroid` (cost 1.2)

an instant photo: the picture in a white frame with a wide bottom, faded and warm (family, nostalgia, summer).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `frame` | 0 to 1 | 0.6 | share | 0 | how wide the white frame (0: none) |
| `fade` | 0 to 1 | 0.5 | share | 0.3 | how faded, lifted and teal in the shadows |
| `warmth` | -1 to 1 | 0.3 | share | 0.5 | cooler (-) or warmer (+) |

Moved by itself: `warmth` by `part` (0.6), `fade` by `quiet` (0.5).

Goes well with: `light_leak`, `grain`, `vignette`.

#### `film_damage` (cost 1)

dust specks and scratches, new each film frame: an old print, a silent film.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how much dust |
| `age` | 0 to 1 | 0.5 | share | 0.3 | how worn: scratches and darker dirt |

Moved by itself: `amount` by `snare` (0.9), `age` by `bass` (0.5).

Goes well with: `sepia`, `gate_weave`, `grain`, `vignette`.

#### `letterbox` (cost 1)

a cinema's black bars, the picture as wide as a film (ballads, epics, soundtracks, road movies).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `ratio` | 1.85 to 2.76 | 2.39 | ratio | 0 | the picture's width to its height (1.85 flat, 2.39 scope) |
| `amount` | 0 to 1 | 1 | share | 0 | how black the bars |

Goes well with: `split_tone`, `grain`, `halation`.

#### `bokeh` (cost 1.3)

bokeh: soft balls of out-of-focus light drifting in the dark parts (Christmas, love songs, city nights, soul and R&B).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.4 | how many lights |
| `size` | 0.5 to 3 | 1 | share | 0.8 | how big |
| `hue` | 0 to 1 | 0.1 | turns | 0.2 | their colour |
| `cover` | 0 to 1 | 0 | share | 0 | the hue (0) or the cover's colour (1) |

Moved by itself: `size` by `vocals` (0.7), `amount` by `chorus` (0.6).

Goes well with: `glow`, `diffusion`, `white_balance`.

### print

#### `halftone` (cost 1.5)

a halftone screen: the picture as dots on paper, black ink or the picture's own colours (newsprint, comics, posters).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 3 to 30 | 8 | px | 6 | the dots' spacing |
| `angle` | 0 to 1 | 0.125 | turns | 0.1 | the screen's angle |
| `ink` | 0 to 1 | 0 | share | 0 | black ink (0) to the picture's colours (1) |
| `paper` | 0 to 1 | 0.3 | share | 0 | white (0) to yellowed (1) paper |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `dot` by `drums` (0.8), `angle` by `part` (0.5).

Goes well with: `duotone`, `posterize_ink`, `curve`.

#### `halftone_cmyk` (cost 2.5)

four-colour process print: cyan, magenta, yellow and black screens at their angles, slightly off register if wanted (comics, magazines, pop art).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 3 to 24 | 7 | px | 4 | the dots' spacing |
| `angle` | 0 to 1 | 0 | turns | 0.05 | the screens turned together |
| `registration` | 0 to 1 | 0.2 | share | 0.6 | how far the plates are off register (up to 4 pixels) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `registration` by `snare` (0.9), `dot` by `kick` (0.6).

Goes well with: `posterize_ink`, `saturation`, `ben_day`.

#### `ben_day` (cost 2)

pop-art comic print: flat bold colours, Ben-Day dots in the midtones, black outlines.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 4 to 20 | 10 | px | 5 | the dots' spacing |
| `levels` | 2 to 5 | 3 | steps | 0 | how many steps each colour has |
| `outline` | 0 to 1 | 0.7 | share | 0.3 | how black the outlines |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `dot` by `kick` (0.8), `outline` by `drums` (0.5).

Goes well with: `halftone_cmyk`, `saturation`, `duotone`.

#### `posterize_ink` (cost 1.5)

cel shading: the tones in a few flat steps and ink lines round the edges (comics, cartoons, anime).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `steps` | 2 to 8 | 4 | steps | 0 | how many tones |
| `ink` | 0 to 1 | 0.7 | share | 0.3 | how dark the lines |
| `width` | 0.5 to 4 | 1.5 | px | 1.5 | how wide the lines |
| `vivid` | 0 to 2 | 1.2 | share | 0.5 | the colours' saturation |

Moved by itself: `width` by `kick` (0.8), `vivid` by `chorus` (0.6).

Goes well with: `halftone`, `ben_day`, `saturation`.

#### `risograph` (cost 2)

a risograph print: two bright inks laid one over the other, slightly off register, grainy (zines, indie, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `first` | 0 to 1 | 0.93 | turns | 0.05 | the first ink's hue (fluorescent pink by default) |
| `second` | 0 to 1 | 0.6 | turns | 0.05 | the second ink's hue (blue by default) |
| `misregister` | 0 to 1 | 0.35 | share | 0.6 | how far the inks are apart (up to 6 pixels) |
| `grain` | 0 to 1 | 0.5 | share | 0.4 | how grainy the ink |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `misregister` by `bass_note` (0.9), `grain` by `drums` (0.5).

Goes well with: `grain`, `halftone`, `canvas`.

#### `xerox` (cost 1)

a photocopy of a photocopy: hard black and white, toner speckle and streaks (punk flyers, zines, hardcore).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `threshold` | 0.2 to 0.8 | 0.5 | share | 0.1 | where black turns white: more black (+) or more white (-) |
| `toner` | 0 to 1 | 0.5 | share | 0.5 | how much speckle and streaking |
| `contrast` | 0 to 1 | 0.7 | share | 0.2 | soft grey (0) to hard black and white (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the copy over the picture |

Moved by itself: `toner` by `snare` (0.9), `threshold` by `bass` (0.4).

Goes well with: `duotone`, `film_damage`, `glitch`.

#### `woodcut` (cost 1)

a woodcut or linocut: the picture carved in black lines that swell in the dark and follow the forms (folk, doom and black metal, protest posters).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `lines` | 3 to 20 | 7 | px | 3 | the lines' spacing |
| `angle` | 0 to 1 | 0.08 | turns | 0.05 | the lines' angle |
| `threshold` | -1 to 1 | 0 | share | 0.4 | darker (-) or lighter (+) overall |
| `ink` | 0 to 1 | 0 | share | 0 | black ink (0) to the cover's colour, darkened (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `threshold` by `bass` (0.6), `lines` by `kick` (0.5).

Goes well with: `duotone`, `canvas`, `vignette`.

#### `engraving` (cost 1.3)

an engraving: the tones in fine hatching, crossed in the darks (banknotes, old books, classical, folk).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `spacing` | 2 to 10 | 4 | px | 2 | the lines' spacing |
| `angle` | 0 to 1 | 0.125 | turns | 0.05 | the first lines' angle |
| `layers` | 1 to 3 | 2 | steps | 0 | how many directions of hatching |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the engraving over the picture |

Moved by itself: `spacing` by `bass` (0.6).

Goes well with: `sepia`, `canvas`, `vignette`.

#### `stipple` (cost 1.3)

stippling: the picture in ink dots, denser where it is dark (pen-and-ink, old scientific plates).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 2 to 10 | 4 | px | 2.5 | the dots' size |
| `contrast` | 0 to 1 | 0.5 | share | 0.3 | how sharply dark and light part |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the stippling over the picture |

Moved by itself: `dot` by `drums` (0.7).

Goes well with: `sepia`, `canvas`.

### paint

#### `kuwahara` (cost 4)

a painterly smear that keeps the edges: flat patches of colour like brushwork (oil, gouache, anime backgrounds).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `radius` | 2 to 10 | 4 | px | 3 | the brush's size |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |

Moved by itself: `radius` by `bass` (0.8).

Goes well with: `canvas`, `oil_paint`, `saturation`.

#### `watercolor` (cost 2.5)

watercolour: soft washes that bleed at wobbly edges, pigment pooling darker there, the paper showing through (folk, childhood, gentle songs).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `wash` | 0 to 1 | 0.6 | share | 0.4 | how far the colours wash together |
| `bleed` | 0 to 1 | 0.5 | share | 0.4 | how wobbly the edges and how dark they pool |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |
| `granulation` | 0 to 1 | 0.4 | share | 0.2 | the pigment's grain in the darks |

Moved by itself: `bleed` by `vocals` (0.7), `wash` by `bass` (0.5).

Goes well with: `canvas`, `lift_gamma_gain`, `saturation`.

#### `oil_paint` (cost 3)

oil paint: the colour smeared in strokes that follow a flow, their impasto ridges lit from one side (classic, ballads, folk portraits).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `brush` | 2 to 12 | 5 | px | 2 | the strokes' size |
| `relief` | 0 to 1 | 0.5 | share | 0.4 | how much the ridges catch the light |
| `smear` | 0 to 1 | 0.6 | share | 0.4 | how long the strokes |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |

Moved by itself: `relief` by `bass` (0.8), `smear` by `vocals` (0.6).

Goes well with: `kuwahara`, `canvas`, `vignette`.

#### `canvas` (cost 1)

the ground showing through: canvas weave or paper fibre over the whole picture, the finish of every painted or drawn look.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `kind` | 0 to 1 | 0 | share | 0 | canvas weave (0) to paper fibre (1) |
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how strongly the ground shows |
| `scale` | 0.5 to 3 | 1 | share | 0.3 | the weave's or fibre's size |

Moved by itself: `amount` by `drums` (0.6).

Goes well with: `oil_paint`, `watercolor`, `charcoal`, `woodcut`.

#### `charcoal` (cost 1.8)

charcoal or chalk on paper: smudged darks in strokes, the paper's tooth catching the dust (blues, folk, grief, sketches).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `smudge` | 0 to 1 | 0.4 | share | 0.4 | how smudged |
| `grain` | 0 to 1 | 0.6 | share | 0.2 | how much of the paper's tooth shows |
| `contrast` | 0 to 1 | 0.5 | share | 0.3 | how dark the darks |

Moved by itself: `smudge` by `bass` (0.7).

Goes well with: `canvas`, `vignette`, `sepia`.

#### `pastel` (cost 1.2)

wax crayon or soft pastel: colour laid in strokes, the paper's tooth showing white through it (children's songs, childhood, sunny pop).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `stroke` | 2 to 12 | 6 | px | 3 | the strokes' size |
| `tooth` | 0 to 1 | 0.5 | share | 0.3 | how much paper shows through |
| `vivid` | 0.5 to 1.5 | 1.1 | share | 0.2 | the colours' saturation |

Moved by itself: `stroke` by `drums` (0.6).

Goes well with: `canvas`, `saturation`, `marker`.

#### `marker` (cost 1.5)

felt-tip markers: flat bold colours in streaky strokes, darker where they overlap, a pen outline (pop, children's songs, humour).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `steps` | 2 to 6 | 3 | steps | 0 | how many tones each colour has |
| `stroke` | 4 to 20 | 10 | px | 3 | the strokes' width |
| `outline` | 0 to 1 | 0.6 | share | 0.4 | how dark the pen outline |

Moved by itself: `outline` by `kick` (0.7).

Goes well with: `canvas`, `pastel`, `posterize_ink`.

#### `pencil` (cost 1.5)

a graphite pencil sketch: outlines, hatching in the shadows crossed in the darkest (singer-songwriters, folk, childhood, memories).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the sketch over the picture |
| `hatch` | 0 to 1 | 0.6 | share | 0.4 | how much hatching in the shadows |
| `spacing` | 2 to 8 | 3.5 | px | 1 | the hatching's spacing |
| `pressure` | 0 to 1 | 0.5 | share | 0.4 | how dark the lines |

Moved by itself: `pressure` by `vocals` (0.6), `hatch` by `drums` (0.5).

Goes well with: `canvas`, `sepia`, `vignette`.

#### `chalkboard` (cost 1.6)

chalk on a dark board: the outlines and the lights in dusty chalk strokes (school days, children's songs, humour, nostalgia).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the board over the picture |
| `board` | 0 to 1 | 0.4 | turns | 0 | the board's hue (green by default, or slate) |
| `chalk` | 0 to 1 | 0.6 | share | 0.4 | outlines only (0) to the lights filled in chalk (1) |

Moved by itself: `chalk` by `chorus` (0.6).

Goes well with: `pastel`, `vignette`.

### medium

#### `stained_glass` (cost 2)

leaded glass: the picture in pieces of coloured glass between dark lead lines, lit from behind (ballads, folk, fantasy).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `cell` | 12 to 80 | 36 | px | 12 | the pieces' size |
| `lead` | 0.5 to 6 | 2.5 | px | 1 | the lead lines' width |
| `glow` | 0 to 1 | 0.5 | share | 0.5 | how much the light shines through each piece's middle |

Moved by itself: `glow` by `vocals` (0.8), `cell` by `bass` (0.5).

Goes well with: `glow`, `vignette`.

#### `mosaic` (cost 1)

a mosaic: the picture in square tiles of one colour each, grout between them (Mediterranean, ancient, summer).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `tile` | 6 to 40 | 14 | px | 6 | the tiles' size |
| `grout` | 0 to 1 | 0.4 | share | 0.2 | how wide and how visible the grout |
| `variety` | 0 to 1 | 0.3 | share | 0.2 | how much each tile's shade varies |

Moved by itself: `tile` by `kick` (0.7).

Goes well with: `saturation`, `vignette`.

#### `cross_stitch` (cost 1)

cross-stitch: the picture in little X stitches of a few thread colours on cloth (folk, family, cosy, Volksmusik).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `stitch` | 4 to 20 | 8 | px | 3 | the stitches' size |
| `levels` | 2 to 5 | 3 | steps | 0 | how many shades of red, of green and of blue the threads come in |
| `fabric` | 0 to 1 | 0.6 | share | 0.2 | how much bare cloth is left in the lights |

Moved by itself: `stitch` by `kick` (0.6).

Goes well with: `canvas`, `vignette`.

#### `cyanotype` (cost 1.6)

a cyanotype's Prussian blue print, or a blueprint's white lines and grid on blue (science, the sea, dreams, plans).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `plan` | 0 to 1 | 0 | share | 0 | a cyanotype print (0) to a blueprint's lines (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the medium over the picture |
| `grid` | 0 to 1 | 0.3 | share | 0.4 | how much of a blueprint's grid |

Moved by itself: `grid` by `drums` (0.6).

Goes well with: `canvas`, `vignette`.

#### `thermal` (cost 1)

a heat camera: the brightness read as heat, in an iron palette or a rainbow (techno, sci-fi, war, horror, sweat and dance).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the camera over the picture |
| `palette` | 0 to 1 | 0 | share | 0 | iron, black to white-hot (0), to a rainbow (1) |
| `noise` | 0 to 1 | 0.3 | share | 0.5 | the sensor's noise |

Moved by itself: `noise` by `high` (0.8).

Goes well with: `crt`, `glitch`.

#### `night_vision` (cost 1)

night vision: a green phosphor picture, noisy and lined, in a round tube (night, war, spies, horror).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the goggles over the picture |
| `noise` | 0 to 1 | 0.4 | share | 0.5 | the tube's noise |
| `lines` | 0 to 1 | 0.3 | share | 0.2 | how strong the scan lines |

Moved by itself: `noise` by `drums` (0.8).

Goes well with: `vignette`, `glitch`.

#### `negative` (cost 1)

a film negative, or an X-ray's cold glow (horror, Halloween, the surreal, sci-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.2 | how much of the negative over the picture |
| `xray` | 0 to 1 | 0.5 | share | 0.4 | a colour negative (0) to an X-ray (1) |
| `contrast` | 0 to 1 | 0.4 | share | 0.2 | how dark the X-ray's dark parts |

Moved by itself: `xray` by `part` (0.6).

Goes well with: `glow`, `grain`.

#### `solarize` (cost 1)

solarisation, the darkroom's turned tones, or liquid chrome's bands (psychedelia, Y2K chrome, glam).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `threshold` | 0.3 to 0.9 | 0.55 | share | 0.15 | where the tones turn back |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the effect over the picture |
| `chrome` | 0 to 1 | 0 | share | 0 | solarised (0) to chrome bands (1) |

Moved by itself: `threshold` by `bass` (0.6).

Goes well with: `hue`, `glow`.

#### `neon_edges` (cost 2)

the picture's edges as glowing neon tubes over a darkened picture (synthwave, the 80s, night and the city, club music).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `width` | 0.5 to 4 | 1.5 | px | 1 | the tubes' width |
| `hue` | 0 to 1 | 0.85 | turns | 0.3 | the neon's colour |
| `glow` | 0 to 1 | 0.6 | share | 0.5 | how far the glow spreads |
| `dark` | 0 to 1 | 0.6 | share | 0.2 | how dark the picture under the neon |
| `cycle` | 0 to 1 | 0 | share | 0.3 | still (0) to a rainbow running through the colours (1) |

Moved by itself: `glow` by `kick` (0.8), `hue` by `part` (0.6).

Goes well with: `glow`, `crt`, `tritone`.

#### `teletext` (cost 2)

teletext's block graphics: each character cell of six blocks lit or dark in one of eight bright colours on black (no letters: shapes only) (the 80s, chiptune, retro TV).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `cell` | 6 to 24 | 12 | px | 5 | a cell's width |
| `cover` | 0 to 1 | 0 | share | 0 | teletext's eight colours (0) to the cover's colour (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the blocks over the picture |

Moved by itself: `cell` by `kick` (0.6).

Goes well with: `crt`, `pixelate`.

#### `cutout` (cost 1.3)

cut paper: the tones as a few layers of coloured paper, each casting a little shadow on the one under it (indie pop, children's songs, whimsy, animation).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `layers` | 2 to 6 | 4 | steps | 0 | how many layers of paper |
| `shadow` | 0 to 1 | 0.6 | share | 0.4 | how deep the layers' shadows |
| `paper` | 0 to 1 | 0.5 | share | 0.2 | how much the paper's fibre shows |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the collage over the picture |

Moved by itself: `shadow` by `kick` (0.7).

Goes well with: `canvas`, `saturation`, `posterize_ink`.

#### `emboss` (cost 1.3)

the picture embossed as a relief in stone, silver or gold, lit from one side (metal, power metal, anthems, epic and classical).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the relief over the picture |
| `light` | 0 to 1 | 0.125 | turns | 0.2 | where the light comes from |
| `metal` | 0 to 1 | 0 | share | 0 | grey stone or silver (0) to gold (1) |
| `depth` | 0 to 1 | 0.5 | share | 0.5 | how deep the relief |

Moved by itself: `depth` by `bass` (0.8), `light` by `bar_phase` (0.5).

Goes well with: `glow`, `vignette`, `bleach_bypass`.

### retro

#### `crt` (cost 2)

an old tube screen: scan lines, the phosphor mask's stripes, a curved glass and its glow (the 80s, video games, synthwave).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `scanlines` | 0 to 1 | 0.5 | share | 0.3 | how strong the scan lines |
| `mask` | 0 to 1 | 0.4 | share | 0.2 | how strong the colour mask's stripes |
| `curve` | 0 to 1 | 0.3 | share | 0 | how curved the glass |
| `glow` | 0 to 1 | 0.3 | share | 0.5 | how much the phosphor glows |

Moved by itself: `glow` by `bass` (0.7), `scanlines` by `drums` (0.5).

Goes well with: `vhs`, `pixelate`, `palette`, `teletext`.

#### `vhs` (cost 1.8)

a worn video tape: lines wobbling, the colour bleeding sideways, noise and a band of bad tracking rolling up (the 80s and 90s, camcorder, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `wobble` | 0 to 1 | 0.4 | share | 0.5 | how much the lines wobble |
| `bleed` | 0 to 1 | 0.5 | share | 0.3 | how far the colour bleeds |
| `noise` | 0 to 1 | 0.3 | share | 0.3 | how much noise and snow |
| `tracking` | 0 to 1 | 0.3 | share | 0.5 | how bad the tracking band |

Moved by itself: `tracking` by `snare` (0.9), `wobble` by `bass` (0.7).

Goes well with: `crt`, `cross_process`, `grain`.

#### `pixelate` (cost 1)

big pixels in a few shades a channel, an ordered dither between them (8-bit and 16-bit games, chiptune).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `size` | 2 to 24 | 6 | px | 5 | the pixels' size |
| `levels` | 2 to 16 | 6 | steps | 0 | how many shades a channel |
| `dither` | 0 to 1 | 0.6 | share | 0.2 | none (0) to a full ordered dither (1) |

Moved by itself: `size` by `kick` (0.7).

Goes well with: `crt`, `palette`, `glitch`.

#### `palette` (cost 1)

an old machine's few colours, dithered: a handheld's four greens, a home computer's black-cyan-magenta-white, an eight-colour micro's, or the cover's own four (chiptune, video-game music).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `palette` | 0 to 3 | 0 | steps | 0 | handheld greens (0), cyan-magenta (1), eight colours (2), the cover's (3) |
| `size` | 2 to 12 | 4 | px | 3 | the pixels' size |
| `dither` | 0 to 1 | 0.5 | share | 0.2 | none (0) to a full ordered dither (1) |

Moved by itself: `size` by `kick` (0.6).

Goes well with: `crt`, `pixelate`.

#### `glitch` (cost 1.5)

digital glitches: blocks of the picture torn sideways and tinted, its colours split, new twelve times a second (Y2K, datamosh, industrial, dubstep drops).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.3 | share | 0.6 | how many blocks tear |
| `block` | 8 to 80 | 32 | px | 16 | the blocks' size |
| `split` | 0 to 1 | 0.4 | share | 0.5 | how far the colours split |

Moved by itself: `amount` by `snare` (1), `split` by `kick` (0.6).

Goes well with: `vhs`, `chromatic`, `xerox`.

#### `anaglyph` (cost 1)

a red-and-cyan 3D picture, the two eyes' views apart (the 50s cinema, B-movies, sci-fi, fun).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `offset` | 1 to 20 | 6 | px | 6 | how far apart the eyes' views |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the 3D over the picture |
| `depth` | 0 to 1 | 0.5 | share | 0.2 | how much brighter parts come forward |

Moved by itself: `offset` by `kick` (0.8).

Goes well with: `grain`, `technicolor`.

#### `chromatic` (cost 1)

the colours split apart, red one way and blue the other, as a cheap lens or a shaken screen does (psychedelia, glitch, drops).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.3 | share | 0.6 | how far the colours split (up to 12 pixels) |
| `radial` | 0 to 1 | 1 | share | 0 | the same way everywhere (0) to out from the middle (1) |
| `angle` | 0 to 1 | 0 | turns | 0.2 | the way they split, when not out from the middle |

Moved by itself: `amount` by `kick` (0.9).

Goes well with: `glitch`, `hue`, `liquid`.

### atmosphere

#### `haze` (cost 1)

haze: the picture fading into the air's colour toward the top and the distance (mountains, summer heat, dreams, longing).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how thick the haze |
| `hue` | 0 to 1 | 0.58 | turns | 0.1 | the air's colour |
| `horizon` | 0 to 1 | 0.6 | share | 0.1 | where the haze starts, up from the bottom |
| `height` | 0.1 to 1 | 0.5 | share | 0.2 | how far up it thickens |

Moved by itself: `amount` by `quiet` (0.6), `height` by `vocals` (0.5).

Goes well with: `god_rays`, `split_tone`, `glow`.

#### `god_rays` (cost 4)

rays of light streaming from a bright point through the picture (gospel, anthems, sunrise, faith, choruses).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the rays |
| `x` | 0 to 1 | 0.5 | share | 0.1 | the light's place across |
| `y` | 0 to 1 | 0.25 | share | 0.1 | the light's place down |
| `length` | 0.1 to 1 | 0.5 | share | 0.3 | how long the rays |

Moved by itself: `amount` by `chorus` (0.7), `length` by `bass` (0.6).

Goes well with: `haze`, `glow`, `lens_flare`.

#### `fog` (cost 1.2)

fog drifting across in slow banks, lying low if wanted (horror, Halloween, dub smoke, grief, mountains).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how thick the fog |
| `scale` | 0.5 to 4 | 1.5 | share | 0.5 | the banks' size: big (low) to small (high) |
| `speed` | 0 to 2 | 0.5 | share | 0.8 | how fast it drifts |
| `hue` | 0 to 1 | 0.6 | turns | 0.1 | its colour |
| `low` | 0 to 1 | 0.5 | share | 0.2 | everywhere (0) to lying low (1) |

Moved by itself: `amount` by `bass` (0.6), `speed` by `energy` (0.6).

Goes well with: `vignette`, `haze`, `night_vision`.

#### `rain` (cost 1.5)

rain falling in streaks, near and far, slanting with the wind (heartbreak, grief, the city at night, blues).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how heavy the rain |
| `slant` | -1 to 1 | 0.2 | share | 0.4 | the wind's slant, left (-) or right (+) |
| `length` | 0 to 1 | 0.5 | share | 0.2 | how long the streaks |
| `speed` | 0.5 to 2 | 1 | share | 0.3 | how fast it falls |

Moved by itself: `amount` by `chorus` (0.8), `slant` by `bass` (0.5).

Goes well with: `fog`, `split_tone`, `glow`.

#### `snow` (cost 1.3)

snow falling softly in layers, near flakes bigger, drifting with the wind (Christmas, winter, cosy songs, Schlager).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how heavy the snow |
| `size` | 0.5 to 3 | 1 | share | 0.5 | the flakes' size |
| `wind` | -1 to 1 | 0.2 | share | 0.5 | blowing left (-) or right (+) |
| `speed` | 0.3 to 2 | 1 | share | 0.3 | how fast it falls |

Moved by itself: `amount` by `chorus` (0.7), `wind` by `bass` (0.5).

Goes well with: `glow`, `white_balance`, `vignette`.

#### `petals` (cost 1.2)

blossom petals or autumn leaves tumbling down (love, spring, K-pop and J-pop ballads, autumn, nostalgia).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how many fall |
| `kind` | 0 to 1 | 0 | share | 0 | pink blossom (0) to autumn leaves (1) |
| `size` | 0.5 to 3 | 1 | share | 0.4 | their size |
| `wind` | -1 to 1 | 0.3 | share | 0.5 | blowing left (-) or right (+) |

Moved by itself: `amount` by `chorus` (0.7), `wind` by `other` (0.6).

Goes well with: `glow`, `diffusion`, `split_tone`.

#### `dust` (cost 1)

dust motes floating slowly, glinting where the light is (old rooms, memories, folk, attics, quiet songs).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how many motes |
| `size` | 0.5 to 3 | 1 | share | 0.6 | their size |
| `lit` | 0 to 1 | 0.6 | share | 0 | everywhere alike (0) to glinting only in the light (1) |

Moved by itself: `amount` by `other` (0.7), `size` by `bass` (0.5).

Goes well with: `god_rays`, `sepia`, `film_damage`.

#### `heat_haze` (cost 1)

heat shimmering the picture, more near the ground (desert, summer, road trips, Americana, flamenco).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how strong the shimmer |
| `scale` | 0.5 to 3 | 1 | share | 0.3 | the ripples' size |
| `ground` | 0 to 1 | 0.6 | share | 0 | everywhere (0) to near the ground (1) |

Moved by itself: `amount` by `energy` (0.7).

Goes well with: `haze`, `white_balance`, `grain`.

#### `underwater` (cost 1.3)

under water: the reds absorbed, the picture wavering, caustic light dancing from above (the sea, dreams, ambient, dub).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much under water |
| `caustics` | 0 to 1 | 0.5 | share | 0.5 | how bright the dancing light |
| `depth` | 0 to 1 | 0.5 | share | 0.3 | how deep: how much red is lost |

Moved by itself: `caustics` by `other` (0.7), `depth` by `bass` (0.5).

Goes well with: `god_rays`, `liquid`, `glow`.

#### `candle` (cost 1)

candlelight: a warm pool of gently flickering light, the rest in shadow (Christmas, chanson, intimate ballads, faith).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.3 | how much candlelight and shadow |
| `warmth` | 0 to 1 | 0.7 | share | 0.2 | pale gold (0) to deep orange (1) |
| `x` | 0 to 1 | 0.5 | share | 0 | the candle's place across |
| `y` | 0 to 1 | 0.65 | share | 0 | the candle's place down |
| `reach` | 0.2 to 1 | 0.6 | share | 0.25 | how far its light reaches |

Moved by itself: `reach` by `vocals` (0.6).

Goes well with: `glow`, `grain`, `vignette`.

#### `aurora` (cost 1.2)

northern lights: glowing curtains rippling in the sky, over the dark parts (winter, space, ambient, wonder).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the curtains |
| `hue` | 0 to 1 | 0.38 | turns | 0.2 | their colour (green by default) |
| `height` | 0 to 1 | 0.35 | share | 0.2 | how far down they reach |
| `speed` | 0 to 2 | 0.5 | share | 0.6 | how fast they ripple |

Moved by itself: `amount` by `other` (0.6), `hue` by `part` (0.5), `speed` by `energy` (0.6).

Goes well with: `starfield`, `glow`, `white_balance`.

#### `starfield` (cost 1)

stars twinkling slowly in the dark parts of the picture (space and stars, night, lullabies, dreams).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.4 | how many stars |
| `twinkle` | 0 to 1 | 0.5 | share | 0.4 | how much they twinkle (slowly) |
| `size` | 0.5 to 3 | 1 | share | 0.6 | their size |

Moved by itself: `twinkle` by `high` (0.8), `size` by `bass` (0.5).

Goes well with: `aurora`, `glow`, `night_vision`.

#### `lens_flare` (cost 1)

a lens flare: a light's glow and streak, its ghosts strung across the frame (summer, road trips, the 80s, stadium rock).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how strong the flare |
| `x` | 0 to 1 | 0.75 | share | 0.1 | the light's place across |
| `y` | 0 to 1 | 0.25 | share | 0.1 | the light's place down |
| `hue` | 0 to 1 | 0.08 | turns | 0.1 | its colour |

Moved by itself: `amount` by `downbeat` (0.7).

Goes well with: `light_leak`, `god_rays`, `glow`.

#### `tilt_shift` (cost 2.5)

a tilted lens: only a band in focus, above and below it blurred, the world looking like a model (city pop, childhood, toy-like whimsy).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `focus` | 0 to 1 | 0.55 | share | 0.15 | where the sharp band is, down the frame |
| `band` | 0.05 to 0.6 | 0.2 | share | 0.1 | how tall the sharp band |
| `blur` | 0 to 1 | 0.6 | share | 0.4 | how blurred the rest |

Moved by itself: `blur` by `bass` (0.6), `focus` by `vocals` (0.4).

Goes well with: `saturation`, `curve`.

#### `confetti` (cost 1.3)

confetti fluttering down in every colour, thicker in the choruses (party, celebrations, disco, Schlager, weddings).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how much confetti |
| `size` | 0.5 to 3 | 1 | share | 0.4 | the pieces' size |
| `wind` | -1 to 1 | 0.2 | share | 0.4 | blowing left (-) or right (+) |

Moved by itself: `amount` by `chorus` (0.7), `wind` by `bass` (0.5).

Goes well with: `glow`, `sparks`, `saturation`.

### psychedelic

#### `hue` (cost 1)

the colours turned round the wheel: shifted, cycling slowly, or a rainbow across the screen, the light kept (psychedelia, a new hue for each section).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `shift` | 0 to 1 | 0 | turns | 1 | how far round the wheel |
| `spread` | 0 to 1 | 0 | share | 0.4 | a rainbow across the screen |
| `speed` | -1 to 1 | 0 | share | 0.3 | cycling backwards (-) or forwards (+), a turn in ten seconds at the most |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the turned colour |

Moved by itself: `shift` by `part` (1), `spread` by `bass` (0.5).

Goes well with: `kaleidoscope`, `liquid`, `saturation`.

#### `kaleidoscope` (cost 1)

a kaleidoscope: the picture folded into mirrored segments round the middle, turning (psychedelia, trance, instrumental breaks).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `segments` | 2 to 16 | 6 | steps | 0 | how many mirrored segments |
| `turn` | 0 to 1 | 0 | turns | 1 | how far turned |
| `zoom` | 0.5 to 2 | 1 | share | 0.4 | zoomed out (low) or in (high) |

Moved by itself: `turn` by `bar_phase` (0.25), `zoom` by `bass` (0.6).

Goes well with: `hue`, `feedback`, `liquid`.

#### `mirror` (cost 1)

the picture mirrored left to right, top to bottom or four ways (symmetry, Rorschach, dance, the surreal).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `mode` | 0 to 2 | 0 | steps | 0 | left onto right (0), top onto bottom (1), four ways (2) |
| `place` | 0.2 to 0.8 | 0.5 | share | 0.2 | where the mirror stands |

Moved by itself: `place` by `other` (0.6).

Goes well with: `kaleidoscope`, `hue`, `feedback`.

#### `liquid` (cost 1.2)

the picture flowing and warping like oil on water (psychedelia, drugs and dreams, the surreal, shoegaze).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how far it flows |
| `scale` | 0.5 to 4 | 1.5 | share | 0.5 | the swirls' size: big (low) to small (high) |
| `speed` | 0 to 2 | 0.5 | share | 0.6 | how fast it flows |

Moved by itself: `amount` by `bass` (0.8), `speed` by `energy` (0.5).

Goes well with: `hue`, `chromatic`, `feedback`.

#### `zoom_blur` (cost 3)

a zoom's blur rushing out from a point, as if the camera lunged (drops, drum and bass, rock's big hits).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.3 | share | 0.6 | how far it rushes |
| `x` | 0 to 1 | 0.5 | share | 0 | the point's place across |
| `y` | 0 to 1 | 0.45 | share | 0 | the point's place down |

Moved by itself: `amount` by `kick` (0.9).

Goes well with: `chromatic`, `glitch`, `feedback`.

### visualizer

#### `feedback` (cost 2)

the frame before fed back under this one: zoomed, turned, swirled and flowing, fading by the second, the new picture over it (trails and echoes; with little of the scene, a full visualizer).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `trail` | 0 to 1 | 0.6 | share | 0.3 | how long the echoes last: a moment (0) to many seconds (1) |
| `zoom` | -1 to 1 | 0.2 | share | 0.6 | the echoes shrinking away (-) or growing out (+) |
| `turn` | -1 to 1 | 0 | share | 0.6 | the echoes turning one way (-) or the other (+) |
| `swirl` | 0 to 1 | 0 | share | 0.6 | the echoes swirling, more in the middle |
| `flow` | 0 to 1 | 0.2 | share | 0.5 | the echoes flowing in a noise's currents |
| `scene` | 0 to 1 | 0.85 | share | 0.4 | how much of the new picture shows over the echoes |
| `hue` | -1 to 1 | 0 | share | 0.4 | the echoes' colour turning as they fade |

Moved by itself: `zoom` by `bass` (0.6), `turn` by `snare` (0.6), `swirl` by `chorus` (0.5), `scene` by `instrumental` (-0.6).

Goes well with: `kaleidoscope`, `scope`, `spectrum_ring`, `hue`.

#### `scope` (cost 1)

an oscilloscope's glowing line across the screen, the stems' last two seconds as a wave (electronic, synth-pop, instrumental breaks).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `height` | 0 to 1 | 0.4 | share | 0.4 | how tall the wave |
| `thick` | 1 to 8 | 2.5 | px | 2 | the line's width |
| `hue` | 0 to 1 | 0.5 | turns | 0.2 | its colour |
| `glow` | 0 to 1 | 0.5 | share | 0.4 | how much it glows |
| `y` | 0 to 1 | 0.35 | share | 0 | where it is, down the screen |
| `cover` | 0 to 1 | 0 | share | 0 | the hue (0) or the cover's colour (1) |

Moved by itself: `glow` by `level` (0.5).

Goes well with: `feedback`, `crt`, `glow`.

#### `spectrum_ring` (cost 1)

a ring of the stems' spectrum, its bars standing out from a circle round the middle, mirrored (EDM, house, trance, hip-hop visualizers).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `radius` | 0.1 to 0.45 | 0.25 | share | 0.08 | the ring's size |
| `length` | 0 to 1 | 0.5 | share | 0.4 | how far the bars reach |
| `hue` | 0 to 1 | 0.6 | turns | 0.2 | its colour |
| `spread` | 0 to 1 | 0.5 | share | 0.2 | one colour (0) to a rainbow round the ring (1) |
| `glow` | 0 to 1 | 0.4 | share | 0.4 | how much the circle glows |

Moved by itself: `radius` by `kick` (0.4), `hue` by `part` (0.5).

Goes well with: `feedback`, `beat_rings`, `kaleidoscope`.

#### `bars` (cost 1)

a spectrum's bars hanging from the top of the screen, clear of the lyrics (pop, disco, club, chiptune).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `count` | 8 to 64 | 32 | steps | 0 | how many bars |
| `height` | 0 to 1 | 0.4 | share | 0.4 | how far down they reach |
| `hue` | 0 to 1 | 0.55 | turns | 0.2 | their colour |
| `spread` | 0 to 1 | 0.5 | share | 0.2 | one colour (0) to a rainbow across (1) |

Moved by itself: `hue` by `part` (0.5).

Goes well with: `pixelate`, `crt`, `feedback`.

#### `beat_rings` (cost 1)

rings sent out from the middle on every beat, the bar's first brighter (dance, house, techno, disco).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.6 | share | 0.4 | how bright the rings |
| `speed` | 0.5 to 2 | 1 | share | 0.4 | how fast they spread |
| `hue` | 0 to 1 | 0.1 | turns | 0.2 | their colour |
| `width` | 1 to 10 | 3 | px | 3 | how wide a ring |

Moved by itself: `amount` by `energy` (0.5).

Goes well with: `spectrum_ring`, `feedback`, `glow`.

#### `sparks` (cost 2)

a burst of sparks flying from somewhere above the lyrics on each kick, snare and bass note, falling as they fade (rock, party, fireworks, celebrations).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.6 | share | 0.4 | how bright the sparks |
| `size` | 0.5 to 3 | 1 | share | 0.5 | their size |
| `hue` | 0 to 1 | 0.08 | turns | 0.2 | their colour |
| `spread` | 0 to 1 | 0.5 | share | 0.4 | how far they fly |

Moved by itself: `amount` by `energy` (0.4).

Goes well with: `glow`, `feedback`, `beat_rings`.

#### `lissajous` (cost 2.5)

a Lissajous figure turning in the middle, its loops from the key, its width the bass and height the voice (ambient, electronic, space, jazz).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `size` | 0.1 to 0.45 | 0.3 | share | 0.08 | how big |
| `thick` | 1 to 6 | 2 | px | 1.5 | the line's width |
| `hue` | 0 to 1 | 0.45 | turns | 0.2 | its colour |
| `glow` | 0 to 1 | 0.5 | share | 0.4 | how much it glows |

Moved by itself: `glow` by `vocals` (0.5).

Goes well with: `feedback`, `scope`, `starfield`.

## Drivers

| driver | speed | what it is, what it is good for |
|---|---|---|
| `beat` | fast | a pulse on every beat, peaking on it: pumping, ticking |
| `downbeat` | fast | a pulse on each bar's first beat: flares, light leaks, accents |
| `beat_phase` | fast | where it is in the beat, 0 to 1 (a ramp): spinning and scrolling on the beat |
| `bar_phase` | fast | where it is in the bar, 0 to 1 (a ramp): slower cycles locked to the bar |
| `tempo` | slow | the tempo, 0.5 at 60 to 1 at 120 to 1.5 at 180 a minute: faster songs move faster |
| `level` | fast | how loud the whole music is now: the general energy |
| `drums` | fast | the drums' level: busy textures, shaking |
| `bass` | fast | the bass's level: breathing grain and halation, pushing zooms |
| `vocals` | fast | the voice's level: glow and bloom that follow the singer |
| `other` | fast | the other instruments' level: colour and wash |
| `kick` | fast | a pulse on each kick-like drum hit (a drum hit on beats 1 and 3): thumps, dot sizes |
| `snare` | fast | a pulse on each snare-like drum hit (on beats 2 and 4): glitches, cuts, turns |
| `bass_note` | fast | a pulse on each bass note's start: wobbles, steps |
| `phrase` | fast | a pulse as each sung line starts: a sweep or a bloom for the line |
| `low` | fast | the low end (bass and kicks): weight, size |
| `mid` | fast | the middle (voice and instruments): colour, body |
| `high` | fast | the top (the drums' bright hits): sparkle, grain, glints |
| `minor` | slow | 1 in a minor key, 0 in a major: cooler or darker grades |
| `key_hue` | slow | the key's place on the circle of fifths, 0 to 1: a hue per key |
| `section` | fast | a swell as a section starts, fading over seconds: big changes |
| `energy` | slow | how loud the section is against the song, 0 to 1: verses low, choruses high |
| `chorus` | slow | 1 while the loudest sections (the choruses) play, easing in and out: more of everything |
| `part` | slow | a new value for each section (steps round 0 to 1), eased: a new hue or place each section |
| `singing` | slow | 1 while a line is sung, easing: the look calms under the words |
| `instrumental` | slow | 1 in a break without singing for a while: more visualizer |
| `quiet` | slow | 1 in the quiet parts, slowly: tighter vignettes, darker |
| `pitch` | fast | the pitch the singer sings now, C2 to C6 as 0 to 1 (live) |
| `melody` | fast | the song's melody's pitch, as the pitch |
| `accuracy` | slow | how well the singer hits the notes lately, 0 to 1 (live) |
| `tier` | slow | the game's tier as a glow, 0 to 1 (live) |
| `streak` | slow | the multiplier's rung, 0 to 1 (live) |
| `progress` | slow | how far the song has gone, 0 to 1: a look that ages with the song |

## Presets

### rock

Rock and metal: guitars, grit and stadium light.

- `rock` Rock: `curve` → `halation` → `grain` → `chromatic` (rock)
- `rock-70s` Rock, 70s film stock: `curve` → `halation` → `grain` → `chromatic` → `split_tone` (rock, 70s)
- `rock-rebellion` Rock, rebellion and protest: `curve` → `halation` → `grain` → `chromatic` (rock, rebellion)
- `rock-freedom` Rock, freedom and road trips: `curve` → `halation` → `grain` → `chromatic` → `lens_flare` → `light_leak` (rock, freedom)
- `rock-heartbreak` Rock, heartbreak: `curve` → `halation` → `grain` → `chromatic` → `white_balance` → `vignette` (rock, heartbreak)
- `rock-70s-rebellion` Rock, 70s film stock, rebellion and protest: `curve` → `halation` → `grain` → `chromatic` → `split_tone` (rock, 70s, rebellion)
- `punk` Punk: `xerox` → `duotone` → `glitch` (punk)
- `punk-80s` Punk, 80s neon and VHS: `xerox` → `duotone` → `glitch` → `vhs` → `crt` (punk, 80s)
- `punk-rebellion` Punk, rebellion and protest: `xerox` → `duotone` → `glitch` → `curve` → `grain` (punk, rebellion)
- `punk-party` Punk, party: `xerox` → `duotone` → `glitch` → `confetti` → `saturation` (punk, party)
- `punk-humour` Punk, humour: `xerox` → `duotone` → `glitch` → `posterize_ink` (punk, humour)
- `punk-80s-rebellion` Punk, 80s neon and VHS, rebellion and protest: `xerox` → `duotone` → `glitch` → `curve` → `vhs` (punk, 80s, rebellion)
- `grunge` Grunge: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` (grunge)
- `grunge-90s` Grunge, 90s camcorder and MTV: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` → `vhs` → `cross_process` (grunge, 90s)
- `grunge-grief` Grunge, grief and death: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` → `saturation` → `fog` (grunge, grief)
- `grunge-rebellion` Grunge, rebellion and protest: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` → `curve` (grunge, rebellion)
- `grunge-longing` Grunge, longing: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` → `haze` → `diffusion` (grunge, longing)
- `grunge-90s-grief` Grunge, 90s camcorder and MTV, grief and death: `bleach_bypass` → `grain` → `film_damage` → `gate_weave` → `saturation` → `vhs` (grunge, 90s, grief)
- `doom-metal` Doom metal: `woodcut` → `duotone` → `liquid` (doom-metal)
- `doom-metal-1920s` Doom metal, 1920s sepia and silent film: `woodcut` → `duotone` → `liquid` → `sepia` → `film_damage` → `gate_weave` (doom-metal, 1920s)
- `doom-metal-grief` Doom metal, grief and death: `woodcut` → `duotone` → `liquid` → `saturation` → `fog` → `vignette` (doom-metal, grief)
- `doom-metal-horror` Doom metal, horror: `woodcut` → `duotone` → `liquid` → `fog` → `vignette` → `negative` (doom-metal, horror)
- `doom-metal-war` Doom metal, war: `woodcut` → `duotone` → `liquid` → `bleach_bypass` → `grain` → `film_damage` (doom-metal, war)
- `doom-metal-1920s-grief` Doom metal, 1920s sepia and silent film, grief and death: `woodcut` → `duotone` → `liquid` → `saturation` → `sepia` → `film_damage` (doom-metal, 1920s, grief)
- `black-metal` Black metal: `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal)
- `black-metal-1920s` Black metal, 1920s sepia and silent film: `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal, 1920s)
- `black-metal-winter` Black metal, winter: `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal, winter)
- `black-metal-horror` Black metal, horror: `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal, horror)
- `black-metal-nature` Black metal, nature (seasons, weather, sea, mountains): `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal, nature)
- `black-metal-1920s-winter` Black metal, 1920s sepia and silent film, winter: `saturation` → `curve` → `grain` → `fog` → `zoom_blur` (black-metal, 1920s, winter)
- `power-metal` Power metal: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal)
- `power-metal-today` Power metal, today's clean digital: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal, today)
- `power-metal-empowerment` Power metal, empowerment: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal, empowerment)
- `power-metal-war` Power metal, war: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal, war)
- `power-metal-freedom` Power metal, freedom and road trips: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal, freedom)
- `power-metal-today-empowerment` Power metal, today's clean digital, empowerment: `emboss` → `god_rays` → `saturation` → `chromatic` (power-metal, today, empowerment)
- `shoegaze` Indie and shoegaze: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze)
- `shoegaze-90s` Indie and shoegaze, 90s camcorder and MTV: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze, 90s)
- `shoegaze-dreams` Indie and shoegaze, dreams and the surreal: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze, dreams)
- `shoegaze-longing` Indie and shoegaze, longing: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze, longing)
- `shoegaze-love` Indie and shoegaze, love: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze, love)
- `shoegaze-90s-dreams` Indie and shoegaze, 90s camcorder and MTV, dreams and the surreal: `diffusion` → `glow` → `cross_process` → `liquid` (shoegaze, 90s, dreams)

### pop

Pop in its many colours: bright, glossy, danceable.

- `pop` Pop: `saturation` → `glow` → `curve` → `chromatic` (pop)
- `pop-today` Pop, today's clean digital: `saturation` → `glow` → `curve` → `chromatic` (pop, today)
- `pop-love` Pop, love: `saturation` → `glow` → `curve` → `chromatic` → `bokeh` (pop, love)
- `pop-party` Pop, party: `saturation` → `glow` → `curve` → `chromatic` → `confetti` (pop, party)
- `pop-empowerment` Pop, empowerment: `saturation` → `glow` → `curve` → `chromatic` (pop, empowerment)
- `pop-today-love` Pop, today's clean digital, love: `saturation` → `glow` → `curve` → `chromatic` (pop, today, love)
- `synth-pop` Synth-pop: `tritone` → `neon_edges` → `scope` (synth-pop)
- `synth-pop-80s` Synth-pop, 80s neon and VHS: `tritone` → `neon_edges` → `scope` → `vhs` → `crt` (synth-pop, 80s)
- `synth-pop-night` Synth-pop, night and city: `tritone` → `neon_edges` → `scope` → `white_balance` → `bokeh` (synth-pop, night)
- `synth-pop-love` Synth-pop, love: `tritone` → `neon_edges` → `scope` → `glow` (synth-pop, love)
- `synth-pop-space` Synth-pop, space and stars: `tritone` → `neon_edges` → `scope` → `starfield` → `aurora` (synth-pop, space)
- `synth-pop-80s-night` Synth-pop, 80s neon and VHS, night and city: `tritone` → `neon_edges` → `scope` → `white_balance` → `vhs` (synth-pop, 80s, night)
- `disco` Disco: `tritone` → `bokeh` → `glow` → `beat_rings` (disco)
- `disco-70s` Disco, 70s film stock: `tritone` → `bokeh` → `glow` → `beat_rings` → `curve` (disco, 70s)
- `disco-party` Disco, party: `tritone` → `bokeh` → `glow` → `beat_rings` → `confetti` (disco, party)
- `disco-love` Disco, love: `tritone` → `bokeh` → `glow` → `beat_rings` (disco, love)
- `disco-night` Disco, night and city: `tritone` → `bokeh` → `glow` → `beat_rings` → `white_balance` (disco, night)
- `disco-70s-party` Disco, 70s film stock, party: `tritone` → `bokeh` → `glow` → `beat_rings` → `confetti` (disco, 70s, party)
- `funk` Funk: `posterize_ink` → `grain` → `chromatic` (funk)
- `funk-70s` Funk, 70s film stock: `posterize_ink` → `grain` → `chromatic` → `curve` → `split_tone` (funk, 70s)
- `funk-party` Funk, party: `posterize_ink` → `grain` → `chromatic` → `confetti` → `saturation` (funk, party)
- `funk-money` Funk, money and hustle: `posterize_ink` → `grain` → `chromatic` → `white_balance` → `glow` (funk, money)
- `funk-humour` Funk, humour: `posterize_ink` → `grain` → `chromatic` (funk, humour)
- `funk-70s-party` Funk, 70s film stock, party: `posterize_ink` → `grain` → `chromatic` → `confetti` → `curve` → `split_tone` (funk, 70s, party)
- `soul-rnb` Soul and R&B: `split_tone` → `diffusion` → `bokeh` → `liquid` (soul-rnb)
- `soul-rnb-70s` Soul and R&B, 70s film stock: `split_tone` → `diffusion` → `bokeh` → `liquid` → `curve` → `grain` (soul-rnb, 70s)
- `soul-rnb-love` Soul and R&B, love: `split_tone` → `diffusion` → `bokeh` → `liquid` (soul-rnb, love)
- `soul-rnb-heartbreak` Soul and R&B, heartbreak: `split_tone` → `diffusion` → `bokeh` → `liquid` → `white_balance` → `vignette` (soul-rnb, heartbreak)
- `soul-rnb-night` Soul and R&B, night and city: `split_tone` → `diffusion` → `bokeh` → `liquid` → `white_balance` (soul-rnb, night)
- `soul-rnb-70s-love` Soul and R&B, 70s film stock, love: `split_tone` → `diffusion` → `bokeh` → `liquid` → `curve` → `grain` (soul-rnb, 70s, love)
- `k-pop` K-pop: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop)
- `k-pop-today` K-pop, today's clean digital: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop, today)
- `k-pop-love` K-pop, love: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop, love)
- `k-pop-party` K-pop, party: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop, party)
- `k-pop-empowerment` K-pop, empowerment: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop, empowerment)
- `k-pop-today-love` K-pop, today's clean digital, love: `saturation` → `glow` → `chromatic` → `zoom_blur` (k-pop, today, love)
- `city-pop` J-pop and city pop: `tritone` → `light_leak` → `vhs` → `gate_weave` (city-pop)
- `city-pop-80s` J-pop and city pop, 80s neon and VHS: `tritone` → `light_leak` → `vhs` → `gate_weave` → `glow` (city-pop, 80s)
- `city-pop-night` J-pop and city pop, night and city: `tritone` → `light_leak` → `vhs` → `gate_weave` → `white_balance` → `bokeh` (city-pop, night)
- `city-pop-summer` J-pop and city pop, summer: `tritone` → `light_leak` → `vhs` → `gate_weave` → `white_balance` → `heat_haze` (city-pop, summer)
- `city-pop-longing` J-pop and city pop, longing: `tritone` → `light_leak` → `vhs` → `gate_weave` → `haze` (city-pop, longing)
- `city-pop-80s-night` J-pop and city pop, 80s neon and VHS, night and city: `tritone` → `light_leak` → `vhs` → `gate_weave` → `white_balance` → `crt` (city-pop, 80s, night)
- `musicals` Musicals: `technicolor` → `saturation` → `glow` → `confetti` (musicals)
- `musicals-1950s` Musicals, 1950s Technicolor: `technicolor` → `saturation` → `glow` → `confetti` → `grain` (musicals, 1950s)
- `musicals-love` Musicals, love: `technicolor` → `saturation` → `glow` → `confetti` → `bokeh` (musicals, love)
- `musicals-dreams` Musicals, dreams and the surreal: `technicolor` → `saturation` → `glow` → `confetti` → `liquid` (musicals, dreams)
- `musicals-humour` Musicals, humour: `technicolor` → `saturation` → `glow` → `confetti` → `posterize_ink` (musicals, humour)
- `musicals-1950s-love` Musicals, 1950s Technicolor, love: `technicolor` → `saturation` → `glow` → `confetti` → `grain` (musicals, 1950s, love)
- `schlager` Schlager and Volksmusik: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` (schlager)
- `schlager-70s` Schlager and Volksmusik, 70s film stock: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` → `curve` (schlager, 70s)
- `schlager-love` Schlager and Volksmusik, love: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` (schlager, love)
- `schlager-hometown` Schlager and Volksmusik, hometown: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` → `white_balance` (schlager, hometown)
- `schlager-party` Schlager and Volksmusik, party: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` (schlager, party)
- `schlager-70s-love` Schlager and Volksmusik, 70s film stock, love: `lift_gamma_gain` → `saturation` → `diffusion` → `polaroid` → `petals` → `curve` (schlager, 70s, love)

### electronic

Electronic music: the club, the drop, the machine.

- `synthwave` Synthwave: `tritone` → `neon_edges` → `bars` → `crt` (synth-pop, 80s neon and VHS, night and city, space and stars)
- `house` House: `beat_rings` → `glow` → `hue` → `chromatic` (house)
- `house-today` House, today's clean digital: `beat_rings` → `glow` → `hue` → `chromatic` → `curve` → `saturation` (house, today)
- `house-party` House, party: `beat_rings` → `glow` → `hue` → `chromatic` → `confetti` (house, party)
- `house-night` House, night and city: `beat_rings` → `glow` → `hue` → `chromatic` → `white_balance` (house, night)
- `house-freedom` House, freedom and road trips: `beat_rings` → `glow` → `hue` → `chromatic` → `lens_flare` → `light_leak` (house, freedom)
- `house-today-party` House, today's clean digital, party: `beat_rings` → `glow` → `hue` → `chromatic` → `confetti` (house, today, party)
- `techno` Techno: `tritone` → `scope` → `glitch` → `zoom_blur` (techno)
- `techno-today` Techno, today's clean digital: `tritone` → `scope` → `glitch` → `zoom_blur` → `curve` (techno, today)
- `techno-night` Techno, night and city: `tritone` → `scope` → `glitch` → `zoom_blur` → `white_balance` (techno, night)
- `techno-rebellion` Techno, rebellion and protest: `tritone` → `scope` → `glitch` → `zoom_blur` → `curve` (techno, rebellion)
- `techno-drugs` Techno, drugs and psychedelia: `tritone` → `scope` → `glitch` → `zoom_blur` → `hue` (techno, drugs)
- `techno-today-night` Techno, today's clean digital, night and city: `tritone` → `scope` → `glitch` → `zoom_blur` → `white_balance` (techno, today, night)
- `trance` Trance: `glow` → `kaleidoscope` → `hue` → `liquid` (trance)
- `trance-y2k` Trance, Y2K chrome and glitch: `glow` → `kaleidoscope` → `hue` → `liquid` → `chromatic` (trance, y2k)
- `trance-space` Trance, space and stars: `glow` → `kaleidoscope` → `hue` → `liquid` → `starfield` (trance, space)
- `trance-dreams` Trance, dreams and the surreal: `glow` → `kaleidoscope` → `hue` → `liquid` (trance, dreams)
- `trance-freedom` Trance, freedom and road trips: `glow` → `kaleidoscope` → `hue` → `liquid` → `lens_flare` (trance, freedom)
- `trance-y2k-space` Trance, Y2K chrome and glitch, space and stars: `glow` → `kaleidoscope` → `hue` → `liquid` → `starfield` (trance, y2k, space)
- `dubstep` Dubstep: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` (dubstep)
- `dubstep-y2k` Dubstep, Y2K chrome and glitch: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` → `solarize` (dubstep, y2k)
- `dubstep-rebellion` Dubstep, rebellion and protest: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` → `curve` (dubstep, rebellion)
- `dubstep-party` Dubstep, party: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` → `confetti` (dubstep, party)
- `dubstep-night` Dubstep, night and city: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` → `white_balance` (dubstep, night)
- `dubstep-y2k-rebellion` Dubstep, Y2K chrome and glitch, rebellion and protest: `glitch` → `chromatic` → `zoom_blur` → `gate_weave` → `curve` (dubstep, y2k, rebellion)
- `drum-and-bass` Drum and bass: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` (drum-and-bass)
- `drum-and-bass-90s` Drum and bass, 90s camcorder and MTV: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` → `vhs` (drum-and-bass, 90s)
- `drum-and-bass-night` Drum and bass, night and city: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` → `white_balance` (drum-and-bass, night)
- `drum-and-bass-freedom` Drum and bass, freedom and road trips: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` → `lens_flare` → `light_leak` (drum-and-bass, freedom)
- `drum-and-bass-space` Drum and bass, space and stars: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` → `starfield` (drum-and-bass, space)
- `drum-and-bass-90s-night` Drum and bass, 90s camcorder and MTV, night and city: `zoom_blur` → `chromatic` → `tritone` → `gate_weave` → `white_balance` → `cross_process` (drum-and-bass, 90s, night)
- `ambient` Ambient: `diffusion` → `haze` → `aurora` → `liquid` (ambient)
- `ambient-today` Ambient, today's clean digital: `diffusion` → `haze` → `aurora` → `liquid` → `curve` → `saturation` (ambient, today)
- `ambient-nature` Ambient, nature (seasons, weather, sea, mountains): `diffusion` → `haze` → `aurora` → `liquid` → `dust` (ambient, nature)
- `ambient-space` Ambient, space and stars: `diffusion` → `haze` → `aurora` → `liquid` → `starfield` (ambient, space)
- `ambient-dreams` Ambient, dreams and the surreal: `diffusion` → `haze` → `aurora` → `liquid` → `hue` (ambient, dreams)
- `ambient-today-nature` Ambient, today's clean digital, nature (seasons, weather, sea, mountains): `diffusion` → `haze` → `aurora` → `liquid` → `curve` → `saturation` (ambient, today, nature)
- `chiptune` Chiptune and video-game music: `palette` → `crt` → `bars` → `gate_weave` (chiptune)
- `chiptune-80s` Chiptune and video-game music, 80s neon and VHS: `palette` → `crt` → `bars` → `gate_weave` → `vhs` (chiptune, 80s)
- `chiptune-humour` Chiptune and video-game music, humour: `palette` → `crt` → `bars` → `gate_weave` → `posterize_ink` (chiptune, humour)
- `chiptune-party` Chiptune and video-game music, party: `palette` → `crt` → `bars` → `gate_weave` → `confetti` → `saturation` (chiptune, party)
- `chiptune-nostalgia` Chiptune and video-game music, nostalgia and childhood: `palette` → `crt` → `bars` → `gate_weave` → `polaroid` → `light_leak` (chiptune, nostalgia)
- `chiptune-80s-humour` Chiptune and video-game music, 80s neon and VHS, humour: `palette` → `crt` → `bars` → `gate_weave` → `posterize_ink` (chiptune, 80s, humour)

### urban

Hip-hop, dub and the street: rhythm and print.

- `boom-bap` Hip-hop, boom bap: `halftone` → `curve` → `grain` → `gate_weave` (boom-bap)
- `boom-bap-90s` Hip-hop, boom bap, 90s camcorder and MTV: `halftone` → `curve` → `grain` → `gate_weave` → `vhs` → `cross_process` (boom-bap, 90s)
- `boom-bap-hometown` Hip-hop, boom bap, hometown: `halftone` → `curve` → `grain` → `gate_weave` → `white_balance` → `polaroid` (boom-bap, hometown)
- `boom-bap-money` Hip-hop, boom bap, money and hustle: `halftone` → `curve` → `grain` → `gate_weave` → `white_balance` (boom-bap, money)
- `boom-bap-rebellion` Hip-hop, boom bap, rebellion and protest: `halftone` → `curve` → `grain` → `gate_weave` (boom-bap, rebellion)
- `boom-bap-90s-hometown` Hip-hop, boom bap, 90s camcorder and MTV, hometown: `halftone` → `curve` → `grain` → `gate_weave` → `white_balance` → `vhs` (boom-bap, 90s, hometown)
- `trap` Hip-hop, trap: `tritone` → `chromatic` → `glitch` → `zoom_blur` (trap)
- `trap-today` Hip-hop, trap, today's clean digital: `tritone` → `chromatic` → `glitch` → `zoom_blur` → `curve` (trap, today)
- `trap-money` Hip-hop, trap, money and hustle: `tritone` → `chromatic` → `glitch` → `zoom_blur` → `white_balance` (trap, money)
- `trap-night` Hip-hop, trap, night and city: `tritone` → `chromatic` → `glitch` → `zoom_blur` → `white_balance` (trap, night)
- `trap-drugs` Hip-hop, trap, drugs and psychedelia: `tritone` → `chromatic` → `glitch` → `zoom_blur` → `hue` (trap, drugs)
- `trap-today-money` Hip-hop, trap, today's clean digital, money and hustle: `tritone` → `chromatic` → `glitch` → `zoom_blur` → `white_balance` (trap, today, money)
- `lo-fi` Hip-hop, lo-fi: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` (lo-fi)
- `lo-fi-90s` Hip-hop, lo-fi, 90s camcorder and MTV: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` → `vhs` (lo-fi, 90s)
- `lo-fi-nostalgia` Hip-hop, lo-fi, nostalgia and childhood: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` → `polaroid` (lo-fi, nostalgia)
- `lo-fi-night` Hip-hop, lo-fi, night and city: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` → `white_balance` (lo-fi, night)
- `lo-fi-longing` Hip-hop, lo-fi, longing: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` → `haze` (lo-fi, longing)
- `lo-fi-90s-nostalgia` Hip-hop, lo-fi, 90s camcorder and MTV, nostalgia and childhood: `cross_process` → `grain` → `vignette` → `canvas` → `gate_weave` → `polaroid` (lo-fi, 90s, nostalgia)
- `reggae-dub` Reggae and dub: `feedback` → `fog` → `white_balance` → `liquid` (reggae-dub)
- `reggae-dub-70s` Reggae and dub, 70s film stock: `feedback` → `fog` → `white_balance` → `liquid` → `curve` → `split_tone` (reggae-dub, 70s)
- `reggae-dub-freedom` Reggae and dub, freedom and road trips: `feedback` → `fog` → `white_balance` → `liquid` → `lens_flare` → `light_leak` (reggae-dub, freedom)
- `reggae-dub-drugs` Reggae and dub, drugs and psychedelia: `feedback` → `fog` → `white_balance` → `liquid` → `hue` → `kaleidoscope` (reggae-dub, drugs)
- `reggae-dub-summer` Reggae and dub, summer: `feedback` → `fog` → `white_balance` → `liquid` → `light_leak` → `heat_haze` (reggae-dub, summer)
- `reggae-dub-70s-freedom` Reggae and dub, 70s film stock, freedom and road trips: `feedback` → `fog` → `white_balance` → `liquid` → `lens_flare` → `curve` (reggae-dub, 70s, freedom)
- `afrobeat` Afrobeat: `risograph` → `saturation` → `grain` → `chromatic` (afrobeat)
- `afrobeat-70s` Afrobeat, 70s film stock: `risograph` → `saturation` → `grain` → `chromatic` → `curve` → `split_tone` (afrobeat, 70s)
- `afrobeat-party` Afrobeat, party: `risograph` → `saturation` → `grain` → `chromatic` → `confetti` (afrobeat, party)
- `afrobeat-empowerment` Afrobeat, empowerment: `risograph` → `saturation` → `grain` → `chromatic` → `glow` (afrobeat, empowerment)
- `afrobeat-summer` Afrobeat, summer: `risograph` → `saturation` → `grain` → `chromatic` → `white_balance` → `light_leak` (afrobeat, summer)
- `afrobeat-70s-party` Afrobeat, 70s film stock, party: `risograph` → `saturation` → `grain` → `chromatic` → `confetti` → `curve` (afrobeat, 70s, party)
- `reggaeton` Reggaeton: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` (reggaeton)
- `reggaeton-today` Reggaeton, today's clean digital: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` → `curve` → `saturation` (reggaeton, today)
- `reggaeton-party` Reggaeton, party: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` → `confetti` (reggaeton, party)
- `reggaeton-love` Reggaeton, love: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` → `bokeh` (reggaeton, love)
- `reggaeton-summer` Reggaeton, summer: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` → `white_balance` → `light_leak` (reggaeton, summer)
- `reggaeton-today-party` Reggaeton, today's clean digital, party: `tritone` → `chromatic` → `beat_rings` → `zoom_blur` → `confetti` (reggaeton, today, party)

### roots

Roots music: jazz, blues, country, folk.

- `jazz` Jazz: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` (jazz)
- `jazz-1950s` Jazz, 1950s Technicolor: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` → `technicolor` (jazz, 1950s)
- `jazz-night` Jazz, night and city: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` → `white_balance` (jazz, night)
- `jazz-love` Jazz, love: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` (jazz, love)
- `jazz-longing` Jazz, longing: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` → `haze` (jazz, longing)
- `jazz-1950s-night` Jazz, 1950s Technicolor, night and city: `saturation` → `split_tone` → `grain` → `bokeh` → `gate_weave` → `white_balance` (jazz, 1950s, night)
- `blues` Blues: `duotone` → `grain` → `vignette` → `rain` (blues)
- `blues-1920s` Blues, 1920s sepia and silent film: `duotone` → `grain` → `vignette` → `rain` → `sepia` → `film_damage` (blues, 1920s)
- `blues-heartbreak` Blues, heartbreak: `duotone` → `grain` → `vignette` → `rain` → `white_balance` (blues, heartbreak)
- `blues-grief` Blues, grief and death: `duotone` → `grain` → `vignette` → `rain` → `saturation` → `fog` (blues, grief)
- `blues-hometown` Blues, hometown: `duotone` → `grain` → `vignette` → `rain` → `white_balance` → `polaroid` (blues, hometown)
- `blues-1920s-heartbreak` Blues, 1920s sepia and silent film, heartbreak: `duotone` → `grain` → `vignette` → `rain` → `white_balance` → `sepia` (blues, 1920s, heartbreak)
- `country` Country and Americana: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` (country)
- `country-70s` Country and Americana, 70s film stock: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` → `split_tone` (country, 70s)
- `country-freedom` Country and Americana, freedom and road trips: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` → `lens_flare` (country, freedom)
- `country-hometown` Country and Americana, hometown: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` → `polaroid` (country, hometown)
- `country-heartbreak` Country and Americana, heartbreak: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` → `rain` (country, heartbreak)
- `country-70s-freedom` Country and Americana, 70s film stock, freedom and road trips: `white_balance` → `curve` → `light_leak` → `grain` → `gate_weave` → `lens_flare` (country, 70s, freedom)
- `folk` Folk: `watercolor` → `canvas` → `liquid` (folk)
- `folk-60s` Folk, 60s psychedelic: `watercolor` → `canvas` → `liquid` → `saturation` → `light_leak` → `grain` (folk, 60s)
- `folk-nature` Folk, nature (seasons, weather, sea, mountains): `watercolor` → `canvas` → `liquid` → `haze` → `dust` (folk, nature)
- `folk-family` Folk, family: `watercolor` → `canvas` → `liquid` → `polaroid` (folk, family)
- `folk-rebellion` Folk, rebellion and protest: `watercolor` → `canvas` → `liquid` → `curve` → `grain` (folk, rebellion)
- `folk-60s-nature` Folk, 60s psychedelic, nature (seasons, weather, sea, mountains): `watercolor` → `canvas` → `liquid` → `haze` → `saturation` → `light_leak` (folk, 60s, nature)
- `celtic` Celtic: `haze` → `god_rays` → `grain` → `fog` (celtic)
- `celtic-1920s` Celtic, 1920s sepia and silent film: `haze` → `god_rays` → `grain` → `fog` (celtic, 1920s)
- `celtic-nature` Celtic, nature (seasons, weather, sea, mountains): `haze` → `god_rays` → `grain` → `fog` (celtic, nature)
- `celtic-mountains` Celtic, mountains: `haze` → `god_rays` → `grain` → `fog` (celtic, mountains)
- `celtic-longing` Celtic, longing: `haze` → `god_rays` → `grain` → `fog` (celtic, longing)
- `celtic-1920s-nature` Celtic, 1920s sepia and silent film, nature (seasons, weather, sea, mountains): `haze` → `god_rays` → `grain` → `fog` (celtic, 1920s, nature)
- `gospel` Gospel: `god_rays` → `glow` → `white_balance` (gospel)
- `gospel-1950s` Gospel, 1950s Technicolor: `god_rays` → `glow` → `white_balance` (gospel, 1950s)
- `gospel-faith` Gospel, faith: `god_rays` → `glow` → `white_balance` (gospel, faith)
- `gospel-empowerment` Gospel, empowerment: `god_rays` → `glow` → `white_balance` (gospel, empowerment)
- `gospel-family` Gospel, family: `god_rays` → `glow` → `white_balance` (gospel, family)
- `gospel-1950s-faith` Gospel, 1950s Technicolor, faith: `god_rays` → `glow` → `white_balance` (gospel, 1950s, faith)
- `chanson` Chanson: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` (chanson)
- `chanson-1950s` Chanson, 1950s Technicolor: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` → `technicolor` (chanson, 1950s)
- `chanson-love` Chanson, love: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` → `glow` (chanson, love)
- `chanson-night` Chanson, night and city: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` → `white_balance` (chanson, night)
- `chanson-longing` Chanson, longing: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` → `haze` (chanson, longing)
- `chanson-1950s-love` Chanson, 1950s Technicolor, love: `saturation` → `split_tone` → `candle` → `grain` → `gate_weave` → `glow` (chanson, 1950s, love)

### classical

Classical, orchestral and the stage.

- `orchestral` Classical and orchestral: `oil_paint` → `canvas` → `vignette` → `liquid` (orchestral)
- `orchestral-1920s` Classical and orchestral, 1920s sepia and silent film: `oil_paint` → `canvas` → `vignette` → `liquid` → `sepia` (orchestral, 1920s)
- `orchestral-nature` Classical and orchestral, nature (seasons, weather, sea, mountains): `oil_paint` → `canvas` → `vignette` → `liquid` → `haze` (orchestral, nature)
- `orchestral-grief` Classical and orchestral, grief and death: `oil_paint` → `canvas` → `vignette` → `liquid` → `saturation` (orchestral, grief)
- `orchestral-faith` Classical and orchestral, faith: `oil_paint` → `canvas` → `vignette` → `liquid` (orchestral, faith)
- `orchestral-1920s-nature` Classical and orchestral, 1920s sepia and silent film, nature (seasons, weather, sea, mountains): `oil_paint` → `canvas` → `vignette` → `liquid` → `haze` (orchestral, 1920s, nature)
- `opera` Opera: `candle` → `glow` → `letterbox` → `dust` (opera)
- `opera-1920s` Opera, 1920s sepia and silent film: `candle` → `glow` → `letterbox` → `dust` → `sepia` → `film_damage` (opera, 1920s)
- `opera-love` Opera, love: `candle` → `glow` → `letterbox` → `dust` → `bokeh` (opera, love)
- `opera-grief` Opera, grief and death: `candle` → `glow` → `letterbox` → `dust` → `saturation` → `vignette` (opera, grief)
- `opera-war` Opera, war: `candle` → `glow` → `letterbox` → `dust` → `bleach_bypass` → `grain` (opera, war)
- `opera-1920s-love` Opera, 1920s sepia and silent film, love: `candle` → `glow` → `letterbox` → `dust` → `sepia` → `film_damage` (opera, 1920s, love)

### world

Music from everywhere: colour, heat and dance.

- `salsa` Salsa: `tritone` → `glow` → `bokeh` → `confetti` (salsa)
- `salsa-70s` Salsa, 70s film stock: `tritone` → `glow` → `bokeh` → `confetti` → `curve` (salsa, 70s)
- `salsa-party` Salsa, party: `tritone` → `glow` → `bokeh` → `confetti` → `saturation` (salsa, party)
- `salsa-love` Salsa, love: `tritone` → `glow` → `bokeh` → `confetti` (salsa, love)
- `salsa-summer` Salsa, summer: `tritone` → `glow` → `bokeh` → `confetti` → `white_balance` (salsa, summer)
- `salsa-70s-party` Salsa, 70s film stock, party: `tritone` → `glow` → `bokeh` → `confetti` → `curve` (salsa, 70s, party)
- `flamenco` Flamenco: `duotone` → `heat_haze` → `vignette` → `chromatic` (flamenco)
- `flamenco-1950s` Flamenco, 1950s Technicolor: `duotone` → `heat_haze` → `vignette` → `chromatic` → `technicolor` → `halation` (flamenco, 1950s)
- `flamenco-love` Flamenco, love: `duotone` → `heat_haze` → `vignette` → `chromatic` → `glow` (flamenco, love)
- `flamenco-heartbreak` Flamenco, heartbreak: `duotone` → `heat_haze` → `vignette` → `chromatic` → `white_balance` → `rain` (flamenco, heartbreak)
- `flamenco-summer` Flamenco, summer: `duotone` → `heat_haze` → `vignette` → `chromatic` → `white_balance` → `light_leak` (flamenco, summer)
- `flamenco-1950s-love` Flamenco, 1950s Technicolor, love: `duotone` → `heat_haze` → `vignette` → `chromatic` → `glow` → `technicolor` (flamenco, 1950s, love)
- `bollywood` Bollywood: `saturation` → `glow` → `bokeh` → `confetti` (bollywood)
- `bollywood-1950s` Bollywood, 1950s Technicolor: `saturation` → `glow` → `bokeh` → `confetti` → `technicolor` (bollywood, 1950s)
- `bollywood-love` Bollywood, love: `saturation` → `glow` → `bokeh` → `confetti` (bollywood, love)
- `bollywood-party` Bollywood, party: `saturation` → `glow` → `bokeh` → `confetti` (bollywood, party)
- `bollywood-family` Bollywood, family: `saturation` → `glow` → `bokeh` → `confetti` → `polaroid` (bollywood, family)
- `bollywood-1950s-love` Bollywood, 1950s Technicolor, love: `saturation` → `glow` → `bokeh` → `confetti` → `technicolor` (bollywood, 1950s, love)

### occasions

Songs for children and for the year's days.

- `childrens` Children's songs: `pastel` → `saturation` → `confetti` (childrens)
- `childrens-today` Children's songs, today's clean digital: `pastel` → `saturation` → `confetti` → `curve` (childrens, today)
- `childrens-humour` Children's songs, humour: `pastel` → `saturation` → `confetti` → `posterize_ink` (childrens, humour)
- `childrens-nature` Children's songs, nature (seasons, weather, sea, mountains): `pastel` → `saturation` → `confetti` → `haze` → `dust` (childrens, nature)
- `childrens-family` Children's songs, family: `pastel` → `saturation` → `confetti` → `polaroid` → `diffusion` (childrens, family)
- `childrens-today-humour` Children's songs, today's clean digital, humour: `pastel` → `saturation` → `confetti` → `posterize_ink` → `curve` (childrens, today, humour)
- `christmas` Christmas songs: `candle` → `bokeh` → `snow` → `gate_weave` (christmas)
- `christmas-1950s` Christmas songs, 1950s Technicolor: `candle` → `bokeh` → `snow` → `gate_weave` → `technicolor` → `grain` (christmas, 1950s)
- `christmas-family` Christmas songs, family: `candle` → `bokeh` → `snow` → `gate_weave` → `polaroid` (christmas, family)
- `christmas-winter` Christmas songs, winter: `candle` → `bokeh` → `snow` → `gate_weave` → `white_balance` (christmas, winter)
- `christmas-faith` Christmas songs, faith: `candle` → `bokeh` → `snow` → `gate_weave` → `glow` (christmas, faith)
- `christmas-1950s-family` Christmas songs, 1950s Technicolor, family: `candle` → `bokeh` → `snow` → `gate_weave` → `polaroid` → `technicolor` (christmas, 1950s, family)
- `halloween` Halloween songs: `duotone` → `fog` → `vignette` → `liquid` (halloween)
- `halloween-1950s` Halloween songs, 1950s Technicolor: `duotone` → `fog` → `vignette` → `liquid` → `technicolor` → `grain` (halloween, 1950s)
- `halloween-horror` Halloween songs, horror: `duotone` → `fog` → `vignette` → `liquid` → `negative` (halloween, horror)
- `halloween-humour` Halloween songs, humour: `duotone` → `fog` → `vignette` → `liquid` → `posterize_ink` (halloween, humour)
- `halloween-party` Halloween songs, party: `duotone` → `fog` → `vignette` → `liquid` → `confetti` → `saturation` (halloween, party)
- `halloween-1950s-horror` Halloween songs, 1950s Technicolor, horror: `duotone` → `fog` → `vignette` → `liquid` → `technicolor` → `grain` (halloween, 1950s, horror)

### eras

The eras: how each decade looked on film, tape and screen.

- `35mm-night-drive` 35mm night drive: `split_tone` → `curve` → `halation` → `grain` → `vignette` (synth-pop, rock, night and city, freedom and road trips, 70s film stock)
- `psychedelic-60s` Psychedelic 60s: `saturation` → `liquid` → `hue` → `kaleidoscope` (rock, 60s psychedelic, drugs and psychedelia, dreams and the surreal)
- `camcorder-90s` 90s camcorder: `cross_process` → `vhs` → `crt` → `gate_weave` (grunge, indie, 90s camcorder and MTV, nostalgia and childhood, family)
- `era-1920s` 1920s sepia and silent film: `sepia` → `film_damage` → `gate_weave` → `vignette` (1920s)
- `era-1920s-love` 1920s sepia and silent film, love: `sepia` → `film_damage` → `gate_weave` → `vignette` → `glow` (1920s, love)
- `era-1920s-grief` 1920s sepia and silent film, grief and death: `sepia` → `film_damage` → `gate_weave` → `vignette` → `saturation` → `fog` (1920s, grief)
- `era-1920s-humour` 1920s sepia and silent film, humour: `sepia` → `film_damage` → `gate_weave` → `vignette` → `posterize_ink` (1920s, humour)
- `era-1920s-night` 1920s sepia and silent film, night and city: `sepia` → `film_damage` → `gate_weave` → `vignette` → `white_balance` → `bokeh` (1920s, night)
- `era-1950s` 1950s Technicolor: `technicolor` → `halation` → `grain` (1950s)
- `era-1950s-love` 1950s Technicolor, love: `technicolor` → `halation` → `grain` → `glow` (1950s, love)
- `era-1950s-party` 1950s Technicolor, party: `technicolor` → `halation` → `grain` → `confetti` → `saturation` (1950s, party)
- `era-1950s-family` 1950s Technicolor, family: `technicolor` → `halation` → `grain` → `polaroid` (1950s, family)
- `era-1950s-space` 1950s Technicolor, space and stars: `technicolor` → `halation` → `grain` → `starfield` → `aurora` (1950s, space)
- `era-60s` 60s psychedelic: `saturation` → `light_leak` → `grain` (60s)
- `era-60s-drugs` 60s psychedelic, drugs and psychedelia: `saturation` → `light_leak` → `grain` → `hue` → `liquid` → `kaleidoscope` (60s, drugs)
- `era-60s-love` 60s psychedelic, love: `saturation` → `light_leak` → `grain` → `glow` → `bokeh` (60s, love)
- `era-60s-rebellion` 60s psychedelic, rebellion and protest: `saturation` → `light_leak` → `grain` → `curve` (60s, rebellion)
- `era-60s-summer` 60s psychedelic, summer: `saturation` → `light_leak` → `grain` → `white_balance` → `heat_haze` (60s, summer)
- `era-70s` 70s film stock: `curve` → `split_tone` → `grain` (70s)
- `era-70s-freedom` 70s film stock, freedom and road trips: `curve` → `split_tone` → `grain` → `lens_flare` → `light_leak` (70s, freedom)
- `era-70s-night` 70s film stock, night and city: `curve` → `split_tone` → `grain` → `white_balance` → `bokeh` (70s, night)
- `era-70s-nostalgia` 70s film stock, nostalgia and childhood: `curve` → `split_tone` → `grain` → `polaroid` → `light_leak` (70s, nostalgia)
- `era-70s-love` 70s film stock, love: `curve` → `split_tone` → `grain` → `glow` → `bokeh` (70s, love)
- `era-80s` 80s neon and VHS: `vhs` → `glow` → `crt` (80s)
- `era-80s-night` 80s neon and VHS, night and city: `vhs` → `glow` → `crt` → `white_balance` (80s, night)
- `era-80s-party` 80s neon and VHS, party: `vhs` → `glow` → `crt` → `saturation` (80s, party)
- `era-80s-space` 80s neon and VHS, space and stars: `vhs` → `glow` → `crt` → `starfield` (80s, space)
- `era-80s-heartbreak` 80s neon and VHS, heartbreak: `vhs` → `glow` → `crt` → `white_balance` (80s, heartbreak)
- `era-90s` 90s camcorder and MTV: `vhs` → `cross_process` (90s)
- `era-90s-nostalgia` 90s camcorder and MTV, nostalgia and childhood: `vhs` → `cross_process` → `polaroid` → `light_leak` (90s, nostalgia)
- `era-90s-rebellion` 90s camcorder and MTV, rebellion and protest: `vhs` → `cross_process` → `curve` → `grain` (90s, rebellion)
- `era-90s-party` 90s camcorder and MTV, party: `vhs` → `cross_process` → `confetti` → `saturation` (90s, party)
- `era-90s-hometown` 90s camcorder and MTV, hometown: `vhs` → `cross_process` → `white_balance` → `polaroid` (90s, hometown)
- `era-y2k` Y2K chrome and glitch: `chromatic` → `glitch` → `solarize` (y2k)
- `era-y2k-party` Y2K chrome and glitch, party: `chromatic` → `glitch` → `solarize` → `confetti` → `saturation` (y2k, party)
- `era-y2k-dreams` Y2K chrome and glitch, dreams and the surreal: `chromatic` → `glitch` → `solarize` → `liquid` → `diffusion` (y2k, dreams)
- `era-y2k-money` Y2K chrome and glitch, money and hustle: `chromatic` → `glitch` → `solarize` → `white_balance` → `glow` (y2k, money)
- `era-y2k-space` Y2K chrome and glitch, space and stars: `chromatic` → `glitch` → `solarize` → `starfield` → `aurora` (y2k, space)
- `era-today` Today's clean digital: `curve` → `saturation` (today)
- `era-today-love` Today's clean digital, love: `curve` → `saturation` → `glow` → `bokeh` (today, love)
- `era-today-empowerment` Today's clean digital, empowerment: `curve` → `saturation` → `god_rays` (today, empowerment)
- `era-today-nature` Today's clean digital, nature (seasons, weather, sea, mountains): `curve` → `saturation` → `haze` → `dust` (today, nature)
- `era-today-winter` Today's clean digital, winter: `curve` → `saturation` → `white_balance` → `snow` (today, winter)

### themes

The lyrics' themes, each in a look of its own, over any genre.

- `theme-love` Love: `glow` → `bokeh` (love)
- `theme-love-film` Love, on film: `glow` → `bokeh` → `grain` → `vignette` (love, 70s)
- `theme-heartbreak` Heartbreak: `white_balance` → `rain` → `vignette` (heartbreak)
- `theme-heartbreak-film` Heartbreak, on film: `white_balance` → `rain` → `vignette` → `grain` (heartbreak, 70s)
- `theme-longing` Longing: `haze` → `diffusion` (longing)
- `theme-longing-film` Longing, on film: `haze` → `diffusion` → `grain` → `vignette` (longing, 70s)
- `theme-party` Party: `confetti` → `saturation` (party)
- `theme-party-film` Party, on film: `confetti` → `saturation` → `grain` → `vignette` (party, 70s)
- `theme-rebellion` Rebellion and protest: `curve` → `grain` (rebellion)
- `theme-rebellion-film` Rebellion and protest, on film: `curve` → `grain` → `vignette` (rebellion, 70s)
- `theme-freedom` Freedom and road trips: `lens_flare` → `light_leak` (freedom)
- `theme-freedom-film` Freedom and road trips, on film: `lens_flare` → `light_leak` → `grain` → `vignette` (freedom, 70s)
- `theme-nostalgia` Nostalgia and childhood: `polaroid` → `light_leak` (nostalgia)
- `theme-nostalgia-film` Nostalgia and childhood, on film: `polaroid` → `light_leak` → `grain` → `vignette` (nostalgia, 70s)
- `theme-family` Family: `polaroid` → `diffusion` (family)
- `theme-family-film` Family, on film: `polaroid` → `diffusion` → `grain` → `vignette` (family, 70s)
- `theme-grief` Grief and death: `saturation` → `fog` → `vignette` (grief)
- `theme-grief-film` Grief and death, on film: `saturation` → `fog` → `vignette` → `grain` (grief, 70s)
- `theme-faith` Faith: `god_rays` → `glow` (faith)
- `theme-faith-film` Faith, on film: `god_rays` → `glow` → `grain` (faith, 70s)
- `theme-nature` Nature (seasons, weather, sea, mountains): `haze` → `dust` (nature)
- `theme-nature-film` Nature (seasons, weather, sea, mountains), on film: `haze` → `dust` → `grain` → `vignette` (nature, 70s)
- `theme-sea` The sea: `underwater` (sea)
- `theme-sea-film` The sea, on film: `underwater` → `grain` → `vignette` (sea, 70s)
- `theme-mountains` Mountains: `haze` (mountains)
- `theme-mountains-film` Mountains, on film: `haze` → `grain` → `vignette` (mountains, 70s)
- `theme-night` Night and city: `white_balance` → `bokeh` → `glow` (night)
- `theme-night-film` Night and city, on film: `white_balance` → `bokeh` → `glow` → `grain` → `vignette` (night, 70s)
- `theme-space` Space and stars: `starfield` → `aurora` (space)
- `theme-space-film` Space and stars, on film: `starfield` → `aurora` → `grain` → `vignette` (space, 70s)
- `theme-dreams` Dreams and the surreal: `liquid` → `diffusion` → `hue` (dreams)
- `theme-dreams-film` Dreams and the surreal, on film: `liquid` → `diffusion` → `hue` → `grain` → `vignette` (dreams, 70s)
- `theme-drugs` Drugs and psychedelia: `hue` → `liquid` → `kaleidoscope` (drugs)
- `theme-drugs-film` Drugs and psychedelia, on film: `hue` → `liquid` → `kaleidoscope` → `grain` → `vignette` (drugs, 70s)
- `theme-money` Money and hustle: `white_balance` → `glow` (money)
- `theme-money-film` Money and hustle, on film: `white_balance` → `glow` → `grain` → `vignette` (money, 70s)
- `theme-war` War: `bleach_bypass` → `grain` → `film_damage` (war)
- `theme-war-film` War, on film: `bleach_bypass` → `grain` → `film_damage` → `vignette` (war, 70s)
- `theme-hometown` Hometown: `white_balance` → `polaroid` (hometown)
- `theme-hometown-film` Hometown, on film: `white_balance` → `polaroid` → `grain` → `vignette` (hometown, 70s)
- `theme-summer` Summer: `white_balance` → `light_leak` → `heat_haze` (summer)
- `theme-summer-film` Summer, on film: `white_balance` → `light_leak` → `heat_haze` → `grain` → `vignette` (summer, 70s)
- `theme-winter` Winter: `white_balance` → `snow` (winter)
- `theme-winter-film` Winter, on film: `white_balance` → `snow` → `grain` → `vignette` (winter, 70s)
- `theme-horror` Horror: `fog` → `vignette` → `negative` (horror)
- `theme-horror-film` Horror, on film: `fog` → `vignette` → `negative` → `grain` (horror, 70s)
- `theme-humour` Humour: `posterize_ink` (humour)
- `theme-humour-film` Humour, on film: `posterize_ink` → `grain` → `vignette` (humour, 70s)
- `theme-empowerment` Empowerment: `god_rays` → `glow` → `saturation` (empowerment)
- `theme-empowerment-film` Empowerment, on film: `god_rays` → `glow` → `saturation` (empowerment, 70s)
- `theme-autumn` Autumn: `petals` → `white_balance` (autumn)
- `theme-autumn-film` Autumn, on film: `petals` → `white_balance` → `grain` → `vignette` (autumn, 70s)
- `theme-spring` Spring: `petals` → `lift_gamma_gain` (spring)
- `theme-spring-film` Spring, on film: `petals` → `lift_gamma_gain` → `grain` → `vignette` (spring, 70s)

### moods

Moods of light: night, golden hour, noir, dreams.

- `noir` Noir: `saturation` → `curve` → `grain` → `vignette` → `gate_weave` (jazz, blues, chanson, night and city, 1920s sepia and silent film)
- `dreamy` Dreamy: `diffusion` → `glow` → `lift_gamma_gain` → `liquid` (indie or shoegaze, dreams and the surreal, longing, love)
- `golden-hour` Golden hour: `white_balance` → `haze` → `light_leak` → `grain` (country and Americana, summer, nostalgia and childhood, freedom and road trips)
- `night` Night: `white_balance` → `lift_gamma_gain` → `starfield` → `aurora` → `vignette` (ambient, night and city, space and stars, lullabies)

### media

The picture painted, drawn and printed.

- `oil-painting` Oil painting: `kuwahara` → `oil_paint` → `canvas` (classical and orchestral, folk, ballads, love)
- `comic-ink` Comic ink: `posterize_ink` → `halftone` (punk, pop, humour, children's songs, 80s)
- `pop-art-halftone` Pop-art halftone: `saturation` → `halftone_cmyk` → `curve` (pop, disco, 60s, party, empowerment)
- `watercolour` Watercolour: `watercolor` → `lift_gamma_gain` → `canvas` → `liquid` (folk, celtic, childhood, nature, love)

### visualizer

Visualizers: the music drawn, the frame fed back, alone or over the picture.

- `tunnel-visualizer` Tunnel visualizer: `feedback` → `spectrum_ring` → `beat_rings` → `hue` (house, techno, trance, instrumental breaks)
- `viz-tunnel` Tunnel: `feedback` → `spectrum_ring` → `hue` (visualizer, instrumental breaks)
- `viz-scope-trails` Scope trails: `feedback` → `scope` (visualizer, instrumental breaks)
- `viz-lissajous-space` Lissajous in space: `starfield` → `feedback` → `lissajous` (visualizer, instrumental breaks)
- `viz-beat-rings` Beat rings: `lift_gamma_gain` → `beat_rings` → `glow` (visualizer, instrumental breaks)
- `viz-spectrum-frame` Spectrum frame: `feedback` → `bars` (visualizer, instrumental breaks)
- `viz-kaleido-feedback` Kaleidoscope feedback: `feedback` → `kaleidoscope` → `hue` (visualizer, instrumental breaks)
- `viz-sparks-night` Sparks at night: `white_balance` → `sparks` → `glow` (visualizer, instrumental breaks)
- `viz-swirl-break` Swirl in the breaks: `feedback` → `spectrum_ring` (visualizer, instrumental breaks)
- `viz-liquid-light` Liquid light show: `liquid` → `hue` → `feedback` (visualizer, instrumental breaks)
- `viz-neon-scope` Neon scope: `neon_edges` → `scope` (visualizer, instrumental breaks)
- `viz-mirror-rings` Mirror rings: `mirror` → `beat_rings` (visualizer, instrumental breaks)
- `viz-drop-rush` Drop rush: `zoom_blur` → `chromatic` → `feedback` (visualizer, instrumental breaks)

## The matrix: genre × theme × era

Each genre's own look, and the same look in its era and for the themes its songs often sing of (the presets' ids; each a starting point to move the knobs from). Any genre takes any era's or theme's layers the same way: `eras` and `themes` above have them alone.

| genre | its look | its era | its themes | era and theme |
|---|---|---|---|---|
| rock | `rock` | `rock-70s` | `rock-rebellion`, `rock-freedom`, `rock-heartbreak` | `rock-70s-rebellion` |
| punk | `punk` | `punk-80s` | `punk-rebellion`, `punk-party`, `punk-humour` | `punk-80s-rebellion` |
| grunge | `grunge` | `grunge-90s` | `grunge-grief`, `grunge-rebellion`, `grunge-longing` | `grunge-90s-grief` |
| doom-metal | `doom-metal` | `doom-metal-1920s` | `doom-metal-grief`, `doom-metal-horror`, `doom-metal-war` | `doom-metal-1920s-grief` |
| black-metal | `black-metal` | `black-metal-1920s` | `black-metal-winter`, `black-metal-horror`, `black-metal-nature` | `black-metal-1920s-winter` |
| power-metal | `power-metal` | `power-metal-today` | `power-metal-empowerment`, `power-metal-war`, `power-metal-freedom` | `power-metal-today-empowerment` |
| shoegaze | `shoegaze` | `shoegaze-90s` | `shoegaze-dreams`, `shoegaze-longing`, `shoegaze-love` | `shoegaze-90s-dreams` |
| pop | `pop` | `pop-today` | `pop-love`, `pop-party`, `pop-empowerment` | `pop-today-love` |
| synth-pop | `synth-pop` | `synth-pop-80s` | `synth-pop-night`, `synth-pop-love`, `synth-pop-space` | `synth-pop-80s-night` |
| disco | `disco` | `disco-70s` | `disco-party`, `disco-love`, `disco-night` | `disco-70s-party` |
| funk | `funk` | `funk-70s` | `funk-party`, `funk-money`, `funk-humour` | `funk-70s-party` |
| soul-rnb | `soul-rnb` | `soul-rnb-70s` | `soul-rnb-love`, `soul-rnb-heartbreak`, `soul-rnb-night` | `soul-rnb-70s-love` |
| k-pop | `k-pop` | `k-pop-today` | `k-pop-love`, `k-pop-party`, `k-pop-empowerment` | `k-pop-today-love` |
| city-pop | `city-pop` | `city-pop-80s` | `city-pop-night`, `city-pop-summer`, `city-pop-longing` | `city-pop-80s-night` |
| musicals | `musicals` | `musicals-1950s` | `musicals-love`, `musicals-dreams`, `musicals-humour` | `musicals-1950s-love` |
| schlager | `schlager` | `schlager-70s` | `schlager-love`, `schlager-hometown`, `schlager-party` | `schlager-70s-love` |
| house | `house` | `house-today` | `house-party`, `house-night`, `house-freedom` | `house-today-party` |
| techno | `techno` | `techno-today` | `techno-night`, `techno-rebellion`, `techno-drugs` | `techno-today-night` |
| trance | `trance` | `trance-y2k` | `trance-space`, `trance-dreams`, `trance-freedom` | `trance-y2k-space` |
| dubstep | `dubstep` | `dubstep-y2k` | `dubstep-rebellion`, `dubstep-party`, `dubstep-night` | `dubstep-y2k-rebellion` |
| drum-and-bass | `drum-and-bass` | `drum-and-bass-90s` | `drum-and-bass-night`, `drum-and-bass-freedom`, `drum-and-bass-space` | `drum-and-bass-90s-night` |
| ambient | `ambient` | `ambient-today` | `ambient-nature`, `ambient-space`, `ambient-dreams` | `ambient-today-nature` |
| chiptune | `chiptune` | `chiptune-80s` | `chiptune-humour`, `chiptune-party`, `chiptune-nostalgia` | `chiptune-80s-humour` |
| boom-bap | `boom-bap` | `boom-bap-90s` | `boom-bap-hometown`, `boom-bap-money`, `boom-bap-rebellion` | `boom-bap-90s-hometown` |
| trap | `trap` | `trap-today` | `trap-money`, `trap-night`, `trap-drugs` | `trap-today-money` |
| lo-fi | `lo-fi` | `lo-fi-90s` | `lo-fi-nostalgia`, `lo-fi-night`, `lo-fi-longing` | `lo-fi-90s-nostalgia` |
| reggae-dub | `reggae-dub` | `reggae-dub-70s` | `reggae-dub-freedom`, `reggae-dub-drugs`, `reggae-dub-summer` | `reggae-dub-70s-freedom` |
| afrobeat | `afrobeat` | `afrobeat-70s` | `afrobeat-party`, `afrobeat-empowerment`, `afrobeat-summer` | `afrobeat-70s-party` |
| reggaeton | `reggaeton` | `reggaeton-today` | `reggaeton-party`, `reggaeton-love`, `reggaeton-summer` | `reggaeton-today-party` |
| jazz | `jazz` | `jazz-1950s` | `jazz-night`, `jazz-love`, `jazz-longing` | `jazz-1950s-night` |
| blues | `blues` | `blues-1920s` | `blues-heartbreak`, `blues-grief`, `blues-hometown` | `blues-1920s-heartbreak` |
| country | `country` | `country-70s` | `country-freedom`, `country-hometown`, `country-heartbreak` | `country-70s-freedom` |
| folk | `folk` | `folk-60s` | `folk-nature`, `folk-family`, `folk-rebellion` | `folk-60s-nature` |
| celtic | `celtic` | `celtic-1920s` | `celtic-nature`, `celtic-mountains`, `celtic-longing` | `celtic-1920s-nature` |
| gospel | `gospel` | `gospel-1950s` | `gospel-faith`, `gospel-empowerment`, `gospel-family` | `gospel-1950s-faith` |
| chanson | `chanson` | `chanson-1950s` | `chanson-love`, `chanson-night`, `chanson-longing` | `chanson-1950s-love` |
| orchestral | `orchestral` | `orchestral-1920s` | `orchestral-nature`, `orchestral-grief`, `orchestral-faith` | `orchestral-1920s-nature` |
| opera | `opera` | `opera-1920s` | `opera-love`, `opera-grief`, `opera-war` | `opera-1920s-love` |
| salsa | `salsa` | `salsa-70s` | `salsa-party`, `salsa-love`, `salsa-summer` | `salsa-70s-party` |
| flamenco | `flamenco` | `flamenco-1950s` | `flamenco-love`, `flamenco-heartbreak`, `flamenco-summer` | `flamenco-1950s-love` |
| bollywood | `bollywood` | `bollywood-1950s` | `bollywood-love`, `bollywood-party`, `bollywood-family` | `bollywood-1950s-love` |
| childrens | `childrens` | `childrens-today` | `childrens-humour`, `childrens-nature`, `childrens-family` | `childrens-today-humour` |
| christmas | `christmas` | `christmas-1950s` | `christmas-family`, `christmas-winter`, `christmas-faith` | `christmas-1950s-family` |
| halloween | `halloween` | `halloween-1950s` | `halloween-horror`, `halloween-humour`, `halloween-party` | `halloween-1950s-horror` |

