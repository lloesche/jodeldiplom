# The looks' catalogue

Made from the passes' own definitions by `cargo run -p jodel-looks --example catalogue`; a test holds it to them. How to use it: `docs/looks.md`.

A look is JSON: an `id`, and `layers`, each a `pass` by name with the `knobs` it sets (by name; left out: the default), how the music moves them (`drive`: a list of `{"knob", "by", "amount"}`, the amount a share of the knob's swing from -1 to 1; left out: the pass's own drivers; `[]`: still), its `mix` over what is under it (0 to 1; `mix` can be driven too) and `keep` (the budget never drops it).

Rules held whatever is asked: every number to its range; at most 6 passes costing 8 in all (over it, the dearest pass not kept goes); the fast drivers together may move 0.08 of the screen's light, the slow ones 0.35; the lyric band's light under 0.55 (encoded).

## Passes

### grade

#### `lift_gamma_gain` (cost 1)

the classic three-way grade: the blacks raised or crushed, the middle brighter or darker, the whites brighter or dimmer.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `lift` | -1 to 1 | 0 | share | 0.3 | raises the blacks (+, a faded film) or crushes them (-, deep shadows) |
| `gamma` | -1 to 1 | 0 | share | 0.3 | the midtones brighter (+) or darker (-), the ends kept |
| `gain` | -1 to 1 | 0 | share | 0.3 | the whites brighter (+) or dimmer (-) |

Goes well with: `curve`, `grain`, `vignette`.

#### `white_balance` (cost 1)

white balance and tint, the light kept: daylight to tungsten, a green or a magenta cast.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `warmth` | -1 to 1 | 0 | share | 0.3 | warmer, orange (+), or cooler, blue (-) |
| `tint` | -1 to 1 | 0 | share | 0.3 | magenta (+) or green (-) |

Goes well with: `curve`, `grain`.

#### `saturation` (cost 1)

how vivid the colours are: saturation for all of them, vibrance for the dull ones only (skin and sky kept).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `saturation` | 0 to 2 | 1 | share | 0.4 | 0 grey, 1 as it is, 2 twice as vivid |
| `vibrance` | -1 to 1 | 0 | share | 0.4 | the dull colours more vivid (+) or duller (-), the vivid ones kept |

Moved by itself: `saturation` by `chorus` (0.4).

Goes well with: `curve`, `duotone`.

#### `duotone` (cost 1)

the picture in two colours, dark to light (the cover's own, or two hues): posters, pop, moods.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dark` | 0 to 1 | 0.7 | turns | 0.1 | the shadows' hue |
| `light` | 0 to 1 | 0.1 | turns | 0.1 | the lights' hue |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the two colours |
| `cover` | 0 to 1 | 0 | share | 0 | from the two hues (0) to the cover's own colours (1) |
| `contrast` | -1 to 1 | 0 | share | 0.3 | flatter (-) or harder (+) from dark to light |

Moved by itself: `light` by `part` (0.6).

Goes well with: `grain`, `halftone`, `curve`.

#### `tritone` (cost 1)

the picture in three colours, shadows, midtones and lights: screen prints, neon nights, sunsets.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dark` | 0 to 1 | 0.75 | turns | 0.1 | the shadows' hue |
| `middle` | 0 to 1 | 0.95 | turns | 0.1 | the midtones' hue |
| `light` | 0 to 1 | 0.12 | turns | 0.1 | the lights' hue |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the three colours |

Moved by itself: `middle` by `part` (0.5).

Goes well with: `grain`, `halftone`, `glow`.

#### `split_tone` (cost 1)

the shadows tinted one way and the lights another (teal and orange, the film look).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `shadows` | 0 to 1 | 0.52 | turns | 0.1 | the shadows' hue |
| `lights` | 0 to 1 | 0.08 | turns | 0.1 | the lights' hue |
| `amount` | 0 to 1 | 0.5 | share | 0.3 | how strongly tinted |
| `balance` | -1 to 1 | 0 | share | 0.3 | where shadows end and lights begin: lower (-) or higher (+) |

Goes well with: `curve`, `grain`, `halation`.

#### `curve` (cost 1)

a tone curve: more or less contrast about a pivot, and the matte fade of printed film.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `contrast` | -1 to 1 | 0 | share | 0.3 | flatter (-) or punchier (+) |
| `pivot` | 0.2 to 0.8 | 0.5 | share | 0.1 | the tone the contrast turns about |
| `fade` | 0 to 1 | 0 | share | 0.3 | the blacks lifted and the whites lowered, matte |

Goes well with: `split_tone`, `grain`.

#### `gradient_map` (cost 1)

the tones mapped onto the cover's three colours, dark to light: a picture made the cover's palette.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much of the cover's palette |

Goes well with: `grain`, `halftone`.

#### `technicolor` (cost 1)

the three-strip 1950s colour: each primary separated and rich, reds deep, skies cyan.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.8 | share | 0.3 | how much of the look |
| `punch` | 0 to 1 | 0.5 | share | 0.3 | how far the dyes separate |

Goes well with: `grain`, `halation`, `curve`.

#### `bleach_bypass` (cost 1)

the silver left in the print: harsh contrast, washed-out colour, metal and war films.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much of the look |
| `contrast` | 0 to 1 | 0.5 | share | 0.3 | how harsh |

Goes well with: `grain`, `vignette`.

#### `cross_process` (cost 1)

slide film in the wrong chemistry: yellow-green lights, blue shadows, hard reds (90s indie, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much of the look |

Goes well with: `grain`, `vignette`, `light_leak`.

#### `infrared` (cost 1)

infrared film: greens turn pink and red, skies deep (or in black and white, foliage glowing white).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.8 | share | 0.3 | how much of the look |
| `mono` | 0 to 1 | 0 | share | 0 | colour infrared (0) or black-and-white infrared (1) |

Goes well with: `glow`, `grain`.

#### `sepia` (cost 1)

an old photograph: warm sepia brown or a daguerreotype's cool silver, faded with age.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.9 | share | 0.3 | how much of the look |
| `tone` | -1 to 1 | 1 | share | 0.2 | a daguerreotype's silver (-1) to warm sepia (+1) |
| `fade` | 0 to 1 | 0.3 | share | 0.3 | how faded with age |

Moved by itself: `fade` by `progress` (0.5).

Goes well with: `film_damage`, `vignette`, `gate_weave`.

### film

#### `grain` (cost 1)

film grain: fine to coarse, mono or coloured, strongest in the midtones, new every film frame.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.35 | share | 0.3 | how much grain |
| `size` | 0.6 to 3 | 1.2 | px | 0.6 | the grains' size |
| `chroma` | 0 to 1 | 0.15 | share | 0.2 | mono (0) to coloured grain (1) |

Moved by itself: `amount` by `bass` (0.5).

Goes well with: `halation`, `vignette`, `curve`, `gate_weave`.

#### `halation` (cost 3)

halation: the red-orange glow film gets round bright lights, the 35 mm look at night.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.3 | how strong the glow |
| `radius` | 2 to 40 | 12 | px | 6 | how far it spreads |
| `warmth` | 0 to 1 | 0.6 | share | 0.2 | pinkish (0) to deep red-orange (1) |

Moved by itself: `amount` by `bass` (0.4).

Goes well with: `grain`, `split_tone`, `vignette`.

#### `glow` (cost 3)

bloom: the bright parts glowing round, tinted if wanted (dreamy, neon, stage light).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how strong the glow |
| `threshold` | 0.3 to 0.95 | 0.65 | share | 0.1 | how bright a part must be to glow |
| `radius` | 4 to 60 | 20 | px | 10 | how far it spreads |
| `hue` | 0 to 1 | 0.1 | turns | 0.1 | the glow's tint |
| `tint` | 0 to 1 | 0.2 | share | 0.2 | white (0) to the hue (1) |

Moved by itself: `amount` by `vocals` (0.6).

Goes well with: `diffusion`, `neon_edges`, `lift_gamma_gain`.

#### `diffusion` (cost 2.5)

a diffusion filter: soft focus, lights blooming gently, blacks a little lifted (dreamy, romantic, 70s).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how soft |
| `radius` | 2 to 30 | 8 | px | 4 | how far the softness reaches |

Moved by itself: `amount` by `vocals` (0.3).

Goes well with: `glow`, `curve`, `grain`.

#### `gate_weave` (cost 1)

gate weave: the whole frame drifting a little, as film does in an old projector.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how far it drifts (up to 2.5 pixels) |
| `speed` | 0.2 to 3 | 1 | share | 0.5 | how fast it drifts |

Moved by itself: `amount` by `bass_note` (0.3).

Goes well with: `grain`, `film_damage`, `sepia`.

#### `light_leak` (cost 1)

a light leak: warm light spilling in from an edge of the frame, drifting (summer, nostalgia, instant cameras).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the leak |
| `hue` | 0 to 1 | 0.06 | turns | 0.1 | its colour (orange, red, magenta ...) |
| `place` | 0 to 1 | 0.15 | turns | 0.2 | where round the frame it comes in |
| `size` | 0 to 1 | 0.5 | share | 0.3 | how far into the frame it reaches |

Moved by itself: `amount` by `downbeat` (0.5).

Goes well with: `grain`, `polaroid`, `cross_process`.

#### `vignette` (cost 1)

a vignette: the edges darkened, round as a lens does or along the frame, soft or hard.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how dark the edges |
| `softness` | 0 to 1 | 0.5 | share | 0.2 | how far in the darkening starts |
| `shape` | 0 to 1 | 0 | share | 0 | round (0) to the frame's shape (1) |

Moved by itself: `amount` by `quiet` (0.6).

Goes well with: `grain`, `curve`, `sepia`.

#### `polaroid` (cost 1.2)

an instant photo: the picture in a white frame with a wide bottom, faded and warm (family, nostalgia, summer).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `frame` | 0 to 1 | 0.6 | share | 0 | how wide the white frame (0: none) |
| `fade` | 0 to 1 | 0.5 | share | 0.2 | how faded, lifted and teal in the shadows |
| `warmth` | -1 to 1 | 0.3 | share | 0.2 | cooler (-) or warmer (+) |

Goes well with: `light_leak`, `grain`, `vignette`.

#### `film_damage` (cost 1)

dust specks and scratches, new each film frame: an old print, a silent film.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how much dust |
| `age` | 0 to 1 | 0.5 | share | 0.2 | how worn: scratches and darker dirt |

Moved by itself: `amount` by `snare` (0.4).

Goes well with: `sepia`, `gate_weave`, `grain`, `vignette`.

### print

#### `halftone` (cost 1.5)

a halftone screen: the picture as dots on paper, black ink or the picture's own colours (newsprint, comics, posters).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 3 to 30 | 8 | px | 4 | the dots' spacing |
| `angle` | 0 to 1 | 0.125 | turns | 0.05 | the screen's angle |
| `ink` | 0 to 1 | 0 | share | 0 | black ink (0) to the picture's colours (1) |
| `paper` | 0 to 1 | 0.3 | share | 0 | white (0) to yellowed (1) paper |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `dot` by `drums` (0.5).

Goes well with: `duotone`, `posterize_ink`, `curve`.

#### `halftone_cmyk` (cost 2.5)

four-colour process print: cyan, magenta, yellow and black screens at their angles, slightly off register if wanted (comics, magazines, pop art).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 3 to 24 | 7 | px | 3 | the dots' spacing |
| `angle` | 0 to 1 | 0 | turns | 0.05 | the screens turned together |
| `registration` | 0 to 1 | 0.2 | share | 0.4 | how far the plates are off register (up to 4 pixels) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `registration` by `snare` (0.5).

Goes well with: `posterize_ink`, `saturation`, `ben_day`.

#### `ben_day` (cost 2)

pop-art comic print: flat bold colours, Ben-Day dots in the midtones, black outlines.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 4 to 20 | 10 | px | 3 | the dots' spacing |
| `levels` | 2 to 5 | 3 | steps | 0 | how many steps each colour has |
| `outline` | 0 to 1 | 0.7 | share | 0.3 | how black the outlines |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `dot` by `kick` (0.4).

Goes well with: `halftone_cmyk`, `saturation`, `duotone`.

#### `posterize_ink` (cost 1.5)

cel shading: the tones in a few flat steps and ink lines round the edges (comics, cartoons, anime).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `steps` | 2 to 8 | 4 | steps | 0 | how many tones |
| `ink` | 0 to 1 | 0.7 | share | 0.3 | how dark the lines |
| `width` | 0.5 to 4 | 1.5 | px | 1 | how wide the lines |
| `vivid` | 0 to 2 | 1.2 | share | 0.3 | the colours' saturation |

Moved by itself: `width` by `kick` (0.4).

Goes well with: `halftone`, `ben_day`, `saturation`.

#### `risograph` (cost 2)

a risograph print: two bright inks laid one over the other, slightly off register, grainy (zines, indie, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `first` | 0 to 1 | 0.93 | turns | 0.05 | the first ink's hue (fluorescent pink by default) |
| `second` | 0 to 1 | 0.6 | turns | 0.05 | the second ink's hue (blue by default) |
| `misregister` | 0 to 1 | 0.35 | share | 0.4 | how far the inks are apart (up to 6 pixels) |
| `grain` | 0 to 1 | 0.5 | share | 0.3 | how grainy the ink |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `misregister` by `bass_note` (0.4).

Goes well with: `grain`, `halftone`, `canvas`.

#### `xerox` (cost 1)

a photocopy of a photocopy: hard black and white, toner speckle and streaks (punk flyers, zines, hardcore).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `threshold` | 0.2 to 0.8 | 0.5 | share | 0.1 | where black turns white: more black (+) or more white (-) |
| `toner` | 0 to 1 | 0.5 | share | 0.4 | how much speckle and streaking |
| `contrast` | 0 to 1 | 0.7 | share | 0.2 | soft grey (0) to hard black and white (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the copy over the picture |

Moved by itself: `toner` by `snare` (0.5).

Goes well with: `duotone`, `film_damage`, `glitch`.

#### `woodcut` (cost 1)

a woodcut or linocut: the picture carved in black lines that swell in the dark and follow the forms (folk, doom and black metal, protest posters).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `lines` | 3 to 20 | 7 | px | 2 | the lines' spacing |
| `angle` | 0 to 1 | 0.08 | turns | 0.05 | the lines' angle |
| `threshold` | -1 to 1 | 0 | share | 0.3 | darker (-) or lighter (+) overall |
| `ink` | 0 to 1 | 0 | share | 0 | black ink (0) to the cover's colour, darkened (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the print over the picture |

Moved by itself: `threshold` by `bass` (0.3).

Goes well with: `duotone`, `canvas`, `vignette`.

#### `engraving` (cost 1.3)

an engraving: the tones in fine hatching, crossed in the darks (banknotes, old books, classical, folk).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `spacing` | 2 to 10 | 4 | px | 1 | the lines' spacing |
| `angle` | 0 to 1 | 0.125 | turns | 0.05 | the first lines' angle |
| `layers` | 1 to 3 | 2 | steps | 0 | how many directions of hatching |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the engraving over the picture |

Goes well with: `sepia`, `canvas`, `vignette`.

#### `stipple` (cost 1.3)

stippling: the picture in ink dots, denser where it is dark (pen-and-ink, old scientific plates).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `dot` | 2 to 10 | 4 | px | 1.5 | the dots' size |
| `contrast` | 0 to 1 | 0.5 | share | 0.3 | how sharply dark and light part |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the stippling over the picture |

Goes well with: `sepia`, `canvas`.

### paint

#### `kuwahara` (cost 4)

a painterly smear that keeps the edges: flat patches of colour like brushwork (oil, gouache, anime backgrounds).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `radius` | 2 to 10 | 4 | px | 2 | the brush's size |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |

Moved by itself: `radius` by `bass` (0.4).

Goes well with: `canvas`, `oil_paint`, `saturation`.

#### `watercolor` (cost 2.5)

watercolour: soft washes that bleed at wobbly edges, pigment pooling darker there, the paper showing through (folk, childhood, gentle songs).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `wash` | 0 to 1 | 0.6 | share | 0.3 | how far the colours wash together |
| `bleed` | 0 to 1 | 0.5 | share | 0.3 | how wobbly the edges and how dark they pool |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |
| `granulation` | 0 to 1 | 0.4 | share | 0.2 | the pigment's grain in the darks |

Moved by itself: `bleed` by `vocals` (0.3).

Goes well with: `canvas`, `lift_gamma_gain`, `saturation`.

#### `oil_paint` (cost 3)

oil paint: the colour smeared in strokes that follow a flow, their impasto ridges lit from one side (classic, ballads, folk portraits).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `brush` | 2 to 12 | 5 | px | 2 | the strokes' size |
| `relief` | 0 to 1 | 0.5 | share | 0.3 | how much the ridges catch the light |
| `smear` | 0 to 1 | 0.6 | share | 0.3 | how long the strokes |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the painting over the picture |

Moved by itself: `relief` by `bass` (0.4).

Goes well with: `kuwahara`, `canvas`, `vignette`.

#### `canvas` (cost 1)

the ground showing through: canvas weave or paper fibre over the whole picture, the finish of every painted or drawn look.

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `kind` | 0 to 1 | 0 | share | 0 | canvas weave (0) to paper fibre (1) |
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how strongly the ground shows |
| `scale` | 0.5 to 3 | 1 | share | 0.3 | the weave's or fibre's size |

Goes well with: `oil_paint`, `watercolor`, `charcoal`, `woodcut`.

#### `charcoal` (cost 1.8)

charcoal or chalk on paper: smudged darks in strokes, the paper's tooth catching the dust (blues, folk, grief, sketches).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `smudge` | 0 to 1 | 0.4 | share | 0.3 | how smudged |
| `grain` | 0 to 1 | 0.6 | share | 0.2 | how much of the paper's tooth shows |
| `contrast` | 0 to 1 | 0.5 | share | 0.3 | how dark the darks |

Goes well with: `canvas`, `vignette`, `sepia`.

#### `pastel` (cost 1.2)

wax crayon or soft pastel: colour laid in strokes, the paper's tooth showing white through it (children's songs, childhood, sunny pop).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `stroke` | 2 to 12 | 6 | px | 2 | the strokes' size |
| `tooth` | 0 to 1 | 0.5 | share | 0.3 | how much paper shows through |
| `vivid` | 0.5 to 1.5 | 1.1 | share | 0.2 | the colours' saturation |

Goes well with: `canvas`, `saturation`, `marker`.

#### `marker` (cost 1.5)

felt-tip markers: flat bold colours in streaky strokes, darker where they overlap, a pen outline (pop, children's songs, humour).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the drawing over the picture |
| `steps` | 2 to 6 | 3 | steps | 0 | how many tones each colour has |
| `stroke` | 4 to 20 | 10 | px | 3 | the strokes' width |
| `outline` | 0 to 1 | 0.6 | share | 0.3 | how dark the pen outline |

Moved by itself: `outline` by `kick` (0.3).

Goes well with: `canvas`, `pastel`, `posterize_ink`.

### medium

#### `stained_glass` (cost 2)

leaded glass: the picture in pieces of coloured glass between dark lead lines, lit from behind (ballads, folk, fantasy).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `cell` | 12 to 80 | 36 | px | 8 | the pieces' size |
| `lead` | 0.5 to 6 | 2.5 | px | 1 | the lead lines' width |
| `glow` | 0 to 1 | 0.5 | share | 0.3 | how much the light shines through each piece's middle |

Moved by itself: `glow` by `vocals` (0.5).

Goes well with: `glow`, `vignette`.

#### `mosaic` (cost 1)

a mosaic: the picture in square tiles of one colour each, grout between them (Mediterranean, ancient, summer).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `tile` | 6 to 40 | 14 | px | 4 | the tiles' size |
| `grout` | 0 to 1 | 0.4 | share | 0.2 | how wide and how visible the grout |
| `variety` | 0 to 1 | 0.3 | share | 0.2 | how much each tile's shade varies |

Moved by itself: `tile` by `kick` (0.3).

Goes well with: `saturation`, `vignette`.

#### `cross_stitch` (cost 1)

cross-stitch: the picture in little X stitches of a few thread colours on cloth (folk, family, cosy, Volksmusik).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `stitch` | 4 to 20 | 8 | px | 2 | the stitches' size |
| `levels` | 2 to 5 | 3 | steps | 0 | how many shades of red, of green and of blue the threads come in |
| `fabric` | 0 to 1 | 0.6 | share | 0.2 | how much bare cloth is left in the lights |

Goes well with: `canvas`, `vignette`.

#### `cyanotype` (cost 1.6)

a cyanotype's Prussian blue print, or a blueprint's white lines and grid on blue (science, the sea, dreams, plans).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `plan` | 0 to 1 | 0 | share | 0 | a cyanotype print (0) to a blueprint's lines (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the medium over the picture |
| `grid` | 0 to 1 | 0.3 | share | 0.2 | how much of a blueprint's grid |

Goes well with: `canvas`, `vignette`.

#### `thermal` (cost 1)

a heat camera: the brightness read as heat, in an iron palette or a rainbow (techno, sci-fi, war, horror, sweat and dance).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the camera over the picture |
| `palette` | 0 to 1 | 0 | share | 0 | iron, black to white-hot (0), to a rainbow (1) |
| `noise` | 0 to 1 | 0.3 | share | 0.3 | the sensor's noise |

Moved by itself: `noise` by `high` (0.5).

Goes well with: `crt`, `glitch`.

#### `night_vision` (cost 1)

night vision: a green phosphor picture, noisy and lined, in a round tube (night, war, spies, horror).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the goggles over the picture |
| `noise` | 0 to 1 | 0.4 | share | 0.3 | the tube's noise |
| `lines` | 0 to 1 | 0.3 | share | 0.2 | how strong the scan lines |

Moved by itself: `noise` by `drums` (0.4).

Goes well with: `vignette`, `glitch`.

#### `negative` (cost 1)

a film negative, or an X-ray's cold glow (horror, Halloween, the surreal, sci-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 1 | share | 0.2 | how much of the negative over the picture |
| `xray` | 0 to 1 | 0.5 | share | 0 | a colour negative (0) to an X-ray (1) |
| `contrast` | 0 to 1 | 0.4 | share | 0.2 | how dark the X-ray's dark parts |

Goes well with: `glow`, `grain`.

#### `solarize` (cost 1)

solarisation, the darkroom's turned tones, or liquid chrome's bands (psychedelia, Y2K chrome, glam).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `threshold` | 0.3 to 0.9 | 0.55 | share | 0.1 | where the tones turn back |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the effect over the picture |
| `chrome` | 0 to 1 | 0 | share | 0 | solarised (0) to chrome bands (1) |

Goes well with: `hue`, `glow`.

#### `neon_edges` (cost 2)

the picture's edges as glowing neon tubes over a darkened picture (synthwave, the 80s, night and the city, club music).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `width` | 0.5 to 4 | 1.5 | px | 1 | the tubes' width |
| `hue` | 0 to 1 | 0.85 | turns | 0.2 | the neon's colour |
| `glow` | 0 to 1 | 0.6 | share | 0.4 | how far the glow spreads |
| `dark` | 0 to 1 | 0.6 | share | 0.2 | how dark the picture under the neon |
| `cycle` | 0 to 1 | 0 | share | 0.3 | still (0) to a rainbow running through the colours (1) |

Moved by itself: `glow` by `kick` (0.5).

Goes well with: `glow`, `crt`, `tritone`.

#### `teletext` (cost 2)

teletext's block graphics: each character cell of six blocks lit or dark in one of eight bright colours on black (no letters: shapes only) (the 80s, chiptune, retro TV).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `cell` | 6 to 24 | 12 | px | 3 | a cell's width |
| `cover` | 0 to 1 | 0 | share | 0 | teletext's eight colours (0) to the cover's colour (1) |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the blocks over the picture |

Moved by itself: `cell` by `kick` (0.3).

Goes well with: `crt`, `pixelate`.

### retro

#### `crt` (cost 2)

an old tube screen: scan lines, the phosphor mask's stripes, a curved glass and its glow (the 80s, video games, synthwave).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `scanlines` | 0 to 1 | 0.5 | share | 0.3 | how strong the scan lines |
| `mask` | 0 to 1 | 0.4 | share | 0.2 | how strong the colour mask's stripes |
| `curve` | 0 to 1 | 0.3 | share | 0 | how curved the glass |
| `glow` | 0 to 1 | 0.3 | share | 0.4 | how much the phosphor glows |

Moved by itself: `glow` by `bass` (0.4).

Goes well with: `vhs`, `pixelate`, `palette`, `teletext`.

#### `vhs` (cost 1.8)

a worn video tape: lines wobbling, the colour bleeding sideways, noise and a band of bad tracking rolling up (the 80s and 90s, camcorder, lo-fi).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `wobble` | 0 to 1 | 0.4 | share | 0.4 | how much the lines wobble |
| `bleed` | 0 to 1 | 0.5 | share | 0.3 | how far the colour bleeds |
| `noise` | 0 to 1 | 0.3 | share | 0.3 | how much noise and snow |
| `tracking` | 0 to 1 | 0.3 | share | 0.4 | how bad the tracking band |

Moved by itself: `tracking` by `snare` (0.5).

Goes well with: `crt`, `cross_process`, `grain`.

#### `pixelate` (cost 1)

big pixels in a few shades a channel, an ordered dither between them (8-bit and 16-bit games, chiptune).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `size` | 2 to 24 | 6 | px | 3 | the pixels' size |
| `levels` | 2 to 16 | 6 | steps | 0 | how many shades a channel |
| `dither` | 0 to 1 | 0.6 | share | 0.2 | none (0) to a full ordered dither (1) |

Moved by itself: `size` by `kick` (0.3).

Goes well with: `crt`, `palette`, `glitch`.

#### `palette` (cost 1)

an old machine's few colours, dithered: a handheld's four greens, a home computer's black-cyan-magenta-white, an eight-colour micro's, or the cover's own four (chiptune, video-game music).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `palette` | 0 to 3 | 0 | steps | 0 | handheld greens (0), cyan-magenta (1), eight colours (2), the cover's (3) |
| `size` | 2 to 12 | 4 | px | 2 | the pixels' size |
| `dither` | 0 to 1 | 0.5 | share | 0.2 | none (0) to a full ordered dither (1) |

Goes well with: `crt`, `pixelate`.

#### `glitch` (cost 1.5)

digital glitches: blocks of the picture torn sideways and tinted, its colours split, new twelve times a second (Y2K, datamosh, industrial, dubstep drops).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.3 | share | 0.6 | how many blocks tear |
| `block` | 8 to 80 | 32 | px | 16 | the blocks' size |
| `split` | 0 to 1 | 0.4 | share | 0.4 | how far the colours split |

Moved by itself: `amount` by `snare` (0.8).

Goes well with: `vhs`, `chromatic`, `xerox`.

#### `anaglyph` (cost 1)

a red-and-cyan 3D picture, the two eyes' views apart (the 50s cinema, B-movies, sci-fi, fun).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `offset` | 1 to 20 | 6 | px | 4 | how far apart the eyes' views |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the 3D over the picture |
| `depth` | 0 to 1 | 0.5 | share | 0.2 | how much brighter parts come forward |

Moved by itself: `offset` by `kick` (0.5).

Goes well with: `grain`, `technicolor`.

#### `chromatic` (cost 1)

the colours split apart, red one way and blue the other, as a cheap lens or a shaken screen does (psychedelia, glitch, drops).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.3 | share | 0.5 | how far the colours split (up to 12 pixels) |
| `radial` | 0 to 1 | 1 | share | 0 | the same way everywhere (0) to out from the middle (1) |
| `angle` | 0 to 1 | 0 | turns | 0.2 | the way they split, when not out from the middle |

Moved by itself: `amount` by `kick` (0.6).

Goes well with: `glitch`, `hue`, `liquid`.

### atmosphere

#### `haze` (cost 1)

haze: the picture fading into the air's colour toward the top and the distance (mountains, summer heat, dreams, longing).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how thick the haze |
| `hue` | 0 to 1 | 0.58 | turns | 0.1 | the air's colour |
| `horizon` | 0 to 1 | 0.6 | share | 0.1 | where the haze starts, up from the bottom |
| `height` | 0.1 to 1 | 0.5 | share | 0.1 | how far up it thickens |

Moved by itself: `amount` by `energy` (-0.3).

Goes well with: `god_rays`, `split_tone`, `glow`.

#### `god_rays` (cost 4)

rays of light streaming from a bright point through the picture (gospel, anthems, sunrise, faith, choruses).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the rays |
| `x` | 0 to 1 | 0.5 | share | 0.1 | the light's place across |
| `y` | 0 to 1 | 0.25 | share | 0.1 | the light's place down |
| `length` | 0.1 to 1 | 0.5 | share | 0.2 | how long the rays |

Moved by itself: `amount` by `chorus` (0.6).

Goes well with: `haze`, `glow`, `lens_flare`.

#### `fog` (cost 1.2)

fog drifting across in slow banks, lying low if wanted (horror, Halloween, dub smoke, grief, mountains).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.3 | how thick the fog |
| `scale` | 0.5 to 4 | 1.5 | share | 0.5 | the banks' size: big (low) to small (high) |
| `speed` | 0 to 2 | 0.5 | share | 0.5 | how fast it drifts |
| `hue` | 0 to 1 | 0.6 | turns | 0.1 | its colour |
| `low` | 0 to 1 | 0.5 | share | 0.2 | everywhere (0) to lying low (1) |

Moved by itself: `amount` by `bass` (0.3).

Goes well with: `vignette`, `haze`, `night_vision`.

#### `rain` (cost 1.5)

rain falling in streaks, near and far, slanting with the wind (heartbreak, grief, the city at night, blues).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how heavy the rain |
| `slant` | -1 to 1 | 0.2 | share | 0.3 | the wind's slant, left (-) or right (+) |
| `length` | 0 to 1 | 0.5 | share | 0.2 | how long the streaks |
| `speed` | 0.5 to 2 | 1 | share | 0.3 | how fast it falls |

Moved by itself: `amount` by `chorus` (0.6).

Goes well with: `fog`, `split_tone`, `glow`.

#### `snow` (cost 1.3)

snow falling softly in layers, near flakes bigger, drifting with the wind (Christmas, winter, cosy songs, Schlager).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how heavy the snow |
| `size` | 0.5 to 3 | 1 | share | 0.5 | the flakes' size |
| `wind` | -1 to 1 | 0.2 | share | 0.3 | blowing left (-) or right (+) |
| `speed` | 0.3 to 2 | 1 | share | 0.3 | how fast it falls |

Moved by itself: `amount` by `chorus` (0.6).

Goes well with: `glow`, `white_balance`, `vignette`.

#### `petals` (cost 1.2)

blossom petals or autumn leaves tumbling down (love, spring, K-pop and J-pop ballads, autumn, nostalgia).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how many fall |
| `kind` | 0 to 1 | 0 | share | 0 | pink blossom (0) to autumn leaves (1) |
| `size` | 0.5 to 3 | 1 | share | 0.4 | their size |
| `wind` | -1 to 1 | 0.3 | share | 0.3 | blowing left (-) or right (+) |

Moved by itself: `amount` by `chorus` (0.5).

Goes well with: `glow`, `diffusion`, `split_tone`.

#### `dust` (cost 1)

dust motes floating slowly, glinting where the light is (old rooms, memories, folk, attics, quiet songs).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how many motes |
| `size` | 0.5 to 3 | 1 | share | 0.4 | their size |
| `lit` | 0 to 1 | 0.6 | share | 0 | everywhere alike (0) to glinting only in the light (1) |

Moved by itself: `amount` by `other` (0.5).

Goes well with: `god_rays`, `sepia`, `film_damage`.

#### `heat_haze` (cost 1)

heat shimmering the picture, more near the ground (desert, summer, road trips, Americana, flamenco).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.4 | how strong the shimmer |
| `scale` | 0.5 to 3 | 1 | share | 0.3 | the ripples' size |
| `ground` | 0 to 1 | 0.6 | share | 0 | everywhere (0) to near the ground (1) |

Moved by itself: `amount` by `energy` (0.4).

Goes well with: `haze`, `white_balance`, `grain`.

#### `underwater` (cost 1.3)

under water: the reds absorbed, the picture wavering, caustic light dancing from above (the sea, dreams, ambient, dub).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.7 | share | 0.3 | how much under water |
| `caustics` | 0 to 1 | 0.5 | share | 0.4 | how bright the dancing light |
| `depth` | 0 to 1 | 0.5 | share | 0.2 | how deep: how much red is lost |

Moved by itself: `caustics` by `other` (0.4).

Goes well with: `god_rays`, `liquid`, `glow`.

#### `candle` (cost 1)

candlelight: a warm pool of gently flickering light, the rest in shadow (Christmas, chanson, intimate ballads, faith).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.3 | how much candlelight and shadow |
| `warmth` | 0 to 1 | 0.7 | share | 0.2 | pale gold (0) to deep orange (1) |
| `x` | 0 to 1 | 0.5 | share | 0 | the candle's place across |
| `y` | 0 to 1 | 0.65 | share | 0 | the candle's place down |
| `reach` | 0.2 to 1 | 0.6 | share | 0.2 | how far its light reaches |

Moved by itself: `reach` by `vocals` (0.4).

Goes well with: `glow`, `grain`, `vignette`.

#### `aurora` (cost 1.2)

northern lights: glowing curtains rippling in the sky, over the dark parts (winter, space, ambient, wonder).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how bright the curtains |
| `hue` | 0 to 1 | 0.38 | turns | 0.2 | their colour (green by default) |
| `height` | 0 to 1 | 0.35 | share | 0.2 | how far down they reach |
| `speed` | 0 to 2 | 0.5 | share | 0.4 | how fast they ripple |

Moved by itself: `amount` by `other` (0.5), `hue` by `part` (0.5).

Goes well with: `starfield`, `glow`, `white_balance`.

#### `starfield` (cost 1)

stars twinkling slowly in the dark parts of the picture (space and stars, night, lullabies, dreams).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.4 | how many stars |
| `twinkle` | 0 to 1 | 0.5 | share | 0.3 | how much they twinkle (slowly) |
| `size` | 0.5 to 3 | 1 | share | 0.4 | their size |

Moved by itself: `twinkle` by `high` (0.5).

Goes well with: `aurora`, `glow`, `night_vision`.

#### `lens_flare` (cost 1)

a lens flare: a light's glow and streak, its ghosts strung across the frame (summer, road trips, the 80s, stadium rock).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.5 | share | 0.5 | how strong the flare |
| `x` | 0 to 1 | 0.75 | share | 0.1 | the light's place across |
| `y` | 0 to 1 | 0.25 | share | 0.1 | the light's place down |
| `hue` | 0 to 1 | 0.08 | turns | 0.1 | its colour |

Moved by itself: `amount` by `downbeat` (0.4).

Goes well with: `light_leak`, `god_rays`, `glow`.

#### `tilt_shift` (cost 2.5)

a tilted lens: only a band in focus, above and below it blurred, the world looking like a model (city pop, childhood, toy-like whimsy).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `focus` | 0 to 1 | 0.55 | share | 0.1 | where the sharp band is, down the frame |
| `band` | 0.05 to 0.6 | 0.2 | share | 0.1 | how tall the sharp band |
| `blur` | 0 to 1 | 0.6 | share | 0.3 | how blurred the rest |

Goes well with: `saturation`, `curve`.

### psychedelic

#### `hue` (cost 1)

the colours turned round the wheel: shifted, cycling slowly, or a rainbow across the screen, the light kept (psychedelia, a new hue for each section).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `shift` | 0 to 1 | 0 | turns | 1 | how far round the wheel |
| `spread` | 0 to 1 | 0 | share | 0.3 | a rainbow across the screen |
| `speed` | -1 to 1 | 0 | share | 0.3 | cycling backwards (-) or forwards (+), a turn in ten seconds at the most |
| `amount` | 0 to 1 | 1 | share | 0.3 | how much of the turned colour |

Moved by itself: `shift` by `part` (1).

Goes well with: `kaleidoscope`, `liquid`, `saturation`.

#### `kaleidoscope` (cost 1)

a kaleidoscope: the picture folded into mirrored segments round the middle, turning (psychedelia, trance, instrumental breaks).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `segments` | 2 to 16 | 6 | steps | 0 | how many mirrored segments |
| `turn` | 0 to 1 | 0 | turns | 1 | how far turned |
| `zoom` | 0.5 to 2 | 1 | share | 0.4 | zoomed out (low) or in (high) |

Moved by itself: `turn` by `bar_phase` (0.25), `zoom` by `bass` (0.3).

Goes well with: `hue`, `feedback`, `liquid`.

#### `mirror` (cost 1)

the picture mirrored left to right, top to bottom or four ways (symmetry, Rorschach, dance, the surreal).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `mode` | 0 to 2 | 0 | steps | 0 | left onto right (0), top onto bottom (1), four ways (2) |
| `place` | 0.2 to 0.8 | 0.5 | share | 0.2 | where the mirror stands |

Moved by itself: `place` by `other` (0.2).

Goes well with: `kaleidoscope`, `hue`, `feedback`.

#### `liquid` (cost 1.2)

the picture flowing and warping like oil on water (psychedelia, drugs and dreams, the surreal, shoegaze).

| knob | range | default | unit | music's swing | what it does |
|---|---|---|---|---|---|
| `amount` | 0 to 1 | 0.4 | share | 0.5 | how far it flows |
| `scale` | 0.5 to 4 | 1.5 | share | 0.5 | the swirls' size: big (low) to small (high) |
| `speed` | 0 to 2 | 0.5 | share | 0.5 | how fast it flows |

Moved by itself: `amount` by `bass` (0.4).

Goes well with: `hue`, `chromatic`, `feedback`.

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

### examples

The hand-made examples: a look each for a kind of song, to start from.

- `35mm-night-drive` 35mm night drive: `split_tone` → `curve` → `halation` → `grain` → `vignette` (synth-pop, rock, night and city, freedom and road trips, 70s film stock)
- `psychedelic-60s` Psychedelic 60s: `saturation` → `liquid` → `hue` → `kaleidoscope` (rock, 60s psychedelic, drugs and psychedelia, dreams and the surreal)
- `oil-painting` Oil painting: `kuwahara` → `oil_paint` → `canvas` (classical and orchestral, folk, ballads, love)
- `comic-ink` Comic ink: `posterize_ink` → `halftone` (punk, pop, humour, children's songs, 80s)
- `pop-art-halftone` Pop-art halftone: `saturation` → `halftone_cmyk` → `curve` (pop, disco, 60s, party, empowerment)
- `noir` Noir: `saturation` → `curve` → `grain` → `vignette` (jazz, blues, chanson, night and city, 1920s sepia and silent film)
- `watercolour` Watercolour: `watercolor` → `lift_gamma_gain` → `canvas` (folk, celtic, childhood, nature, love)
- `synthwave` Synthwave: `tritone` → `neon_edges` → `bars` → `crt` (synth-pop, 80s neon and VHS, night and city, space and stars)
- `dreamy` Dreamy: `diffusion` → `glow` → `lift_gamma_gain` → `saturation` (indie or shoegaze, dreams and the surreal, longing, love)
- `golden-hour` Golden hour: `white_balance` → `haze` → `light_leak` → `grain` (country and Americana, summer, nostalgia and childhood, freedom and road trips)
- `night` Night: `white_balance` → `lift_gamma_gain` → `starfield` → `vignette` (ambient, night and city, space and stars, lullabies)
- `camcorder-90s` 90s camcorder: `cross_process` → `vhs` → `crt` (grunge, indie, 90s camcorder and MTV, nostalgia and childhood, family)
- `tunnel-visualizer` Tunnel visualizer: `feedback` → `spectrum_ring` → `beat_rings` → `hue` (house, techno, trance, instrumental breaks)

