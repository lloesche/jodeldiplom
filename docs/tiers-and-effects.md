# Tiers, rungs, effects and sounds

How the karaoke game rewards singing, as built: the tier table, the rungs
of the multiplier, the effects library, the music clock they all move by,
the sounds the conductor plays, and the themes over all of it. A new tier is one row of a table; a
new effect is one module and one line; nothing matches on a tier by name.

## What it is for

The rewards build over a song. A phrase sung well is rated Good, Great,
Awesome, Amazing or Diva; words hit in a row climb a multiplier from ×2 to
×32, each rung named after a tier. The higher the singing, the more the
stage shows: rainbow bands, a glow round the lane, flames, marquee bulbs,
coins, angels, the screen's edges on fire, a word slammed down ("DIVA!").
The sounds grow with it: more notes, longer, in the song's key and on its
beat. The lyrics may be covered a little, never much.

## The tier table: `player/src/tiers.rs`

`TIERS` is the only place a tier is defined: a `TierSpec` a row, Good
first, the top last. A tier is `Tier(u8)`, its place in the table
(`Tier::ALL`, `Tier::TOP`, `Tier::spec`, `Tier::name`, `Tier::bar`).

| Field | What it sets |
|---|---|
| `name` | the name shown (the badge, the verdict, the diploma) |
| `bar`, `pitch_bar` | the shine it takes, as a share of what the original singer reaches; a pitch share too, where the row asks for one |
| `earned` | it must be earned in a row: it counts only right after a phrase that counted at least the row below (`Tier::counted`) |
| `streak` | words hit in a row for its rung of the multiplier |
| `color`, `fill` | its colour (badge, progress mark, sticker) and how it fills the meter (a colour, or the rainbow) |
| `glow` | how strongly the singing glows while at it, 0 to 1 (the top's 1) |
| `bands`, `lane_light` | the rainbow's bands from the playhead, and how much it lights the lane |
| `fx` | the effects it lists ([below](#the-effects-library-playersrcguifx)) |
| `backdrop_glow`, `backdrop_rays`, `backdrop_beat` | how the backdrop lights its subject, its spotlight rays, and how hard it moves with the beat |
| `mark` | the size of its phrases' marks on the song's progress |
| `sticker` | its sticker on the diploma (glints, a halo) |
| `prismatic` | it shines in the rainbow: its sticker's colour runs through it, its slammed word's face is it |
| `layers` | the layers of its sounds ([below](#the-sounds-playersrcsfx)) |
| `call` | the word the announcer calls a phrase reaching it out with, its name (its rung's is its multiplier's: `call::times`) |

A phrase reaching a row whose `fx.on_phrase` fires the jackpot's burst
hits the jackpot's sound too (`TierSpec::jackpot`). Where a row's moment
(`tiers::Landing`) lands is one rule (`tiers::lands`): a rung's on the
next beat, with its stinger; a phrase's on a bar's first beat with room
for `MOMENT_BEATS` before a sung line, with the jackpot. The slam, the
jackpot's sound and the announcer all land by it.

The rows today:

| Tier | Bar | Rung at | × | Effects held | On a phrase | On its rung | At its rung |
|---|---|---|---|---|---|---|---|
| Good | 0.60 | 8 words | ×2 | rainbow | step-up | | |
| Great | 0.72 | 20 | ×4 | rainbow, lane glow | step-up | | |
| Awesome | 0.80 | 40 | ×8 | rainbow, glow | step-up | coin fountain | coin flick, bulbs 0.6 |
| Amazing | 0.87 | 75 | ×16 | rainbow, glow, lyrics on fire 0.3 | step-up | coin fountain | coin flick, bulbs, flames |
| Diva | 0.94, pitch 0.92, earned | 120 | ×32 | rainbow, glow, angels, lyrics on fire 0.6 | step-up, jackpot, slam | coin fountain, coin rain, slam | coin flick, bulbs, flames, edge fire |

How a phrase gets its tier is the shine's (`player/src/core/shine`): its
coverage of the words times its quality, each measure a share of the
original singer's same phrase. The bars are shares of the phrase's par,
what the original's own everyday hearings reach there, so the original is
Diva on its own song and a good take reaches it a few phrases a song.
`examples/shine.rs` measures that on real songs and takes.

### Rungs

`core::singing::rung(streak)` is the last row whose `streak` the words in
a row reach (none below the first: ×1). The multiplier is ×2 at the
first rung and doubles a rung (`Tier::multiplier`). The moments say when
the streak goes up a rung and when it is lost from one
(`Cue::StreakUp(tier)`, `Cue::StreakLost(tier)`). The badge over the
playhead reads the rung's name and the streak: "Amazing 82 in a row ×16".

### Looks held between rows

The singing's glow eases from row to row. A look held while at a tier (the
backdrop's light, an effect's strength) is the straight line between the
rows' values at their glows, none below the first (`tiers::held`). At a
row's glow it is that row's value exactly.

### What reads the table

The meter's marks and fills, the rainbow's bands, the backdrop's light,
the badge, the diploma's stickers and tally, the sounds' layers. The
shaders get the table as generated constants (`const TIERS`, the marks,
the fills: `tiers::wgsl_array`, `gui::stage::shapes::glory`); no tier number is written
in WGSL.

`every_row_is_in_order` holds the rows to their order: each bar, streak,
glow, band count and backdrop light above the row before, each held look
no less, each row looking other than the one before, its rung adding
sound layers no row below adds, its chime's other. Adding a row is
adding a line to `TIERS` and seeing that test pass.

## The effects library: `player/src/gui/fx/`

An effect is a module with a `const SPEC: Spec`, and one line in the
`registry!` of `gui/fx/mod.rs`, which makes the `Fx` enum. The registry's
order is the order within a slot (the lane's glow before its flames keeps
the light's order). A tier's row lists effects as `Use { fx, strength,
mods }` in four lists (`fx::Effects`):

- `held`: while the singing is at the tier;
- `on_phrase`: fired once when a phrase reaches it;
- `on_rung`: fired once when the streak reaches its rung;
- `at_rung`: held while the streak is at its rung.

`fx::Board` eases every effect's strength frame by frame from those lists:
one clock for every held effect (the singing's glow), the rung's held
effects eased the same way, fired ones played out over the frames after
(`Playing`: the coin rain's bursts on the beats, the slam waiting for a
bar's first beat).

### How an effect draws

The stage asks the effects to draw at fixed points of its draw order (a
`Layer`, with its geometry as an `At`): under the tubes, round the arrow,
from the playhead, from the meter, round the lane, over the screen, from a
phrase's mark, as a screen pass, and over everything. An effect draws in
one or more ways:

- **stage shapes:** instances of its own kinds, drawn by a pipeline of
  its own (`stage::gpu::Group::Shapes`) with only its WGSL: the rainbow's
  bands, the lane's glow, the flames;
- **screen passes:** shapes over the finished frame, given the lyric band
  and the score, badge and meter to keep clear (`fx::Clear`): the edges
  on fire, the marquee bulbs;
- **lyric treatments:** a function of a letter's colour and coverage, a
  pipeline each (`stage::gpu::Group::Lyric`), on the shouts or the line
  sung: the rainbow, the lyrics on fire, frosted;
- **bursts:** a record for the particles' pass, worked out in its vertex
  shader: the step-up, the jackpot, the angels, the coins;
- **big words** over everything, lettered from the stage's atlas: the
  slam, shaking the screen.

| Effect | Module | Draws as |
|---|---|---|
| Rainbow | `rainbow.rs` | bands from the playhead, a swirl round the arrow, the rim of the word sung, the shouts in rainbow |
| StepUp | `step_up.rs` | a burst of stars and notes from the playhead each tier a phrase rises |
| Jackpot | `jackpot.rs` | coins and stars from the meter and the phrase's mark |
| Glow | `glow.rs` | a warm glow round the lane, pulsing on the beat (light) |
| Flames | `flames.rs` | flames along the lane's edges (light); they set the bulbs on fire |
| Angels | `angels.rs` | flights of angels down the screen while at the top (light) |
| EdgeFire | `edge_fire.rs` | the screen's edges on fire (a screen pass) |
| Bulbs | `bulbs.rs` | marquee bulbs round the lane, chasing on the beat (a screen pass) |
| LyricFire | `lyric_fire.rs` | the line sung on fire (a lyric treatment) |
| LyricFrost | `lyric_frost.rs` | the line frosted (a lyric treatment) |
| Fountain, Rain, Flick | `coins.rs` | coins from the playhead, down the whole screen, out of each word hit |
| Slam | `slam.rs` | the tier's name or the multiplier slammed down behind the playhead |
| Fog | `fog.rs` | fog creeping in at the screen's edges, eyes blinking in it (a screen pass; held by a theme, no row lists it) |

The kinds of shape are numbered in one scheme: an effect's kinds from
`FX_KINDS` (32), four each (`Fx::kind`). Each effect's WGSL names its
helpers after it (`<ident>_shape`, `lyric_<ident>`), so all of them share
the stage's shader; the fire every burning effect draws is one
(`fx::helpers::fire`). Every effect's look is four numbers
(`Spec::params`), sent to the shaders as one uniform (`looks`).

### Held to the maths, and to the frame budget

- Each effect's maths is in Rust too, its shader's reference
  (`Spec::reference`); `gui/shader_check.rs` runs the WGSL on a GPU and
  holds it to that.
- `the_registry_is_whole_and_its_names_unique` checks names, prefixes,
  kinds and pipelines.
- `player/tests/fx_digest.rs` runs a scripted Diva run and pins a digest
  of the stage's draw list, so a refactor can be shown to change nothing;
  a change on purpose takes the new digest from the test's output.
- `examples/frame_cost.rs` times each step of a frame on the GPU, per
  tier and rung; on the Mac a frame must stay under 4 ms and each step
  under 2.5 ms (`--budget 4 --step-budget 2.5`, `PERFORMANCE.md`).

## The music clock: `player/src/gui/fx/music.rs`

Everything that moves with the song moves by one clock, worked out once a
frame (`fx::Music`) through the one reader of the music,
`jodel_looks::drive`: the beat from the analysis's grid (`drive::beat`:
the phases, the pulses on each beat and each bar's first, the tempo, how
sure the grid is: where it is unsure the beat fades and the looks run on
their own clock), the lanes' levels against how loud they get
(`drive::Range`), the melody's pitch and a swell as a section starts. The
song's look moves by the same numbers: every driver of the looks'
catalogue (`jodel_looks::Driver`: the song's curves read at the time, the
singing's set live over them by `core::karaoke::live`), the frame's
`Signals`, which the backdrop reads once for the look and the effects.

A row can move an effect with the music through `Mod { param, by, amount
}`: `param` is its strength or one of its look's numbers, `by` a driver,
as a synth routes a modulation (no row uses one today: the effects that
move with the beat, the glow, the bulbs, the fire, do it in their own
maths). The same signals go to the stage's pipelines as a small uniform,
so a shader can use the beat at no cost. The backdrop's pulse (`gui::backdrop::pulse`) reads the same clock.

Where something lands on the beat is one rule, the grid's
(`jodel_rhythm::grid::Grid::land`): a look fired with a sound lands with
it.

## The sounds: `player/src/sfx/`

Every sound is played live by our own synthesizer (`jodel-synth`) on the
desk's effects input, on its own strip (the Effects strip). The conductor
(`sfx::conductor`) plans them on the UI thread, where the game's moments
are read, as timed notes on the song's stream frames; the mixer thread
puts them on the synth's frames as it mixes, so each lands inside the
period it falls in.

### What plays, and when

- **At once:** a word's pop on the pitch ladder, its coin's clink at the
  rungs that have one, the rescue shot's zap, the wind-down of a streak
  lost.
- **On the next beat**, a pickup at once: a rung's stinger, a phrase's
  chime, the diploma's points and its best streak's rung.
- **On a downbeat with room before the next sung line:** the top rung's
  payout, the top tier's jackpot.
- **Held:** a riser into a high rung, a shimmer pad while the streak is
  high, glitter after each word at the top. A streak lost lets them go.
- **The diploma's own:** its paper, pen, stamps and tunes, each on fixed
  notes and the same every time (`sfx/conductor/sheet.rs`), timed inside
  to what the stage draws.

### In the song's key

The key is the analysis's, shifted as the song is heard, as a pentatonic
(`sfx/conductor/key.rs`): no half step to clash with the song's chords.
Each word hit in a row pops a step higher, and the word that reaches a
rung lands on the tonic an octave over where the climb began
(`sfx/conductor/ladder.rs`; Peggle's rising peg hits, US patent
8,128,476). With no key known, the pitched layers stay out and what rings
sits on fixed notes high over the song.

### What a row adds

A row's `layers` (`sfx/conductor/layers.rs`) say what its sounds add over
the base ones:

- the streak's its rung adds (`Streak`, `Layers::adds`; a rung plays its
  own and every rung's below, `at_rung`): a glint over each pop, the
  stinger's octave and bell, the register's ka-ching, a coin's clink after
  each word, the reels running down when it is lost, a brass chord, a
  riser, the shimmer pad, glitter, the payout, the multiplier called;
- a phrase's chime reaching it (`Chime`): plucks up the chord, a short
  brass hit or a "ta-daa", a choir.

How big a sound is beyond its layers follows from its row's share of the
way to the top: more notes and longer, never only louder. So a new row
sounds right with no new code.

### Made of atoms

Every sound is notes of atoms (`sfx/conductor/palette.rs`): the pop, its
body and glint, three coins, the zap, ticks and pips, a pluck, bells,
brass, a choir's "aah", pads, glitter, a riser, the register's clack and
ching, the slam's drums, and the diploma's paper, pen and stamps. An atom
is one synth patch. Their levels set them against each other; the synth's
bus (compressor, then a limiter at -1 dBFS) lifts them together and keeps
a pile of them from clipping. The bus follows each song's loudness as the
karaoke plays it (`sfx/conductor/calibration.rs`), so the sounds sit the
same over a loud song and a quiet one; the Effects strip's fader is the
user's on top.

### The craft behind it

The choices come from how reward sounds are made in games and slot
machines, without the deceit:

- the size of a sound follows the size of what it celebrates, and only
  real achievements get one;
- a scale in the song's key climbs with the streak, and a rung resolves
  on the tonic;
- sounds are stacked in time (a transient, the body, sparkle after it),
  not only in loudness;
- the big ones are quantised to the beat, with a pickup at once so the
  feedback stays instant;
- a riser leads into a rung that is one word away;
- the small sounds stay soft and varied (round-robin, a little different
  each time, fewer when they come thick), so the rare ones have room;
- the diploma's count-up is short and quickening, and ends on a full
  chord: the peak and the end are what is remembered.

## Themes: `player/src/theme.rs`

A theme is one data file under `player/src/theme/`: how the tiers, their
effects and their sounds look and sound, and how the picture is graded.
What it overrides (a tier's colour, the count-in's accent, an atom's
patch) it may leave out, and that is as the tier's row or the palette has
it; how it dresses the effects it always says. Everything that has a
colour, a dress or a patch a theme may change reads it through the theme
in force; the setting `karaoke_theme` in the settings file picks it
(`"classic"`, the default, or `"spooky"`).

| Field of `Theme` | What it says | Read through |
|---|---|---|
| `colors` | the tiers' colours, one a tier, Good first (their badge, verdict, marks, stickers, the slam) | `Theme::color` |
| `accent` | the count-in's accent | `Theme::accent` |
| `fire` | the fire's colours (`Ramp`: the screen's edges, the lane's flames, the line sung) | the shaders' `theme.fire` |
| `bulbs`, `bulb` | the marquee's bulbs' colours (`Palette`) and what a bulb is (`BulbShape`: a bulb, a jack-o'-lantern) | the shaders' `theme.bulbs`, `theme.bulb` |
| `coins` | the colours the coins come in (none: gold) | `fx::coins::candied` |
| `angels` | what comes down at the top (`Flight`: angels, ghosts) | `fx::angels` |
| `sounds` | the synth's patch for one of the sounds' atoms | `Theme::patch` |
| `always` | effects held throughout, whatever the singing (screen passes only) | `Theme::holds` |
| `grade` | the backdrop's grade (`Grade`: the tint its colours take once desaturated, and how much) | the backdrop's shader |

The shaders read a theme's dress from one uniform (`Theme::uniform`, its
WGSL `struct Theme` generated by `theme::wgsl`), apart from the effects'
looks (`Spec::params`, which the music moves): no mod of a look changes a
theme.

- **Classic** (`theme::CLASSIC`, `theme/classic.rs`) is the game as made.
  With it the stage's digest, the frame_cost scenes and every conductor
  sound are the same as before themes, to the byte.
- **Spooky** (`theme/spooky.rs`) is its Halloween. The tiers are slime
  green, pumpkin, witch's purple, a ghostly mint and moonlight, and the
  count-in is pumpkin. The fire burns green (the screen's edges, the
  lane's flames, the line sung). The marquee's bulbs are jack-o'-lanterns,
  their candles lit as the chase runs; the coins are candy, the angels
  ghosts. The backdrop is graded cold, its gold rays turned to moonlight,
  and the fog (`fx::Fog`) creeps in at the edges with eyes blinking in it.
  The pops bubble like a cauldron over a skeleton's xylophone, the clinks
  are sweets in a jar, the chimes toll as bells, the choir sings a
  ghost's "ooh", and a streak lost wails down like a theremin.

A theme that needs a dress an effect cannot wear yet adds it to `Theme`
as a typed field (and to its uniform where the shaders read it), every
theme saying its own.

### Adding a theme

1. A `static` `Theme` in a module of its own under `player/src/theme/`,
   with its name, its dress, and the overrides it makes.
2. Its place in `theme::THEMES`. The setting reads it by its name.
3. `every_theme_overrides_what_there_is_once` checks that each atom is
   overridden once and that what it holds throughout is a screen pass. A
   test of its own (as `spooky_is_its_own`) says what makes it other than
   classic.
4. Look at it with `examples/frame_cost.rs --theme NAME` (it must stay
   within the frame budget) and hear it with `examples/conductor.rs
   OUT_DIR --theme NAME`.

## Not built yet

A sung announcer (the slam's words called out, sung in the song's key)
is on the later list ([todo.md](todo.md)), and so is choosing the theme
in the window.
