# The window

How the player's window is built: an iced application around the pure
core, its screens made of iced's widgets in the native look
([look.md](look.md)), and Karaoke mode drawn by a GPU stage of our own.
The window began as a terminal's cells rasterised into pixels; the screens
moved to iced one at a time, and the cells are gone.

## iced, pinned

The window is iced's development version, pinned to one commit
(`98bd48f`, iced 0.15.0-dev) because it is on the same `wgpu` (30) as
`jodel-gpu`: the player has one GPU stack, the window's frames and the
networks on one device and one queue. An upgrade of iced is an upgrade of
`wgpu` for every compute kernel too, so it is done on purpose, in a window
of its own. `iced_test` is taken from the same commit, for the headless
tests and examples.

Its features: the `wgpu` renderer without `wgpu`'s own backends (ours pick
them: Vulkan on Windows and Linux, Metal on the Mac, the backends the
models run on too), the CPU renderer (`vello-cpu`) where there is no
GPU, `canvas`, images without codecs (ours decode), `advanced` for our
own widgets. `JODELDIPLOM_SOFTWARE=1` runs on the CPU renderer as a
machine without a GPU would.

## The pieces

| Module | What it is |
|---|---|
| `gui::window` | the application: `Gui` (its state), `Message`, `update`, `view`, `subscription`; the phases (setup, running, leaving) |
| `gui::screens` | what shows, screen by screen, as iced elements |
| `gui::look`, `gui::widgets` | the look's tokens and the shared widgets every screen is built from |
| `gui::keys`, `gui::pointer` | the keyboard as the core's keys; what a press holds (a grab) and the wheel |
| `gui::pacer`, `gui::paint`, `gui::wake` | when the frames come and what each draws; the player's threads waking the window |
| `gui::mixer` | the Mixer tab: the strips laid out in points, the faders and meters as canvases |
| `gui::lanes` | the Transition and Karaoke tabs' lanes, painted in tiles and scrolled on the GPU |
| `gui::stage`, `gui::stage::gpu`, `gui::stage::shapes`, `gui::fx`, `gui::backdrop`, `gui::bloom`, `gui::particles`, `gui::present` | Karaoke mode on the GPU |
| `gui::cover` | macOS: Karaoke mode covering the screen ([macos-window.md](macos-window.md)) |

## Messages

The core is the logic; the window turns what happens into its actions.

- **Keys** go through `gui::keys::pressed` into the core's `Key`, and
  `core::keymap::action` says what they do in the state now. The clock
  goes in first (`Action::Tick`): what a key means can depend on the time.
- **Widgets** send the core's actions themselves (`Message::Act`), or a
  command and its shifted form (`Message::OpShifted`: Enter Karaoke Mode,
  shift for where the song is).
- **A press that holds the pointer** is a grab (`Message::Grab`,
  `gui::pointer::Grab`): the seek bar, a queue row, a Transition block's
  lanes or its view, a slider. While it is held the window listens to
  every move, anywhere, and the grab turns each into the core's actions
  (a share across a track, a point in a lane); the release lands it.
  Every button's release comes to the window, whichever widget it was
  over, so nothing stays held.
- **The wheel** goes through one accumulator (`gui::pointer::Wheel`), so a
  touchpad's pixels add up to whole steps: three rows a step for the
  lists, a step a notch for a slider or a zoom (`Message::Steps`), a fader
  on the Mixer tab (`Message::DeskWheel`).
- **The player's threads** (the engine, the workers, setup) call
  `Wake::notify`; the window takes one `Message::Wake` a burst and pumps
  the app.
- **The window's own events:** opened, resized, rescaled, closing, and
  unfocused (Ctrl or the demo key let go of in another window is never
  heard, so it is let go of here).

## The core keeps no geometry

The core knows nothing of points or cells. The window measures and
reports the facts of its size that change what a key does, as a
`core::Fit` (`Action::Fit`, sent when it changes): how many rows a page
of the list or the help is, how wide the Mixer's desk is (which strips
show, so the arrow keys go round the same ones), how many letters fit a
Transition lane (the fit-lyrics zoom; a letter is as wide as the main
font's M, `gui::lanes::Metrics::at`), how near the bottom the pointer
brings Karaoke mode's controls up, whether a spectrum shows. The rules
read the numbers.

## When the frames come

One decision a tick (`gui::paint::Paint::plan`): how often the ticks come
and what is drawn now.

- **At the display's refresh** while something moves smoothly (a lane
  following the playhead, a fader gliding, Karaoke mode). The tick comes
  from inside each redraw (`gui::pacer`: a widget round everything that
  sends it before the redraw is drawn and asks for the next), so the
  frame shown is the one ticked for it. iced's own frame subscription
  hands the redraw on after it, which on a 60 Hz display missed every
  other refresh. Two ticks are at least 4 ms apart: redraws that do not
  wait for the display (a window going in or out of fullscreen) came
  thousands a second.
- **30 a second** while only the spectrum and the clock move.
- **Four a second at rest** (`REST`), whatever else: the engine's and the
  workers' news taken in, what is due done.
- **Nothing** while the window is covered: when the refresh's ticks stop
  coming, the window draws nothing until they come again.

Background GPU work gives way while the window draws at the refresh, and
while its ticks come late, the window starved of the GPU
(`jodel_gpu::pace::tick`).

## What shows

`gui::screens::view` builds the window's content from the state on every
message:

- **The setup screen** from the first frame of every start, while setup
  chooses the graphics card, finds (or gets) the player's files and starts
  the player in the background (`cli::start_setup`); the player shows once
  it has started. The graphics card is chosen once the window has opened:
  set up at the same time, the window's own waited for it.
- **Whole screens** in iced's widgets: Now Playing, Settings and Help (the
  last two over the bottom bar), with the command palette over them.
- **The main screen:** the list pane with its tabs (the browser, the
  queue, the Transition tab, the Karaoke tab, the Mixer tab), the right
  column (the cover, the track's links, the signal path) or the summary
  under the list, the bottom bar (transport, seek bar, spectrum), the
  Karaoke tab's sync panel, and the command palette over all of it. They
  sit in one `stack!` in a fixed order, each slot an element or nothing,
  so each widget keeps its place in iced's tree (the tabs' pill slides
  from one tab to the next).
- **Karaoke mode:** the whole window, drawn by the GPU (below).

Each screen's content comes from a model in `view/` (what it says: rows,
labels, tones, hints from the keymap), and its view in `gui/screens/`
maps the model to widgets. Models are plain data, tested as text; views
are tested in iced's simulator.

### Text

iced's text for the whole application, in Nunito (`look::font`), the
player's other built-in fonts loaded beside it (Sarasa Fixed J for
Japanese, Chinese and Korean, Noto Emoji). The words along the lanes and
Karaoke mode's text are drawn by our own glyph rasteriser
(`gui::big_text`, ab_glyph) into the lanes' tiles (in Atkinson
Hyperlegible Next) or the stage's atlas (in Karaoke mode's own faces:
Nunito, Fredoka, M PLUS Rounded 1c).

### Pictures

- **Covers:** iced images, each art scaled once to the pixels it shows at
  and cached by its id (`gui::widgets::cover`), corners rounded, a shade
  under text over them.
- **The spectrum:** a canvas from the model's plot (`gui::widgets::spectrum`),
  which both renderers draw.
- **Lanes:** each lane is painted in tiles once (`gui::lanes::tiles`), and
  a shader widget scrolls the tiles on the GPU (`gui::lanes::gpu`: one
  pipeline for every lane, a texture array a lane, only the new tiles
  sent each frame); on the CPU renderer an image of the part showing.

## Karaoke mode on the GPU

With a GPU, Karaoke mode is one surface drawn in iced's render pass, in
this order (`gui::present::Painter::PARTS`):

1. **ground:** the backdrop (`gui::backdrop`): the song's cover blurred,
   or with the game its story's scenes moving with the music, a
   triangle over the window reading it;
2. **stage:** the stage's shapes (`gui::stage`, `gui::stage::gpu`): the lane,
   the bubbles as tubes with their liquid, the arrow, the lyrics, the
   count-in, the score, the meter, the badge, Next, the result's
   diploma;
3. **bursts:** the particles (`gui::particles`);
4. **composite:** the light that blooms (`gui::bloom`), drawn apart at a
   texel a point and added in;
5. **fx:** the effects' screen passes (`gui::fx`), in strips;
6. **top:** the shapes over everything (the slam's big words).

The stage is instances of a few shapes, each a signed distance in the
fragment shader, or text from a glyph atlas. It is drawn by one pipeline
a group of shapes (`gui::stage::gpu::groups`), each compiled with only its
group's code: Apple's GPUs run a shader at the pace of its dearest code
for every pixel, so one shader for all of them made a plain pill as dear
as the diploma's paper. The paper, once it holds still, is baked into a
texture and read back. `PERFORMANCE.md` holds the rules and the budgets.

The controls' bar (the voice slider, the seek bar, the times, the keys)
comes up only with the pointer near the bottom and fades
(`core::karaoke::controls`). The pointer hides when it has been still a
while, as in a video player (`core::karaoke::pointer`). Karaoke mode makes
the window fullscreen; on macOS the window covers the screen itself
([macos-window.md](macos-window.md)).

Karaoke needs a GPU: without one the core refuses it
(`core::karaoke::not_singable`), and a window drawn on the CPU renderer
shows only a line saying so in its place
(`gui::screens::karaoke::mode::NO_STAGE`).

## Checking a change to the window

- `cargo test -p jodeldiplom`: the core's rules, the models' text, the
  simulator's tests of the screens (`player/tests/screens.rs`), and
  `player/tests/interactions.rs`, which checks every test
  `INTERACTIONS.md` names exists.
- `examples/shots.rs`: the screens as the window draws them, headless,
  with a fixture track playing, each with its model's text. Look at them
  before and after; nothing lost, and the new one at least as clean.
- `examples/look_sheet.rs`: every token and widget on one sheet.
- `examples/view_cost.rs`: what building and laying out the view costs a
  frame (it is built again every tick at the refresh).
- `examples/frame_cost.rs`: Karaoke mode's GPU steps against the budget.
- The real window with `JODELDIPLOM_TRACE=FILE`: a line every 5 s on the
  frames (how many, how many late, the longest gap, how long drawing
  took).
- `--exit-after SECONDS --save-frame FILE.png` saves the window's last
  frame.
