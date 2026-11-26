# The native look

Every screen drawn in iced's widgets takes its look from one place: the
tokens in `player/src/gui/look.rs` and the shared widgets in
`player/src/gui/widgets/`. A screen never has numbers of its own; a widget
it needs and does not find is added there, made for every screen.

The look replaced the terminal's cells with the same player, nothing lost:
every key, click, double click, drag, wheel and hover still works, every
piece of information is still shown, and `player/INTERACTIONS.md` is still
the contract.

## Principles

- **Calm and dark, one accent.** The music and the covers are the colour.
  The chrome is layered dark surfaces, and one accent (the Okabe–Ito
  yellow) marks what is active.
- **Proportional type**, with figures all one width where numbers change
  (times, rates, counts). Monospace only where alignment is the point.
- **Space over lines.** Grouping comes from spacing and surfaces, not
  box-drawing borders. An edge only where a surface needs one: 1 point,
  low contrast.
- **Soft geometry.** Rounded corners: 6 points for controls, 10 for panels,
  pills for chips, toggles and tracks.
- **Motion that explains.** Hover and selection ease in over 120 ms,
  panels and overlays over 160 ms. Nothing animates for its own sake, and
  a calm screen still rests (30 frames a second while only the spectrum
  moves, otherwise none).
- **Readable across the room** for Karaoke, comfortable at a desk for
  browsing. Sizes come from the type scale.

## Tokens: `gui::look`

The colours start from the player's palette (`view::palette`), so it stays
recognisable.

| Token | Value | Use |
|---|---|---|
| `BG` | #111115 | the window's ground |
| `SURFACE` | #17171d | panels, the list pane, the bottom bar |
| `RAISED` | #1f1f27 | cards, menus, the palette overlay, hovered rows |
| `SELECTED` | #2a2a36 | the selected row (with the accent's bar at its left) |
| `EDGE` | #2c2c36 | the 1-point edges that remain, the empty part of a track |
| `TEXT` | #dedee4 | primary text |
| `TEXT_DIM` | #9a9aa8 | secondary text, labels |
| `TEXT_FAINT` | #6e6e7c | hints, what cannot be used now |
| `ACCENT` | #f0e442 | the play head, the active tab, the focused control, the selection bar |
| `ACCENT_INK` | #16160f | text on the accent |
| `GOOD`, `WARN`, `FAIL` | #2ecc71, #f0a03c, #eb5050 | status |
| `FOLDER` | #82b4ff | folders in the browser |
| `SOURCE` | #96a0c8 | the file's rate, the spectrum's curve |

`look::alpha` gives a colour at an opacity; `look::tint` mixes a tone
over a ground as the eye sees it, so a faint tint stays faint.

| Scale | Values (points) |
|---|---|
| Spacing (`space`) | 4 within a group, 8 between controls, 12 inside a row or card, 16 a panel's padding, 24 between a panel's groups, 32 round a whole screen |
| Radii (`radius`) | 6 controls, 10 panels and cards, 999 pills |
| Type (`size`) | 12 caption, 13 label (semi-bold, dim), 14 body, 18 title (bold), 28 display (bold: Now Playing's title) |
| Heights (`height`) | 34 a list row, 28 a dense row, 34 a button, 26 a chip, 24 a small button, 6 a progress bar, 4 a track, 16 an icon |
| Lines (`line`) | 1 an edge, 2 the focus ring |
| Motion (`motion`) | 120 ms hover and selection, 160 ms panels and overlays |

The face is Nunito, built in (`look::font`): medium for body text,
semi-bold for labels, buttons and numbers, bold for titles. Its figures
are all one width, so numbers that change do not jitter. Without the
built-in fonts (some tests) it is iced's sans-serif.

## Shared widgets: `gui::widgets`

- **Buttons** (`button`): `primary` (the accent's fill: the one to take),
  `secondary` (raised), `ghost` (text alone, raised under the pointer),
  `icon` (a square ghost round an icon), `round` (a disc in the accent
  round an icon), `chip` (a small pill: a switch, struck through while
  off). Each has hover, pressed, focused (a 2-point accent ring) and
  disabled states; a label can lead with an icon and name its key.
- **Surfaces** (`panel`): a panel (the list pane, a screen's body), a card
  on it, a tinted note for what needs attention, a status disc, tags on
  pills, a key's cap.
- **Rows** (`rows`): a list's row, an icon, a title and its details at the
  right, raised under the pointer, selected with the accent's bar at its
  left.
- **Items** (`item`): a card's rows that take a click (a setting, a device,
  a folder) and segmented controls.
- **Switches and checks** (`switch`): a pill with its knob, a box or a
  disc, lit in the accent while on; the row they are on takes the click.
- **Tabs** (`tabs`): a segmented control whose accent pill slides to the
  tab shown; a tab can carry a count.
- **Track** (`track`): the seek bar's or a slider's line, thin, filled in
  the accent, a thumb under the pointer or while held; a press is a grab.
- **Progress** (`progress`), **spectrum** (`spectrum`, a canvas both
  renderers draw), **cover** (`cover`, rounded, a soft shadow, a shade
  under text over it), **link** (`link`, underlined under the pointer).
- **Key hints** (`hints`), **keeping what the keys are on in view**
  (`reveal`), and **an area that takes the pointer** over a picture drawn
  by something else (`area`, the Transition tab's lanes).
- **Icons** (`icons`): vector paths in one module, one stroke weight
  (two units of a 24-unit square), round ends, sized from the type scale,
  in the colour of the text round them: check, cross, retry, download,
  alert, play, pause, previous, next, stop, settings, help, folder, note,
  playlist, queue, up, chevron, search, command. No icon fonts and no
  emoji for the chrome.

## Checking a screen

- `cargo run -p jodeldiplom --example look_sheet` draws every token and
  widget, in every state, on one sheet, as the window draws.
- `cargo run -p jodeldiplom --example shots` draws the screens as the
  window does, headless, at 1280 × 800 at scale 1 and 1512 × 982 at scale
  2 (a laptop's), each with its model's text: nothing lost, and the new
  one at least as clean.
- `examples/view_cost.rs` keeps the view's cost a frame in view: it is
  built again on every tick at the display's refresh.
