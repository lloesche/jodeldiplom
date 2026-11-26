# The window on macOS

What the window does on macOS that it does not do elsewhere, and why. Most
of it was found by trying things on a Mac and measuring, so each point
says what was seen, what was tried and what stayed.

## Karaoke mode covers the screen; it is not a fullscreen Space

**Seen.** In a fullscreen Space, Karaoke mode showed at 30 frames a second
on a 60 Hz display while the windowed tabs reached 60, with the GPU at
2 ms a frame. The words in the bubbles smeared to the eye.

**Tried.** A switch made Karaoke mode a frameless window the size of the
screen instead of a Space, to tell whether the Space was the cause. It
ran at the display's refresh.

**Kept.** `player/src/gui/cover.rs`: entering Karaoke mode the window
covers its screen as fullscreen was before Spaces.

- The title bar goes: the style loses `Titled`, `Resizable` and
  `Miniaturizable`.
- The window takes the screen's whole frame and cannot be moved.
- The menu bar and the Dock hide (the application's presentation
  options).
- Leaving, the style, the frame, the presentation options and the window's
  moving all come back as they were (kept in a thread-local while it
  covers the screen).

It runs as a task on iced's window (`iced::window::run`), on the main
thread, as AppKit needs. On Windows and Linux the window goes fullscreen
through iced (`window::Mode::Fullscreen`).

## The keyboard after the style changes

**Seen.** After entering or leaving Karaoke mode the keys went nowhere: a
window whose style mask changed is no longer the key window.

**Kept.** After each change the window is made key and ordered front
again, and its view made the first responder (`makeKeyAndOrderFront`,
`makeFirstResponder`).

## The frame tick, and the Metal layer that was not the cause

**Seen.** Karaoke mode ran at about 34 uneven frames a second on a 60 Hz
display, the words smeared.

**Tried, in turn:**

1. The Metal layer's drawables. iced configures its surface for the least
   latency; on Metal that looked like a layer of two drawables (one on
   screen, one being drawn), where a frame not through by the refresh
   waits a whole refresh more. Not waiting for the display at all ran at
   60 but tore. The window put three drawables back on every redraw,
   after iced had configured the surface.
2. The window's view, to find its layer under, was asked for once the
   window was known: its opening can come before the window listens, so
   the view had never been found. The trace then said what the layer
   had.
3. A translucent layer (composited, never shown directly by the display),
   behind a switch, to see whether showing frames directly was the cause.

The trace showed the layer had three drawables already. The layer code
went.

**The cause.** The tick: iced's frame subscription hands each redraw on
after it, so the tick for a frame lands after the frame, and on a 60 Hz
display missed every other refresh. Now a widget round everything the
window shows sends the tick from inside each redraw, before it is drawn
(`player/src/gui/pacer.rs`). The fullscreen half of the 30 frames was the
Space (above).

**Seen too.** Redraws that do not wait for the display came thousands a
second while the window went in or out of fullscreen. Two ticks are at
least 4 ms apart.

## The Dock icon

macOS has no window icons, and a bare executable (not in the app bundle,
whose `Info.plist` names the icon file) shows a generic one in the Dock.
The window sets the application's icon image (drawn by
`player/src/icon.rs` in Apple's style) on the main thread once the
window has opened. The app bundle (`player/macos/make-app.sh`) carries
`assets/jodeldiplom.icns`.

## Small things

- **Cmd+V pastes,** as Ctrl+V does elsewhere (`player/src/gui/keys.rs`).
- **The folder to start in:** Finder starts an app in `/`, so a player
  run from its bundle starts in `~/Music` (else the home folder) when no
  folder is given (`cli::default_folder`).
- **The microphone:** without `NSMicrophoneUsageDescription` in
  `Info.plist`, macOS never asks and the microphone gives only exact
  zeros. A microphone that gives nothing but zeros for a while is said on
  the Karaoke tab, with where to allow it.
- **The volume keys:** the system's volume keys do not reach the device
  while the player plays on it, so the player has its own (`alt+up`,
  `alt+down`, the device's volume a step).
- **The app is not notarised:** after the first attempt to open it, allow
  it under System Settings, Privacy & Security (Open Anyway).

## Beside the window: the GPU and the sound

Learnt on the same Macs, and kept in the code:

- **Apple's GPUs run a shader at the pace of its dearest code, for every
  pixel.** One stage shader for every shape cost about 4 ms a screen
  whatever was drawn (11 ms on the diploma, on an M4 Max, against 0.1 to
  0.3 ms on an RTX 5080). The stage is now a pipeline a group of shapes,
  each compiled with only its own code (`gui/stage/gpu/groups.rs`), with
  the same frames to the byte.
- **Metal writes timestamps only at a pass's start and end,** and Apple's
  GPUs start a pass that needs nothing before it early, beside what is
  still drawing. `frame_cost` marks each step with a pass of its own,
  clearing one pixel, once the GPU has done the step
  (`gui/present/stamps.rs`).
- **A sample on a seam can fall either side.** Apple's GPUs put the meter
  ring's sample on its level line and the sticker's at its middle, the
  other side from the Rust reference; the shader checks hold a point
  turned about another within the sine's rounding, and take no sample at
  the ring's centre, where the angle is undefined
  (`gui/shader_check.rs`).
- **Retina screens:** the light that blooms is drawn at a texel a point
  (half each way on a Retina screen) and the backdrop at the share of the
  window that still gives each of the cover's texels a pixel; at 3024 ×
  1964 a frame costs a third less GPU time.
- **The mixer's thread is real time.** As an ordinary thread it could be
  held up behind a busy window and the sound stuttered. While it does
  audio work it runs under Mach's time-constraint policy, as Core Audio's
  own threads do, which also keeps it on a performance core in Low Power
  Mode (`output/src/backend/priority.rs`).
