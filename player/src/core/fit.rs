//! What the window's size does to the keys and the pointer (the layout
//! metrics), as the window
//! reports it ([`crate::core::Action::Fit`]): how many rows a page is, which
//! of the desk's strips are shown, how wide a lane is. The core keeps no
//! geometry of its own; the window measures, the rules read the numbers.

/// The facts of the window's size that change what a key or the pointer
/// does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fit {
    /// The rows the browser's and the queue's list shows, the filter line's
    /// row included (it takes the last while the browser is filtered):
    /// the page step, and the rows kept in view.
    pub list_rows: u16,
    /// The rows the help's text shows, the search line's row included (it
    /// takes the last while the help is searched).
    pub help_rows: u16,
    /// The Mixer tab's desk's width, in points: which of its strips are
    /// shown ([`crate::core::desk::strips_shown`]).
    pub desk_width: f32,
    /// The width of the Transition tab's lanes, in letters of the words
    /// along them, a cell wide each (0: none shown): what fits the
    /// lyrics' words.
    pub lane_letters: u16,
    /// The least share of the height, from the bottom, where the pointer
    /// brings Karaoke mode's controls up: the controls' bar and two lines.
    pub controls_reach: f32,
    /// Whether the screen shows a spectrum (it is computed only then).
    pub spectrum: bool,
}
