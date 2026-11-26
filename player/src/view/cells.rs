//! The cell renderer's measure of text: its width in cells.

use unicode_width::UnicodeWidthStr;

/// The width of a text in cells.
pub fn width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}
