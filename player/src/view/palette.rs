//! The player's colours, as plain RGB for every way it is drawn: the
//! window's own widgets and pixels read them as they are, the cell widgets
//! make their cell colours from them.

/// The window background.
pub const BG: [u8; 3] = [17, 17, 21];
/// Highlights and the play head: the Okabe–Ito yellow, apart from the
/// drums' orange.
pub const ACCENT: [u8; 3] = [240, 228, 66];
/// Secondary text and borders.
pub const DIM: [u8; 3] = [120, 120, 132];
/// Text.
pub const TEXT: [u8; 3] = [222, 222, 228];
/// Borders.
pub const BORDER: [u8; 3] = [64, 64, 74];
/// The selected row.
pub const SELECTED: [u8; 3] = [46, 46, 58];
/// Buttons.
pub const BUTTON: [u8; 3] = [38, 38, 47];
/// Good news (done, chosen).
pub const GREEN: [u8; 3] = [46, 204, 113];
/// Warnings.
pub const WARN: [u8; 3] = [240, 160, 60];
/// Errors.
pub const FAIL: [u8; 3] = [235, 80, 80];
/// The file's rate, and the spectrum's curve.
pub const SOURCE: [u8; 3] = [150, 160, 200];
/// Folders.
pub const FOLDER: [u8; 3] = [130, 180, 255];
/// The unlit part of a level and a slider in cells.
pub const TRACK: [u8; 3] = [60, 60, 70];
