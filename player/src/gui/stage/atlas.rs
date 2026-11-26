//! The stage's text kept in its atlas on the GPU ([`Atlas`]), shelf by
//! shelf, each run of text (its face, text and size) once.

use std::collections::HashMap;

use super::super::big_text::{BigText, Face};
use super::gpu::{AtlasUpload, ATLAS};

/// Text in a face at a size (a quarter pixel's steps).
type Key = (Face, String, u32);

/// Text kept in the atlas, shelf by shelf: where each (face, text, size)
/// is, and what is new to it since the frame before.
#[derive(Default)]
pub(super) struct Atlas {
    places: HashMap<Key, Placed>,
    /// Where the next goes: across, the shelf's top, the shelf's height.
    pub(super) pen: (u32, u32, u32),
    /// The text to put into the atlas on the GPU.
    pub(super) uploads: Vec<AtlasUpload>,
}

impl Atlas {
    /// Where `text` at `px` in `big`'s face is in the atlas (put there now
    /// if it is not): its place, and the size it was made at as a share of
    /// `px`. Text wider than the atlas (a long lyric line on a large
    /// screen) is made smaller to fit, its quad drawn larger by as much.
    pub(super) fn place(&mut self, big: &mut BigText, text: &str, px: f32) -> Option<Placed> {
        self.place_or(big, text, px, true)
    }

    /// As [`Atlas::place`], but none where the atlas is too full for it:
    /// never started again, so the places of the text already put in this
    /// frame keep (an effect's big word waits for room).
    pub(super) fn place_if_room(
        &mut self,
        big: &mut BigText,
        text: &str,
        px: f32,
    ) -> Option<Placed> {
        self.place_or(big, text, px, false)
    }

    /// [`Atlas::place`], starting the atlas again where it is full if
    /// `again`, else none.
    fn place_or(&mut self, big: &mut BigText, text: &str, px: f32, again: bool) -> Option<Placed> {
        let key = (big.face(), text.to_owned(), (px * 4.0).round() as u32);
        if let Some(&place) = self.places.get(&key) {
            return Some(place);
        }
        let (mut w, mut h, mut coverage) = big.coverage(text, px);
        let mut made = 1.0;
        if w.max(h) > ATLAS as usize {
            made = (ATLAS - 8) as f32 / w.max(h) as f32;
            (w, h, coverage) = big.coverage(text, px * made);
        }
        let (w, h) = (w as u32, h as u32);
        if w > ATLAS || h > ATLAS {
            return None;
        }
        let next_shelf = self.pen.0 + w > ATLAS;
        let pen_y = if next_shelf {
            self.pen.1 + self.pen.2 + 2
        } else {
            self.pen.1
        };
        if pen_y + h > ATLAS && !again {
            return None;
        }
        if next_shelf {
            self.pen = (0, pen_y, 0);
        }
        if self.pen.1 + h > ATLAS {
            // Full: started again (what shows is put back as it is asked
            // for).
            self.places.clear();
            self.pen = (0, 0, 0);
        }
        let rect = (self.pen.0, self.pen.1, w, h);
        self.pen.0 += w + 2;
        self.pen.2 = self.pen.2.max(h);
        let place = Placed { rect, made };
        self.places.insert(key, place);
        self.uploads.push(AtlasUpload { rect, coverage });
        Some(place)
    }
}

/// Text's place in the atlas: across, down, wide, high (atlas pixels), and
/// the size it was made at as a share of the size asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Placed {
    pub(super) rect: (u32, u32, u32, u32),
    pub(super) made: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_atlas_keeps_text_once_and_starts_again_when_full() {
        let mut big = BigText::default();
        let mut atlas = Atlas::default();
        let a = atlas.place(&mut big, "hello", 40.0).unwrap();
        assert_eq!(atlas.place(&mut big, "hello", 40.0), Some(a));
        assert_eq!(atlas.uploads.len(), 1);
        assert_eq!(a.made, 1.0);
        let b = atlas.place(&mut big, "world", 40.0).unwrap().rect;
        assert!(b.0 >= a.rect.0 + a.rect.2);
        // Full: started again from the top.
        atlas.pen = (0, ATLAS - 2, 0);
        let c = atlas.place(&mut big, "again", 40.0).unwrap().rect;
        assert_eq!((c.0, c.1), (0, 0));
        let gone = (Face::Main, "hello".to_owned(), 160);
        assert!(!atlas.places.contains_key(&gone));
    }

    /// A lyric line wider than the atlas at its size (a large screen's)
    /// is made smaller to fit, not left out.
    #[test]
    fn text_wider_than_the_atlas_is_made_smaller() {
        let mut big = BigText::default();
        let mut atlas = Atlas::default();
        let line = "Makes you feel like every corner is black and white tonight";
        let p = atlas.place(&mut big, line, 120.0).unwrap();
        assert!(p.made < 1.0, "{p:?}");
        assert!(p.rect.2 <= ATLAS && p.rect.2 > ATLAS / 2, "{p:?}");
    }
}
