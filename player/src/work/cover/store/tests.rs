//! Each cover is taken apart once: the same art in two files, the same art
//! encoded again in one album; but two covers, and two covers of an album
//! that only look alike, are each their own.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::work::art::sample_cover;
use crate::work::cover::NearMap;

/// Takes a cover apart by copying it, counting.
#[derive(Default)]
pub(crate) struct Counting {
    pub(crate) calls: AtomicUsize,
}

impl Splitter for Counting {
    fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let near = NearMap {
            width: 4,
            height: 4,
            near: vec![100; 16],
        };
        Ok(CoverLayers {
            plate: image.clone(),
            subject: None,
            depth: near.clone(),
            plate_depth: near,
            outline: GrayImage::new(4, 4),
            palette: vec![Swatch {
                rgb: [1, 2, 3],
                share: 1.0,
            }],
        })
    }
}

fn store(tag: &str) -> CoverStore {
    let dir = std::env::temp_dir().join(format!("jodel-covers-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    CoverStore::new(dir)
}

fn song(name: &str, album: &str, image: Option<RgbImage>) -> SongCover {
    SongCover {
        song: name.to_owned(),
        albums: vec![album.to_owned()],
        image,
    }
}

/// `image` through JPEG at `quality` and back, at `side`.
fn reencoded(image: &RgbImage, quality: u8, side: u32) -> RgbImage {
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality)
        .encode_image(image)
        .unwrap();
    let back = image::load_from_memory(&jpeg).unwrap().to_rgb8();
    image::imageops::resize(&back, side, side, FilterType::Triangle)
}

#[test]
fn the_same_art_in_two_files_is_taken_apart_once() {
    let s = store("same");
    let c = Counting::default();
    let art = sample_cover(96);
    // Two albums even (a compilation's track and the album's).
    let a = s
        .layers(&song("one", "A", Some(art.clone())), &c)
        .unwrap()
        .unwrap();
    let b = s.layers(&song("two", "B", Some(art)), &c).unwrap().unwrap();
    assert_eq!(c.calls.load(Ordering::SeqCst), 1);
    assert_eq!(a, b);
    assert!(s.ready("one") && s.ready("two"));
    assert_eq!(s.linked("one"), s.linked("two"));
    // Ready at once from the song's entry: no image needed.
    let again = s.layers(&song("two", "B", None), &c).unwrap();
    assert_eq!(again.as_deref(), Some(&*a));
    assert_eq!(c.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn the_same_art_encoded_again_in_one_album_is_reused() {
    let s = store("again");
    let c = Counting::default();
    let art = sample_cover(256);
    let copy = reencoded(&art, 55, 200);
    assert_ne!(pixel_hash(&art), pixel_hash(&copy));
    s.layers(&song("one", "A", Some(art)), &c).unwrap();
    s.layers(&song("two", "A", Some(copy.clone())), &c).unwrap();
    assert_eq!(c.calls.load(Ordering::SeqCst), 1);
    assert_eq!(s.linked("one"), s.linked("two"));
    // In another album it is a cover of its own.
    s.layers(&song("three", "B", Some(copy)), &c).unwrap();
    assert_eq!(c.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn covers_of_one_album_that_only_look_alike_are_not_shared() {
    // A templated single: the same artwork, another title in a band across
    // it, and a deluxe edition's cover with a brighter corner.
    let s = store("alike");
    let c = Counting::default();
    let art = sample_cover(256);
    let mut single = art.clone();
    for y in 180..230 {
        for x in 20..236 {
            single.put_pixel(x, y, image::Rgb([240, 230, 40]));
        }
    }
    let mut deluxe = art.clone();
    for y in 0..96 {
        for x in 0..96 {
            let p = deluxe.get_pixel_mut(x, y);
            p.0 = p.0.map(|v| v.saturating_add(70));
        }
    }
    for other in [&single, &deluxe] {
        assert!(!same_picture(&thumb(&art), &thumb(other)));
    }
    let a_copy = reencoded(&art, 60, 200);
    assert!(same_picture(&thumb(&art), &thumb(&a_copy)));
    s.layers(&song("album", "A", Some(art)), &c).unwrap();
    // Even where the album's entry says the perceptual hashes agree (as
    // they may for covers this alike), the pixels decide.
    let Some(Some(hash)) = s.linked("album") else {
        panic!("linked");
    };
    for (name, other) in [("single", single), ("deluxe", deluxe)] {
        s.add_to_album("A", &hash, &perceptual(&other)).unwrap();
        s.layers(&song(name, "A", Some(other)), &c).unwrap();
    }
    assert_eq!(c.calls.load(Ordering::SeqCst), 3);
    assert_ne!(s.linked("album"), s.linked("single"));
    assert_ne!(s.linked("album"), s.linked("deluxe"));
}

#[test]
fn two_covers_are_taken_apart_twice_and_no_cover_is_no_work() {
    let s = store("two");
    let c = Counting::default();
    let other = image::imageops::rotate90(&sample_cover(96));
    s.layers(&song("one", "A", Some(sample_cover(96))), &c)
        .unwrap();
    s.layers(&song("two", "A", Some(other)), &c).unwrap();
    assert_eq!(c.calls.load(Ordering::SeqCst), 2);
    assert_ne!(s.linked("one"), s.linked("two"));
    assert_eq!(s.layers(&song("none", "A", None), &c).unwrap(), None);
    assert_eq!(s.linked("none"), Some(None));
    assert!(s.ready("none"));
    assert_eq!(c.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn a_result_made_otherwise_is_made_again() {
    let s = store("made");
    let c = Counting::default();
    s.layers(&song("one", "A", Some(sample_cover(64))), &c)
        .unwrap();
    let Some(Some(hash)) = s.linked("one") else {
        panic!("linked");
    };
    // The same layout, made by other models: read, but not current.
    let mut kept = s.load(&hash).unwrap();
    kept.made = "other-models split-0".into();
    std::fs::write(s.result_file(&hash), to_bytes(&kept)).unwrap();
    assert_eq!(
        from_bytes(&std::fs::read(s.result_file(&hash)).unwrap()),
        Some(kept)
    );
    assert!(!s.ready("one"));
    assert!(s.load(&hash).is_none());
    s.layers(&song("one", "A", Some(sample_cover(64))), &c)
        .unwrap();
    assert_eq!(c.calls.load(Ordering::SeqCst), 2);
    assert!(s.ready("one"));
    // Another layout's version byte is not read at all.
    let mut bytes = std::fs::read(s.result_file(&hash)).unwrap();
    bytes[4] += 1;
    assert_eq!(from_bytes(&bytes), None);
}

/// While the data folder moves and after, a cover taken apart is written
/// into the new place (by a store made with the old one's name, as the
/// player that chose the move holds it), and one taken apart before is
/// found where it is: nothing is taken apart again, nothing is left behind.
#[test]
fn covers_go_into_the_new_place_while_and_after_the_data_folder_moves() {
    let dir = std::env::temp_dir().join(format!("jodel-covers-moved-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (old, new) = (dir.join("old"), dir.join("new"));
    let (before, during, after) = (
        CoverStore::new(old.join("covers")),
        CoverStore::new(old.join("covers")),
        CoverStore::new(old.join("covers")),
    );
    let c = Counting::default();
    before
        .layers(&song("before", "A", Some(sample_cover(64))), &c)
        .unwrap();
    let settings = dir.join("settings");
    crate::store::mover::begin(&settings, &[(old.clone(), new.clone())]).unwrap();
    let other = image::imageops::rotate90(&sample_cover(64));
    during
        .layers(&song("during", "B", Some(other)), &c)
        .unwrap();
    assert!(new.join("covers/songs").read_dir().unwrap().count() == 1);
    // Found by the new place's name too (a player started meanwhile).
    let started = CoverStore::new(new.join("covers"));
    assert!(started.ready("before") && started.ready("during"));
    crate::store::mover::finish(&settings);
    assert!(!old.exists(), "nothing left behind");
    after.layers(&song("after", "C", None), &c).unwrap();
    assert!(new.join("covers/songs/after.txt").is_file());
    assert!(!old.exists(), "nothing written into the old place");
    for song in ["before", "during", "after"] {
        assert!(started.ready(song), "{song}");
    }
    assert_eq!(c.calls.load(Ordering::SeqCst), 2);
    let _ = std::fs::remove_dir_all(&dir);
}
