//! A song's scenes taken apart ([`take_apart`]): each picture's parts
//! derived on the GPU ([`SceneNets`]: the two models,
//! [`super::GpuNets`]'s) and kept, as the last step of the song's story
//! in its preparation (`crate::work::story::worker`); then, as the song
//! plays, made ready for the backdrop from what is kept ([`prepare`],
//! each scene's texels, on the processor alone).

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use image::RgbImage;
use jodel_gpu::vision::prep::Rgb;
use jodel_gpu::vision::{Depth, Matte};

use super::split::{derive, Derived};
use super::{picture_hash, Ready, Scene, SceneStore, Scenes, Story};

/// What derives a scene's parts on the GPU: the depth and matte models
/// (or a stand-in), each as the caller holds it.
pub trait SceneNets {
    /// The depth of `picture`.
    fn depth(&self, picture: &Rgb) -> Result<Depth, String>;
    /// The matte of `picture`'s subject.
    fn matte(&self, picture: &Rgb) -> Result<Matte, String>;
}

/// How long deriving a scene's parts took, step by step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Took {
    /// Its depth, on the GPU.
    pub depth: Duration,
    /// Its faces, heads, split and the warp's nearness.
    pub faces: Duration,
    /// Its subject's matte, on the GPU (none for a warp).
    pub matte: Duration,
    /// Its layers and the plate filled (none for a warp).
    pub split: Duration,
}

impl Took {
    /// All of it.
    pub fn total(&self) -> Duration {
        self.depth + self.faces + self.matte + self.split
    }
}

impl fmt::Display for Took {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        write!(
            f,
            "{:.0} ms (depth {:.0}, faces {:.0}, matte {:.0}, split {:.0})",
            ms(self.total()),
            ms(self.depth),
            ms(self.faces),
            ms(self.matte),
            ms(self.split)
        )
    }
}

/// A scene's picture as its file holds it: its bytes' hash (what its parts
/// are kept by) and its pixels.
fn picture_of(scene: &Scene) -> Result<(String, RgbImage), String> {
    let path = &scene.picture;
    let bytes = std::fs::read(crate::store::mover::locate(path))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let picture = image::load_from_memory(&bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgb8();
    Ok((picture_hash(&bytes), picture))
}

/// Whether every one of `scenes` has its parts kept in `store` (each
/// picture read for its hash).
pub fn taken_apart(scenes: &Scenes, store: &SceneStore) -> Result<bool, String> {
    for scene in &scenes.scenes {
        if store.load(&picture_of(scene)?.0).is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// How taking a story's scenes apart ended ([`take_apart`]).
#[derive(Clone, Debug, PartialEq)]
pub enum TakenApart {
    /// Every scene: how long deriving each took (none: kept already).
    All(Vec<Option<Took>>),
    /// Given up, no longer wanted (what is kept stays).
    GivenUp,
}

/// `scenes` taken apart: each one's parts not kept in `store` derived by
/// `nets` and kept, while `still` says they are wanted (asked before each
/// picture: what is kept stays, so a story given up goes on from there the
/// next time).
pub fn take_apart(
    scenes: &Scenes,
    store: &SceneStore,
    nets: &dyn SceneNets,
    still: &dyn Fn() -> bool,
) -> Result<TakenApart, String> {
    let mut took = Vec::new();
    for scene in &scenes.scenes {
        let (hash, picture) = picture_of(scene)?;
        if store.load(&hash).is_some() {
            took.push(None);
            continue;
        }
        if !still() {
            return Ok(TakenApart::GivenUp);
        }
        let (parts, t) = derive(&picture, hash, nets)?;
        store.keep(&parts)?;
        took.push(Some(t));
    }
    Ok(TakenApart::All(took))
}

/// `scenes` made ready for the backdrop from their parts kept in `store`;
/// it fails on a scene whose parts are not kept (not taken apart yet).
pub fn prepare(scenes: &Scenes, store: &SceneStore) -> Result<Arc<Story>, String> {
    let (mut ready, mut hashes) = (Vec::new(), Vec::new());
    for scene in &scenes.scenes {
        let (hash, picture) = picture_of(scene)?;
        let parts: Derived = store
            .load(&hash)
            .ok_or_else(|| format!("{}: not taken apart", scene.picture.display()))?;
        ready.push(Ready::new((scene.start, scene.end), &picture, &parts));
        hashes.push(hash);
    }
    Ok(Arc::new(Story::new(&hashes, ready)))
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: a scene's pictures as the drawing step leaves them"
)]
pub(crate) mod tests {
    use super::*;
    use crate::work::scenes::{Moved, Scene};
    use image::RgbImage;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Depth and a matte from the picture's red: a red disc nearer (and
    /// the subject), the rest far; the depth's calls counted.
    #[derive(Default)]
    pub(crate) struct ByRed {
        pub(crate) calls: AtomicUsize,
    }

    impl SceneNets for ByRed {
        fn depth(&self, p: &Rgb) -> Result<Depth, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Depth {
                width: p.width,
                height: p.height,
                values: p
                    .data
                    .chunks(3)
                    .map(|c| if c[0] > 128 { 1.0 } else { 5.0 })
                    .collect(),
            })
        }

        fn matte(&self, p: &Rgb) -> Result<Matte, String> {
            Ok(Matte {
                width: p.width,
                height: p.height,
                values: p
                    .data
                    .chunks(3)
                    .map(|c| if c[0] > 128 { 1.0 } else { 0.0 })
                    .collect(),
            })
        }
    }

    /// A 160 × 90 picture: a red disc round (`cx`, 45) on grey, or grey
    /// alone with a soft slope of light (no subject).
    pub(crate) fn picture(disc: Option<f32>) -> RgbImage {
        RgbImage::from_fn(160, 90, |x, y| {
            let inside = disc.is_some_and(|cx| {
                let (dx, dy) = (x as f32 - cx, y as f32 - 45.0);
                dx * dx + dy * dy < 400.0
            });
            if inside {
                image::Rgb([220, 30, 30])
            } else {
                image::Rgb([90, 90, (40 + y) as u8])
            }
        })
    }

    /// The store of a data folder in `dir`, and a song's two scenes put
    /// where the drawing step leaves them (the song at `song`, a file
    /// there): a disc in front, then a slope of light; their pictures.
    pub(crate) fn drawn(store: &SceneStore, song: &Path) -> [PathBuf; 2] {
        let folder = store.folder(&crate::work::scenes::song_key(song));
        std::fs::create_dir_all(&folder).unwrap();
        let paths = ["00.png", "01.png"].map(|n| folder.join(n));
        picture(Some(60.0)).save(&paths[0]).unwrap();
        picture(None).save(&paths[1]).unwrap();
        let scenes = |(start, end), picture: &PathBuf| Scene {
            start,
            end,
            picture: picture.clone(),
        };
        Scenes {
            made: "drawn by hand".into(),
            scenes: vec![
                scenes((0.0, 8.0), &paths[0]),
                scenes((8.0, 20.0), &paths[1]),
            ],
            look: crate::work::scenes::look(),
        }
        .write(&folder)
        .unwrap();
        paths
    }

    /// The times of every scene taken apart (none given up).
    fn all(taken: Result<TakenApart, String>) -> Vec<Option<Took>> {
        match taken {
            Ok(TakenApart::All(took)) => took,
            other => panic!("every scene taken apart: {other:?}"),
        }
    }

    #[test]
    fn a_song_s_scenes_are_taken_apart_once_and_made_ready_from_what_is_kept() {
        let dir = std::env::temp_dir().join(format!("jodel-scenes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let song = dir.join("a song.wav");
        durable::write_atomic(&song, b"a song").unwrap();
        let store = SceneStore::new(dir.join("scenes"), dir.join("scene-parts"));
        // Nothing left for the song: no story.
        assert_eq!(store.scenes_of(&song), None);
        let paths = drawn(&store, &song);
        let scenes = store.scenes_of(&song).expect("the song's scenes");
        let nets = ByRed::default();
        // Not taken apart yet: nothing to make ready.
        assert_eq!(taken_apart(&scenes, &store), Ok(false));
        assert!(prepare(&scenes, &store).is_err());
        // No longer wanted: given up before the first picture, nothing
        // derived.
        let given_up = take_apart(&scenes, &store, &nets, &|| false);
        assert_eq!(given_up, Ok(TakenApart::GivenUp));
        assert_eq!(nets.calls.load(Ordering::SeqCst), 0);
        let took = all(take_apart(&scenes, &store, &nets, &|| true));
        assert_eq!(nets.calls.load(Ordering::SeqCst), 2);
        assert!(took.iter().all(Option::is_some));
        assert_eq!(taken_apart(&scenes, &store), Ok(true));
        let story = prepare(&scenes, &store).unwrap();
        let [disc, plain] = &story.scenes[..] else {
            panic!("two scenes");
        };
        // The disc in front: two layers, the nearest the disc, the camera
        // drifting to it; the plate grey where the disc was.
        let Moved::Layers { cuts, plate } = &disc.moved else {
            panic!("layers");
        };
        assert_eq!(cuts.layers(), 2);
        assert_eq!(disc.shape_size, (80, 45));
        let texel = |r: &Ready, x: usize, y: usize| {
            let i = (y * r.shape_size.0 as usize + x) * 4;
            [r.shape[i], r.shape[i + 1], r.shape[i + 2]]
        };
        assert!(texel(disc, 30, 22)[2] > 250 && texel(disc, 70, 10)[2] < 5);
        assert!(
            (disc.focus[0] - 60.0 / 160.0).abs() < 0.03,
            "{:?}",
            disc.focus
        );
        let at = (45 * 160 + 60) * 4;
        assert!(plate[at] < 130, "{:?}", &plate[at..at + 4]);
        // The slope of light alone: the warp.
        assert_eq!(plain.moved, Moved::Warp);
        assert_eq!(disc.span, (0.0, 8.0));
        // Kept by picture, in a cache apart from the songs' scenes: taken
        // apart again, the models are not asked again.
        assert_eq!(
            std::fs::read_dir(dir.join("scene-parts")).unwrap().count(),
            2
        );
        let again = all(take_apart(&scenes, &store, &nets, &|| true));
        assert!(again.iter().all(Option::is_none));
        assert_eq!(nets.calls.load(Ordering::SeqCst), 2);
        let ready = prepare(&scenes, &store).unwrap();
        assert_eq!(ready.id, story.id);
        for (a, b) in ready.scenes.iter().zip(&story.scenes) {
            assert!(a.shape == b.shape && a.moved == b.moved && a.picture == b.picture);
        }
        // A picture drawn again: that scene alone is taken apart again.
        picture(Some(100.0)).save(&paths[1]).unwrap();
        assert_eq!(taken_apart(&scenes, &store), Ok(false));
        let took = all(take_apart(&scenes, &store, &nets, &|| true));
        assert_eq!(nets.calls.load(Ordering::SeqCst), 3);
        assert!(took[0].is_none() && took[1].is_some());
        assert_ne!(prepare(&scenes, &store).unwrap().id, story.id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
