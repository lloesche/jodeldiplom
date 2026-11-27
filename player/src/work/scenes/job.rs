//! A song's scenes made ready for the backdrop ([`prepare`], [`story`]):
//! each picture's parts as kept, or derived on the GPU ([`SceneNets`]: the
//! cover's two models, `crate::work::cover::GpuSplitter`'s) and kept;
//! then each scene's texels made ([`super::Ready`]), on the covers'
//! thread, after the song playing's cover.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use jodel_gpu::vision::prep::Rgb;
use jodel_gpu::vision::{Depth, Matte};

use super::split::derive;
use super::{picture_hash, Ready, SceneStore, Scenes, Story};

/// What derives a scene's parts on the GPU: the cover's depth and matte
/// models (or a stand-in), each as the caller holds it.
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

/// What the one making a story ready is asked as it goes: whether it is
/// still wanted (asked before each picture taken apart: none is, once it
/// is not), and a word before each picture the models take apart.
pub type Asked<'a> = (&'a dyn Fn() -> bool, &'a mut dyn FnMut());

/// `scenes` made ready: each one's parts as kept in `store` for its
/// picture, else derived by `nets` (none: not derived, and it fails on
/// them) and kept, while still wanted (`asked`: what is kept stays, so a
/// story given up goes on from there the next time); the scenes ready for
/// the backdrop, and how long deriving each took (none: kept).
pub fn prepare(
    scenes: &Scenes,
    store: &SceneStore,
    nets: Option<&dyn SceneNets>,
    (still, running): Asked,
) -> Result<(Arc<Story>, Vec<Option<Took>>), String> {
    let (mut took, mut ready, mut hashes) = (Vec::new(), Vec::new(), Vec::new());
    for scene in &scenes.scenes {
        let path = &scene.picture;
        let bytes = std::fs::read(crate::store::mover::locate(path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let hash = picture_hash(&bytes);
        let picture = image::load_from_memory(&bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .to_rgb8();
        let (parts, t) = match store.load(&hash) {
            Some(parts) => (parts, None),
            None => {
                let nets = nets.ok_or_else(|| format!("{}: not prepared", path.display()))?;
                if !still() {
                    return Err("no longer wanted".into());
                }
                running();
                let (parts, t) = derive(&picture, hash.clone(), nets)?;
                store.keep(&parts)?;
                (parts, Some(t))
            }
        };
        ready.push(Ready::new((scene.start, scene.end), &picture, &parts));
        took.push(t);
        hashes.push(hash);
    }
    Ok((Arc::new(Story::new(&hashes, ready)), took))
}

/// The story of the song at `song`: its scenes in `store`, made ready
/// ([`prepare`], `asked` as it goes; each picture derived now noted in the
/// trace with its time); none where it has none, or they cannot be made
/// ready (why noted in the trace).
pub fn story(
    store: &SceneStore,
    song: &Path,
    nets: Option<&dyn SceneNets>,
    asked: Asked,
) -> Option<Arc<Story>> {
    let scenes = store.scenes_of(song)?;
    let name = song.file_name().unwrap_or_default().to_string_lossy();
    match prepare(&scenes, store, nets, asked) {
        Ok((story, took)) => {
            for (i, t) in took.iter().enumerate() {
                if let Some(t) = t {
                    crate::trace::note(format_args!("scenes: {name} scene {i}: {t}"));
                }
            }
            Some(story)
        }
        Err(why) => {
            crate::trace::note(format_args!("scenes: {name}: {why}"));
            None
        }
    }
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
    use std::path::PathBuf;
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
        let folder = store.folder(&crate::work::cover::song_key(song));
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
            scenes: vec![
                scenes((0.0, 8.0), &paths[0]),
                scenes((8.0, 20.0), &paths[1]),
            ],
        }
        .write(&folder)
        .unwrap();
        paths
    }

    #[test]
    fn a_song_s_scenes_are_derived_once_and_read_back_after() {
        let dir = std::env::temp_dir().join(format!("jodel-scenes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let song = dir.join("a song.wav");
        durable::write_atomic(&song, b"a song").unwrap();
        let store = SceneStore::new(dir.join("scenes"), dir.join("scene-parts"));
        // Nothing left for the song: no story.
        assert_eq!(store.scenes_of(&song), None);
        let (yes, mut quiet) = (|| true, || {});
        assert!(story(&store, &song, None, (&yes, &mut quiet)).is_none());
        let paths = drawn(&store, &song);
        let scenes = store.scenes_of(&song).expect("the song's scenes");
        let nets = ByRed::default();
        // Without the models, nothing kept: it fails.
        assert!(prepare(&scenes, &store, None, (&yes, &mut quiet)).is_err());
        // No longer wanted: given up before the first picture, nothing
        // derived.
        let (no, mut told) = (|| false, 0);
        let given_up = prepare(&scenes, &store, Some(&nets), (&no, &mut || told += 1));
        assert!(given_up.is_err() && told == 0);
        assert_eq!(nets.calls.load(Ordering::SeqCst), 0);
        let mut told = 0;
        let (story, took) =
            prepare(&scenes, &store, Some(&nets), (&yes, &mut || told += 1)).unwrap();
        assert_eq!(told, 2);
        assert_eq!(nets.calls.load(Ordering::SeqCst), 2);
        assert!(took.iter().all(Option::is_some));
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
        // Kept by picture, in a cache apart from the songs' scenes: read
        // back the same, the models not asked again.
        assert_eq!(
            std::fs::read_dir(dir.join("scene-parts")).unwrap().count(),
            2
        );
        let (again, took) = prepare(&scenes, &store, None, (&yes, &mut quiet)).unwrap();
        assert!(took.iter().all(Option::is_none));
        assert_eq!(again.id, story.id);
        for (a, b) in again.scenes.iter().zip(&story.scenes) {
            assert!(a.shape == b.shape && a.moved == b.moved && a.picture == b.picture);
        }
        // A picture drawn again: that scene alone is derived again.
        picture(Some(100.0)).save(&paths[1]).unwrap();
        let (redrawn, took) = prepare(&scenes, &store, Some(&nets), (&yes, &mut quiet)).unwrap();
        assert_eq!(nets.calls.load(Ordering::SeqCst), 3);
        assert!(took[0].is_none() && took[1].is_some());
        assert_ne!(redrawn.id, story.id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
