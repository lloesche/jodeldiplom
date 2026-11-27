//! A song's story scenes, karaoke's backdrop in the game
//! (`crate::gui::backdrop::scenes`).
//!
//! The story's drawing step leaves a song's scenes in its folder in the
//! data folder's `scenes/` ([`SceneStore::folder`]): a picture a scene and
//! their description with the song's look ([`Scenes`], [`FILE`]),
//! `crate::work::story::draw`'s. The story's last step in the song's
//! preparation takes each picture apart once ([`take_apart`], on the GPU
//! by [`GpuNets`], `split.rs`: its depth and the matte of its subject, its
//! heads held flat by the faces found ([`FaceNet`], on the processor), two
//! or three layers or a warp, the plate behind the layers filled) and
//! keeps it by the picture's bytes in a cache of its own ([`SceneStore`],
//! made again where [`VERSION`] changes). As the song plays, its scenes'
//! texels are made ready for the GPU from what is kept ([`Ready`], on a
//! thread of their own: [`PlayingStory`]), so a scene showing costs the
//! window no more than sending them over.

mod describe;
mod faces;
mod job;
mod layers;
mod models;
mod playing;
mod split;
mod store;

use std::path::Path;

use image::RgbImage;

pub use describe::{Scene, Scenes, FILE};
pub use faces::{Face, FaceNet};
pub use job::{prepare, take_apart, taken_apart, SceneNets, TakenApart, Took};
pub use models::{
    GpuNets, SceneModels, DEPTH_BYTES, DEPTH_FILE, DEPTH_SHA256, DEPTH_URL, MATTE_BYTES,
    MATTE_FILE, MATTE_SHA256, MATTE_URL,
};
pub use playing::PlayingStory;
pub use split::{alphas, derive, half, Cuts, Derived, Head, Split};
pub use store::{from_bytes, to_bytes, SceneStore};

/// A song's key, its scenes' folder's name ([`SceneStore::folder`]) and
/// its pictures' seeds' (`crate::work::story::draw::seed`): as the lyrics'
/// cache keys a song (its path, size and modification time: a file
/// retagged is drawn afresh), or its path's hash where it cannot be read.
pub fn song_key(path: &Path) -> String {
    crate::work::lyrics::store::cache_key(path)
        .unwrap_or_else(|| crate::store::cache::name(&path.display().to_string()))
}

#[cfg(test)]
pub(crate) use describe::tests::look;
#[cfg(test)]
pub(crate) use job::tests::{drawn, ByRed};

use crate::work::picture::{resize, rgba, Planes, Swatch};

/// What the parts kept are made by: the models and the way the pictures
/// are taken apart. Parts made otherwise are made again.
pub const VERSION: &str = "da3-small ben2-base yunet scenes-3";

/// A picture's file's hash (of its `bytes`): what its parts are kept by.
pub fn picture_hash(bytes: &[u8]) -> String {
    format!("{:016x}", jodel_dsp::fnv1a(bytes))
}

/// How a scene moves, as its textures have it.
#[derive(Clone, Debug, PartialEq)]
pub enum Moved {
    /// By a warp of its nearness.
    Warp,
    /// In layers.
    Layers {
        /// Where it is cut.
        cuts: Cuts,
        /// The plate behind them, four bytes a pixel, the picture's size.
        plate: Vec<u8>,
    },
}

impl Moved {
    /// How many layers (1: the warp).
    pub fn count(&self) -> u32 {
        match self {
            Moved::Warp => 1,
            Moved::Layers { cuts, .. } => cuts.layers(),
        }
    }
}

/// A scene ready for the backdrop: its span in the song, its picture and
/// shape as their textures take them, how it moves, where its subject is,
/// its palette.
#[derive(Clone, Debug, PartialEq)]
pub struct Ready {
    /// Its span, in the song's seconds.
    pub span: (f64, f64),
    /// Its picture's size.
    pub size: (u32, u32),
    /// The picture, four bytes a pixel.
    pub picture: Vec<u8>,
    /// The shape's size: half the picture's.
    pub shape_size: (u32, u32),
    /// Four bytes a texel: the nearness the warp and the haze go by (its
    /// heads held flat, smoothed), the middle layer's share, the
    /// nearest's, and 255.
    pub shape: Vec<u8>,
    /// How it moves (its plate with it, in layers).
    pub moved: Moved,
    /// Where its subject is (the picture's shares): the camera drifts
    /// towards it.
    pub focus: [f32; 2],
    /// Its main colours, the largest share first.
    pub palette: Vec<Swatch>,
}

impl Ready {
    /// The scene spanning `span` of `picture`, from its `parts`.
    pub fn new(span: (f64, f64), picture: &RgbImage, parts: &Derived) -> Ready {
        let size = picture.dimensions();
        let alphas = match &parts.split {
            Split::Layers { alphas, .. } => Some(alphas),
            Split::Warp => None,
        };
        let (sw, sh) = alphas.map_or(split::half(size), |a| {
            (a.width() as usize, a.height() as usize)
        });
        let near = resize(&Planes::from(&parts.near), sw, sh);
        let layer = |i: usize, k: usize| alphas.map_or(0, |a| a.as_raw()[i * 2 + k]);
        let shape = (0..sw * sh)
            .flat_map(|i| [split::byte(near.data[i]), layer(i, 0), layer(i, 1), 255])
            .collect();
        let moved = match &parts.split {
            Split::Warp => Moved::Warp,
            Split::Layers { cuts, plate, .. } => Moved::Layers {
                cuts: *cuts,
                plate: rgba(plate),
            },
        };
        Ready {
            span,
            size,
            picture: rgba(picture),
            shape_size: (sw as u32, sh as u32),
            shape,
            moved,
            focus: focus(parts),
            palette: parts.palette.clone(),
        }
    }
}

/// Where a scene's subject is (the picture's shares): the middle of its
/// nearest layer; else of its faces, the larger counting more; else of its
/// nearest fifth; else its middle.
fn focus(parts: &Derived) -> [f32; 2] {
    let centroid = |(w, h): (usize, usize), weight: &dyn Fn(usize) -> f32| {
        let (mut sx, mut sy, mut sum) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..w * h {
            let k = f64::from(weight(i));
            sx += k * ((i % w) as f64 + 0.5);
            sy += k * ((i / w) as f64 + 0.5);
            sum += k;
        }
        (sum > 1e-3 * (w * h) as f64)
            .then(|| [(sx / sum / w as f64) as f32, (sy / sum / h as f64) as f32])
    };
    if let Split::Layers { alphas, .. } = &parts.split {
        let size = (alphas.width() as usize, alphas.height() as usize);
        let raw = alphas.as_raw();
        if let Some(c) = centroid(size, &|i| f32::from(raw[i * 2 + 1]) / 255.0) {
            return c;
        }
    }
    if !parts.heads.is_empty() {
        let (mut x, mut y, mut sum) = (0.0, 0.0, 0.0);
        for head in &parts.heads {
            let [fx, fy, fw, fh] = head.face;
            let k = fw * fh;
            x += k * (fx + fw / 2.0);
            y += k * (fy + fh / 2.0);
            sum += k;
        }
        return [x / sum, y / sum];
    }
    let near = &parts.near;
    let mut sorted = near.near.clone();
    sorted.sort_unstable();
    let fifth = sorted[sorted.len() * 4 / 5].max(1);
    let size = (near.width as usize, near.height as usize);
    centroid(size, &|i| f32::from(u8::from(near.near[i] >= fifth))).unwrap_or([0.5, 0.5])
}

/// A song's scenes ready for the backdrop: which they are (a hash of
/// their pictures' and their spans), and each scene, in the song's order
/// (the song's look over them is its [`Scenes::look`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Story {
    /// Which they are: what tells two stories apart.
    pub id: u64,
    /// The scenes.
    pub scenes: Vec<Ready>,
}

impl Story {
    /// The `scenes`, of pictures of `hashes`.
    pub fn new(hashes: &[String], scenes: Vec<Ready>) -> Story {
        let spans: Vec<String> = (scenes.iter())
            .map(|s| format!("{} {}", s.span.0, s.span.1))
            .collect();
        let which = format!("{} {}", hashes.join(" "), spans.join(" "));
        Story {
            id: jodel_dsp::fnv1a(which.as_bytes()),
            scenes,
        }
    }
}
