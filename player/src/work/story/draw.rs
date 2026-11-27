//! A song's story drawn for the karaoke, from its plan ([`Drawing`]): the
//! image model's two units of work. In: the plan's pictures as the image
//! model is asked them ([`pictures::pictures`]: a portrait of each
//! character, and each scene with the portraits of those in it as its
//! reference pictures). Out: the pictures, and last their description with
//! the song's look ([`Scenes`]), in the song's folder in the data folder's
//! `scenes/` ([`SceneStore::folder`]), where the covers' worker takes them
//! up. The models they need, opened by whoever runs them (the image
//! model's needs said before: [`memory`]):
//!
//! - [`Drawing::encode`]: the text encoder (Qwen3-4B, as an [`Encoder`])
//!   for the prompt of every picture still to draw;
//! - [`Drawing::draw`]: the transformer and the decoder (FLUX.2 [klein]
//!   4B, as a [`Painter`]) for every picture still to draw, the portraits
//!   first, each kept as it lands and its prompt let go, each scene told to
//!   the preparation view ([`Story::Scene`]), given up as soon as the song
//!   is no longer wanted; then [`Drawing::keep`] describes them.
//!
//! The same plan always draws the same pictures: each picture's seed is
//! the song's and the picture's ([`seed`]). A picture is kept in a file
//! named by what made it ([`VERSION`], its seed, its size, the files of
//! its reference pictures and its prompt), so an unchanged plan draws
//! nothing again, and a plan made again draws again what it changed; the
//! description is named by what it holds ([`Drawing::drawn`]).
//!
//! A prompt's embeddings take about 16 MB (512 tokens of 7680 floats):
//! between its two turns a song holds that much for each picture still to
//! draw (some 400 MB for a song of 25 pictures), each let go once its
//! picture is drawn.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use image::{imageops, DynamicImage, ImageFormat, RgbImage};
use jodel_gpu::image::{Drawer, Embeds, Picture, Reference, TextEncoder};
use jodel_gpu::Footprint;

use crate::core::prep::Story;
use crate::core::story::pictures::{self, Ask, Pictures, MOST_REFERENCES};
use crate::core::story::Plan;
use crate::store::{cache, mover};
use crate::work::picture::codec::png;
use crate::work::scenes::{Scene, SceneStore, Scenes};
use crate::work::song_name;

/// What draws the pictures, as their files' names say it: the models
/// (FLUX.2 [klein] 4B in BFL's FP8 release, four steps; Qwen3-4B in
/// Qwen's) and how they are asked ([`pictures`]). Another raises it, and
/// every picture is drawn again.
pub const VERSION: u32 = 1;

/// A reference picture's side: a portrait's head ([`head`]) made this
/// size, as the scenes take a character in (the bake-off's).
pub const REFERENCE: usize = 512;

/// A portrait's head, as the bake-off cropped it: a square in the middle
/// across, its side this share of the portrait's, ...
const HEAD_SIDE: f64 = 0.625;
/// ... this share of the portrait's side from its top.
const HEAD_TOP: f64 = 0.0625;

/// How a drawing's turn ended ([`Drawing::draw`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    /// Every picture drawn.
    All,
    /// Given up: the song is no longer wanted (what is drawn is kept).
    GivenUp,
}

/// A text encoder: a prompt's embeddings for the image model.
pub trait Encoder {
    /// `prompt`'s embeddings.
    fn encode(&mut self, prompt: &str) -> Result<Embeds, String>;
}

impl Encoder for TextEncoder {
    fn encode(&mut self, prompt: &str) -> Result<Embeds, String> {
        TextEncoder::encode(self, prompt)
    }
}

/// An image model: a picture of a size from a prompt's embeddings, a seed
/// and its reference pictures.
pub trait Painter {
    /// The picture.
    fn paint(
        &mut self,
        embeds: &Embeds,
        seed: u64,
        size: (usize, usize),
        refs: &[Reference],
    ) -> Result<Picture, String>;
}

impl Painter for Drawer {
    fn paint(
        &mut self,
        embeds: &Embeds,
        seed: u64,
        size: (usize, usize),
        refs: &[Reference],
    ) -> Result<Picture, String> {
        Drawer::paint(self, embeds, seed, size, refs)
    }
}

/// What the image model needs on the GPU (nothing read): its weights,
/// and the working buffers of its largest picture, a portrait or a scene
/// with the most reference pictures.
pub fn memory() -> Footprint {
    let refs = [(REFERENCE, REFERENCE); MOST_REFERENCES];
    let (portrait, scene) = (pictures::PORTRAIT, pictures::SCENE);
    let portrait = Drawer::planned(portrait.0, portrait.1, &[]);
    let scene = Drawer::planned(scene.0, scene.1, &refs);
    Footprint {
        resident: portrait.resident.max(scene.resident),
        working: portrait.working.max(scene.working),
    }
}

/// What a picture is of: a character's portrait or a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A portrait.
    Portrait,
    /// A scene.
    Scene,
}

impl Kind {
    /// Its name, as its pictures' files are named.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Portrait => "portrait",
            Kind::Scene => "scene",
        }
    }
}

/// A picture of a drawing: of what, and which among its kind.
type Which = (Kind, usize);

/// The seed of the `index`th picture of `kind` of the song of `song_key`
/// ([`crate::work::cover::song_key`]).
pub fn seed(song_key: &str, kind: Kind, index: usize) -> u64 {
    jodel_dsp::fnv1a(format!("{song_key} {} {index}", kind.name()).as_bytes())
}

/// A picture of a drawing: what the image model is asked, its seed, and
/// its file in the song's folder.
struct Planned {
    ask: Ask,
    seed: u64,
    file: PathBuf,
}

/// The prompts' embeddings of a drawing's pictures still to draw
/// ([`Drawing::encode`]), in the order they are drawn, each let go once
/// its picture is.
pub struct Encoded(VecDeque<(Which, Embeds)>);

impl Encoded {
    /// The bytes they hold.
    pub fn bytes(&self) -> usize {
        (self.0.iter())
            .map(|(_, e)| std::mem::size_of_val(e.values.as_slice()))
            .sum()
    }
}

/// A song's story to draw: the song's file, its plan, its folder, and its
/// pictures, each named by what makes it.
pub struct Drawing {
    path: PathBuf,
    plan: Plan,
    folder: PathBuf,
    portraits: Vec<Planned>,
    scenes: Vec<Planned>,
}

impl Drawing {
    /// The drawing of `plan`, of the song at `path`, kept in `store`
    /// (looks at the song's file for its key: not for the window's
    /// thread).
    pub fn new(path: PathBuf, plan: Plan, store: &SceneStore) -> Drawing {
        let key = crate::work::cover::song_key(&path);
        let folder = store.folder(&key);
        let Pictures { portraits, scenes } = pictures::pictures(&plan);
        let planned = |kind: Kind, asks: Vec<Ask>, portraits: &[Planned]| -> Vec<Planned> {
            (asks.into_iter().enumerate())
                .map(|(index, ask)| {
                    let seed = seed(&key, kind, index);
                    let refs: Vec<String> = (ask.refs.iter())
                        .map(|&k| file_name(&portraits[k].file).into_owned())
                        .collect();
                    let (w, h) = ask.size;
                    let made = format!(
                        "{VERSION}\n{seed}\n{w}x{h}\n{}\n{}",
                        refs.join(" "),
                        ask.prompt
                    );
                    let name = format!("{}-{index:02}-{}.png", kind.name(), cache::name(&made));
                    Planned {
                        file: folder.join(name),
                        ask,
                        seed,
                    }
                })
                .collect()
        };
        let portraits = planned(Kind::Portrait, portraits, &[]);
        let scenes = planned(Kind::Scene, scenes, &portraits);
        Drawing {
            path,
            plan,
            folder,
            portraits,
            scenes,
        }
    }

    /// The song's file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The picture `which`.
    fn planned(&self, (kind, index): Which) -> &Planned {
        match kind {
            Kind::Portrait => &self.portraits[index],
            Kind::Scene => &self.scenes[index],
        }
    }

    /// Its pictures, in the order they are drawn: the portraits, then the
    /// scenes.
    fn all(&self) -> impl Iterator<Item = (Which, &Planned)> {
        let portraits = (self.portraits.iter().enumerate()).map(|(k, p)| ((Kind::Portrait, k), p));
        let scenes = (self.scenes.iter().enumerate()).map(|(k, p)| ((Kind::Scene, k), p));
        portraits.chain(scenes)
    }

    /// Its scenes as they are kept once drawn: each scene's span and
    /// picture, the plan's look, and what made them ([`Drawing::made`]).
    fn described(&self) -> Scenes {
        Scenes {
            made: self.made(),
            scenes: (self.plan.scenes.iter().zip(&self.scenes))
                .map(|(s, p)| Scene {
                    start: s.from,
                    end: s.to,
                    picture: p.file.clone(),
                })
                .collect(),
            look: self.plan.look.clone(),
        }
    }

    /// What its description is named by: a hash of what it holds (each
    /// scene's span and picture, the look), as the plan has them.
    fn made(&self) -> String {
        let look = serde_json::to_string(&self.plan.look).expect("a look writes as JSON");
        let mut holds = format!("{VERSION}\n{look}\n");
        for (s, p) in self.plan.scenes.iter().zip(&self.scenes) {
            holds += &format!("{} {} {}\n", s.from, s.to, file_name(&p.file));
        }
        cache::name(&holds)
    }

    /// Its pictures not in its folder, in the order they are drawn.
    fn missing(&self) -> Vec<Which> {
        (self.all())
            .filter(|(_, p)| !mover::locate(&p.file).is_file())
            .map(|(which, _)| which)
            .collect()
    }

    /// Whether a picture of it is still to draw (not in its folder).
    pub fn to_draw(&self) -> bool {
        !self.missing().is_empty()
    }

    /// Whether it is drawn: every picture in its folder, and described
    /// there by what made the description as this plan has it.
    pub fn drawn(&self) -> bool {
        !self.to_draw() && Scenes::read(&self.folder).is_some_and(|s| s.made == self.made())
    }

    /// The text encoder's unit of work: the prompt of each picture still
    /// to draw, encoded by `encoder`, in order.
    pub fn encode(&self, encoder: &mut dyn Encoder) -> Result<Encoded, String> {
        let mut encoded = VecDeque::new();
        for which in self.missing() {
            encoded.push_back((which, encoder.encode(&self.planned(which).ask.prompt)?));
        }
        Ok(Encoded(encoded))
    }

    /// The image model's unit of work: each picture of `encoded` painted
    /// by `painter` with its seed and its reference pictures (the
    /// portraits' heads, painted now or read from the folder), kept in the
    /// folder as it lands and its prompt let go, each scene told to `tell`;
    /// given up ([`Drawn::GivenUp`]) before a picture once `still` says the
    /// song is no longer wanted.
    pub fn draw(
        &self,
        encoded: &mut Encoded,
        painter: &mut dyn Painter,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String> {
        let mut heads: Vec<Option<Reference>> = vec![None; self.portraits.len()];
        while let Some(&(which, ref embeds)) = encoded.0.front() {
            if !still() {
                return Ok(Drawn::GivenUp);
            }
            let started = Instant::now();
            let p = self.planned(which);
            let mut refs = Vec::new();
            for &k in &p.ask.refs {
                let head = match &heads[k] {
                    Some(head) => head.clone(),
                    None => head_of(&read(&self.portraits[k].file)?),
                };
                heads[k] = Some(head.clone());
                refs.push(head);
            }
            let painting = Instant::now();
            let painted = painter.paint(embeds, p.seed, p.ask.size, &refs)?;
            let painting = painting.elapsed();
            encoded.0.pop_front();
            let (w, h) = (painted.width as u32, painted.height as u32);
            let image = RgbImage::from_raw(w, h, painted.rgb)
                .ok_or_else(|| format!("a picture of {w} × {h} without its pixels"))?;
            let file = mover::current(&p.file);
            durable::write_atomic(&file, &png(|c| image.write_to(c, ImageFormat::Png)))
                .map_err(|e| format!("{}: {e}", file.display()))?;
            let (kind, index) = which;
            crate::trace::note(format_args!(
                "story: {}: {} {index} drawn in {:.1} s ({:.1} s painted)",
                song_name(&self.path),
                kind.name(),
                started.elapsed().as_secs_f64(),
                painting.as_secs_f64()
            ));
            match kind {
                Kind::Portrait => heads[index] = Some(head_of(&image)),
                Kind::Scene => tell(Story::Scene {
                    index,
                    of: self.scenes.len(),
                    picture: Arc::new(DynamicImage::ImageRgb8(image).to_rgba8()),
                }),
            }
        }
        Ok(Drawn::All)
    }

    /// The drawing step's last write: its scenes described in its folder
    /// (every picture in it before), then the pictures no longer its own
    /// let go.
    pub fn keep(&self) -> Result<(), String> {
        self.described().write(&self.folder)?;
        let folder = mover::current(&self.folder);
        let own: Vec<_> = self.all().map(|(_, p)| file_name(&p.file)).collect();
        let stale = (std::fs::read_dir(&folder).into_iter().flatten().flatten())
            .map(|e| e.path())
            .filter(|f| f.extension().is_some_and(|x| x == "png") && !own.contains(&file_name(f)));
        for file in stale {
            let _ = std::fs::remove_file(file);
        }
        let _ = durable::sync_dir(&folder);
        Ok(())
    }
}

/// A file's name, as the pictures are named and listed by.
fn file_name(file: &Path) -> std::borrow::Cow<'_, str> {
    file.file_name().unwrap_or_default().to_string_lossy()
}

/// The picture in `file`.
fn read(file: &Path) -> Result<RgbImage, String> {
    let at = mover::locate(file);
    let image = image::open(&at).map_err(|e| format!("{}: {e}", at.display()))?;
    Ok(image.to_rgb8())
}

/// Where the head is in a portrait of side `side`: its left, its top and
/// its side, in pixels.
fn head(side: u32) -> (u32, u32, u32) {
    let head = (f64::from(side) * HEAD_SIDE).round() as u32;
    let top = (f64::from(side) * HEAD_TOP).round() as u32;
    ((side - head) / 2, top, head)
}

/// A portrait's head as a reference picture ([`head`], made [`REFERENCE`]
/// square).
fn head_of(portrait: &RgbImage) -> Reference {
    let (x, y, side) = head(portrait.width().min(portrait.height()));
    let crop = imageops::crop_imm(portrait, x, y, side, side).to_image();
    let side = REFERENCE as u32;
    let made = imageops::resize(&crop, side, side, imageops::FilterType::Lanczos3);
    Reference {
        width: REFERENCE,
        height: REFERENCE,
        rgb: made.into_raw(),
    }
}

#[cfg(test)]
mod tests {
    /// A portrait's head is the bake-off's crop of it.
    #[test]
    fn a_portrait_s_head_is_its_upper_middle() {
        let side = crate::core::story::pictures::PORTRAIT.0 as u32;
        assert_eq!(super::head(side), (192, 64, 640));
    }
}
