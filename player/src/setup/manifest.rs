//! What setup gets: the models the player needs, each with where its files
//! come from (at a fixed revision), their lengths and SHA-256s
//! ([`manifest`]), and the models found, by what they are ([`Found`]).

use std::path::{Path, PathBuf};

use crate::work::lyrics::models;
use crate::work::story_models as story;

/// A file's length and SHA-256: what it is checked against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum {
    /// Its length in bytes.
    pub bytes: u64,
    /// Its SHA-256, in lower-case hex.
    pub sha256: String,
}

/// A file downloaded as it is, checked against its [`Sum`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Where it comes from.
    pub url: String,
    /// What it is checked against.
    pub sum: Sum,
}

/// How a needed model is got.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// One file, downloaded as it is.
    Download(Part),
    /// A model's folder: its files downloaded as they are, each at its
    /// path in the folder (`/` between folders, as in the model's own
    /// repository).
    Folder(Vec<(String, Part)>),
    /// One file made here from a checkpoint's files, downloaded as a
    /// folder beside it, then packed at [`super::PACK_BITS`] bits.
    Pack {
        /// What the packed file is checked against.
        packed: Sum,
        /// The checkpoint's files, each at its path in its folder.
        parts: Vec<(String, Part)>,
        /// Why it is packed, as the screen says it.
        why: String,
    },
}

/// A model the player needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Need {
    /// Its path in the store (`/` between folders): its file, or its
    /// folder ([`Source::Folder`]).
    pub file: String,
    /// What it is, as the screen names it ("the word-timing model").
    pub title: String,
    /// Whose it is ("wav2vec2 by Meta"), said where it is downloaded.
    pub credit: String,
    /// What it is for, as the screen says it.
    pub why: String,
    /// How it is got.
    pub source: Source,
}

/// `name` (`/` between folders) under `dir`.
pub fn under(dir: &Path, name: &str) -> PathBuf {
    name.split('/').fold(dir.to_path_buf(), |p, c| p.join(c))
}

impl Need {
    /// Where it is in the models folder `dir`.
    pub fn at(&self, dir: &Path) -> PathBuf {
        under(dir, &self.file)
    }

    /// The files it downloads, each with where it goes for the model's
    /// file or folder at `at` (for a packed file, the folder its
    /// checkpoint is downloaded into). Every path setup hands on is one
    /// of these, as setup puts it; setup's checks read each through
    /// [`crate::store::mover::locate`] (while the data folder moves, a
    /// file not yet moved is where it was), and so must a reader.
    pub fn files(&self, at: &Path) -> Vec<(PathBuf, &Part)> {
        match &self.source {
            Source::Download(part) => vec![(at.to_path_buf(), part)],
            Source::Folder(files) | Source::Pack { parts: files, .. } => files
                .iter()
                .map(|(name, part)| (under(at, name), part))
                .collect(),
        }
    }

    /// The files it keeps, with it at `at` (its file, or its folder), each
    /// with what it is checked against.
    pub fn expected(&self, at: &Path) -> Vec<(PathBuf, &Sum)> {
        match &self.source {
            Source::Pack { packed, .. } => vec![(at.to_path_buf(), packed)],
            _ => self
                .files(at)
                .into_iter()
                .map(|(path, part)| (path, &part.sum))
                .collect(),
        }
    }

    /// What it keeps, in bytes.
    pub fn bytes(&self) -> u64 {
        self.expected(Path::new(""))
            .iter()
            .map(|(_, sum)| sum.bytes)
            .sum()
    }

    /// What it downloads, in bytes.
    pub fn downloaded_bytes(&self) -> u64 {
        self.files(Path::new(""))
            .iter()
            .map(|(_, part)| part.sum.bytes)
            .sum()
    }

    /// Where it is made here from what is downloaded: what the packed
    /// file is checked against, and why it is packed.
    pub fn packed(&self) -> Option<(&Sum, &str)> {
        match &self.source {
            Source::Pack { packed, why, .. } => Some((packed, why)),
            Source::Download(_) | Source::Folder(_) => None,
        }
    }
}

/// The models the player needs beside its executable, with their
/// checksums: the speech models, and where this machine has a usable GPU
/// (they run nowhere else) karaoke story mode's models and its scenes'.
pub fn manifest() -> Vec<Need> {
    manifest_for(jodel_gpu::gpu_name().is_ok())
}

/// The models the player has found, by what they are ([`Found::of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    /// The word-timing model ([`models::MODEL_FILE`]).
    pub speech: PathBuf,
    /// The speech recogniser ([`models::ASR_FILE`]).
    pub asr: PathBuf,
    /// The story scenes' depth and cut-out models, where the machine needs
    /// them.
    pub scenes: Option<crate::work::scenes::SceneModels>,
    /// Karaoke story mode's models, where the machine needs them.
    pub story: Option<story::StoryModels>,
}

impl Found {
    /// The models among `paths` (setup's, each with its model's path in the
    /// store); `None` without the speech models. A group (the scenes', the
    /// story's) is there whole or not at all.
    pub fn of(paths: &[(String, PathBuf)]) -> Option<Found> {
        let named = |file: &str| {
            paths
                .iter()
                .find(|(f, _)| f == file)
                .map(|(_, p)| p.clone())
        };
        let scenes = (|| {
            Some(crate::work::scenes::SceneModels {
                depth: named(crate::work::scenes::DEPTH_FILE)?,
                matte: named(crate::work::scenes::MATTE_FILE)?,
            })
        })();
        let story = (|| {
            Some(story::StoryModels {
                language: named(story::LANGUAGE_FILE)?,
                transformer: named(story::TRANSFORMER_DIR)?,
                pipeline: named(story::PIPELINE_DIR)?,
                encoder: named(story::ENCODER_DIR)?,
            })
        })();
        Some(Found {
            speech: named(models::MODEL_FILE)?,
            asr: named(models::ASR_FILE)?,
            scenes,
            story,
        })
    }
}

/// A file's [`Sum`].
fn sum(bytes: u64, sha256: &str) -> Sum {
    Sum {
        bytes,
        sha256: sha256.to_owned(),
    }
}

/// The files in `table` (each one's path, length and SHA-256), each at its
/// path, downloaded from `base`.
fn parts(base: &str, table: &[(&str, u64, &str)]) -> Vec<(String, Part)> {
    table
        .iter()
        .map(|&(name, bytes, sha)| {
            let part = Part {
                url: format!("{base}/{name}"),
                sum: sum(bytes, sha),
            };
            (name.to_owned(), part)
        })
        .collect()
}

/// A model of one file, downloaded from `url`.
fn download(url: &str, bytes: u64, sha256: &str) -> Source {
    Source::Download(Part {
        url: url.to_owned(),
        sum: sum(bytes, sha256),
    })
}

/// [`manifest`] for a machine with a usable GPU (`gpu`) or without.
pub fn manifest_for(gpu: bool) -> Vec<Need> {
    let need = |file: &str, title: &str, credit: &str, why: &str, source| Need {
        file: file.to_owned(),
        title: title.to_owned(),
        credit: credit.to_owned(),
        why: why.to_owned(),
        source,
    };
    let mut needs = vec![
        need(
            models::MODEL_FILE,
            "the word-timing model",
            "wav2vec2 by Meta",
            "It hears where each sung word falls, so karaoke and the Transition tab's lyrics \
             light up word by word.",
            download(models::MODEL_URL, models::MODEL_BYTES, models::MODEL_SHA256),
        ),
        need(
            models::ASR_FILE,
            "the speech recogniser",
            "Qwen3-ASR 0.6B by Alibaba's Qwen team",
            "It listens to the singing to check that lyrics found online belong to the song.",
            Source::Pack {
                packed: sum(models::ASR_BYTES, models::ASR_SHA256),
                parts: parts(models::ASR_BASE, &models::ASR_PARTS),
                why: format!(
                    "Packed at {} bits, with our calibration from 1,000 songs, it keeps its \
                     accuracy in a third of the memory: 589 MB instead of 1.9 GB.",
                    super::PACK_BITS
                ),
            },
        ),
    ];
    if !gpu {
        return needs;
    }
    let folder = |base: &str, table: &[(&str, u64, &str)]| Source::Folder(parts(base, table));
    needs.extend([
        need(
            crate::work::scenes::DEPTH_FILE,
            "the depth model",
            "Depth Anything 3 Small by ByteDance Seed",
            "It sees how near each part of a story's scene is, so the game's background can \
             move it in depth.",
            download(
                crate::work::scenes::DEPTH_URL,
                crate::work::scenes::DEPTH_BYTES,
                crate::work::scenes::DEPTH_SHA256,
            ),
        ),
        need(
            crate::work::scenes::MATTE_FILE,
            "the scenes' cut-out model",
            "BEN2 by Prama",
            "It cuts what stands out off a story's scene, so the game's background can move \
             it on its own.",
            download(
                crate::work::scenes::MATTE_URL,
                crate::work::scenes::MATTE_BYTES,
                crate::work::scenes::MATTE_SHA256,
            ),
        ),
        need(
            story::LANGUAGE_FILE,
            "the story's language model",
            "Gemma 4 E4B by Google DeepMind, at 4 bits by Unsloth",
            "It reads a song's lyrics and writes karaoke's story mode: its characters, and a \
             scene for each part of the song.",
            download(
                story::LANGUAGE_URL,
                story::LANGUAGE_BYTES,
                story::LANGUAGE_SHA256,
            ),
        ),
        need(
            story::TRANSFORMER_DIR,
            "the image model",
            "FLUX.2 [klein] 4B by Black Forest Labs, at 8 bits",
            "It draws the story's scenes, karaoke's background in story mode.",
            folder(story::TRANSFORMER_BASE, &story::TRANSFORMER_FILES),
        ),
        need(
            story::PIPELINE_DIR,
            "the image model's decoder and settings",
            "FLUX.2 [klein] 4B by Black Forest Labs",
            "It turns what the image model draws into the scene's pixels.",
            folder(story::PIPELINE_BASE, &story::PIPELINE_FILES),
        ),
        need(
            story::ENCODER_DIR,
            "the image model's text encoder",
            "Qwen3-4B by Alibaba's Qwen team, at 8 bits",
            "It reads each scene's description for the image model.",
            folder(story::ENCODER_BASE, &story::ENCODER_FILES),
        ),
    ]);
    needs
}
