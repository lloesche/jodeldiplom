//! The models that time the words: their files, where they are looked
//! for, and their checksums.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The speech model's file name.
pub const MODEL_FILE: &str = "wav2vec2-base-960h.safetensors";
/// Where it is downloaded from (Apache 2.0, by Meta).
pub const MODEL_URL: &str =
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors";
/// Its length in bytes.
pub const MODEL_BYTES: u64 = 377_607_901;
/// Its SHA-256.
pub const MODEL_SHA256: &str = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a";

/// The speech model where it is (not checked: [`crate::setup`] checks it):
/// `models/` beside the executable, or in the app bundle's
/// `Contents/Resources/models/` (where one sent along with the player
/// sits); else `models/` in the data folder, where setup puts it. `None`
/// when it is in none of them.
pub fn model_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let beside = exe.as_deref().and_then(Path::parent).map(Path::to_path_buf);
    let data = crate::dirs::Data::current().map(|d| d.0);
    model_in(beside.as_deref(), data.as_deref())
}

/// [`model_path`] for an executable in `exe_dir` and the data folder
/// `data_dir`.
pub fn model_in(exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    file_in(MODEL_FILE, exe_dir, data_dir)
}

/// The model file `name` in the places [`model_path`] looks.
fn file_in(name: &str, exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    let mut places = Vec::new();
    if let Some(dir) = exe_dir {
        places.push(dir.join(MODEL_DIR));
        places.push(dir.join("..").join("Resources").join(MODEL_DIR));
    }
    if let Some(dir) = data_dir {
        places.push(dir.join(MODEL_DIR));
    }
    places
        .into_iter()
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// The speech recogniser's file: Qwen3-ASR-0.6B (Apache 2.0, by Alibaba's
/// Qwen team) packed at 4 bits ([`jodel_lyrics::asr::packed`]), about 0.5 GB.
pub const ASR_FILE: &str = "qwen3-asr-0.6b-q4.jdlasr";
/// The packed file's length in bytes. Packing is the same on every machine
/// (each group of weights is fitted on its own, in a fixed order), so the
/// file is too: setup checks it against [`ASR_SHA256`]. A change to how it
/// is packed ([`jodel_lyrics::asr::packed::VERSION`], the calibration) makes
/// another file: these two are then measured again.
pub const ASR_BYTES: u64 = 588_661_472;
/// The packed file's SHA-256.
pub const ASR_SHA256: &str = "4a480c989e5d47e225a31fd798ec258c65e4c6c8f1bef51ca6f65df1396df151";
/// The checkpoint it is packed from, at a fixed revision.
pub(crate) const ASR_BASE: &str =
    "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0";
/// The checkpoint's files: name, length, SHA-256.
pub(crate) const ASR_PARTS: [(&str, u64, &str); 3] = [
    (
        "model.safetensors",
        1_876_091_704,
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
    ),
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "tokenizer_config.json",
        12_487,
        "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
    ),
];

/// The speech recogniser's file, where the build put it (as
/// [`model_path`]).
pub fn asr_model_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let beside = exe.as_deref().and_then(Path::parent).map(Path::to_path_buf);
    let data = crate::dirs::Data::current().map(|d| d.0);
    asr_model_in(beside.as_deref(), data.as_deref())
}

/// [`asr_model_path`] for an executable in `exe_dir` and the data folder
/// `data_dir`.
pub fn asr_model_in(exe_dir: Option<&Path>, data_dir: Option<&Path>) -> Option<PathBuf> {
    file_in(ASR_FILE, exe_dir, data_dir)
}

/// The folder the speech model sits in, where it is looked for.
pub const MODEL_DIR: &str = "models";

/// The speech model and the speech recogniser, as setup found them
/// ([`crate::setup`]): there, and checked against their built-in checksums.
/// Read through [`crate::mover::locate`]: while the data folder moves, one
/// not yet moved is read where it was.
pub struct Models {
    /// The speech model.
    pub speech: PathBuf,
    /// The speech recogniser.
    pub asr: PathBuf,
}

impl Models {
    /// The models at `speech` and `asr`.
    pub fn new(speech: PathBuf, asr: PathBuf) -> Arc<Models> {
        Arc::new(Models { speech, asr })
    }

    /// Where the speech model is to be read now.
    pub fn speech_now(&self) -> PathBuf {
        crate::mover::locate(&self.speech)
    }

    /// Where the speech recogniser is to be read now.
    pub fn asr_now(&self) -> PathBuf {
        crate::mover::locate(&self.asr)
    }

    /// Whether the speech recogniser is there.
    pub fn has_asr(&self) -> bool {
        jodel_lyrics::asr::packed::PackedModel::is_current(&self.asr_now())
    }
}
