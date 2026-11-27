//! Each network's footprint, worked out from its shapes before it is set
//! up, against what the card's budget says it took once set up: the
//! background separator and the beat network (their weights built in),
//! and the speech model, the speech recogniser and the cover's two where
//! their files are in the folder `JODEL_MODELS` names. One test, so
//! nothing else in the process holds or frees memory meanwhile; skipped
//! where there is no usable GPU, or no budget is told. The bounds are a
//! sanity check: at least most of the buffers worked out, at most them all
//! with the weights' upload (about two and a half times the weights, at
//! most), a device's own few dozen megabytes either way.

use std::path::{Path, PathBuf};

use jodel_gpu::memory::budget;
use jodel_gpu::{Footprint, GpuAsr, GpuDepth, GpuMatte, GpuNetwork, GpuSeparator, GpuSpeech};

const MB: u64 = 1 << 20;

/// The models' folder: `JODEL_MODELS`, where it is set.
fn models() -> Option<PathBuf> {
    let models = std::env::var_os("JODEL_MODELS").map(PathBuf::from);
    if models.is_none() {
        eprintln!("no models' folder (JODEL_MODELS): their networks skipped");
    }
    models
}

/// What the card holds now.
fn used() -> u64 {
    budget().map_or(0, |b| b.usage)
}

/// `footprint` against what setting up by `set_up` took (what it set up
/// let go of after, in a turn); skipped where it does not set up.
fn holds<T>(what: &str, footprint: Footprint, set_up: impl FnOnce() -> Result<T, String>) {
    let before = used();
    let Ok(made) = set_up() else {
        eprintln!("{what}: does not set up here: skipped");
        return;
    };
    let took = used().saturating_sub(before);
    eprintln!(
        "{what}: worked out {} MB kept + {} MB working, took {} MB",
        footprint.resident / MB,
        footprint.working / MB,
        took / MB
    );
    assert!(
        took + 64 * MB >= footprint.resident * 3 / 4,
        "{what}: took {} MB, worked out {} MB",
        took / MB,
        footprint.resident / MB
    );
    assert!(
        took <= footprint.total() * 5 / 4 + 64 * MB,
        "{what}: took {} MB, worked out {} MB at most",
        took / MB,
        footprint.total() / MB
    );
    let _turn = jodel_gpu::setting_up();
    drop(made);
}

/// The bytes of `path`.
fn bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

#[test]
fn each_network_takes_about_what_its_shapes_say() {
    if budget().is_none() || jodel_gpu::gpu_name().is_err() {
        eprintln!("no usable GPU, or no budget told: skipped");
        return;
    }
    // The background separator (the track playing's takes the allocator's
    // large blocks, and no scheduler plans it).
    holds(
        "separator",
        GpuSeparator::background_resident().footprint(),
        || {
            let separator = GpuSeparator::background().filter(|s| !s.plays());
            separator.ok_or_else(|| "no background separator".to_owned())
        },
    );
    holds(
        "beat network",
        GpuNetwork::shared_resident().footprint(),
        GpuNetwork::new,
    );
    let Some(models) = models() else {
        return;
    };
    let speech = models.join("wav2vec2-base-960h.safetensors");
    holds("speech model", GpuSpeech::footprint(bytes(&speech)), || {
        GpuSpeech::new(&jodel_lyrics::speech::Wav2Vec2::open(&speech)?)
    });
    let asr = models.join("qwen3-asr-0.6b-q4.jdlasr");
    holds("speech recogniser", GpuAsr::footprint(bytes(&asr)), || {
        GpuAsr::new(&jodel_lyrics::asr::packed::PackedModel::open(&asr)?)
    });
    let (depth, matte) = (
        models.join("da3-small.safetensors"),
        models.join("ben2-base.safetensors"),
    );
    if let (Ok(d), Ok(m)) = (std::fs::read(&depth), std::fs::read(&matte)) {
        let (fd, fm) = (
            GpuDepth::footprint(d.clone()).unwrap(),
            GpuMatte::footprint(m.clone()).unwrap(),
        );
        let both = Footprint {
            resident: fd.resident + fm.resident,
            working: fd.working.max(fm.working),
        };
        holds("the cover's two", both, || {
            Ok((GpuDepth::new(d)?, GpuMatte::new(m)?))
        });
    }
}
