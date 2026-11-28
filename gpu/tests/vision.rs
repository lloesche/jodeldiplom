//! The cover networks on the GPU against their authors' PyTorch outputs, on
//! the generated scene (`jodel_gpu::vision::scene`): Depth Anything 3
//! Small's depth at 504 × 504 and BEN2's matte at 1024 × 1024, each averaged
//! over blocks (`fixtures/scene.depth.f32`: 8 × 8, `scene.matte.f32`:
//! 16 × 16; made by the official code in full precision on the same
//! pixels). The weights: `JODEL_DEPTH_MODEL` and `JODEL_MATTE_MODEL`, else
//! the player's data folder's; skipped with a note where they or a usable
//! GPU are not.

use std::path::PathBuf;

use jodel_gpu::vision::check::{pooled, scene, stats};
use jodel_gpu::vision::prep;
use jodel_gpu::{GpuDepth, GpuMatte};
use jodel_tensors::SafeTensors;

fn fixture(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The checkpoint named by `var`, else the data folder's `file` (mapped,
/// as the player opens it).
fn checkpoint(var: &str, file: &str) -> Option<SafeTensors> {
    let path = std::env::var_os(var).map(PathBuf::from).or_else(|| {
        let data = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        }?;
        Some(data.join("Jodeldiplom Data/models").join(file))
    })?;
    let tensors = SafeTensors::open(&path).ok();
    if tensors.is_none() {
        eprintln!("no {file} here ({var}): skipped");
    }
    tensors
}

#[test]
fn the_depth_is_the_official_models() {
    let Some(tensors) = checkpoint("JODEL_DEPTH_MODEL", "da3-small.safetensors") else {
        return;
    };
    let depth = match GpuDepth::new(&tensors) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no usable GPU ({e}): skipped");
            return;
        }
    };
    let image = scene(504);
    let raw = depth.raw(&prep::normalise(&image), 504, 504).unwrap();
    let ours: Vec<f32> = raw.chunks_exact(2).map(|v| v[0].exp()).collect();
    let s = stats(&pooled(&ours, 504, 504, 8), &fixture("scene.depth.f32"));
    eprintln!("depth against the reference: {s:?}");
    assert!(s.correlation > 0.99999, "{s:?}");
    assert!(s.max_rel < 0.005, "{s:?}");
    // The figure stands in front of the floor behind it, the sky is
    // farthest.
    let at = |x: usize, y: usize| ours[y * 504 + x];
    assert!(
        at(252, 340) < at(100, 300),
        "the figure nearer than the floor beside it"
    );
    assert!(
        at(252, 20) > at(252, 480),
        "the sky farther than the floor in front"
    );
}

#[test]
fn the_matte_is_the_official_models() {
    let Some(tensors) = checkpoint("JODEL_MATTE_MODEL", "ben2-base.safetensors") else {
        return;
    };
    let matte = match GpuMatte::new(&tensors) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("no usable GPU ({e}): skipped");
            return;
        }
    };
    let ours = matte.raw(&prep::normalise(&scene(1024))).unwrap();
    let s = stats(&pooled(&ours, 1024, 1024, 16), &fixture("scene.matte.f32"));
    eprintln!("matte against the reference: {s:?}");
    assert!(s.correlation > 0.9999, "{s:?}");
    assert!(s.max_abs < 0.01, "{s:?}");
    // The figure is the subject; the sky is not.
    assert!(ours[600 * 1024 + 512] > 0.9);
    assert!(ours[40 * 1024 + 100] < 0.1);
}
