//! The cover networks against their authors' PyTorch outputs.
//!
//! `vision_check scene DIR` writes the generated scenes the tests use
//! (`scene504.ppm`, `scene1024.ppm`) for the reference to run on.
//!
//! `vision_check ref ROOT [NAME...]` runs Depth Anything 3 Small and BEN2 on
//! the inputs the reference script dumped (`ROOT/ref/NAME.da3in.f32`,
//! `NAME.ben2in.f32`, shapes in `NAME.json`), so the networks see exactly
//! what PyTorch saw, compares the outputs (`NAME.da3.f32`, `NAME.ben2.f32`;
//! BEN2's official half-precision run `NAME.ben2h.f32` as a yardstick) and
//! writes ours to `ROOT/ours/`. The weights: `JODEL_DEPTH_MODEL` and
//! `JODEL_MATTE_MODEL`, else the data folder's.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_gpu::vision::check::{scene, stats};
use jodel_gpu::{GpuDepth, GpuMatte};
use jodel_lyrics::json::Json;

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[expect(clippy::disallowed_methods, reason = "debt (P6): its comparison dumps")]
fn write_f32s(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn model(var: &str, file: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA"))
            .join("Jodeldiplom Data/models")
            .join(file)
    })
}

#[expect(clippy::disallowed_methods, reason = "debt (P6): its comparison dumps")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("scene") => {
            let dir = PathBuf::from(&args[1]);
            for side in [504, 1024] {
                let s = scene(side);
                let mut ppm = format!("P6\n{side} {side}\n255\n").into_bytes();
                ppm.extend(&s.data);
                std::fs::write(dir.join(format!("scene{side}.ppm")), ppm).unwrap();
            }
        }
        Some("ref") => reference(Path::new(&args[1]), &args[2..]),
        _ => eprintln!("usage: vision_check scene DIR | ref ROOT [NAME...]"),
    }
}

fn reference(root: &Path, only: &[String]) {
    let started = Instant::now();
    let depth =
        GpuDepth::new(std::fs::read(model("JODEL_DEPTH_MODEL", "da3-small.safetensors")).unwrap())
            .unwrap_or_else(|e| panic!("depth: {e}"));
    let depth_setup = started.elapsed();
    let started = Instant::now();
    let matte =
        GpuMatte::new(std::fs::read(model("JODEL_MATTE_MODEL", "ben2-base.safetensors")).unwrap())
            .unwrap_or_else(|e| panic!("matte: {e}"));
    println!(
        "on {}: set up in {:.2} s and {:.2} s",
        depth.name(),
        depth_setup.as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    let refs = root.join("ref");
    let ours = root.join("ours");
    std::fs::create_dir_all(&ours).unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&refs)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_suffix(".json").map(str::to_owned)
        })
        .filter(|n| only.is_empty() || only.contains(n))
        .collect();
    names.sort();
    let (mut dt, mut mt) = (Vec::new(), Vec::new());
    println!("cover       depth: corr  max|d|/range  mean|d|/range   matte: corr  max|d|  mean|d|  (fp16 ref: max|d| mean|d|)");
    for name in names {
        let meta =
            Json::parse(&std::fs::read_to_string(refs.join(format!("{name}.json"))).unwrap())
                .unwrap();
        let shape = meta.get("da3").and_then(Json::as_array).unwrap();
        let (h, w) = (
            shape[0].as_f64().unwrap() as usize,
            shape[1].as_f64().unwrap() as usize,
        );
        let input = f32s(&refs.join(format!("{name}.da3in.f32")));
        let mut raw = Vec::new();
        for _ in 0..2 {
            let t = Instant::now();
            raw = depth.raw(&input, w, h).unwrap();
            dt.push(t.elapsed().as_secs_f64());
        }
        let ours_d: Vec<f32> = raw.chunks_exact(2).map(|v| v[0].exp()).collect();
        let ref_d = f32s(&refs.join(format!("{name}.da3.f32")));
        let ds = stats(&ours_d, &ref_d);
        write_f32s(&ours.join(format!("{name}.da3.f32")), &ours_d);

        let input = f32s(&refs.join(format!("{name}.ben2in.f32")));
        let mut ours_m = Vec::new();
        for _ in 0..2 {
            let t = Instant::now();
            ours_m = matte.raw(&input).unwrap();
            mt.push(t.elapsed().as_secs_f64());
        }
        let ref_m = f32s(&refs.join(format!("{name}.ben2.f32")));
        let half = f32s(&refs.join(format!("{name}.ben2h.f32")));
        let ms = stats(&ours_m, &ref_m);
        let hs = stats(&half, &ref_m);
        write_f32s(&ours.join(format!("{name}.ben2.f32")), &ours_m);
        println!(
            "{name}  {:.6} {:.5} {:.6}   {:.6} {:.4} {:.6}   ({:.4} {:.6})",
            ds.correlation,
            ds.max_rel,
            ds.mean_rel,
            ms.correlation,
            ms.max_abs,
            ms.mean_abs,
            hs.max_abs,
            hs.mean_abs
        );
    }
    let ((dw, dp), (mw, mp)) = (depth.gpu_sizes(), matte.gpu_sizes());
    println!(
        "on the GPU: depth {} MB of weights and {} MB of working buffers, matte {} MB and {} MB",
        dw >> 20,
        dp >> 20,
        mw >> 20,
        mp >> 20
    );
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2] * 1000.0
    };
    println!(
        "median: depth {:.1} ms, matte {:.1} ms a cover",
        median(&mut dt),
        median(&mut mt)
    );
    if std::env::var_os("JODEL_VISION_PROFILE").is_some() {
        let input = vec![0.1f32; 504 * 504 * 3];
        let show = |what: &str, rows: Vec<(String, usize, f64)>| {
            let total: f64 = rows.iter().map(|r| r.2).sum();
            println!("{what}: {total:.1} ms in single dispatches");
            for (k, n, ms) in rows.iter().take(14) {
                println!("  {ms:8.2} ms  {n:4}x  {k}");
            }
        };
        show("depth", depth.profile(&input, 504, 504).unwrap());
        show(
            "matte",
            matte.profile(&vec![0.1f32; 1024 * 1024 * 3]).unwrap(),
        );
    }
}
