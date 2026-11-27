//! Pictures from FLUX.2 [klein] 4B on the GPU (`jodel_gpu::image`): a
//! prompt, a seed and reference pictures to a PNG; the bake-off's two
//! scenes, timed and held against its pictures; or each part held against
//! the PyTorch reference's dumps.
//!
//! ```text
//! cargo run --release -p jodel-gpu --example klein -- MODELS --prompt TEXT [--seed N] [--ref PNG]... [--size 1344x768] --out PNG
//! cargo run --release -p jodel-gpu --example klein -- MODELS --bakeoff DIR --out DIR [--turns]
//! cargo run --release -p jodel-gpu --example klein -- MODELS --check DIR
//! ```
//!
//! `MODELS` holds `Qwen__Qwen3-4B-FP8/` (Qwen's FP8 release, with its
//! tokenizer), `black-forest-labs__FLUX.2-klein-4b-fp8/` (BFL's FP8
//! transformer) and `black-forest-labs__FLUX.2-klein-4B/` (the
//! autoencoder in its `vae/`), each in its repository's own layout. `--bakeoff DIR` renders the two scenes of the bake-off
//! (`DIR/rust/ref/tokens.json`: their prompts, seeds and references, the
//! references' 512 × 512 crops in `DIR/rust/refs/`) after a first picture
//! to warm up, prints the load times, the seconds per picture (and the
//! GPU's time for it, from its submissions' stamps) and the GPU memory at
//! its peak, and compares each with the bake-off's
//! (`DIR/quick/klein-*.png`); `--turns` frees the text encoder before the
//! transformer loads (a card of 8 GB), else both stay loaded. `--check DIR`
//! holds the tokens, the noise, the encoder's hidden states, the
//! references' latents, the transformer's first step, the decoder and the
//! whole picture against the reference's f32 dumps in `DIR/rust/ref/`.
//! `--full` runs the painter in f32 throughout (no tensor cores; the text
//! encoder always is: its f16 inputs cost the picture ten times its own
//! rounding, for 0.1 s a prompt); `--profile` prints each kernel's GPU
//! time for one picture.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_gpu::image::{noise, Embeds, Painter, Part, Picture, Precision, TextEncoder, Tokenizer};
use jodel_gpu::profile;
use jodel_tensors::json::Json;
use jodel_tensors::SafeTensors;

const QWEN: &str = "Qwen__Qwen3-4B-FP8";
const KLEIN: &str = "black-forest-labs__FLUX.2-klein-4b-fp8";
const VAE: &str = "black-forest-labs__FLUX.2-klein-4B";

struct Args {
    models: PathBuf,
    prompt: Option<String>,
    seed: u64,
    refs: Vec<PathBuf>,
    size: (usize, usize),
    out: Option<PathBuf>,
    bakeoff: Option<PathBuf>,
    check: Option<PathBuf>,
    precision: Precision,
    turns: bool,
    profile: bool,
}

fn args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        models: PathBuf::from(it.next().expect("MODELS")),
        prompt: None,
        seed: 0,
        refs: Vec::new(),
        size: (1344, 768),
        out: None,
        bakeoff: None,
        check: None,
        precision: Precision::Fast,
        turns: false,
        profile: false,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} wants a value"));
        match flag.as_str() {
            "--prompt" => a.prompt = Some(value()),
            "--seed" => a.seed = value().parse().expect("a seed"),
            "--ref" => a.refs.push(value().into()),
            "--size" => {
                let v = value();
                let (w, h) = v.split_once('x').expect("WxH");
                a.size = (w.parse().expect("a width"), h.parse().expect("a height"));
            }
            "--out" => a.out = Some(value().into()),
            "--bakeoff" => a.bakeoff = Some(value().into()),
            "--check" => a.check = Some(value().into()),
            "--full" => a.precision = Precision::Full,
            "--turns" => a.turns = true,
            "--profile" => a.profile = true,
            other => panic!("unknown {other}"),
        }
    }
    a
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64)
}

/// A PNG as a reference picture.
fn read_png(path: &Path) -> Picture {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8();
    Picture {
        width: img.width() as usize,
        height: img.height() as usize,
        rgb: img.into_raw(),
    }
}

/// A picture to a PNG, through a durable file.
fn write_png(path: &Path, p: &Picture) {
    let img = image::RgbImage::from_raw(p.width as u32, p.height as u32, p.rgb.clone())
        .expect("a picture");
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .expect("PNG");
    if let Some(dir) = path.parent() {
        durable::create_dir_all(dir).expect("the folder");
    }
    durable::write_atomic(path, &bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The mean absolute difference (0..255) and PSNR of two pictures.
fn pixel_error(a: &Picture, b: &Picture) -> (f64, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let n = a.rgb.len() as f64;
    let (mut abs, mut sq) = (0.0, 0.0);
    for (x, y) in a.rgb.iter().zip(&b.rgb) {
        let d = f64::from(*x) - f64::from(*y);
        abs += d.abs();
        sq += d * d;
    }
    (
        abs / n,
        10.0 * (255.0f64.powi(2) / (sq / n).max(1e-9)).log10(),
    )
}

/// The GPU's memory at its peak so far (its allocator's, sampled).
struct Peak(u64);

impl Peak {
    fn sample(&mut self, bytes: Option<u64>) {
        self.0 = self.0.max(bytes.unwrap_or(0));
    }
}

/// One picture from the command line.
fn single(a: &Args) {
    let prompt = a.prompt.as_deref().expect("--prompt");
    let t = Instant::now();
    let mut encoder = TextEncoder::load(&a.models.join(QWEN), Precision::Full).unwrap();
    let embeds = encoder.encode(prompt).unwrap();
    drop(encoder);
    println!("prompt encoded in {:.1} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let mut painter =
        Painter::load(&a.models.join(KLEIN), &a.models.join(VAE), a.precision).unwrap();
    println!("painter loaded in {:.1} s", t.elapsed().as_secs_f64());
    let refs: Vec<Picture> = a.refs.iter().map(|p| read_png(p)).collect();
    let t = Instant::now();
    if a.profile {
        profile::start(profile::Detail::Kernels);
    }
    let picture = painter.paint(&embeds, a.seed, a.size, &refs).unwrap();
    if a.profile {
        print_profile();
    }
    println!("painted in {:.2} s", t.elapsed().as_secs_f64());
    write_png(a.out.as_deref().expect("--out"), &picture);
}

fn print_profile() {
    let p = profile::take();
    let mut kernels = p.kernels.clone();
    kernels.sort_by_key(|k| std::cmp::Reverse(k.gpu));
    let total: f64 = kernels.iter().map(|k| k.gpu.as_secs_f64()).sum();
    println!("GPU time by kernel ({total:.3} s in all):");
    for k in kernels.iter().take(12) {
        println!(
            "  {:<14} {:>6} dispatches {:>8.1} ms ({:.0} %)",
            k.kernel,
            k.calls,
            k.gpu.as_secs_f64() * 1e3,
            100.0 * k.gpu.as_secs_f64() / total
        );
    }
    let mut shapes = p.shapes.clone();
    shapes.sort_by_key(|s| std::cmp::Reverse(s.typical));
    println!("the costliest dispatches by shape (workgroups):");
    for s in shapes.iter().take(12) {
        println!(
            "  {:<14} {:>5}x{:<4} {:>5} dispatches {:>8.1} ms",
            s.kernel,
            s.groups[0],
            s.groups[1],
            s.calls,
            s.typical.as_secs_f64() * 1e3
        );
    }
}

/// The bake-off's scenes: their tag, prompt, seed and references.
fn scenes(dir: &Path) -> Vec<(String, String, u64, Vec<PathBuf>)> {
    let text = std::fs::read_to_string(dir.join("rust/ref/tokens.json")).expect("tokens.json");
    let Json::Object(all) = Json::parse(&text).expect("JSON") else {
        panic!("tokens.json is not an object")
    };
    let sid = |tag: &str| if tag == "lsd" { "thunder" } else { tag }.to_owned();
    all.iter()
        .filter(|(k, _)| k.as_str() != "tests")
        .map(|(tag, v)| {
            let refs = v
                .get("refs")
                .and_then(Json::as_array)
                .unwrap_or(&[])
                .iter()
                .map(|c| {
                    dir.join(format!(
                        "rust/refs/{}_{}.png",
                        sid(tag),
                        c.as_str().unwrap()
                    ))
                })
                .collect();
            (
                tag.clone(),
                v.get("prompt").and_then(Json::as_str).unwrap().to_owned(),
                v.get("seed").and_then(Json::as_f64).unwrap() as u64,
                refs,
            )
        })
        .collect()
}

/// The bake-off's two scenes, timed, held against its pictures.
fn bakeoff(a: &Args, dir: &Path) {
    let out = a.out.clone().expect("--out");
    let scenes = scenes(dir);
    let mut peak = Peak(0);
    println!(
        "planned: encoder {} + {} working; painter {} + {} working at {}x{} with a 512² reference",
        gb(TextEncoder::footprint().resident),
        gb(TextEncoder::footprint().working),
        gb(Painter::footprint(a.size.0, a.size.1, &[(512, 512)]).resident),
        gb(Painter::footprint(a.size.0, a.size.1, &[(512, 512)]).working),
        a.size.0,
        a.size.1
    );
    let t = Instant::now();
    let mut encoder = TextEncoder::load(&a.models.join(QWEN), Precision::Full).unwrap();
    let encoder_load = t.elapsed();
    let t = Instant::now();
    let embeds: Vec<Embeds> = scenes
        .iter()
        .map(|s| encoder.encode(&s.1).unwrap())
        .collect();
    let encoding = t.elapsed().as_secs_f64() / scenes.len() as f64;
    let encoder_bytes = encoder.bytes();
    peak.sample(encoder.reserved());
    let mut kept = None;
    if a.turns {
        drop(encoder);
    } else {
        kept = Some(encoder);
    }
    let t = Instant::now();
    let mut painter =
        Painter::load(&a.models.join(KLEIN), &a.models.join(VAE), a.precision).unwrap();
    let painter_load = t.elapsed();
    println!(
        "{} ({}), {}",
        painter.gpu_name(),
        if painter.tensor_cores() {
            "tensor cores"
        } else {
            "f32"
        },
        if a.turns {
            "the encoder freed before the painter loaded"
        } else {
            "both kept loaded"
        }
    );
    println!(
        "load: text encoder {:.1} s ({} on the GPU), painter {:.1} s",
        encoder_load.as_secs_f64(),
        gb(encoder_bytes.resident),
        painter_load.as_secs_f64()
    );
    println!("encoding: {encoding:.3} s a prompt");
    let pictures: Vec<Vec<Picture>> = scenes
        .iter()
        .map(|s| s.3.iter().map(|p| read_png(p)).collect())
        .collect();
    // Warm-up: the first scene once.
    let t = Instant::now();
    painter
        .paint(&embeds[0], scenes[0].2, a.size, &pictures[0])
        .unwrap();
    println!("warm-up picture: {:.2} s", t.elapsed().as_secs_f64());
    peak.sample(painter.reserved());
    for (i, (tag, _, seed, _)) in scenes.iter().enumerate() {
        let t = Instant::now();
        let picture = painter
            .paint(&embeds[i], *seed, a.size, &pictures[i])
            .unwrap();
        let took = t.elapsed().as_secs_f64();
        peak.sample(painter.reserved());
        // Again with each submission stamped: the GPU's own time (the wall
        // time also counts other work on the machine and the GPU).
        profile::start(profile::Detail::Submissions);
        painter
            .paint(&embeds[i], *seed, a.size, &pictures[i])
            .unwrap();
        let gpu: f64 = profile::take()
            .kernels
            .iter()
            .map(|k| k.gpu.as_secs_f64())
            .sum();
        let path = out.join(format!(
            "klein-{tag}{}.png",
            if a.precision == Precision::Full {
                "-f32"
            } else {
                ""
            }
        ));
        write_png(&path, &picture);
        let quick = read_png(&dir.join(format!("quick/klein-{tag}.png")));
        let (mad, psnr) = pixel_error(&picture, &quick);
        let f32_ref = read_png(&dir.join(format!("rust/ref/f32_{tag}.png")));
        let (mad32, psnr32) = pixel_error(&picture, &f32_ref);
        println!(
            "{tag}: {took:.2} s ({gpu:.2} s on the GPU; {} references); against the bake-off's PNG: mean |Δ| {mad:.2}, PSNR {psnr:.1} dB; against the f32 reference: mean |Δ| {mad32:.2}, PSNR {psnr32:.1} dB -> {}",
            pictures[i].len(),
            path.display()
        );
    }
    let bytes = painter.bytes();
    println!(
        "GPU memory: painter {} + {} working{}; the device's allocator at its peak {}",
        gb(bytes.resident),
        gb(bytes.working),
        kept.as_ref().map_or(String::new(), |e| format!(
            ", text encoder {} + {}",
            gb(e.bytes().resident),
            gb(e.bytes().working)
        )),
        gb(peak.0)
    );
    if a.profile {
        profile::start(profile::Detail::Kernels);
        let t = Instant::now();
        painter
            .paint(&embeds[0], scenes[0].2, a.size, &pictures[0])
            .unwrap();
        println!("profiled picture: {:.2} s", t.elapsed().as_secs_f64());
        print_profile();
    }
}

/// The largest and the root-mean-square difference of `got` from `want`,
/// each against `want`'s root mean square.
fn error(got: &[f32], want: &[f32]) -> String {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    let (mut most, mut sq) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = f64::from(g - w).abs();
        most = most.max(d);
        sq += d * d;
    }
    let drms = (sq / want.len() as f64).sqrt();
    format!(
        "max |Δ| {most:.3e}, rms Δ {drms:.3e} (rms {rms:.3e}: {:.2e} of it)",
        drms / rms
    )
}

/// `[c][n]` as `[n][c]`.
fn transposed(x: &[f32], c: usize, n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|i| (0..c).map(move |j| x[j * n + i]))
        .collect()
}

/// Each part against the reference's dumps.
fn check(a: &Args, dir: &Path) {
    let r = dir.join("rust/ref");
    let open = |name: &str| SafeTensors::open(&r.join(name)).unwrap();
    let scenes = scenes(dir);
    let text = std::fs::read_to_string(r.join("tokens.json")).unwrap();
    let tokens = Json::parse(&text).unwrap();
    // The tokens.
    let tok = Tokenizer::open(&a.models.join(QWEN)).unwrap();
    for (tag, prompt, _, _) in &scenes {
        let (ids, real) = tok.prompt(prompt).unwrap();
        let want: Vec<u32> = tokens
            .get(tag)
            .and_then(|v| v.get("ids"))
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as u32)
            .collect();
        println!(
            "tokens {tag}: {} of {} as the reference's ({real} the prompt's)",
            ids.iter().zip(&want).filter(|(a, b)| a == b).count(),
            want.len()
        );
    }
    for t in tokens.get("tests").and_then(Json::as_array).unwrap() {
        let s = t.get("text").and_then(Json::as_str).unwrap();
        let want: Vec<u32> = t
            .get("ids")
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as u32)
            .collect();
        let got = tok.tokens(s).unwrap();
        println!(
            "tokens {s:?}: {}",
            if got == want {
                "same".to_owned()
            } else {
                format!("{got:?} against {want:?}")
            }
        );
    }
    // The noise.
    let n = open("noise.safetensors");
    for (tag, _, seed, _) in &scenes {
        let want = n.get(&format!("{tag}_bf16"), &[1, 128, 48, 84]).unwrap();
        let got = noise(*seed, want.len());
        let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        println!(
            "noise {tag}: {same} of {} the same, {}",
            want.len(),
            error(&got, &want)
        );
    }
    // The encoder.
    let t = Instant::now();
    let mut encoder = TextEncoder::load(&a.models.join(QWEN), Precision::Full).unwrap();
    println!("text encoder loaded in {:.1} s", t.elapsed().as_secs_f64());
    let mut reference_embeds = Vec::new();
    for (tag, prompt, _, _) in &scenes {
        let t = Instant::now();
        let got = encoder.encode(prompt).unwrap();
        println!("encoded {tag} in {:.3} s", t.elapsed().as_secs_f64());
        for kind in ["f32", "bf16"] {
            let e = open(&format!("enc_{tag}_{kind}.safetensors"));
            for (j, layer) in ["h9", "h18", "h27"].iter().enumerate() {
                let want = e.get(layer, &[512, 2560]).unwrap();
                let mine: Vec<f32> = got
                    .values
                    .chunks(7680)
                    .flat_map(|row| row[j * 2560..(j + 1) * 2560].to_vec())
                    .collect();
                println!(
                    "encoder {tag} {layer} against the {kind} reference: {}",
                    error(&mine, &want)
                );
            }
        }
        let e = open(&format!("enc_{tag}_f32.safetensors"));
        let parts: Vec<Vec<f32>> = ["h9", "h18", "h27"]
            .iter()
            .map(|l| e.get(l, &[512, 2560]).unwrap())
            .collect();
        reference_embeds.push(Embeds {
            values: (0..512)
                .flat_map(|i| {
                    parts
                        .iter()
                        .flat_map(move |p| p[i * 2560..(i + 1) * 2560].to_vec())
                })
                .collect(),
        });
    }
    drop(encoder);
    // The painter's parts, from the reference's inputs.
    let mut painter =
        Painter::load(&a.models.join(KLEIN), &a.models.join(VAE), a.precision).unwrap();
    let size = (1344, 768);
    for (i, (tag, _, seed, refs)) in scenes.iter().enumerate() {
        let pictures: Vec<Picture> = refs.iter().map(|p| read_png(p)).collect();
        let start = n.get(&format!("{tag}_bf16"), &[1, 128, 48, 84]).unwrap();
        let rf = open(&format!("refs_{tag}.safetensors"));
        let got = painter
            .run(
                Part::References,
                &reference_embeds[i],
                &start,
                size,
                &pictures,
            )
            .unwrap();
        let names: Vec<String> = tokens
            .get(tag)
            .and_then(|v| v.get("refs"))
            .and_then(Json::as_array)
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_owned())
            .collect();
        let want: Vec<f32> = names
            .iter()
            .flat_map(|c| {
                transposed(
                    &rf.get(&format!("{c}_latent"), &[128, 32, 32]).unwrap(),
                    128,
                    1024,
                )
            })
            .collect();
        println!("references' latents {tag}: {}", error(&got, &want));
        let run = open(&format!("run_{tag}.safetensors"));
        let got = painter
            .run(
                Part::FirstStep,
                &reference_embeds[i],
                &start,
                size,
                &pictures,
            )
            .unwrap();
        let want = run.get("step0_pred", &[4032, 128]).unwrap();
        println!("first step {tag}: {}", error(&got, &want));
        let latents = run.get("final_packed", &[4032, 128]).unwrap();
        let got = painter
            .run(Part::Decode, &reference_embeds[i], &latents, size, &[])
            .unwrap();
        let want = transposed(&run.get("image", &[3, 768, 1344]).unwrap(), 3, 768 * 1344);
        println!("decoder {tag}: {}", error(&got, &want));
        let picture = painter
            .paint(&reference_embeds[i], *seed, size, &pictures)
            .unwrap();
        let f32_ref = read_png(&r.join(format!("f32_{tag}.png")));
        let (mad, psnr) = pixel_error(&picture, &f32_ref);
        let quick = read_png(&dir.join(format!("quick/klein-{tag}.png")));
        let (qmad, qpsnr) = pixel_error(&picture, &quick);
        println!("picture {tag} (the reference's embeddings): against the f32 reference mean |Δ| {mad:.2}, PSNR {psnr:.1} dB; against the bake-off's PNG mean |Δ| {qmad:.2}, PSNR {qpsnr:.1} dB");
    }
}

fn main() {
    let a = args();
    if let Some(dir) = a.check.clone() {
        check(&a, &dir);
    } else if let Some(dir) = a.bakeoff.clone() {
        bakeoff(&a, &dir);
    } else {
        single(&a);
    }
}
