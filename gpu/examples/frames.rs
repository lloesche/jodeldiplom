//! A window's frames on the GPU while the story's models work: a thread
//! stands for the window, a frame every 16.7 ms on a device of its own (a
//! pass over a 2560 × 1440 picture, as a window's frame), timed from its
//! submission until the GPU has done it; meanwhile background work (as the
//! player's story makes it: [`jodel_gpu::pace::background`]) paints a
//! picture with klein, then writes with Gemma, each model's weights put up
//! first (some 13 GB in all), then puts up the scenes' depth and cut-out
//! networks. It prints each model's load's time, and for each part (each
//! model's load on its own) the frames' latencies (median, 95th and 99th
//! percentile, the longest), and the work's own speed (the picture's
//! seconds, Gemma's prompt of some 3,000 tokens and its tokens a second).
//!
//! ```text
//! cargo run --release -p jodel-gpu --example frames -- MODELS [--told] [--tokens N]
//! ```
//!
//! `MODELS` is the player's models folder (its `story/`: klein's
//! transformer, autoencoder and text encoder, and Gemma's GGUF; the depth
//! and cut-out checkpoints beside it).
//! `--told` tells the pace of each frame, late or not, as the player's
//! window tells it of its ticks ([`jodel_gpu::pace::tick`], nothing moving
//! smoothly). It tests how background work answers late frames, not how
//! the window finds them: the stand-in has no surface to get its frames
//! from and no ticks, so its late frame is one the GPU took a quarter of a
//! frame past its slot to do (the next frame starts that late, as a
//! window's next tick does when getting its frame waited), where the
//! player times the gaps between its ticks (`gui::paint`'s `Lateness`).
//! The player's own trace (its frames' line: ticks late, those starved,
//! the share) shows how that goes. Each part's line says how many frames
//! came late, and in how many background work was giving way. Without
//! `--told` the frames say nothing.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_gpu::image::{Drawer, TextEncoder};
use jodel_gpu::{Gemma, GpuDepth, GpuMatte, Precision};
use jodel_tensors::SafeTensors;

/// A frame's interval.
const FRAME: Duration = Duration::from_micros(16_667);
/// The picture a frame writes: 2560 × 1440.
const PIXELS: u32 = 2560 * 1440;

/// The parts frames are told apart by: the rest (the start, the prompt's
/// encoding), klein's picture, Gemma's prompt, Gemma's steps, each model's
/// load (its weights going up, its first plan's or context's room made:
/// klein's text encoder and drawer, some 8 GB; Gemma, some 5 GB; the
/// scenes' depth and cut-out), and each model's first use after its load
/// (klein's first picture and Gemma's first words, which set the plans up
/// and are the first work on the weights put up).
const PARTS: [&str; 11] = [
    "the rest",
    "klein",
    "gemma's prompt",
    "gemma's steps",
    "text encoder loading",
    "drawer loading",
    "gemma loading",
    "depth loading",
    "cut-out loading",
    "klein's first picture",
    "gemma's first words",
];

/// The window's stand-in: its own device, a pass a frame.
struct Window {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl Window {
    fn new() -> Window {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("a GPU");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("a device");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frames"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> picture: array<u32>;
                 @compute @workgroup_size(256)
                 fn frame(@builtin(global_invocation_id) id: vec3<u32>) {
                     let i = id.x + id.y * 65535u * 256u;
                     if (i < arrayLength(&picture)) {
                         let x = f32(i % 2560u) / 2560.0;
                         let y = f32(i / 2560u) / 1440.0;
                         let c = vec3<f32>(x, y, fract(x * y * 7.0));
                         picture[i] = pack4x8unorm(vec4<f32>(c, 1.0));
                     }
                 }"
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frames"),
            layout: None,
            module: &module,
            entry_point: Some("frame"),
            compilation_options: Default::default(),
            cache: None,
        });
        let picture = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frames"),
            size: u64::from(PIXELS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frames"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: picture.as_entire_binding(),
            }],
        });
        Window {
            device,
            queue,
            pipeline,
            group,
        }
    }

    /// A frame, from its submission until the GPU has done it.
    fn frame(&self) -> Duration {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(PIXELS.div_ceil(256), 1, 1);
        }
        let started = Instant::now();
        let index = self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("a frame");
        started.elapsed()
    }
}

/// Each part's frames: their latencies, and whether background work was
/// giving way then ([`jodel_gpu::pace::duty`] below 1).
type Latencies = Arc<Mutex<[Vec<(Duration, bool)>; PARTS.len()]>>;

/// A frame the GPU took this long to do is late: the next starts a quarter
/// of a frame past its slot.
const LATE: Duration = Duration::from_micros(16_667 + 16_667 / 4);

/// The window's thread: a frame every [`FRAME`] until `stop`, each one's
/// latency kept under the part `part` says; `told`: the pace is told of
/// each late frame ([`LATE`]).
fn window(
    told: bool,
    part: Arc<AtomicU8>,
    stop: Arc<AtomicBool>,
) -> (std::thread::JoinHandle<()>, Latencies) {
    let latencies: Latencies = Arc::default();
    let kept = Arc::clone(&latencies);
    let thread = std::thread::spawn(move || {
        let window = Window::new();
        let mut next = Instant::now();
        while !stop.load(Ordering::Acquire) {
            let giving = jodel_gpu::pace::duty() < 1.0;
            let took = window.frame();
            if told {
                jodel_gpu::pace::tick(took >= LATE, false);
            }
            let at = usize::from(part.load(Ordering::Acquire));
            kept.lock().unwrap()[at].push((took, giving));
            next += FRAME;
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            } else {
                next = now;
            }
        }
    });
    (thread, latencies)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The frames' line for a part.
fn summary(kept: Vec<(Duration, bool)>) -> String {
    if kept.is_empty() {
        return "no frames".into();
    }
    let giving = kept.iter().filter(|f| f.1).count();
    let mut frames: Vec<Duration> = kept.into_iter().map(|f| f.0).collect();
    frames.sort();
    let at = |q: f64| frames[((frames.len() - 1) as f64 * q).round() as usize];
    let late = frames.iter().filter(|&&d| d >= LATE).count();
    format!(
        "{} frames: latency median {:.2} ms, 95 % {:.2} ms, 99 % {:.2} ms, longest {:.2} ms; \
         {} late; background work giving way in {}",
        frames.len(),
        ms(at(0.5)),
        ms(at(0.95)),
        ms(at(0.99)),
        ms(frames[frames.len() - 1]),
        late,
        giving
    )
}

/// `load` run, its time told in `lines` as `what`'s load.
fn timed<T>(lines: &mut Vec<String>, what: &str, load: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let made = load();
    lines.push(format!(
        "{what}: loaded in {:.2} s",
        started.elapsed().as_secs_f64()
    ));
    made
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(&args[0]);
    let models = root.join("story");
    let told = args.iter().any(|a| a == "--told");
    let tokens: usize = args
        .iter()
        .position(|a| a == "--tokens")
        .map_or(300, |i| args[i + 1].parse().expect("a number of tokens"));
    jodel_gpu::pace::background();
    let part = Arc::new(AtomicU8::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (thread, latencies) = window(told, Arc::clone(&part), Arc::clone(&stop));
    std::thread::sleep(Duration::from_secs(2));
    let mut lines = Vec::new();
    {
        part.store(4, Ordering::Release);
        let mut encoder = timed(&mut lines, "text encoder", || {
            TextEncoder::load(&models.join("qwen3-4b-fp8"), Precision::Fast).unwrap()
        });
        part.store(0, Ordering::Release);
        let embeds = encoder
            .encode("A lighthouse on a cliff at dusk, its lamp lit, waves below")
            .unwrap();
        drop(encoder);
        part.store(5, Ordering::Release);
        let mut drawer = timed(&mut lines, "drawer", || {
            Drawer::load(
                &models.join("flux2-klein-4b-fp8"),
                &models.join("flux2-klein-4b"),
                Precision::Fast,
            )
            .unwrap()
        });
        // A first picture sets the plans up.
        part.store(9, Ordering::Release);
        drawer.paint(&embeds, 1, (1344, 768), &[]).unwrap();
        part.store(1, Ordering::Release);
        let started = Instant::now();
        drawer.paint(&embeds, 1, (1344, 768), &[]).unwrap();
        lines.push(format!(
            "klein: a picture in {:.2} s",
            started.elapsed().as_secs_f64()
        ));
        part.store(0, Ordering::Release);
    }
    {
        part.store(6, Ordering::Release);
        let gemma = timed(&mut lines, "gemma", || {
            Gemma::open(&models.join("gemma-4-E4B-it-Q4_K_M.gguf"), Precision::Fast).unwrap()
        });
        // A long prompt, as the story's are (its attention the longest
        // dispatches): some 3,000 tokens.
        let verse = "The keeper climbs the winding stair at dusk, the lamp is lit, the \
                     gulls go quiet, and far below the waves keep time against the rocks \
                     while ships pass by in the dark and nobody hears him sing. ";
        let prompt = gemma.tokens().chat(&format!(
            "Here is a song's text. Tell its story in about three hundred words.\n\n{}",
            verse.repeat(64)
        ));
        gemma.reserve(prompt.len() + tokens + 64).unwrap();
        part.store(10, Ordering::Release);
        gemma
            .generate(&prompt, 16, |_| std::ops::ControlFlow::Continue(()))
            .unwrap();
        part.store(2, Ordering::Release);
        // The steps from the first tokens read on.
        let made = gemma
            .generate(&prompt, tokens, |_| {
                part.store(3, Ordering::Release);
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        part.store(0, Ordering::Release);
        lines.push(format!(
            "gemma: a prompt of {} tokens in {:.3} s, then {} tokens at {:.1} a second",
            made.prompt_tokens,
            made.prompt_time.as_secs_f64(),
            made.tokens.len(),
            made.tokens.len() as f64 / made.step_time.as_secs_f64().max(1e-9)
        ));
    }
    // The scenes' depth and cut-out: their weights up, then let go.
    let open = |name: &str| SafeTensors::open(&root.join(name)).unwrap();
    part.store(7, Ordering::Release);
    drop(timed(&mut lines, "depth", || {
        GpuDepth::new(&open("da3-small.safetensors")).unwrap()
    }));
    part.store(8, Ordering::Release);
    drop(timed(&mut lines, "cut-out", || {
        GpuMatte::new(&open("ben2-base.safetensors")).unwrap()
    }));
    part.store(0, Ordering::Release);
    std::thread::sleep(Duration::from_secs(1));
    stop.store(true, Ordering::Release);
    thread.join().unwrap();
    let all = std::mem::take(&mut *latencies.lock().unwrap());
    println!(
        "{} ({})",
        jodel_gpu::gpu_name().unwrap_or_default(),
        if told {
            "late frames told"
        } else {
            "no frame told"
        }
    );
    for line in lines {
        println!("{line}");
    }
    for (name, frames) in PARTS.iter().zip(all) {
        println!("{name}: {}", summary(frames));
    }
}
