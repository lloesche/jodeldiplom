//! Gemma 4 E4B-it on the GPU ([`jodel_gpu::Gemma`]): packs the checkpoint,
//! or writes greedily after a prompt (one user turn, the chat template's)
//! and says how long it took and what it held.
//!
//! ```text
//! cargo run --release -p jodel-gpu --example gemma -- pack MODEL_DIR OUT.jdlgemma
//! cargo run --release -p jodel-gpu --example gemma -- run MODEL.jdlgemma PROMPT.txt OUT.txt [MAX_TOKENS]
//! cargo run --release -p jodel-gpu --example gemma -- logits MODEL.jdlgemma PROMPT.txt OUT.f32
//! ```
//!
//! `run` writes the text to `OUT.txt`, the prompt's and the answer's token
//! ids to `OUT.txt.ids` (a line each), and prints the load time, the
//! prompt's and the answer's tokens a second and the GPU memory held;
//! with `JODEL_GEMMA_PROFILE` set it writes again, profiled
//! ([`jodel_gpu::profile`]), and prints the kernels' GPU time. `logits`
//! writes the logits after the prompt (f32s).

use std::path::Path;
use std::time::Instant;

use jodel_gpu::llm::{self, Memory};
use jodel_gpu::{profile, Gemma};

fn write(path: &str, bytes: &[u8]) {
    durable::write_atomic(Path::new(path), bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

fn open(model: &str) -> Gemma {
    let started = Instant::now();
    let gemma = Gemma::open(Path::new(model)).unwrap();
    println!(
        "loaded on {} in {:.2} s ({:?})",
        gemma.name(),
        started.elapsed().as_secs_f64(),
        gemma.precision().unwrap()
    );
    gemma
}

fn prompt(gemma: &Gemma, file: &str) -> Vec<u32> {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
    gemma.tokens().chat(&text)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("pack") => {
            let started = Instant::now();
            llm::pack(Path::new(&args[1]), Path::new(&args[2]), |what| {
                println!("{:7.1} s  {what}", started.elapsed().as_secs_f64());
            })
            .unwrap();
        }
        Some("logits") => {
            let gemma = open(&args[1]);
            let ids = prompt(&gemma, &args[2]);
            gemma.reserve(ids.len() + 1).unwrap();
            let logits = gemma.logits(&ids).unwrap();
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            write(&args[3], &bytes);
            println!("{} prompt tokens, {} logits", ids.len(), logits.len());
        }
        Some("run") => {
            let most: usize = args.get(4).map_or(2500, |n| n.parse().unwrap());
            let gemma = open(&args[1]);
            let ids = prompt(&gemma, &args[2]);
            let context = ids.len() + most;
            let planned = Memory::of(context);
            gemma.reserve(context).unwrap();
            let held = gemma.memory().unwrap();
            println!(
                "a context of {context}: weights {:.0} MB, context {:.0} MB (planned {:.0} MB)",
                mb(held.weights),
                mb(held.context),
                mb(planned.context)
            );
            let mut written = 0;
            let out = gemma
                .generate(&ids, most, |tokens| {
                    written += tokens.len();
                    eprint!("\r{written} tokens");
                })
                .unwrap();
            eprintln!();
            let prompt_rate = out.prompt_tokens as f64 / out.prompt_time.as_secs_f64();
            let step_rate = out.tokens.len() as f64 / out.step_time.as_secs_f64();
            println!(
                "prompt: {} tokens in {:.3} s ({prompt_rate:.0} tokens a second)",
                out.prompt_tokens,
                out.prompt_time.as_secs_f64()
            );
            println!(
                "answer: {} tokens in {:.2} s ({step_rate:.1} tokens a second; {} steps; {})",
                out.tokens.len(),
                out.step_time.as_secs_f64(),
                out.steps,
                if out.stopped {
                    "stopped"
                } else {
                    "at the limit"
                }
            );
            println!(
                "GPU memory held: {:.0} MB",
                mb(gemma.memory().unwrap().total())
            );
            let text = gemma.tokens().decode(&out.tokens);
            write(&args[3], text.as_bytes());
            let lines: Vec<String> = ids
                .iter()
                .chain(&out.tokens)
                .map(|t| t.to_string())
                .collect();
            write(&format!("{}.ids", args[3]), lines.join("\n").as_bytes());
            if std::env::var_os("JODEL_GEMMA_PROFILE").is_some() {
                profile::start(profile::Detail::Kernels);
                let again = gemma.generate(&ids, most, |_| {}).unwrap();
                let p = profile::take();
                assert_eq!(again.tokens, out.tokens, "the same answer profiled");
                println!(
                    "profiled: GPU {:.2} s in {} dispatches",
                    p.kernels.iter().map(|k| k.gpu.as_secs_f64()).sum::<f64>(),
                    p.kernels.iter().map(|k| k.calls).sum::<u64>()
                );
                for k in p.kernels.iter().take(14) {
                    println!(
                        "  {:10} {:8} calls  {:8.3} s  (typical {:.3} s)",
                        k.kernel,
                        k.calls,
                        k.gpu.as_secs_f64(),
                        k.typical.as_secs_f64()
                    );
                }
            }
        }
        _ => eprintln!(
            "gemma pack MODEL_DIR OUT | run MODEL PROMPT OUT [MAX] | logits MODEL PROMPT OUT"
        ),
    }
}
