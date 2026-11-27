//! What this GPU really does ([`jodel_gpu::probe`]): the report printed
//! and written to a text file (`gpu-probe.txt` in the current folder, or
//! the path given).
//!
//! ```text
//! cargo run --release -p jodel-gpu --example probe [-- REPORT.txt]
//! ```

use std::path::PathBuf;

fn main() {
    let path = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("gpu-probe.txt"), PathBuf::from);
    let report = match jodel_gpu::probe::run() {
        Ok(report) => report.to_string(),
        Err(e) => format!("GPU probe: no GPU to probe: {e}\n"),
    };
    print!("{report}");
    match durable::write_atomic(&path, report.as_bytes()) {
        Ok(()) => println!("\nwritten to {}", path.display()),
        Err(e) => eprintln!("\n{}: {e}", path.display()),
    }
}
