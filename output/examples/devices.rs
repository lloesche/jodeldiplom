//! Lists the output devices and the format each takes: its mix format, at
//! the rate the system runs it at.
//!
//! `cargo run -p jodel-output --example devices`
//!
//! Plays nothing: the system is only asked for each device's mix format.

use jodel_output::{devices, mix_format, BACKEND};

fn main() {
    println!("backend: {BACKEND}");
    let list = match devices() {
        Ok(list) => list,
        Err(e) => {
            eprintln!("listing devices failed: {e}");
            std::process::exit(1);
        }
    };
    println!("{} output device(s)", list.len());
    for (i, device) in list.iter().enumerate() {
        println!();
        println!(
            "[{i}] {}{}",
            device.name,
            if device.is_default { "  (default)" } else { "" }
        );
        println!("    id: {}", device.id);
        match mix_format(&device.id) {
            Ok(format) => println!("    takes {format}"),
            Err(e) => println!("    cannot be asked: {e}"),
        }
    }
}
