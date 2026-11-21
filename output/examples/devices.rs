//! Lists the output devices and what each accepts exclusively.
//!
//! `cargo run -p jodel-output --example devices`
//!
//! Plays nothing: the capability probe only asks the driver which formats
//! it would accept.

use jodel_output::{capabilities, devices, SampleFormat, BACKEND, PROBE_RATES};

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
        let caps = match capabilities(&device.id) {
            Ok(caps) => caps,
            Err(e) => {
                println!("    probe failed: {e}");
                continue;
            }
        };
        if caps.is_empty() {
            println!("    accepts nothing exclusively at the probed rates");
            continue;
        }
        for channels in caps.channel_counts() {
            println!("    exclusive, {channels} ch (ok = accepted, - = refused)");
            print!("    {:>7}", "rate");
            for f in SampleFormat::ALL {
                print!(" {:>9}", f.name());
            }
            println!();
            for rate in PROBE_RATES {
                let accepted = caps.formats(rate, channels);
                print!("    {rate:>7}");
                for f in SampleFormat::ALL {
                    print!(" {:>9}", if accepted.contains(&f) { "ok" } else { "-" });
                }
                println!();
            }
        }
    }
}
