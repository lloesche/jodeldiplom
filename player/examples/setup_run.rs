//! Runs the player's setup ([`jodeldiplom::setup`]) without a screen: the
//! files the player needs got into the data folder in PARENT (downloaded,
//! packed, checked against their built-in checksums), the steps and their
//! progress printed. For checking setup from a clean start, and that
//! packing here gives the file the checksum was made from.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example setup_run -- PARENT
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodeldiplom::setup::{manifest, Places, Real, Setup, StepState};
use jodeldiplom::store::dirs::Data;

fn main() {
    let Some(parent) = std::env::args().nth(1) else {
        eprintln!("usage: setup_run PARENT");
        std::process::exit(2);
    };
    let data = Data::in_parent(Path::new(&parent));
    let places = Places {
        look: vec![data.models()],
        store: data.models(),
    };
    let started = Instant::now();
    let setup = Setup::start(manifest, places, Arc::new(Real), Arc::new(|| {}), None);
    let mut shown = (usize::MAX, u64::MAX);
    loop {
        let s = setup.status();
        if let Some(paths) = &s.finished {
            for (i, step) in s.steps.iter().enumerate() {
                println!(
                    "{} {i}: {}",
                    if step.state == StepState::Done {
                        "✓"
                    } else {
                        "·"
                    },
                    step.title
                );
            }
            println!("finished in {:.0} s:", started.elapsed().as_secs_f64());
            for p in paths {
                println!("  {}", p.display());
            }
            return;
        }
        if let Some(f) = &s.failure {
            eprintln!("failed: {}\n  {}", f.what, f.advice);
            std::process::exit(1);
        }
        if let Some(i) = s.current {
            let tenth = (s.step_share() * 10.0) as u64;
            if (i, tenth) != shown {
                shown = (i, tenth);
                println!(
                    "[{:5.1} %] {} — {} of {}",
                    s.overall() * 100.0,
                    s.steps[i].title,
                    s.done,
                    s.total
                );
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}
