//! `--diag`: what the player finds on this system, for a bug report. It
//! reads only: the build and the system, the settings, where the caches
//! are, every output device with what it takes, the GPU side (every
//! adapter, whether the stem separator sets up on it, how long a chunk
//! takes), and the last
//! analyses the player made (what was found, how long each part
//! took), the microphones (the one karaoke uses listened to for two
//! seconds) and where the speech model is. Nothing is played.

use std::io::Write;
use std::process::Command;

use crate::cli::{self, Args};

/// Prints the report to standard output, section by section (the GPU
/// check takes a few seconds).
pub fn run(args: &Args) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    let mut section = |title: &str, lines: Vec<String>| {
        let _ = writeln!(out, "{title}");
        for line in lines {
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out);
        let _ = out.flush();
    };
    section("Build", build());
    section("System", system());
    let settings = cli::settings(args)?;
    let folders = cli::folders(&settings)?;
    // While the data folder moves, what has not moved yet is found where
    // it is.
    crate::store::mover::resume(&folders.settings);
    let data = &folders.data;
    section("Settings", settings_lines(&settings));
    section("Caches", caches(&folders));
    let (main, emoji) = crate::gui::fonts::built_in_fonts();
    section(
        "Window fonts",
        vec![format!(
            "Sarasa Fixed J built in: {main}, Noto Emoji built in: {emoji}"
        )],
    );
    section(&format!("Output ({})", jodel_output::BACKEND), output());
    section(
        &format!("Microphones ({})", jodel_output::input::INPUT_BACKEND),
        microphones(settings.effective.karaoke_mic_device.as_deref()),
    );
    section(
        "Speech model (karaoke's word timing)",
        vec![match crate::work::lyrics::models::model_path(&data.0) {
            Some(path) => format!("found: {}", path.display()),
            None => format!(
                "not found (looked beside the binary, in the app's Resources, and in the data folder) for {}",
                crate::work::lyrics::models::MODEL_FILE
            ),
        }],
    );
    section(
        "Speech recogniser (whether fetched lyrics fit the song)",
        vec![match crate::work::lyrics::models::asr_model_path(&data.0) {
            Some(path) => format!("found: {}", path.display()),
            None => format!(
                "not found (looked beside the binary, in the app's Resources, and in the data folder) for {}",
                crate::work::lyrics::models::ASR_FILE
            ),
        }],
    );
    section("GPU (wgpu)", gpu(&settings.effective));
    section(
        "Story scenes taken apart (the game's background)",
        scenes(data),
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let recent = crate::work::rhythm::recent(&data.analyses());
    section(
        &format!(
            "Last analyses (newest first, up to {})",
            crate::work::rhythm::RECENT_KEPT
        ),
        recent_lines(&recent, now),
    );
    Ok(())
}

/// The kept analyses ([`crate::work::rhythm::recent`]) for the report: each
/// one's lines, its time as how long ago.
pub fn recent_lines(entries: &[String], now: u64) -> Vec<String> {
    if entries.is_empty() {
        return vec!["none yet (analyses taken from the cache are not counted)".into()];
    }
    let mut lines = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            lines.push(String::new());
        }
        for line in entry.lines() {
            match line.strip_prefix("at ").and_then(|t| t.parse::<u64>().ok()) {
                Some(at) => lines.push(ago(now.saturating_sub(at))),
                None => lines.push(line.to_owned()),
            }
        }
    }
    lines
}

/// Whether the story's scenes are taken apart on this machine: the two
/// models' weights, whether they set up on the GPU, how long a picture of
/// 1024 × 1024 takes (set up and the first, then one), the GPU memory
/// they hold.
fn scenes(data: &crate::store::dirs::Data) -> Vec<String> {
    use crate::setup::Lookup;
    use crate::work::scenes::{self, GpuNets};
    use crate::work::stages::{self, Stage};
    let places = crate::setup::Places::player(data);
    let found: crate::setup::Paths = crate::setup::manifest()
        .iter()
        .filter_map(|need| match crate::setup::look(need, &places) {
            Lookup::Checked(p) | Lookup::Unchecked(p) => Some((need.file.clone(), p)),
            Lookup::Missing => None,
        })
        .collect();
    let Some(models) = crate::setup::Found::of(&found).and_then(|f| f.scenes) else {
        return vec![format!(
            "the models are not here ({} and {}; setup gets them where there is a usable GPU): no stories are made",
            scenes::DEPTH_FILE,
            scenes::MATTE_FILE
        )];
    };
    let mut lines = vec![
        format!("found: {}", models.depth.display()),
        format!("found: {}", models.matte.display()),
    ];
    let nets = GpuNets::new(models);
    let picture = crate::work::art::sample_cover(1024);
    let derive = || {
        let started = std::time::Instant::now();
        let derived = stages::run(&[Stage::Scenes], "a picture", || {
            scenes::derive(&picture, "sample".into(), &nets)
        });
        derived.map(|(_, took)| (started.elapsed(), took))
    };
    match derive().and_then(|(first, _)| Ok((first, derive()?))) {
        Err(e) => lines.push(format!("runs here: no ({e}); no stories are made")),
        Ok((first, (_, took))) => {
            let (resident, most) = nets.gpu_bytes();
            lines.push(format!(
                "runs here: yes; set up and the first picture {} ms, then a picture {took}; on the GPU {} MB between pictures, at most {} MB while one is taken apart",
                first.as_millis(),
                resident >> 20,
                most >> 20
            ));
        }
    }
    lines
}

/// `12 s ago`, `5 min ago`, `3 h ago`, `2 days ago`.
fn ago(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds} s ago"),
        60..=3_599 => format!("{} min ago", seconds / 60),
        3_600..=86_399 => format!("{} h ago", seconds / 3_600),
        _ => format!("{} days ago", seconds / 86_400),
    }
}

fn build() -> Vec<String> {
    let bin = crate::BIN;
    vec![
        format!(
            "{bin} {}, {} build",
            env!("CARGO_PKG_VERSION"),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        ),
        format!("for {} {}", std::env::consts::OS, std::env::consts::ARCH),
        format!("beat analysis version {}", jodel_rhythm::VERSION),
    ]
}

/// The first line a command prints, if it runs.
fn first_line(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_owned)
}

fn system() -> Vec<String> {
    let mut lines = Vec::new();
    match std::env::consts::OS {
        "macos" => {
            let version = first_line("sw_vers", &["-productVersion"]).unwrap_or_default();
            let build = first_line("sw_vers", &["-buildVersion"]).unwrap_or_default();
            lines.push(format!("macOS {version} ({build})"));
            if let Some(cpu) = first_line("sysctl", &["-n", "machdep.cpu.brand_string"]) {
                lines.push(format!("processor: {cpu}"));
            }
            if let Some(model) = first_line("sysctl", &["-n", "hw.model"]) {
                lines.push(format!("model: {model}"));
            }
        }
        "linux" => {
            let release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
            let name = release
                .lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|n| n.trim_matches('"').to_owned())
                .unwrap_or_else(|| "Linux".into());
            let kernel = first_line("uname", &["-r"]).unwrap_or_default();
            lines.push(format!("{name}, kernel {kernel}"));
            let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            if let Some(cpu) = cpuinfo
                .lines()
                .find_map(|l| l.strip_prefix("model name"))
                .and_then(|l| l.split(':').nth(1))
            {
                lines.push(format!("processor: {}", cpu.trim()));
            }
        }
        "windows" => {
            if let Some(version) = first_line("cmd", &["/C", "ver"]) {
                lines.push(version);
            }
            if let Ok(cpu) = std::env::var("PROCESSOR_IDENTIFIER") {
                lines.push(format!("processor: {cpu}"));
            }
        }
        other => lines.push(other.to_owned()),
    }
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
    lines.push(format!(
        "{cores} processors; the beat network uses {} threads where it runs on the processor",
        jodel_tensors::kernels::threads()
    ));
    lines
}

fn settings_lines(settings: &cli::Settings) -> Vec<String> {
    let mut lines = vec![match &settings.file {
        Some(path) if path.is_file() => format!("file: {}", path.display()),
        Some(path) => format!("file: {} (not there yet: the defaults)", path.display()),
        None => "file: none (the defaults)".into(),
    }];
    lines.extend(settings.warnings.iter().map(|w| format!("warning: {w}")));
    lines.push("in force (with the command line's options):".into());
    lines.extend(
        settings
            .effective
            .to_text()
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .map(|l| format!("  {l}")),
    );
    lines
}

fn caches(folders: &crate::store::dirs::Folders) -> Vec<String> {
    vec![
        format!("beat analyses: {}", folders.data.analyses().display()),
        format!("beat grids fixed by hand: {}", folders.grids().display()),
    ]
}

fn output() -> Vec<String> {
    let devices = match jodel_output::devices() {
        Ok(devices) => devices,
        Err(error) => return vec![format!("listing the devices failed: {error}")],
    };
    if devices.is_empty() {
        return vec!["no output devices".into()];
    }
    let mut lines = Vec::new();
    for device in devices {
        lines.push(format!(
            "{}{}",
            device.name,
            if device.is_default { " (default)" } else { "" }
        ));
        lines.push(format!("  id: {}", device.id));
        lines.push(match jodel_output::mix_format(&device.id) {
            Ok(format) => format!("  takes: {}", crate::desk_wiring::format_summary(format)),
            Err(error) => format!("  takes: cannot be asked: {error}"),
        });
    }
    lines
}

/// How long `--diag` listens to the microphone.
const LISTEN: std::time::Duration = std::time::Duration::from_secs(2);

/// The capture devices, and what comes in from the one karaoke uses
/// (`chosen`, else the default) over [`LISTEN`].
fn microphones(chosen: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    match jodel_output::input::input_devices() {
        Ok(list) if list.is_empty() => lines.push("no capture devices".into()),
        Ok(list) => {
            for d in list {
                lines.push(format!(
                    "{}{}",
                    d.name,
                    if d.is_default { " (default)" } else { "" }
                ));
                lines.push(format!("  id: {}", d.id));
            }
        }
        Err(error) => return vec![format!("listing them failed: {error}")],
    }
    let input = match crate::app::mic::open_mic(chosen) {
        Ok(input) => input,
        Err(error) => {
            lines.push(format!("opening karaoke's one failed: {error}"));
            return lines;
        }
    };
    let info = input.info().clone();
    lines.push(format!(
        "karaoke's one: {}, {} Hz, {} channels, a period of {} frames",
        info.name, info.rate, info.channels, info.period_frames
    ));
    // The meter falls between packets: its highest over the time.
    let started = std::time::Instant::now();
    let mut peak = 0.0f32;
    while started.elapsed() < LISTEN {
        std::thread::sleep(std::time::Duration::from_millis(20));
        peak = peak.max(input.peak());
    }
    let status = input.status();
    lines.push(if status.packet_frames == 0 {
        format!("in {} s: nothing came in at all", LISTEN.as_secs())
    } else if input.only_silence(LISTEN / 2) {
        let hint = if cfg!(target_os = "macos") {
            ": the system gives silence when the program (or the terminal it runs in) is not allowed the microphone, see System Settings › Privacy & Security › Microphone; or the input is muted"
        } else {
            ": the input is muted, or its level is at zero"
        };
        format!("in {} s: only exact silence came in{hint}", LISTEN.as_secs())
    } else {
        format!(
            "in {} s: sound came in, its peak {:.1} dB below full scale",
            LISTEN.as_secs(),
            -20.0 * f64::from(peak.max(1e-9)).log10()
        )
    });
    if let Some(failure) = input.failure() {
        lines.push(format!("capture stopped: {failure}"));
    }
    lines
}

fn gpu(config: &crate::config::Config) -> Vec<String> {
    let mut lines = jodel_gpu::report();
    let separates = lines.iter().any(|l| l.starts_with("separator set up: yes"));
    lines.push(match (config.stems, separates) {
        (false, _) => "stems: off in the settings, so the beat analysis listens to the mix".into(),
        (true, true) => {
            "stems: on; the beat analysis listens to stems separated on this GPU".into()
        }
        (true, false) => {
            "stems: on, but no GPU separates them, so the beat analysis listens to the mix".into()
        }
    });
    let network = lines
        .iter()
        .any(|l| l.starts_with("beat network set up: yes"));
    lines.push(if network {
        "beat network: on this GPU".into()
    } else {
        format!(
            "beat network: on the processor ({} threads)",
            jodel_tensors::kernels::threads()
        )
    });
    lines
}
