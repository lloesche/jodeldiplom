//! What this GPU really does ([`run`]: a [`Report`]), measured with small
//! compute kernels on the GPU every network here uses ([`crate::chosen`]),
//! so kernels can be chosen per GPU instead of every one written for the
//! weakest: the adapter and what it claims; whether lane writes into one
//! vector race; what an index past a binding's end reads and writes; the
//! f32 maths at the edges and its accuracy; f16; subgroups; the fast
//! paths' features (the packed int8 dot, cooperative matrices) and what
//! the language model picks; throughput (arithmetic, memory, the cost of
//! zeroing workgroup memory); barriers and atomics. A probe a GPU cannot
//! run says why and the others go on. About two seconds on a desktop GPU.
//!
//! The report is read through its summary ([`Report::lane_writes_race`]
//! and the others) and printed whole ([`Report`]'s `Display`);
//! `cargo run --release -p jodel-gpu --example probe` prints it and writes
//! it to a text file.

mod bounds;
mod elementwise;
mod fast;
mod gpu;
mod half;
mod lanes;
mod maths;
mod rates;
mod subgroups;
mod sync;
#[cfg(test)]
mod tests;
mod text;

use std::fmt;
use std::time::{Duration, Instant};

use self::gpu::Gpu;

/// Why a probe found nothing: the GPU lacks what it needs, or it failed.
#[derive(Clone, Debug)]
enum Miss {
    Unsupported(String),
    Failed(String),
}

impl From<String> for Miss {
    fn from(why: String) -> Miss {
        Miss::Failed(why)
    }
}

/// What a probe found: measured, not run (and why), or failed (and how).
#[derive(Clone, Debug)]
enum Found<T> {
    Measured(T),
    Skipped(String),
    Failed(String),
}

impl<T> Found<T> {
    /// The finding, where it was measured.
    fn measured(&self) -> Option<&T> {
        match self {
            Found::Measured(t) => Some(t),
            _ => None,
        }
    }

    fn of(result: Result<T, Miss>) -> Found<T> {
        match result {
            Ok(t) => Found::Measured(t),
            Err(Miss::Unsupported(why)) => Found::Skipped(why),
            Err(Miss::Failed(why)) => Found::Failed(why),
        }
    }

    /// Its lines: `lines` of the finding, or why there is none.
    fn lines(&self, lines: impl FnOnce(&T) -> Vec<String>) -> Vec<String> {
        match self {
            Found::Measured(t) => lines(t),
            Found::Skipped(why) => vec![why.clone()],
            Found::Failed(why) => {
                // wgpu's errors come over many lines, some empty.
                let mut lines: Vec<String> = why
                    .lines()
                    .map(str::trim_end)
                    .filter(|l| !l.trim().is_empty())
                    .map(String::from)
                    .collect();
                if let Some(first) = lines.first_mut() {
                    *first = format!("failed: {first}");
                }
                lines
            }
        }
    }

    /// Its finding as one table cell (`cell` of it), or why there is none
    /// (a failure's first line).
    fn cell(&self, cell: impl FnOnce(&T) -> String) -> String {
        match self {
            Found::Measured(t) => cell(t),
            Found::Skipped(why) => why.clone(),
            Found::Failed(why) => format!("failed: {}", why.lines().next().unwrap_or("")),
        }
    }
}

/// `Err(Miss::Unsupported)` where `gpu` lacks `feature` (named `name`).
fn needs(gpu: &Gpu, feature: wgpu::Features, name: &str) -> Result<(), Miss> {
    if gpu.has(feature) {
        Ok(())
    } else {
        Err(Miss::Unsupported(format!("not supported: no {name}")))
    }
}

/// One check of many cases: its name, the cases, those wrong.
#[derive(Clone, Debug)]
struct Check {
    name: &'static str,
    cases: u64,
    wrong: u64,
}

/// The adapter and what it claims.
#[derive(Clone, Debug)]
struct Identity {
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    features: wgpu::Features,
    language: wgpu::WgslLanguageFeatures,
    coop: Vec<wgpu::CooperativeMatrixProperties>,
}

/// Everything the probe found on one GPU: read through its summary, or
/// printed whole.
#[derive(Clone, Debug)]
pub struct Report {
    identity: Identity,
    lanes: Found<Vec<lanes::Case>>,
    bounds: Found<bounds::Bounds>,
    edges: Found<Vec<maths::Edge>>,
    accuracy: Found<Vec<maths::Accuracy>>,
    half: Found<half::Half>,
    subgroups: Found<subgroups::Subgroups>,
    dot: Found<fast::Dot>,
    coop: Found<Vec<fast::Coop>>,
    rates: rates::Rates,
    zeroing: Found<rates::Zeroing>,
    sync: Found<Vec<Check>>,
    took: Duration,
}

impl Report {
    /// Whether writing one lane of a vector lost values here (`None`:
    /// not measured).
    pub fn lane_writes_race(&self) -> Option<bool> {
        self.lanes.measured().map(|c| lanes::races(c))
    }

    /// Whether an index past a binding's end read its last element.
    pub fn out_of_bounds_clamps(&self) -> Option<bool> {
        self.bounds.measured().map(bounds::clamps)
    }

    /// Whether `exp`, `exp2` or `tanh` missed IEEE's limit (∞, 0, ±1) or
    /// WGSL's bound at one of the edges (huge arguments among them).
    pub fn fast_maths_breaks(&self) -> Option<bool> {
        self.edges.measured().map(|e| maths::fast_maths_breaks(e))
    }

    /// Whether `dot4I8Packed` is right and faster than unpacking by hand.
    pub fn dot4_native(&self) -> Option<bool> {
        let right = self.dot.measured()?.signed_wrong == 0;
        let d = self.rates.of(rates::Kind::Dot4I8)?;
        let h = self.rates.of(rates::Kind::Dot4ByHand)?;
        Some(right && fast::native(d, h))
    }
}

/// Probes the process's GPU ([`crate::chosen`]); `Err` only where there
/// is no GPU or no device on it.
pub fn run() -> Result<Report, String> {
    let started = Instant::now();
    let adapter = crate::chosen().adapter.clone()?;
    let identity = Identity {
        info: adapter.get_info(),
        limits: adapter.limits(),
        features: adapter.features(),
        language: crate::chosen().instance.wgsl_language_features(),
        coop: adapter.cooperative_matrix_properties(),
    };
    let gpu = Gpu::open("jodel-gpu probe", gpu::WANTED)?;
    let lanes = Found::of(lanes::probe(&gpu));
    let bounds = Found::of(bounds::probe(&gpu));
    let edges = Found::of(maths::edges(&gpu));
    let accuracy = Found::of(maths::accuracy(&gpu));
    let half = Found::of(
        needs(&gpu, wgpu::Features::SHADER_F16, "SHADER_F16").and_then(|_| half::probe(&gpu)),
    );
    let claimed = (
        identity.info.subgroup_min_size,
        identity.info.subgroup_max_size,
    );
    let subgroups = Found::of(
        needs(&gpu, wgpu::Features::SUBGROUP, "SUBGROUP")
            .and_then(|_| subgroups::probe(&gpu, claimed)),
    );
    let dot = Found::of(fast::dot(&gpu));
    // The rate of the cooperative matrices: one subgroup a workgroup, of
    // the one size measured.
    let subgroup = subgroups
        .measured()
        .filter(|s| s.sizes.len() == 1)
        .and_then(|s| s.sizes.first().copied());
    let coop = Found::of(fast::coop(&identity, subgroup));
    let rates = rates::probe(&gpu);
    let zeroing = Found::of(rates::zeroing(&gpu));
    let sync = Found::of(sync::probe(&gpu));
    Ok(Report {
        identity,
        lanes,
        bounds,
        edges,
        accuracy,
        half,
        subgroups,
        dot,
        coop,
        rates,
        zeroing,
        sync,
        took: started.elapsed(),
    })
}

/// The identity's lines.
fn identity_lines(id: &Identity) -> Vec<String> {
    let info = &id.info;
    let l = &id.limits;
    let mb = |b: u64| format!("{} MB", b >> 20);
    let has = |f: wgpu::Features| if id.features.contains(f) { "yes" } else { "no" };
    let language =
        |f: wgpu::WgslLanguageFeatures| if id.language.contains(f) { "yes" } else { "no" };
    let mut rows = vec![
        vec![
            "adapter".to_string(),
            format!("{} ({:?})", info.name, info.device_type),
        ],
        vec!["backend".into(), format!("{:?}", info.backend)],
        vec![
            "ids".into(),
            format!("vendor 0x{:04x}, device 0x{:04x}", info.vendor, info.device),
        ],
        vec![
            "driver".into(),
            format!("{} {}", info.driver, info.driver_info)
                .trim()
                .to_string(),
        ],
        vec!["max buffer".into(), mb(l.max_buffer_size)],
        vec![
            "max storage binding".into(),
            mb(l.max_storage_buffer_binding_size),
        ],
        vec![
            "workgroup storage".into(),
            format!("{} KB", l.max_compute_workgroup_storage_size / 1024),
        ],
        vec![
            "invocations a workgroup".into(),
            format!(
                "{} (x at most {}, workgroups a dimension {})",
                l.max_compute_invocations_per_workgroup,
                l.max_compute_workgroup_size_x,
                l.max_compute_workgroups_per_dimension
            ),
        ],
        vec![
            "subgroup size (claimed)".into(),
            format!("{} to {}", info.subgroup_min_size, info.subgroup_max_size),
        ],
        vec!["SHADER_F16".into(), has(wgpu::Features::SHADER_F16).into()],
        vec![
            "SUBGROUP".into(),
            format!(
                "{} (SUBGROUP_BARRIER {})",
                has(wgpu::Features::SUBGROUP),
                has(wgpu::Features::SUBGROUP_BARRIER)
            ),
        ],
        vec![
            "cooperative matrices".into(),
            if id
                .features
                .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            {
                let shapes: Vec<String> = id
                    .coop
                    .iter()
                    .map(|p| {
                        format!(
                            "{}x{}x{} {:?}->{:?}",
                            p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type
                        )
                        .to_lowercase()
                    })
                    .collect();
                format!("yes (experimental): {}", shapes.join(", "))
            } else {
                "no".into()
            },
        ],
        vec![
            "packed int8 dot".into(),
            format!(
                "WGSL packed_4x8_integer_dot_product {}; naga writes it as {}",
                language(wgpu::WgslLanguageFeatures::Packed4x8IntegerDotProduct),
                fast::written(info.backend)
            ),
        ],
        vec![
            "TIMESTAMP_QUERY".into(),
            has(wgpu::Features::TIMESTAMP_QUERY).into(),
        ],
        vec![
            "SHADER_INT64, SHADER_F64".into(),
            format!(
                "{}, {}",
                has(wgpu::Features::SHADER_INT64),
                has(wgpu::Features::SHADER_F64)
            ),
        ],
    ];
    rows.iter_mut().for_each(|r| r[0] += ":");
    text::table(&rows)
}

/// A heading and its section's lines, indented.
fn section(out: &mut fmt::Formatter<'_>, heading: &str, lines: &[String]) -> fmt::Result {
    writeln!(out, "\n== {heading}")?;
    for line in lines {
        if line.is_empty() {
            writeln!(out)?;
        } else {
            writeln!(out, "  {line}")?;
        }
    }
    Ok(())
}

impl fmt::Display for Report {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backend = self.identity.info.backend;
        writeln!(
            out,
            "GPU probe: {} ({:?}), {:.1} s",
            self.identity.info.name,
            backend,
            self.took.as_secs_f64()
        )?;
        let yes = |v: Option<bool>| match v {
            Some(true) => "YES",
            Some(false) => "no",
            None => "not measured",
        };
        section(
            out,
            "Summary",
            &[
                format!(
                    "lane writes into one vector race: {}",
                    yes(self.lane_writes_race())
                ),
                format!(
                    "an index past a binding's end reads its last element: {}",
                    yes(self.out_of_bounds_clamps())
                ),
                format!(
                    "exp/exp2/tanh miss their limits or WGSL's bound: {}",
                    yes(self.fast_maths_breaks())
                ),
                format!("dot4I8Packed native: {}", yes(self.dot4_native())),
                format!(
                    "Gemma's Fast path here: {}",
                    if crate::llm::q8_on(backend) {
                        "q8"
                    } else {
                        "f32 (Full)"
                    }
                ),
            ],
        )?;
        section(
            out,
            "1. Identity (what the adapter claims)",
            &identity_lines(&self.identity),
        )?;
        section(
            out,
            "2. Lane writes (a store to one lane of a vector another invocation also writes)",
            &self.lanes.lines(|c| lanes::lines(c)),
        )?;
        section(
            out,
            "3. Out of bounds (a binding of 777 words: 194 whole vec4s and three words)",
            &self.bounds.lines(bounds::lines),
        )?;
        section(
            out,
            "4. Maths at the edges (inputs from a buffer)",
            &self.edges.lines(|e| maths::edge_lines(e)),
        )?;
        section(
            out,
            "5. f32 accuracy (against the processor's f64)",
            &self.accuracy.lines(|a| maths::accuracy_lines(a)),
        )?;
        section(out, "6. f16", &self.half.lines(half::lines))?;
        section(out, "7. Subgroups", &self.subgroups.lines(subgroups::lines))?;
        let mut fast_lines = self.dot.lines(|d| fast::dot_lines(d, backend, &self.rates));
        fast_lines.push(String::new());
        fast_lines.push("cooperative matrices:".into());
        fast_lines.extend(self.coop.lines(|c| fast::coop_lines(c)));
        fast_lines.push(String::new());
        fast_lines.extend(fast::gemma(backend, self.dot.measured(), &self.rates));
        section(out, "8. Fast-path features", &fast_lines)?;
        let mut rate_lines = self.rates.lines();
        rate_lines.push(String::new());
        rate_lines.extend(self.zeroing.lines(rates::zeroing_lines));
        section(out, "9. Throughput", &rate_lines)?;
        section(
            out,
            "10. Barriers and atomics",
            &self.sync.lines(|c| text::check_lines(c)),
        )
    }
}
