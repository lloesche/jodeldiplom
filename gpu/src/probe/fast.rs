//! The fast paths' features ([`dot`], [`coop`], [`gemma`]): whether
//! `dot4I8Packed` and `dot4U8Packed` give the right sums and whether they
//! are native (faster than unpacking the bytes by hand); whether the
//! cooperative matrices the adapter lists multiply right, and how fast,
//! and whether each of their loads and their store by rows (`coopLoadT`,
//! `coopStoreT`) is right; and what the language model picks here (its
//! `Fast` path or its f32 `Full` one).

use super::gpu::{caught, Bound, Gpu, Noise, COOP};
use super::half::{from_f16, to_f16};
use super::rates::{self, Kind, Rates};
use super::text::{share, si, table};
use super::{Found, Identity, Miss};

/// The packed dots checked.
const PAIRS: usize = 1 << 16;

/// The packed dots' sums checked against the processor's: pairs checked,
/// and wrong ones for the signed and the unsigned dot.
#[derive(Clone, Debug)]
pub(super) struct Dot {
    pairs: u64,
    pub(super) signed_wrong: u64,
    unsigned_wrong: u64,
}

/// `dot4I8Packed` of `a` and `b` as the processor does it.
pub(super) fn dot4_i8(a: u32, b: u32) -> i32 {
    (0..4)
        .map(|k| i32::from((a >> (8 * k)) as u8 as i8) * i32::from((b >> (8 * k)) as u8 as i8))
        .sum()
}

/// `dot4U8Packed` of `a` and `b` as the processor does it.
pub(super) fn dot4_u8(a: u32, b: u32) -> u32 {
    (0..4)
        .map(|k| ((a >> (8 * k)) & 255) * ((b >> (8 * k)) & 255))
        .sum()
}

const DOT: &str = "@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let p = input[g.x];
    out[g.x] = vec2<u32>(bitcast<u32>(dot4I8Packed(p.x, p.y)), dot4U8Packed(p.x, p.y));
}
";

/// The packed dots on `gpu`.
pub(super) fn dot(gpu: &Gpu) -> Result<Dot, Miss> {
    let mut noise = Noise::new(0xd07);
    let mut words: Vec<u32> = (0..PAIRS * 2).map(|_| noise.word()).collect();
    // The edges of the bytes' range too.
    words[..8].copy_from_slice(&[
        0x8080_8080,
        0x8080_8080,
        0x7f7f_7f7f,
        0x8080_8080,
        0xffff_ffff,
        0xffff_ffff,
        0x7f7f_7f7f,
        0x7f7f_7f7f,
    ]);
    let kernel = gpu.kernel(DOT, "main")?;
    let input = gpu.words(&words)?;
    let out = gpu.zeros(PAIRS as u64 * 8)?;
    gpu.run(
        &kernel,
        &[Bound::All(&input), Bound::All(&out)],
        [(PAIRS / 256) as u32, 1, 1],
        1,
    )?;
    let got = gpu.read(&out)?;
    let (mut signed_wrong, mut unsigned_wrong) = (0, 0);
    for (pair, sums) in words.chunks_exact(2).zip(got.chunks_exact(2)) {
        signed_wrong += u64::from(sums[0] as i32 != dot4_i8(pair[0], pair[1]));
        unsigned_wrong += u64::from(sums[1] != dot4_u8(pair[0], pair[1]));
    }
    Ok(Dot {
        pairs: PAIRS as u64,
        signed_wrong,
        unsigned_wrong,
    })
}

/// How `dot4I8Packed` is done here, from its rate against the same by
/// hand: native where it is at least half as fast again.
pub(super) fn native(dot: f64, by_hand: f64) -> bool {
    dot >= 1.5 * by_hand
}

/// How `backend` writes `dot4I8Packed` (naga's writers, wgpu 30).
pub(super) fn written(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => {
            "OpSDot where the device has VK_KHR_shader_integer_dot_product (else shifts and multiplies)"
        }
        wgpu::Backend::Metal => {
            "four multiply-adds of packed_char4 bytes (Metal has no packed dot instruction)"
        }
        _ => "shifts and multiplies",
    }
}

/// The packed dots as report lines, with their rates.
pub(super) fn dot_lines(dot: &Dot, backend: wgpu::Backend, rates: &Rates) -> Vec<String> {
    let mut lines = vec![
        format!(
            "dot4I8Packed: wrong {}; dot4U8Packed: wrong {}",
            share(dot.signed_wrong, dot.pairs),
            share(dot.unsigned_wrong, dot.pairs)
        ),
        format!("naga writes it on {backend:?} as {}", written(backend)),
    ];
    match (rates.of(Kind::Dot4I8), rates.of(Kind::Dot4ByHand)) {
        (Some(d), Some(h)) => lines.push(format!(
            "measured: {} against {} by hand ({:.2}x): {}",
            si(d, "MAC/s"),
            si(h, "MAC/s"),
            d / h,
            if native(d, h) {
                "native"
            } else {
                "EMULATED (no faster than unpacking)"
            }
        )),
        _ => lines.push("measured: no rate (see the rates)".into()),
    }
    lines
}

/// One cooperative matrix configuration the adapter lists, and what it
/// did.
#[derive(Clone, Debug)]
pub(super) struct Coop {
    /// The tile: m × n × k.
    shape: [u32; 3],
    ab: wgpu::CooperativeScalarType,
    cr: wgpu::CooperativeScalarType,
    /// How its product came out, its loads and store by rows, and its rate
    /// (FLOP/s).
    product: Found<Product>,
    rows: Found<[Outcome; 4]>,
    rate: Found<f64>,
}

/// How a product with one of its operations by rows came out against the
/// processor's: right, wrong, or its output never written (the sentinel
/// it was filled with still there: the kernel faulted or never ran).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Right,
    Wrong,
    Unwritten,
}

/// The operations checked by rows, one at a time (the others by columns):
/// A loaded, B loaded, C loaded, and the sums stored.
const BY_ROWS: [&str; 4] = ["A", "B", "C", "store"];

/// What the output is filled with before a check: no product of the
/// check's values (small integers and halves) comes near it, and it is
/// exact in f16.
pub(super) const SENTINEL: f32 = 1234.0;

/// How a cooperative matrix product came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Product {
    /// A·B with the matrices read and written by rows.
    Rows,
    /// B·A as rows read: the matrices are read and written by columns.
    Columns,
    Wrong,
    /// Its output never written (the sentinel still there).
    Unwritten,
}

/// The WGSL name of a cooperative matrix scalar, where the probe takes it.
fn scalar(t: wgpu::CooperativeScalarType) -> Option<&'static str> {
    match t {
        wgpu::CooperativeScalarType::F32 => Some("f32"),
        wgpu::CooperativeScalarType::F16 => Some("f16"),
        _ => None,
    }
}

/// The rate's kernel ([`coop_rate`]) of one `n` × `n` tile of `ab` inputs
/// and `cr` sums, in workgroups of `workgroup`: each workgroup's four
/// accumulators loaded from places of their own in `c`, each multiplied
/// into `rounds` times, and stored back there (so no product is the same
/// as another, and none is dropped).
fn coop_source(n: u32, ab: &str, cr: &str, rounds: u32, workgroup: u32) -> String {
    let tile = n * n;
    let mut body = format!(
        "    let ma = coopLoad<coop_mat{n}x{n}<{ab}, A>>(&a[0], {n}u);
    let mb = coopLoad<coop_mat{n}x{n}<{ab}, B>>(&b[0], {n}u);
    let at = w.x * {}u;
",
        4 * tile
    );
    for k in 0..4 {
        body += &format!(
            "    var c{k} = coopLoad<coop_mat{n}x{n}<{cr}, C>>(&c[at + {}u], {n}u);\n",
            k * tile
        );
    }
    body += &format!("    for (var k = 0u; k < {rounds}u; k++) {{\n");
    for k in 0..4 {
        body += &format!("        c{k} = coopMultiplyAdd(ma, mb, c{k});\n");
    }
    body += "    }\n";
    for k in 0..4 {
        body += &format!("    coopStore(c{k}, &c[at + {}u], {n}u);\n", k * tile);
    }
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> a: array<{ab}>;
@group(0) @binding(1) var<storage, read> b: array<{ab}>;
@group(0) @binding(2) var<storage, read_write> c: array<{cr}>;
@compute @workgroup_size({workgroup})
fn main(@builtin(workgroup_id) w: vec3<u32>) {{
{body}}}
"
    )
}

/// The words of `values` stored as `ty` (f16s two to a word).
fn stored(values: &[f32], ty: &str) -> Vec<u32> {
    if ty == "f16" {
        values
            .chunks(2)
            .map(|p| u32::from(to_f16(p[0])) | (u32::from(to_f16(*p.get(1).unwrap_or(&0.0))) << 16))
            .collect()
    } else {
        values.iter().map(|v| v.to_bits()).collect()
    }
}

/// `words` of `ty` as values.
fn loaded(words: &[u32], ty: &str, len: usize) -> Vec<f32> {
    if ty == "f16" {
        words
            .iter()
            .flat_map(|&w| [from_f16(w as u16), from_f16((w >> 16) as u16)])
            .take(len)
            .collect()
    } else {
        words.iter().take(len).map(|&w| f32::from_bits(w)).collect()
    }
}

/// One tile's A·B + C (`a`, `b`, `c` `n` × `n`, by rows in memory), each
/// of A, B and C loaded by rows (`coopLoadT`) where `rows` says, else by
/// columns, and the sums stored by rows (`coopStoreT`) where `rows[3]`
/// says: what its output buffer holds.
fn by_rows_source(n: u32, ab: &str, cr: &str, rows: [bool; 4], workgroup: u32) -> String {
    let load = |by_rows: bool| if by_rows { "coopLoadT" } else { "coopLoad" };
    let store = if rows[3] { "coopStoreT" } else { "coopStore" };
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> a: array<{ab}>;
@group(0) @binding(1) var<storage, read> b: array<{ab}>;
@group(0) @binding(2) var<storage, read> c: array<{cr}>;
@group(0) @binding(3) var<storage, read_write> d: array<{cr}>;
@compute @workgroup_size({workgroup})
fn main() {{
    let ma = {}<coop_mat{n}x{n}<{ab}, A>>(&a[0], {n}u);
    let mb = {}<coop_mat{n}x{n}<{ab}, B>>(&b[0], {n}u);
    let mc = {}<coop_mat{n}x{n}<{cr}, C>>(&c[0], {n}u);
    {store}(coopMultiplyAdd(ma, mb, mc), &d[0], {n}u);
}}
",
        load(rows[0]),
        load(rows[1]),
        load(rows[2]),
    )
}

/// What [`by_rows_source`]'s output holds, as the processor reckons it:
/// a matrix by rows reads (r, c) at r·n + c, by columns at c·n + r.
pub(super) fn by_rows_expected(
    (a, b, c): (&[f32], &[f32], &[f32]),
    n: usize,
    rows: [bool; 4],
) -> Vec<f32> {
    let at = |by_rows: bool, r: usize, col: usize| if by_rows { r * n + col } else { col * n + r };
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for col in 0..n {
            let sum: f32 = (0..n)
                .map(|k| a[at(rows[0], r, k)] * b[at(rows[1], k, col)])
                .sum();
            out[at(rows[3], r, col)] = sum + c[at(rows[2], r, col)];
        }
    }
    out
}

/// How `got` compares with `want` (to `tolerance`), or whether the
/// sentinel is still there.
pub(super) fn outcome(want: &[f32], got: &[f32], tolerance: f32) -> Outcome {
    if got.contains(&SENTINEL) {
        Outcome::Unwritten
    } else if near(want, got, tolerance) {
        Outcome::Right
    } else {
        Outcome::Wrong
    }
}

/// Whether `got` is `want` to `tolerance` (relative, past 1).
pub(super) fn near(want: &[f32], got: &[f32], tolerance: f32) -> bool {
    want.iter()
        .zip(got)
        .all(|(w, g)| (w - g).abs() <= tolerance * (1.0 + w.abs()))
}

/// Each operation of [`BY_ROWS`] by rows in its own product of one tile,
/// against the processor's, its output first filled with the sentinel.
fn by_rows(gpu: &Gpu, n: u32, ab: &str, cr: &str) -> Result<[Outcome; 4], String> {
    let len = (n * n) as usize;
    let mut noise = Noise::new(0xc00d);
    let (a, b, c) = (
        tile_values(len, &mut noise),
        tile_values(len, &mut noise),
        tile_values(len, &mut noise),
    );
    let (ba, bb, bc) = (
        gpu.words(&stored(&a, ab))?,
        gpu.words(&stored(&b, ab))?,
        gpu.words(&stored(&c, cr))?,
    );
    let mut found = [Outcome::Wrong; 4];
    for (i, slot) in found.iter_mut().enumerate() {
        let rows = std::array::from_fn(|j| j == i);
        let kernel = gpu.kernel(&by_rows_source(n, ab, cr, rows, 64), "main")?;
        let d = gpu.words(&stored(&vec![SENTINEL; len], cr))?;
        gpu.run(
            &kernel,
            &[
                Bound::All(&ba),
                Bound::All(&bb),
                Bound::All(&bc),
                Bound::All(&d),
            ],
            [1, 1, 1],
            1,
        )?;
        let got = loaded(&gpu.read(&d)?, cr, len);
        *slot = outcome(
            &by_rows_expected((&a, &b, &c), n as usize, rows),
            &got,
            1e-3,
        );
    }
    Ok(found)
}

/// The `n` × `n` product of `a` and `b`, both by rows.
pub(super) fn product(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0; n * n];
    for r in 0..n {
        for col in 0..n {
            c[r * n + col] = (0..n).map(|k| a[r * n + k] * b[k * n + col]).sum();
        }
    }
    c
}

/// How `got` compares with A·B and B·A (by rows), to `tolerance`.
pub(super) fn which_product(
    a: &[f32],
    b: &[f32],
    got: &[f32],
    n: usize,
    tolerance: f32,
) -> Product {
    if got.contains(&SENTINEL) {
        Product::Unwritten
    } else if near(&product(a, b, n), got, tolerance) {
        Product::Rows
    } else if near(&product(b, a, n), got, tolerance) {
        Product::Columns
    } else {
        Product::Wrong
    }
}

/// A tile's values: small integers and halves, exact in f16, their
/// products' sums too.
fn tile_values(len: usize, noise: &mut Noise) -> Vec<f32> {
    (0..len)
        .map(|_| (noise.word() % 9) as f32 * 0.5 - 2.0)
        .collect()
}

/// Every cooperative matrix configuration the adapter lists (`identity`),
/// on a device of their own; `subgroup` the size measured, for the rate
/// (one subgroup a workgroup).
pub(super) fn coop(identity: &Identity, subgroup: Option<u32>) -> Result<Vec<Coop>, Miss> {
    if !identity.features.contains(COOP) {
        return Err(Miss::Unsupported(
            "not supported: no SHADER_F16 with EXPERIMENTAL_COOPERATIVE_MATRIX".into(),
        ));
    }
    if identity.coop.is_empty() {
        return Err(Miss::Unsupported(
            "not supported: the adapter lists no configuration".into(),
        ));
    }
    // The device's setup is outside the probe's guards (a panic in it is
    // caught here).
    let gpu = caught(|| Gpu::open("jodel-gpu probe coop", COOP))?;
    let mut found = Vec::new();
    for p in &identity.coop {
        let square = p.m_size == p.n_size && p.n_size == p.k_size && matches!(p.m_size, 8 | 16);
        let (product, rows, rate) = match (square, scalar(p.ab_type).zip(scalar(p.cr_type))) {
            (true, Some((ab, cr))) => {
                let product = Found::of(multiply(&gpu, p.m_size, ab, cr).map_err(Miss::from));
                let rows = Found::of(by_rows(&gpu, p.m_size, ab, cr).map_err(Miss::from));
                let rate = match (&product, subgroup) {
                    (Found::Measured(Product::Wrong | Product::Unwritten), _) => {
                        Found::Skipped("not timed: no right product".into())
                    }
                    (Found::Measured(_), Some(sg)) => {
                        Found::of(coop_rate(&gpu, p.m_size, ab, cr, sg))
                    }
                    (Found::Measured(_), None) => {
                        Found::Skipped("not timed: no one subgroup size was measured".into())
                    }
                    _ => Found::Skipped("not timed: the product was not measured".into()),
                };
                (product, rows, rate)
            }
            _ => (
                Found::Skipped("not tried (the probe takes square tiles of 8 or 16)".into()),
                Found::Skipped("not tried".into()),
                Found::Skipped("not timed: not tried".into()),
            ),
        };
        found.push(Coop {
            shape: [p.m_size, p.n_size, p.k_size],
            ab: p.ab_type,
            cr: p.cr_type,
            product,
            rows,
            rate,
        });
    }
    Ok(found)
}

/// The bytes of `len` values stored as `ty` (f16s two to a word).
fn bytes(len: usize, ty: &str) -> u64 {
    let words = if ty == "f16" { len.div_ceil(2) } else { len };
    4 * words as u64
}

/// One tile's product of random matrices, everything by columns (C
/// zeros), over the sentinel ([`by_rows_source`]), compared with the
/// processor's.
fn multiply(gpu: &Gpu, n: u32, ab: &str, cr: &str) -> Result<Product, String> {
    let len = (n * n) as usize;
    let mut noise = Noise::new(0xc00b);
    let (a, b) = (tile_values(len, &mut noise), tile_values(len, &mut noise));
    let kernel = gpu.kernel(&by_rows_source(n, ab, cr, [false; 4], 64), "main")?;
    let (ba, bb) = (gpu.words(&stored(&a, ab))?, gpu.words(&stored(&b, ab))?);
    let c = gpu.zeros(bytes(len, cr))?;
    let d = gpu.words(&stored(&vec![SENTINEL; len], cr))?;
    gpu.run(
        &kernel,
        &[
            Bound::All(&ba),
            Bound::All(&bb),
            Bound::All(&c),
            Bound::All(&d),
        ],
        [1, 1, 1],
        1,
    )?;
    let got = loaded(&gpu.read(&d)?, cr, len);
    Ok(which_product(&a, &b, &got, n as usize, 1e-3))
}

/// The rate of one configuration: four accumulating products a round in
/// each subgroup (a workgroup).
fn coop_rate(gpu: &Gpu, n: u32, ab: &str, cr: &str, subgroup: u32) -> Result<f64, Miss> {
    const ROUNDS: u32 = 512;
    const GROUPS: u32 = 4096;
    let len = (n * n) as usize;
    let kernel = gpu.kernel(&coop_source(n, ab, cr, ROUNDS, subgroup), "main")?;
    let mut noise = Noise::new(0xc00c);
    let (a, b) = (tile_values(len, &mut noise), tile_values(len, &mut noise));
    let (ba, bb) = (gpu.words(&stored(&a, ab))?, gpu.words(&stored(&b, ab))?);
    let c = gpu.zeros(bytes(GROUPS as usize * 4 * len, cr))?;
    let work = f64::from(GROUPS) * f64::from(ROUNDS) * 4.0 * 2.0 * f64::from(n).powi(3);
    Ok(gpu.rate(
        &kernel,
        &[Bound::All(&ba), Bound::All(&bb), Bound::All(&c)],
        [GROUPS, 1, 1],
        work,
        rates::TARGET,
    )?)
}

/// The cooperative matrices as report lines.
pub(super) fn coop_lines(found: &[Coop]) -> Vec<String> {
    let mut rows = vec![vec![
        "tile".to_string(),
        "inputs".into(),
        "sums".into(),
        "product".into(),
        "by rows (A, B, C loaded; stored)".into(),
        "rate".into(),
    ]];
    for c in found {
        rows.push(vec![
            format!("{}x{}x{}", c.shape[0], c.shape[1], c.shape[2]),
            format!("{:?}", c.ab).to_lowercase(),
            format!("{:?}", c.cr).to_lowercase(),
            c.product.cell(|p| {
                match p {
                    Product::Rows => "right (by rows)",
                    Product::Columns => "right (loads and stores by columns)",
                    Product::Wrong => "WRONG",
                    Product::Unwritten => "NEVER WRITTEN",
                }
                .into()
            }),
            c.rows.cell(|found| {
                BY_ROWS
                    .iter()
                    .zip(found)
                    .map(|(op, o)| {
                        let how = match o {
                            Outcome::Right => "right",
                            Outcome::Wrong => "WRONG",
                            Outcome::Unwritten => "NEVER WRITTEN",
                        };
                        format!("{op} {how}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }),
            c.rate.cell(|r| si(*r, "FLOP/s")),
        ]);
    }
    table(&rows)
}

/// What the language model picks here (`llm::q8_on`), and what its fast
/// path would gain.
pub(super) fn gemma(backend: wgpu::Backend, dot: Option<&Dot>, rates: &Rates) -> Vec<String> {
    let q8 = crate::llm::q8_on(backend);
    let mut lines = vec![format!(
        "Gemma with Precision::Fast here takes its rows at {} (llm::q8_on({backend:?}) is {q8})",
        if q8 {
            "8 bits: the Fast path"
        } else {
            "f32: the Full path"
        }
    )];
    let right = dot.is_some_and(|d| d.signed_wrong == 0);
    match (
        rates.of(Kind::Dot4I8),
        rates.of(Kind::Dot4ByHand),
        rates.of(Kind::F32Fma),
    ) {
        (Some(d), Some(h), Some(f)) => {
            // An f32 multiply-add is two flops; the Full path does one a
            // weight, the q8 path one int8 MAC.
            let ratio = d / (f / 2.0);
            lines.push(format!(
                "the q8 path {} here: dot4I8Packed {}, {} ({}), against {} f32 multiply-adds: {ratio:.2}x",
                if right { "would run" } else { "would NOT run right" },
                if right { "right" } else { "WRONG" },
                si(d, "MAC/s"),
                if native(d, h) { "native" } else { "emulated" },
                si(f / 2.0, "FMA/s"),
            ));
            lines.push(format!(
                "estimate: a step (one token) reads the same 4- and 6-bit weights on both paths, so it gains little; a prompt's tiled multiplies are compute bound and would go about {ratio:.1}x {}",
                if ratio >= 1.0 { "as fast" } else { "as fast (slower)" }
            ));
        }
        _ => lines.push("estimate: no rates to compare (see the rates)".into()),
    }
    lines
}
