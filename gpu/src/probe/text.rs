//! How the probe's report writes its numbers: tables lined up, rates with
//! SI prefixes, f32 values with their special cases named, checks with
//! what of them was wrong.

use super::Check;

/// `rows` as lines, each column as wide as its widest cell, two spaces
/// between; the last column is not padded.
pub(super) fn table(rows: &[Vec<String>]) -> Vec<String> {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..columns)
        .map(|c| {
            rows.iter()
                .filter_map(|r| r.get(c))
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    rows.iter()
        .map(|row| {
            let mut line = String::new();
            for (c, cell) in row.iter().enumerate() {
                if c + 1 == row.len() {
                    line += cell;
                } else {
                    line += cell;
                    let pad = widths[c] - cell.chars().count() + 2;
                    line.extend(std::iter::repeat_n(' ', pad));
                }
            }
            line.trim_end().to_string()
        })
        .collect()
}

/// `x` with an SI prefix (k, M, G, T) and three significant digits, then
/// `unit`: `si(1.5e12, "FLOP/s")` is "1.50 TFLOP/s".
pub(super) fn si(x: f64, unit: &str) -> String {
    let (scaled, prefix) = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]
        .into_iter()
        .find(|&(scale, _)| x.abs() >= scale)
        .map_or((x, ""), |(scale, p)| (x / scale, p));
    let digits = if scaled.abs() >= 100.0 {
        0
    } else if scaled.abs() >= 10.0 {
        1
    } else {
        2
    };
    format!("{scaled:.digits$} {prefix}{unit}")
}

/// An f32 for the report: `inf`, `-inf`, `NaN`, `0`, `-0`, a denormal
/// marked so, and others in six significant digits.
pub(super) fn value(v: f32) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.into()
    } else if v == 0.0 {
        if v.is_sign_negative() { "-0" } else { "0" }.into()
    } else if v.is_subnormal() {
        format!("{v:.5e} (denormal)")
    } else if (1e-4..1e6).contains(&v.abs()) {
        round6(v)
    } else {
        format!("{v:.5e}")
    }
}

/// `v` printed with at most six significant digits (no trailing zeros).
fn round6(v: f32) -> String {
    let digits = (5 - v.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.digits$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// `part` of `whole` as a share: "0 of 1048576", "12 of 1048576 (0.0011 %)".
pub(super) fn share(part: u64, whole: u64) -> String {
    if part == 0 {
        format!("0 of {whole}")
    } else {
        let p = part as f64 * 100.0 / whole.max(1) as f64;
        let p = if p >= 1.0 {
            format!("{p:.1}")
        } else {
            format!("{p:.2e}")
        };
        format!("{part} of {whole} ({p} %)")
    }
}

/// A largest error in ulps for a table: the number, or `NaN` where a
/// result was NaN (the most ulps there are).
pub(super) fn ulps_cell(ulps: u64) -> String {
    if ulps == u64::MAX {
        "NaN".into()
    } else {
        ulps.to_string()
    }
}

/// Checks as a table: each one's name and what of its cases was wrong.
pub(super) fn check_lines(checks: &[Check]) -> Vec<String> {
    let rows: Vec<Vec<String>> = checks
        .iter()
        .map(|c| {
            vec![
                c.name.to_string(),
                format!("wrong {}", share(c.wrong, c.cases)),
            ]
        })
        .collect();
    table(&rows)
}
