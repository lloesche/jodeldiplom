//! A reference as text, as the cache keeps it ([`crate::cache::SHINE`]):
//! `version N` (the measuring's [`super::reference::VERSION`]), then a line
//! a sung line: `none`, or its yardstick and par tab apart: the share of it
//! voiced, the pitch off, the spread, the range (`-`: none), the par's
//! shine and pitch, the words' frames, the long notes held (each a comma
//! list) and the voicing (its runs, unvoiced first, a comma list). Numbers
//! as written exactly read back the same.

use std::fmt::Write as _;

use super::measure::Yardstick;
use super::reference::{Par, Phrase, Reference, VERSION};

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |x| x.to_string())
}

fn list(v: &[f64]) -> String {
    v.iter().map(f64::to_string).collect::<Vec<_>>().join(",")
}

/// The voicing's runs: unvoiced first (perhaps 0), then voiced, in turn.
fn runs(mask: &[bool]) -> String {
    let mut out = Vec::new();
    let (mut on, mut n) = (false, 0usize);
    for &v in mask {
        if v == on {
            n += 1;
        } else {
            out.push(n.to_string());
            (on, n) = (v, 1);
        }
    }
    out.push(n.to_string());
    out.join(",")
}

fn unruns(text: &str) -> Option<Vec<bool>> {
    let mut out = Vec::new();
    let mut on = false;
    for n in text.split(',') {
        out.extend(std::iter::repeat_n(on, n.parse().ok()?));
        on = !on;
    }
    Some(out)
}

fn unlist(text: &str) -> Option<Vec<f64>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    text.split(',').map(|v| v.parse().ok()).collect()
}

fn unopt(text: &str) -> Option<Option<f64>> {
    if text == "-" {
        Some(None)
    } else {
        text.parse().ok().map(Some)
    }
}

impl Reference {
    /// As the cache keeps it (its header aside).
    pub fn to_text(&self) -> String {
        let mut out = format!("version {VERSION}\n");
        for p in &self.phrases {
            let Some(p) = p else {
                out.push_str("none\n");
                continue;
            };
            let y = &p.yardstick;
            let _ = writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                y.voiced,
                opt(y.off),
                opt(y.spread),
                opt(y.range),
                p.par.shine,
                p.par.pitch,
                list(&y.word_hits),
                list(&y.held),
                runs(&y.mask)
            );
        }
        out
    }

    /// A reference from [`Reference::to_text`]; none for one measured
    /// another way (another [`VERSION`]) or not read.
    pub fn from_text(text: &str) -> Option<Reference> {
        let mut lines = text.lines();
        let version: u32 = lines.next()?.strip_prefix("version ")?.parse().ok()?;
        if version != VERSION {
            return None;
        }
        let mut phrases = Vec::new();
        for line in lines {
            if line == "none" {
                phrases.push(None);
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 9 {
                return None;
            }
            let yardstick = Yardstick {
                voiced: f[0].parse().ok()?,
                off: unopt(f[1])?,
                spread: unopt(f[2])?,
                range: unopt(f[3])?,
                word_hits: unlist(f[6])?,
                held: unlist(f[7])?,
                mask: unruns(f[8])?,
            };
            let par = Par {
                shine: f[4].parse().ok()?,
                pitch: f[5].parse().ok()?,
            };
            phrases.push(Some(Phrase { yardstick, par }));
        }
        Some(Reference { phrases })
    }
}
