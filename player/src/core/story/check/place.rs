//! The scenes placed on the song's lines by their first lines alone
//! ([`place`]: for now, what is asked kept; [`for_good`]: once the
//! follow-up settled it), so that every line is in one scene; first lines
//! written as times read as the lines sung then ([`by_the_clock`]).

use super::{Again, Drafted};

/// The line a first line `written` names, of a song of `n` lines: a whole
/// number among them.
fn line(written: f64, n: usize) -> Option<usize> {
    (written.fract() == 0.0 && written >= 0.0 && written < n as f64).then_some(written as usize)
}

/// Where `scenes` start, read by `read` from each scene and its first line
/// written: the first at line 0, each other where `read` puts it past
/// line 0 (none where it is nowhere).
fn firsts(scenes: &[Drafted], read: impl Fn(&Drafted, f64) -> Option<usize>) -> Vec<Option<usize>> {
    (scenes.iter().enumerate())
        .map(|(k, s)| match k {
            0 => Some(0),
            _ => s.written.and_then(|w| read(s, w)).filter(|&l| l > 0),
        })
        .collect()
}

/// `scenes`' first lines (and the scenes they show again) read as times
/// where the model went by the clock, each as the line of `starts` (the
/// song's lines' starts) starting nearest that time, said in `notes`:
/// for certain where it wrote them as minutes:seconds (`"1:32"`, as the
/// prompt tells the times); for the numbers, where two of them or more
/// are no line of the song (past its last) but within its length (`end`,
/// seconds), and read as seconds more of the scenes go up in order
/// ([`rising`]) than read as lines. A plan by lines whose last scenes
/// overshoot stands almost whole as lines (they are asked where they
/// start), while seconds collapse it onto a few lines.
pub(super) fn by_the_clock(
    scenes: &mut [Drafted],
    starts: &[f64],
    end: f64,
    notes: &mut Vec<String>,
) {
    let n = starts.len();
    let nearest = |w: f64| {
        (0..n)
            .min_by(|&a, &b| (starts[a] - w).abs().total_cmp(&(starts[b] - w).abs()))
            .unwrap_or(0)
    };
    let mut timed = Vec::new();
    for s in scenes.iter_mut() {
        if let Some(t) = s.times.0 {
            let line = nearest(t);
            timed.push(format!("{t} s line {line}"));
            s.written = Some(line as f64);
        }
        if let Some(t) = s.times.1 {
            s.same_as = Some(Again::Written(nearest(t) as i64));
        }
    }
    if !timed.is_empty() {
        notes.push(format!(
            "first lines written as minutes:seconds read as the lines sung then: {}",
            timed.join(", ")
        ));
    }
    let past: Vec<f64> = (scenes.iter())
        .filter(|s| s.times.0.is_none())
        .filter_map(|s| s.written.filter(|&w| w >= n as f64 && w <= end))
        .collect();
    let [seconds, _, ..] = past[..] else {
        return;
    };
    let as_lines = rising(&firsts(scenes, |_, w| line(w, n))).len();
    let as_times = firsts(scenes, |s, w| match s.times.0 {
        Some(_) => line(w, n),
        None => (w >= 0.0 && w <= end).then(|| nearest(w)),
    });
    if rising(&as_times).len() <= as_lines {
        return;
    }
    let mut read = Vec::new();
    for s in scenes.iter_mut() {
        if let Some(w) = s.written.filter(|_| s.times.0.is_none()) {
            let line = nearest(w);
            read.push(format!("{w} s line {line}"));
            s.written = Some(line as f64);
        }
        // The scene it shows again is named by its first line too.
        if let (Some(Again::Written(w)), None) = (s.same_as, s.times.1) {
            s.same_as = Some(Again::Written(nearest(w as f64) as i64));
        }
    }
    notes.push(format!(
        "the scenes' first lines read as seconds ({seconds} is no line, within the song's \
         {end:.0} s): {}",
        read.join(", ")
    ));
}

/// `scenes` placed on the song's `n` lines: the first at line 0; the most
/// of the others whose first lines go up in order (each a line of the
/// song) standing as written; the others between two standing (or after
/// the last) each placed for now among the lines there, in their order
/// (as written where that is among them, else spread evenly), where each
/// may go kept to ask (room left for the others); those past the lines
/// there left out, their lines shown by the scene before (said in
/// `notes`).
pub(super) fn place(scenes: &mut Vec<Drafted>, n: usize, notes: &mut Vec<String>) {
    if scenes[0].written != Some(0.0) {
        notes.push(format!(
            "the first scene starts at line 0 (written: {:?})",
            scenes[0].written
        ));
    }
    let starts = firsts(scenes, |_, w| line(w, n));
    let standing = rising(&starts);
    let m = scenes.len();
    let mut out = vec![false; m];
    for (k, s) in scenes.iter_mut().enumerate() {
        if standing.contains(&k) {
            s.first = starts[k].expect("standing");
            s.standing = true;
        }
    }
    // Each run of scenes not standing, between the standing one before it
    // (the first scene always stands) and the next (or the song's end).
    let mut k = 1;
    while k < m {
        if scenes[k].standing {
            k += 1;
            continue;
        }
        let run = k;
        while k < m && !scenes[k].standing {
            k += 1;
        }
        let before = scenes[run - 1].first;
        let next = scenes.get(k).map_or(n, |s| s.first);
        let room = next - before - 1;
        let placed = (k - run).min(room);
        let mut last = before;
        for (j, s) in scenes[run..k].iter_mut().enumerate() {
            if j >= placed {
                out[run + j] = true;
                continue;
            }
            let (lo, hi) = (before + 1 + j, next - placed + j);
            let spread = before + 1 + j * room / placed;
            let at = (s.written).map_or(spread, |w| (w.max(0.0) as usize).clamp(lo, hi));
            s.first = at.max(last + 1);
            s.bounds = Some((lo, hi));
            last = s.first;
        }
    }
    let mut kept = Vec::new();
    for (k, s) in std::mem::take(scenes).into_iter().enumerate() {
        if out[k] {
            notes.push(format!(
                "scene {k}'s first line {:?} fits nowhere between the scenes around it: left out",
                s.written
            ));
        } else {
            kept.push(s);
        }
    }
    *scenes = kept;
}

/// The places of the longest run of `starts` going up (each one there),
/// the first always among them (the first of equal runs).
fn rising(starts: &[Option<usize>]) -> Vec<usize> {
    let m = starts.len();
    let (mut best, mut before) = (vec![0usize; m], vec![None; m]);
    for k in 0..m {
        let Some(s) = starts[k] else {
            continue;
        };
        best[k] = 1;
        for j in 0..k {
            if starts[j].is_some_and(|p| p < s) && best[j] + 1 > best[k] {
                best[k] = best[j] + 1;
                before[k] = Some(j);
            }
        }
    }
    let longest = (0..m)
        .filter(|&k| starts[k].is_some())
        .max_by_key(|&k| (best[k], std::cmp::Reverse(k)));
    let Some(mut k) = longest else {
        return Vec::new();
    };
    let mut run = vec![k];
    while let Some(j) = before[k] {
        run.push(j);
        k = j;
    }
    run.reverse();
    run
}

/// `scenes`, settled, placed for good on the song's `n` lines: the first
/// at line 0, those standing as they are, each other after the one before
/// it (moved there where its answer put it before: said in `notes`). Each
/// placed still has a line of its own before the next standing one: its
/// bounds ([`place`]) leave room for those after it.
pub(super) fn for_good(
    mut scenes: Vec<Drafted>,
    n: usize,
    notes: &mut Vec<String>,
) -> Vec<Drafted> {
    // A scene placed goes before the next standing one (which stays).
    let limits: Vec<usize> = (0..scenes.len())
        .map(|k| {
            let next = scenes[k + 1..].iter().find(|s| s.standing);
            next.map_or(n, |s| s.first)
        })
        .collect();
    let mut after = 0;
    for (k, s) in scenes.iter_mut().enumerate() {
        if k == 0 {
            s.first = 0;
        } else if !s.standing && s.first < after {
            notes.push(format!(
                "scene {k}'s first line {} moved to {after}, after the scene before",
                s.first
            ));
            s.first = after;
        }
        debug_assert!(
            s.standing || k == 0 || s.first < limits[k],
            "a line of its own"
        );
        after = s.first + 1;
    }
    scenes
}
