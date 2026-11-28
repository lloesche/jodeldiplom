//! The scenes placed on the song's lines by their first lines alone
//! ([`place`]: for now, what is asked kept; [`for_good`]: once the
//! follow-up settled it), so that every line is in one scene.

use super::Drafted;

/// `scenes` placed on the song's `n` lines: the first at line 0; the most
/// of the others whose first lines go up in order (each a line of the
/// song) standing as written; the others between two standing (or after
/// the last) each placed for now among the lines there, in their order
/// (as written where that is among them, else spread evenly), where each
/// may go kept to ask (room left for the others); those past the lines
/// there left out, their lines shown by the scene before (said in
/// `notes`).
pub(super) fn place(scenes: &mut Vec<Drafted>, n: usize, notes: &mut Vec<String>) {
    let line = |w: Option<f64>| {
        w.filter(|w| w.fract() == 0.0 && *w >= 0.0 && *w < n as f64)
            .map(|w| w as usize)
    };
    if scenes[0].written != Some(0.0) {
        notes.push(format!(
            "the first scene starts at line 0 (written: {:?})",
            scenes[0].written
        ));
    }
    let starts: Vec<Option<usize>> = (scenes.iter().enumerate())
        .map(|(k, s)| match k {
            0 => Some(0),
            _ => line(s.written).filter(|&l| l > 0),
        })
        .collect();
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
