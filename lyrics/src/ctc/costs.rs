//! What the path pays besides the model's log-probabilities, one source of
//! cost a function, each added to the [`Costs`] of an alignment's
//! [`Job`] in the order [`super::align_path`] puts them in.

use super::path::{blank_before, states_of, token_at, token_state};
use super::{Costs, Job, Path, StayCost, PINNED};

/// The blanks inside a word (the blank before its letter i, between its
/// letters i - 1 and i) cost where the voice pauses
/// ([`super::CtcSettings::inside_word`]); nothing when the pauses are not
/// known.
pub(super) fn pauses(job: &Job, costs: &mut Costs) {
    if job.hints.pauses.is_empty() {
        return;
    }
    let ids = &job.tokens.ids;
    let sep = job.vocab.separator;
    let inside = job.settings.inside_word * job.clock.frame_seconds as f32;
    let by_state = (0..job.states())
        .map(|s| {
            let between = blank_before(s)
                .is_some_and(|i| i > 0 && i < ids.len() && ids[i - 1] != sep && ids[i] != sep);
            if between {
                inside
            } else {
                0.0
            }
        })
        .collect();
    costs.stay.push(StayCost::Marked {
        by_state,
        frames: Some(job.hints.pauses.clone()),
    });
}

/// Each word set by hand ([`crate::WordEdit::pin`]) holds the path: its
/// letters cost [`PINNED`] a frame outside its times, and on its frames
/// every state but its own and the word breaks either side of it costs as
/// much (a break may fall on its first or last frame, so two words set end
/// to end still fit).
pub(super) fn pinned(job: &Job, costs: &mut Costs) {
    let mut letters = vec![None; job.states()];
    let mut holds = vec![None; job.clock.frames];
    let mut any = false;
    for w in &job.tokens.words {
        let (Some(range), Some((a, b))) = (&w.tokens, job.pin(w)) else {
            continue;
        };
        let within = job.clock.outward(a, b);
        let own = states_of(range.clone());
        for slot in &mut letters[own.clone()] {
            *slot = Some((within.clone(), PINNED));
        }
        let around = own.start.saturating_sub(2)..(own.end + 2).min(job.states());
        for hold in &mut holds[within] {
            *hold = Some((around.clone(), PINNED));
        }
        any = true;
    }
    if any {
        costs.stay.push(StayCost::Outside(letters));
        costs.stay.push(StayCost::Held(holds));
    }
}

/// A word a recogniser heard costs to start outside where it heard it
/// ([`super::CtcSettings::window_penalty`]).
pub(super) fn heard_windows(job: &Job, costs: &mut Costs) {
    let penalty = job.settings.window_penalty;
    if penalty <= 0.0 || job.hints.windows.is_empty() {
        return;
    }
    for (k, w) in job.tokens.words.iter().enumerate() {
        let (Some(range), Some((a, b))) = (&w.tokens, job.heard_window(k)) else {
            continue;
        };
        costs.entry[token_state(range.start)].push((job.clock.entry_window(a, b), penalty));
    }
}

/// A line's words cost to start outside its window in `windows`, and its
/// letters (and the blanks between them) to be there
/// ([`super::CtcSettings::scaffold`]).
pub(super) fn scaffold(job: &Job, costs: &mut Costs, windows: &[Option<(f64, f64)>]) {
    let penalty = job.settings.scaffold;
    if penalty <= 0.0 || windows.is_empty() {
        return;
    }
    let line_window = |l: usize| {
        let &(from, to) = windows.get(l)?.as_ref()?;
        Some(job.clock.entry_window(from, to))
    };
    let per_frame = penalty * job.clock.frame_seconds as f32;
    let mut outside = vec![None; job.states()];
    for (l, tokens) in &job.lines {
        if let Some(within) = line_window(*l) {
            for c in &mut outside[states_of(tokens.clone())] {
                *c = Some((within.clone(), per_frame));
            }
        }
    }
    costs.stay.push(StayCost::Outside(outside));
    for w in &job.tokens.words {
        if let (Some(range), Some(within)) = (&w.tokens, line_window(w.at.line)) {
            costs.entry[token_state(range.start)].push((within, penalty));
        }
    }
}

/// A line costs over a stretch where the glimpses heard no words
/// ([`super::CtcSettings::unheard`]).
pub(super) fn unheard(job: &Job, costs: &mut Costs) {
    if job.settings.unheard <= 0.0 || job.hints.glimpses.is_empty() {
        return;
    }
    let cost = job.settings.unheard * job.clock.frame_seconds as f32;
    costs.stay.push(StayCost::Marked {
        by_state: job.spanned(cost),
        frames: Some(crate::glimpses::unheard(&job.hints.glimpses, job.clock)),
    });
}

/// Each line's first word costs to start outside where the glimpses near
/// where `path` put it heard it start, within the part the transcript has
/// it in when it has it ([`super::CtcSettings::glimpse_penalty`]). Whether
/// any does.
pub(super) fn glimpsed(job: &Job, costs: &mut Costs, path: &Path) -> bool {
    let penalty = job.settings.glimpse_penalty;
    if penalty <= 0.0 || job.hints.glimpses.is_empty() {
        return false;
    }
    let spans = path.token_frames(job.tokens.ids.len());
    let mut first = vec![None; job.lyrics.lines.len()];
    for (l, tokens) in &job.lines {
        first[*l] = spans[tokens.start]
            .as_ref()
            .map(|r| job.clock.time(r.start));
    }
    let starts = crate::glimpses::line_starts(job.lyrics, &job.hints.glimpses, &first);
    let mut any = false;
    let firsts = job.tokens.words.iter().enumerate();
    for (k, w) in firsts.filter(|(_, w)| w.at.word == 0) {
        let (Some(range), Some(start)) = (&w.tokens, starts[w.at.line]) else {
            continue;
        };
        let (c, d) = job
            .heard_window(k)
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        let (a, b) = (start.from.max(c), start.to.min(d));
        if a < b {
            costs.entry[token_state(range.start)].push((job.clock.entry_window(a, b), penalty));
            any = true;
        }
    }
    any
}

/// The opening, from the first word to the first island `path` hears,
/// pulled together: every frame off the letters between costs
/// ([`super::CtcSettings::opening_pull`]). Whether there is one.
pub(super) fn opening(job: &Job, costs: &mut Costs, path: &Path) -> bool {
    if job.settings.opening_pull <= 0.0 {
        return false;
    }
    let Some((first, island)) = job.first_island(path) else {
        return false;
    };
    let ids = &job.tokens.ids;
    let sep = job.vocab.separator;
    let pull = job.settings.opening_pull * job.clock.frame_seconds as f32;
    let by_state = (0..job.states())
        .map(|s| {
            let inside = s > token_state(first) && s < token_state(island);
            let on_letter = token_at(s).is_some_and(|k| ids[k] != sep);
            if inside && !on_letter {
                pull
            } else {
                0.0
            }
        })
        .collect();
    costs.stay.push(StayCost::Marked {
        by_state,
        frames: None,
    });
    true
}
