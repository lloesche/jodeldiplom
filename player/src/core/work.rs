//! The background work on the files: their details read for what is on
//! screen, their beats analysed ahead of the queue, the whole queue got
//! ready, and how far each file is.

use std::path::{Path, PathBuf};

use crate::browser::EntryKind;
use crate::core::browse::{browser_rows, list_cursor, list_height, row_entry};
use crate::core::{
    karaoke, Effect, Facts, Gpu, Heard, Job, Prepared, Rhythm, RhythmRecipe, Rhythmed, Screen,
    Side, State, RHYTHM_AHEAD,
};

/// The files on screen whose details are not known yet.
pub fn visible_missing(state: &State) -> Vec<PathBuf> {
    // Karaoke (the tab or the mode): the song it shows and the next two,
    // their titles and artists shown before they play.
    if state.screen == Screen::Karaoke || state.showing(Side::Karaoke) {
        let Some(index) = karaoke::song_index(state) else {
            return Vec::new();
        };
        return state
            .queue
            .paths
            .iter()
            .skip(index)
            .take(3)
            .filter(|p| !state.details.contains_key(*p))
            .cloned()
            .collect();
    }
    if state.screen != Screen::Main {
        return Vec::new();
    }
    let height = list_height(state, state.side);
    let (_, scroll) = list_cursor(state, state.side);
    let paths: Vec<PathBuf> = match state.side {
        Side::Browser => browser_rows(state)
            .into_iter()
            .skip(scroll)
            .take(height)
            .filter_map(|row| row_entry(state, row))
            .filter(|e| e.kind == EntryKind::File)
            .map(|e| e.path.clone())
            .collect(),
        Side::Queue => state
            .queue
            .paths
            .iter()
            .skip(scroll)
            .take(height)
            .cloned()
            .collect(),
        Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth => Vec::new(),
    };
    paths
        .into_iter()
        .filter(|p| !state.details.contains_key(p))
        .collect()
}

/// Asks for everything on screen still unknown, including what is already
/// asked for (the worker replaces its queue with this), when some of it
/// is not asked for yet.
pub(crate) fn want_details(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = visible_missing(state);
    if wanted.iter().any(|p| !state.pending.contains(p)) {
        state.pending.extend(wanted.iter().cloned());
        fx.push(Effect::WantDetails(wanted));
    }
}

/// The files whose beats should be known: the playing track and the next
/// [`RHYTHM_AHEAD`] in the queue; none while the tempo is off.
fn rhythm_wanted(state: &State) -> Vec<PathBuf> {
    match state.queue.current {
        Some(current) if state.config.bpm => {
            // The track a transition just left, while its block slides away.
            let from = match state.transition.slide {
                Some(_) => current.saturating_sub(1),
                None => current,
            };
            state
                .queue
                .paths
                .iter()
                .skip(from)
                .take(1 + RHYTHM_AHEAD + current - from)
                .cloned()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The recipe in force: the methods, and stems when they are on and there
/// is a GPU; `None` while stems are on and the GPU is still being looked
/// for, so nothing is analysed twice.
fn rhythm_recipe(state: &State) -> Option<RhythmRecipe> {
    let stems = match state.gpu {
        _ if !state.config.stems => false,
        Gpu::Looking => return None,
        Gpu::Absent(_) => false,
        Gpu::Present(_) => true,
    };
    Some(RhythmRecipe {
        methods: state.config.methods_in_force(),
        stems,
    })
}

/// When the files to analyse or the recipe change, forgets the analyses
/// no longer wanted (all of them, for another recipe) and asks for those
/// not done yet, in order.
pub(crate) fn want_rhythm(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = rhythm_wanted(state);
    let Some(recipe) = rhythm_recipe(state) else {
        return;
    };
    // Nothing wanted before or now: another recipe changes nothing.
    if wanted == state.rhythm_wanted && (recipe == state.rhythm_recipe || wanted.is_empty()) {
        state.rhythm_recipe = recipe;
        return;
    }
    if recipe != state.rhythm_recipe {
        state.rhythm.clear();
    }
    state.rhythm.retain(|path, _| wanted.contains(path));
    let mut ask = Vec::new();
    for path in &wanted {
        let rhythm = state.rhythm.entry(path.clone()).or_insert(Rhythm::Pending);
        if matches!(rhythm, Rhythm::Pending) {
            ask.push(path.clone());
        }
    }
    state.rhythm_wanted = wanted;
    state.rhythm_recipe = recipe.clone();
    fx.push(Effect::WantRhythm(ask, recipe));
}

/// [`crate::core::Action::Rhythm`]: a file's beats, kept only for a file
/// still wanted, analysed by the recipe in force.
pub(crate) fn rhythm(
    state: &mut State,
    path: PathBuf,
    recipe: RhythmRecipe,
    result: Result<std::sync::Arc<Rhythmed>, String>,
) {
    if state.rhythm_wanted.contains(&path) && recipe == state.rhythm_recipe {
        let rhythm = match result {
            Ok(analysis) => Rhythm::Ready(analysis),
            Err(_) => Rhythm::Failed,
        };
        state.rhythm.insert(path, rhythm);
    }
}

/// Gets the whole queue ready ahead when it or the track playing changed:
/// from the track playing (or the current one) to the end, then from the
/// start; analysed by the recipe in force (none while the tempo is off)
/// and their lyrics looked up and timed, into the caches.
pub(crate) fn prepare(state: &mut State, fx: &mut Vec<Effect>) {
    let recipe = if state.config.bpm {
        // Stems on and the GPU still being looked for: not yet.
        let Some(recipe) = rhythm_recipe(state) else {
            return;
        };
        Some(recipe)
    } else {
        None
    };
    let paths = &state.queue.paths;
    let from = state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.queue.current)
        .unwrap_or(0)
        .min(paths.len());
    let order = || paths[from..].iter().chain(&paths[..from]);
    let old = &state.prepared;
    if old.recipe == recipe && old.order.len() == paths.len() && old.order.iter().eq(order()) {
        return;
    }
    let order: Vec<PathBuf> = order().cloned().collect();
    // Another recipe: the analyses made by the last are not this one's.
    if state.prepared.recipe != recipe {
        for jobs in state.ready.values_mut() {
            jobs.retain(|j| *j != Job::Rhythm);
        }
        for facts in state.facts.values_mut() {
            (facts.key, facts.stems, facts.tempo) = (None, None, None);
        }
    }
    state.prepared = Prepared {
        order: order.clone(),
        recipe: recipe.clone(),
    };
    fx.push(Effect::Prepare(order, recipe));
}

/// [`crate::core::Action::Working`]: a worker started (`on`) or stopped
/// `job` for `path`.
pub(crate) fn working(state: &mut State, path: PathBuf, job: Job, on: bool) {
    let jobs = state.working.entry(path.clone()).or_default();
    jobs.retain(|j| *j != job);
    if on {
        jobs.push(job);
    }
    if jobs.is_empty() {
        state.working.remove(&path);
    }
}

/// [`crate::core::Action::Heard`]: part of what a file's work finds out.
pub(crate) fn heard(state: &mut State, path: PathBuf, part: Heard) {
    let facts = state.facts.entry(path).or_default();
    match part {
        Heard::Key(found) => facts.key = Some(found),
        Heard::Stems => facts.stems = Some(true),
    }
}

/// [`crate::core::Action::Ready`]: `job` is done for `path`, with what it
/// found.
pub(crate) fn ready(state: &mut State, path: PathBuf, job: Job, facts: Option<Facts>) {
    match facts {
        Some(Facts::Rhythm(f)) => state.facts.entry(path.clone()).or_default().analysed(&f),
        Some(Facts::Lyrics(f)) => state.facts.entry(path.clone()).or_default().lyrics = Some(f),
        Some(Facts::Cover(f)) => state.facts.entry(path.clone()).or_default().cover = Some(f),
        None => {}
    }
    let jobs = state.ready.entry(path).or_default();
    if !jobs.contains(&job) {
        jobs.push(job);
    }
}

/// How far a file in the queue is got ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// A worker is at it.
    Working,
    /// Not yet: some work for it is still to come.
    Waiting,
    /// Everything is in the caches: its analysis and waveforms (while the
    /// tempo is on), its lyrics, looked up and timed, and its cover taken
    /// apart (where they are got ready).
    Done,
}

/// How far `path` is got ready ([`Readiness`]).
pub fn readiness(state: &State, path: &Path) -> Readiness {
    if state.working.contains_key(path) {
        return Readiness::Working;
    }
    if still_to_do(state, path).is_empty() {
        Readiness::Done
    } else {
        Readiness::Waiting
    }
}

/// The jobs the queue is got ready by ([`State::preparing`]) not done for
/// `path` yet ([`job_done`]), in the order the queue's icons show them:
/// none once it is fully prepared.
pub fn still_to_do(state: &State, path: &Path) -> Vec<Job> {
    (state.preparing.iter().copied())
        .filter(|&job| !job_done(state, path, job))
        .collect()
}

/// What `job` gets ready, in the words the queue's icons are told by:
/// its analysis and waveforms (and its stems, where they are separated),
/// its lyrics, its cover taken apart.
pub fn job_words(state: &State, job: Job) -> &'static str {
    match job {
        Job::Rhythm if state.prepared.recipe.as_ref().is_some_and(|r| r.stems) => {
            "its analysis, waveforms and stems"
        }
        Job::Rhythm => "its analysis and waveforms",
        Job::Lyrics => "its lyrics, found or heard, and timed",
        Job::Cover => "its cover taken apart",
    }
}

/// Whether `job` is done for `path`: told by its worker, or known here
/// already (an analysis wanted, lyrics shown with their words timed), and
/// no worker at it now.
pub fn job_done(state: &State, path: &Path, job: Job) -> bool {
    if state
        .working
        .get(path)
        .is_some_and(|jobs| jobs.contains(&job))
    {
        return false;
    }
    let told = state
        .ready
        .get(path)
        .is_some_and(|jobs| jobs.contains(&job));
    match job {
        Job::Rhythm => {
            state.prepared.recipe.is_none()
                || told
                || matches!(
                    state.rhythm.get(path),
                    Some(Rhythm::Ready(_) | Rhythm::Failed)
                )
        }
        Job::Lyrics => {
            told || match state.karaoke.lyrics.get(path) {
                Some(karaoke::SongLyrics::Found { shown, .. }) => {
                    shown.timing.detail == jodel_lyrics::Detail::Words
                        || state.karaoke.model == karaoke::ModelState::Missing
                }
                Some(karaoke::SongLyrics::Pending) | None => false,
                Some(_) => true,
            }
        }
        Job::Cover => told,
    }
}
