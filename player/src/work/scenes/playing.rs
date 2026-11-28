//! The song playing's story for the app, on a thread of its own
//! ([`PlayingStory`]): its look (the song's own, read from its scenes'
//! description: one small file) as the song starts, and its scenes'
//! texels for the game's backdrop, made from their parts as the song's
//! preparation kept them ([`super::prepare`]: on the processor alone,
//! nothing loaded on the GPU), only while that backdrop is wanted
//! ([`PlayingStory::show_scenes`]); both again when its story is finished
//! while it plays ([`PlayingStory::on_drawn`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use jodel_looks::Look;

use super::{SceneStore, Story};
use crate::work::outbox::Notify;
use crate::work::Queue;

/// What the thread is asked: the song playing, whether its scenes are
/// wanted, and whether what is held for it is to be made afresh.
#[derive(Default)]
struct Asked {
    playing: Option<PathBuf>,
    scenes: bool,
    fresh: bool,
}

/// What is held for the song playing: its look, and its scenes ready.
#[derive(Default)]
struct Held {
    look: Option<Arc<Look>>,
    story: Option<Arc<Story>>,
}

struct Shared {
    queue: Queue<Asked>,
    held: Mutex<Held>,
}

impl Shared {
    fn held(&self) -> MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The song playing's story's thread.
pub struct PlayingStory {
    shared: Arc<Shared>,
}

impl Drop for PlayingStory {
    fn drop(&mut self) {
        self.shared.queue.stop();
    }
}

impl PlayingStory {
    /// Starts the thread: the stories in `store`, `notify` told when the
    /// song playing's look or scenes are held.
    pub fn start(store: SceneStore, notify: Notify) -> PlayingStory {
        let shared = Arc::new(Shared {
            queue: Queue::new(Asked::default()),
            held: Mutex::new(Held::default()),
        });
        let thread = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("playing story".into())
            .spawn(move || work(&thread, &store, &*notify))
            .expect("a thread");
        PlayingStory { shared }
    }

    /// The song playing is now the one at `path`: the last one's look and
    /// scenes go at once (under the queue's lock, so the thread never holds
    /// them for this one), this one's are read.
    pub fn want(&self, path: &Path) {
        self.shared.queue.change(|asked| {
            *self.shared.held() = Held::default();
            asked.playing = Some(path.to_owned());
            asked.fresh = true;
        });
    }

    /// Whether the song playing's scenes are wanted (the game's backdrop
    /// or the Shader tab shows): made ready as they come to be, let go as
    /// they go.
    pub fn show_scenes(&self, scenes: bool) {
        self.shared.queue.change(|asked| {
            if asked.scenes != scenes {
                asked.scenes = scenes;
                if scenes {
                    asked.fresh = true;
                } else {
                    self.shared.held().story = None;
                }
            }
        });
    }

    /// What to tell when a song's story is finished (its scenes drawn and
    /// taken apart): where it is the song playing, what is held for it is
    /// read again.
    pub fn on_drawn(&self) -> impl Fn(&Path) + Send + Sync + 'static {
        let shared = Arc::clone(&self.shared);
        move |path: &Path| {
            shared.queue.change(|asked| {
                if asked.playing.as_deref() == Some(path) {
                    asked.fresh = true;
                }
            });
        }
    }

    /// The song playing's look, its story's (none: it has none, or not
    /// yet).
    pub fn look(&self) -> Option<Arc<Look>> {
        self.shared.held().look.clone()
    }

    /// The song playing's scenes ready, while they are wanted (none: it has
    /// none, or not yet).
    pub fn story(&self) -> Option<Arc<Story>> {
        self.shared.held().story.clone()
    }
}

/// The thread: each song asked for afresh read (its look, and its scenes
/// made ready where they are wanted) and held while it is still the one
/// asked for (none asked anew since, under the queue's lock).
fn work(shared: &Shared, store: &SceneStore, notify: &(dyn Fn() + Send + Sync)) {
    while let Some((path, scenes)) = shared.queue.next(|asked| {
        let fresh = std::mem::take(&mut asked.fresh);
        (asked.playing.clone().filter(|_| fresh)).map(|p| (p, asked.scenes))
    }) {
        let described = store.scenes_of(&path);
        let look = described.as_ref().map(|s| Arc::new(s.look.clone()));
        let story = (described.as_ref().filter(|_| scenes)).and_then(|s| {
            (super::prepare(s, store))
                .map_err(|why| missed(&path, &why))
                .ok()
        });
        let held = {
            let asked = shared.queue.lock();
            let held = !asked.fresh && asked.playing.as_ref() == Some(&path);
            if held {
                let story = story.filter(|_| asked.scenes);
                *shared.held() = Held { look, story };
            }
            held
        };
        if held {
            notify();
        }
    }
}

/// The song at `path`'s scenes not made ready, for `why` (noted in the
/// trace).
fn missed(path: &Path, why: &str) {
    let name = crate::work::song_name(path);
    crate::trace::note(format_args!("scenes: {name}: {why}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::scenes::{drawn, look, take_apart, ByRed};
    use std::time::{Duration, Instant};

    /// Waits until `ok`, for a few seconds at most.
    fn until(ok: impl Fn() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if ok() {
                return true;
            }
            std::thread::yield_now();
        }
        false
    }

    /// The song playing's look is held as soon as its scenes are described;
    /// its scenes only once they are taken apart, and only while they are
    /// wanted; another song playing lets both go at once.
    #[test]
    fn the_song_playing_s_story_is_held_once_it_is_finished_and_wanted() {
        let dir = jodel_testkit::TempDir::new("playing-story");
        let song = dir.join("a song.wav");
        durable::write_atomic(&song, b"a song").unwrap();
        let store = SceneStore::new(dir.join("scenes"), dir.join("scene-parts"));
        let playing = PlayingStory::start(store.clone(), Arc::new(|| {}));
        playing.want(&song);
        drawn(&store, &song);
        // Drawn, not taken apart: its look, no scenes (not wanted yet).
        (playing.on_drawn())(&song);
        assert!(until(|| playing.look().is_some_and(|l| *l == look())));
        assert!(playing.story().is_none());
        let scenes = store.scenes_of(&song).unwrap();
        take_apart(&scenes, &store, &ByRed::default(), &|| true).unwrap();
        (playing.on_drawn())(&song);
        std::thread::sleep(Duration::from_millis(200));
        assert!(playing.story().is_none(), "not wanted");
        playing.show_scenes(true);
        assert!(until(|| playing
            .story()
            .is_some_and(|s| s.scenes.len() == 2)));
        playing.show_scenes(false);
        assert!(playing.story().is_none() && playing.look().is_some());
        playing.show_scenes(true);
        assert!(until(|| playing.story().is_some()));
        playing.want(&dir.join("another.wav"));
        assert!(playing.story().is_none() && playing.look().is_none());
    }
}
