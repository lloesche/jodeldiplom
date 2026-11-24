//! The covers' step of a song's preparation: a thread of its own beside
//! the analysis and the lyrics, getting the queue's covers taken apart in
//! the queue's order (the song playing first), each told to the app as
//! [`Job::Cover`] (working, then ready), and holding the song playing's
//! layers for the app ([`CoverWorker::current`]). Its GPU work gives way
//! to the window's frames ([`jodel_gpu::pace`]); the models are set up on
//! the first cover and let go when the work has been done for a while.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use image::RgbImage;
use jodel_gpu::vision::prep::Rgb;
use jodel_gpu::{GpuDepth, GpuMatte};
use jodel_playback::{Cover, Tags, TrackInfo};

use super::store::{CoverStore, SongCover, Splitter};
use super::{album_keys, split, Apart, CoverLayers, CoverModels, Layers, Parts};
use crate::browser::Notify;
use crate::core::{Action, Facts, Job};

/// How long the models stay set up after the last cover.
const KEEP_MODELS: Duration = Duration::from_secs(30);
/// The first wait before setting the models up again after the GPU
/// failed; doubled each time, up to [`MOST_BACKOFF`].
const FIRST_BACKOFF: Duration = Duration::from_secs(30);
const MOST_BACKOFF: Duration = Duration::from_secs(30 * 60);

/// The models, set up when first needed.
enum Models {
    NotYet,
    Ready(Box<(GpuDepth, GpuMatte)>),
    /// The weights cannot be read: for good (setup checked them, so this
    /// is a file gone since).
    NoWeights(String),
    /// The GPU failed: tried again after `until`, the wait doubled.
    Failed {
        why: String,
        until: Instant,
        wait: Duration,
    },
}

/// The covers taken apart by the two models on the GPU.
pub struct GpuSplitter {
    files: CoverModels,
    models: Mutex<Models>,
}

impl GpuSplitter {
    /// The splitter of the weights in `files`; its models are set up on the
    /// first cover.
    pub fn new(files: CoverModels) -> GpuSplitter {
        GpuSplitter {
            files,
            models: Mutex::new(Models::NotYet),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Models> {
        self.models.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets the models up if they are not, as their state allows.
    fn set_up(&self, models: &mut Models) -> Result<(), String> {
        let wait = match &*models {
            Models::Ready(_) => return Ok(()),
            Models::NoWeights(why) => return Err(why.clone()),
            Models::Failed { why, until, .. } if Instant::now() < *until => return Err(why.clone()),
            Models::Failed { wait, .. } => (*wait * 2).min(MOST_BACKOFF),
            Models::NotYet => FIRST_BACKOFF,
        };
        let read =
            |path: &Path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
        let (depth, matte) = match (read(&self.files.depth), read(&self.files.matte)) {
            (Ok(d), Ok(m)) => (d, m),
            (Err(e), _) | (_, Err(e)) => {
                *models = Models::NoWeights(e.clone());
                return Err(e);
            }
        };
        let made = GpuDepth::new(depth).and_then(|d| Ok((d, GpuMatte::new(matte)?)));
        *models = match made {
            Ok(m) => Models::Ready(Box::new(m)),
            Err(why) => Models::Failed {
                why: why.clone(),
                until: Instant::now() + wait,
                wait,
            },
        };
        match &*models {
            Models::Failed { why, .. } => Err(why.clone()),
            _ => Ok(()),
        }
    }

    /// What the models find in `image`, setting them up first if need be.
    /// Each model's working buffers go once it has run, so only one
    /// model's are held at a time (their weights stay).
    pub fn parts(&self, image: &RgbImage) -> Result<Parts, String> {
        let mut models = self.lock();
        self.set_up(&mut models)?;
        let Models::Ready(m) = &*models else {
            unreachable!("set up");
        };
        let (depth, matte) = (&m.0, &m.1);
        let rgb = Rgb::new(
            image.width() as usize,
            image.height() as usize,
            image.as_raw().clone(),
        );
        let found = depth.depth(&rgb);
        depth.release();
        let found = found?;
        let cut = matte.matte(&rgb);
        matte.release();
        Ok(Parts {
            depth: found,
            matte: cut?,
        })
    }

    /// The bytes the models keep on the GPU between covers (their
    /// weights), and at most while a cover is taken apart (the weights and
    /// the larger model's working buffers: one runs at a time).
    pub fn gpu_bytes(&self) -> (u64, u64) {
        match &*self.lock() {
            Models::Ready(m) => {
                let ((dw, dp), (mw, mp)) = (m.0.gpu_sizes(), m.1.gpu_sizes());
                (dw + mw, dw + mw + dp.max(mp))
            }
            _ => (0, 0),
        }
    }
}

impl Splitter for GpuSplitter {
    fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
        Ok(split(image, &self.parts(image)?))
    }

    fn rest(&self) {
        let mut models = self.lock();
        if matches!(*models, Models::Ready(_)) {
            *models = Models::NotYet;
        }
    }
}

/// The song playing, whose layers the app wants.
struct Wanted {
    path: PathBuf,
    cover: Option<Cover>,
    tags: Tags,
    /// Its cover's bytes' hash.
    bytes: Option<u64>,
}

/// The song playing's layers, with the hash of its cover's bytes as the
/// song brought them ([`bytes_hash`]): a song after it with the very same
/// bytes (an album's) keeps them.
struct Held {
    bytes: Option<u64>,
    layers: Layers,
}

/// The hash of a song's cover's bytes as its file has them (none without
/// a cover): cheap enough for the window's thread.
fn bytes_hash(cover: Option<&Cover>) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    cover.map(|c| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        c.data.hash(&mut h);
        h.finish()
    })
}

struct Queue {
    wanted: Option<Wanted>,
    prepare: VecDeque<PathBuf>,
    stop: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
    /// The song playing's layers, once ready.
    current: Mutex<Option<Held>>,
}

/// The covers' worker thread.
pub struct CoverWorker {
    shared: Arc<Shared>,
}

impl Drop for CoverWorker {
    fn drop(&mut self) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stop = true;
        self.shared.ready.notify_all();
    }
}

impl CoverWorker {
    /// Starts the thread: covers kept in `dir`, taken apart by the models
    /// in `models`; each song prepared is told on `sender` (and `notify`
    /// called, also when the song playing's layers arrive).
    pub fn start(
        dir: PathBuf,
        models: CoverModels,
        sender: Sender<Action>,
        notify: Notify,
    ) -> CoverWorker {
        CoverWorker::start_with(
            CoverStore::new(dir),
            Box::new(GpuSplitter::new(models)),
            sender,
            notify,
        )
    }

    /// [`CoverWorker::start`] with a store and a splitter of its own.
    pub fn start_with(
        store: CoverStore,
        splitter: Box<dyn Splitter>,
        sender: Sender<Action>,
        notify: Notify,
    ) -> CoverWorker {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                wanted: None,
                prepare: VecDeque::new(),
                stop: false,
            }),
            ready: Condvar::new(),
            current: Mutex::new(None),
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("covers".into())
            .spawn(move || {
                jodel_gpu::pace::background();
                work(&worker, &store, &*splitter, &sender, &notify);
            })
            .expect("a thread");
        CoverWorker { shared }
    }

    /// The song playing is now `info`'s: its own layers are made (or found)
    /// first, and the last one's go at once unless its cover's bytes are
    /// the very same (an album's songs one after another keep them; the
    /// worker keeps them too where the store gives the song the same cover
    /// result).
    pub fn want(&self, info: &TrackInfo) {
        let bytes = bytes_hash(info.cover.as_ref());
        {
            let mut current = self
                .shared
                .current
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if current
                .as_ref()
                .is_none_or(|h| bytes.is_none() || h.bytes != bytes)
            {
                *current = None;
            }
        }
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .wanted = Some(Wanted {
            path: info.path.clone(),
            cover: info.cover.clone(),
            tags: info.tags.clone(),
            bytes,
        });
        self.shared.ready.notify_all();
    }

    /// The song playing's layers, once ready.
    pub fn current(&self) -> Option<Layers> {
        self.shared
            .current
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|h| Arc::clone(&h.layers))
    }

    /// Gets `paths`' covers taken apart, in this order (what was asked for
    /// before and not started is dropped).
    pub fn prepare(&self, paths: Vec<PathBuf>) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prepare = paths.into();
        self.shared.ready.notify_all();
    }
}

/// A song's cover's step done: its layers (when asked for and it has a
/// cover), whether it has them; or why not, when its file could not be
/// read or its cover not taken apart (nothing is kept then: it is tried
/// again).
enum Taken {
    Done { layers: Option<Layers>, has: bool },
    Failed(String),
}

/// Whether [`take_apart`] gives a song's layers back: not, or so, with the
/// layers held now (kept rather than read again where they are the same
/// cover result).
enum Back<'a> {
    No,
    Yes(Option<&'a Layers>),
}

/// Takes `path`'s cover apart (or finds it taken apart), its cover and
/// tags `given` (the song playing's) or read from its file, its layers
/// given `back` or not. `working` is told while work is under way.
fn take_apart(
    store: &CoverStore,
    splitter: &dyn Splitter,
    path: &Path,
    given: Option<(Option<Cover>, Tags)>,
    back: Back,
    working: &mut dyn FnMut(bool),
) -> Taken {
    let key = super::song_key(path);
    let layers = matches!(back, Back::Yes(_));
    if store.ready(&key) {
        let Some(Some(hash)) = store.linked(&key) else {
            return Taken::Done {
                layers: None,
                has: false,
            };
        };
        let held = match back {
            Back::Yes(held) => held.filter(|h| h.key == hash).cloned(),
            Back::No => None,
        };
        let load = || {
            store
                .load(&hash)
                .map(|k| Arc::new(Apart::new(hash.clone(), k.layers)))
        };
        let found = layers.then(|| held.or_else(load)).flatten();
        return Taken::Done {
            layers: found,
            has: true,
        };
    }
    working(true);
    let (cover, tags) = match given {
        Some(g) => g,
        None => match jodel_playback::source::probe(path) {
            Ok(info) => (info.cover, info.tags),
            Err(e) => {
                working(false);
                return Taken::Failed(format!("{}: {e}", path.display()));
            }
        },
    };
    let image = match cover {
        None => None,
        Some(c) => match crate::art::decode(&c.data) {
            Some(image) => Some(image),
            None => {
                working(false);
                return Taken::Failed(format!("{}: its cover is not an image", path.display()));
            }
        },
    };
    let song = SongCover {
        song: key,
        albums: album_keys(path, Some(&tags)),
        image,
    };
    let made = store.layers(&song, splitter);
    working(false);
    match made {
        Ok(found) => Taken::Done {
            has: found.is_some(),
            layers: found.filter(|_| layers),
        },
        Err(why) => Taken::Failed(why),
    }
}

fn work(
    shared: &Shared,
    store: &CoverStore,
    splitter: &dyn Splitter,
    sender: &Sender<Action>,
    notify: &Notify,
) {
    let mut last_work: Option<Instant> = None;
    loop {
        let (wanted, path) = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if q.stop {
                    return;
                }
                if let Some(w) = q.wanted.take() {
                    break (Some(w), None);
                }
                if let Some(p) = q.prepare.pop_front() {
                    break (None, Some(p));
                }
                // Nothing to do: the models go after a while.
                let wait = match last_work {
                    Some(at) if at.elapsed() < KEEP_MODELS => KEEP_MODELS - at.elapsed(),
                    Some(_) => {
                        splitter.rest();
                        last_work = None;
                        continue;
                    }
                    None => Duration::from_secs(3600),
                };
                q = shared
                    .ready
                    .wait_timeout(q, wait)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        let tell = |action: Action| {
            let sent = sender.send(action).is_ok();
            notify();
            sent
        };
        let (path, given, bytes) = match (wanted, path) {
            (Some(w), _) => (w.path, Some((w.cover, w.tags)), w.bytes),
            (None, Some(p)) => (p, None, None),
            (None, None) => continue,
        };
        let playing = given.is_some();
        let held = shared
            .current
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|h| Arc::clone(&h.layers));
        let mut working = |on: bool| {
            if on {
                last_work = Some(Instant::now());
            }
            tell(Action::Working {
                path: path.clone(),
                job: Job::Cover,
                on,
            });
        };
        let back = if playing {
            Back::Yes(held.as_ref())
        } else {
            Back::No
        };
        let taken = take_apart(store, splitter, &path, given, back, &mut working);
        let has = match taken {
            Taken::Done { layers, has } => {
                if playing {
                    // Still the song playing (no other wanted since).
                    let still = shared
                        .queue
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .wanted
                        .is_none();
                    if still {
                        *shared.current.lock().unwrap_or_else(|e| e.into_inner()) =
                            layers.map(|layers| Held { bytes, layers });
                    }
                }
                has
            }
            Taken::Failed(why) => {
                crate::trace::note(format_args!("covers: {why}"));
                false
            }
        };
        if !tell(Action::Ready {
            path,
            job: Job::Cover,
            facts: Some(Facts::Cover(has)),
        }) {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cover::store::tests::Counting;
    use std::sync::atomic::Ordering;

    /// A splitter shared with the test, to count its calls.
    struct Shared2(Arc<Counting>);

    impl Splitter for Shared2 {
        fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
            self.0.split(image)
        }
    }

    fn temp(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jodel-cover-worker-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Waits for `count` covers to be told ready; their facts.
    fn ready(
        rx: &std::sync::mpsc::Receiver<Action>,
        count: usize,
    ) -> Vec<(PathBuf, Option<Facts>)> {
        let mut out = Vec::new();
        while out.len() < count {
            if let Action::Ready { path, facts, .. } =
                rx.recv_timeout(Duration::from_secs(20)).expect("told")
            {
                out.push((path, facts));
            }
        }
        out
    }

    #[test]
    fn an_albums_songs_one_after_another_take_its_cover_apart_once() {
        let dir = temp("album");
        let album = dir.join("Album");
        std::fs::create_dir_all(&album).unwrap();
        crate::art::sample_cover(128)
            .save(album.join("cover.png"))
            .unwrap();
        let tracks: Vec<PathBuf> = (1..=3)
            .map(|i| {
                let p = album.join(format!("0{i}.wav"));
                std::fs::write(&p, crate::keep::wav(&vec![0i16; 4410 * i], 44_100)).unwrap();
                p
            })
            .collect();
        // And a file that cannot be read: told, not kept as "no cover".
        let broken = album.join("04.wav");
        std::fs::write(&broken, b"not audio").unwrap();
        let counting = Arc::new(Counting::default());
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = CoverWorker::start_with(
            CoverStore::new(dir.join("covers")),
            Box::new(Shared2(Arc::clone(&counting))),
            tx,
            Arc::new(|| {}),
        );
        let mut all = tracks.clone();
        all.push(broken.clone());
        worker.prepare(all);
        let told = ready(&rx, 4);
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);
        for (path, facts) in told {
            let has = path != broken;
            assert_eq!(facts, Some(Facts::Cover(has)), "{}", path.display());
        }
        let store = CoverStore::new(dir.join("covers"));
        assert!(store.linked(&crate::cover::song_key(&broken)).is_none());
        // The song playing: its layers held for the app; another song
        // playing clears them at once.
        let info = jodel_playback::source::probe(&tracks[0]).unwrap();
        worker.want(&info);
        ready(&rx, 1);
        let first = worker.current().expect("held");
        // The album's next song, the same cover: the very layers stay,
        // neither dropped nor read again.
        let next = jodel_playback::source::probe(&tracks[1]).unwrap();
        assert!(next.cover.is_some());
        worker.want(&next);
        assert!(worker.current().is_some_and(|l| Arc::ptr_eq(&l, &first)));
        ready(&rx, 1);
        assert!(worker.current().is_some_and(|l| Arc::ptr_eq(&l, &first)));
        let mut none = info.clone();
        none.cover = None;
        none.path = dir.join("elsewhere.wav");
        worker.want(&none);
        assert!(worker.current().is_none());
        ready(&rx, 1);
        assert!(worker.current().is_none());
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1);
    }
}
