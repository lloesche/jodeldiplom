//! The covers' step of a song's preparation: a thread of its own beside
//! the analysis and the lyrics, getting the queue's covers taken apart in
//! the queue's order (the song playing first), each told to the app as
//! [`Job::Cover`] (working, then ready), and holding the song playing's
//! layers for the app ([`CoverWorker::current`]); after the song playing's
//! cover, its story scenes made ready with the same models
//! ([`crate::work::scenes::story`], told as the cover's work while its
//! models run, given up as soon as another song is wanted; held for the
//! app: [`CoverWorker::story`]). Its GPU work gives way
//! to the window's frames ([`jodel_gpu::pace`]); the models are set up on
//! the first cover and let go when the work has been done for a while.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use image::RgbImage;
use jodel_gpu::vision::prep::Rgb;
use jodel_gpu::vision::{Depth, Matte};
use jodel_gpu::{GpuDepth, GpuMatte};
use jodel_playback::{Cover, Tags, TrackInfo};

use super::store::{CoverStore, SongCover, Splitter};
use super::{
    album_keys, identity, split, Apart, CoverLayers, CoverModels, FaceNet, Layers, Parts, TextNet,
};
use crate::core::{Action, Facts, Job};
use crate::work::outbox::Outbox;
use crate::work::scenes::{SceneNets, SceneStore, Story};
use crate::work::{Next, Queue};

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

    /// What `run` makes of the two models, setting them up first if need
    /// be (and for the next picture again where their GPU was lost).
    fn with_models<T>(
        &self,
        run: impl FnOnce(&GpuDepth, &GpuMatte) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut models = self.lock();
        self.set_up(&mut models)?;
        let Models::Ready(m) = &*models else {
            unreachable!("set up");
        };
        let done = run(&m.0, &m.1);
        let lost = m.0.is_lost() || m.1.is_lost();
        if done.is_err() && lost {
            *models = Models::NotYet;
        }
        done
    }

    /// What the models find in `image`: its depth, then its matte (one
    /// model's working buffers held at a time); the lettering and the
    /// faces found on the processor after them.
    pub fn parts(&self, image: &RgbImage) -> Result<Parts, String> {
        let rgb = Rgb::new(
            image.width() as usize,
            image.height() as usize,
            image.as_raw().clone(),
        );
        Ok(Parts {
            depth: SceneNets::depth(self, &rgb)?,
            matte: SceneNets::matte(self, &rgb)?,
            text: TextNet::built_in().find(image),
            faces: FaceNet::built_in().find(image),
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

/// Each model's working buffers let go once it has run (its weights stay).
impl SceneNets for GpuSplitter {
    fn depth(&self, picture: &Rgb) -> Result<Depth, String> {
        self.with_models(|depth, _| {
            let found = depth.depth(picture);
            depth.release();
            found
        })
    }

    fn matte(&self, picture: &Rgb) -> Result<Matte, String> {
        self.with_models(|_, matte| {
            let cut = matte.matte(picture);
            matte.release();
            cut
        })
    }
}

impl Splitter for GpuSplitter {
    fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
        Ok(split(image, &self.parts(image)?))
    }

    fn nets(&self) -> Option<&dyn SceneNets> {
        Some(self)
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
    /// Its cover's [`identity`].
    bytes: Option<u64>,
}

/// The song playing's layers, with the [`identity`] of its cover as the
/// song brought it: a song after it with the very same bytes (an album's)
/// keeps them.
struct Held {
    bytes: Option<u64>,
    layers: Layers,
}

/// What the worker is asked for: the song playing first, then the songs
/// to get ready.
#[derive(Default)]
struct Asked {
    wanted: Option<Wanted>,
    prepare: VecDeque<PathBuf>,
}

/// The next song to take apart: the one playing, or one to get ready.
enum Ask {
    Playing(Box<Wanted>),
    Prepare(PathBuf),
}

struct Shared {
    queue: Queue<Asked>,
    /// The song playing's layers, once ready.
    current: Mutex<Option<Held>>,
    /// The song playing's story scenes, once ready.
    story: Mutex<Option<Arc<Story>>>,
}

/// The covers' worker thread.
pub struct CoverWorker {
    shared: Arc<Shared>,
}

impl Drop for CoverWorker {
    fn drop(&mut self) {
        self.shared.queue.stop();
    }
}

impl CoverWorker {
    /// Starts the thread: covers kept in `dir`, taken apart by the models
    /// in `models`, the songs' story scenes in `scenes`; each song prepared
    /// is told on `outbox` (which is also woken when the song playing's
    /// layers or story arrive).
    pub fn start(
        dir: PathBuf,
        models: CoverModels,
        scenes: SceneStore,
        outbox: Outbox,
    ) -> CoverWorker {
        let splitter = Box::new(GpuSplitter::new(models));
        CoverWorker::start_with(CoverStore::new(dir), scenes, splitter, outbox)
    }

    /// [`CoverWorker::start`] with stores and a splitter of its own.
    pub fn start_with(
        store: CoverStore,
        scenes: SceneStore,
        splitter: Box<dyn Splitter>,
        outbox: Outbox,
    ) -> CoverWorker {
        let shared = Arc::new(Shared {
            queue: Queue::new(Asked::default()),
            current: Mutex::new(None),
            story: Mutex::new(None),
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("covers".into())
            .spawn(move || {
                jodel_gpu::pace::background();
                work(&worker, (&store, &scenes), &*splitter, &outbox);
            })
            .expect("a thread");
        CoverWorker { shared }
    }

    /// The song playing is now `info`'s: its own layers are made (or found)
    /// first, and the last one's go at once unless its cover's bytes are
    /// the very same (an album's songs one after another keep them; the
    /// worker keeps them too where the store gives the song the same cover
    /// result); its story goes at once. Both let go under the queue's lock
    /// as the song is asked for, so the worker never holds the last song's
    /// for it (the lock taken first, always).
    pub fn want(&self, info: &TrackInfo) {
        let bytes = info.cover.as_ref().map(identity);
        self.shared.queue.change(|asked| {
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
            *self.shared.story.lock().unwrap_or_else(|e| e.into_inner()) = None;
            asked.wanted = Some(Wanted {
                path: info.path.clone(),
                cover: info.cover.clone(),
                tags: info.tags.clone(),
                bytes,
            });
        });
    }

    /// The song playing's story scenes, once ready (none: it has none).
    pub fn story(&self) -> Option<Arc<Story>> {
        self.shared
            .story
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
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
            .change(|asked| asked.prepare = paths.into());
    }
}

/// A song's cover's step done: its layers (when asked for and it has a
/// cover, or taken apart now), whether it has them, and how long taking it
/// apart took when it was taken apart now (not found kept); or why not,
/// when its file could not be read or its cover not taken apart (nothing
/// is kept then: it is tried again).
enum Taken {
    Done {
        layers: Option<Layers>,
        has: bool,
        split: Option<Duration>,
    },
    Failed(String),
}

/// Whether [`take_apart`] gives a song's layers back: not, or so, with the
/// layers held now (kept rather than read again where they are the same
/// cover result).
enum Back<'a> {
    No,
    Yes(Option<&'a Layers>),
}

/// A splitter that keeps how long its taking a picture apart took.
struct Timed<'a> {
    splitter: &'a dyn Splitter,
    took: Mutex<Option<Duration>>,
}

impl Splitter for Timed<'_> {
    fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
        let started = Instant::now();
        let split = self.splitter.split(image);
        *self.took.lock().unwrap_or_else(|e| e.into_inner()) = Some(started.elapsed());
        split
    }
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
                split: None,
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
            split: None,
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
        Some(c) => match crate::work::art::decode(&c.data) {
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
    let timed = Timed {
        splitter,
        took: Mutex::new(None),
    };
    let made = store.layers(&song, &timed);
    working(false);
    let split = timed.took.into_inner().unwrap_or_else(|e| e.into_inner());
    match made {
        Ok(found) => Taken::Done {
            has: found.is_some(),
            layers: found.filter(|_| layers || split.is_some()),
            split,
        },
        Err(why) => Taken::Failed(why),
    }
}

fn work(
    shared: &Shared,
    (store, scenes): (&CoverStore, &SceneStore),
    splitter: &dyn Splitter,
    outbox: &Outbox,
) {
    let mut last_work: Option<Instant> = None;
    loop {
        // Nothing to do: the models go after a while.
        let rest = last_work.map_or(Duration::MAX, |at| KEEP_MODELS.saturating_sub(at.elapsed()));
        let next = shared.queue.next_within(rest, |asked| {
            (asked.wanted.take().map(|w| Ask::Playing(Box::new(w))))
                .or_else(|| asked.prepare.pop_front().map(Ask::Prepare))
        });
        let ask = match next {
            Next::Do(ask) => ask,
            Next::Idle => {
                splitter.rest();
                last_work = None;
                continue;
            }
            Next::Stop => return,
        };
        let (path, given, bytes) = match ask {
            Ask::Playing(w) => (w.path, Some((w.cover, w.tags)), w.bytes),
            Ask::Prepare(p) => (p, None, None),
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
            outbox.send(Action::Working {
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
            Taken::Done { layers, has, split } => {
                if let Some(took) = split {
                    crate::trace::note(format_args!(
                        "covers: {} taken apart in {:.1} s",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        took.as_secs_f64()
                    ));
                    // Taken apart now: its layers, as they were made.
                    if let Some(made) = &layers {
                        let cut = crate::core::prep::CoverCut(Arc::clone(made));
                        if !outbox.send(Action::Heard {
                            path: path.clone(),
                            part: crate::core::Heard::Cover(cut),
                        }) {
                            return;
                        }
                    }
                }
                if playing {
                    // Still the song playing (no other wanted since: held
                    // under the queue's lock, as `want` lets go).
                    let asked = shared.queue.lock();
                    if asked.wanted.is_none() {
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
        if !outbox.send(Action::Ready {
            path: path.clone(),
            job: Job::Cover,
            facts: Some(Facts::Cover(has)),
        }) {
            return;
        }
        // The song playing's story scenes, after its cover, with its
        // models (told as the cover's work while they run), given up as
        // soon as another song is wanted; held while it is still the song
        // playing (under the queue's lock, as `want` lets go).
        if playing {
            let still = || shared.queue.lock().wanted.is_none();
            let mut ran = false;
            let mut running = || {
                if !ran {
                    ran = true;
                    working(true);
                }
            };
            let story =
                crate::work::scenes::story(scenes, &path, splitter.nets(), (&still, &mut running));
            if ran {
                working(false);
            }
            let held = {
                let asked = shared.queue.lock();
                let held = asked.wanted.is_none();
                if held {
                    *shared.story.lock().unwrap_or_else(|e| e.into_inner()) = story;
                }
                held
            };
            if held {
                (outbox.notify())();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::cover::store::tests::Counting;
    use crate::work::scenes::{drawn, ByRed};
    use std::sync::atomic::Ordering;

    /// A splitter shared with the test, to count its calls.
    struct Shared2(Arc<Counting>);

    impl Splitter for Shared2 {
        fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
            self.0.split(image)
        }
    }

    /// A splitter with the scenes' stand-in models.
    struct WithNets(Counting, ByRed);

    impl Splitter for WithNets {
        fn split(&self, image: &RgbImage) -> Result<CoverLayers, String> {
            self.0.split(image)
        }

        fn nets(&self) -> Option<&dyn SceneNets> {
            Some(&self.1)
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
        let mut cover = durable::AtomicFile::create(&album.join("cover.png")).unwrap();
        crate::work::art::sample_cover(128)
            .write_to(&mut cover, image::ImageFormat::Png)
            .unwrap();
        cover.commit().unwrap();
        let tracks: Vec<PathBuf> = (1..=3)
            .map(|i| {
                let p = album.join(format!("0{i}.wav"));
                durable::write_atomic(&p, &jodel_dsp::wav::pcm16(44_100, 1, &vec![0i16; 4410 * i]))
                    .unwrap();
                p
            })
            .collect();
        // And a file that cannot be read: told, not kept as "no cover".
        let broken = album.join("04.wav");
        durable::write_atomic(&broken, b"not audio").unwrap();
        let counting = Arc::new(Counting::default());
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = CoverWorker::start_with(
            CoverStore::new(dir.join("covers")),
            SceneStore::new(dir.join("scenes"), dir.join("scene-parts")),
            Box::new(Shared2(Arc::clone(&counting))),
            Outbox::new(tx, Arc::new(|| {})),
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
        assert!(store
            .linked(&crate::work::cover::song_key(&broken))
            .is_none());
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

    /// The song playing's story scenes, where the drawing step left them,
    /// are made ready after its cover with the splitter's models and held
    /// for the app; another song playing lets them go at once.
    #[test]
    fn the_song_playing_s_story_is_made_ready_after_its_cover() {
        let dir = temp("story");
        let song = |name: &str| {
            let p = dir.join(name);
            durable::write_atomic(&p, &jodel_dsp::wav::pcm16(44_100, 1, &[0i16; 4410])).unwrap();
            jodel_playback::source::probe(&p).unwrap()
        };
        let (first, other) = (song("01.wav"), song("02.wav"));
        let scenes = SceneStore::new(dir.join("scenes"), dir.join("scene-parts"));
        drawn(&scenes, &first.path);
        let (tx, rx) = std::sync::mpsc::channel();
        let splitter = WithNets(Counting::default(), ByRed::default());
        let worker = CoverWorker::start_with(
            CoverStore::new(dir.join("covers")),
            scenes,
            Box::new(splitter),
            Outbox::new(tx, Arc::new(|| {})),
        );
        worker.want(&first);
        ready(&rx, 1);
        let started = Instant::now();
        let story = loop {
            if let Some(story) = worker.story() {
                break story;
            }
            assert!(started.elapsed() < Duration::from_secs(20), "no story");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(story.scenes.len(), 2);
        assert_eq!(story.scenes[0].moved.count(), 2);
        worker.want(&other);
        assert!(worker.story().is_none());
        ready(&rx, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
