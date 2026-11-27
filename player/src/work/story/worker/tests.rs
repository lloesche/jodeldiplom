//! Stories on stand-in models (no GPU): a round planning, then encoding,
//! then drawing, the models asked only for work; what is kept and drawn
//! again (a song of a real length drawn once); the scenes described with
//! their times and the look; a song no longer wanted given up between its
//! pictures; no models, or no timed lines, no story; the thread taking up
//! the songs wanted, their lyrics changed too. The models' turns among
//! threads are the stages' (`crate::work::stages`); the whole of it on
//! the real models, from the app, is `tests/story.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Mutex;
use std::time::Duration;

use jodel_gpu::image::{Embeds, Picture, Reference};
use jodel_lyrics::{Line as Sung, Source, Word};
use jodel_testkit::TempDir;

use super::*;
use crate::core::story::pictures::{PORTRAIT, SCENE};
use crate::core::story::tests::{song, GOOD};
use crate::work::scenes::Scenes;
use crate::work::story::draw::{Encoder, Painter, REFERENCE};
use crate::work::story::{Writer, Written};

/// A song's length, seconds: not a round one, as no real song's is.
const LENGTH: f64 = 213.4567;

/// The songs' rate: one their length in seconds has no end of decimals at.
const RATE: u32 = 7_000;

/// The songs' frames: [`LENGTH`] at [`RATE`], as near as there are.
fn frames() -> usize {
    (LENGTH * f64::from(RATE)) as usize
}

/// What the stand-in models did, in order.
type Log = Arc<Mutex<Vec<String>>>;

/// Stand-in models: the language model answers `answer`, the encoder
/// encodes a prompt as its length, the image model paints a picture of
/// its seed's colour (`painted` counting them; where `gate` says, it
/// waits after its picture of that count until let go); each logs what
/// it does; none runs where `none`.
struct Fakes {
    log: Log,
    answer: Mutex<String>,
    painted: AtomicUsize,
    gate: Mutex<Option<(usize, Receiver<()>)>>,
    none: bool,
}

impl Fakes {
    fn new() -> Fakes {
        Fakes {
            log: Log::default(),
            answer: Mutex::new(GOOD.to_owned()),
            painted: AtomicUsize::new(0),
            gate: Mutex::new(None),
            none: false,
        }
    }

    /// The pictures painted so far.
    fn painted(&self) -> usize {
        self.painted.load(Ordering::SeqCst)
    }

    /// Runs where there are models.
    fn up(&self) -> Result<(), String> {
        if self.none {
            return Err("no GPU".into());
        }
        Ok(())
    }

    /// The log, emptied.
    fn took(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }

    /// `what` logged.
    fn did(&self, what: String) {
        self.log.lock().unwrap().push(what);
    }
}

/// The stand-ins as their kinds of model.
struct Writes<'a>(&'a Fakes, String);
struct Encodes<'a>(&'a Fakes);
struct Paints<'a>(&'a Fakes);

impl Writer for Writes<'_> {
    fn write(&self, _: &str, _: usize, seen: &mut dyn FnMut(&str)) -> Result<Written, Failure> {
        self.0.did("plan".into());
        seen(&self.1);
        Ok(Written {
            text: self.1.clone(),
            stopped: true,
            ..Written::default()
        })
    }
}

impl Encoder for Encodes<'_> {
    fn encode(&mut self, prompt: &str) -> Result<Embeds, String> {
        self.0.did("encode".into());
        Ok(Embeds {
            values: vec![prompt.len() as f32],
        })
    }
}

impl Painter for Paints<'_> {
    fn paint(
        &mut self,
        _: &Embeds,
        seed: u64,
        (width, height): (usize, usize),
        refs: &[Reference],
    ) -> Result<Picture, String> {
        let sizes = refs.iter().map(|r| (r.width, r.height));
        assert!(sizes.into_iter().all(|s| s == (REFERENCE, REFERENCE)));
        self.0.did(paint((width, height), refs.len()));
        let n = self.0.painted.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some((at, go)) = &*self.0.gate.lock().unwrap() {
            if n == *at {
                go.recv().expect("let go");
            }
        }
        Ok(Picture {
            width,
            height,
            rgb: vec![seed as u8; width * height * 3],
        })
    }
}

/// How a picture painted is logged.
fn paint((w, h): (usize, usize), refs: usize) -> String {
    format!("paint {w}x{h} {refs}")
}

impl Models for Fakes {
    fn plan(&self, job: &StoryJob, tell: &mut dyn FnMut(Story)) -> Result<Plan, Failure> {
        self.up().map_err(Failure::Model)?;
        let answer = self.answer.lock().unwrap().clone();
        job.run(&Writes(self, answer), tell)
    }

    fn encode(&self, drawing: &Drawing) -> Result<Encoded, String> {
        self.up()?;
        drawing.encode(&mut Encodes(self))
    }

    fn draw(
        &self,
        drawing: &Drawing,
        encoded: &mut Encoded,
        still: &dyn Fn() -> bool,
        tell: &mut dyn FnMut(Story),
    ) -> Result<Drawn, String> {
        self.up()?;
        drawing.draw(encoded, &mut Paints(self), still, tell)
    }
}

/// The lyrics of [`song`], every word of a line between its start and
/// its end; `timed` false: none timed at all.
fn lyrics(timed: bool) -> Arc<Lyrics> {
    let lines = (song().lines.iter())
        .map(|l| {
            let mut words: Vec<Word> = l.text.split(' ').map(Word::new).collect();
            if timed {
                words[0].start = Some(l.start);
                words.last_mut().expect("a word").end = Some(l.end);
            }
            Sung {
                words,
                ..Sung::default()
            }
        })
        .collect();
    Arc::new(Lyrics {
        lines,
        source: Source::Lrclib,
        timing: Default::default(),
    })
}

/// A data folder's places in a new folder for `tag`, and `songs` songs
/// in it ([`frames`] of silence).
fn setting(tag: &str, songs: usize) -> (TempDir, Places, Vec<PathBuf>) {
    let silence = vec![0i16; frames()];
    let dir = TempDir::new(&format!("stories-{tag}"));
    let places = Places {
        stories: dir.join("stories"),
        scenes: SceneStore::new(dir.join("scenes"), dir.join("scene-parts")),
    };
    let paths = (0..songs)
        .map(|k| {
            let path = dir.join(format!("{k:02}.wav"));
            durable::write_atomic(&path, &jodel_dsp::wav::pcm16(RATE, 1, &silence)).unwrap();
            path
        })
        .collect();
    (dir, places, paths)
}

/// The song at `path` with [`lyrics`] timed.
fn wanted(path: &Path) -> StorySong {
    StorySong {
        path: path.to_owned(),
        lyrics: lyrics(true),
    }
}

/// An outbox, and what it is sent.
fn outbox() -> (Outbox, Receiver<Action>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (Outbox::new(tx, Arc::new(|| {})), rx)
}

/// The scenes `rx` was told of, "N/M", of each song (none of its words).
fn told(rx: &Receiver<Action>) -> Vec<(PathBuf, String)> {
    rx.try_iter()
        .filter_map(|action| match action {
            Action::Heard {
                path,
                part: Heard::Story(Story::Scene { index, of, .. }),
            } => Some((path, format!("{index}/{of}"))),
            _ => None,
        })
        .collect()
}

/// The plan [`GOOD`] makes of [`song`] as long as the songs written.
fn good() -> Plan {
    let song = Song {
        duration: frames() as f64 / f64::from(RATE),
        ..song()
    };
    crate::core::story::check::read(GOOD, &song).unwrap()
}

/// The folder of the song at `path`'s scenes in `places`.
fn folder(places: &Places, path: &Path) -> PathBuf {
    places.scenes.folder(&crate::work::cover::song_key(path))
}

/// One round of `songs`, each always wanted; the songs told drawn.
fn round_of(songs: &[StorySong], models: &Fakes, places: &Places, out: &Outbox) -> Vec<PathBuf> {
    let drawn = Mutex::new(Vec::new());
    let on_drawn = |path: &Path| drawn.lock().unwrap().push(path.to_owned());
    round(songs, models, places, &|_| true, out, &on_drawn);
    drawn.into_inner().unwrap()
}

/// A round plans every song wanted, then encodes their prompts, then
/// draws their pictures: a song's portraits first, then its scenes, each
/// with the portraits of those in it; each scene told as it lands, and
/// the song told drawn once described. Its folder then holds a picture a
/// scene, with the plan's times and look.
#[test]
fn a_round_plans_every_song_then_encodes_then_draws() {
    let (_dir, places, paths) = setting("order", 2);
    let fakes = Fakes::new();
    let (out, rx) = outbox();
    let songs: Vec<StorySong> = paths.iter().map(|p| wanted(p)).collect();
    assert_eq!(round_of(&songs, &fakes, &places, &out), paths);
    let plan = good();
    let pictures = plan.characters.len() + plan.scenes.len();
    let mut want = vec!["plan".to_owned(); paths.len()];
    want.extend(vec!["encode".to_owned(); paths.len() * pictures]);
    for _ in &paths {
        want.extend(vec![paint(PORTRAIT, 0); plan.characters.len()]);
        want.extend((plan.scenes.iter()).map(|s| paint(SCENE, s.characters.len())));
    }
    assert_eq!(fakes.took(), want);
    let news = told(&rx);
    let of = plan.scenes.len();
    for path in &paths {
        let scenes: Vec<&str> = (news.iter())
            .filter(|(p, _)| p == path)
            .map(|(_, n)| n.as_str())
            .collect();
        let want: Vec<String> = (0..of).map(|k| format!("{k}/{of}")).collect();
        assert_eq!(scenes, want);
        // Its scenes as the plan times them, the plan's look over them.
        let kept = Scenes::read(&folder(&places, path)).expect("described");
        let spans: Vec<(f64, f64)> = kept.scenes.iter().map(|s| (s.start, s.end)).collect();
        let planned: Vec<(f64, f64)> = (plan.scenes.iter())
            .map(|s| (millis(s.from), millis(s.to)))
            .collect();
        assert_eq!(spans, planned);
        assert_eq!(kept.look, plan.look);
        let first = image::open(&kept.scenes[0].picture).unwrap();
        assert_eq!((first.width() as usize, first.height() as usize), SCENE);
    }
}

/// `t` as a description keeps it, to the millisecond.
fn millis(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

/// A song drawn is not drawn again, nor told of (whatever its length):
/// its plan kept, its pictures there, its description made as its plan
/// makes it. A picture gone is drawn again alone (its scene's portraits
/// read back from the folder); a plan made again draws again only the
/// pictures it changed, and lets the old ones go.
#[test]
fn a_song_drawn_is_not_drawn_again() {
    let (_dir, places, paths) = setting("kept", 1);
    let fakes = Fakes::new();
    let (out, rx) = outbox();
    let song = [wanted(&paths[0])];
    round_of(&song, &fakes, &places, &out);
    fakes.took();
    told(&rx);
    assert!(round_of(&song, &fakes, &places, &out).is_empty());
    assert_eq!(fakes.took(), Vec::<String>::new());
    assert!(rx.try_iter().next().is_none(), "nothing told");
    // The second scene's picture gone: it alone, with its two portraits.
    let folder = folder(&places, &paths[0]);
    let second = Scenes::read(&folder).unwrap().scenes[1].picture.clone();
    std::fs::remove_file(&second).unwrap();
    assert_eq!(round_of(&song, &fakes, &places, &out), paths);
    assert_eq!(fakes.took(), ["encode".to_owned(), paint(SCENE, 2)]);
    assert_eq!(told(&rx).len(), 1);
    assert!(second.is_file());
    // New lyrics, a new plan: one scene's prompt is another, so that
    // picture alone is drawn again, the old one let go.
    let before = std::fs::read_dir(&folder).unwrap().count();
    *fakes.answer.lock().unwrap() = GOOD.replace("Lonely lighthouse", "Tall lighthouse");
    let mut other = (*lyrics(true)).clone();
    other.lines[0].words[0].text = "We".into();
    let other = StorySong {
        path: paths[0].clone(),
        lyrics: Arc::new(other),
    };
    round_of(&[other], &fakes, &places, &out);
    let want = ["plan".to_owned(), "encode".into(), paint(SCENE, 0)];
    assert_eq!(fakes.took(), want);
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), before);
    let kept = Scenes::read(&folder).unwrap();
    assert!(kept.scenes.iter().all(|s| s.picture.is_file()));
}

/// A song no longer wanted is given up between its pictures: those drawn
/// kept, no description, nothing told drawn; the next round draws only
/// the rest.
#[test]
fn a_song_no_longer_wanted_is_given_up_between_its_pictures() {
    let (_dir, places, paths) = setting("given-up", 1);
    let fakes = Fakes::new();
    let (out, _rx) = outbox();
    let song = [wanted(&paths[0])];
    // Wanted until its third picture is painted.
    const DRAWN: usize = 3;
    let still = |_: &StorySong| fakes.painted() < DRAWN;
    let drawn = Mutex::new(Vec::new());
    let on_drawn = |path: &Path| drawn.lock().unwrap().push(path.to_owned());
    round(&song, &fakes, &places, &still, &out, &on_drawn);
    assert_eq!(fakes.painted(), DRAWN);
    assert!(drawn.lock().unwrap().is_empty());
    let folder = folder(&places, &paths[0]);
    assert!(Scenes::read(&folder).is_none());
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), DRAWN);
    assert_eq!(round_of(&song, &fakes, &places, &out), paths);
    let plan = good();
    let all = plan.characters.len() + plan.scenes.len();
    assert_eq!(fakes.painted(), all);
}

/// No models to run (no GPU): nothing kept, nothing told; a song with no
/// timed lines asks no model at all.
#[test]
fn without_models_or_timed_lines_there_is_no_story() {
    let (dir, places, paths) = setting("none", 1);
    let (out, rx) = outbox();
    let fakes = Fakes {
        none: true,
        ..Fakes::new()
    };
    assert!(round_of(&[wanted(&paths[0])], &fakes, &places, &out).is_empty());
    assert!(!dir.join("stories").exists() && !dir.join("scenes").exists());
    assert!(rx.try_iter().next().is_none());
    let fakes = Fakes::new();
    let untimed = StorySong {
        path: paths[0].clone(),
        lyrics: lyrics(false),
    };
    assert!(round_of(&[untimed], &fakes, &places, &out).is_empty());
    assert_eq!(fakes.took(), Vec::<String>::new());
    assert!(!dir.join("stories").exists() && !dir.join("scenes").exists());
}

/// A song is still wanted while it is in the list with the very same
/// lyrics: not with others (found or timed again), nor out of it.
#[test]
fn a_song_is_still_wanted_with_its_very_lyrics() {
    let (a, b) = (wanted(Path::new("a.flac")), wanted(Path::new("b.flac")));
    let asked = Asked {
        songs: vec![a.clone()],
        times: 1,
    };
    assert!(asked.still(&a));
    assert!(!asked.still(&wanted(Path::new("a.flac"))), "other lyrics");
    assert!(!asked.still(&b));
}

/// The thread takes up the songs wanted, each drawn once; asked again
/// with a song's lyrics changed, it plans that song again (and draws what
/// its new plan changed).
#[test]
fn the_worker_takes_up_the_songs_wanted_and_their_new_lyrics() {
    let (_dir, places, paths) = setting("thread", 2);
    let fakes = Arc::new(Fakes::new());
    let (worker, _told, drawn) = started(&fakes, places);
    let wait = || drawn.recv_timeout(WAIT).expect("drawn");
    worker.want(paths.iter().map(|p| wanted(p)).collect());
    let mut both = vec![wait(), wait()];
    both.sort();
    assert_eq!(both, paths);
    let plans = |log: Vec<String>| log.iter().filter(|e| *e == "plan").count();
    assert_eq!(plans(fakes.took()), 2);
    let mut other = (*lyrics(true)).clone();
    other.lines[0].words[0].text = "We".into();
    *fakes.answer.lock().unwrap() = GOOD.replace("Lonely lighthouse", "Tall lighthouse");
    worker.want(vec![StorySong {
        path: paths[1].clone(),
        lyrics: Arc::new(other),
    }]);
    assert_eq!(wait(), paths[1]);
    assert_eq!(plans(fakes.took()), 1);
    drop(worker);
}

/// How long a test waits for the thread.
const WAIT: Duration = Duration::from_secs(60);

/// The thread on `fakes`, its stories kept in `places`; what it tells on
/// its outbox, and the songs it tells drawn.
fn started(
    fakes: &Arc<Fakes>,
    places: Places,
) -> (StoryWorker, Receiver<Action>, Receiver<PathBuf>) {
    let (out, told) = outbox();
    let (tx, drawn) = std::sync::mpsc::channel();
    let tx = Mutex::new(tx);
    let on_drawn: OnDrawn = Arc::new(move |path: &Path| {
        let _ = tx.lock().unwrap().send(path.to_owned());
    });
    let models = Arc::clone(fakes) as Arc<dyn Models>;
    (
        StoryWorker::start(models, places, out, on_drawn),
        told,
        drawn,
    )
}

/// A song no longer wanted while the thread draws it is given up there,
/// and the song still wanted drawn: the one given up never told drawn.
#[test]
fn the_worker_gives_up_a_song_no_longer_wanted() {
    let (_dir, places, paths) = setting("thread-given-up", 2);
    let fakes = Arc::new(Fakes::new());
    // Its first song's painter waits as it paints its second scene (the
    // first told).
    let portraits = good().characters.len();
    let (go, gate) = std::sync::mpsc::channel();
    *fakes.gate.lock().unwrap() = Some((portraits + 2, gate));
    let (worker, told, drawn) = started(&fakes, places);
    let (a, b) = (wanted(&paths[0]), wanted(&paths[1]));
    worker.want(vec![a, b.clone()]);
    loop {
        let news = told.recv_timeout(WAIT).expect("told");
        if let Action::Heard {
            path,
            part: Heard::Story(Story::Scene { .. }),
        } = news
        {
            assert_eq!(path, paths[0], "the first song drawn first");
            break;
        }
    }
    worker.want(vec![b]);
    go.send(()).unwrap();
    assert_eq!(drawn.recv_timeout(WAIT), Ok(paths[1].clone()));
    drop(worker);
    assert!(drawn.try_iter().all(|p| p != paths[0]), "given up");
}

/// A plan written again (its model set up anew, after its device was
/// lost) tells only the words past those told before, however its pieces
/// fall.
#[test]
fn a_plan_written_again_tells_only_its_words_past_those_told() {
    let mut words = Vec::new();
    let tell = &mut |story: Story| {
        if let Story::Words(w) = story {
            words.push(w);
        }
    };
    let told = &mut 0;
    let runs: [&[&str]; 2] = [&["Once ", "upon é"], &["Once up", "on é the", " end"]];
    for run in runs {
        let mut tell = past(told, tell);
        for piece in run {
            tell(Story::Words((*piece).into()));
        }
    }
    assert_eq!(words.concat(), "Once upon é the end");
}
