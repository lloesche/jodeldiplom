//! Stories on stand-in models (no GPU): a round looking at every song,
//! then planning, then encoding, then drawing, then taking the scenes
//! apart, the models asked only for work; each song told working in its
//! turns and ready once finished; what is kept and drawn again (a song of
//! a real length drawn once); the scenes described with their times and
//! the look; a song no longer wanted given up between its pictures; no
//! models, or no timed lines, no story; the thread taking up the songs
//! wanted, in their order, their lyrics settled again too. The models'
//! turns among threads are the stages' (`crate::work::stages`); the whole
//! of it on the real models, from the app, is `tests/story.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use jodel_gpu::image::{Embeds, Picture, Reference};
use jodel_lyrics::{Line as Sung, Source, Word};
use jodel_testkit::TempDir;

use super::*;
use crate::core::prep::Picture as Begun;
use crate::core::story::pictures::{PORTRAIT, SCENE};
use crate::core::story::tests::{song, GOOD};
use crate::work::scenes::ByRed;
use crate::work::story::draw::{Encoder, Painter, REFERENCE};
use crate::work::story::{Writer, Writing, Written};

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
/// waits after its picture of that count until let go), the scenes are
/// taken apart by their red; each logs what it does; none runs where
/// `none`, the encoder fails where `no_encoder`.
struct Fakes {
    log: Log,
    answer: Mutex<String>,
    painted: AtomicUsize,
    gate: Mutex<Option<(usize, Receiver<()>)>>,
    none: bool,
    no_encoder: bool,
    cut: bool,
}

impl Fakes {
    fn new() -> Fakes {
        Fakes {
            log: Log::default(),
            answer: Mutex::new(GOOD.to_owned()),
            painted: AtomicUsize::new(0),
            gate: Mutex::new(None),
            none: false,
            no_encoder: false,
            cut: false,
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
    fn write(&self, _: &str, _: usize, seen: &mut dyn FnMut(Writing)) -> Result<Written, Failure> {
        self.0.did("plan".into());
        seen(Writing::Answer(&self.1));
        Ok(Written {
            text: self.1.clone(),
            stopped: !self.0.cut,
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
        if self.no_encoder {
            self.did("no encoder".into());
            return Err("out of memory".into());
        }
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

    fn take_apart(
        &self,
        scenes: &Scenes,
        store: &SceneStore,
        still: &dyn Fn() -> bool,
    ) -> Result<TakenApart, String> {
        self.up()?;
        self.did("cut".into());
        scenes::take_apart(scenes, store, &ByRed::default(), still)
    }
}

/// The lyrics of [`song`], every word of a line between its start and
/// its end; `timed` false: none timed at all; `first` the first word.
fn lyrics(timed: bool, first: &str) -> Lyrics {
    let mut lines: Vec<Sung> = (song().lines.iter())
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
    lines[0].words[0].text = first.into();
    Lyrics {
        lines,
        source: Source::Lrclib,
        timing: Default::default(),
    }
}

/// The songs' lyrics as the stand-in lyrics worker settled them: every
/// song's the same, changed by the test.
type Settled = Arc<Mutex<Result<Option<Lyrics>, String>>>;

/// A data folder's places in a new folder for `tag`, and `songs` songs
/// in it ([`frames`] of silence), their lyrics timed; what settles their
/// lyrics.
fn setting(tag: &str, songs: usize) -> (TempDir, Places, Vec<PathBuf>, Settled) {
    let silence = vec![0i16; frames()];
    let dir = TempDir::new(&format!("stories-{tag}"));
    let settled: Settled = Arc::new(Mutex::new(Ok(Some(lyrics(true, "I")))));
    let read = Arc::clone(&settled);
    let places = Places {
        stories: dir.join("stories"),
        scenes: SceneStore::new(dir.join("scenes"), dir.join("scene-parts")),
        lyrics: Arc::new(move |_| read.lock().unwrap().clone()),
    };
    let paths = (0..songs)
        .map(|k| {
            let path = dir.join(format!("{k:02}.wav"));
            durable::write_atomic(&path, &jodel_dsp::wav::pcm16(RATE, 1, &silence)).unwrap();
            path
        })
        .collect();
    (dir, places, paths, settled)
}

/// The song at `path` with its lyrics settled the first time.
fn wanted(path: &Path) -> StorySong {
    StorySong {
        path: path.to_owned(),
        lyrics: 1,
    }
}

/// An outbox, and what it is sent.
fn outbox() -> (Outbox, Receiver<Action>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (Outbox::new(tx, Arc::new(|| {})), rx)
}

/// What `rx` was told, of each song: each scene drawn ("N/M"), a scene
/// begun ("drawing scene N of M"), the steps begun, working ("on",
/// "off") and ready ("story", "none", or "not now: why"); none of its
/// words.
fn told(rx: &Receiver<Action>) -> Vec<(PathBuf, String)> {
    rx.try_iter()
        .filter_map(|action| match action {
            Action::Heard {
                path,
                part: Heard::Story(story),
            } => {
                let said = match story {
                    Story::Words(_) | Story::Drawing(Begun::Portrait(..)) => return None,
                    Story::Scene { index, of, .. } => format!("{index}/{of}"),
                    Story::Drawing(Begun::Scene(n, of)) => format!("drawing scene {n} of {of}"),
                    Story::Planning => "planning".into(),
                    Story::Reading => "reading".into(),
                    Story::Thinking => "thinking".into(),
                    Story::Cutting => "cutting".into(),
                };
                Some((path, said))
            }
            Action::Working { path, job, on } => {
                assert_eq!(job, Job::Story);
                Some((path, if on { "on" } else { "off" }.into()))
            }
            Action::Ready { path, job, facts } => {
                assert_eq!(job, Job::Story);
                let said = match facts {
                    Some(Facts::Story {
                        fact: StoryFact::Made,
                        ..
                    }) => "story".into(),
                    Some(Facts::Story {
                        fact: StoryFact::NoStory,
                        ..
                    }) => "none".into(),
                    Some(Facts::Story {
                        fact: StoryFact::NotNow(why),
                        ..
                    }) => format!("not now: {why}"),
                    other => panic!("{other:?}"),
                };
                Some((path, said))
            }
            _ => None,
        })
        .collect()
}

/// What `news` says of the song at `path`.
fn of(news: &[(PathBuf, String)], path: &Path) -> Vec<String> {
    (news.iter())
        .filter(|(p, _)| p == path)
        .map(|(_, n)| n.clone())
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
    places.scenes.folder(&crate::work::scenes::song_key(path))
}

/// One round of `songs`, each wanted while `still` says, with `tried`
/// from the rounds before; the songs told finished to the song playing's
/// story, and what comes next.
fn next_round(
    songs: &[StorySong],
    (models, places, out): (&Fakes, &Places, &Outbox),
    still: &dyn Fn(&StorySong) -> bool,
    tried: &mut Tried,
) -> (Vec<PathBuf>, Next) {
    let drawn = Mutex::new(Vec::new());
    let on_drawn = |path: &Path| drawn.lock().unwrap().push(path.to_owned());
    let ask = Ask {
        models,
        places,
        still,
        outbox: out,
        on_drawn: &on_drawn,
    };
    let next = round(songs, &ask, tried);
    (drawn.into_inner().unwrap(), next)
}

/// [`next_round`]'s songs told finished.
fn round_with(
    songs: &[StorySong],
    with: (&Fakes, &Places, &Outbox),
    still: &dyn Fn(&StorySong) -> bool,
    tried: &mut Tried,
) -> Vec<PathBuf> {
    next_round(songs, with, still, tried).0
}

/// One round of `songs`, each always wanted, as a thread's first.
fn round_of(songs: &[StorySong], models: &Fakes, places: &Places, out: &Outbox) -> Vec<PathBuf> {
    round_with(
        songs,
        (models, places, out),
        &|_| true,
        &mut Tried::default(),
    )
}

/// A round plans every song wanted, then encodes their prompts, then
/// draws their pictures (a song's portraits first, then its scenes, each
/// with the portraits of those in it), then takes their scenes apart: each
/// song told working in each of its turns with the step begun, each scene
/// begun and told as it lands, and told ready once its scenes are taken
/// apart. Its folder then holds a picture a scene, with the plan's times
/// and look, and its scenes' parts are kept.
#[test]
fn a_round_plans_every_song_then_encodes_then_draws_then_takes_apart() {
    let (_dir, places, paths, _) = setting("order", 2);
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
    want.extend(vec!["cut".to_owned(); paths.len()]);
    assert_eq!(fakes.took(), want);
    let news = told(&rx);
    let n = plan.scenes.len();
    let mut drawing = Vec::new();
    for k in 0..n {
        drawing.push(format!("drawing scene {} of {n}", k + 1));
        drawing.push(format!("{k}/{n}"));
    }
    for path in &paths {
        let mut want: Vec<String> = ["on", "planning", "off", "on", "reading", "off", "on"]
            .map(String::from)
            .into();
        want.extend(drawing.iter().cloned());
        want.extend(["off", "on", "cutting", "off", "story"].map(String::from));
        assert_eq!(of(&news, path), want);
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
        assert_eq!(scenes::taken_apart(&kept, &places.scenes), Ok(true));
    }
}

/// `t` as a description keeps it, to the millisecond.
fn millis(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

/// A song made is not made again (whatever its length): its plan kept,
/// its pictures there, its description made as its plan makes it, its
/// scenes taken apart; it is told ready at once, no model asked, and a
/// thread's next round does not look at it again. A picture gone is drawn
/// again alone (its scene's portraits read back from the folder) and
/// taken apart; a plan made again draws again only the pictures it
/// changed, and lets the old ones go.
#[test]
fn a_song_made_is_not_made_again() {
    let (_dir, places, paths, settled) = setting("kept", 1);
    let fakes = Fakes::new();
    let (out, rx) = outbox();
    let song = [wanted(&paths[0])];
    let mut tried = Tried::default();
    let always = |_: &StorySong| true;
    let ask = (&fakes, &places, &out);
    round_with(&song, ask, &always, &mut tried);
    fakes.took();
    told(&rx);
    assert!(round_of(&song, &fakes, &places, &out).is_empty());
    assert_eq!(fakes.took(), Vec::<String>::new());
    assert_eq!(of(&told(&rx), &paths[0]), ["story"], "ready at once");
    assert!(round_with(&song, ask, &always, &mut tried).is_empty());
    assert!(rx.try_iter().next().is_none(), "not looked at again");
    // The second scene's picture gone: it alone, with its two portraits.
    let folder = folder(&places, &paths[0]);
    let second = Scenes::read(&folder).unwrap().scenes[1].picture.clone();
    std::fs::remove_file(&second).unwrap();
    assert_eq!(round_of(&song, &fakes, &places, &out), paths);
    let cut = "cut".to_owned();
    assert_eq!(
        fakes.took(),
        ["encode".to_owned(), paint(SCENE, 2), cut.clone()]
    );
    assert_eq!(told(&rx).iter().filter(|(_, n)| n.contains('/')).count(), 1);
    assert!(second.is_file());
    // New lyrics, a new plan: one scene's prompt is another, so that
    // picture alone is drawn again, the old one let go.
    let before = std::fs::read_dir(&folder).unwrap().count();
    *fakes.answer.lock().unwrap() = GOOD.replace("Lonely lighthouse", "Tall lighthouse");
    *settled.lock().unwrap() = Ok(Some(lyrics(true, "We")));
    let again = StorySong {
        lyrics: 2,
        ..song[0].clone()
    };
    round_with(&[again], ask, &always, &mut tried);
    let want = ["plan".to_owned(), "encode".into(), paint(SCENE, 0), cut];
    assert_eq!(fakes.took(), want);
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), before);
    let kept = Scenes::read(&folder).unwrap();
    assert!(kept.scenes.iter().all(|s| s.picture.is_file()));
}

/// A song no longer wanted is given up between its pictures: those drawn
/// kept, no description, not told ready; the next round draws only the
/// rest.
#[test]
fn a_song_no_longer_wanted_is_given_up_between_its_pictures() {
    let (_dir, places, paths, _) = setting("given-up", 1);
    let fakes = Fakes::new();
    let (out, rx) = outbox();
    let song = [wanted(&paths[0])];
    // Wanted until its third picture is painted.
    const DRAWN: usize = 3;
    let still = |_: &StorySong| fakes.painted() < DRAWN;
    let drawn = round_with(
        &song,
        (&fakes, &places, &out),
        &still,
        &mut Tried::default(),
    );
    assert_eq!(fakes.painted(), DRAWN);
    assert!(drawn.is_empty());
    assert!(!of(&told(&rx), &paths[0]).iter().any(|n| n == "story"));
    let folder = folder(&places, &paths[0]);
    assert!(Scenes::read(&folder).is_none());
    assert_eq!(std::fs::read_dir(&folder).unwrap().count(), DRAWN);
    assert_eq!(round_of(&song, &fakes, &places, &out), paths);
    let plan = good();
    let all = plan.characters.len() + plan.scenes.len();
    assert_eq!(fakes.painted(), all);
}

/// No models to run (no GPU): nothing kept, the song told ready as not
/// made now and why, and the round asks to be tried again after a pause;
/// its lyrics not read now, the same, no model asked. A song with no timed
/// lines, or no lyrics, asks no model at all and is told ready with none
/// for good, nothing to try again.
#[test]
fn without_models_or_timed_lines_there_is_no_story() {
    let (dir, places, paths, settled) = setting("none", 1);
    let (out, rx) = outbox();
    let fakes = Fakes {
        none: true,
        ..Fakes::new()
    };
    let song = [wanted(&paths[0])];
    let with = (&fakes, &places, &out);
    let always = |_: &StorySong| true;
    let (drawn, next) = next_round(&song, with, &always, &mut Tried::default());
    assert!(drawn.is_empty() && matches!(next, Next::At(_)));
    assert!(!dir.join("stories").exists() && !dir.join("scenes").exists());
    let news = of(&told(&rx), &paths[0]);
    assert_eq!(news, ["on", "planning", "off", "not now: no GPU"]);
    let fakes = Fakes::new();
    let with = (&fakes, &places, &out);
    *settled.lock().unwrap() = Err("the network drive is asleep".into());
    let (_, next) = next_round(&song, with, &always, &mut Tried::default());
    assert!(matches!(next, Next::At(_)));
    let news = of(&told(&rx), &paths[0]);
    assert_eq!(news, ["not now: the network drive is asleep"]);
    for lyrics in [Some(lyrics(false, "I")), None] {
        *settled.lock().unwrap() = Ok(lyrics);
        let (drawn, next) = next_round(&song, with, &always, &mut Tried::default());
        assert!(drawn.is_empty() && next == Next::Wait);
        assert_eq!(fakes.took(), Vec::<String>::new());
        assert_eq!(of(&told(&rx), &paths[0]), ["none"]);
    }
    assert!(!dir.join("stories").exists() && !dir.join("scenes").exists());
}

/// A round plans only as many songs as their prompts would fill
/// [`ENCODED_AHEAD_MB`] with, then draws them, and leaves the rest for the
/// next round (at once): the songs wanted first are made first.
#[test]
fn a_round_plans_no_more_songs_than_it_can_draw() {
    let pictures = good().characters.len() + good().scenes.len();
    let per_song = pictures * PROMPT_BYTES;
    let songs = AHEAD.div_ceil(per_song) + 1;
    let (_dir, places, paths, _) = setting("bounded", songs);
    let fakes = Fakes::new();
    let (out, _rx) = outbox();
    let wanted_all: Vec<StorySong> = paths.iter().map(|p| wanted(p)).collect();
    let mut tried = Tried::default();
    let always = |_: &StorySong| true;
    let with = (&fakes, &places, &out);
    let (drawn, next) = next_round(&wanted_all, with, &always, &mut tried);
    assert_eq!(next, Next::Round);
    assert_eq!(drawn, paths[..songs - 1], "the first, in order");
    let plans = |log: Vec<String>| log.iter().filter(|e| *e == "plan").count();
    assert_eq!(plans(fakes.took()), songs - 1);
    let (drawn, next) = next_round(&wanted_all, with, &always, &mut tried);
    assert_eq!((drawn, next), (paths[songs - 1..].to_vec(), Next::Wait));
    assert_eq!(plans(fakes.took()), 1);
}

/// A song left without its story now (its encoder failing) is left out
/// of the rounds until it is to be tried again: it neither runs again at
/// once nor fills the round's room, so the songs after it are planned in
/// the next round; once every song waits so, the round asks for none at
/// once, but for one when the first is to be tried again.
#[test]
fn a_song_not_made_now_rests_and_holds_no_other_off() {
    let pictures = good().characters.len() + good().scenes.len();
    let songs = AHEAD.div_ceil(pictures * PROMPT_BYTES) + 1;
    let (_dir, places, paths, _) = setting("resting", songs);
    let fakes = Fakes {
        no_encoder: true,
        ..Fakes::new()
    };
    let (out, _rx) = outbox();
    let wanted_all: Vec<StorySong> = paths.iter().map(|p| wanted(p)).collect();
    let mut tried = Tried::default();
    let always = |_: &StorySong| true;
    let with = (&fakes, &places, &out);
    let count = |log: &[String], what: &str| log.iter().filter(|e| *e == what).count();
    let (_, next) = next_round(&wanted_all, with, &always, &mut tried);
    assert_eq!(next, Next::Round, "songs left for the next");
    let log = fakes.took();
    assert_eq!(count(&log, "plan"), songs - 1);
    assert_eq!(count(&log, "no encoder"), songs - 1);
    // The songs that failed rest: the last is planned, and fails too.
    let (_, next) = next_round(&wanted_all, with, &always, &mut tried);
    let log = fakes.took();
    assert_eq!((count(&log, "plan"), count(&log, "no encoder")), (1, 1));
    let Next::At(at) = next else {
        panic!("tried again later, not at once: {next:?}");
    };
    assert!(at > Instant::now() + RETRY / 2);
    // All resting: no model asked.
    let (_, again) = next_round(&wanted_all, with, &always, &mut tried);
    assert_eq!(fakes.took(), Vec::<String>::new());
    assert_eq!(again, Next::At(at));
}

/// A song left without its story now is left out for [`RETRY`], twice as
/// long each time again in a row for the same lyrics, at most
/// [`MOST_RETRY`]; other lyrics, or the song finished, start afresh.
#[test]
fn a_song_not_made_now_waits_longer_each_time_again() {
    let mut tried = Tried::default();
    let song = wanted(Path::new("a.flac"));
    let now = Instant::now();
    assert_eq!(tried.fail(&song, now), RETRY);
    assert!(tried.rests(&song, now) && !tried.rests(&song, now + RETRY));
    assert_eq!(tried.fail(&song, now), RETRY * 2);
    let waits: Vec<Duration> = (0..6).map(|_| tried.fail(&song, now)).collect();
    assert_eq!(waits.last(), Some(&MOST_RETRY));
    assert!(waits.windows(2).all(|w| w[0] <= w[1]));
    let other = StorySong {
        lyrics: 2,
        ..song.clone()
    };
    assert!(!tried.rests(&other, now), "other lyrics are looked at");
    assert_eq!(tried.fail(&other, now), RETRY);
    tried.finish(&other);
    assert!(tried.rests(&other, now + MOST_RETRY), "finished for them");
    assert_eq!(tried.due([&song].into_iter()), None);
}

/// An answer cut off at its length is the song's for good (written
/// greedily, it would be cut off again): told ready with no story once,
/// kept so, and never planned again for the same ask, the round asking
/// for no retry; a round after (a run after) finds it kept.
#[test]
fn a_plan_cut_off_is_refused_once_and_not_tried_again() {
    let (_dir, places, paths, _) = setting("cut-off", 1);
    let fakes = Fakes {
        cut: true,
        answer: Mutex::new(GOOD[..GOOD.len() / 2].to_owned()),
        ..Fakes::new()
    };
    let (out, rx) = outbox();
    let song = [wanted(&paths[0])];
    let with = (&fakes, &places, &out);
    let always = |_: &StorySong| true;
    let (drawn, next) = next_round(&song, with, &always, &mut Tried::default());
    assert!(drawn.is_empty());
    assert_eq!(next, Next::Wait, "not tried again");
    assert_eq!(fakes.took(), ["plan"], "asked once, not twice");
    assert_eq!(of(&told(&rx), &paths[0]), ["on", "planning", "off", "none"]);
    let (_, next) = next_round(&song, with, &always, &mut Tried::default());
    assert_eq!(next, Next::Wait);
    assert_eq!(
        fakes.took(),
        Vec::<String>::new(),
        "kept: not planned again"
    );
    assert_eq!(of(&told(&rx), &paths[0]), ["none"]);
}

/// A song is still wanted while it is in the list of the very same
/// lyrics: not of others (settled again), nor out of it.
#[test]
fn a_song_is_still_wanted_with_its_very_lyrics() {
    let (a, b) = (wanted(Path::new("a.flac")), wanted(Path::new("b.flac")));
    let asked = Asked {
        songs: vec![a.clone()],
        times: 1,
    };
    assert!(asked.still(&a));
    let again = StorySong { lyrics: 2, ..a };
    assert!(!asked.still(&again), "other lyrics");
    assert!(!asked.still(&b));
}

/// The thread takes up the songs wanted in their order (the first planned
/// first), each made once; asked again with a song's lyrics settled anew,
/// it plans that song again (and draws what its new plan changed), the
/// other not looked at again.
#[test]
fn the_worker_takes_up_the_songs_wanted_in_order_and_their_new_lyrics() {
    let (_dir, places, paths, settled) = setting("thread", 2);
    let fakes = Arc::new(Fakes::new());
    let (worker, told_rx, drawn) = started(&fakes, places);
    let wait = || drawn.recv_timeout(WAIT).expect("drawn");
    let songs: Vec<StorySong> = paths.iter().rev().map(|p| wanted(p)).collect();
    worker.want(songs.clone());
    assert_eq!([wait(), wait()], [paths[1].clone(), paths[0].clone()]);
    let plans = |log: Vec<String>| log.iter().filter(|e| *e == "plan").count();
    assert_eq!(plans(fakes.took()), 2);
    let planning: Vec<PathBuf> = (told(&told_rx).into_iter())
        .filter(|(_, n)| n == "planning")
        .map(|(p, _)| p)
        .collect();
    assert_eq!(planning, [paths[1].clone(), paths[0].clone()], "in order");
    *settled.lock().unwrap() = Ok(Some(lyrics(true, "We")));
    *fakes.answer.lock().unwrap() = GOOD.replace("Lonely lighthouse", "Tall lighthouse");
    let again = StorySong {
        lyrics: 2,
        ..songs[1].clone()
    };
    worker.want(vec![songs[0].clone(), again]);
    assert_eq!(wait(), paths[0]);
    assert_eq!(plans(fakes.took()), 1);
    drop(worker);
}

/// How long a test waits for the thread.
const WAIT: Duration = Duration::from_secs(60);

/// The thread on `fakes`, its stories kept in `places`; what it tells on
/// its outbox, and the songs it tells finished.
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
/// and the song still wanted made: the one given up never told finished.
#[test]
fn the_worker_gives_up_a_song_no_longer_wanted() {
    let (_dir, places, paths, _) = setting("thread-given-up", 2);
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

/// A song left without its story now rests, and the thread waits to try
/// it again; a new ask meanwhile is taken up at once: the other song
/// wanted is made well within the wait.
#[test]
fn the_worker_takes_up_a_new_ask_while_a_song_rests() {
    let (_dir, places, paths, _) = setting("thread-resting", 2);
    let asleep = paths[0].clone();
    let places = Places {
        lyrics: Arc::new(move |path: &Path| {
            if path == asleep {
                Err("the network drive is asleep".into())
            } else {
                Ok(Some(lyrics(true, "I")))
            }
        }),
        ..places
    };
    let fakes = Arc::new(Fakes::new());
    let (worker, told, drawn) = started(&fakes, places);
    let (a, b) = (wanted(&paths[0]), wanted(&paths[1]));
    worker.want(vec![a.clone()]);
    loop {
        if let Action::Ready {
            facts:
                Some(Facts::Story {
                    fact: StoryFact::NotNow(_),
                    ..
                }),
            ..
        } = told.recv_timeout(WAIT).expect("told")
        {
            break;
        }
    }
    let asked = Instant::now();
    worker.want(vec![a, b]);
    assert_eq!(drawn.recv_timeout(RETRY / 2), Ok(paths[1].clone()));
    assert!(asked.elapsed() < RETRY / 2, "taken up at once");
    drop(worker);
}

/// A plan written again (its model set up anew, after its device was
/// lost) tells only the words past those told before, however its pieces
/// fall, and not its thought again.
#[test]
fn a_plan_written_again_tells_only_its_words_past_those_told() {
    let (mut words, mut thoughts) = (Vec::new(), 0);
    let tell = &mut |story: Story| match story {
        Story::Words(w) => words.push(w),
        Story::Thinking => thoughts += 1,
        _ => {}
    };
    let told = &mut 0;
    let runs: [&[&str]; 2] = [&["Once ", "upon é"], &["Once up", "on é the", " end"]];
    for run in runs {
        let mut tell = past(told, tell);
        tell(Story::Thinking);
        for piece in run {
            tell(Story::Words((*piece).into()));
        }
    }
    assert_eq!(words.concat(), "Once upon é the end");
    assert_eq!(thoughts, 1, "a run again tells no second thought");
}
