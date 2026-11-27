//! What goes into [`crate::core::update`] and what comes out.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_playback::{Command, Event};

use crate::browser::{Details, Entry};
use crate::core::keymap::Op;
use crate::core::{DeviceChoice, SettingsRow, Side};

/// Work the workers do on a file in the background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Job {
    /// Its beats, bars and stems analysed.
    Rhythm,
    /// Its lyrics looked up, or their words timed.
    Lyrics,
    /// Its karaoke story planned, its pictures drawn and its scenes taken
    /// apart ([`crate::work::story`]).
    Story,
}

impl Job {
    /// Every job.
    pub const ALL: [Job; 3] = [Job::Rhythm, Job::Lyrics, Job::Story];

    /// The job whose result this one is made from (the story from the
    /// lyrics): another result of it makes this one's to be made again.
    pub fn needs(self) -> Option<Job> {
        match self {
            Job::Story => Some(Job::Lyrics),
            Job::Rhythm | Job::Lyrics => None,
        }
    }
}

/// What a file's analysis found, as the queue's icons show it.
#[derive(Clone, Debug, PartialEq)]
pub struct RhythmFacts {
    /// Its tempo, when its beat is clear.
    pub bpm: Option<f64>,
    /// Its key on the Camelot wheel (`8A`), when found.
    pub key: Option<String>,
    /// Whether its stems were separated (and kept).
    pub stems: bool,
}

impl RhythmFacts {
    /// The facts of `analysis`.
    pub fn of(analysis: &jodel_rhythm::Analysis) -> RhythmFacts {
        RhythmFacts {
            bpm: analysis.summary.map(|s| s.bpm).filter(|b| *b > 0.0),
            key: analysis.key.as_ref().map(|k| k.camelot()),
            stems: analysis.stems,
        }
    }
}

/// How a file's lyrics were settled, as the queue's icons show it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LyricsFacts {
    /// Found, and their words timed by us.
    Timed,
    /// Found, their words not timed (no speech model, or it failed).
    Untimed,
    /// None found anywhere: heard from the voice by a speech recogniser
    /// (and timed), so they may be wrong.
    Heard,
    /// The track has no words.
    Instrumental,
    /// None found.
    NotFound,
    /// They could not be looked up (asked again later).
    Failed,
}

/// A file's lyrics as settled: how, as the queue's icons show it, and
/// which they are (a fingerprint of them as shown: settled again the same,
/// the same fingerprint), what the story is made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SettledLyrics {
    /// How.
    pub how: LyricsFacts,
    /// Which: [`SettledLyrics::print`] of them as shown, 0 with none.
    pub print: u64,
}

impl SettledLyrics {
    /// How `answer` settles them, and which they are.
    pub fn of(answer: &crate::work::lyrics::Answer) -> SettledLyrics {
        use crate::work::lyrics::Answer;
        let print = match answer {
            Answer::Found { lyrics, timed } => {
                SettledLyrics::print(timed.as_ref().unwrap_or(lyrics))
            }
            Answer::Instrumental | Answer::NotFound | Answer::Failed(_) => 0,
        };
        SettledLyrics {
            how: LyricsFacts::of(answer),
            print,
        }
    }

    /// The fingerprint of `lyrics`: a hash of them as Enhanced LRC (their
    /// words and every time).
    pub fn print(lyrics: &jodel_lyrics::Lyrics) -> u64 {
        jodel_dsp::fnv1a(lyrics.to_lrc().as_bytes())
    }
}

/// How a song's story stands once its worker told it ([`Facts::Story`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoryFact {
    /// Made: planned, drawn and its scenes taken apart.
    Made,
    /// None, for good: no lyrics or no timed sung lines, or a plan that
    /// did not hold (the game cannot be played over it).
    NoStory,
    /// Not made now, and why (a model that failed, a file that could not
    /// be read): tried again later.
    NotNow(String),
}

impl LyricsFacts {
    /// How `answer` settles them.
    pub fn of(answer: &crate::work::lyrics::Answer) -> LyricsFacts {
        use crate::work::lyrics::Answer;
        match answer {
            Answer::Found { lyrics, .. }
                if matches!(lyrics.source, jodel_lyrics::Source::Transcribed(_)) =>
            {
                LyricsFacts::Heard
            }
            Answer::Found { timed: Some(_), .. } => LyricsFacts::Timed,
            Answer::Found { .. } => LyricsFacts::Untimed,
            Answer::Instrumental => LyricsFacts::Instrumental,
            Answer::NotFound => LyricsFacts::NotFound,
            Answer::Failed(_) => LyricsFacts::Failed,
        }
    }
}

/// What a finished job's facts make of it ([`QueueFacts::take`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Took {
    /// Done.
    Done,
    /// Done, its result another than before: every job made from it is
    /// to be done again ([`Job::needs`]).
    Changed,
    /// Not done now: still to come, its why kept.
    StillToCome,
    /// Of what is no longer so: dropped, nothing taken in.
    Stale,
}

/// What a file's work found so far, as the queue's icons show it: each
/// `None` until that part is done, then whether the file has it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QueueFacts {
    /// Its key.
    pub key: Option<bool>,
    /// Its stems, separated and kept.
    pub stems: Option<bool>,
    /// Its tempo (a clear beat).
    pub tempo: Option<bool>,
    /// Its lyrics, once settled.
    pub lyrics: Option<SettledLyrics>,
    /// Its story, as its worker last told it.
    pub story: Option<StoryFact>,
}

impl QueueFacts {
    /// Takes in the whole analysis's facts.
    pub fn analysed(&mut self, f: &RhythmFacts) {
        self.key = Some(f.key.is_some());
        self.stems = Some(f.stems);
        self.tempo = Some(f.bpm.is_some());
    }

    /// Takes in what finished `facts` found, and what that makes of its
    /// job ([`Took`]): lyrics with another fingerprint than before changed
    /// (what the jobs made from them are made of: none before, not
    /// changed; settled otherwise with the same words, not changed
    /// either); a story not made now still to come; a story of other
    /// lyrics than those settled now (they changed while it was made)
    /// stale, not taken in.
    pub fn take(&mut self, facts: Facts) -> Took {
        match facts {
            Facts::Rhythm(f) => {
                self.analysed(&f);
                Took::Done
            }
            Facts::Lyrics(f) => match self.lyrics.replace(f) {
                Some(was) if was.print != f.print => Took::Changed,
                _ => Took::Done,
            },
            Facts::Story { lyrics, .. } if lyrics != self.lyrics_print() => Took::Stale,
            Facts::Story { fact, .. } => {
                let took = match fact {
                    StoryFact::NotNow(_) => Took::StillToCome,
                    StoryFact::Made | StoryFact::NoStory => Took::Done,
                };
                self.story = Some(fact);
                took
            }
        }
    }

    /// The fingerprint of the lyrics settled (0: none settled, or none
    /// found), what a story is made of.
    pub fn lyrics_print(&self) -> u64 {
        self.lyrics.map_or(0, |l| l.print)
    }

    /// What `job` found, forgotten: it is to be done again.
    pub fn forget(&mut self, job: Job) {
        match job {
            Job::Rhythm => (self.key, self.stems, self.tempo) = (None, None, None),
            Job::Lyrics => self.lyrics = None,
            Job::Story => self.story = None,
        }
    }
}

/// A part of a song's preparation told while it goes on, in the order
/// they come ([`Action::Heard`]): the key once the file is decoded, the
/// stems as they are separated and once they all are, the beat once
/// decided; the lyrics' steps and what they find; the song's story as it
/// is planned, written and drawn ([`crate::core::prep`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Heard {
    /// The key was looked for: the one found.
    Key(Option<jodel_rhythm::key::Key>),
    /// A chunk of the stems separated: the stems so far.
    Separating(Arc<crate::core::prep::Separating>),
    /// The stems are separated and kept.
    Stems,
    /// The beat decided.
    Beats(Arc<crate::core::prep::BeatsFound>),
    /// A step of the lyrics begun, or what it found.
    Lyrics(crate::core::prep::Words),
    /// The song's story, as it is planned, written and drawn.
    Story(crate::core::prep::Story),
}

/// What a finished job found ([`Action::Ready`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Facts {
    /// The analysis's.
    Rhythm(RhythmFacts),
    /// The lyrics'.
    Lyrics(SettledLyrics),
    /// The story's (not made now: the job still to come), and the
    /// fingerprint of the lyrics it was made of
    /// ([`SettledLyrics::print`]).
    Story {
        /// The lyrics' fingerprint.
        lyrics: u64,
        /// How it stands.
        fact: StoryFact,
    },
}

/// Every input to the UI core.
#[derive(Clone, Debug)]
pub enum Action {
    /// A named operation (from a key, a button or the palette).
    Op(Op),
    /// A character typed into the filter or the palette.
    Type(char),
    /// Text pasted: from the clipboard after [`Effect::ReadClipboard`]. It goes into the palette or the filter,
    /// and opens the filter on the main screen.
    Paste(String),

    /// The Ctrl key, as the frontend saw it: held in the browser, it
    /// previews the file selected ([`crate::core::preview`]).
    Ctrl(crate::core::preview::Ctrl),
    /// The key of a held row let go of (or the window left while it was
    /// held; [`crate::core::keymap::Binding::held`]): what it does ends,
    /// where it was going on (the demo stops singing at once,
    /// [`crate::core::singing::Demo`]; a key of the Synth tab's keyboard
    /// lets its note go).
    Released(crate::core::keymap::Op),
    /// Select a list row (mouse).
    SelectRow(usize),
    /// A click on a list row: selects it, and a second click soon after
    /// on the same row does what `enter` does on it.
    ClickRow(usize),
    /// Queue a list row (right click).
    EnqueueRow(usize),
    /// The queue was saved as a playlist there, or why not.
    QueueSaved(Result<PathBuf, String>),
    /// A playlist's tracks, or why they could not be read.
    PlaylistRead {
        /// The playlist.
        path: PathBuf,
        /// Its tracks.
        tracks: Result<Vec<PathBuf>, String>,
        /// Where they go.
        how: Placing,
    },
    /// The queue row being dragged is over this row: it moves there.
    QueueDrag(usize),
    /// The drag of a queue row ended.
    QueueDragEnd,
    /// Scroll the list by rows (mouse wheel).
    Scroll(i32),
    /// Show this side of the list pane.
    ShowSide(Side),
    /// Seek to a fraction of the track (a click on the progress bar, or the
    /// end of a drag).
    SeekTo(f64),
    /// The progress bar is being dragged to this fraction.
    SeekDrag(f64),
    /// Open the settings at this row (the device name in the signal path
    /// opens the device list).
    OpenSettings(SettingsRow),
    /// Select a settings row and step its value (`-1`, `0`: select only, or
    /// for the device row open the list, `1`).
    SettingsClick(usize, i32),
    /// Choose a device from the open list.
    ChooseDevice(usize),
    /// A row of the data folder's list clicked ([`crate::core::FolderPick::rows`]).
    FolderRow(usize),
    /// A folder's subfolders listed for the data folder's list (`None`:
    /// the drives).
    FoldersListed(Option<PathBuf>, Vec<PathBuf>),
    /// Run the palette's match at this index.
    PaletteRun(usize),
    /// Scroll the help screen.
    HelpScroll(i32),
    /// A point of a Transition block's lanes, `time` seconds into its track,
    /// was pressed (`extend`: dragged to, selecting bars). A second press on
    /// the same beat soon after is a double click: the playing track plays
    /// from there, the next track's cue goes there.
    TransitionPoint {
        /// The block.
        block: super::lanes::Block,
        /// Seconds into its track.
        time: f64,
        /// Dragged to rather than pressed.
        extend: bool,
    },
    /// A Transition block's overview was clicked: the playing track seeks
    /// there, the next one's view goes there.
    TransitionJump {
        /// The block.
        block: super::lanes::Block,
        /// Seconds into its track.
        time: f64,
    },
    /// The wheel over the Transition tab: zoom in (negative) or out; with
    /// `scroll` (over the whole track), the block's view moves by that
    /// many tenths instead.
    TransitionWheel {
        /// The block under the pointer.
        block: super::lanes::Block,
        /// Steps, negative up.
        steps: i32,
        /// Shift held: scroll rather than zoom.
        scroll: bool,
        /// Seconds into the block's track under the pointer (the lanes): a
        /// zoom keeps it where it is.
        at: Option<f64>,
    },
    /// The secondary button drags a view: a Transition block's or the
    /// lyrics editor's.
    Pan {
        /// The view.
        of: super::timeview::PanOf,
        /// Pressed, moved or let go of.
        input: super::timeview::PanInput,
    },
    /// A Karaoke slider was pressed or dragged to a share of its range.
    KaraokeSlide {
        /// The slider.
        slider: super::sliders::Slider,
        /// Where, 0 to 1.
        share: f64,
    },
    /// The drag of a Karaoke slider ended.
    KaraokeSlideEnd,
    /// The wheel turned over a Karaoke slider (no drag starts).
    KaraokeStep {
        /// The slider.
        slider: super::sliders::Slider,
        /// By how many of its steps ([`super::sliders::Slider::stepped`]):
        /// up while positive.
        by: i32,
    },
    /// What the app tells of the sync under way ([`crate::core::sync`]).
    Sync(crate::core::sync::SyncEvent),
    /// The mouse on the Mixer tab's desk.
    Desk(crate::core::desk::DeskAction),
    /// The pointer and the window on the Synth tab.
    Synth(crate::core::synth_tab::SynthAction),
    /// The pointer and the window on the Shader tab.
    Shader(crate::core::shader_tab::ShaderAction),
    /// The pointer moved (Karaoke mode shows its pointer, and its controls
    /// while it is near the bottom).
    PointerMoved {
        /// The share of the window's height below the pointer: 0 at the
        /// bottom, 1 at the top.
        from_bottom: f32,
    },
    /// The pointer left the window (Karaoke mode's controls go in a while).
    PointerLeft,

    /// What the window's size does to the keys and the pointer, measured
    /// again (the window reports it whenever it changes).
    Fit(super::Fit),
    /// The clock: milliseconds since the start.
    Tick(u64),

    /// A folder was listed (or failed to be).
    Listed {
        /// The folder.
        dir: PathBuf,
        /// Its rows, or the error.
        result: Result<Vec<Entry>, String>,
    },
    /// A file's details were read.
    Details(PathBuf, Details),
    /// These files were not read after all (no longer on screen).
    DetailsDropped(Vec<PathBuf>),
    /// Search results.
    SearchHits {
        /// The search.
        id: u64,
        /// What it found since the last batch.
        hits: Vec<Entry>,
    },
    /// A search finished.
    SearchDone {
        /// The search.
        id: u64,
    },
    /// A folder walk for queueing finished.
    Walked {
        /// What the walk was for.
        purpose: WalkPurpose,
        /// The playable files found, in order.
        paths: Vec<PathBuf>,
    },
    /// What was typed after `/` is not an existing folder, file or drive:
    /// it is searched for below the folder instead.
    JumpMissed {
        /// What was typed.
        text: String,
    },
    /// What was typed after `/` is an existing folder or file: show this
    /// folder, with this file selected.
    JumpTo {
        /// The folder.
        dir: PathBuf,
        /// The file to select in it.
        select: Option<PathBuf>,
    },
    /// The output devices were listed.
    Devices(Result<Vec<DeviceChoice>, String>),
    /// The playback engine reported something.
    Engine(Event),
    /// A file's beats were analysed by a recipe (or could not be).
    Rhythm(
        PathBuf,
        super::RhythmRecipe,
        Result<Arc<super::Rhythmed>, String>,
    ),
    /// The GPU for stems, by name, or why there is none.
    Gpu(Result<String, String>),
    /// What was found of a file's lyrics.
    Lyrics(PathBuf, crate::work::lyrics::Answer),
    /// The lyrics editor: opened, closed, an action of its, or the song's
    /// edits forgotten ([`crate::core::lyric_edit`]).
    LyricsEdit(crate::core::lyric_edit::LyricsEdit),
    /// A worker started (`on`) or ended work on a file: its analysis, or
    /// its lyrics looked up or timed.
    Working {
        /// The file.
        path: PathBuf,
        /// Which work.
        job: Job,
        /// Started, else ended.
        on: bool,
    },
    /// What a job makes of a file is in the cache (or nothing more can be
    /// made of it): nothing is left to do for it.
    Ready {
        /// The file.
        path: PathBuf,
        /// Which work.
        job: Job,
        /// What it found, for the queue's icons (none when it failed).
        facts: Option<Facts>,
    },
    /// A part of a song's preparation is done or begun, the rest still
    /// going on (the queue's icons light up one after the other, its
    /// captions and the preparation view follow it).
    Heard {
        /// The file.
        path: PathBuf,
        /// Which part.
        part: Heard,
    },
    /// A failure outside the engine (a config file that could not be
    /// written, a device that could not be opened).
    Failed(String),
    /// Something to tell from outside the core (a recording done).
    Info(String),
    /// The playback engine of this start (the app's count of engine
    /// starts) failed and is gone; the app starts another, with the queue.
    EngineGone(u64),
}

/// What a folder walk is for: its files put in the queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkPurpose {
    /// The folder's name, for the message.
    pub name: String,
    /// Where they go.
    pub how: Placing,
}

/// Where tracks chosen in the browser go (`enter`, `shift+enter`,
/// `alt+enter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placing {
    /// After the queue.
    Append,
    /// After the queue, and the first of them plays now (the queue kept).
    AppendPlay,
    /// In place of the queue, playing.
    Replace,
}

/// Work for the app to carry out after an update.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// Send a command to the playback engine.
    Engine(Command),
    /// Tell the preview's players ([`crate::core::preview`]).
    Preview(crate::core::preview::Preview),
    /// List a folder (answered with [`Action::Listed`]).
    ListDir(PathBuf),
    /// List a folder's subfolders for the data folder's list, or the
    /// drives (`None`) (answered with [`Action::FoldersListed`]).
    ListFolders(Option<PathBuf>),
    /// Move the data folder into this folder (the settings save it once the
    /// move is written down).
    MoveData(PathBuf),
    /// Go on with a move of the data folder that stopped (a full disk).
    RetryMove,
    /// Read the details of exactly these files now (answered with
    /// [`Action::Details`] and [`Action::DetailsDropped`]).
    WantDetails(Vec<PathBuf>),
    /// Start a recursive search (answered with [`Action::SearchHits`] and
    /// [`Action::SearchDone`]); a running one is cancelled.
    Search {
        /// Matches results to the search.
        id: u64,
        /// Where to search.
        root: PathBuf,
        /// The text to search for.
        query: String,
    },
    /// Cancel the running search.
    CancelSearch,
    /// Check whether what was typed after `/` names an existing folder,
    /// file or drive, relative to `from` unless absolute (answered with
    /// [`Action::JumpTo`] if it does, else [`Action::JumpMissed`]).
    Jump {
        /// What was typed.
        text: String,
        /// The folder it was typed in.
        from: PathBuf,
    },
    /// Collect every playable file below a folder (answered with
    /// [`Action::Walked`]).
    Walk {
        /// The folder.
        root: PathBuf,
        /// What for.
        purpose: WalkPurpose,
    },
    /// Analyse the beats of exactly these files, in this order (answered
    /// with [`Action::Rhythm`]); what was asked for before and not started
    /// is dropped.
    WantRhythm(Vec<PathBuf>, super::RhythmRecipe),
    /// Look up the lyrics of exactly these files, in this order (answered
    /// with [`Action::Lyrics`]); what was asked for before and not started
    /// is dropped.
    WantLyrics(Vec<PathBuf>),
    /// Get the stories of exactly these songs made, in this order, from
    /// their lyrics as settled (told with [`Heard::Story`], working and
    /// ready as [`Job::Story`]); the songs no longer asked for are given
    /// up.
    WantStories(Vec<crate::core::StorySong>),
    /// Keep a song's lyrics as edited by hand (`None`: forget its edits),
    /// and time it again with them (answered with [`Action::Lyrics`] once
    /// timed).
    SaveLyricEdits(PathBuf, Option<jodel_lyrics::Lyrics>),
    /// Give the mixer the voice's curve over the song's words, or end it
    /// ([`crate::core::voice_curve`]).
    VoiceCurve(crate::core::voice_curve::VoiceCurve),
    /// Get these files ready ahead, in this order, after what is wanted:
    /// their analysis by the recipe (`None`: none) and their lyrics, into
    /// the caches; what was asked for before and not started is dropped.
    Prepare(Vec<PathBuf>, Option<super::RhythmRecipe>),
    /// Karaoke: use the next microphone (input device), and keep it in the
    /// settings.
    NextMic,
    /// Karaoke: record what the echo canceller works with, for tuning it.
    RecordEcho,
    /// Karaoke: play the sync's clicks and listen, or a press, or stop
    /// ([`crate::core::sync`]).
    Sync(crate::core::sync::SyncCommand),
    /// Set a strip of the mixer as the Mixer tab has it.
    Desk(crate::core::desk::DeskChange),
    /// Play what the Synth tab asks on the effects input.
    Bench(crate::core::synth_tab::Bench),
    /// Keep a theme's atom's patch as edited on the Synth tab, in the data
    /// folder (`None`: forget what was kept, the built-in one again).
    KeepPatch {
        /// The theme.
        theme: &'static crate::theme::Theme,
        /// The atom.
        atom: crate::sfx::conductor::Atom,
        /// The patch.
        patch: Option<jodel_synth::Patch>,
    },
    /// Keep a look as a preset of the user's own, in the data folder (the
    /// Shader tab's).
    KeepLook(jodel_looks::Look),
    /// Put this text on the clipboard.
    CopyText(String),
    /// List the output devices (answered with [`Action::Devices`]).
    ListDevices,
    /// Play on this device from now on (`None`: the system default),
    /// carrying the queue and the position over.
    SwitchDevice(Option<String>),
    /// Read the clipboard's text (answered with [`Action::Paste`], or a
    /// message when it holds none).
    ReadClipboard,
    /// Keep a beat grid fixed by hand for a file (in the grid store).
    SaveGrid(PathBuf, super::grid::Grid),
    /// Forget a file's grid fixed by hand.
    ResetGrid(PathBuf),
    /// Write the settings to the config file.
    SaveConfig,
    /// Save these tracks as a playlist among the saved queues (answered
    /// with [`Action::QueueSaved`]).
    SaveQueue(Vec<PathBuf>),
    /// Read a playlist's tracks (answered with [`Action::PlaylistRead`]).
    ReadPlaylist {
        /// The playlist.
        path: PathBuf,
        /// Where its tracks go.
        how: Placing,
    },
    /// Leave.
    Quit,
}
