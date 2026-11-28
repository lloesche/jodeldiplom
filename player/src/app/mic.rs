//! The microphone and the singing heard through it, for the
//! [`App`](super::App): the microphone opened in the background while
//! Karaoke shows and closed after, the next one chosen, its reading for
//! each frame, the singer's pitch held against the song's notes, the
//! singing kept ([`crate::store::keep`]), a recording for the echo canceller,
//! and the workers the game's singing needs (the original singer's phrases,
//! the song's loudness).
//!
//! The microphone is only listened to: nothing of it is played.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use jodel_output::input::Input;
use jodel_output::mixer::Mixer;

use crate::core::{self, Action, State};
use crate::work::outbox::Notify;

/// Opens a microphone: the input with this id, or the system's default.
pub type MicOpener = Arc<dyn Fn(Option<&str>) -> Result<Input, String> + Send + Sync>;

/// Lists the microphones (input devices) as (id, name).
pub type MicLister = Arc<dyn Fn() -> Result<Vec<(String, String)>, String> + Send + Sync>;

/// The real microphone with this id, or the system's default input.
pub fn open_mic(id: Option<&str>) -> Result<Input, String> {
    let id = match id {
        Some(id) => id.to_owned(),
        None => {
            jodel_output::input::default_input()
                .map_err(|e| e.to_string())?
                .id
        }
    };
    jodel_output::input::open_input(&id).map_err(|e| e.to_string())
}

/// The real input devices.
pub fn list_mics() -> Result<Vec<(String, String)>, String> {
    jodel_output::input::input_devices()
        .map(|list| list.into_iter().map(|d| (d.id, d.name)).collect())
        .map_err(|e| e.to_string())
}

/// How long a microphone giving only exact zeros is open before the tab
/// says so.
const MIC_SILENT_AFTER: Duration = Duration::from_millis(1_500);

/// How often the microphone's line goes into the trace while it is open.
const MIC_NOTE_EVERY: Duration = Duration::from_secs(1);

/// A microphone opened in the background (or why not), with the count of
/// the ask it answers.
type Opened = (u64, Result<Input, String>);

/// The microphone, open while Karaoke shows, and the singing heard through
/// it.
pub(crate) struct MicSession {
    /// The microphone open; how one opens, how they are listed, and why
    /// it last did not open (not tried again until Karaoke shows anew).
    input: Option<Input>,
    open: MicOpener,
    list: MicLister,
    failed: bool,
    /// The microphone being opened (its ask), and the count of asks; the
    /// channel the openings come back on.
    opening: Option<u64>,
    asks: u64,
    opened_tx: Sender<Opened>,
    opened_rx: Receiver<Opened>,
    notify: Notify,
    /// When the microphone's line last went into the trace.
    noted: Option<Instant>,
    /// The singer's voice the cleaner gave since the last frame.
    sung: Vec<jodel_output::VoiceBlock>,
    /// The singer's pitch, tracked at the voice's rate.
    yin: Option<(u32, jodel_rhythm::yin::Yin)>,
    /// The singing of the song in Karaoke mode, kept ([`crate::store::keep`]).
    kept: Option<crate::store::keep::Kept>,
    /// A recording of the echo canceller under way.
    echo_recording: Option<jodel_output::EchoRecording>,
    /// The original singer's phrases, worked out for the shine.
    shine: crate::work::shine::ShineWorker,
    /// The songs' loudness as played, measured for the game's sounds.
    loudness: crate::work::loudness::LoudnessWorker,
}

impl MicSession {
    /// A session opening microphones with `open` and listing them with
    /// `list`, its analyses kept beside `analyses` (none: not kept), each
    /// result told through `notify`.
    pub(crate) fn new(
        open: MicOpener,
        list: MicLister,
        analyses: Option<&Path>,
        notify: Notify,
    ) -> MicSession {
        let (opened_tx, opened_rx) = mpsc::channel();
        MicSession {
            input: None,
            open,
            list,
            failed: false,
            opening: None,
            asks: 0,
            opened_tx,
            opened_rx,
            notify: Arc::clone(&notify),
            noted: None,
            sung: Vec::new(),
            yin: None,
            kept: None,
            echo_recording: None,
            shine: crate::work::shine::ShineWorker::new(
                analyses.map_or_else(Default::default, crate::work::shine::Folders::beside),
                Arc::clone(&notify),
            ),
            loudness: crate::work::loudness::LoudnessWorker::new(
                analyses.map_or_else(Default::default, crate::work::loudness::Folders::beside),
                notify,
            ),
        }
    }

    /// The microphone open.
    pub(crate) fn input(&self) -> Option<&Input> {
        self.input.as_ref()
    }

    /// Whether a microphone is being opened.
    pub(crate) fn opening(&self) -> bool {
        self.opening.is_some()
    }

    /// Gives `mixer` the microphone open (or none): its port on the desk.
    pub(crate) fn give(&self, mixer: Option<&Mixer>) {
        if let Some(mixer) = mixer {
            mixer.set_mic(self.input.clone());
        }
    }

    /// Opens the microphone while Karaoke shows (in the background: an
    /// audio device can take a while to answer) and closes it after,
    /// giving `mixer` the change.
    pub(crate) fn turn(&mut self, state: &mut State, mixer: Option<&Mixer>) {
        let want = core::karaoke::mic_wanted(state);
        if want && self.input.is_none() && self.opening.is_none() && !self.failed {
            self.asks += 1;
            let ask = self.asks;
            self.opening = Some(ask);
            let id = state.config.karaoke_mic_device.clone();
            let (open, tx, notify) = (
                Arc::clone(&self.open),
                self.opened_tx.clone(),
                Arc::clone(&self.notify),
            );
            thread::spawn(move || {
                let result = open(id.as_deref());
                if tx.send((ask, result)).is_ok() {
                    notify();
                }
            });
        } else if !want {
            self.failed = false;
            // One being opened is dropped when it comes.
            self.opening = None;
            if self.input.take().is_some() {
                state.karaoke.mic = core::karaoke::MicShown::default();
                self.give(mixer);
            }
        }
        if let Some(input) = &self.input {
            input.set_echo_cancel(state.config.karaoke_echo);
        }
    }

    /// Takes in what came from the background: a microphone opened (or
    /// not), taken if it is the one still wanted and given to `mixer`; the
    /// original singer's phrases, for the notes they were worked out for
    /// (the song's words timed anew since: asked again); the songs'
    /// loudness, for the game's sounds to follow. Whether anything did.
    pub(crate) fn pump(&mut self, state: &mut State, mixer: Option<&Mixer>) -> bool {
        let mut any = false;
        while let Ok((ask, result)) = self.opened_rx.try_recv() {
            any = true;
            self.opened(state, mixer, ask, result);
        }
        while let Some(found) = self.shine.take() {
            let singing = &mut state.karaoke.singing;
            let made = &found.made;
            if singing.is_made_of(&made.path, &made.lyrics, &made.waveform) {
                singing.set_reference(found.reference);
                any = true;
            }
        }
        while let Some(found) = self.loudness.take() {
            state.karaoke.loudness.insert(found.path, found.lufs);
            any = true;
        }
        any
    }

    fn opened(
        &mut self,
        state: &mut State,
        mixer: Option<&Mixer>,
        ask: u64,
        result: Result<Input, String>,
    ) {
        if self.opening != Some(ask) {
            return;
        }
        self.opening = None;
        match result {
            Ok(input) => {
                state.karaoke.mic = core::karaoke::MicShown {
                    name: Some(input.info().name.clone()),
                    error: None,
                    silent: false,
                };
                input.set_echo_cancel(state.config.karaoke_echo);
                self.input = Some(input);
                self.give(mixer);
            }
            Err(why) => {
                self.failed = true;
                state.karaoke.mic = core::karaoke::MicShown {
                    name: None,
                    error: Some(why),
                    silent: false,
                };
            }
        }
    }

    /// `i`: the next microphone, set in the settings (not saved) and the
    /// one open closed (`mixer` told); its name, or why there is none.
    pub(crate) fn next(
        &mut self,
        state: &mut State,
        mixer: Option<&Mixer>,
    ) -> Result<String, String> {
        let list = match (self.list)() {
            Ok(list) if !list.is_empty() => list,
            Ok(_) => return Err("No microphone found".into()),
            Err(why) => return Err(format!("Listing microphones failed: {why}")),
        };
        let current = self
            .input
            .as_ref()
            .map(|m| m.info().id.clone())
            .or_else(|| state.config.karaoke_mic_device.clone());
        let at = current
            .and_then(|id| list.iter().position(|(i, _)| *i == id))
            .map_or(0, |i| (i + 1) % list.len());
        let (id, name) = list[at].clone();
        state.config.karaoke_mic_device = Some(id);
        self.input = None;
        self.failed = false;
        self.opening = None;
        state.karaoke.mic = core::karaoke::MicShown::default();
        self.give(mixer);
        Ok(name)
    }

    /// Records 30 s of what the echo canceller works with, beside the
    /// settings file (`recordings`), for tuning it to a room: what to say.
    pub(crate) fn record_echo(&mut self, state: &State) -> Action {
        let Some(input) = &self.input else {
            return Action::Failed("No microphone open".into());
        };
        let dir = recordings(state);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let path = dir.join(format!("echo-{stamp}.wav"));
        let started = durable::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|()| input.record_echo(&path, 30.0));
        match started {
            Ok(recording) => {
                self.echo_recording = Some(recording);
                Action::Info(format!(
                    "Recording 30 s for the echo canceller (sing over the music): {}",
                    path.display()
                ))
            }
            Err(why) => Action::Failed(format!("Could not record: {why}")),
        }
    }

    /// A recording of the echo canceller done: where it is.
    pub(crate) fn echo_recorded(&mut self) -> Option<Action> {
        let done = self.echo_recording.as_mut().and_then(|r| r.done())?;
        self.echo_recording = None;
        Some(match done {
            Ok(path) => Action::Info(format!("Recorded: {}", path.display())),
            Err(why) => Action::Failed(format!("The recording failed: {why}")),
        })
    }

    /// The microphone read once for the frame into `state` (the Karaoke
    /// tab and the Mixer tab draw from it), and the voice the cleaner gave
    /// since the last frame taken: what it was.
    pub(crate) fn listen(&mut self, state: &mut State) -> &[jodel_output::VoiceBlock] {
        state.mic = None;
        self.sung.clear();
        if let Some(mic) = &self.input {
            let status = mic.status();
            let echo = mic.echo_status();
            let reading = core::mic::MicReading {
                rate: mic.rate(),
                peak: mic.peak(),
                voice: mic.voice_peak(),
                echo_on: mic.echo_cancel(),
                removed_db: echo.removed_db,
                learnt: echo.learnt,
                delay: echo.delay,
                drift_ppm: status.adjust * 1e6,
                fill_ms: status.buffered().as_secs_f64() * 1000.0,
                xruns: status.underruns + status.overruns,
            };
            state.mic = Some(reading);
            state.karaoke.mic.silent = mic.only_silence(MIC_SILENT_AFTER);
            // The voice cleaned, block by block with where in the song it
            // was sung: for scoring the singing once that comes. Taken so
            // the cleaner's queue keeps the newest.
            mic.take_voice(&mut self.sung);
            if crate::trace::on() && self.noted.is_none_or(|t| t.elapsed() >= MIC_NOTE_EVERY) {
                self.noted = Some(Instant::now());
                note_mic(mic, reading.voice, &echo);
            }
        }
        &self.sung
    }

    /// The singer's pitch from the voice the cleaner gave this frame
    /// ([`MicSession::listen`]), each frame of it at the time in the song
    /// it was sung (the output frame playing as it was caught, less how
    /// late the voice comes: [`crate::core::lag`]), held against the song's
    /// notes in Karaoke mode ([`crate::core::singing`]), and the delay
    /// estimated from it. No microphone: the notes made all the same,
    /// nothing heard. The demo key held, the demo sings instead
    /// ([`crate::core::singing::Demo`]).
    pub(crate) fn sing(&mut self, state: &mut State) {
        // Every frame, whatever is heard: the demo let go of hands over to
        // the microphone on the song's time.
        let (now, late) = (state.heard(), state.karaoke.lag.late);
        state.karaoke.demo.follow(now, late);
        let heard = self.input.is_some();
        let song = core::karaoke::song_index(state).filter(|_| core::karaoke::in_mode(state));
        // Where in the song each block was sung, through the head it was
        // heard on (a hand-over moves the microphone's tap).
        let Some(index) = song else {
            return;
        };
        let Some(clock) = core::voice_time::VoiceClock::of(state, index) else {
            return;
        };
        let path = state.queue.paths[index].clone();
        let lyrics = match core::karaoke::lyrics_of(state, index) {
            Some(core::karaoke::SongLyrics::Found { shown, .. }) => Arc::clone(shown),
            _ => return,
        };
        let waveform = match state.rhythm.get(&path) {
            Some(core::Rhythm::Ready(r)) => r.waveform.clone(),
            _ => None,
        };
        let Some(waveform) = waveform.filter(|w| w.melody.is_some()) else {
            return;
        };
        let game = state.show().scored;
        let sung_now = state.karaoke.sung_clock(state.heard());
        // Estimated even with a delay measured: the singing corrects it
        // ([`crate::core::lag::correction`]).
        let karaoke = &mut state.karaoke;
        let (singing, estimate) = (&mut karaoke.singing, &mut karaoke.lag.estimate);
        // The microphone's own delay (the demo's path returns below before
        // it is used): what its frames are placed by and Keep records.
        let (late, source) = (karaoke.lag.late, karaoke.lag.source);
        let demo = &mut karaoke.demo;
        singing.prepare(&path, &lyrics, &waveform);
        // The song's loudness as played, for the game's sounds (the game
        // on): read or measured off this thread, set as it comes
        // ([`MicSession::pump`]).
        if game && !karaoke.loudness.contains_key(&path) {
            self.loudness.want(&path);
        }
        // Without a microphone (nor the demo) the notes are still made (the
        // lane shows them), nothing heard.
        if !heard && !demo.held() {
            return;
        }
        // The original singer's phrases, for the shine (the game on, a
        // microphone open or the demo singing): worked out (or read) off
        // this thread, set as they come ([`MicSession::pump`]).
        if game && singing.reference().is_none() {
            self.shine.want(&path, &lyrics, &waveform);
        }
        // Kept as it is heard, where that is on.
        if state.config.karaoke_keep && self.kept.as_ref().is_none_or(|k| k.song != path) {
            self.kept = Some(crate::store::keep::Kept::new(
                path.clone(),
                Arc::clone(&lyrics),
                Arc::clone(&waveform),
            ));
        }
        // The singer's pitch comes from one place: the demo key held, the
        // demo, up to the singing's time now (the microphone not listened
        // to, the delay not estimated from it, the song's singing not
        // kept; the pitch tracker starts afresh after); else the
        // microphone, through the pitch tracker.
        if demo.held() {
            demo.sing(singing, sung_now);
            if let Some(kept) = self.kept.as_mut().filter(|k| k.song == path) {
                kept.sung_by_demo();
            }
            self.yin = None;
            return;
        }
        // A frame's pitch is that of the window before it: its middle.
        let centre = jodel_rhythm::yin::WINDOW / 2.0;
        let placed = clock.place(&self.sung);
        let mut kept = self.kept.as_mut().filter(|_| state.config.karaoke_keep);
        if let Some(kept) = kept.as_mut() {
            for &(block, at, continues) in &placed {
                let how = crate::store::keep::Sung::new(late, source.key(), singing.shift());
                kept.push(&block.samples, block.rate, at, continues, how);
            }
        }
        // The frames with a pitch: a take without a voice is not saved.
        let mut voiced = 0;
        for &(block, at, continues) in &placed {
            let fresh = match &self.yin {
                Some((rate, _)) => *rate != block.rate || !continues,
                None => true,
            };
            if fresh {
                self.yin = Some((block.rate, jodel_rhythm::yin::Yin::new(block.rate)));
            }
            let Some((_, yin)) = self.yin.as_mut() else {
                continue;
            };
            let rate = f64::from(block.rate.max(1));
            for (i, &x) in block.samples.iter().enumerate() {
                yin.push(&[x], |cents| {
                    voiced += usize::from(cents.is_some());
                    let t = at + i as f64 / rate - centre;
                    demo.microphone((&mut *singing, &mut *estimate), (t, late), cents);
                });
            }
        }
        if let Some(kept) = kept {
            kept.voiced(voiced);
        }
    }

    /// The singing kept saved, once its song is over: another song in
    /// the mode, the mode left, or keeping turned off (`now`: whatever).
    /// What to say of it.
    pub(crate) fn save_kept(&mut self, state: &State, now: bool) -> Option<Action> {
        let kept = self.kept.as_ref()?;
        let still = core::karaoke::in_mode(state)
            && state.config.karaoke_keep
            && core::karaoke::song_index(state)
                .and_then(|i| state.queue.paths.get(i))
                .is_some_and(|p| *p == kept.song);
        if still && !now {
            return None;
        }
        let kept = self.kept.take()?;
        if !kept.worth_saving() {
            return None;
        }
        Some(match kept.save(&recordings(state)) {
            Ok(folder) => Action::Info(format!("Your singing kept: {}", folder.display())),
            Err(why) => Action::Failed(format!("Your singing could not be kept: {why}")),
        })
    }
}

/// Where recordings go: `recordings` beside the settings file.
fn recordings(state: &State) -> PathBuf {
    match state.config_file.as_deref().and_then(Path::parent) {
        Some(dir) => dir.join("recordings"),
        None => std::env::temp_dir(),
    }
}

/// The microphone and its echo canceller, for the trace: the voice's peak
/// as captured and as cleaned (the music's echo out of it), what the
/// canceller does, and how the capture keeps up with the output (a skip
/// moves the echo's delay).
fn note_mic(mic: &Input, cleaned: Option<f32>, echo: &jodel_output::aec::EchoStatus) {
    let dbfs = |v: f32| 20.0 * v.max(1e-6).log10();
    let status = mic.status();
    let cleaned = cleaned.map_or_else(
        || "no output tapped".to_owned(),
        |v| format!("{:.0} dBFS cleaned", dbfs(v)),
    );
    let echo_line = if !mic.echo_cancel() {
        "off".to_owned()
    } else if echo.learnt {
        format!(
            "{:.0} dB out ({:.0} by the filter), the echo {} late",
            echo.removed_db,
            echo.filter_db,
            echo.delay
                .map_or_else(|| "?".to_owned(), |d| format!("{:.0} ms", d * 1000.0))
        )
    } else {
        "learning".to_owned()
    };
    crate::trace::note(format_args!(
        "microphone: {:.0} dBFS captured, {cleaned}; echo canceller {echo_line}; \
         {:.1} ms buffered (target {} frames at {} Hz), clocks {:+.0} ppm apart, \
         {} underruns, {} resyncs, {} overruns, {} blocks lost to the canceller, \
         {} cleaned blocks not taken",
        dbfs(mic.peak()),
        status.buffered().as_secs_f64() * 1000.0,
        status.target,
        status.rate,
        status.adjust * 1e6,
        status.underruns,
        status.resyncs,
        status.overruns,
        status.blocks_dropped,
        status.voice_dropped,
    ));
}
