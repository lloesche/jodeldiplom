//! The speech model's word timing, and the voice it times the words
//! against.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_gpu::{GpuAsr, GpuSpeech, Kept, Resident};
use jodel_lyrics::aligner::{Acoustic, CtcAligner};
use jodel_lyrics::asr::Transcript;
use jodel_lyrics::ctc::Emissions;
use jodel_lyrics::lines::transcript_lyrics;
use jodel_lyrics::prepare::{self, Others};
use jodel_lyrics::speech::Wav2Vec2;
use jodel_lyrics::{Lyrics, Maker};

use super::models::Models;
use super::{relined, WordTimer, TOO_LITTLE_HEARD};
use crate::work::stages::{self, Stage};
use crate::work::voice::{voice, voice_16k, Voice};

/// The word timing of the speech model: the voice separated on the GPU,
/// wav2vec2 on the GPU or half the processor's threads (the rest left to
/// playback), and the CTC alignment over the whole song (the words' times
/// ours; the lines held near their source's synced times only where those
/// fit the recording: [`jodel_lyrics::scaffold`]). Any text is first
/// checked against what the speech recogniser hears in the voice (on the
/// GPU, where it and the recogniser are there), one from LRCLIB swapped for
/// one of LRCLIB's others when it does not fit ([`prepare`]); what it
/// heard, in the transcript's parts and in glimpses
/// ([`jodel_lyrics::glimpses`]), pins the words to where it was sung.
///
/// A song's timing is three jobs of the GPU's stages, one after another:
/// its voice separated (where its stems are not kept), heard by the
/// recogniser ([`Stage::Hearing`]), timed by the speech model
/// ([`Stage::Timing`]). Both models are kept on the GPU as the stages
/// plan ([`Kept`], registered for their stages as the timer is made), set
/// up in their turns.
pub struct SpeechTimer {
    models: Arc<Models>,
    /// The player's kept stems, where the voice is taken from when there.
    stems: Option<PathBuf>,
    /// The speech model on the GPU.
    gpu: Arc<Kept<GpuSpeech>>,
    /// The speech model on the processor, read where the GPU's does not
    /// set up.
    cpu: Mutex<Option<Arc<Wav2Vec2>>>,
    /// The recogniser on the GPU.
    asr: Arc<Kept<GpuAsr>>,
}

/// The bytes of the file at `path`, where it is there.
fn bytes_of(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// The speech model on the GPU as the aligner takes it.
struct OnGpu(Arc<Kept<GpuSpeech>>);

impl Acoustic for OnGpu {
    fn maker(&self) -> Maker {
        jodel_lyrics::speech::maker()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        self.0.run(|m| Ok(m.emissions(vocals_16k))).ok().flatten()
    }
}

impl SpeechTimer {
    /// The timer for `models` (each read when first needed), the voice
    /// taken from the stems kept in `stems` where they are there; its
    /// models registered for their stages.
    pub fn new(models: Arc<Models>, stems: Option<PathBuf>) -> SpeechTimer {
        let (on_disk, read) = (Arc::clone(&models), Arc::clone(&models));
        let gpu = Arc::new(Kept::new(
            move || bytes_of(&on_disk.speech_now()).map(GpuSpeech::footprint),
            Box::new(move || {
                let speech = GpuSpeech::new(&Wav2Vec2::open(&read.speech_now())?)?;
                let name = speech.name().to_owned();
                Ok((speech, name))
            }),
        ));
        let (on_disk, read) = (Arc::clone(&models), Arc::clone(&models));
        let asr = Arc::new(Kept::new(
            move || bytes_of(&on_disk.asr_now()).map(GpuAsr::footprint),
            Box::new(move || {
                let packed = jodel_lyrics::asr::packed::PackedModel::open(&read.asr_now())?;
                let asr = GpuAsr::new(&packed)?;
                let name = asr.name().to_owned();
                Ok((asr, name))
            }),
        ));
        stages::register(Stage::Timing, Arc::clone(&gpu) as Arc<dyn Resident>);
        stages::register(Stage::Hearing, Arc::clone(&asr) as Arc<dyn Resident>);
        SpeechTimer {
            models,
            stems,
            gpu,
            cpu: Mutex::new(None),
            asr,
        }
    }

    /// The speech recogniser, where it is there.
    fn recogniser(&self) -> Option<&Kept<GpuAsr>> {
        self.models.has_asr().then_some(&*self.asr)
    }

    /// `f` given the speech recogniser `asr` (where it is there) on the GPU
    /// where it sets up: in the recogniser's turn.
    fn hearing<T>(
        &self,
        path: &Path,
        asr: Option<&Kept<GpuAsr>>,
        f: impl FnOnce(Option<&Kept<GpuAsr>>) -> T,
    ) -> T {
        let Some(asr) = asr else {
            return f(None);
        };
        let name = crate::work::song_name(path);
        stages::run(&[Stage::Hearing], &name, || f(asr.usable().then_some(asr)))
    }

    /// `f` given the speech model that times the words: in its turn.
    fn timing<T>(
        &self,
        path: &Path,
        f: impl FnOnce(Arc<dyn Acoustic + Send + Sync>) -> T,
    ) -> Result<T, String> {
        if !self.models.speech_now().is_file() {
            return Err("the speech model is not here".into());
        }
        let name = crate::work::song_name(path);
        stages::run(&[Stage::Timing], &name, || Ok(f(self.acoustic()?)))
    }

    /// The speech model that times the words: on the GPU where it sets up
    /// (25 times as fast, and the processor left to playback), else on half
    /// the processor's threads (read once).
    fn acoustic(&self) -> Result<Arc<dyn Acoustic + Send + Sync>, String> {
        if self.gpu.usable() {
            return Ok(Arc::new(OnGpu(Arc::clone(&self.gpu))));
        }
        let mut cpu = self.cpu.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(model) = cpu.as_ref() {
            return Ok(Arc::clone(model) as _);
        }
        let mut model = Wav2Vec2::open(&self.models.speech_now())?;
        let threads = (thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1);
        crate::trace::note(format_args!(
            "lyrics: the speech model runs on {threads} processor threads (not on the GPU)"
        ));
        model.set_threads(threads);
        let model = Arc::new(model);
        *cpu = Some(Arc::clone(&model));
        Ok(model)
    }
}

impl WordTimer for SpeechTimer {
    fn ready(&self) -> bool {
        self.models.speech_now().is_file()
    }

    fn maker(&self) -> Maker {
        jodel_lyrics::aligner::aligner_maker(&jodel_lyrics::speech::maker())
    }

    fn time(&self, path: &Path, lyrics: &Lyrics, others: &Others) -> Result<Lyrics, String> {
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        // Any text checked against what is sung (one from LRCLIB swapped
        // for one of its others when it does not fit: `others` has none
        // for the file's own); what was heard, in the transcript's parts
        // and in glimpses, also pins the words to where it was.
        let heard = self.hearing(path, self.recogniser(), |asr| {
            steps.listen(asr, &voice, |_| Ok(()))
        });
        let (lyrics, fit, parts, glimpses) = match heard {
            Ok((heard, glimpses)) => {
                let (lyrics, fit) = prepare::text(lyrics, &heard, others);
                (lyrics, fit, heard.parts, glimpses)
            }
            Err(_) => (lyrics.clone(), None, Vec::new(), Vec::new()),
        };
        let mut timed = self.timing(path, |model| {
            steps.align(model, &voice, &lyrics, (&parts, &glimpses))
        })?;
        timed.timing.fit = fit;
        steps.note(path);
        Ok(timed)
    }

    fn time_edited(&self, path: &Path, lyrics: &Lyrics) -> Result<Lyrics, String> {
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        // The text stays as edited; what was heard still pins its words to
        // where they were sung.
        let (parts, glimpses) = self
            .hearing(path, self.recogniser(), |asr| {
                steps.listen(asr, &voice, |_| Ok(()))
            })
            .map_or_else(|_| (Vec::new(), Vec::new()), |(heard, g)| (heard.parts, g));
        let timed = self.timing(path, |model| {
            steps.align(model, &voice, lyrics, (&parts, &glimpses))
        })?;
        steps.note(path);
        Ok(timed)
    }

    fn transcribe(&self, path: &Path) -> Result<(Lyrics, Lyrics), String> {
        let asr = self.recogniser().ok_or("no speech recogniser here")?;
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        let (heard, glimpses) = self.hearing(path, Some(asr), |asr| {
            steps.listen(asr, &voice, |heard| {
                if prepare::enough_heard(&heard.text) {
                    Ok(())
                } else {
                    Err(TOO_LITTLE_HEARD.to_owned())
                }
            })
        })?;
        let text = transcript_lyrics(&heard, asr_maker());
        let timed = self.timing(path, |model| {
            steps.align(model, &voice, &text, (&heard.parts, &glimpses))
        })?;
        steps.note(path);
        // The lines made again where the words' timing says they break.
        Ok(relined(&timed))
    }
}

/// How long a song's timing took, step by step, for the trace; with the
/// trace on, also what the separator and the recogniser did in their steps.
#[derive(Default)]
struct Steps {
    voiced: Duration,
    /// The voice was separated now (not taken from the kept stems).
    separated: bool,
    /// The separators' GPU work meanwhile (other work on them included).
    separator: Option<jodel_gpu::pace::Tally>,
    heard: Step,
    glimpsed: Step,
    aligned: Duration,
}

/// A step of the recogniser's: how long it took, and (with the trace on)
/// what the recogniser did in it.
#[derive(Default)]
struct Step {
    took: Duration,
    hearing: Option<jodel_gpu::Hearing>,
}

impl Steps {
    /// The voice of `path` ([`track_voice`], from the stems kept in
    /// `stems`), its time, and where it came from.
    fn voice(&mut self, stems: Option<&Path>, path: &Path) -> Result<Vec<f32>, String> {
        let tally = jodel_gpu::GpuSeparator::tally_all;
        let before = crate::trace::on().then(tally);
        let started = Instant::now();
        let (voice, separated) = track_voice(stems, path)?;
        self.voiced = started.elapsed();
        self.separated = separated;
        self.separator = before
            .filter(|_| separated)
            .map(|before| tally().since(&before));
        Ok(voice)
    }

    /// The words of `lyrics` timed by `model` against `voice`, what was
    /// heard (its `parts` and `glimpses`) pinning them; how long it took.
    fn align(
        &mut self,
        model: Arc<dyn Acoustic + Send + Sync>,
        voice: &[f32],
        lyrics: &Lyrics,
        (parts, glimpses): (&[jodel_lyrics::asr::Part], &[jodel_lyrics::asr::Part]),
    ) -> Lyrics {
        let started = Instant::now();
        let timed = CtcAligner::new(model).align(voice, lyrics, parts, glimpses);
        self.aligned = started.elapsed();
        timed
    }

    /// What the recogniser `asr` (none: it does not set up here) hears in
    /// `voice`: its transcript, checked by `enough` before its glimpses
    /// are heard (none where they fail).
    fn listen(
        &mut self,
        asr: Option<&Kept<GpuAsr>>,
        voice: &[f32],
        enough: impl FnOnce(&Transcript) -> Result<(), String>,
    ) -> Result<(Transcript, Vec<jodel_lyrics::asr::Part>), String> {
        let asr = asr.ok_or("the speech recogniser does not set up here")?;
        let heard = self.hear(asr, || asr.run(|a| a.transcribe(voice)), |s| &mut s.heard)?;
        enough(&heard)?;
        let glimpses = self
            .hear(asr, || asr.run(|a| a.glimpses(voice)), |s| &mut s.glimpsed)
            .unwrap_or_default();
        Ok((heard, glimpses))
    }

    /// `f`, a step of the recogniser `asr`'s, into the step `step` picks.
    fn hear<T>(
        &mut self,
        asr: &Kept<GpuAsr>,
        f: impl FnOnce() -> T,
        step: impl FnOnce(&mut Self) -> &mut Step,
    ) -> T {
        let hearing = || asr.run(|a| a.hearing()).ok();
        let before = crate::trace::on().then(hearing).flatten();
        let started = Instant::now();
        let out = f();
        let step = step(self);
        step.took = started.elapsed();
        step.hearing = before.and_then(|b| Some(hearing()?.since(&b)));
        out
    }

    /// The trace's lines for `path`: the steps' times, what the separator
    /// did for the voice (when it separated it), and the recogniser's work
    /// in the transcript and in the glimpses.
    fn note(&self, path: &Path) {
        if !crate::trace::on() {
            return;
        }
        let s = |d: Duration| d.as_secs_f64();
        let name = crate::work::song_name(path);
        let from = if self.separated {
            "separated now"
        } else {
            "its kept stems"
        };
        crate::trace::note(format_args!(
            "lyrics: {name} timed: the voice {:.1} s ({from}), heard {:.1} s, glimpses {:.1} s, aligned {:.1} s",
            s(self.voiced),
            s(self.heard.took),
            s(self.glimpsed.took),
            s(self.aligned),
        ));
        if let Some(separator) = self.separator {
            crate::trace::note(format_args!(
                "lyrics: {name}, the separators for the voice (other work on them meanwhile included): {separator}"
            ));
        }
        for (what, step) in [
            ("the transcript", &self.heard),
            ("the glimpses", &self.glimpsed),
        ] {
            if let Some(hearing) = &step.hearing {
                crate::trace::note(format_args!(
                    "lyrics: {name}, the recogniser in {what}: {hearing}"
                ));
            }
        }
    }
}

/// What made a transcript's text: the speech recogniser, as packed.
pub fn asr_maker() -> Maker {
    Maker::new(
        "qwen3-asr-0.6b-q4",
        u32::from(jodel_lyrics::asr::packed::VERSION),
    )
}

/// The voice of `path`, mono at 16 kHz ([`voice`]), and whether it was
/// separated now.
fn track_voice(stems: Option<&Path>, path: &Path) -> Result<(Vec<f32>, bool), String> {
    match voice(stems, path)? {
        Voice::Kept(file) => Ok((voice_16k(*file)?, false)),
        Voice::Separated(voice) => Ok((voice, true)),
    }
}
