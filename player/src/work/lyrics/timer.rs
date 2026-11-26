//! The speech model's word timing, and the voice it times the words
//! against.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_lyrics::aligner::{Acoustic, CtcAligner};
use jodel_lyrics::lines::transcript_lyrics;
use jodel_lyrics::prepare::{self, Others};
use jodel_lyrics::{Lyrics, Maker};

use super::models::Models;
use super::{relined, WordTimer, TOO_LITTLE_HEARD};

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
pub struct SpeechTimer {
    models: Arc<Models>,
    /// The player's kept stems, where the voice is taken from when there.
    stems: Option<PathBuf>,
    model: Mutex<Option<Speech>>,
    /// The recogniser once read: `None` inside when it cannot run here.
    /// One whose GPU was lost is read again.
    asr: Mutex<Option<Option<Arc<jodel_gpu::GpuAsr>>>>,
}

/// The speech model as read: on the GPU, or on the processor.
enum Speech {
    Gpu(Arc<jodel_gpu::GpuSpeech>),
    Cpu(Arc<jodel_lyrics::speech::Wav2Vec2>),
}

impl Speech {
    fn acoustic(&self) -> Arc<dyn Acoustic + Send + Sync> {
        match self {
            Speech::Gpu(gpu) => Arc::clone(gpu) as _,
            Speech::Cpu(cpu) => Arc::clone(cpu) as _,
        }
    }
}

impl SpeechTimer {
    /// The timer for `models` (each read when first needed), the voice
    /// taken from the stems kept in `stems` where they are there.
    pub fn new(models: Arc<Models>, stems: Option<PathBuf>) -> SpeechTimer {
        SpeechTimer {
            models,
            stems,
            model: Mutex::new(None),
            asr: Mutex::new(None),
        }
    }

    /// The speech recogniser on the GPU, when it is there and there is one.
    fn recogniser(&self) -> Option<Arc<jodel_gpu::GpuAsr>> {
        if !self.models.has_asr() {
            return None;
        }
        let mut kept = self.asr.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(&*kept, Some(Some(asr)) if asr.is_lost()) {
            crate::trace::note("lyrics: the GPU was lost; the speech recogniser is read again");
            *kept = None;
        }
        if let Some(asr) = kept.as_ref() {
            return asr.clone();
        }
        let asr = jodel_lyrics::asr::packed::PackedModel::open(&self.models.asr_now())
            .and_then(|packed| jodel_gpu::GpuAsr::new(&packed))
            .ok()
            .map(Arc::new);
        *kept = Some(asr.clone());
        asr
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
        let model = self.acoustic()?;
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        // Any text checked against what is sung (one from LRCLIB swapped
        // for one of its others when it does not fit: `others` has none
        // for the file's own); what was heard, in the transcript's parts
        // and in glimpses, also pins the words to where it was.
        let (lyrics, fit, parts, glimpses) = match self.recogniser() {
            Some(asr) => match steps.hear(&asr, || asr.transcribe(&voice), |s| &mut s.heard) {
                Ok(heard) => {
                    let (lyrics, fit) = prepare::text(lyrics, &heard, others);
                    let glimpses = steps.hear(&asr, || asr.glimpses(&voice), |s| &mut s.glimpsed);
                    (lyrics, fit, heard.parts, glimpses.unwrap_or_default())
                }
                Err(_) => (lyrics.clone(), None, Vec::new(), Vec::new()),
            },
            None => (lyrics.clone(), None, Vec::new(), Vec::new()),
        };
        let started = Instant::now();
        let mut timed = CtcAligner::new(model).align(&voice, &lyrics, &parts, &glimpses);
        steps.aligned = started.elapsed();
        timed.timing.fit = fit;
        steps.note(path);
        Ok(timed)
    }

    fn time_edited(&self, path: &Path, lyrics: &Lyrics) -> Result<Lyrics, String> {
        let model = self.acoustic()?;
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        // The text stays as edited; what was heard still pins its words to
        // where they were sung.
        let (parts, glimpses) = match self.recogniser() {
            Some(asr) => match steps.hear(&asr, || asr.transcribe(&voice), |s| &mut s.heard) {
                Ok(heard) => {
                    let glimpses = steps.hear(&asr, || asr.glimpses(&voice), |s| &mut s.glimpsed);
                    (heard.parts, glimpses.unwrap_or_default())
                }
                Err(_) => (Vec::new(), Vec::new()),
            },
            None => (Vec::new(), Vec::new()),
        };
        let started = Instant::now();
        let timed = CtcAligner::new(model).align(&voice, lyrics, &parts, &glimpses);
        steps.aligned = started.elapsed();
        steps.note(path);
        Ok(timed)
    }

    fn transcribe(&self, path: &Path) -> Result<(Lyrics, Lyrics), String> {
        let asr = self.recogniser().ok_or("no speech recogniser here")?;
        let model = self.acoustic()?;
        let mut steps = Steps::default();
        let voice = steps.voice(self.stems.as_deref(), path)?;
        let heard = steps.hear(&asr, || asr.transcribe(&voice), |s| &mut s.heard)?;
        if !prepare::enough_heard(&heard.text) {
            return Err(TOO_LITTLE_HEARD.into());
        }
        let text = transcript_lyrics(&heard, asr_maker());
        let glimpses = steps
            .hear(&asr, || asr.glimpses(&voice), |s| &mut s.glimpsed)
            .unwrap_or_default();
        let started = Instant::now();
        let timed = CtcAligner::new(model).align(&voice, &text, &heard.parts, &glimpses);
        steps.aligned = started.elapsed();
        steps.note(path);
        // The lines made again where the words' timing says they break.
        Ok(relined(&timed))
    }
}

impl SpeechTimer {
    /// The speech model that times the words, read once (and again where
    /// its GPU was lost): on the GPU where there is one (25 times as fast,
    /// and the processor left to playback), else on half the processor's
    /// threads.
    fn acoustic(&self) -> Result<Arc<dyn Acoustic + Send + Sync>, String> {
        if !self.models.speech_now().is_file() {
            return Err("the speech model is not here".into());
        }
        let mut kept = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(&*kept, Some(Speech::Gpu(gpu)) if gpu.is_lost()) {
            crate::trace::note("lyrics: the GPU was lost; the speech model is read again");
            *kept = None;
        }
        if let Some(m) = kept.as_ref() {
            return Ok(m.acoustic());
        }
        let mut cpu = jodel_lyrics::speech::Wav2Vec2::open(&self.models.speech_now())?;
        let m = match jodel_gpu::GpuSpeech::new(&cpu) {
            Ok(gpu) => {
                crate::trace::note("lyrics: the speech model runs on the GPU");
                Speech::Gpu(Arc::new(gpu))
            }
            Err(why) => {
                let threads = thread::available_parallelism().map_or(2, |n| n.get());
                crate::trace::note(format_args!(
                    "lyrics: the speech model runs on {} processor threads ({why})",
                    (threads / 2).max(1)
                ));
                cpu.set_threads((threads / 2).max(1));
                Speech::Cpu(Arc::new(cpu))
            }
        };
        let acoustic = m.acoustic();
        *kept = Some(m);
        Ok(acoustic)
    }
}

/// How long a song's timing took, step by step, for the trace; with the
/// trace on, also what the separator and the recogniser did in their steps.
#[derive(Default)]
struct Steps {
    voiced: Duration,
    /// The voice was separated now (not taken from the kept stems).
    separated: bool,
    /// The separator's GPU work meanwhile (other work on it included).
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
        let separator = jodel_gpu::GpuSeparator::for_this_thread()
            .filter(|_| crate::trace::on())
            .map(|g| (g, g.tally()));
        let started = Instant::now();
        let (voice, separated) = track_voice(stems, path)?;
        self.voiced = started.elapsed();
        self.separated = separated;
        self.separator = separator
            .filter(|_| separated)
            .map(|(g, before)| g.tally().since(&before));
        Ok(voice)
    }

    /// `f`, a step of the recogniser `asr`'s, into the step `step` picks.
    fn hear<T>(
        &mut self,
        asr: &jodel_gpu::GpuAsr,
        f: impl FnOnce() -> T,
        step: impl FnOnce(&mut Self) -> &mut Step,
    ) -> T {
        let before = crate::trace::on().then(|| asr.hearing().ok()).flatten();
        let started = Instant::now();
        let out = f();
        let step = step(self);
        step.took = started.elapsed();
        step.hearing = before.and_then(|b| Some(asr.hearing().ok()?.since(&b)));
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
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
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
                "lyrics: {name}, the separator for the voice (other work on it meanwhile included): {separator}"
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

/// The voice of `path`, mono at 16 kHz: from the track's stems kept in
/// `stems` (once its analysis has them in their file, if it is separating
/// them now: [`crate::store::stems::open_waiting`]), else separated now
/// (true).
fn track_voice(stems: Option<&Path>, path: &Path) -> Result<(Vec<f32>, bool), String> {
    let kept = stems.and_then(|dir| crate::store::stems::open_waiting(dir, path));
    match kept {
        Some(file) => Ok((voice_16k(file)?, false)),
        None => Ok((separate_16k(path, false)?.0, true)),
    }
}
/// The voice in a track's kept stems, mono at 16 kHz: the vocals' two
/// channels averaged and resampled. Nothing is separated.
pub fn voice_16k(mut file: jodel_playback::StemFile) -> Result<Vec<f32>, String> {
    use jodel_playback::stems::Stem;
    let mut r = jodel_playback::Resampler::new(jodel_playback::stems::RATE, 16_000, 1);
    let (mut voice, mut out) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < file.frames() {
        let stems = file
            .read(at, jodel_playback::stems::RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        let [l, rt] = &stems[Stem::Vocals];
        if l.is_empty() {
            break;
        }
        at += l.len() as u64;
        voice.clear();
        voice.extend(
            l.iter()
                .zip(rt)
                .map(|(a, b)| (f64::from(*a) + f64::from(*b)) / 2.0),
        );
        r.process_f64(&voice, &mut out);
    }
    r.finish_f64(&mut out);
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// The voice of `path` (and, with `mix`, the whole mix), mono at 16 kHz:
/// the file decoded, separated into its stems on the GPU, the vocals'
/// two channels averaged.
pub fn separate_16k(path: &Path, mix: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
    use jodel_playback::stems::{Stem, StereoFeed};
    use jodel_rhythm::stems::{Splitter, StemBlock};
    let gpu = jodel_gpu::GpuSeparator::for_this_thread().ok_or("no GPU for the stems")?;
    let mut source = jodel_playback::Source::open(path).map_err(|e| e.to_string())?;
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    let mut feed = StereoFeed::new(rate);
    let mut splitter = Splitter::overlapped(gpu);
    let mut voice: Vec<f64> = Vec::new();
    let mut whole: Vec<f64> = Vec::new();
    let mut take = |b: &StemBlock| {
        let [l, r] = b.stems[Stem::Vocals];
        if voice.len() < b.start + l.len() {
            voice.resize(b.start + l.len(), 0.0);
        }
        for (i, (a, c)) in l.iter().zip(r).enumerate() {
            voice[b.start + i] = (f64::from(*a) + f64::from(*c)) / 2.0;
        }
    };
    let mut failed = None;
    let mut put = |left: Vec<f32>, right: Vec<f32>| {
        if mix {
            whole.extend(
                left.iter()
                    .zip(&right)
                    .map(|(a, b)| (f64::from(*a) + f64::from(*b)) / 2.0),
            );
        }
        if let Err(e) = splitter.push(&left, &right, &mut take) {
            failed.get_or_insert(e);
        }
    };
    let mut block = Vec::new();
    loop {
        block.clear();
        let frames = source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?;
        if frames == 0 {
            break;
        }
        feed.push(&block, channels, &mut put);
    }
    feed.finish(&mut put);
    if let Some(e) = failed {
        return Err(e);
    }
    splitter.finish(&mut take)?;
    let to_16k = |x: &[f64]| {
        let mut r = jodel_playback::Resampler::new(jodel_playback::stems::RATE, 16_000, 1);
        let mut out = Vec::new();
        r.process_f64(x, &mut out);
        r.finish_f64(&mut out);
        out.into_iter().map(|v| v as f32).collect::<Vec<f32>>()
    };
    Ok((to_16k(&voice), to_16k(&whole)))
}
