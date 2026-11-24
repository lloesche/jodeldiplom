//! The speech model's word timing, and the voice it times the words
//! against.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_lyrics::{Lyrics, Maker};

use super::models::Models;
use super::{
    prepare_text, relined, transcript_lyrics, Others, WordTimer, HEARD_LEAST, TOO_LITTLE_HEARD,
};

/// The word timing of the speech model: the voice separated on the GPU,
/// wav2vec2 on the GPU or half the processor's threads (the rest left to
/// playback), and the CTC alignment over the whole song (the words' times
/// ours; the lines held near their source's synced times only where those
/// fit the recording: [`jodel_lyrics::scaffold`]). Any text is first
/// checked against what the speech recogniser hears in the voice (on the
/// GPU, where it and the recogniser are there), one from LRCLIB swapped for
/// one of LRCLIB's others when it does not fit ([`fitting`]); what it
/// heard, in the transcript's parts and in glimpses
/// ([`jodel_lyrics::glimpses`]), pins the words to where it was sung.
pub struct SpeechTimer {
    models: Arc<Models>,
    model: Mutex<Option<Arc<dyn jodel_lyrics::aligner::Acoustic + Send + Sync>>>,
    /// The recogniser once read: `None` inside when it cannot run here.
    asr: Mutex<Option<Option<Arc<jodel_gpu::GpuAsr>>>>,
}

impl SpeechTimer {
    /// The timer for `models` (each read when first needed).
    pub fn new(models: Arc<Models>) -> SpeechTimer {
        SpeechTimer {
            models,
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
        use jodel_lyrics::aligner::CtcAligner;
        let model = self.acoustic()?;
        let started = Instant::now();
        let voice = track_voice(path)?;
        let voiced = started.elapsed();
        // Any text checked against what is sung (one from LRCLIB swapped
        // for one of its others when it does not fit: `others` has none
        // for the file's own); what was heard, in the transcript's parts
        // and in glimpses, also pins the words to where it was.
        let mut steps = Steps::default();
        let (lyrics, fit, parts, glimpses) = match self.recogniser() {
            Some(asr) => match steps.of(|| asr.transcribe(&voice), |s| &mut s.heard) {
                Ok(heard) => {
                    let (lyrics, fit) = prepare_text(lyrics, &heard, others);
                    let glimpses = steps.of(|| asr.glimpses(&voice), |s| &mut s.glimpsed);
                    (lyrics, fit, heard.parts, glimpses.unwrap_or_default())
                }
                Err(_) => (lyrics.clone(), None, Vec::new(), Vec::new()),
            },
            None => (lyrics.clone(), None, Vec::new(), Vec::new()),
        };
        let mut timed = steps.of(
            || CtcAligner::new(model).align_heard_in(&voice, &lyrics, &parts, &glimpses),
            |s| &mut s.aligned,
        );
        timed.timing.fit = fit;
        steps.note(path, voiced);
        Ok(timed)
    }

    fn transcribe(&self, path: &Path) -> Result<(Lyrics, Lyrics), String> {
        use jodel_lyrics::aligner::CtcAligner;
        let asr = self.recogniser().ok_or("no speech recogniser here")?;
        let model = self.acoustic()?;
        let started = Instant::now();
        let voice = track_voice(path)?;
        let voiced = started.elapsed();
        let mut steps = Steps::default();
        let heard = steps.of(|| asr.transcribe(&voice), |s| &mut s.heard)?;
        if jodel_lyrics::fit::words(&heard.text).len() < HEARD_LEAST {
            return Err(TOO_LITTLE_HEARD.into());
        }
        let text = transcript_lyrics(&heard, asr_maker());
        let glimpses = steps.of(|| asr.glimpses(&voice), |s| &mut s.glimpsed);
        let timed = steps.of(
            || {
                let glimpses = glimpses.unwrap_or_default();
                CtcAligner::new(model).align_heard_in(&voice, &text, &heard.parts, &glimpses)
            },
            |s| &mut s.aligned,
        );
        steps.note(path, voiced);
        // The lines made again where the words' timing says they break.
        Ok(relined(&timed))
    }
}

impl SpeechTimer {
    /// The speech model that times the words, read once: on the GPU where
    /// there is one (25 times as fast, and the processor left to
    /// playback), else on half the processor's threads.
    fn acoustic(&self) -> Result<Arc<dyn jodel_lyrics::aligner::Acoustic + Send + Sync>, String> {
        if !self.models.speech_now().is_file() {
            return Err("the speech model is not here".into());
        }
        let mut kept = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = kept.as_ref() {
            return Ok(Arc::clone(m));
        }
        let mut cpu = jodel_lyrics::speech::Wav2Vec2::open(&self.models.speech_now())?;
        let m: Arc<dyn jodel_lyrics::aligner::Acoustic + Send + Sync> =
            match jodel_gpu::GpuSpeech::new(&cpu) {
                Ok(gpu) => {
                    crate::trace::note("lyrics: the speech model runs on the GPU");
                    Arc::new(gpu)
                }
                Err(why) => {
                    let threads = thread::available_parallelism().map_or(2, |n| n.get());
                    crate::trace::note(format_args!(
                        "lyrics: the speech model runs on {} processor threads ({why})",
                        (threads / 2).max(1)
                    ));
                    cpu.set_threads((threads / 2).max(1));
                    Arc::new(cpu)
                }
            };
        *kept = Some(Arc::clone(&m));
        Ok(m)
    }
}

/// How long a song's timing took, step by step, for the trace.
#[derive(Default)]
struct Steps {
    heard: Duration,
    glimpsed: Duration,
    aligned: Duration,
}

impl Steps {
    /// `f`, its time added to the step `field` picks.
    fn of<T>(
        &mut self,
        f: impl FnOnce() -> T,
        field: impl FnOnce(&mut Self) -> &mut Duration,
    ) -> T {
        let started = Instant::now();
        let out = f();
        *field(self) += started.elapsed();
        out
    }

    /// The trace's line for `path`, its voice got in `voiced`.
    fn note(&self, path: &Path, voiced: Duration) {
        let s = |d: Duration| d.as_secs_f64();
        crate::trace::note(format_args!(
            "lyrics: {} timed: the voice {:.1} s, heard {:.1} s, glimpses {:.1} s, aligned {:.1} s",
            path.file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            s(voiced),
            s(self.heard),
            s(self.glimpsed),
            s(self.aligned),
        ));
    }
}

/// What made a transcript's text: the speech recogniser, as packed.
pub fn asr_maker() -> Maker {
    Maker::new(
        "qwen3-asr-0.6b-q4",
        u32::from(jodel_lyrics::asr::packed::VERSION),
    )
}

/// The voice of `path`, mono at 16 kHz: from the track's kept stems, else
/// separated now.
fn track_voice(path: &Path) -> Result<Vec<f32>, String> {
    let kept = crate::stems_cache::dir().and_then(|dir| crate::stems_cache::open(&dir, path));
    match kept {
        Some(file) => voice_16k(file),
        None => Ok(separate_16k(path, false)?.0),
    }
}
/// The voice in a track's kept stems, mono at 16 kHz: the vocals' two
/// channels averaged and resampled. Nothing is separated.
pub fn voice_16k(mut file: jodel_playback::StemFile) -> Result<Vec<f32>, String> {
    use jodel_playback::stems::Stem;
    let vocals = Stem::ALL
        .iter()
        .position(|&s| s == Stem::Vocals)
        .unwrap_or(2);
    let mut r = jodel_playback::Resampler::new(jodel_playback::stems::RATE, 16_000, 1);
    let (mut voice, mut out) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < file.frames() {
        let stems = file
            .read(at, jodel_playback::stems::RATE as usize * 10)
            .map_err(|e| e.to_string())?;
        let [l, rt] = &stems[vocals];
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
    let gpu = if jodel_gpu::pace::is_background() {
        jodel_gpu::GpuSeparator::background()
    } else {
        jodel_gpu::GpuSeparator::shared()
    }
    .ok_or("no GPU for the stems")?;
    let vocals = Stem::ALL
        .iter()
        .position(|&s| s == Stem::Vocals)
        .unwrap_or(2);
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
        let [l, r] = b.stems[vocals];
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
