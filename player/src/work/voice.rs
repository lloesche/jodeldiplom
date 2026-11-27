//! A song's voice, as its words' timing and its singer's phrases take it:
//! from its kept stems, else separated now on the GPU, in the separator's
//! turn of the GPU's stages ([`Stage::Stems`]).

use std::path::Path;

use crate::work::stages::{self, Stage};

/// A song's voice as its words' timing and its singer's phrases take it
/// ([`voice`]).
pub enum Voice {
    /// Its stems kept: their file.
    Kept(Box<jodel_playback::StemFile>),
    /// Separated now: mono at 16 kHz.
    Separated(Vec<f32>),
}

/// The voice of `path`: its stems kept in `stems` (once its analysis has
/// them in their file, if it is writing them now:
/// [`crate::store::stems::open_waiting`]), else separated now in the
/// separator's turn, where they are looked for once more first (an
/// analysis queued before kept them meanwhile; not waited for there: one
/// queued after would wait for this turn).
pub fn voice(stems: Option<&Path>, path: &Path) -> Result<Voice, String> {
    crate::work::rhythm::register_networks();
    let kept = |open: fn(&Path, &Path) -> Option<jodel_playback::StemFile>| {
        stems.and_then(|dir| open(dir, path))
    };
    if let Some(file) = kept(crate::store::stems::open_waiting) {
        return Ok(Voice::Kept(Box::new(file)));
    }
    let name = crate::work::song_name(path);
    stages::run(&[Stage::Stems], &name, || {
        match kept(crate::store::stems::open) {
            Some(file) => Ok(Voice::Kept(Box::new(file))),
            None => Ok(Voice::Separated(separate_now(path, false)?.0)),
        }
    })
}

/// The voice in a track's kept stems, mono at 16 kHz: the vocals' two
/// channels averaged and resampled. Nothing is separated.
pub fn voice_16k(mut file: jodel_playback::StemFile) -> Result<Vec<f32>, String> {
    use jodel_playback::stems::Stem;
    let mut r =
        jodel_playback::Resampler::new(jodel_playback::stems::RATE, jodel_lyrics::VOCALS_RATE, 1);
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
/// the file decoded, separated into its stems on the GPU (in the stems'
/// turn: [`Stage::Stems`]), the vocals' two channels averaged.
pub fn separate_16k(path: &Path, mix: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
    crate::work::rhythm::register_networks();
    let name = crate::work::song_name(path);
    stages::run(&[Stage::Stems], &name, || separate_now(path, mix))
}

/// [`separate_16k`], now.
fn separate_now(path: &Path, mix: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
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
        let mut r = jodel_playback::Resampler::new(
            jodel_playback::stems::RATE,
            jodel_lyrics::VOCALS_RATE,
            1,
        );
        let mut out = Vec::new();
        r.process_f64(x, &mut out);
        r.finish_f64(&mut out);
        out.into_iter().map(|v| v as f32).collect::<Vec<f32>>()
    };
    Ok((to_16k(&voice), to_16k(&whole)))
}
