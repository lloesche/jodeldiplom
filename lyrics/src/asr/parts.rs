//! A voice heard in parts: where it is cut, a loop in what was written cut
//! short, and the parts' words put together as one [`Transcript`]. The
//! same for the CPU's recogniser ([`super::Qwen3Asr`]) and a GPU's.

use super::text;

/// The longest part of a voice heard on its own, in seconds: the decoder
/// keeps to what it hears in a part this long, where over minutes it may
/// loop or stop early.
pub const PART_SECONDS: f64 = 30.0;
/// A part ends at the quietest moment of the voice after this many
/// seconds of it.
pub const PART_LEAST: f64 = 20.0;

/// What the model heard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// The language it named ("English"), empty when none.
    pub language: String,
    /// The words.
    pub text: String,
    /// The tokens it wrote.
    pub ids: Vec<u32>,
    /// The parts it was heard in, and their words.
    pub parts: Vec<Part>,
}

/// A stretch of a voice heard on its own, and what was heard in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Part {
    /// Where it starts and ends, in seconds.
    pub start: f64,
    /// Where it ends.
    pub end: f64,
    /// The words heard in it.
    pub text: String,
}

/// Where `voice` (16 kHz) is cut into parts: sample offsets, from 0 to its
/// length, at most [`PART_SECONDS`] apart, each cut at the quietest 0.2 s
/// from [`PART_LEAST`] on (so a word is rarely cut).
pub fn parts_of(voice: &[f32]) -> Vec<usize> {
    let rate = f64::from(crate::VOCALS_RATE);
    let most = (PART_SECONDS * rate) as usize;
    let least = (PART_LEAST * rate) as usize;
    let window = (0.2 * rate) as usize;
    let step = window / 4;
    let mut cuts = vec![0];
    let mut at = 0;
    while voice.len() - at > most {
        let mut best = (f64::MAX, at + most);
        let mut w = at + least;
        while w + window <= at + most {
            let energy: f64 = voice[w..w + window]
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum();
            if energy < best.0 {
                best = (energy, w + window / 2);
            }
            w += step;
        }
        at = best.1;
        cuts.push(at);
    }
    cuts.push(voice.len());
    cuts
}

/// Where the tokens have fallen into a loop, if they have: the last 32
/// all repeat a cycle of at most 4. Then the tokens to keep (up to two of
/// the cycles).
pub fn loop_start(ids: &[u32]) -> Option<usize> {
    const SPAN: usize = 32;
    if ids.len() < SPAN {
        return None;
    }
    let tail = &ids[ids.len() - SPAN..];
    (1..=4)
        .find(|&p| (p..SPAN).all(|i| tail[i] == tail[i - p]))
        .map(|p| ids.len() - SPAN + 2 * p)
}

/// The transcript of `voice` heard in parts ([`parts_of`]), `hear` giving
/// each part's tokens: the parts' words joined, the first language named.
/// The first error `hear` gives, if any.
pub fn transcribe_parts<E>(
    voice: &[f32],
    vocab: &text::Vocab,
    mut hear: impl FnMut(&[f32]) -> Result<Vec<u32>, E>,
) -> Result<Transcript, E> {
    let rate = f64::from(crate::VOCALS_RATE);
    let cuts = parts_of(voice);
    let mut out = Transcript::default();
    for pair in cuts.windows(2) {
        let ids = hear(&voice[pair[0]..pair[1]])?;
        let (language, words) = text::split_language(&vocab.decode(&ids));
        if out.language.is_empty() || out.language == "None" {
            out.language = language;
        }
        if !words.is_empty() {
            if !out.text.is_empty() {
                out.text.push(' ');
            }
            out.text.push_str(&words);
        }
        out.parts.push(Part {
            start: pair[0] as f64 / rate,
            end: pair[1] as f64 / rate,
            text: words,
        });
        out.ids.extend(ids);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_voice_is_cut_where_it_is_quiet() {
        let rate = 16_000;
        // 70 s of a tone, silent at 24 s and 52 s.
        let mut voice: Vec<f32> = (0..70 * rate)
            .map(|i| (i as f32 * 0.03).sin() * 0.5)
            .collect();
        for q in [24, 52] {
            for v in &mut voice[q * rate..q * rate + rate / 2] {
                *v = 0.0;
            }
        }
        let cuts = parts_of(&voice);
        assert_eq!(cuts.len(), 4, "{cuts:?}");
        assert_eq!((cuts[0], cuts[3]), (0, voice.len()));
        for (cut, quiet) in cuts[1..3].iter().zip([24, 52]) {
            let at = *cut as f64 / f64::from(rate as u32);
            assert!(at >= quiet as f64 && at <= quiet as f64 + 0.5, "{at}");
        }
        assert_eq!(parts_of(&voice[..20 * rate]), vec![0, 20 * rate]);
    }

    #[test]
    fn a_loop_is_cut_after_two_turns() {
        let mut ids: Vec<u32> = (0..10).collect();
        assert_eq!(loop_start(&ids), None);
        for _ in 0..16 {
            ids.extend([7, 8]);
        }
        assert_eq!(loop_start(&ids), Some(10 + 4));
        let chorus: Vec<u32> = (0..40).map(|i| i % 9).collect();
        assert_eq!(loop_start(&chorus), None, "a longer cycle is a chorus");
    }
}
