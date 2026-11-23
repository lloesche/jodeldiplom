//! The speech recogniser's pieces heard together against each heard on
//! its own: the same tokens. On the model when it is here
//! (`JODEL_ASR_MODEL`, else the data folder's) and a voice when one is
//! given (`JODEL_ASR_VOICE`), else sung vowels; skipped with a note where
//! the model or a usable GPU is not.

mod common;

use jodel_gpu::GpuAsr;

/// Glimpses heard together, and a shorter piece among them, get each the
/// tokens it gets alone.
#[test]
fn pieces_heard_together_get_the_tokens_they_get_alone() {
    let Some(asr) = common::recogniser() else {
        return;
    };
    let mut pieces = common::pieces(6, jodel_lyrics::glimpses::GLIMPSE);
    pieces.insert(2, pieces[4][..20_000].to_vec());
    let all: Vec<&[f32]> = pieces.iter().map(|p| &p[..]).collect();
    let together = asr.tokens_of(&all).unwrap();
    // Each wrote a few tokens (its language at least): the decoder ran.
    assert!(together.iter().all(|t| t.len() >= 3), "{together:?}");
    for (piece, heard) in pieces.iter().zip(&together) {
        assert_eq!(&asr.tokens(piece).unwrap(), heard);
    }
    eprintln!("{together:?}");
}

/// A voice's glimpses heard together are each as its transcript alone.
#[test]
fn glimpses_are_heard_as_each_alone() {
    let Some(asr) = common::recogniser() else {
        return;
    };
    let voice: Vec<f32> = common::pieces(4, 2.0).concat();
    // The first glimpse writes a few tokens (its language at least).
    assert!(asr.tokens(&voice[..48_000]).unwrap().len() >= 3);
    let together = asr.glimpses(&voice).unwrap();
    let alone =
        jodel_lyrics::glimpses::hear_glimpses(&voice, |piece| Ok(asr.transcribe(piece)?.text))
            .unwrap();
    assert_eq!(together, alone);
}
