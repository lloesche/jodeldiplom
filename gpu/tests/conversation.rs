//! The language model going on in a conversation from what its cache
//! holds (`Gemma::go_on`), on the model itself: the second turn's answer
//! is the one the whole conversation written out and run from the start
//! gives, the first turn's prompt and answer not run again.

use std::ops::ControlFlow;
use std::path::PathBuf;

use jodel_gpu::{Gemma, Precision};

/// The tokens each answer may take: few, so that the two paths' roundings
/// cannot part (see the test).
const MOST: usize = 48;

/// `JODEL_GEMMA` names the model's file (Unsloth's
/// `gemma-4-E4B-it-Q4_K_M.gguf`); skipped where it is not there. A first
/// turn (its prompt longer than a chunk, so the second turn starts in the
/// middle of one), then a second gone on from the cache: its answer, token
/// for token, is the answer to the whole conversation written out (the
/// first prompt, the first answer, the second turn) and run from the
/// start, and only the second turn's tokens (and the last one written) ran
/// through the decoder for it. Going on from another length than the
/// cache holds, or with no conversation to go on (its room made again), is
/// an error. The two runs take the first answer's keys and values by
/// different paths (the steps', the prompt's chunks), so token for token
/// holds only for a short answer ([`MOST`]); at 8 bits the roundings part
/// after some 70 tokens. Unverified on Apple's GPUs (the f32 path): run it
/// on the Mac before calling it working there.
#[test]
#[ignore = "runs the language model: JODEL_GEMMA"]
fn a_conversation_goes_on_from_its_cache() {
    let Some(model) = std::env::var_os("JODEL_GEMMA").map(PathBuf::from) else {
        eprintln!("skipped: JODEL_GEMMA names the model's file");
        return;
    };
    let gemma = Gemma::open(&model, Precision::Fast).expect("the model");
    let tokens = gemma.tokens();
    let mut first = String::from("Here is a list of harbours and their lights:\n");
    for k in 0..60 {
        first.push_str(&format!(
            "Harbour {k} has a light that turns every {k} seconds.\n"
        ));
    }
    first.push_str("Name one harbour of the list in a short sentence.");
    let prompt = tokens.chat(&first);
    assert!(prompt.len() > 512, "{} tokens", prompt.len());
    gemma.reserve(4096).expect("its room");
    let all = |_: &[u32]| ControlFlow::Continue(());
    let answer = gemma.generate(&prompt, MOST, all).expect("an answer");
    let next = tokens.next_turn("Now name another one, and say how often its light turns.");
    let whole_first = [&prompt[..], &answer.tokens].concat();
    let after = (whole_first.len(), *whole_first.last().expect("tokens"));
    for other in [(after.0 + 1, after.1), (after.0, after.1 + 1)] {
        assert!(
            gemma.go_on(other, &next, MOST, all).is_err(),
            "another conversation"
        );
    }
    let went_on = gemma.go_on(after, &next, MOST, all).expect("going on");
    assert_eq!(went_on.prompt_tokens, next.len() + 1);
    let whole = [&prompt[..], &answer.tokens, &next].concat();
    let from_start = gemma.generate(&whole, MOST, all).expect("from the start");
    eprintln!(
        "first: {:?}\nwent on: {:?}\nfrom the start: {:?}",
        tokens.decode(&answer.tokens),
        tokens.decode(&went_on.tokens),
        tokens.decode(&from_start.tokens)
    );
    assert_eq!(went_on.tokens, from_start.tokens);
    gemma.reserve(4096).expect("its room again");
    assert!(
        gemma.go_on(after, &next, MOST, all).is_err(),
        "nothing to go on"
    );
}
