//! The language model past 32K positions, on the model itself: a word
//! given at the start of a prompt of 33 000 tokens is found at its end.
//! Only the global layers see that far (the sliding layers' rings have
//! gone round 33 times by then), so it holds the global layers' keys and
//! values, each in a buffer of its own, over the whole context.

use std::path::PathBuf;

use jodel_gpu::{Gemma, Precision};

/// The prompt's tokens at the least.
const PAST: usize = 33_000;

/// `JODEL_GEMMA` names the model's file (Unsloth's
/// `gemma-4-E4B-it-Q4_K_M.gguf`); skipped where it is not there. The word
/// comes first, then plain numbered lines past [`PAST`] tokens, then the
/// question; the greedy answer names the word. The room held is what
/// `Gemma::planned` said, but for the prompt's numbers (recorded as it
/// runs).
#[test]
#[ignore = "runs the language model: JODEL_GEMMA"]
fn a_word_from_33k_tokens_back_is_found() {
    let Some(model) = std::env::var_os("JODEL_GEMMA").map(PathBuf::from) else {
        eprintln!("skipped: JODEL_GEMMA names the model's file");
        return;
    };
    let gemma = Gemma::open(&model, Precision::Fast).expect("the model");
    let line = |k: usize| format!("Line {k}: the harbour lights were on and the boats came in.\n");
    let each = gemma.tokens().encode(&line(1000)).len();
    let mut text = String::from("Remember this: the password is heliotrope.\n\n");
    // A tenth more lines than the 1000th's tokens say (the early ones are
    // shorter).
    for k in 0..PAST * 11 / 10 / each {
        text.push_str(&line(k));
    }
    text.push_str("\nWhat is the password? Answer with the word alone.");
    let ids = gemma.tokens().chat(&text);
    assert!(ids.len() > PAST, "{} tokens", ids.len());
    let most = 16;
    let context = ids.len() + most;
    gemma.reserve(context).expect("its room");
    let planned = Gemma::planned(&model, context).expect("its plan");
    let held = gemma.held();
    assert_eq!(held.resident, planned.resident);
    assert!(held.working <= planned.working && held.working * 101 / 100 >= planned.working);
    let answer = gemma.generate_text(&ids, most, |_| {}).expect("an answer");
    eprintln!(
        "{} prompt tokens in {:.1} s: {:?}",
        ids.len(),
        answer.generated.prompt_time.as_secs_f64(),
        answer.text
    );
    assert!(
        answer.text.to_lowercase().contains("heliotrope"),
        "{:?}",
        answer.text
    );
}
