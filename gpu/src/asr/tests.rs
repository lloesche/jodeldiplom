//! The recogniser's kernels: their source through naga's validation and
//! its Metal and HLSL writers (on any machine); and, on the model when it
//! is here (`JODEL_ASR_MODEL`, else the data folder's, skipped with a note
//! where it or a usable GPU is not), the fused step against the unfused
//! one and a piece past the room of pieces heard together heard again
//! alone.

use super::*;

#[path = "../../tests/common/mod.rs"]
mod common;

/// The kernels parse and validate, and Metal's and DirectX's writers take
/// them (as wgpu would, on a Mac and through DirectX 12).
#[test]
fn the_kernels_validate_and_write_for_metal_and_directx() {
    // unpack2x16float (the scales and offsets); no f16 arithmetic, no
    // subgroups.
    let entries: Vec<&str> = Kernel::ALL.iter().map(|k| k.entry()).collect();
    crate::check_source(
        &source(),
        naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        &entries,
    );
}

/// A fused step gives the unfused step's logits but for rounding (the
/// unfused attention is the prompt's, which sums otherwise), and the same
/// next token wherever the best logit is clear of the next.
#[test]
fn a_fused_step_is_the_unfused_one() {
    let Some(asr) = common::recogniser() else {
        return;
    };
    let pieces = common::pieces(3, jodel_lyrics::glimpses::GLIMPSE);
    let all: Vec<&[f32]> = pieces.iter().map(|p| &p[..]).collect();
    let features = mel::features_of(&all);
    let frames = features[0].1;
    let all: Vec<&[f32]> = features.iter().map(|f| &f.0[..]).collect();
    let mut inner = asr.lock().unwrap();
    let shape = Shape::new(all.len(), frames);
    let mut logits = Vec::new();
    for fused in [true, false] {
        inner.prompted(&all, &shape).unwrap();
        let mut ops = Vec::new();
        for k in 0..3 {
            let pos = shape.rows + k;
            if fused {
                inner.fused_ops(&mut ops, &shape, pos);
            } else {
                inner.unfused_ops(&mut ops, &shape, pos, 1);
            }
            inner.next_ops(&mut ops, &shape, (0, 1), pos + 1);
        }
        logits.push(
            inner
                .run(
                    &ops,
                    Some((Buf::Logits, 0, shape.pieces * VOCAB)),
                    Runs::Steps,
                )
                .unwrap(),
        );
    }
    let scale = logits[1].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = logits[0]
        .iter()
        .zip(&logits[1])
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("logits {scale:.2} at most, {worst:e} apart at most");
    assert!(worst <= 1e-4 * scale, "{worst} of {scale}");
    for (fused, unfused) in logits[0].chunks(VOCAB).zip(logits[1].chunks(VOCAB)) {
        let best = |v: &[f32]| {
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
            (order[0], v[order[0]] - v[order[1]])
        };
        let (a, margin) = best(unfused);
        if margin > 4.0 * worst {
            assert_eq!(best(fused).0, a);
        }
    }
}

/// With little room for pieces heard together, those that have not ended
/// in it are heard again alone, and every piece gets the tokens it gets
/// alone.
#[test]
fn a_piece_past_the_room_together_is_heard_again_alone() {
    let Some(asr) = common::recogniser() else {
        return;
    };
    let mut pieces = common::pieces(5, jodel_lyrics::glimpses::GLIMPSE);
    pieces.push(pieces[3][..30_000].to_vec());
    pieces.push(pieces[1][..30_000].to_vec());
    let all: Vec<&[f32]> = pieces.iter().map(|p| &p[..]).collect();
    let alone: Vec<Vec<u32>> = pieces.iter().map(|p| asr.tokens(p).unwrap()).collect();
    let longest = alone.iter().map(Vec::len).max().unwrap();
    assert!(longest >= 3, "the pieces wrote {longest} tokens at most");
    // Room for none: a read of four steps at most, then again alone.
    asr.lock().unwrap().new_at_once = 0;
    let features = mel::features_of(&all);
    let frames = features[0].1;
    let first: Vec<&[f32]> = features[..5].iter().map(|f| &f.0[..]).collect();
    let heard = {
        let mut inner = asr.lock().unwrap();
        inner.hear(&first, frames).unwrap()
    };
    assert!(
        heard.iter().any(Option::is_none),
        "every piece ended in the room: {heard:?}"
    );
    let together = asr.tokens_of(&all).unwrap();
    assert_eq!(together, alone);
}
