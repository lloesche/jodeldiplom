//! The buffer and the control timeline together, as the stem lanes use
//! them: whatever the stems do (come in pieces, stall, start again for a
//! jump, end, are given back, start afresh after a gap), the parts promised
//! are always there, and the words keep the promise.

use jodel_output::lanes::Parts;

use super::buffer::Buffer;
use super::control::{Supply, Timeline};

const E: u64 = 10;
/// Frames made past a write at the least (the longest seam's crossfade).
const MARGIN: u64 = 30;
/// Frames kept behind what is written, and the most given back.
const BACK: u64 = 60;

/// Whether `words` keep the promise: no `None` right after `Present`, and
/// at least `E` frames of `Ending` before a `None` after one.
fn kept(words: &[Parts]) -> Result<(), String> {
    let mut run = 0;
    for (i, w) in words.iter().enumerate() {
        match (i.checked_sub(1).map(|j| words[j]), w) {
            (Some(Parts::Present), Parts::None) => {
                return Err(format!("None after Present at {i}"))
            }
            (Some(Parts::Ending), Parts::None) if run < E => {
                return Err(format!("None after {run} frames of Ending at {i}"))
            }
            _ => {}
        }
        run = if *w == Parts::Ending { run + 1 } else { 0 };
    }
    Ok(())
}

/// The stream makes its frames, a few at a time, until the buffer reaches
/// `to` (none while it is held up).
fn make(b: &mut Buffer, stream: &mut Option<i64>, to: u64, stalled: bool) {
    while let Some(at) = stream.as_mut().filter(|_| !stalled) {
        if b.end() >= to {
            break;
        }
        let k = 1 + (at.unsigned_abs() % 7);
        let piece: Vec<i32> = (0..k as i64).map(|i| (*at + i) as i32).collect();
        b.push(*at, &piece);
        *at += k as i64;
    }
}

#[test]
fn the_parts_promised_are_always_there() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |m: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % m.max(1)
    };
    let (mut seams, mut ends) = (0, 0);
    for _ in 0..100 {
        let mut t = Timeline::new(E);
        let mut b = Buffer::new(1);
        let mut words: Vec<Parts> = Vec::new();
        let mut from = 0u64;
        // The stream's next frame, and whether it is held up.
        let mut stream: Option<i64> = Some(0);
        let mut stalled = false;
        // The furthest written: the output gives back no more than it holds,
        // what was written within BACK of that.
        let mut high = 0u64;
        for _ in 0..400 {
            high = high.max(from);
            match next(100) {
                0..=4 => stalled = !stalled,
                5..=9 => {
                    // A joined start: a run-up before the frame written
                    // next, crossfaded into what is there.
                    let fade = next(MARGIN + 1) as usize;
                    let keep = t.promised(from).max(from + fade as u64);
                    b.restart(from, keep, fade);
                    // From no later than the parts kept end, as the lanes
                    // start them.
                    let mut first = from as i64 - next(20) as i64;
                    if !b.is_empty() && b.end() < from && b.end() + BACK >= from {
                        first = first.min(b.end() as i64);
                    }
                    stream = Some(first);
                    seams += 1;
                }
                10..=12 => {
                    // A gap (a seek's flush): afresh from here.
                    kept(&words).unwrap_or_else(|e| panic!("{e}: {words:?}"));
                    words.clear();
                    t.clear(from);
                    b.clear(from);
                    stream = Some(from as i64);
                }
                13..=15 => {
                    // No more parts: what is promised stays.
                    b.restart(from, t.promised(from), 0);
                    stream = None;
                    ends += 1;
                }
                16..=20 if !words.is_empty() && from + BACK > high => {
                    let most = (from + BACK - high).min(words.len() as u64);
                    let back = next(most) + 1;
                    from -= back;
                    words.truncate(words.len() - back as usize);
                    t.given_back(from);
                }
                _ => {
                    let n = next(16) + 1;
                    let absent = t.state(from) == super::control::State::Absent;
                    if !absent {
                        make(&mut b, &mut stream, from + n + MARGIN, stalled);
                    }
                    let there = b.there(from);
                    assert!(
                        there >= t.promised(from),
                        "promised to {} but there to {there} (from {from})",
                        t.promised(from)
                    );
                    let s = Supply {
                        there,
                        coming: stream.is_some(),
                        hold: 40,
                        through: false,
                        main_end: None,
                    };
                    let plan = t.plan(from, n, s);
                    let took = next(n + 1);
                    for o in from..from + took {
                        let w = plan.word(o);
                        assert!(w == Parts::None || o < there, "a word with no parts at {o}");
                    }
                    t.commit(&plan, from, took);
                    words.extend((from..from + took).map(|o| plan.word(o)));
                    from += took;
                    // After the write, the rest made toward what is held
                    // ahead.
                    make(&mut b, &mut stream, from + 50, stalled);
                    let behind = from.saturating_sub(BACK);
                    b.trim(behind);
                    t.trim(behind);
                    kept(&words).unwrap_or_else(|e| panic!("{e}: {words:?}"));
                }
            }
        }
    }
    assert!(seams > 1_000 && ends > 500, "{seams} seams, {ends} ends");
}
