//! The passes' knobs as their units say.

use jodel_looks::pass::Unit;
use jodel_looks::passes::PASSES;

/// Every knob in turns spans a whole turn (0 to 1), so going round from
/// its top to its bottom is no jump: its number goes round, never held.
#[test]
fn every_turns_knob_spans_a_whole_turn() {
    let mut turns = 0;
    for pass in PASSES {
        for k in pass.knobs.iter().filter(|k| k.unit == Unit::Turns) {
            turns += 1;
            assert!(
                (k.lo, k.hi) == (0.0, 1.0),
                "{}'s {}: {} to {} turns",
                pass.name,
                k.name,
                k.lo,
                k.hi
            );
            assert!(
                k.wraps() && k.clamp(1.25) == 0.25,
                "{}'s {}",
                pass.name,
                k.name
            );
        }
    }
    assert!(turns > 30, "{turns}");
}
