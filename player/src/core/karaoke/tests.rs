//! The count-in's tests: a number a second, and the title before it.

use super::*;

#[test]
fn a_count_in_is_a_number_a_second_then_the_song_from_its_start() {
    let c = Countdown::new(0);
    assert_eq!((c.beat, c.on), (1.0, 0.0));
    assert!((c.lead() - 3.0).abs() < 1e-12);
    // "3" as the silence starts, "2" a second later, "1" the last.
    assert_eq!(c.at(-3.0), Some((3, 0.0)));
    assert_eq!(c.at(-1.5).map(|(n, _)| n), Some(2));
    let (n, through) = c.at(-0.25).unwrap();
    assert_eq!(n, 1);
    assert!((through - 0.75).abs() < 1e-9);
    assert_eq!(c.at(0.0), None);
    assert!(c.over(0.0) && !c.over(-0.01));
}

/// After Next the title shows and fades, then the count runs, then
/// the singing: the music waits for all that where the singing starts
/// sooner, and plays at once (counted in over its intro) where not.
#[test]
fn the_title_fades_before_the_count_and_the_music_waits_for_it() {
    let title = TITLE_HOLD + TITLE_FADE;
    // Sung from the start: the title and the count in silence.
    let soon = Countdown::after_title(0, 0.0);
    assert!((soon.lead() - (title + 3.0)).abs() < 1e-12);
    // The title (from the lead's start) is gone as "3" comes.
    let starts = -soon.lead();
    assert_eq!(soon.at(starts + title - 0.01), None);
    assert_eq!(soon.at(starts + title).map(|(n, _)| n), Some(3));
    // Sung 2.5 s in: the music waits for what it lacks.
    let near = Countdown::after_title(0, 2.5);
    assert!((near.lead() - (title + 0.5)).abs() < 1e-12);
    // Sung 20 s in: no wait; the count over the intro.
    let far = Countdown::after_title(0, 20.0);
    assert_eq!(far.lead(), 0.0);
    assert_eq!(far.at(17.0).map(|(n, _)| n), Some(3));
}
