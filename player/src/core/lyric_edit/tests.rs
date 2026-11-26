use super::*;
use jodel_lyrics::Source;

/// "one two three / four five", each word a second from 0 s, half a second
/// long.
fn song() -> Lyrics {
    let mut l = Lyrics::parse_text("one two three\nfour five\n", Source::Lrclib);
    for (k, w) in l.lines.iter_mut().flat_map(|l| &mut l.words).enumerate() {
        (w.start, w.end, w.confidence) = (Some(k as f64), Some(k as f64 + 0.5), 0.9);
    }
    l
}

fn editor() -> LyricEditor {
    LyricEditor::open(PathBuf::from("/music/a.flac"), &song())
}

fn at(line: usize, word: usize) -> WordAt {
    WordAt { line, word }
}

fn times(e: &LyricEditor, a: WordAt) -> (f64, f64) {
    let w = &e.lyrics.lines[a.line].words[a.word];
    (w.start.unwrap(), w.end.unwrap())
}

fn text(e: &LyricEditor) -> Vec<String> {
    e.lyrics.lines.iter().map(Line::text).collect()
}

fn near(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

#[test]
fn the_selection_goes_over_lines_and_stops_at_the_ends() {
    let mut e = editor();
    assert_eq!(e.selected, Some(at(0, 0)));
    assert!(!e.apply(Edit::Previous));
    assert_eq!(e.selected, Some(at(0, 0)));
    for _ in 0..3 {
        e.apply(Edit::Next);
    }
    assert_eq!(e.selected, Some(at(1, 0)));
    e.apply(Edit::Select(at(1, 1)));
    e.apply(Edit::Next);
    assert_eq!(e.selected, Some(at(1, 1)));
    // A word that is not there is not selected.
    e.apply(Edit::Select(at(5, 0)));
    assert_eq!(e.selected, Some(at(1, 1)));
    assert!(!e.can_undo());
}

#[test]
fn a_word_moved_and_nudged_is_set_by_hand_and_its_neighbours_make_room() {
    let mut e = editor();
    e.apply(Edit::Select(at(0, 1)));
    // "two" moved 0.8 s later, over "three" (2.0 to 2.5).
    assert!(e.apply(Edit::Move(0.8)));
    assert!(near(times(&e, at(0, 1)), (1.8, 2.3)));
    let two = e.word().unwrap();
    assert!(two.pinned && two.confidence == GIVEN);
    assert_eq!(times(&e, at(0, 2)), (times(&e, at(0, 1)).1, 2.5));
    // Made longer at its end, shorter at its start.
    assert!(e.apply(Edit::NudgeEnd(0.1)));
    assert!(e.apply(Edit::NudgeStart(0.05)));
    assert!(near(times(&e, at(0, 1)), (1.85, 2.4)));
    // Never shorter than the shortest.
    e.apply(Edit::NudgeStart(5.0));
    assert!(near(times(&e, at(0, 1)), (2.4 - SHORTEST, 2.4)));
    // Never across a word set by hand: "three" set, then "two" pushed
    // into it stops at its start.
    e.apply(Edit::Select(at(0, 2)));
    assert!(e.apply(Edit::SetTimes(2.6, 2.9)));
    e.apply(Edit::Select(at(0, 1)));
    e.apply(Edit::Move(10.0));
    assert!(near(times(&e, at(0, 1)), (2.6 - SHORTEST, 2.6)));
    // Its letters stay inside it.
    let two = e.word().unwrap();
    assert!(two.letters.iter().all(|&t| f64::from(t) <= SHORTEST + 1e-6));
}

#[test]
fn marks_and_the_voices_level_are_set_and_cleared() {
    let mut e = editor();
    e.apply(Edit::Select(at(1, 0)));
    assert!(e.apply(Edit::NotSung(true)));
    assert!(e.apply(Edit::Voice(Some(-3.0))));
    let w = e.word().unwrap();
    assert!(w.not_sung && w.voice_db == Some(-3.0));
    // Setting what is already set is no change.
    assert!(!e.apply(Edit::NotSung(true)));
    assert!(e.apply(Edit::Voice(None)));
    assert_eq!(e.word().unwrap().voice_db, None);
}

#[test]
fn the_text_is_edited_word_by_word_and_line_by_line() {
    let mut e = editor();
    e.apply(Edit::Select(at(0, 1)));
    // A word replaced keeps its times; by two words, the second put in
    // after it.
    assert!(e.apply(Edit::ReplaceWord("too".into())));
    assert_eq!(times(&e, at(0, 1)), (1.0, 1.5));
    assert!(e.apply(Edit::ReplaceWord("too (oh)".into())));
    assert_eq!(text(&e), ["one too (oh) three", "four five"]);
    assert!(e.lyrics.lines[0].words[2].backing);
    // Put in between its neighbours: timed between them.
    let oh = times(&e, at(0, 2));
    assert!(oh.0 >= 1.5 && oh.1 <= 2.0, "{oh:?}");
    e.apply(Edit::Select(at(0, 0)));
    assert!(e.apply(Edit::InsertWord {
        after: false,
        text: "and".into()
    }));
    assert_eq!(e.selected, Some(at(0, 0)));
    assert_eq!(text(&e)[0], "and one too (oh) three");
    assert!(e.apply(Edit::DeleteWord));
    assert_eq!(text(&e)[0], "one too (oh) three");
    // A line replaced by as many words keeps their times; by others, they
    // are spread over its time.
    e.apply(Edit::Select(at(1, 1)));
    assert!(e.apply(Edit::ReplaceLine("for fife".into())));
    assert_eq!(times(&e, at(1, 1)), (4.0, 4.5));
    assert!(e.apply(Edit::ReplaceLine("four five six".into())));
    assert_eq!(times(&e, at(1, 0)).0, 3.0);
    assert_eq!(times(&e, at(1, 2)).1, 4.5);
    assert!(e.apply(Edit::InsertLine {
        after: true,
        text: "seven".into()
    }));
    assert_eq!(text(&e), ["one too (oh) three", "four five six", "seven"]);
    assert_eq!(e.selected, Some(at(2, 0)));
    assert!(e.apply(Edit::DeleteLine));
    assert_eq!(e.lyrics.lines.len(), 2);
    // A word replaced by nothing is taken out; the last of its line takes
    // the line with it.
    e.apply(Edit::Select(at(1, 0)));
    for _ in 0..3 {
        assert!(e.apply(Edit::ReplaceWord(" ".into())));
    }
    assert_eq!(text(&e), ["one too (oh) three"]);
}

#[test]
fn every_change_is_undone_and_redone_until_a_new_one() {
    let mut e = editor();
    let start = e.lyrics.clone();
    e.apply(Edit::Select(at(0, 1)));
    e.apply(Edit::Move(0.25));
    let moved = e.lyrics.clone();
    e.apply(Edit::NotSung(true));
    let marked = e.lyrics.clone();
    assert!(e.apply(Edit::Undo));
    assert_eq!(e.lyrics, moved);
    assert!(e.apply(Edit::Undo));
    assert_eq!(e.lyrics, start);
    assert!(!e.apply(Edit::Undo));
    assert!(e.apply(Edit::Redo));
    assert_eq!(e.lyrics, moved);
    assert!(e.apply(Edit::Redo));
    assert_eq!(e.lyrics, marked);
    assert!(!e.apply(Edit::Redo));
    // A new change after an undo: nothing left to redo.
    e.apply(Edit::Undo);
    e.apply(Edit::Voice(Some(0.0)));
    assert!(!e.can_redo() && e.can_undo());
    // Undoing a deleted line's change keeps a word selected.
    e.apply(Edit::Select(at(1, 1)));
    e.apply(Edit::DeleteLine);
    e.apply(Edit::Undo);
    assert!(e.word().is_some());
}

#[test]
fn the_song_timed_again_with_the_same_edits_is_taken() {
    let mut e = editor();
    e.apply(Edit::Select(at(0, 1)));
    e.apply(Edit::Move(0.8));
    let mut timed = e.lyrics.clone();
    timed.lines[0].words[2].start = Some(2.35);
    assert!(e.timed_again(&timed));
    assert_eq!(times(&e, at(0, 2)).0, 2.35);
    // Other edits (made since): not taken.
    let mut other = timed.clone();
    other.lines[0].words[0].not_sung = true;
    assert!(!e.timed_again(&other));
}
