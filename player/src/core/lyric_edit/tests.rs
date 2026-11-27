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
    assert!(!e.apply(Edit::Select(Selection::Previous)));
    assert_eq!(e.selected, Some(at(0, 0)));
    for _ in 0..3 {
        e.apply(Edit::Select(Selection::Next));
    }
    assert_eq!(e.selected, Some(at(1, 0)));
    e.apply(Edit::Select(Selection::Word(at(1, 1))));
    e.apply(Edit::Select(Selection::Next));
    assert_eq!(e.selected, Some(at(1, 1)));
    // A word that is not there is not selected.
    e.apply(Edit::Select(Selection::Word(at(5, 0))));
    assert_eq!(e.selected, Some(at(1, 1)));
    assert!(!e.can_undo());
}

#[test]
fn a_word_moved_and_nudged_is_set_by_hand_and_its_neighbours_make_room() {
    let mut e = editor();
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    // "two" moved 0.8 s later, over "three" (2.0 to 2.5).
    assert!(e.apply(Edit::Change(Change::Move(0.8))));
    assert!(near(times(&e, at(0, 1)), (1.8, 2.3)));
    let two = e.word().unwrap();
    assert!(two.edit.pin.is_some() && two.confidence == jodel_lyrics::GIVEN);
    assert_eq!(times(&e, at(0, 2)), (times(&e, at(0, 1)).1, 2.5));
    // Made longer at its end, shorter at its start.
    assert!(e.apply(Edit::Change(Change::NudgeEnd(0.1))));
    assert!(e.apply(Edit::Change(Change::NudgeStart(0.05))));
    assert!(near(times(&e, at(0, 1)), (1.85, 2.4)));
    // Never shorter than the shortest.
    e.apply(Edit::Change(Change::NudgeStart(5.0)));
    assert!(near(times(&e, at(0, 1)), (2.4 - SHORTEST, 2.4)));
    // Never across a word set by hand: "three" set, then "two" pushed
    // into it stops at its start.
    e.apply(Edit::Select(Selection::Word(at(0, 2))));
    assert!(e.apply(Edit::Change(Change::SetTimes(2.6, 2.9))));
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    e.apply(Edit::Change(Change::Move(10.0)));
    assert!(near(times(&e, at(0, 1)), (2.6 - SHORTEST, 2.6)));
    // Its letters stay inside it.
    let two = e.word().unwrap();
    assert!(two.letters.iter().all(|&t| f64::from(t) <= SHORTEST + 1e-6));
}

#[test]
fn marks_and_the_voices_level_are_set_and_cleared() {
    let mut e = editor();
    e.apply(Edit::Select(Selection::Word(at(1, 0))));
    assert!(e.apply(Edit::Change(Change::NotSung(true))));
    assert!(e.apply(Edit::Change(Change::Voice(Some(-3.0)))));
    let w = e.word().unwrap();
    assert!(w.edit.not_sung && w.edit.voice_db == Some(-3.0));
    // Setting what is already set is no change.
    assert!(!e.apply(Edit::Change(Change::NotSung(true))));
    assert!(e.apply(Edit::Change(Change::Voice(None))));
    assert_eq!(e.word().unwrap().edit.voice_db, None);
}

#[test]
fn the_text_is_edited_word_by_word_and_line_by_line() {
    let mut e = editor();
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    // A word replaced keeps its times; by two words, the second put in
    // after it.
    assert!(e.apply(Edit::Change(Change::ReplaceWord("too".into()))));
    assert_eq!(times(&e, at(0, 1)), (1.0, 1.5));
    assert!(e.apply(Edit::Change(Change::ReplaceWord("too (oh)".into()))));
    assert_eq!(text(&e), ["one too (oh) three", "four five"]);
    assert!(e.lyrics.lines[0].words[2].backing);
    // Put in between its neighbours: timed between them.
    let oh = times(&e, at(0, 2));
    assert!(oh.0 >= 1.5 && oh.1 <= 2.0, "{oh:?}");
    e.apply(Edit::Select(Selection::Word(at(0, 0))));
    assert!(e.apply(Edit::Change(Change::InsertWord {
        after: false,
        text: "and".into()
    })));
    assert_eq!(e.selected, Some(at(0, 0)));
    assert_eq!(text(&e)[0], "and one too (oh) three");
    assert!(e.apply(Edit::Change(Change::DeleteWord)));
    assert_eq!(text(&e)[0], "one too (oh) three");
    // A line replaced by as many words keeps their times; by others, they
    // are spread over its time.
    e.apply(Edit::Select(Selection::Word(at(1, 1))));
    assert!(e.apply(Edit::Change(Change::ReplaceLine("for fife".into()))));
    assert_eq!(times(&e, at(1, 1)), (4.0, 4.5));
    assert!(e.apply(Edit::Change(Change::ReplaceLine("four five six".into()))));
    assert_eq!(times(&e, at(1, 0)).0, 3.0);
    assert_eq!(times(&e, at(1, 2)).1, 4.5);
    assert!(e.apply(Edit::Change(Change::InsertLine {
        after: true,
        text: "seven".into()
    })));
    assert_eq!(text(&e), ["one too (oh) three", "four five six", "seven"]);
    assert_eq!(e.selected, Some(at(2, 0)));
    assert!(e.apply(Edit::Change(Change::DeleteLine)));
    assert_eq!(e.lyrics.lines.len(), 2);
    // A word replaced by nothing is taken out; the last of its line takes
    // the line with it.
    e.apply(Edit::Select(Selection::Word(at(1, 0))));
    for _ in 0..3 {
        assert!(e.apply(Edit::Change(Change::ReplaceWord(" ".into()))));
    }
    assert_eq!(text(&e), ["one too (oh) three"]);
}

#[test]
fn every_change_is_undone_and_redone_until_a_new_one() {
    let mut e = editor();
    let start = e.lyrics.clone();
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    e.apply(Edit::Change(Change::Move(0.25)));
    let moved = e.lyrics.clone();
    e.apply(Edit::Change(Change::NotSung(true)));
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
    e.apply(Edit::Change(Change::Voice(Some(0.0))));
    assert!(!e.can_redo() && e.can_undo());
    // Undoing a deleted line's change keeps a word selected.
    e.apply(Edit::Select(Selection::Word(at(1, 1))));
    e.apply(Edit::Change(Change::DeleteLine));
    e.apply(Edit::Undo);
    assert!(e.word().is_some());
}

#[test]
fn the_song_timed_again_with_the_same_edits_is_taken() {
    let mut e = editor();
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    e.apply(Edit::Change(Change::Move(0.8)));
    let mut timed = e.lyrics.clone();
    timed.lines[0].words[2].start = Some(2.35);
    assert!(e.timed_again(&timed));
    assert_eq!(times(&e, at(0, 2)).0, 2.35);
    // Other edits (made since): not taken.
    let mut other = timed.clone();
    other.lines[0].words[0].edit.not_sung = true;
    assert!(!e.timed_again(&other));
}

/// A drag that leads nowhere the word may go (an untimed word) changes
/// nothing: the lyrics are as the drag found them.
#[test]
fn a_drag_that_cannot_be_made_leaves_the_lyrics_as_found() {
    let mut e = editor();
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    let from = e.lyrics.clone();
    assert!(e.drag(&from, Change::Move(0.25)));
    assert_ne!(e.lyrics, from);
    let mut untimed = from.clone();
    untimed[at(0, 1)].clear_times();
    e.lyrics = untimed.clone();
    assert!(!e.drag(&untimed, Change::Move(0.25)));
    assert_eq!(e.lyrics, untimed);
}

/// A word made longer over the next one: the next starts where it ends,
/// its letters kept where they were sung (shifted with its start), within
/// it.
#[test]
fn the_word_after_keeps_its_letters_where_they_were_sung() {
    let mut e = editor();
    e.lyrics[at(0, 2)].letters = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    e.apply(Edit::Select(Selection::Word(at(0, 1))));
    // "two" (1.0 to 1.5) to 2.2: "three" (2.0 to 2.5) from 2.2.
    assert!(e.apply(Edit::Change(Change::NudgeEnd(0.7))));
    let three = &e.lyrics[at(0, 2)];
    assert!(near(times(&e, at(0, 2)), (2.2, 2.5)));
    let letters: Vec<f64> = three.letters.iter().map(|&t| f64::from(t)).collect();
    for (got, want) in letters.iter().zip([0.0, 0.0, 0.0, 0.1, 0.2]) {
        assert!((got - want).abs() < 1e-6, "{letters:?}");
    }
}
