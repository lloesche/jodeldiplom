use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::Source;

use super::*;
use crate::config::Config;
use crate::core::karaoke::SongLyrics;
use crate::core::{update, Action, Side};

/// A queue of one song, "one two three / four five" on it, each word a
/// second from 10 s, half a second long; the Karaoke tab shown.
fn state() -> State {
    let mut l = Lyrics::parse_text("one two three\nfour five\n", Source::Lrclib);
    for (k, w) in l.lines.iter_mut().flat_map(|l| &mut l.words).enumerate() {
        (w.start, w.end, w.confidence) = (Some(10.0 + k as f64), Some(10.5 + k as f64), 0.9);
    }
    let mut s = State::new("/m".into(), Config::default());
    s.side = Side::Karaoke;
    let path = PathBuf::from("/m/a.flac");
    s.queue.paths = vec![path.clone()];
    s.karaoke.lyrics_wanted = vec![path.clone()];
    let l = Arc::new(l);
    s.karaoke.lyrics.insert(
        path,
        SongLyrics::Found {
            lyrics: Arc::clone(&l),
            shown: l,
        },
    );
    s
}

fn op(s: &mut State, op: Op) -> Vec<Effect> {
    update(s, Action::Op(op))
}

fn input(s: &mut State, input: Input) -> Vec<Effect> {
    update(s, Action::LyricsEdit(LyricsEdit::Screen(input)))
}

fn editor(s: &State) -> &LyricEditor {
    s.karaoke.editor.as_ref().expect("the editor is open")
}

fn times(s: &State, line: usize, word: usize) -> (f64, f64) {
    let w = &editor(s).lyrics.lines[line].words[word];
    (w.start.unwrap(), w.end.unwrap())
}

fn saves(fx: &[Effect]) -> usize {
    fx.iter()
        .filter(|e| matches!(e, Effect::SaveLyricEdits(..)))
        .count()
}

fn at(line: usize, word: usize) -> WordAt {
    WordAt { line, word }
}

/// `e` on the Karaoke tab opens the editor on its screen, the first word
/// selected a third of the way into the view; `esc` there closes it, back
/// to the tab; what leaves the screen does nothing there.
#[test]
fn the_editor_opens_on_its_screen_and_esc_closes_it() {
    let mut s = state();
    op(&mut s, Op::LyricsEditor);
    assert_eq!(s.screen, Screen::Lyrics);
    assert_eq!(editor(&s).selected, Some(at(0, 0)));
    let (a, b) = window(&s).unwrap();
    assert!((b - a - SPAN).abs() < 1e-9 && (10.0 - a - SPAN / 3.0).abs() < 1e-9);
    // The palette, the settings, the tabs: not from here.
    for o in [Op::Palette, Op::Settings, Op::ToggleSide, Op::Help] {
        op(&mut s, o);
        assert_eq!(
            (s.screen, s.palette.is_none()),
            (Screen::Lyrics, true),
            "{o:?}"
        );
    }
    op(&mut s, Op::Back);
    assert_eq!((s.screen, s.side), (Screen::Main, Side::Karaoke));
    assert!(s.karaoke.editor.is_none());
    // Without lyrics: it says so and stays shut.
    s.karaoke.lyrics.clear();
    op(&mut s, Op::LyricsEditor);
    assert_eq!(s.screen, Screen::Main);
    assert!(s.message.is_some());
}

/// A word held by its end and dragged: its end goes with the pointer, set
/// by hand; kept once, as one step of the history, when let go of.
#[test]
fn a_word_dragged_by_its_end_is_set_by_hand_in_one_step() {
    let mut s = state();
    op(&mut s, Op::LyricsEditor);
    let hold = Input::Hold {
        word: at(0, 1),
        part: Part::End,
        time: 11.5,
        slop: 0.01,
    };
    assert_eq!(saves(&input(&mut s, hold)), 0);
    assert_eq!(editor(&s).selected, Some(at(0, 1)));
    for t in [11.6, 11.7, 11.8] {
        assert_eq!(saves(&input(&mut s, Input::Drag(t))), 0);
    }
    assert!((times(&s, 0, 1).1 - 11.8).abs() < 1e-9);
    let fx = input(&mut s, Input::Drop);
    assert_eq!(saves(&fx), 1);
    let w = &editor(&s).lyrics.lines[0].words[1];
    assert!(w.pinned);
    assert!(editor(&s).screen.held.is_none());
    // One step back: as it was.
    op(&mut s, Op::LyricsUndo);
    assert_eq!(times(&s, 0, 1), (11.0, 11.5));
    assert!(!editor(&s).lyrics.lines[0].words[1].pinned);
    // Its middle moves all of it, its start only the start.
    let hold = |part, time| Input::Hold {
        word: at(0, 1),
        part,
        time,
        slop: 0.01,
    };
    input(&mut s, hold(Part::Middle, 11.2));
    input(&mut s, Input::Drag(11.0));
    input(&mut s, Input::Drop);
    assert!(near(times(&s, 0, 1), (10.8, 11.3)));
    input(&mut s, hold(Part::Start, 10.8));
    input(&mut s, Input::Drag(10.9));
    input(&mut s, Input::Drop);
    assert!(near(times(&s, 0, 1), (10.9, 11.3)));
}

fn near(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

/// A press that trembles within its slop changes nothing (no word set by
/// hand); a second click on the word soon after writes it; a press on the
/// lane while writing keeps what was written.
#[test]
fn a_click_selects_and_a_second_writes_the_word() {
    let mut s = state();
    op(&mut s, Op::LyricsEditor);
    let click = |s: &mut State, now: u64| {
        update(s, Action::Tick(now));
        input(
            s,
            Input::Hold {
                word: at(1, 0),
                part: Part::Middle,
                time: 13.2,
                slop: 0.05,
            },
        );
        input(s, Input::Drag(13.22));
        input(s, Input::Drop)
    };
    assert_eq!(saves(&click(&mut s, 1_000)), 0);
    assert!(!editor(&s).lyrics.lines[1].words[0].pinned);
    assert!(!writing(&s));
    click(&mut s, 1_000 + DOUBLE_CLICK_MS / 2);
    assert!(writing(&s));
    assert_eq!(
        editor(&s).screen.draft,
        Some(Draft {
            line: false,
            text: "four".into()
        })
    );
    input(&mut s, Input::Draft("for".into()));
    let fx = input(&mut s, Input::Seek(12.0));
    assert_eq!(saves(&fx), 1);
    assert!(!writing(&s));
    assert_eq!(editor(&s).lyrics.lines[1].words[0].text, "for");
}

/// The keys: the word after and the line below; its start and end nudged
/// and the whole word moved; not sung; the voice's level up from the
/// slider's and back; written and left or kept; undo and redo.
#[test]
fn the_keys_nudge_mark_level_write_and_undo() {
    let mut s = state();
    s.config.karaoke_voice = 100;
    op(&mut s, Op::LyricsEditor);
    op(&mut s, Op::LyricsNext);
    assert_eq!(editor(&s).selected, Some(at(0, 1)));
    op(&mut s, Op::Down);
    assert_eq!(editor(&s).selected, Some(at(1, 0)));
    op(&mut s, Op::Up);
    assert_eq!(editor(&s).selected, Some(at(0, 0)));
    op(&mut s, Op::LyricsNext);
    op(&mut s, Op::LyricsStartLater);
    op(&mut s, Op::LyricsEndEarlier);
    assert!(near(times(&s, 0, 1), (11.0 + NUDGE, 11.5 - NUDGE)));
    op(&mut s, Op::LyricsMoveLater);
    assert!(near(times(&s, 0, 1), (11.0 + 2.0 * NUDGE, 11.5)));
    assert!(editor(&s).word().unwrap().pinned);
    op(&mut s, Op::LyricsNotSung);
    assert!(editor(&s).word().unwrap().not_sung);
    // Not sung, the voice at the slider's top (0 dB): raised 6 dB, then a
    // step louder from there is the most.
    op(&mut s, Op::LyricsLouder);
    assert_eq!(editor(&s).word().unwrap().voice_db, Some(LEVEL_MOST));
    op(&mut s, Op::LyricsQuieter);
    assert_eq!(
        editor(&s).word().unwrap().voice_db,
        Some(LEVEL_MOST - LEVEL_STEP)
    );
    op(&mut s, Op::LyricsLevelReset);
    assert_eq!(editor(&s).word().unwrap().voice_db, None);
    // Written, then left: as it was; written again and kept.
    op(&mut s, Op::LyricsWrite);
    input(&mut s, Input::Draft("too".into()));
    op(&mut s, Op::Back);
    assert_eq!(editor(&s).word().unwrap().text, "two");
    assert_eq!(
        s.screen,
        Screen::Lyrics,
        "esc left the text, not the editor"
    );
    op(&mut s, Op::LyricsWriteLine);
    input(&mut s, Input::Draft("one too three".into()));
    op(&mut s, Op::Accept);
    assert_eq!(editor(&s).lyrics.lines[0].text(), "one too three");
    op(&mut s, Op::LyricsUndo);
    assert_eq!(editor(&s).lyrics.lines[0].text(), "one two three");
    op(&mut s, Op::LyricsRedo);
    assert_eq!(editor(&s).lyrics.lines[0].text(), "one too three");
    // The line written, its first word selected: taken out.
    op(&mut s, Op::LyricsDelete);
    assert_eq!(editor(&s).lyrics.lines[0].text(), "too three");
}

/// The wheel zooms round the time under the pointer; the secondary button
/// drags the view by the span over the lane's width a point.
#[test]
fn the_view_zooms_round_the_pointer_and_pans() {
    let mut s = state();
    op(&mut s, Op::LyricsEditor);
    let (a, b) = window(&s).unwrap();
    input(&mut s, Input::Zoom { steps: 1, at: 12.0 });
    let (c, d) = window(&s).unwrap();
    assert!(d - c < b - a);
    let share = |x: f64, (a, b): (f64, f64)| (x - a) / (b - a);
    assert!((share(12.0, (a, b)) - share(12.0, (c, d))).abs() < 1e-9);
    // Dragged left by a tenth of the lane: a tenth of the span later.
    input(
        &mut s,
        Input::PanStart {
            at: 500.0,
            width: 1000.0,
        },
    );
    input(&mut s, Input::Pan(400.0));
    input(&mut s, Input::PanEnd);
    let (e, _) = window(&s).unwrap();
    assert!((e - (c + 0.1 * (d - c))).abs() < 1e-6);
    // Not played: no playhead, nothing to seek.
    assert_eq!(playhead(&s), None);
    let fx = input(&mut s, Input::Seek(20.0));
    assert!(!fx.iter().any(|e| matches!(e, Effect::Engine(_))), "{fx:?}");
}
