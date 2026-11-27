//! A view of a stretch of a song's time, as the Transition tab's blocks
//! and the lyrics editor show one ([`TimeView`]): held where it was put,
//! or following the playhead (leading it, [`TimeView::leading`], or a page
//! at a time, [`TimeView::paged`]); zoomed round a time
//! ([`zoomed_start`]), a time brought into it ([`TimeView::lead_to`]),
//! dragged by the pointer ([`Pan`], the drag of either view: [`PanOf`]),
//! and zoomed to fit the lyrics' words ([`fit_lyrics`], measuring them by
//! [`crate::core::lanes::words_left_out`] or [`words_cut`]).
//!
//! A view's span is its owner's: the editor keeps its own, the Transition
//! tab's comes from its zoom (bars of the track).

use jodel_lyrics::Word;

use crate::core::lanes::Block;

/// The share of the span shown before the playhead while a view leads it,
/// and before a time brought into view.
pub const LEAD: f64 = 1.0 / 3.0;

/// Where a view of a song's time is, and the drag of it under way.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimeView {
    place: Place,
    pan: Option<Pan>,
}

/// Where a view is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    /// Held with this many seconds at its left edge.
    Held(f64),
    /// Following the playhead; a page at a time, its pages counted from
    /// `from` seconds.
    Following {
        /// Where its pages count from.
        from: f64,
    },
}

impl Default for Place {
    /// Following the playhead.
    fn default() -> Place {
        Place::Following { from: 0.0 }
    }
}

impl TimeView {
    /// A view held with `start` at its left edge.
    pub fn held(start: f64) -> TimeView {
        TimeView {
            place: Place::Held(start),
            pan: None,
        }
    }

    /// Whether it follows the playhead.
    pub fn is_following(&self) -> bool {
        matches!(self.place, Place::Following { .. })
    }

    /// Where it was put: its left edge held, or where its pages count from.
    fn anchor(&self) -> f64 {
        match self.place {
            Place::Held(start) | Place::Following { from: start } => start,
        }
    }

    /// Its left edge while it leads the playhead, `span` across: held,
    /// where it was put; following, the playhead [`LEAD`] of the way in.
    /// `None` following a playhead there is not.
    pub fn leading(&self, span: f64, playhead: Option<f64>) -> Option<f64> {
        match self.place {
            Place::Held(start) => Some(start),
            Place::Following { .. } => playhead.map(|p| p - LEAD * span),
        }
    }

    /// Its left edge while it follows the playhead a page at a time, `span`
    /// across: held, where it was put; following, a page on (`share` of
    /// the span) each time the playhead gets that far across, from where
    /// its pages count (there while the song is not played).
    pub fn paged(&self, span: f64, playhead: Option<f64>, share: f64) -> f64 {
        match (self.place, playhead) {
            (Place::Following { from }, Some(p)) => {
                let page = share * span;
                from + ((p - from) / page).floor() * page
            }
            _ => self.anchor(),
        }
    }

    /// Held with `start` at its left edge.
    pub fn hold(&mut self, start: f64) {
        self.place = Place::Held(start);
    }

    /// Following the playhead again; by pages, counted from where it was
    /// put.
    pub fn follow(&mut self) {
        self.follow_from(self.anchor());
    }

    /// Following the playhead again; by pages, counted from `from`.
    pub fn follow_from(&mut self, from: f64) {
        self.place = Place::Following { from };
    }

    /// Put at `start` as it is: held there, or following with its pages
    /// counted from there.
    pub fn move_to(&mut self, start: f64) {
        self.place = match self.place {
            Place::Held(_) => Place::Held(start),
            Place::Following { .. } => Place::Following { from: start },
        };
    }

    /// Held with `time` [`LEAD`] of the way into a view `span` across.
    pub fn lead_to(&mut self, time: f64, span: f64) {
        self.hold(lead_start(time, span));
    }

    /// A drag of it starts, the pointer `at` across, its left edge `start`
    /// seconds, a unit across worth `per_unit` seconds ([`Pan`]); it is held
    /// once the pointer moves.
    pub fn start_pan(&mut self, at: f32, start: f64, per_unit: f64) {
        self.pan = Some(Pan {
            at,
            start,
            per_unit,
        });
    }

    /// The pointer dragging it is `at` across: held where that puts it.
    pub fn pan_to(&mut self, at: f32) {
        if let Some(pan) = self.pan {
            self.hold(pan.start_at(at));
        }
    }

    /// Its drag ends.
    pub fn end_pan(&mut self) {
        self.pan = None;
    }

    /// Whether it is dragged.
    pub fn panning(&self) -> bool {
        self.pan.is_some()
    }
}

/// The left edge of a view `span` across with `time` [`LEAD`] of the way in
/// (not before the song).
pub fn lead_start(time: f64, span: f64) -> f64 {
    (time - LEAD * span).max(0.0)
}

/// The left edge of a view `span` across with `time` in its middle (not
/// before the song).
pub fn centred_start(time: f64, span: f64) -> f64 {
    (time - 0.5 * span).max(0.0)
}

/// The left edge of a view zoomed to `span` across from the one showing
/// `window` (from, to seconds), `at` (seconds) staying where it was across
/// it; `outside` of the way across where it was not in view.
pub fn zoomed_start((start, end): (f64, f64), span: f64, at: f64, outside: f64) -> f64 {
    let share = if end > start && (start..=end).contains(&at) {
        (at - start) / (end - start)
    } else {
        outside
    };
    (at - share * span).max(0.0)
}

/// A drag of a view with the secondary button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pan {
    /// Where across the pointer was pressed, in the window's units.
    pub at: f32,
    /// The view's left edge then, in seconds.
    pub start: f64,
    /// Seconds a unit across is worth (negative: the view follows the
    /// pointer, as over the whole track; positive: the waveform does).
    pub per_unit: f64,
}

impl Pan {
    /// The view's left edge with the pointer `at` across: dragged right,
    /// the waveform goes right, to earlier times.
    pub fn start_at(&self, at: f32) -> f64 {
        let moved = f64::from(at) - f64::from(self.at);
        (self.start - moved * self.per_unit).max(0.0)
    }
}

/// Which view the secondary button drags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanOf {
    /// A Transition block's, by its lanes or (`overview`) its whole track.
    Block {
        /// The block.
        block: Block,
        /// Over the whole track.
        overview: bool,
    },
    /// The lyrics editor's.
    Lyrics,
}

/// Whether the view `of` is dragged.
pub fn panning(state: &crate::core::State, of: PanOf) -> bool {
    match of {
        PanOf::Block { block, .. } => crate::core::transition::tab::block(state, block)
            .view
            .panning(),
        PanOf::Lyrics => {
            crate::core::lyric_edit::editor(state).is_some_and(|e| e.screen.view.panning())
        }
    }
}

/// What the secondary button does to the view it drags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PanInput {
    /// Pressed `at` across (in the window's units), over a part `width`
    /// wide: the drag starts.
    Start {
        /// Where.
        at: f32,
        /// The width of what it was pressed on.
        width: f32,
    },
    /// The pointer is `at` across.
    To(f32),
    /// Let go of.
    End,
}

/// What a fit counts the words that no zoom shows whole by ([`widest_fit`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Floor {
    /// Those not shown whole at this many letters a second (the Transition
    /// tab's lanes: [`crate::core::lanes::LYRICS_MOST_PER`]).
    PerSecond(f64),
    /// Those the narrowest of the spans does not show whole (the lyrics
    /// editor's).
    Narrowest,
}

/// What `z` zooms to: of `spans` (the seconds across a view `letters`
/// letters wide, the narrowest first), the widest at which `cut(per)` (how
/// many words a view at `per` letters a second does not show whole) is no
/// more than `floor` has (the words no zoom shows whole). Its index, and
/// whether one is; else the narrowest's, 0.
pub fn widest_fit(
    spans: &[f64],
    letters: u16,
    floor: Floor,
    cut: &dyn Fn(f64) -> usize,
) -> (usize, bool) {
    let per = |span: f64| f64::from(letters) / span.max(1e-3);
    let least = cut(match floor {
        Floor::PerSecond(per) => per,
        Floor::Narrowest => spans.first().map_or(0.0, |&s| per(s)),
    });
    let fits = |span: f64| cut(per(span)) <= least;
    match (0..spans.len()).rev().find(|&k| fits(spans[k])) {
        Some(k) => (k, true),
        None => (0, false),
    }
}

/// How many of `words` a view at `per` letters a second shows cut short:
/// each a box as long as it is sung, which shows its text whole where its
/// letters and `pad` more fit in it (the lyrics editor's boxes).
pub fn words_cut(words: &[&jodel_lyrics::Word], per: f64, pad: f64) -> usize {
    words
        .iter()
        .filter(|w| match (w.start, w.end) {
            (Some(start), Some(end)) => (end - start) * per < w.text.chars().count() as f64 + pad,
            _ => false,
        })
        .count()
}

/// What the key that zooms a view to fit its lyrics is called, beside it.
pub const FIT_LYRICS: &str = "fit lyrics";

/// What `z` says without timed words to fit the zoom to.
pub const NOTHING_TO_FIT: &str = "No timed lyrics to fit the zoom to";

/// What `z` says once it has zoomed `what` (`the playing track`) to show
/// `shown` (`16 bars`): fitted to its lyrics, or (`fits` false) in all
/// the way with words still too tight.
pub fn fit_said(what: &str, shown: &str, fits: bool) -> String {
    if fits {
        format!("Zoomed {what} to fit its lyrics: {shown}")
    } else {
        format!("Zoomed {what} in all the way ({shown}); some words still do not fit")
    }
}

/// What a view's `z` fits its lyrics over ([`fit_lyrics`]).
pub struct LyricsFit<'a> {
    /// What the view shows, as said (`the song`, `the playing track`).
    pub what: &'a str,
    /// The lyrics' words fitted ([`crate::core::lanes::fitted_words`]).
    pub words: &'a [&'a Word],
    /// The letters across the view (0: not measured yet).
    pub letters: u16,
    /// The spans it shows, the narrowest first.
    pub spans: &'a [f64],
    /// What counts the words no span shows whole.
    pub floor: Floor,
    /// How many of the words a view at so many letters a second does not
    /// show whole.
    pub cut: &'a dyn Fn(&[&Word], f64) -> usize,
    /// How a span (its index) is said: `16 bars`, `4.2 s`.
    pub shown: &'a dyn Fn(usize) -> String,
}

/// What `z` comes to ([`fit_lyrics`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Fitted {
    /// No timed words to fit to.
    Nothing,
    /// The span to zoom to (its index), and what is said of it.
    Zoom {
        /// The span's index.
        index: usize,
        /// What is said.
        said: String,
    },
}

impl Fitted {
    /// What is said of it.
    pub fn said(&self) -> &str {
        match self {
            Fitted::Nothing => NOTHING_TO_FIT,
            Fitted::Zoom { said, .. } => said,
        }
    }
}

/// `z` on a view: the widest of its spans at which its lyrics' words show
/// whole but those no span shows so ([`widest_fit`]), and what is said of
/// it ([`fit_said`]). `None` while the view is not measured.
pub fn fit_lyrics(fit: &LyricsFit) -> Option<Fitted> {
    if fit.letters == 0 {
        return None;
    }
    if fit.words.is_empty() {
        return Some(Fitted::Nothing);
    }
    let cut = |per| (fit.cut)(fit.words, per);
    let (index, fits) = widest_fit(fit.spans, fit.letters, fit.floor, &cut);
    let said = fit_said(fit.what, &(fit.shown)(index), fits);
    Some(Fitted::Zoom { index, said })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held, a view stays; leading, the playhead a third in; by pages, a
    /// page on each time the playhead gets that far, from where it began.
    #[test]
    fn a_view_is_held_leads_the_playhead_or_pages_after_it() {
        let held = TimeView::held(12.0);
        assert_eq!(held.leading(9.0, Some(40.0)), Some(12.0));
        assert_eq!(held.paged(9.0, Some(40.0), 0.5), 12.0);
        let leading = TimeView::default();
        assert!(leading.is_following());
        assert_eq!(leading.leading(9.0, Some(40.0)), Some(37.0));
        assert_eq!(leading.leading(9.0, None), None);
        let mut paging = TimeView::default();
        paging.follow_from(10.0);
        assert_eq!(paging.paged(10.0, Some(14.0), 0.5), 10.0);
        assert_eq!(paging.paged(10.0, Some(16.0), 0.5), 15.0);
        assert_eq!(paging.paged(10.0, None, 0.5), 10.0);
        // Moved as it is: still following, its pages from there.
        paging.move_to(20.0);
        assert_eq!(paging.paged(10.0, Some(26.0), 0.5), 25.0);
        let mut v = TimeView::default();
        v.lead_to(30.0, 9.0);
        assert_eq!(v, TimeView::held(27.0));
        v.follow();
        assert_eq!(v.paged(9.0, None, 0.5), 27.0, "its pages from where it was");
        v.lead_to(1.0, 9.0);
        assert_eq!(v, TimeView::held(0.0));
        assert_eq!(centred_start(10.0, 4.0), 8.0);
        assert_eq!(centred_start(1.0, 4.0), 0.0);
    }

    /// Zoomed, the time under the pointer stays where it was across the
    /// view; one out of view comes in where asked.
    #[test]
    fn a_zoom_keeps_the_time_where_it_was_across() {
        assert_eq!(zoomed_start((10.0, 20.0), 5.0, 12.5, LEAD), 11.25);
        assert_eq!(zoomed_start((10.0, 20.0), 6.0, 40.0, 0.5), 37.0);
        assert_eq!(zoomed_start((10.0, 20.0), 400.0, 11.0, 0.5), 0.0);
    }

    /// A drag moves the view the other way, in proportion, never before 0,
    /// holding it; let go of, the pointer moves it no more.
    #[test]
    fn a_drag_moves_the_view_with_the_pointer() {
        let pan = Pan {
            at: 100.0,
            start: 10.0,
            per_unit: 0.1,
        };
        assert_eq!(pan.start_at(80.0), 12.0);
        assert_eq!(pan.start_at(300.0), 0.0);
        let mut v = TimeView::default();
        v.start_pan(100.0, 10.0, 0.1);
        assert!(v.panning() && v.is_following());
        v.pan_to(80.0);
        assert_eq!(v.leading(5.0, Some(50.0)), Some(12.0));
        v.end_pan();
        v.pan_to(0.0);
        assert!(!v.panning());
        assert_eq!(v.leading(5.0, Some(50.0)), Some(12.0));
    }

    /// The widest span that cuts no more words than the floor does.
    #[test]
    fn the_fit_is_the_widest_span_that_cuts_no_more_than_the_floor() {
        let spans = [1.0, 2.0, 4.0, 8.0];
        // A word cut at under 10 letters a second; 20 letters across.
        let cut = |per: f64| usize::from(per < 10.0);
        assert_eq!(widest_fit(&spans, 20, Floor::Narrowest, &cut), (1, true));
        assert_eq!(
            widest_fit(&spans, 20, Floor::PerSecond(4.0), &cut),
            (3, true)
        );
        let never = |_: f64| 1;
        assert_eq!(widest_fit(&spans, 20, Floor::Narrowest, &never), (3, true));
        let worse = |per: f64| usize::from(per < 100.0) * 2;
        assert_eq!(
            widest_fit(&spans, 20, Floor::PerSecond(400.0), &worse),
            (0, false)
        );
    }

    /// `z`: nothing before the view is measured, said so without words,
    /// else the widest span that fits, said.
    #[test]
    fn z_fits_a_view_and_says_so() {
        let mut word = Word::new("hello");
        (word.start, word.end) = (Some(1.0), Some(1.5));
        let words = [&word];
        let cut = |words: &[&Word], per: f64| words_cut(words, per, 0.0);
        let shown = |k: usize| format!("span {k}");
        let fit = LyricsFit {
            what: "the song",
            words: &words,
            letters: 20,
            spans: &[1.0, 2.0, 4.0],
            floor: Floor::Narrowest,
            cut: &cut,
            shown: &shown,
        };
        // 5 letters in 0.5 s: 10 letters a second, 20 across: 2 s.
        let fitted = fit_lyrics(&fit).unwrap();
        assert_eq!(
            fitted,
            Fitted::Zoom {
                index: 1,
                said: fit_said("the song", "span 1", true)
            }
        );
        assert!(fit_lyrics(&LyricsFit { letters: 0, ..fit }).is_none());
        let none = LyricsFit { words: &[], ..fit };
        assert_eq!(fit_lyrics(&none).unwrap().said(), NOTHING_TO_FIT);
    }
}
