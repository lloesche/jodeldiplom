//! A timing trace for finding what holds the player up: with
//! `JODELDIPLOM_TRACE=FILE` set, [`note`] appends a line to `FILE` with the
//! milliseconds since the first note; without it, notes cost a check.
//! The window notes its start's steps and slow frames, the app its slow
//! updates ([`SLOW`]).

use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Updates and frames longer than this are noted.
pub const SLOW: Duration = Duration::from_millis(30);

struct Trace {
    file: Mutex<File>,
    started: Instant,
}

#[expect(
    clippy::disallowed_methods,
    reason = "the debug trace (JODELDIPLOM_TRACE) is a log written as it happens, not a file replaced whole"
)]
fn trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os("JODELDIPLOM_TRACE")?;
            let file = File::create(path).ok()?;
            Some(Trace {
                file: Mutex::new(file),
                started: Instant::now(),
            })
        })
        .as_ref()
}

/// Whether the trace is on.
pub fn on() -> bool {
    trace().is_some()
}

/// Notes `what`, when the trace is on.
pub fn note(what: impl Display) {
    if let Some(t) = trace() {
        let ms = t.started.elapsed().as_secs_f64() * 1000.0;
        if let Ok(mut file) = t.file.lock() {
            let _ = writeln!(file, "{ms:9.1} ms  {what}");
        }
    }
}

/// Notes `what` and how long it took since `since`, when that is longer
/// than [`SLOW`] (or `always`).
pub fn took(what: impl Display, since: Instant, always: bool) {
    let took = since.elapsed();
    if on() && (always || took >= SLOW) {
        note(format_args!(
            "{what}: {:.1} ms",
            took.as_secs_f64() * 1000.0
        ));
    }
}

/// The window's frames over a few seconds, for a line in the trace: how
/// many, the longest gap between two, how long drawing and showing took,
/// and how background GPU work gave way ([`jodel_gpu::pace`]).
#[derive(Default)]
pub struct Frames {
    since: Option<Instant>,
    count: u32,
    late: u32,
    worst_gap: Duration,
    draw: (Duration, Duration),
    show: (Duration, Duration),
}

/// How often [`Frames`] writes its line.
const FRAMES_EVERY: Duration = Duration::from_secs(5);

impl Frames {
    /// A frame drawn in `draw` and shown in `show`, `gap` after the last
    /// one of what moves (`None`: not one of those); `late`: later than it
    /// was wanted. `what`: what the window shows. Writes the line every
    /// few seconds, when the trace is on.
    pub fn frame(
        &mut self,
        gap: Option<Duration>,
        late: bool,
        draw: Duration,
        show: Duration,
        what: &str,
    ) {
        if !on() {
            return;
        }
        let now = Instant::now();
        let since = *self.since.get_or_insert(now);
        self.count += 1;
        self.late += u32::from(late);
        if let Some(gap) = gap {
            self.worst_gap = self.worst_gap.max(gap);
        }
        self.draw = (self.draw.0 + draw, self.draw.1.max(draw));
        self.show = (self.show.0 + show, self.show.1.max(show));
        let span = now - since;
        if span < FRAMES_EVERY {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let n = f64::from(self.count.max(1));
        let gpu = jodel_gpu::pace::report();
        note(format_args!(
            "frames ({what}): {} in {:.1} s ({:.0} a second), {} late, longest gap {:.0} ms; \
             drawing {:.1} ms on average, {:.1} at most; showing {:.1} ms on average, {:.1} at most; \
             background GPU work: {} frames late while it ran, {} rests, {:.0} ms in all, \
             {:.0} ms waiting for the stems playing, its share now {:.0} %",
            self.count,
            span.as_secs_f64(),
            n / span.as_secs_f64(),
            self.late,
            ms(self.worst_gap),
            ms(self.draw.0) / n,
            ms(self.draw.1),
            ms(self.show.0) / n,
            ms(self.show.1),
            gpu.late,
            gpu.rests,
            ms(gpu.rested),
            ms(gpu.waited),
            gpu.duty * 100.0,
        ));
        *self = Frames::default();
    }
}
