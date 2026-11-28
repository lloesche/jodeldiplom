//! What a [`super::Kept`] network and a [`super::SharedDevice`] hold the
//! same way ([`Retrying`]): what is set up, made when first needed, and a
//! setup that failed standing a while ([`RETRY`] at first, doubled after
//! each failure in a row up to [`MOST_RETRY`]) before the next one is
//! tried.

use std::time::{Duration, Instant};

/// How long a first failed setup stands before the next use tries again.
pub(super) const RETRY: Duration = Duration::from_secs(30);
/// The longest a failed setup stands (each failure doubles the wait).
pub(super) const MOST_RETRY: Duration = Duration::from_secs(30 * 60);

/// How long a failed setup stands: `first`, else twice as long as the
/// failure `before` it, up to [`MOST_RETRY`].
pub(super) fn stands(before: Option<Duration>, first: Duration) -> Duration {
    before.map_or(first, |before| (before * 2).min(MOST_RETRY))
}

/// What is set up now (`on`; its holder takes it away once its device is
/// lost), and when the last setup failed, why, and how long that stands.
pub(super) struct Retrying<T> {
    pub(super) on: Option<T>,
    failed: Option<(Instant, String, Duration)>,
}

impl<T> Retrying<T> {
    /// Nothing set up yet (for a `static`).
    pub(super) const fn new() -> Retrying<T> {
        Retrying {
            on: None,
            failed: None,
        }
    }

    /// What is set up, made by `set_up` where nothing is; why not where a
    /// failed setup still stands, or this one fails (a setup that panics
    /// fails), then standing `first`, or twice as long as the failure
    /// before it.
    pub(super) fn ready(
        &mut self,
        first: Duration,
        set_up: impl FnOnce() -> Result<T, String>,
    ) -> Result<&mut T, String> {
        if self.on.is_none() {
            if let Some((at, why, stands)) = &self.failed {
                if at.elapsed() < *stands {
                    return Err(why.clone());
                }
            }
            let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(set_up))
                .unwrap_or_else(|_| Err("setting it up panicked".into()));
            match made {
                Ok(on) => {
                    self.on = Some(on);
                    self.failed = None;
                }
                Err(why) => {
                    let before = self.failed.as_ref().map(|&(_, _, before)| before);
                    self.failed = Some((Instant::now(), why.clone(), stands(before, first)));
                    return Err(why);
                }
            }
        }
        Ok(self.on.as_mut().expect("set up"))
    }

    /// The last failed setup no longer stands: the next use tries at once
    /// (what failed it is mended).
    pub(super) fn forgive(&mut self) {
        self.failed = None;
    }

    /// How long the last failed setup stands, while it is the last.
    #[cfg(test)]
    pub(super) fn standing(&self) -> Option<Duration> {
        self.failed.as_ref().map(|f| f.2)
    }
}
