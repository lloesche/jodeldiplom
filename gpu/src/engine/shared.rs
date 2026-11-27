//! A device the process keeps for the parts of models that each hold
//! their own memory on it ([`SharedDevice`]: the cover's vision networks',
//! the picture models'): set up when first asked for and kept (never torn
//! down while it works), set up again on the next ask once it is lost (the
//! parts on the lost one say so); a setup that failed stands a while
//! ([`RETRY`] at first, doubled after each failure in a row up to half an
//! hour: [`Retrying`]), as a [`super::Kept`] network's does.

use std::sync::{Arc, Mutex};

use super::retrying::{Retrying, RETRY};
use super::OnDevice;
use crate::Turned;

/// A device set up by `set_up` when first asked for, shared by every part
/// that asks for it.
pub(crate) struct SharedDevice<G> {
    set_up: fn() -> Result<G, String>,
    slot: Mutex<Retrying<Arc<Turned<G>>>>,
}

impl<G: OnDevice> SharedDevice<G> {
    /// The device `set_up` makes, not set up yet (for a `static`).
    pub(crate) const fn new(set_up: fn() -> Result<G, String>) -> SharedDevice<G> {
        SharedDevice {
            set_up,
            slot: Mutex::new(Retrying::new()),
        }
    }

    /// The device, set up now where it is not or was lost; why not where
    /// it cannot be (a failed setup stands a while).
    pub(crate) fn get(&self) -> Result<Arc<Turned<G>>, String> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if slot.on.as_ref().is_some_and(|on| on.lost()) {
            slot.on = None;
        }
        let on = slot.ready(RETRY, || Ok(Arc::new(Turned::new((self.set_up)()?))))?;
        Ok(Arc::clone(on))
    }
}
