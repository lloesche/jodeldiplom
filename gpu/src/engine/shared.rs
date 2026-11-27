//! A device the process keeps for the parts of models that each hold
//! their own memory on it ([`SharedDevice`]: the picture models'): set up
//! when first asked for and kept (never torn down while it works), set up
//! again on the next ask once it is lost (the parts on the lost one say
//! so); a setup that failed stands for [`RETRY`], as a [`super::Kept`]
//! network's does.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::kept::RETRY;
use super::OnDevice;
use crate::Turned;

/// The device now, and when the last setup failed (and why).
struct Slot<G> {
    on: Option<Arc<Turned<G>>>,
    failed: Option<(Instant, String)>,
}

/// A device set up by `set_up` when first asked for, shared by every part
/// that asks for it.
pub(crate) struct SharedDevice<G> {
    set_up: fn() -> Result<G, String>,
    slot: Mutex<Slot<G>>,
}

impl<G: OnDevice> SharedDevice<G> {
    /// The device `set_up` makes, not set up yet (for a `static`).
    pub(crate) const fn new(set_up: fn() -> Result<G, String>) -> SharedDevice<G> {
        SharedDevice {
            set_up,
            slot: Mutex::new(Slot {
                on: None,
                failed: None,
            }),
        }
    }

    /// The device, set up now where it is not or was lost; why not where
    /// it cannot be (a failed setup stands for [`RETRY`]).
    pub(crate) fn get(&self) -> Result<Arc<Turned<G>>, String> {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(on) = slot.on.as_ref().filter(|on| !on.lost()) {
            return Ok(Arc::clone(on));
        }
        slot.on = None;
        if let Some((at, why)) = &slot.failed {
            if at.elapsed() < RETRY {
                return Err(why.clone());
            }
        }
        let made = std::panic::catch_unwind(self.set_up)
            .unwrap_or_else(|_| Err("setting the GPU up panicked".into()));
        match made {
            Ok(on) => {
                let on = Arc::new(Turned::new(on));
                slot.on = Some(Arc::clone(&on));
                slot.failed = None;
                Ok(on)
            }
            Err(why) => {
                slot.failed = Some((Instant::now(), why.clone()));
                Err(why)
            }
        }
    }
}
