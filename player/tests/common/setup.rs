//! Setup ([`jodeldiplom::setup`]) on a made-up web, for the setup screen's
//! tests and shots: the first download comes at once, the next stalls
//! part-way (the screen shows a step under way) or fails (the screen shows
//! the failure and its buttons).

// Each test or example takes what it needs of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodeldiplom::setup::{manifest_for, Part, Places, Setup, Status, Work};

/// What the made-up web does after its first download.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Web {
    /// The next download gets `share` of its part after `after`, then
    /// stalls for good.
    Stalls { share: f64, after: Duration },
    /// Every other download fails: this computer is offline.
    Offline,
}

/// Setup's work on the made-up web; it counts the downloads asked for.
pub struct Made {
    web: Web,
    pub downloads: AtomicUsize,
}

impl Work for Made {
    fn download(
        &self,
        part: &Part,
        to: &Path,
        progress: &mut dyn FnMut(u64),
    ) -> Result<(), String> {
        if self.downloads.fetch_add(1, Ordering::SeqCst) == 0 {
            progress(part.sum.bytes);
            return durable::write_atomic(to, b"made up").map_err(|e| e.to_string());
        }
        match self.web {
            Web::Offline => Err("the connection timed out".into()),
            Web::Stalls { share, after } => {
                std::thread::sleep(after);
                progress((part.sum.bytes as f64 * share) as u64);
                loop {
                    std::thread::park();
                }
            }
        }
    }

    fn pack(
        &self,
        _parts: &Path,
        _to: &Path,
        _progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String> {
        Err("nothing to pack".into())
    }

    fn free_space(&self, _dir: &Path) -> Option<u64> {
        Some(1 << 40)
    }

    fn memory(&self) -> Option<u64> {
        Some(32 << 30)
    }
}

/// Setup started on `web` into an empty folder under `dir` (the player's
/// whole list, as on a machine with a GPU), and its work.
pub fn start(web: Web, dir: &Path) -> (Setup, Arc<Made>) {
    let store: PathBuf = dir.join("models");
    let places = Places {
        look: vec![store.clone()],
        store,
    };
    let made = Arc::new(Made {
        web,
        downloads: AtomicUsize::new(0),
    });
    let setup = Setup::start(
        || manifest_for(true),
        places,
        made.clone(),
        Arc::new(|| {}),
        None,
    );
    (setup, made)
}

/// `setup`'s status once `ready` holds for it (within ten seconds).
pub fn when(setup: &Setup, ready: impl Fn(&Status) -> bool) -> Status {
    let started = Instant::now();
    loop {
        let status = setup.status();
        if ready(&status) {
            return status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "setup never got there: {status:?}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}
