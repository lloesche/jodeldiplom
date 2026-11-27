//! What the workspace's tests share, a dev-dependency only: one real-time
//! test of a binary at a time ([`real_time`], [`heavy`]), the repository's
//! PCM fixtures ([`fixture`]), a temporary folder removed when it goes
//! ([`TempDir`]), an allocator that counts its calls ([`Counting`]), and
//! with the `gpu` feature the shader checks' compute runner (`gpu`).

#[cfg(feature = "gpu")]
pub mod gpu;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Held by a test that plays on a device in real time (a simulated clock
/// that plays a period every period of wall-clock time) for as long as it
/// runs: one such test of a binary at a time, and none beside one that
/// holds [`heavy`]. Side by side on a small, loaded machine (a debug
/// build, every test of a binary at once), their threads would starve each
/// other of the periods they must keep.
pub fn real_time() -> RwLockWriteGuard<'static, ()> {
    REAL_TIME.write().unwrap_or_else(|e| e.into_inner())
}

/// Held by a test that keeps the processor busy (separating stems, say) in
/// a binary with [`real_time`] tests: any number of them at once, but none
/// while one of those runs.
pub fn heavy() -> RwLockReadGuard<'static, ()> {
    REAL_TIME.read().unwrap_or_else(|e| e.into_inner())
}

static REAL_TIME: RwLock<()> = RwLock::new(());

/// A file of the repository's PCM fixtures (`tests/fixtures/pcm`).
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/pcm")
        .join(name)
}

/// A new empty folder under the system's temporary folder, removed (with
/// what is in it) when dropped: left behind, the tests' audio files filled
/// a small temporary file system run after run.
pub struct TempDir(PathBuf);

impl TempDir {
    /// A folder named after `tag`, this process and a count: tests running
    /// at once never share one.
    pub fn new(tag: &str) -> TempDir {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("jodel-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a temporary folder");
        TempDir(dir)
    }

    /// Where it is.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A call an allocator takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Call {
    /// Memory asked for: `alloc`, `alloc_zeroed` or `realloc`.
    Alloc,
    /// Memory given back: `dealloc`.
    Free,
}

/// The system's allocator, telling its function of every call before it
/// passes the call on as it came. A test binary that counts what a thread
/// allocates makes it its own:
///
/// ```ignore
/// #[global_allocator]
/// static ALLOCATOR: jodel_testkit::Counting = jodel_testkit::Counting(count);
/// ```
///
/// The function runs inside the allocator: it must not allocate itself.
pub struct Counting(pub fn(Call));

// SAFETY: every call goes to the system allocator as it came, with the
// same layout and pointer; the function only counts on the way.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        (self.0)(Call::Alloc);
        // SAFETY: the caller's contract is passed on as it is.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        (self.0)(Call::Free);
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        (self.0)(Call::Alloc);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        (self.0)(Call::Alloc);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_temporary_folder_goes_when_dropped() {
        let (a, b) = (TempDir::new("kit"), TempDir::new("kit"));
        assert_ne!(a.path(), b.path());
        std::fs::create_dir_all(a.join("inner")).unwrap();
        let kept = a.to_path_buf();
        drop(a);
        assert!(!kept.exists());
        assert!(b.is_dir());
    }

    #[test]
    fn the_fixtures_are_there() {
        assert!(fixture("ramp-44100-16bit.flac").is_file());
    }
}
