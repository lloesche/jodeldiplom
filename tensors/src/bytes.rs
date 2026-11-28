//! A file's bytes, read only ([`Bytes`]): mapped into memory from the
//! file, as a model's file is read, or held in memory (bytes made whole:
//! the tests' files).
//!
//! Mapped, the pages are the file's own: the system reads each as it is
//! first touched (from its file cache where the file is there), zeroes
//! none, and can drop them again under pressure, where a copy read into
//! the heap is the process's own memory beside the file cache's. A loader
//! keeps the mapping only while it loads ([`Bytes`] dropped with its
//! reader once the weights are up), so a file is mapped only for that
//! while: setup's repair replaces a damaged model's files (a rename over
//! each) after its load has failed and let go.
//!
//! A mapping is sound only while its file is not written in place or
//! truncated: the player replaces a model's file only whole, by a rename,
//! which leaves a mapping reading the file it mapped (measured on Windows
//! 11's NTFS and on Linux: the rename over a mapped file goes through).
//! What a mapping cannot do is fail softly. A read the system cannot
//! make under it (a bad sector, a drive unplugged or a network folder
//! gone while a model loads), and on Linux and macOS a file truncated in
//! place under it, ends the process (`EXCEPTION_IN_PAGE_ERROR`,
//! `SIGBUS`) where a read was an error the repair took. A file that is
//! short already is refused before anything is mapped: setup checks every
//! model file's length at the start, and a reader checks each place its
//! header gives against the length mapped.
//!
//! The mapping is the system's own call (`mmap`, `MapViewOfFile`): this
//! crate depends on nothing.

use std::fs::File;
use std::io;
use std::path::Path;

/// A file's bytes, read only: mapped from the file, or held in memory
/// (from a `Vec<u8>`).
pub struct Bytes(Inner);

enum Inner {
    Mapped(Map),
    Held(Vec<u8>),
}

impl Bytes {
    /// The file at `path`, mapped (an empty file held: the systems map no
    /// empty view). Its length is the file's as it is opened; a reader
    /// checks every place its header gives against it.
    ///
    /// # Safety
    ///
    /// The file is not written in place or truncated while the bytes live
    /// (replaced whole, by a rename, it may be).
    pub(crate) unsafe fn open(path: &Path) -> io::Result<Bytes> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a file too long to map"))?;
        if len == 0 {
            return Ok(Bytes(Inner::Held(Vec::new())));
        }
        Ok(Bytes(Inner::Mapped(Map::new(&file, len)?)))
    }

    /// A model's file at `path` mapped: a checkpoint, a GGUF file, the
    /// speech recogniser's packed file. The file is to be replaced only
    /// whole (by a rename) while the bytes live.
    pub fn model(path: &Path) -> io::Result<Bytes> {
        // SAFETY: model files are written only through `durable`, whole
        // under a temporary name and renamed over the old one (setup's
        // download and repair, the packed file's writer): never written
        // in place or truncated.
        unsafe { Bytes::open(path) }
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(bytes: Vec<u8>) -> Bytes {
        Bytes(Inner::Held(bytes))
    }
}

impl std::ops::Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.0 {
            Inner::Mapped(map) => map.bytes(),
            Inner::Held(bytes) => bytes,
        }
    }
}

/// A file's whole length mapped read only, unmapped when dropped.
struct Map {
    at: *const u8,
    len: usize,
}

// SAFETY: the view is read only and owned by this value alone: any thread
// may read it, and the one that drops it unmaps it.
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    /// `len` bytes of `file` (its length now, not 0) mapped, and the
    /// system asked to read them all ahead: a load touches a checkpoint's
    /// tensors out of the file's order, a piece at a time between its
    /// uploads, and pages read only as each is touched come slowly from a
    /// cold disk (klein's drawer, its 4 GB file not in the file cache, on
    /// the 5080's machine from a SATA disk: 85 s so, 12 s read ahead, where
    /// reading the file whole takes about 10; from the file cache the same
    /// either way). A hint: where it is refused, the pages come as they are
    /// touched.
    fn new(file: &File, len: usize) -> io::Result<Map> {
        let at = sys::map(file, len)?;
        sys::read_ahead(at, len);
        Ok(Map { at, len })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view of `len` bytes from `at` stays mapped while
        // `self` lives (its file left as [`Bytes::open`] asks).
        unsafe { std::slice::from_raw_parts(self.at, self.len) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        sys::unmap(self.at, self.len);
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *const c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(
            process: *mut c_void,
            entries: usize,
            ranges: *const Range,
            flags: u32,
        ) -> i32;
    }

    /// `WIN32_MEMORY_RANGE_ENTRY`.
    #[repr(C)]
    struct Range {
        at: *const c_void,
        bytes: usize,
    }

    /// The system asked to read the view's `len` bytes from `at` into its
    /// file cache, in large reads (`PrefetchVirtualMemory`).
    pub(super) fn read_ahead(at: *const u8, len: usize) {
        let range = Range {
            at: at.cast(),
            bytes: len,
        };
        // SAFETY: one range within a view of this process; a hint only,
        // its failure ignored.
        unsafe {
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
        }
    }

    /// `len` bytes of `file` mapped read only: where they begin. A file
    /// shorter by now than `len` is an error (the view is refused).
    pub(super) fn map(file: &File, len: usize) -> io::Result<*const u8> {
        // SAFETY: `file` is an open handle with read access; the mapping
        // (the file's length, unnamed) is closed once the view holds it.
        unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return Err(io::Error::last_os_error());
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len);
            let refused = io::Error::last_os_error();
            CloseHandle(mapping);
            if view.is_null() {
                return Err(refused);
            }
            Ok(view.cast_const().cast())
        }
    }

    pub(super) fn unmap(at: *const u8, _len: usize) {
        // SAFETY: `at` is a view `map` made, unmapped once.
        unsafe {
            UnmapViewOfFile(at.cast());
        }
    }
}

// `mmap` as declared below takes a 64-bit `off_t`: so it does on 64-bit
// Linux and on macOS, not on 32-bit glibc.
#[cfg(all(unix, not(target_pointer_width = "64")))]
compile_error!("jodel-tensors maps files on 64-bit systems only");

#[cfg(all(unix, target_pointer_width = "64"))]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    /// `MADV_WILLNEED`, on Linux and macOS alike.
    const MADV_WILLNEED: i32 = 3;

    /// The system asked to read the mapping's `len` bytes from `at` ahead
    /// (`madvise`'s `MADV_WILLNEED`).
    pub(super) fn read_ahead(at: *const u8, len: usize) {
        // SAFETY: the whole of a mapping `map` made; a hint only, its
        // failure ignored.
        unsafe {
            madvise(at.cast_mut().cast(), len, MADV_WILLNEED);
        }
    }

    /// `len` bytes of `file` mapped read only: where they begin.
    pub(super) fn map(file: &File, len: usize) -> io::Result<*const u8> {
        // SAFETY: a new private, read-only mapping of an open descriptor;
        // the mapping keeps the file referenced after the descriptor
        // closes.
        let at = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        // `MAP_FAILED`.
        if at as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(at.cast_const().cast())
    }

    pub(super) fn unmap(at: *const u8, len: usize) {
        // SAFETY: `at` and `len` are a mapping `map` made, unmapped once.
        unsafe {
            munmap(at.cast_mut().cast(), len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_testkit::TempDir;

    /// A file mapped reads as it is; an empty one is held (nothing to map).
    #[test]
    fn a_file_mapped_reads_as_it_is() {
        let dir = TempDir::new("bytes-mapped");
        let path = dir.join("f");
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        durable::write_atomic(&path, &bytes).unwrap();
        let mapped = Bytes::model(&path).unwrap();
        assert!(matches!(mapped.0, Inner::Mapped(_)));
        assert_eq!(&mapped[..], &bytes[..]);
        drop(mapped);
        durable::write_atomic(&path, b"").unwrap();
        let empty = Bytes::model(&path).unwrap();
        assert!(matches!(empty.0, Inner::Held(_)) && empty.is_empty());
        assert!(Bytes::model(&dir.join("gone")).is_err());
    }

    /// A file replaced whole while it is mapped (as setup's repair replaces
    /// a model's damaged file: `durable`'s rename over it) is replaced, on
    /// Windows (NTFS's POSIX renames) as elsewhere, and the mapping goes on
    /// reading the file it mapped; mapped again, the new file is read.
    #[test]
    fn a_file_replaced_while_mapped_reads_as_it_was() {
        let dir = TempDir::new("bytes-replaced");
        let path = dir.join("f");
        durable::write_atomic(&path, b"the old bytes").unwrap();
        let mapped = Bytes::model(&path).unwrap();
        let replaced = durable::write_atomic(&path, b"the new bytes, longer");
        assert!(replaced.is_ok(), "{replaced:?}");
        assert_eq!(&mapped[..], b"the old bytes");
        assert_eq!(&Bytes::model(&path).unwrap()[..], b"the new bytes, longer");
    }
}
