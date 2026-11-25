//! macOS: Core Audio HAL properties, for the output
//! ([`super::coreaudio`]) and the input ([`super::coreaudio_input`]): reading
//! and setting them, hog mode, and listening to them.

use crate::error::Error;
use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2_core_audio::{
    kAudioDevicePropertyHogMode, kAudioDevicePropertyStreams, kAudioHardwarePropertyDevices,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject, AudioObjectAddPropertyListenerBlock,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertyListenerProc,
    AudioObjectRemovePropertyListenerBlock, AudioObjectSetPropertyData,
};
use objc2_core_foundation::{CFRetained, CFString};
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::{null, NonNull};

pub(super) const GLOBAL: u32 = kAudioObjectPropertyScopeGlobal;
pub(super) const OUTPUT: u32 = kAudioObjectPropertyScopeOutput;

pub(super) fn addr(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

pub(super) fn status_error(call: &'static str, status: i32) -> Error {
    Error::Backend {
        call,
        code: i64::from(status),
        message: String::new(),
    }
}

/// Reads a fixed-size property.
pub(super) fn get<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
    call: &'static str,
) -> Result<T, Error> {
    // SAFETY: `value` is a plain-data out buffer of the size we pass.
    unsafe {
        let mut value: T = std::mem::zeroed();
        let mut size = size_of::<T>() as u32;
        let status = AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast(),
        );
        if status == 0 {
            Ok(value)
        } else {
            Err(status_error(call, status))
        }
    }
}

/// Reads an array property.
pub(super) fn get_vec<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
) -> Vec<T> {
    // SAFETY: the buffer is sized from the property's reported size.
    unsafe {
        let mut size = 0u32;
        if AudioObjectGetPropertyDataSize(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
        ) != 0
            || (size as usize) < size_of::<T>()
        {
            return Vec::new();
        }
        let mut values: Vec<T> = (0..size as usize / size_of::<T>())
            .map(|_| std::mem::zeroed())
            .collect();
        let mut size = (values.len() * size_of::<T>()) as u32;
        let status = AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(values.as_mut_ptr().cast()),
        );
        if status != 0 {
            return Vec::new();
        }
        values.truncate(size as usize / size_of::<T>());
        values
    }
}

pub(super) fn set<T: Copy>(
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
    value: T,
    call: &'static str,
) -> Result<(), Error> {
    // SAFETY: passes a plain-data value of its own size.
    let status = unsafe {
        AudioObjectSetPropertyData(
            object,
            NonNull::from(&address),
            0,
            null(),
            size_of::<T>() as u32,
            NonNull::from(&value).cast(),
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(status_error(call, status))
    }
}

/// Reads a CFString property and releases it.
pub(super) fn get_string(object: AudioObjectID, selector: u32) -> Option<String> {
    let p: *const CFString =
        get(object, addr(selector, GLOBAL), "AudioObjectGetPropertyData").ok()?;
    // SAFETY: string properties are returned with +1 retain count.
    let s = unsafe { CFRetained::from_raw(NonNull::new(p as *mut CFString)?) };
    Some(s.to_string())
}

pub(super) fn output_streams(device: AudioObjectID) -> Vec<AudioObjectID> {
    get_vec(device, addr(kAudioDevicePropertyStreams, OUTPUT))
}

pub(super) fn system_devices() -> Vec<AudioObjectID> {
    get_vec(
        kAudioObjectSystemObject as AudioObjectID,
        addr(kAudioHardwarePropertyDevices, GLOBAL),
    )
}

/// Whether a process (never this one: it takes no hog mode) holds `device`
/// in hog mode, as far as it can be read.
pub(super) fn hogged(device: AudioObjectID) -> bool {
    get::<i32>(
        device,
        addr(kAudioDevicePropertyHogMode, GLOBAL),
        "get hog mode",
    )
    .is_ok_and(|pid| pid != -1)
}

/// What the HAL calls a listener block with: how many addresses changed,
/// and they.
type Notice = dyn Fn(u32, NonNull<AudioObjectPropertyAddress>);

/// A property listener: `proc_` called with `client` whenever one of the
/// properties it was added for changes, until it is dropped. The calls
/// come one at a time on a serial queue of the listener's own, and dropping
/// it waits for one under way: once it is dropped nothing calls with
/// `client` any more, and the caller may free it (keeping it alive until
/// then). It must not be dropped from inside `proc_` (that would wait on
/// itself).
pub(super) struct Listener {
    queue: DispatchRetained<DispatchQueue>,
    added: Vec<(AudioObjectID, AudioObjectPropertyAddress, RcBlock<Notice>)>,
}

// SAFETY: the blocks are called only by the HAL, on `queue`, and released
// on drop after the HAL let them go and the queue is drained; the queue is
// thread-safe. `client` is only handed to `proc_`.
unsafe impl Send for Listener {}

impl Listener {
    /// Listens to each `(object, address)` of `on` that the HAL takes.
    pub(super) fn add(
        on: &[(AudioObjectID, AudioObjectPropertyAddress)],
        proc_: AudioObjectPropertyListenerProc,
        client: *mut c_void,
    ) -> Listener {
        let queue = DispatchQueue::new("jodel.coreaudio.listener", DispatchQueueAttr::SERIAL);
        let added = on
            .iter()
            .filter_map(|&(object, address)| {
                let block: RcBlock<Notice> = RcBlock::new(
                    move |count: u32, addresses: NonNull<AudioObjectPropertyAddress>| {
                        if let Some(proc_) = proc_ {
                            // SAFETY: the caller keeps `client` alive while
                            // the listener lives, and its drop waits for
                            // this call.
                            unsafe { proc_(object, count, addresses, client) };
                        }
                    },
                );
                // SAFETY: registers the block on our queue; removed on drop.
                let status = unsafe {
                    AudioObjectAddPropertyListenerBlock(
                        object,
                        NonNull::from(&address),
                        Some(&queue),
                        RcBlock::as_ptr(&block),
                    )
                };
                (status == 0).then_some((object, address, block))
            })
            .collect();
        Listener { queue, added }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        for (object, address, block) in &self.added {
            // SAFETY: removes what `add` registered, with the same queue and
            // block.
            unsafe {
                AudioObjectRemovePropertyListenerBlock(
                    *object,
                    NonNull::from(address),
                    Some(&self.queue),
                    RcBlock::as_ptr(block),
                );
            }
        }
        // A call the HAL queued before the removal runs before this one:
        // once it returns, none is left.
        self.queue.exec_sync(|| {});
    }
}
