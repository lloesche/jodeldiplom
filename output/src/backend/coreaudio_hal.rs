//! macOS: Core Audio HAL properties, for the output
//! ([`super::coreaudio`]) and the input ([`super::coreaudio_input`]): reading
//! and setting them, hog mode, and listening to them.

use crate::error::Error;
use objc2_core_audio::{
    kAudioDevicePropertyHogMode, kAudioDevicePropertyStreams, kAudioHardwarePropertyDevices,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject, AudioObjectAddPropertyListener,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectPropertyListenerProc, AudioObjectRemovePropertyListener,
    AudioObjectSetPropertyData,
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

/// A property listener: `proc_` called with `client` whenever one of the
/// properties it was added for changes, until it is dropped. The caller
/// keeps `client` alive until then.
pub(super) struct Listener {
    added: Vec<(AudioObjectID, AudioObjectPropertyAddress)>,
    proc_: AudioObjectPropertyListenerProc,
    client: *mut c_void,
}

// SAFETY: the HAL's listener calls are thread-safe; `client` is only handed
// back to the HAL.
unsafe impl Send for Listener {}

impl Listener {
    /// Listens to each `(object, address)` of `on` that the HAL takes.
    pub(super) fn add(
        on: &[(AudioObjectID, AudioObjectPropertyAddress)],
        proc_: AudioObjectPropertyListenerProc,
        client: *mut c_void,
    ) -> Listener {
        let added = on
            .iter()
            .copied()
            .filter(|(object, address)| {
                // SAFETY: registers `proc_` with `client`, removed on drop.
                unsafe {
                    AudioObjectAddPropertyListener(*object, NonNull::from(address), proc_, client)
                        == 0
                }
            })
            .collect();
        Listener {
            added,
            proc_,
            client,
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        for (object, address) in &self.added {
            // SAFETY: removes what `add` registered.
            unsafe {
                AudioObjectRemovePropertyListener(
                    *object,
                    NonNull::from(address),
                    self.proc_,
                    self.client,
                );
            }
        }
    }
}
